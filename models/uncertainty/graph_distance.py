"""Graph-native uncertainty: how unlike its neighbors is a node?

This is the project's novel uncertainty signal, and unlike the other methods it is
fully model-agnostic -- it reads node attributes and graph structure, never the
network. That makes it applicable to every detector, including ones that cannot
host an external uncertainty head.

Three things were wrong with the original implementation, and each one silently
degraded the measurement rather than failing:

**Categorical attributes were dropped.** Demographics were read behind
``isinstance(value, (int, float))``, which is ``False`` for ``np.int64`` -- the
dtype pandas produces -- so gender, race, and age contributed exactly nothing.
Treating them as L2 magnitudes was also wrong in principle: race code 3 is not
"three times further" than code 1. They are now a separate Hamming mismatch term.

**Continuous attributes were unnormalized.** ``blur`` and ``brightness`` run into
the hundreds while ``symmetry_*`` and ``emotion_*`` live in [0, 1], so the
Euclidean distance was effectively a blur-score distance and every other attribute
was numerically invisible. Statistics are now fitted once (robust median/IQR,
because blur is heavy-tailed) and reused.

**The degree penalty was folded into every distance.** That made the values
uninterpretable -- was a high score attribute dissimilarity, or just a low-degree
node? -- and made the degree-only ablation impossible. Since "graph distance
predicts error" could easily be "low-degree nodes are harder", that ablation is the
control this method needs, so the penalty is now its own reported key.

On fitting scope: statistics are fitted **once on the training graph** and reused
for val, test, and OOD. Fitting per split would be a silent invalidation of the
shift experiment -- an OOD generator whose images are globally blurrier would be
renormalized until it looked distributionally identical to in-distribution data,
erasing exactly the signal the method is supposed to detect.
"""

import hashlib
import math
import numbers

import numpy as np
import torch


CONTINUOUS_ATTRIBUTES = [
    "blur",
    "brightness",
    "contrast",
    "compression",
    "symmetry_eye",
    "symmetry_mouth",
    "symmetry_nose",
    "symmetry_overall",
    "emotion_angry",
    "emotion_disgust",
    "emotion_fear",
    "emotion_happy",
    "emotion_sad",
    "emotion_surprise",
    "emotion_neutral",
]

CATEGORICAL_ATTRIBUTES = [
    "Ground Truth Gender",
    "Ground Truth Race",
    "Ground Truth Age",
]

#: Reported score names. ``degree_penalty`` is deliberately a first-class method so
#: it can serve as the ablation control for the distance-based ones.
AVAILABLE_METHODS = (
    "attribute_distance",
    "embedding_distance",
    "hybrid_distance",
    "degree_penalty",
)

#: Score for a node with no neighbors, or a pair whose distance is undefined.
#: Scores land in roughly [0, 2], so this is a deliberately mid-to-high sentinel.
ISOLATED_SENTINEL = 1.0

_EPSILON = 1e-6


def _as_float(value):
    """Coerce an attribute value to float, or None if it is not numeric.

    Accepts Python and numpy scalars alike. The original
    ``isinstance(value, (int, float))`` test rejected ``np.int64``, which is what
    made the demographic attributes vanish.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.generic) and np.issubdtype(value.dtype, np.number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GraphDistanceUncertainty:
    """Fitted, cached scorer for graph-distance uncertainty.

    Usage::

        scorer = GraphDistanceUncertainty(methods).fit(train_graph.get_nodes())
        scorer.precompute(train_graph)          # optional: score every node once
        scores = scorer.compute(batch_nodes)    # {method: [N, 1] float32}
    """

    def __init__(self, methods, penalty_weight=1.0, categorical_weight=1.0, robust=True):
        unknown = [method for method in methods if method not in AVAILABLE_METHODS]
        if unknown:
            raise ValueError(
                f"unknown graph uncertainty method(s) {unknown}; "
                f"available: {list(AVAILABLE_METHODS)}"
            )
        self.methods = tuple(methods)
        self.penalty_weight = float(penalty_weight)
        self.categorical_weight = float(categorical_weight)
        self.robust = bool(robust)

        self._center = None
        self._scale = None
        self._stats_hash = None
        self._vector_cache = {}
        self._node_scores = None
        self._embedding_coverage = None

    # -- fitting ------------------------------------------------------------ #

    def fit(self, nodes):
        """Fit per-attribute location and scale over ``nodes``.

        Robust by default: median and IQR rather than mean and standard deviation,
        because ``blur`` is heavy-tailed and a handful of extreme frames would
        otherwise set the scale for everything.
        """
        nodes = list(nodes)
        if not nodes:
            raise ValueError("cannot fit graph-distance statistics on an empty node list")

        matrix = np.stack([self._build_raw_vector(node) for node in nodes], axis=0)

        with np.errstate(invalid="ignore"):
            if self.robust:
                center = np.nanmedian(matrix, axis=0)
                lower = np.nanpercentile(matrix, 25, axis=0)
                upper = np.nanpercentile(matrix, 75, axis=0)
                scale = upper - lower
            else:
                center = np.nanmean(matrix, axis=0)
                scale = np.nanstd(matrix, axis=0)

        # A zero-variance attribute contributes nothing; a scale of 1 keeps its
        # standardized value at 0 instead of producing inf/nan.
        scale = np.where(np.isfinite(scale) & (scale > _EPSILON), scale, 1.0)

        self._center = np.nan_to_num(center, nan=0.0).astype(np.float32)
        self._scale = scale.astype(np.float32)
        self._stats_hash = self._compute_stats_hash()

        with_embedding = sum(1 for node in nodes if self._embedding_of(node) is not None)
        self._embedding_coverage = with_embedding / len(nodes)

        self._vector_cache.clear()
        self._node_scores = None
        return self

    @property
    def is_fitted(self):
        return self._center is not None and self._scale is not None

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "GraphDistanceUncertainty must be fit() on the training graph before "
                "scoring. Scoring against per-batch statistics would make values "
                "incomparable across batches and splits."
            )

    @property
    def stats_hash(self):
        return self._stats_hash

    @property
    def embedding_coverage(self):
        """Fraction of fitted nodes that carried a usable face embedding.

        Worth surfacing: a missing embedding used to fall through to a flat
        sentinel, fabricating a bimodal score distribution that reads like signal.
        """
        return self._embedding_coverage

    def statistics_for(self, attribute):
        self._require_fitted()
        index = CONTINUOUS_ATTRIBUTES.index(attribute)
        return float(self._center[index]), float(self._scale[index])

    def _compute_stats_hash(self):
        digest = hashlib.sha256()
        digest.update(np.asarray(self._center, dtype=np.float32).tobytes())
        digest.update(np.asarray(self._scale, dtype=np.float32).tobytes())
        digest.update(repr((self.robust, self.categorical_weight)).encode("utf-8"))
        return digest.hexdigest()[:16]

    def state_dict(self):
        self._require_fitted()
        return {
            "center": self._center.tolist(),
            "scale": self._scale.tolist(),
            "robust": self.robust,
            "penalty_weight": self.penalty_weight,
            "categorical_weight": self.categorical_weight,
            "continuous_attributes": list(CONTINUOUS_ATTRIBUTES),
            "categorical_attributes": list(CATEGORICAL_ATTRIBUTES),
            "embedding_coverage": self._embedding_coverage,
            "stats_hash": self._stats_hash,
        }

    def load_state_dict(self, state):
        saved_attributes = state.get("continuous_attributes")
        if saved_attributes is not None and list(saved_attributes) != CONTINUOUS_ATTRIBUTES:
            raise ValueError(
                "graph-distance statistics were fitted over a different attribute set "
                f"({saved_attributes}); refusing to load, since the vectors would not align"
            )
        self._center = np.asarray(state["center"], dtype=np.float32)
        self._scale = np.asarray(state["scale"], dtype=np.float32)
        self.robust = bool(state.get("robust", self.robust))
        self.penalty_weight = float(state.get("penalty_weight", self.penalty_weight))
        self.categorical_weight = float(state.get("categorical_weight", self.categorical_weight))
        self._embedding_coverage = state.get("embedding_coverage")
        self._stats_hash = state.get("stats_hash") or self._compute_stats_hash()
        self._vector_cache.clear()
        self._node_scores = None
        return self

    # -- per-node vectors --------------------------------------------------- #

    def _build_raw_vector(self, node):
        attributes = getattr(node, "attributes", {}) or {}
        values = []
        for name in CONTINUOUS_ATTRIBUTES:
            value = _as_float(attributes.get(name))
            values.append(np.nan if value is None else value)
        return np.asarray(values, dtype=np.float32)

    def _standardized_vector(self, node):
        """Standardized attribute vector, cached per node id."""
        key = getattr(node, "node_id", id(node))
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached

        raw = self._build_raw_vector(node)
        standardized = (np.nan_to_num(raw, nan=0.0) - self._center) / self._scale
        # A missing attribute contributes zero rather than pulling toward the median.
        standardized = np.where(np.isnan(raw), 0.0, standardized).astype(np.float32)
        self._vector_cache[key] = standardized
        return standardized

    @staticmethod
    def _embedding_of(node):
        attributes = getattr(node, "attributes", {}) or {}
        embedding = attributes.get("face_embedding")
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if embedding.size == 0:
            return None
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        return embedding / norm

    def invalidate(self, node_ids=None):
        """Drop cached vectors and scores.

        Must be called when the graph is mutated -- ``GraphReductionManager``
        removes and restores nodes mid-training, which changes both adjacency and
        degree.
        """
        if node_ids is None:
            self._vector_cache.clear()
            self._node_scores = None
            return
        for node_id in node_ids:
            self._vector_cache.pop(node_id, None)
        self._node_scores = None

    # -- pairwise components ------------------------------------------------ #

    def continuous_distance(self, node, other):
        """Normalized Euclidean distance over standardized continuous attributes."""
        self._require_fitted()
        left = self._standardized_vector(node)
        right = self._standardized_vector(other)
        # Divide by sqrt(dim) so the result is a per-attribute RMS difference and
        # does not grow merely because more attributes are present.
        return float(np.linalg.norm(left - right) / math.sqrt(len(left)))

    def categorical_mismatch(self, node, other):
        """Fraction of categorical attributes that differ.

        Hamming, not L2: label codes have no magnitude, so any two distinct values
        are equally dissimilar. A value missing on either side counts as a
        mismatch, since "unknown" is not evidence of similarity.
        """
        left = getattr(node, "attributes", {}) or {}
        right = getattr(other, "attributes", {}) or {}
        if not CATEGORICAL_ATTRIBUTES:
            return 0.0

        mismatches = 0
        for name in CATEGORICAL_ATTRIBUTES:
            if name not in left or name not in right:
                mismatches += 1
                continue
            if _as_float(left[name]) != _as_float(right[name]):
                mismatches += 1
        return mismatches / len(CATEGORICAL_ATTRIBUTES)

    def attribute_distance(self, node, other):
        """Continuous distance plus the weighted categorical mismatch."""
        continuous = self.continuous_distance(node, other)
        categorical = self.categorical_mismatch(node, other)
        return continuous + self.categorical_weight * categorical

    def embedding_distance(self, node, other):
        """Cosine distance in [0, 2], or None if either embedding is unusable."""
        left = self._embedding_of(node)
        right = self._embedding_of(other)
        if left is None or right is None or left.shape != right.shape:
            return None
        similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
        return 1.0 - similarity

    def degree_penalty(self, node):
        """Penalty that decays with degree: sparsely connected nodes are riskier."""
        degree = len(node.get_adjacent_nodes()) if hasattr(node, "get_adjacent_nodes") else 0
        return self.penalty_weight / math.sqrt(float(degree) + 1.0)

    # -- scoring ------------------------------------------------------------ #

    def score_node(self, node):
        """All requested scores for one node, computed in a single neighbor pass."""
        self._require_fitted()
        neighbors = node.get_adjacent_nodes() if hasattr(node, "get_adjacent_nodes") else []
        scores = {}

        if "degree_penalty" in self.methods:
            scores["degree_penalty"] = self.degree_penalty(node)

        distance_methods = [
            method for method in self.methods
            if method in ("attribute_distance", "embedding_distance", "hybrid_distance")
        ]
        if not distance_methods:
            return scores

        if not neighbors:
            for method in distance_methods:
                scores[method] = ISOLATED_SENTINEL
            return scores

        # One pass over neighbors computing both components, then derive hybrid from
        # the values already in hand rather than recomputing them. Previously all
        # three methods each walked the neighbors independently, and hybrid redid
        # the work of the other two.
        attribute_values = []
        embedding_values = []
        hybrid_values = []
        for neighbor in neighbors:
            attribute = self.attribute_distance(node, neighbor)
            embedding = self.embedding_distance(node, neighbor)

            if math.isfinite(attribute):
                attribute_values.append(attribute)
            if embedding is not None and math.isfinite(embedding):
                embedding_values.append(embedding)

            components = [
                value for value in (attribute, embedding)
                if value is not None and math.isfinite(value)
            ]
            if components:
                hybrid_values.append(sum(components) / len(components))

        for method, values in (
            ("attribute_distance", attribute_values),
            ("embedding_distance", embedding_values),
            ("hybrid_distance", hybrid_values),
        ):
            if method in distance_methods:
                scores[method] = float(np.mean(values)) if values else ISOLATED_SENTINEL
        return scores

    def precompute(self, graph):
        """Score every node in ``graph`` once and cache the result.

        Graph-distance uncertainty is a static function of the graph, yet it was
        recomputed for every batch inside the training loop. Precomputing turns
        ``compute`` into a dictionary lookup.
        """
        self._require_fitted()
        nodes = graph.get_nodes() if hasattr(graph, "get_nodes") else list(graph)
        self._node_scores = {
            getattr(node, "node_id", id(node)): self.score_node(node) for node in nodes
        }
        return self

    def compute(self, nodes):
        """Scores for ``nodes`` as ``{method: [N, 1] float32}``, input order preserved."""
        self._require_fitted()
        nodes = list(nodes)
        per_method = {method: [] for method in self.methods}

        for node in nodes:
            if self._node_scores is not None:
                key = getattr(node, "node_id", id(node))
                scores = self._node_scores.get(key) or self.score_node(node)
            else:
                scores = self.score_node(node)
            for method in self.methods:
                per_method[method].append(scores.get(method, ISOLATED_SENTINEL))

        return {
            method: torch.tensor(values, dtype=torch.float32).unsqueeze(1)
            for method, values in per_method.items()
            if values
        }


# --------------------------------------------------------------------------- #
# Module-level entry points
# --------------------------------------------------------------------------- #

def compute_graph_uncertainty(node, methods, penalty_weight=1.0, standardizer=None):
    """Scores for a single node. Requires a fitted ``standardizer``."""
    if standardizer is None:
        raise RuntimeError(
            "compute_graph_uncertainty requires a fitted `standardizer`. Fitting on the "
            "fly would produce values that are not comparable across batches or splits, "
            "which is the scale bug this API exists to avoid."
        )
    return standardizer.score_node(node)


def compute_batch_graph_uncertainty(nodes, methods, penalty_weight=1.0, standardizer=None):
    """Scores for a batch of nodes as ``{method: [N, 1] float32}``.

    ``standardizer`` is required. There is deliberately no lazy per-batch fallback:
    batch-level statistics would silently reintroduce the incomparability this
    module was rewritten to remove.
    """
    if standardizer is None:
        raise RuntimeError(
            "compute_batch_graph_uncertainty requires a fitted `standardizer` "
            "(GraphDistanceUncertainty.fit(train_nodes)). Without one, statistics would be "
            "fitted per batch and the resulting values would not be comparable across "
            "batches, splits, or runs."
        )
    return standardizer.compute(nodes)
