"""What a training batch actually contained, recorded per batch.

The I-value work has repeatedly been misread because nothing recorded *what was selected*.
Two conclusions had to be retracted for exactly that reason: that `loss_ewma` concentrated on
a small hard set (it covered 134,096 unique nodes of 150,000 draws, the broadest of any arm),
and that sharper hard-sample selection was what hurt (it had in fact trained on the *easiest*
data in the sweep). Both were recoverable only afterwards, by mining `node_id` out of a
diagnostic written for a different purpose.

The measurement this exists for is **batch diversity**. An i-value walk takes its argmax over
the current node's k-NN neighbours, and k-NN neighbours are similar faces by construction, so
a batch drawn by walking should contain near-duplicates where an i.i.d. batch does not. Low
effective batch diversity is a standard cause of worse SGD, and it would explain why a 16x
improvement in how well the estimator ranks learning gain changed detector accuracy not at
all: the walk, not the ranking, would be the bottleneck.

Cheap by construction: the `face_embedding` is already on every node and already used to build
the graph, so this is one small matrix product per batch and no extra forward passes.
"""

import numpy as np

#: Demographic axes recorded per batch.
#:
#: These are the keys as they exist on a *node*, which are NOT the `gt_race`/`gt_gender` names
#: the record tables use -- those are produced later by the record collector. Getting this
#: wrong is not a harmless miss: the first Phase 0 run reported `race_coverage = 0.0000` for
#: both arms, which reads as "every batch contained one race" when it actually meant "the
#: attribute was never found". Each name is tried in order, and a batch where none is present
#: reports None rather than 0.0.
COMPOSITION_ATTRIBUTES = {
    "gender": ("Ground Truth Gender", "gt_gender", "gender"),
    "race": ("Ground Truth Race", "gt_race", "race"),
    "age": ("Ground Truth Age", "gt_age", "age"),
}

#: Attribute key holding the face embedding, as in `dataloaders/knn_edges.EMBEDDING_KEY`.
EMBEDDING_KEY = "face_embedding"


def batch_diversity(nodes):
    """Mean pairwise cosine *distance* between the batch's face embeddings.

    1.0 means mutually orthogonal, 0.0 means identical. Returns `(diversity, n_embedded)`;
    diversity is None when fewer than two nodes carry an embedding, which is reported rather
    than silently returned as 0.0 -- ~7.7% of nodes have no embedding at all and would
    otherwise read as a perfectly redundant batch.
    """
    vectors = []
    for node in nodes:
        value = getattr(node, "attributes", {}).get(EMBEDDING_KEY)
        if value is None:
            continue
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vectors.append(vector / norm)

    if len(vectors) < 2:
        return None, len(vectors)

    matrix = np.stack(vectors)
    similarities = matrix @ matrix.T
    # Off-diagonal mean, each unordered pair once.
    upper = np.triu_indices(len(vectors), k=1)
    return float(1.0 - similarities[upper].mean()), len(vectors)


def batch_composition(nodes):
    """Label balance and demographic composition of one batch, as a flat dict."""
    row = {}
    labels = []
    for node in nodes:
        try:
            labels.append(float(node.get_label()))
        except Exception:
            continue
    if labels:
        row["n_labelled"] = len(labels)
        row["frac_positive"] = float(np.mean(labels))

    for name, candidates in COMPOSITION_ATTRIBUTES.items():
        present = []
        for node in nodes:
            attributes = getattr(node, "attributes", {}) or {}
            for key in candidates:
                if key in attributes and attributes[key] is not None:
                    present.append(attributes[key])
                    break
        if not present:
            # None, not 0.0. A silent zero here reads as "one value in every batch".
            row[f"{name}_distinct"] = None
            row[f"{name}_coverage"] = None
            continue
        # Distinct values present, as a fraction of the batch: a batch drawn by walking a
        # similarity graph should concentrate on fewer demographic values than an i.i.d. one.
        row[f"{name}_distinct"] = len(set(present))
        row[f"{name}_coverage"] = len(set(present)) / len(present)
    return row


class SelectionDiagnostic:
    """Accumulates one row per training batch; written by the caller at the end of a run."""

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.rows = []

    def record(self, nodes, epoch=0, losses=None, selector=""):
        """Record one batch. A no-op unless enabled, so the hot path pays almost nothing."""
        if not self.enabled or not nodes:
            return

        diversity, n_embedded = batch_diversity(nodes)
        row = {
            "epoch": int(epoch),
            "selector": selector,
            "batch_size": len(nodes),
            "batch_diversity": diversity,
            "n_embedded": n_embedded,
            "n_unique": len({getattr(n, "node_id", id(n)) for n in nodes}),
        }
        row.update(batch_composition(nodes))

        if losses is not None and len(losses):
            values = np.asarray(losses, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                row["loss_min"] = float(finite.min())
                row["loss_median"] = float(np.median(finite))
                row["loss_max"] = float(finite.max())
                row["loss_mean"] = float(finite.mean())
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


__all__ = [
    "COMPOSITION_ATTRIBUTES", "EMBEDDING_KEY", "SelectionDiagnostic",
    "batch_composition", "batch_diversity",
]
