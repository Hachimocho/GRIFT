"""Graph-distance uncertainty: standardization, categoricals, and caching.

Three bugs are covered.

A5: demographics were read through ``isinstance(value, (int, float))``, which is
False for ``np.int64`` -- the dtype pandas hands over -- so gender, race, and age
contributed *nothing* to the distance. Treating label codes as L2 magnitudes was
also wrong on its own terms: race code 3 is not "three times further" than code 1.

A6: the attribute vector mixed unbounded ``blur``/``brightness`` (values in the
hundreds) with ``symmetry_*`` and ``emotion_*`` in [0, 1], unnormalized, so the
distance was effectively a blur-score distance and every other attribute was
numerically invisible.

A7: ``hybrid_distance`` recomputed what the other two methods had already
computed, and per-node attribute vectors were rebuilt for every (node, neighbor)
pair inside the training loop.
"""

import numpy as np
import pytest
import torch

from edges.Edge import Edge
from models.uncertainty.graph_distance import (
    CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES, GraphDistanceUncertainty,
    compute_batch_graph_uncertainty,
)
from nodes.atrnode import AttributeNode

ALL_METHODS = ("attribute_distance", "embedding_distance", "hybrid_distance", "degree_penalty")


def make_node(node_id, **attribute_overrides):
    """A node with every continuous attribute at a mid-scale baseline."""
    attributes = {name: 0.5 for name in CONTINUOUS_ATTRIBUTES}
    attributes.update({
        "blur": 100.0, "brightness": 100.0, "contrast": 50.0, "compression": 1.0,
    })
    for name in CATEGORICAL_ATTRIBUTES:
        attributes[name] = np.int64(0)
    attributes["face_embedding"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    attributes.update(attribute_overrides)
    return AttributeNode(node_id, "train", None, [], 0, attributes, threshold=50)


def connect(*nodes):
    for index in range(len(nodes) - 1):
        edge = Edge(nodes[index], nodes[index + 1], x=None)
        nodes[index].add_edge(edge)
        nodes[index + 1].add_edge(edge)
    return list(nodes)


def fitted(nodes, methods=ALL_METHODS, **kwargs):
    return GraphDistanceUncertainty(methods=methods, **kwargs).fit(nodes)


# --------------------------------------------------------------------------- #
# A5: categorical attributes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [np.int64, np.int32, int, np.float64, float])
def test_categorical_difference_registers_for_every_numeric_dtype(dtype):
    """A pair differing only in a demographic must have non-zero distance.

    Before the fix this was exactly 0.0 for np.int64 -- the dtype the real dataset
    uses -- so gender, race, and age were silently ignored.
    """
    same = make_node("a", **{"Ground Truth Gender": dtype(0)})
    other = make_node("b", **{"Ground Truth Gender": dtype(1)})
    nodes = connect(same, other)
    scorer = fitted(nodes, methods=("attribute_distance",))

    scores = scorer.compute(nodes)["attribute_distance"]
    assert scores[0].item() > 0.0, (
        f"a gender difference stored as {dtype.__name__} produced zero distance"
    )


def test_identical_categoricals_contribute_nothing():
    nodes = connect(make_node("a"), make_node("b"))
    scorer = fitted(nodes, methods=("attribute_distance",))
    assert scorer.categorical_mismatch(nodes[0], nodes[1]) == 0.0


def test_categorical_mismatch_is_a_fraction():
    """All three demographics differing gives a mismatch fraction of 1.0."""
    first = make_node("a", **{name: np.int64(0) for name in CATEGORICAL_ATTRIBUTES})
    second = make_node("b", **{name: np.int64(1) for name in CATEGORICAL_ATTRIBUTES})
    scorer = fitted(connect(first, second), methods=("attribute_distance",))
    assert scorer.categorical_mismatch(first, second) == pytest.approx(1.0)

    partial = make_node("c", **{
        "Ground Truth Gender": np.int64(1),
        "Ground Truth Race": np.int64(0),
        "Ground Truth Age": np.int64(0),
    })
    assert scorer.categorical_mismatch(first, partial) == pytest.approx(1.0 / 3.0)


def test_categorical_distance_is_not_ordinal():
    """Race code 3 must not read as "further" than race code 1.

    Categorical codes carry no magnitude, so any two distinct values are equally
    dissimilar. An L2 treatment made code 3 three times as distant as code 1.
    """
    base = make_node("a", **{"Ground Truth Race": np.int64(0)})
    near = make_node("b", **{"Ground Truth Race": np.int64(1)})
    far = make_node("c", **{"Ground Truth Race": np.int64(7)})
    scorer = fitted(connect(base, near, far), methods=("attribute_distance",))
    assert scorer.categorical_mismatch(base, near) == scorer.categorical_mismatch(base, far)


def test_missing_categorical_counts_as_a_mismatch():
    present = make_node("a")
    absent = make_node("b")
    del absent.attributes["Ground Truth Gender"]
    scorer = fitted(connect(present, absent), methods=("attribute_distance",))
    assert scorer.categorical_mismatch(present, absent) > 0.0


# --------------------------------------------------------------------------- #
# A6: standardization
# --------------------------------------------------------------------------- #

def test_symmetry_difference_is_visible_alongside_a_large_blur_difference():
    """The core A6 regression.

    Unstandardized, a 300-unit blur gap swamps a 0.9 symmetry gap so completely
    that changing symmetry does not measurably move the distance. After
    standardization both features contribute on comparable scales.
    """
    population = [
        make_node(f"p{index}", blur=100.0 + index * 60.0, symmetry_eye=index / 10.0)
        for index in range(10)
    ]
    scorer = fitted(population, methods=("attribute_distance",))

    anchor = make_node("anchor", blur=100.0, symmetry_eye=0.0)
    blur_only = make_node("blur", blur=400.0, symmetry_eye=0.0)
    blur_and_symmetry = make_node("both", blur=400.0, symmetry_eye=0.9)

    only = scorer.continuous_distance(anchor, blur_only)
    both = scorer.continuous_distance(anchor, blur_and_symmetry)
    assert both > only * 1.05, (
        f"adding a 0.9 symmetry difference barely changed the distance "
        f"({only:.6f} -> {both:.6f}); the blur scale is still dominating"
    )


def test_fit_computes_robust_statistics():
    """Median/IQR, not mean/std -- blur is heavy-tailed with real outliers."""
    population = [make_node(f"p{index}", blur=float(index)) for index in range(9)]
    population.append(make_node("outlier", blur=100_000.0))
    scorer = fitted(population, methods=("attribute_distance",))

    center, scale = scorer.statistics_for("blur")
    assert 0.0 <= center <= 20.0, f"a single extreme outlier moved the center to {center}"
    assert scale > 0.0


def test_fit_is_order_independent():
    population = [make_node(f"p{index}", blur=float(index * 13 % 97)) for index in range(12)]
    forward = fitted(population, methods=("attribute_distance",))
    backward = fitted(list(reversed(population)), methods=("attribute_distance",))
    assert forward.stats_hash == backward.stats_hash


def test_stats_hash_changes_with_the_population():
    first = fitted([make_node(f"a{i}", blur=float(i)) for i in range(6)])
    second = fitted([make_node(f"b{i}", blur=float(i * 5)) for i in range(6)])
    assert first.stats_hash != second.stats_hash


def test_zero_variance_attribute_does_not_divide_by_zero():
    population = [make_node(f"p{index}") for index in range(6)]  # every value identical
    scorer = fitted(population, methods=("attribute_distance",))
    distance = scorer.continuous_distance(population[0], population[1])
    assert np.isfinite(distance) and distance == pytest.approx(0.0)


def test_state_dict_roundtrip_preserves_scores():
    population = [make_node(f"p{index}", blur=float(index * 30)) for index in range(8)]
    scorer = fitted(population)
    expected = scorer.continuous_distance(population[0], population[3])

    restored = GraphDistanceUncertainty(methods=ALL_METHODS)
    restored.load_state_dict(scorer.state_dict())
    assert restored.stats_hash == scorer.stats_hash
    assert restored.continuous_distance(population[0], population[3]) == pytest.approx(expected)


def test_unfitted_scorer_refuses_to_score():
    """Scoring without fitted statistics would silently reproduce the A6 bug."""
    nodes = connect(make_node("a"), make_node("b"))
    scorer = GraphDistanceUncertainty(methods=("attribute_distance",))
    with pytest.raises(RuntimeError, match="fit"):
        scorer.compute(nodes)


# --------------------------------------------------------------------------- #
# A7: one pass, cached vectors, separated degree penalty
# --------------------------------------------------------------------------- #

def test_degree_penalty_is_its_own_key():
    """The penalty must be reported separately, not folded into each distance.

    It used to be added into every distance, which made the values uninterpretable
    (was this attribute dissimilarity or just a low-degree node?) and made the
    degree-only ablation impossible to run.
    """
    nodes = connect(make_node("a"), make_node("b"), make_node("c"))
    scorer = fitted(nodes)
    scores = scorer.compute(nodes)

    assert "degree_penalty" in scores
    assert scores["degree_penalty"].shape == (3, 1)
    # The middle node has degree 2, the ends degree 1, so its penalty is lower.
    assert scores["degree_penalty"][1].item() < scores["degree_penalty"][0].item()


def test_distances_exclude_the_degree_penalty():
    nodes = connect(make_node("a", blur=100.0), make_node("b", blur=400.0))
    scorer = fitted(nodes, methods=("attribute_distance", "degree_penalty"), penalty_weight=10.0)
    scores = scorer.compute(nodes)
    # With a penalty weight of 10 a folded-in penalty would dominate everything.
    assert scores["attribute_distance"].max().item() < 10.0


def test_attribute_vectors_are_built_once_per_node():
    """A7 regression: vectors were rebuilt for every (node, neighbor) pair."""
    nodes = connect(*[make_node(f"n{index}", blur=float(index * 20)) for index in range(6)])
    scorer = fitted(nodes)

    calls = {"count": 0}
    original = scorer._build_raw_vector

    def counting(node):
        calls["count"] += 1
        return original(node)

    scorer._build_raw_vector = counting
    scorer.invalidate()
    scorer.compute(nodes)
    assert calls["count"] <= len(nodes), (
        f"built {calls['count']} vectors for {len(nodes)} nodes -- vectors are not cached"
    )


def test_hybrid_is_the_mean_of_its_components():
    nodes = connect(
        make_node("a", blur=100.0),
        make_node("b", blur=400.0, face_embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
    )
    scorer = fitted(nodes)
    scores = scorer.compute(nodes)
    expected = (scores["attribute_distance"] + scores["embedding_distance"]) / 2.0
    assert torch.allclose(scores["hybrid_distance"], expected, atol=1e-6)


def test_precompute_matches_direct_computation():
    nodes = connect(*[make_node(f"n{index}", blur=float(index * 25)) for index in range(6)])
    from graphs.HyperGraph import HyperGraph

    scorer = fitted(nodes)
    direct = scorer.compute(nodes)
    scorer.precompute(HyperGraph(nodes))
    cached = scorer.compute(nodes)
    for method in direct:
        assert torch.allclose(direct[method], cached[method], atol=1e-6), f"{method} differs"


def test_output_order_follows_input_order():
    nodes = connect(*[make_node(f"n{index}", blur=float(index * 30)) for index in range(5)])
    scorer = fitted(nodes)

    forward = scorer.compute(nodes)["attribute_distance"].squeeze().tolist()
    reversed_scores = scorer.compute(list(reversed(nodes)))["attribute_distance"].squeeze().tolist()
    assert forward == pytest.approx(list(reversed(reversed_scores)))


def test_scores_are_invariant_to_edge_list_order():
    """Adjacency order must not change the aggregate distance."""
    nodes = connect(*[make_node(f"n{index}", blur=float(index * 30)) for index in range(5)])
    scorer = fitted(nodes)
    before = scorer.compute(nodes)["attribute_distance"].clone()

    for node in nodes:
        node.edges.reverse()
    scorer.invalidate()
    after = scorer.compute(nodes)["attribute_distance"]
    assert torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def test_identical_embeddings_are_zero_distance():
    nodes = connect(make_node("a"), make_node("b"))
    scorer = fitted(nodes, methods=("embedding_distance",))
    assert scorer.embedding_distance(nodes[0], nodes[1]) == pytest.approx(0.0, abs=1e-6)


def test_antipodal_embeddings_are_maximally_distant():
    first = make_node("a", face_embedding=np.array([1.0, 0.0], dtype=np.float32))
    second = make_node("b", face_embedding=np.array([-1.0, 0.0], dtype=np.float32))
    scorer = fitted(connect(first, second), methods=("embedding_distance",))
    assert scorer.embedding_distance(first, second) == pytest.approx(2.0, abs=1e-6)


def test_zero_norm_embedding_yields_no_distance():
    first = make_node("a")
    second = make_node("b", face_embedding=np.zeros(4, dtype=np.float32))
    scorer = fitted(connect(first, second), methods=("embedding_distance",))
    assert scorer.embedding_distance(first, second) is None


def test_missing_embedding_yields_no_distance():
    first = make_node("a")
    second = make_node("b")
    del second.attributes["face_embedding"]
    scorer = fitted(connect(first, second), methods=("embedding_distance",))
    assert scorer.embedding_distance(first, second) is None


def test_embedding_coverage_is_reported():
    """Coverage must be visible: a missing embedding silently became a flat sentinel.

    `_embedding_distance` returning None made the old code assign `1.0 + penalty`,
    fabricating a bimodal score distribution that looks like real signal.
    """
    nodes = [make_node(f"p{index}") for index in range(4)]
    for node in nodes[2:]:
        del node.attributes["face_embedding"]
    scorer = fitted(connect(*nodes))
    assert scorer.embedding_coverage == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Isolated nodes and misc
# --------------------------------------------------------------------------- #

def test_isolated_node_gets_the_maximal_sentinel():
    node = make_node("lonely")
    scorer = fitted([node], penalty_weight=1.0)
    scores = scorer.compute([node])
    assert scores["attribute_distance"][0].item() == pytest.approx(1.0)
    assert scores["degree_penalty"][0].item() == pytest.approx(1.0)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        GraphDistanceUncertainty(methods=("not_a_method",))


def test_output_dtype_and_shape():
    nodes = connect(*[make_node(f"n{index}") for index in range(4)])
    scores = fitted(nodes).compute(nodes)
    for method, tensor in scores.items():
        assert tensor.dtype == torch.float32, f"{method} is {tensor.dtype}"
        assert tensor.shape == (4, 1), f"{method} has shape {tensor.shape}"


def test_module_level_helper_requires_a_standardizer_in_strict_mode():
    """The legacy entrypoint must not silently fall back to an unfitted fit.

    A batch-level fit would make values incomparable across batches, quietly
    reintroducing the scale bug it was supposed to fix.
    """
    nodes = connect(make_node("a"), make_node("b"))
    with pytest.raises(RuntimeError, match="standardizer"):
        compute_batch_graph_uncertainty(nodes, ("attribute_distance",))


def test_module_level_helper_accepts_a_standardizer():
    nodes = connect(make_node("a", blur=100.0), make_node("b", blur=400.0))
    scorer = fitted(nodes, methods=("attribute_distance",))
    scores = compute_batch_graph_uncertainty(
        nodes, ("attribute_distance",), standardizer=scorer
    )
    assert scores["attribute_distance"].shape == (2, 1)
