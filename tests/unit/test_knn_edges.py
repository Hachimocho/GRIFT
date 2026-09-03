"""Candidate-edge generation, and why it stopped being all-pairs.

`all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]` materialised every pair as a
Python list *before* any similarity filtering. Measured, that one line costs 0.7 GiB at 5,000
nodes, 74.5 GiB at 50,000, and ~76 TiB at the full corpus's 1.6M -- on a 541 GiB machine. A
second wall sat behind it: at the project's default thresholds a measured 1,304-node split
kept 823,814 edges, 97.0% of all possible pairs, average degree 1,264.

k-NN makes candidates O(k*N) and the graph sparse by construction. The filtering step is
unchanged, so an edge still means what it meant -- there are just no longer N^2 chances.
"""

import numpy as np
import pytest

from dataloaders.knn_edges import (
    ALL_PAIRS_NODE_LIMIT, DEFAULT_KNN_NEIGHBOURS, EDGE_CONSTRUCTION_MODES,
    EdgeConstructionError, all_pairs_candidate_edges, candidate_edges, embedding_matrix,
    knn_candidate_edges,
)


class FakeNode:
    def __init__(self, index, dim=32, embedding=True):
        self.node_id = f"n{index:05d}"
        rng = np.random.default_rng(index)
        self.attributes = (
            {"face_embedding": rng.normal(size=dim).astype(np.float32)} if embedding else {}
        )


def nodes(count=200, **kwargs):
    return [FakeNode(index, **kwargs) for index in range(count)]


# -- shape and scaling ---------------------------------------------------------- #

@pytest.mark.parametrize("k", [5, 20, 50])
def test_edge_count_scales_with_k_not_with_n_squared(k):
    made = nodes(300)
    edges = knn_candidate_edges(made, k=k)
    all_pairs = len(made) * (len(made) - 1) // 2

    average_degree = 2 * len(edges) / len(made)
    # Symmetrisation can push realised degree above k, but not by an order of magnitude.
    assert k <= average_degree <= 3 * k
    assert len(edges) < all_pairs / 5


def test_every_pair_is_ordered_and_unique():
    edges = knn_candidate_edges(nodes(120), k=10)
    assert all(i < j for i, j in edges)
    assert len(edges) == len(set(edges))


def test_output_is_sorted_and_reproducible():
    """A set's iteration order is not stable across processes, so the list is sorted."""
    made = nodes(150)
    first = knn_candidate_edges(made, k=15)
    assert first == sorted(first)
    assert first == knn_candidate_edges(made, k=15)


def test_indices_restrict_the_search_and_keep_the_caller_space():
    made = nodes(200)
    subset = list(range(50, 90))
    edges = knn_candidate_edges(made, k=8, indices=subset)
    touched = {index for pair in edges for index in pair}
    assert touched <= set(subset)


def test_k_is_clamped_to_the_group_size():
    edges = knn_candidate_edges(nodes(6), k=100)
    assert edges, "a group smaller than k must still produce edges"
    assert all(i < j for i, j in edges)


@pytest.mark.parametrize("count", [0, 1])
def test_degenerate_groups_produce_nothing(count):
    assert knn_candidate_edges(nodes(count), k=5) == []


# -- the memory guard ----------------------------------------------------------- #

def test_all_pairs_is_refused_above_the_limit():
    """An early error beats an OOM kill halfway through a graph build."""
    with pytest.raises(EdgeConstructionError, match="Refusing"):
        all_pairs_candidate_edges(ALL_PAIRS_NODE_LIMIT + 1)


def test_the_refusal_quantifies_the_cost_and_names_the_fix():
    with pytest.raises(EdgeConstructionError) as caught:
        all_pairs_candidate_edges(1_600_000)
    message = str(caught.value)
    assert "GiB" in message
    assert "--edge-construction knn" in message


def test_all_pairs_still_works_under_the_limit():
    edges = all_pairs_candidate_edges(50)
    assert len(edges) == 50 * 49 // 2


def test_knn_has_no_such_limit():
    """The whole point: it must scale where all-pairs cannot."""
    edges = knn_candidate_edges(nodes(2_000), k=10)
    assert len(edges) < 2_000 * 1_999 // 2 / 50


# -- dispatch ------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", EDGE_CONSTRUCTION_MODES)
def test_both_modes_dispatch(mode):
    edges = candidate_edges(nodes(80), mode=mode, k=10)
    assert edges and all(i < j for i, j in edges)


def test_an_unknown_mode_is_refused():
    with pytest.raises(EdgeConstructionError, match="unknown edge construction"):
        candidate_edges(nodes(10), mode="magic")


def test_all_pairs_over_a_subset_stays_in_the_caller_space():
    made = nodes(60)
    subset = [3, 9, 27]
    edges = candidate_edges(made, mode="all_pairs", indices=subset)
    assert sorted(edges) == [(3, 9), (3, 27), (9, 27)]


def test_knn_is_a_subset_of_all_pairs():
    """It removes candidates; it must never invent one."""
    made = nodes(100)
    knn = set(candidate_edges(made, mode="knn", k=10))
    every = set(candidate_edges(made, mode="all_pairs"))
    assert knn <= every


# -- embeddings ----------------------------------------------------------------- #

def test_a_missing_embedding_becomes_a_zero_row_not_a_dropped_node():
    """Row order has to match `indices` exactly or the caller's mapping breaks."""
    made = nodes(10) + nodes(5, embedding=False)
    matrix, missing = embedding_matrix(made)
    assert matrix.shape[0] == len(made)
    assert missing == 5
    assert not matrix[-1].any()


def test_ragged_embeddings_are_padded_to_one_width():
    made = nodes(4, dim=32) + nodes(4, dim=16)
    matrix, _missing = embedding_matrix(made)
    assert matrix.shape == (8, 32)


def test_nodes_without_embeddings_still_yield_edges():
    edges = knn_candidate_edges(nodes(40, embedding=False), k=5)
    assert edges, "zero embeddings cluster together, which is a dense component not a crash"


# -- wiring --------------------------------------------------------------------- #

def test_both_live_builders_use_the_shared_helper(repo_root):
    """The k-NN code already existed in `_build_graph_clustered`, which the live path never
    calls, gated behind a 5,000-node subgroup threshold. Both `_build_graph_standard`
    implementations must go through the shared module instead."""
    import os

    for name in ("UnclusteredDeepfakeDataloader", "HierarchicalDeepfakeDataloader"):
        source = open(os.path.join(repo_root, "dataloaders", f"{name}.py")).read()
        assert "knn_edges" in source, f"{name} does not use the shared helper"
        assert "for j in range(i + 1, n_nodes)" not in source, (
            f"{name} still enumerates all pairs inline"
        )
