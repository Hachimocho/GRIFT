"""The I-value candidate pool.

Without it the argmax ranges only over the current node's graph neighbours. On the measured
full-scale graph that is ~8 nodes joined *by similarity*, so their I-values are correlated
and the effective choice is narrower still -- a selection pressure weak enough to blunt even
a perfect estimator. The pool widens the choice while keeping the draw on the traversal's own
seeded stream, so reproducibility is unaffected.
"""

import pytest

from tests.helpers.factories import DummyTrainer, build_ring_graph
from traversals.IValueTraversal import IValueTraversal


def _traversal(graph, trainer, pool=0, **kwargs):
    return IValueTraversal(graph, num_pointers=1, num_steps=10_000, trainer=trainer,
                           candidate_pool=pool, **kwargs)


def test_default_is_off_so_existing_runs_are_unchanged():
    graph, _nodes, _edges = build_ring_graph(count=12)
    assert _traversal(graph, DummyTrainer()).candidate_pool == 0


@pytest.mark.parametrize("pool", [0, 1, 8, 64])
def test_pool_is_normalised_to_a_non_negative_int(pool):
    graph, _nodes, _edges = build_ring_graph(count=12)
    assert _traversal(graph, DummyTrainer(), pool=pool).candidate_pool == pool


@pytest.mark.parametrize("bad", [-5, None])
def test_nonsense_pool_sizes_collapse_to_off(bad):
    graph, _nodes, _edges = build_ring_graph(count=12)
    assert _traversal(graph, DummyTrainer(), pool=bad).candidate_pool == 0


def test_candidates_without_a_pool_are_exactly_the_neighbours():
    graph, nodes, _edges = build_ring_graph(count=12)
    traversal = _traversal(graph, DummyTrainer(), pool=0)
    neighbours = nodes[0].get_adjacent_nodes()
    assert traversal._candidates(neighbours, set()) == neighbours


def test_a_pool_widens_the_choice():
    graph, nodes, _edges = build_ring_graph(count=40)
    traversal = _traversal(graph, DummyTrainer(), pool=10)
    neighbours = nodes[0].get_adjacent_nodes()
    widened = traversal._candidates(neighbours, set())
    assert len(widened) > len(neighbours)
    # The neighbours must still be considered, not replaced.
    for neighbour in neighbours:
        assert neighbour in widened


def test_no_duplicates_and_visited_nodes_are_excluded():
    graph, nodes, _edges = build_ring_graph(count=40)
    traversal = _traversal(graph, DummyTrainer(), pool=30)
    neighbours = nodes[0].get_adjacent_nodes()
    visited = {nodes[5], nodes[6], nodes[7]}
    widened = traversal._candidates(neighbours, visited)
    ids = [node.node_id for node in widened]
    assert len(ids) == len(set(ids)), "a node must not be offered twice"
    for node in visited:
        assert node not in widened[len(neighbours):], "already-visited nodes must be skipped"


def test_the_draw_comes_from_the_traversals_seeded_stream():
    """Same stream state -> same pool, which is what keeps a run reproducible.

    Asserted by injecting the stream rather than by building two traversals and comparing:
    `Traversal.rng` hands out one *shared* memoised stream per class name, so a second
    instance continues the first one's sequence by design. Determinism here is a property of
    a run, not of an instance -- the thing worth pinning is that the draw goes through
    `self.rng` at all, rather than through the process-global `random`.
    """
    import random

    graph, nodes, _edges = build_ring_graph(count=60)
    neighbours = nodes[0].get_adjacent_nodes()

    traversal = _traversal(graph, DummyTrainer(), pool=15)
    traversal._rng = random.Random(1234)
    first = [n.node_id for n in traversal._candidates(neighbours, set())]

    traversal._rng = random.Random(1234)
    second = [n.node_id for n in traversal._candidates(neighbours, set())]

    assert first == second
    # And a different stream state gives a different pool, so it is really being consulted.
    traversal._rng = random.Random(999)
    assert [n.node_id for n in traversal._candidates(neighbours, set())] != first


def test_selection_can_reach_a_high_value_node_outside_the_neighbourhood():
    """The point of the pool: a strong candidate the walk cannot see locally."""
    graph, nodes, _edges = build_ring_graph(count=60)
    far = nodes[30].node_id
    trainer = DummyTrainer(i_values={far: 1.0})
    traversal = _traversal(graph, trainer, pool=59)
    neighbours = nodes[0].get_adjacent_nodes()
    assert far not in {n.node_id for n in neighbours}
    widened = traversal._candidates(neighbours, set())
    assert far in {n.node_id for n in widened}
