"""`Node.get_adjacent_nodes` is memoised; these tests attack the staleness risk.

A stale neighbour list would not raise -- it would quietly change which nodes a traversal
can reach, so training would still "work" and every number would be subtly wrong. Each
mutation path in the codebase therefore gets a test that the cache notices it.
"""

import pickle

import pytest

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from nodes.Node import Node


def _node(node_id):
    return Node(node_id=node_id, split="train", data=None, edges=[], label=0)


def _link(left, right):
    edge = Edge(left, right, None)
    left.add_edge(edge)
    right.add_edge(edge)
    return edge


def test_repeated_calls_agree_and_reuse_the_cache():
    a, b = _node("a"), _node("b")
    _link(a, b)
    first = a.get_adjacent_nodes()
    second = a.get_adjacent_nodes()
    assert first == [b]
    assert second is first  # actually cached, not just recomputed to the same value


def test_add_edge_is_seen():
    a, b, c = _node("a"), _node("b"), _node("c")
    _link(a, b)
    assert a.get_adjacent_nodes() == [b]
    _link(a, c)  # length changes
    assert a.get_adjacent_nodes() == [b, c]


def test_in_place_removal_is_seen():
    # The GraphReductionManager path: node1.edges.remove(edge), no method call.
    a, b, c = _node("a"), _node("b"), _node("c")
    ab = _link(a, b)
    _link(a, c)
    assert set(a.get_adjacent_nodes()) == {b, c}
    a.edges.remove(ab)
    assert a.get_adjacent_nodes() == [c]


def test_reassignment_is_seen():
    # The dataloader path: node.edges = [] between grid-search iterations.
    a, b = _node("a"), _node("b")
    _link(a, b)
    assert a.get_adjacent_nodes() == [b]
    a.edges = []
    assert a.get_adjacent_nodes() == []


def test_reordering_by_canonicalize_is_seen():
    a, b, c = _node("a"), _node("b"), _node("c")
    _link(a, b)
    _link(a, c)
    before = list(a.get_adjacent_nodes())
    a.edges = list(reversed(a.edges))  # same length, new list object
    assert a.get_adjacent_nodes() == list(reversed(before))


def test_edge_repointing_is_seen_on_the_other_endpoint():
    # Identity and length both unchanged: only the explicit invalidation catches this.
    a, b, c = _node("a"), _node("b"), _node("c")
    edge = _link(a, b)
    assert a.get_adjacent_nodes() == [b]
    edge.set_node2(c)
    assert a.get_adjacent_nodes() == [c]


def test_set_nodes_is_seen():
    a, b, c, d = _node("a"), _node("b"), _node("c"), _node("d")
    edge = _link(a, b)
    assert a.get_adjacent_nodes() == [b]
    edge.set_nodes(a, d)
    assert a.get_adjacent_nodes() == [d]
    assert c not in a.get_adjacent_nodes()


def test_remove_nodes_is_seen_by_surviving_neighbours():
    a, b, c = _node("a"), _node("b"), _node("c")
    _link(a, b)
    _link(a, c)
    graph = HyperGraph([a, b, c])
    assert set(a.get_adjacent_nodes()) == {b, c}
    graph.remove_nodes([b])
    assert a.get_adjacent_nodes() == [c]


def test_duplicate_neighbours_are_preserved():
    # Two parallel edges must still yield the neighbour twice, as before memoising.
    a, b = _node("a"), _node("b")
    _link(a, b)
    _link(a, b)
    assert a.get_adjacent_nodes() == [b, b]


def test_self_loop_is_excluded():
    a = _node("a")
    edge = Edge(a, a, None)
    a.add_edge(edge)
    assert a.get_adjacent_nodes() == []


def test_degree_matches_the_neighbour_list():
    a, b, c = _node("a"), _node("b"), _node("c")
    _link(a, b)
    _link(a, c)
    assert a.get_degree() == len(a.get_adjacent_nodes()) == 2
    a.edges.clear()
    assert a.get_degree() == 0


def test_cache_is_not_pickled():
    a, b = _node("a"), _node("b")
    _link(a, b)
    a.get_adjacent_nodes()  # populate
    assert a.__dict__.get('_adjacency_cache') is not None
    state = a.__getstate__()
    assert '_adjacency_cache' not in state


def test_unpickled_node_rebuilds_rather_than_raising():
    a, b = _node("a"), _node("b")
    _link(a, b)
    a.get_adjacent_nodes()
    restored = pickle.loads(pickle.dumps(a))
    # __init__ never ran on `restored`; the class-level default must carry it.
    assert [n.node_id for n in restored.get_adjacent_nodes()] == ["b"]


def test_explicit_invalidation_forces_a_rebuild():
    a, b = _node("a"), _node("b")
    _link(a, b)
    first = a.get_adjacent_nodes()
    a.invalidate_adjacency_cache()
    assert a.get_adjacent_nodes() is not first
    assert a.get_adjacent_nodes() == [b]


@pytest.mark.parametrize("size", [1, 2, 8, 64])
def test_matches_an_uncached_reference_implementation(size):
    hub = _node("hub")
    spokes = [_node(f"s{i}") for i in range(size)]
    for spoke in spokes:
        _link(hub, spoke)

    def uncached(node):
        out = []
        for edge in node.edges:
            for other in edge.get_nodes():
                if other != node:
                    out.append(other)
        return out

    assert hub.get_adjacent_nodes() == uncached(hub)
    hub.edges.remove(hub.edges[0])
    assert hub.get_adjacent_nodes() == uncached(hub)
