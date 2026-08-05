"""HyperGraph operations, including the edge-ordering properties traversals rely on."""

import pytest

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from nodes.Node import Node


def make_node(node_id):
    return Node(node_id=node_id, split="train", data=[], edges=[], label=0)


def make_triangle():
    a, b, c = make_node("a"), make_node("b"), make_node("c")
    for left, right in ((a, b), (b, c), (c, a)):
        edge = Edge(left, right, x=None)
        left.add_edge(edge)
        right.add_edge(edge)
    return [a, b, c]


def test_hypergraph_basic_operations():
    nodes = [make_node("n1"), make_node("n2")]
    graph = HyperGraph(nodes)
    assert len(graph) == 2
    assert graph.get_node(0) is nodes[0]
    assert graph.get_nodes() == nodes

    third = make_node("n3")
    graph.add_node(third)
    assert len(graph) == 3
    graph.set_node(1, third)
    assert graph.get_node(1) is third
    graph.remove_node(1)
    assert len(graph) == 2


def test_node_data_map_populated_at_construction():
    nodes = [make_node("n1"), make_node("n2")]
    graph = HyperGraph(nodes)
    assert graph._node_data_map == {"n1": nodes[0], "n2": nodes[1]}


def test_edge_list_and_num_edges_triangle():
    graph = HyperGraph(make_triangle())
    assert set(graph.get_edge_list()) == {("a", "b"), ("b", "c"), ("a", "c")}
    assert graph.num_edges() == 3


def make_attr_node(node_id):
    """A node that is *truthy*. See test_add_edges_from_list_node_truthiness_trap."""
    from nodes.atrnode import AttributeNode
    return AttributeNode(node_id, "train", None, [], 0, {"blur": 1.0}, threshold=50)


def test_add_edges_from_list_roundtrip():
    nodes = [make_attr_node("a"), make_attr_node("b"), make_attr_node("c")]
    graph = HyperGraph(nodes)
    graph.add_edges_from_list([("a", "b"), ("b", "c")])
    assert set(graph.get_edge_list()) == {("a", "b"), ("b", "c")}


def test_add_edges_from_list_node_truthiness_trap():
    """`add_edges_from_list` tests nodes for truthiness, not for None.

    It does `if node1 and node2:`, and `Node.__len__` returns `len(self.data)`,
    so a base Node whose data is empty is *falsy* and every edge referencing it
    is silently skipped -- reported only as "N edges skipped" in stdout.

    Production is unaffected because `AttributeNode.__len__` returns its
    attribute count, which is non-zero. Pinned so the latent footgun is visible:
    the correct guard is `if node1 is not None and node2 is not None`.
    """
    nodes = [make_node("a"), make_node("b")]  # base Node, data == [] -> falsy
    assert not nodes[0], "precondition: a base Node with empty data is falsy"
    graph = HyperGraph(nodes)
    graph.add_edges_from_list([("a", "b")])
    assert graph.get_edge_list() == [], "current (surprising) behavior: edge silently skipped"


@pytest.mark.xfail(
    strict=True,
    reason="C4: get_edge_list returns list(set) of string-id tuples, so its order is "
           "PYTHONHASHSEED-dependent. Fixed by sorting in group C.",
)
def test_get_edge_list_is_deterministically_ordered():
    """`get_edge_list` must not depend on set iteration order.

    It builds a set of (node_id, node_id) string tuples and returns
    `list(edge_set)`. Because `Node.__hash__` hashes a *string* node_id, that
    order varies with PYTHONHASHSEED across processes -- and the result is used
    to write the pickle graph cache, so a cache written under one hash seed
    replays its edges in a different order than one written under another.
    """
    graph = HyperGraph(make_triangle())
    assert graph.get_edge_list() == sorted(graph.get_edge_list())


def test_get_random_node_is_a_member():
    graph = HyperGraph(make_triangle())
    assert graph.get_random_node() in graph.get_nodes()


def _line_graph(count=6):
    nodes = [make_node(str(index)) for index in range(count)]
    for index in range(count - 1):
        edge = Edge(nodes[index], nodes[index + 1], x=None)
        nodes[index].add_edge(edge)
        nodes[index + 1].add_edge(edge)
    return HyperGraph(nodes), nodes


def test_k_hop_subgraph_one_hop_raises():
    """`k_hop_subgraph(node, 1)` is unconditionally broken.

    For k=1 the accumulated set holds only the seed's neighbors, so the trailing
    `k_hop_nodes.remove(node)` raises KeyError. It happens to work for k>=2
    because the seed gets re-added as a neighbour-of-a-neighbour. Both
    `k_hop_subgraph` and `k_hop_list` have no callers anywhere in the repo, so
    this is dead code -- pinned rather than fixed, so if it is ever put to use
    the breakage is already documented.
    """
    graph, nodes = _line_graph()
    with pytest.raises(KeyError):
        graph.k_hop_subgraph(nodes[2], 1)


def test_k_hop_subgraph_ordering_is_deterministic():
    graph, nodes = _line_graph()
    first = [node.node_id for node in graph.k_hop_subgraph(nodes[2], 2).get_nodes()]
    second = [node.node_id for node in graph.k_hop_subgraph(nodes[2], 2).get_nodes()]
    assert first == second, "k-hop expansion walks a set of Node objects, so its order must be pinned"
