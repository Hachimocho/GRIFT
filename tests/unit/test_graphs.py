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


def test_get_edge_list_is_deterministically_ordered():
    """`get_edge_list` must not depend on set iteration order.

    It builds a set of (node_id, node_id) string tuples. Because `Node.__hash__`
    hashes a *string* node_id, returning `list(edge_set)` gave an order that varied
    with PYTHONHASHSEED across processes -- and this list is written to the pickle
    graph cache, so a cache written under one hash seed replayed its edges in a
    different order than one written under another. Edge order then determines
    traversal tie-breaks.
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


def test_k_hop_subgraph_one_hop_returns_the_neighbors():
    """`k_hop_subgraph(node, 1)` used to raise KeyError unconditionally.

    For k=1 the accumulated set holds only the seed's neighbors, so the trailing
    `k_hop_nodes.remove(node)` could not find the seed. It happened to work for
    k>=2, where the seed is re-added as a neighbour-of-a-neighbour. Now uses
    `discard`.
    """
    graph, nodes = _line_graph()
    one_hop = graph.k_hop_subgraph(nodes[2], 1)
    assert {node.node_id for node in one_hop.get_nodes()} == {"1", "3"}


def test_k_hop_subgraph_excludes_the_seed_node():
    graph, nodes = _line_graph()
    two_hop = graph.k_hop_subgraph(nodes[2], 2)
    assert nodes[2].node_id not in {node.node_id for node in two_hop.get_nodes()}


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_k_hop_subgraph_ordering_is_deterministic(hops):
    graph, nodes = _line_graph()
    first = [node.node_id for node in graph.k_hop_subgraph(nodes[2], hops).get_nodes()]
    second = [node.node_id for node in graph.k_hop_subgraph(nodes[2], hops).get_nodes()]
    assert first == second, "k-hop expansion walks a set of Node objects, so its order must be pinned"
    assert first == sorted(first), "and that order should be by node id"


def test_k_hop_list_ordering_is_deterministic():
    graph, nodes = _line_graph()
    first = [node.node_id for node in graph.k_hop_list(nodes[2], 2)]
    second = [node.node_id for node in graph.k_hop_list(nodes[2], 2)]
    assert first == second
    assert first[0] == nodes[2].node_id, "the seed node leads the k-hop list"


def test_canonicalize_edge_order_sorts_adjacency_by_peer_id():
    graph, nodes = _line_graph()
    middle = nodes[2]
    middle.edges.reverse()
    graph.canonicalize_edge_order()
    peers = [
        (edge.get_nodes()[1] if edge.get_nodes()[0] is middle else edge.get_nodes()[0]).node_id
        for edge in middle.edges
    ]
    assert peers == sorted(peers)


def test_canonicalize_edge_order_is_idempotent():
    graph, _ = _line_graph()
    graph.canonicalize_edge_order()
    first = [[id(edge) for edge in node.edges] for node in graph.nodes]
    graph.canonicalize_edge_order()
    assert [[id(edge) for edge in node.edges] for node in graph.nodes] == first
