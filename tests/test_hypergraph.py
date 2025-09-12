from graphs.HyperGraph import HyperGraph
from nodes.Node import Node
from edges.Edge import Edge

from conftest import build_dummy_graph


def test_hypergraph_len_and_nodes():
    graph, nodes, _ = build_dummy_graph(5)
    assert len(graph) == 5
    assert isinstance(graph.get_node(0), Node)
    assert graph.get_nodes()[0] is nodes[0]


def test_hypergraph_edge_list_and_add_edges_from_list():
    graph, nodes, edges = build_dummy_graph(4)
    initial_edges = set(tuple(sorted((e.get_node1().node_id, e.get_node2().node_id))) for e in edges)
    edge_list = set(graph.get_edge_list())
    assert initial_edges == edge_list

    # Add a new edge and ensure it's reflected
    new = Edge(nodes[0], nodes[2], x=None)
    nodes[0].add_edge(new)
    nodes[2].add_edge(new)
    el2 = set(graph.get_edge_list())
    assert tuple(sorted((nodes[0].node_id, nodes[2].node_id))) in el2

    # Build a fresh graph with fresh Node objects (no prior edges)
    fresh_nodes = [Node(n.node_id, 'train', {}, [], 0) for n in nodes]
    g2 = HyperGraph(nodes=fresh_nodes)
    g2.add_edges_from_list(list(initial_edges))
    assert set(g2.get_edge_list()) == initial_edges

