from graphs.HyperGraph import HyperGraph
from nodes.Node import Node
from edges.Edge import Edge


def make_node(node_id):
    return Node(node_id=node_id, split="train", data=[], edges=[], label=0)


def make_triangle_graph():
    a = make_node("a")
    b = make_node("b")
    c = make_node("c")
    e1 = Edge(a, b, x=None)
    e2 = Edge(b, c, x=None)
    e3 = Edge(c, a, x=None)
    for n, e in [(a,e1),(b,e1),(b,e2),(c,e2),(c,e3),(a,e3)]:
        n.add_edge(e)
    return [a,b,c]


def test_hypergraph_basic_operations():
    nodes = [make_node("n1"), make_node("n2")]
    hg = HyperGraph(nodes)
    assert len(hg) == 2
    assert hg.get_node(0) is nodes[0]
    assert hg.get_nodes() == nodes

    n3 = make_node("n3")
    hg.add_node(n3)
    assert len(hg) == 3
    hg.set_node(1, n3)
    assert hg.get_node(1) is n3
    hg.remove_node(1)
    assert len(hg) == 2


def test_edge_list_and_num_edges_triangle():
    nodes = make_triangle_graph()
    hg = HyperGraph(nodes)
    edges = hg.get_edge_list()
    assert set(edges) == {("a","b"),("b","c"),("a","c")}
    assert hg.num_edges() == 3


def test_add_edges_from_list():
    a = make_node("a")
    b = make_node("b")
    c = make_node("c")
    hg = HyperGraph([a,b,c])
    hg.add_edges_from_list([("a","b"),("b","c")])
    edges = set(hg.get_edge_list())
    assert edges == {("a","b"),("b","c")}

