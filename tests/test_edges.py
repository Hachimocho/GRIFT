from edges.Edge import Edge
from nodes.Node import Node


class DummyNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id=node_id, split="train", data=None, edges=[], label=node_id)


def test_edge_getters_setters_and_weight():
    n1 = DummyNode("a")
    n2 = DummyNode("b")

    e = Edge(n1, n2, x={"v": 1}, traversal_weight=3)

    assert e.get_node1() is n1
    assert e.get_node2() is n2
    assert e.get_nodes() == (n1, n2)

    assert e.get_data() == {"v": 1}
    e.set_data({"v": 2})
    assert e.get_data() == {"v": 2}

    assert e.get_traversal_weight() == 3
    e.set_traversal_weight(7)
    assert e.get_traversal_weight() == 7

    n3 = DummyNode("c")
    e.set_node1(n3)
    assert e.get_node1() is n3
    e.set_nodes(n1, n2)
    assert e.get_nodes() == (n1, n2)

