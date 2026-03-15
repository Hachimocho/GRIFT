from nodes.Node import Node
from nodes.atrnode import AttributeNode


def make_node(node_id="n1", split="train", data=[1,2,3], label=0):
    return Node(node_id=node_id, split=split, data=data, edges=[], label=label)


def test_node_len_and_data_accessors():
    n = make_node()
    assert len(n) == 3
    assert n.get_data() == [1,2,3]
    n.set_data([4])
    assert n.get_data() == [4]


def test_node_equality_and_hash():
    a = make_node(node_id="id1")
    b = make_node(node_id="id1")
    c = make_node(node_id="id2")
    assert a == b
    assert a != c
    s = {a, b, c}
    assert len(s) == 2


def test_node_split_and_label():
    n = make_node(split="val", label=1)
    assert n.get_split() == "val"
    n.set_split("test")
    assert n.get_split() == "test"
    assert n.get_label() == 1


def test_node_adjacent_nodes():
    from edges.Edge import Edge
    a = make_node("a")
    b = make_node("b")
    c = make_node("c")
    e1 = Edge(a, b, x=None)
    e2 = Edge(a, c, x=None)
    a.add_edge(e1)
    a.add_edge(e2)
    b.add_edge(e1)
    c.add_edge(e2)
    adj = set(a.get_adjacent_nodes())
    assert adj == {b, c}


def test_attribute_node_similarity_and_match():
    import numpy as np
    attrs1 = {
        'face_embedding': np.ones(4, dtype=float),
        'blur': 10.0,
        'symmetry_overall': 0.9,
        'emotion_happy': 1.0,
    }
    attrs2 = {
        'face_embedding': np.ones(4, dtype=float),
        'blur': 15.0,
        'symmetry_overall': 0.85,
        'emotion_happy': 1.0,
    }
    an1 = AttributeNode("a", "train", None, [], 0, attrs1, threshold=50)
    an2 = AttributeNode("b", "train", None, [], 0, attrs2, threshold=50)
    assert an1.match(an2) is True

    an1.remove_attribute('emotion_happy')
    an1.add_attribute(1.0, 'emotion_happy')
    assert 'emotion_happy' in an1.attributes

