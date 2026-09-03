"""Node and Edge contracts.

Ported from the abandoned unit-test branches, plus pins for two subtle
behaviors that the uncertainty work depends on (see the last two tests).
"""

import numpy as np
import pytest

from edges.Edge import Edge
from nodes.atrnode import AttributeNode
from nodes.Node import Node
from nodes.RandomNode import RandomNode


def make_node(node_id="n1", split="train", data=(1, 2, 3), label=0):
    return Node(node_id=node_id, split=split, data=list(data), edges=[], label=label)


def test_node_len_and_data_accessors():
    node = make_node()
    assert len(node) == 3
    assert node.get_data() == [1, 2, 3]
    node.set_data([4])
    assert node.get_data() == [4]


def test_node_equality_and_hash_by_id():
    first = make_node(node_id="id1")
    duplicate = make_node(node_id="id1")
    other = make_node(node_id="id2")
    assert first == duplicate
    assert first != other
    assert len({first, duplicate, other}) == 2


def test_node_split_and_label():
    node = make_node(split="val", label=1)
    assert node.get_split() == "val"
    node.set_split("test")
    assert node.get_split() == "test"
    assert node.get_label() == 1


def test_node_adjacent_nodes():
    a, b, c = make_node("a"), make_node("b"), make_node("c")
    ab, ac = Edge(a, b, x=None), Edge(a, c, x=None)
    a.add_edge(ab)
    a.add_edge(ac)
    b.add_edge(ab)
    c.add_edge(ac)
    assert set(a.get_adjacent_nodes()) == {b, c}


def test_node_neighbor_aliases():
    a, b = make_node("a"), make_node("b")
    edge = Edge(a, b, x=None)
    a.add_edge(edge)
    b.add_edge(edge)
    assert a.get_neighbors() == a.get_adjacent_nodes()
    assert a.get_degree() == 1


def test_get_adjacent_nodes_counts_duplicate_edges_separately():
    """Pins current behavior: adjacency has edge multiplicity, not set semantics.

    `get_adjacent_nodes` walks `self.edges` and appends one entry per edge, so a
    duplicated edge yields a duplicated neighbor and inflates `get_degree()`.
    That in turn inflates the graph-distance degree penalty, which divides by
    sqrt(degree + 1). Recorded here so a future dedupe is a deliberate change
    rather than a surprise.
    """
    a, b = make_node("a"), make_node("b")
    for _ in range(2):
        edge = Edge(a, b, x=None)
        a.add_edge(edge)
        b.add_edge(edge)
    assert a.get_adjacent_nodes() == [b, b]
    assert a.get_degree() == 2


def test_edge_getters_setters_and_weight():
    n1, n2, n3 = make_node("a"), make_node("b"), make_node("c")
    edge = Edge(n1, n2, x={"v": 1}, traversal_weight=3)

    assert edge.get_node1() is n1
    assert edge.get_node2() is n2
    assert edge.get_nodes() == (n1, n2)

    assert edge.get_data() == {"v": 1}
    edge.set_data({"v": 2})
    assert edge.get_data() == {"v": 2}

    assert edge.get_traversal_weight() == 3
    edge.set_traversal_weight(7)
    assert edge.get_traversal_weight() == 7

    edge.set_node1(n3)
    assert edge.get_node1() is n3
    edge.set_nodes(n1, n2)
    assert edge.get_nodes() == (n1, n2)


def test_random_node_match_bounds():
    other = make_node("x")
    assert RandomNode(split="train", data=None, edges=[], label=0, match_chance=1.0).match(other) is True
    assert RandomNode(split="train", data=None, edges=[], label=0, match_chance=0.0).match(other) is False


def test_attribute_node_match_on_similar_attributes():
    shared_embedding = np.ones(4, dtype=np.float32)
    first = AttributeNode("a", "train", None, [], 0, {
        "face_embedding": shared_embedding,
        "blur": 10.0,
        "symmetry_overall": 0.9,
        "emotion_happy": 1.0,
    }, threshold=50)
    second = AttributeNode("b", "train", None, [], 0, {
        "face_embedding": shared_embedding.copy(),
        "blur": 15.0,
        "symmetry_overall": 0.85,
        "emotion_happy": 1.0,
    }, threshold=50)
    assert first.match(second) is True


def test_attribute_node_add_remove_attribute():
    node = AttributeNode("a", "train", None, [], 0, {"emotion_happy": 1.0}, threshold=50)
    node.remove_attribute("emotion_happy")
    assert "emotion_happy" not in node.attributes
    node.add_attribute(1.0, "emotion_happy")
    assert node.attributes["emotion_happy"] == 1.0


def test_attribute_node_match_returns_false_with_no_common_attributes():
    first = AttributeNode("a", "train", None, [], 0, {"blur": 1.0}, threshold=50)
    second = AttributeNode("b", "train", None, [], 0, {"contrast": 1.0}, threshold=50)
    assert first.match(second) is False


@pytest.mark.parametrize("value_type", [np.int64, np.int32, int])
def test_attribute_node_categorical_similarity_uses_exact_match(value_type):
    """Pins that numpy-int demographics fall through to exact match here.

    `compute_similarity` has the same `isinstance(v, (int, float))` blind spot as
    `graph_distance.py`, but here the fallthrough is *correct*: exact match is
    the right semantics for a categorical label code. Pinned so the benign case
    stays benign if that isinstance check is ever "fixed" in the wrong direction.
    """
    node = AttributeNode("a", "train", None, [], 0, {}, threshold=50)
    # Note: for numpy inputs the `value1 == value2` fallthrough returns np.bool_,
    # not Python bool, so compare by value rather than identity.
    assert bool(node.compute_similarity(node, "Ground Truth Race", value_type(2), value_type(2))) is True
    assert bool(node.compute_similarity(node, "Ground Truth Race", value_type(2), value_type(3))) is False
