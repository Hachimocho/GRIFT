from nodes.RandomNode import RandomNode
from nodes.Node import Node


def test_random_node_match_probability_one():
    rn = RandomNode(split="train", data=None, edges=[], label=0, match_chance=1.0)
    assert rn.match(Node(node_id="x", split="train", data=None, edges=[], label=0)) is True


def test_random_node_match_probability_zero():
    rn = RandomNode(split="train", data=None, edges=[], label=0, match_chance=0.0)
    # Could be False always
    assert rn.match(Node(node_id="x", split="train", data=None, edges=[], label=0)) is False

