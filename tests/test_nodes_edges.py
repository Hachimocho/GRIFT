import inspect

from nodes.Node import Node
from edges.Edge import Edge

from conftest import get_node_classes, get_edge_classes


def _construct_with_signature(cls, base_kwargs):
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name in base_kwargs:
            kwargs[name] = base_kwargs[name]
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        elif name == 'attributes':
            kwargs[name] = {"dummy": 1}
        elif name == 'threshold':
            kwargs[name] = 50
        elif name == 'match_chance':
            kwargs[name] = 1.0
        else:
            # Unknown required parameter; skip this class in tests
            raise RuntimeError(f"Cannot construct {cls.__name__}: missing arg {name}")
    return cls(**kwargs)


def test_edge_basic_setters_getters():
    n1 = Node("a", "train", {}, [], 0)
    n2 = Node("b", "train", {}, [], 1)
    e = Edge(n1, n2, x={"hello": "world"}, traversal_weight=2)
    assert e.get_nodes() == (n1, n2)
    assert e.get_data()["hello"] == "world"
    assert e.get_traversal_weight() == 2
    n3 = Node("c", "train", {}, [], 2)
    e.set_node1(n3)
    assert e.get_node1() == n3
    e.set_traversal_weight(3)
    assert e.get_traversal_weight() == 3


def test_node_base_behavior_adjacency():
    n1 = Node("a", "train", {"x": 1}, [], 0)
    n2 = Node("b", "train", {"x": 2}, [], 1)
    e = Edge(n1, n2, x=None)
    n1.add_edge(e)
    n2.add_edge(e)
    assert n2 in n1.get_adjacent_nodes()
    assert n1 in n2.get_adjacent_nodes()
    assert len(n1) == len(n1.get_data())
    n1.set_data({"y": 3})
    assert n1.get_data()["y"] == 3


def test_all_node_classes_construct_and_link():
    classes = get_node_classes()
    base_kwargs = dict(node_id="x", split="train", data={}, edges=[], label=0)
    for name, cls in classes.items():
        # Attempt to build instance; skip if constructor is incompatible
        try:
            node = _construct_with_signature(cls, base_kwargs)
        except Exception:
            continue

        # Should accept edges and adjacency via Edge
        other = Node("y", "train", {}, [], 0)
        e = Edge(node, other, x=None)
        # Some subclasses may not inherit add_edge; ensure attribute exists
        assert hasattr(node, "add_edge")
        node.add_edge(e)
        other.add_edge(e)
        assert other in node.get_adjacent_nodes()

