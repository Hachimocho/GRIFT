import inspect
import os
import sys
import random
from typing import Dict, List, Tuple, Type

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nodes.Node import Node
from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph

# Import all exported classes from these packages (their __init__ files expose classes)
import nodes as nodes_pkg
import edges as edges_pkg
import traversals as traversals_pkg


def seed_all(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)


seed_all()


def get_node_classes() -> Dict[str, Type[Node]]:
    classes = {}
    for name, obj in inspect.getmembers(nodes_pkg, inspect.isclass):
        try:
            if issubclass(obj, Node):
                classes[name] = obj
        except Exception:
            continue
    return classes


def get_edge_classes() -> Dict[str, Type[Edge]]:
    classes = {}
    for name, obj in inspect.getmembers(edges_pkg, inspect.isclass):
        try:
            if issubclass(obj, Edge):
                classes[name] = obj
        except Exception:
            continue
    return classes


def get_traversal_classes() -> Dict[str, type]:
    classes = {}
    for name, obj in inspect.getmembers(traversals_pkg, inspect.isclass):
        # Base class name is Traversal; include subclasses we know are usable in tests
        if name in {"RandomTraversal", "ComprehensiveTraversal", "IValueTraversal", "IValueTraversalSubcluster"}:
            classes[name] = obj
    return classes


def make_dummy_attribute(value_dim: int = 4) -> Dict[str, object]:
    return {
        "race": random.choice(["A", "B"]),
        "gender": random.choice(["M", "F"]),
        "embedding": np.random.rand(value_dim),
        "blur": float(np.random.randint(0, 100)),
    }


def create_dummy_nodes(num_nodes: int = 6) -> List[Node]:
    # Prefer AttributeNode if available; fall back to base Node
    node_classes = get_node_classes()
    AttributeNode = node_classes.get("AttributeNode", None)
    nodes: List[Node] = []
    for i in range(num_nodes):
        node_id = f"n{i}"
        split = random.choice(["train", "val", "test"])
        if AttributeNode is not None:
            node = AttributeNode(
                node_id=node_id,
                split=split,
                data={"idx": i},
                edges=[],
                label=int(i % 2),
                attributes=make_dummy_attribute(),
                threshold=75,
            )
        else:
            node = Node(
                node_id=node_id,
                split=split,
                data={"idx": i},
                edges=[],
                label=int(i % 2),
            )
        nodes.append(node)
    return nodes


def connect_ring(nodes: List[Node]) -> List[Edge]:
    edges: List[Edge] = []
    n = len(nodes)
    for i in range(n):
        n1 = nodes[i]
        n2 = nodes[(i + 1) % n]
        e = Edge(n1, n2, x={"w": 1.0})
        n1.add_edge(e)
        n2.add_edge(e)
        edges.append(e)
    return edges


def build_dummy_graph(num_nodes: int = 6) -> Tuple[HyperGraph, List[Node], List[Edge]]:
    nodes = create_dummy_nodes(num_nodes)
    edges = connect_ring(nodes)
    graph = HyperGraph(nodes)
    return graph, nodes, edges

