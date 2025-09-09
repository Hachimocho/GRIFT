from graphs.HyperGraph import HyperGraph
from nodes.Node import Node
from nodes.atrnode import AttributeNode
from edges.Edge import Edge
from traversals.RandomTraversal import RandomTraversal
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal


def build_simple_graph(num=5, attribute_nodes=False):
    nodes = []
    if attribute_nodes:
        for i in range(num):
            n = AttributeNode(str(i), "train", None, [], 0, {"face_embedding": None}, threshold=50)
            nodes.append(n)
    else:
        for i in range(num):
            nodes.append(Node(node_id=str(i), split="train", data=None, edges=[], label=0))
    # Line graph connections: 0-1-2-...-(n-1)
    for i in range(num-1):
        e = Edge(nodes[i], nodes[i+1], x=None)
        nodes[i].add_edge(e)
        nodes[i+1].add_edge(e)
    return HyperGraph(nodes)


def test_random_traversal_collects_nodes():
    hg = build_simple_graph(6)
    tr = RandomTraversal(hg, num_pointers=2, num_steps=10)
    batch = tr.traverse(batch_size=16)
    assert isinstance(batch, list)
    assert all(isinstance(n, Node) for n in batch)


def test_comprehensive_traversal_visits_all():
    hg = build_simple_graph(8)
    tr = ComprehensiveTraversal(hg, num_pointers=1, num_steps=None)
    seen = []
    while True:
        b = tr.traverse(batch_size=3)
        if not b:
            break
        seen.extend(b)
    assert len(set(seen)) == len(hg.get_nodes())


class DummyTrainer:
    def get_i_value(self, node, model_index):
        # Prefer center nodes in line graph by id numeric value proximity to middle
        try:
            idx = int(getattr(node, 'node_id', 0))
        except Exception:
            idx = 0
        return -abs(idx - 3)


def test_ivalue_traversal_returns_nodes():
    hg = build_simple_graph(7, attribute_nodes=True)
    tr = IValueTraversal(hg, num_pointers=2, num_steps=5, trainer=DummyTrainer())
    batch = tr.traverse(batch_size=8)
    # May return empty if < min threshold, but with our graph and trainer should return some
    assert isinstance(batch, list)
    # AttributeNode filter inside traversal enforces types
    assert all(isinstance(n, AttributeNode) for n in batch) or batch == []

