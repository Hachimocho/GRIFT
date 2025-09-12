import pytest

from .conftest import build_dummy_graph, get_traversal_classes


def test_random_and_comprehensive_traversals_collect_nodes():
    graph, nodes, _ = build_dummy_graph(8)
    tclasses = get_traversal_classes()

    if 'RandomTraversal' in tclasses:
        T = tclasses['RandomTraversal']
        t = T(graph=graph, num_pointers=2, num_steps=10)
        batch = t.traverse(batch_size=16)
        assert isinstance(batch, list)
        # Should return at most all nodes
        assert all(n in nodes for n in batch)

    if 'ComprehensiveTraversal' in tclasses:
        T = tclasses['ComprehensiveTraversal']
        t = T(graph=graph, num_pointers=1, num_steps=None)
        seen = []
        while True:
            b = t.traverse(batch_size=3)
            if not b:
                break
            seen.extend(b)
        assert set(seen).issubset(set(nodes))


@pytest.mark.filterwarnings('ignore:.*louvain.*')
def test_ivalue_traversal_runs_without_trainer():
    tclasses = get_traversal_classes()
    if 'IValueTraversal' not in tclasses:
        pytest.skip("IValueTraversal not available")
    graph, nodes, _ = build_dummy_graph(10)
    T = tclasses['IValueTraversal']
    t = T(graph=graph, num_pointers=2, num_steps=3, trainer=None)
    out = t.traverse(batch_size=8)
    # May be empty if not enough AttributeNodes; only assert type
    assert isinstance(out, list)

