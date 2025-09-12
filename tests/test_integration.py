import pytest

from .conftest import build_dummy_graph, get_traversal_classes


def test_integration_dummy_graph_with_random_traversal():
    graph, nodes, _ = build_dummy_graph(12)
    tclasses = get_traversal_classes()
    if 'RandomTraversal' not in tclasses:
        pytest.skip('RandomTraversal not available')

    T = tclasses['RandomTraversal']
    t = T(graph=graph, num_pointers=3, num_steps=20)

    all_seen = set()
    for _ in range(5):
        batch = t.traverse(batch_size=10)
        if not batch:
            break
        for n in batch:
            all_seen.add(n)
    assert len(all_seen) > 0
    assert all(n in nodes for n in all_seen)

