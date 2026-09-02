"""Cumulative comprehensive coverage.

The default `ComprehensiveTraversal` clears `visited` in `reset_pointers`, and
`BasicTrainingCapability` calls that every epoch, so it re-samples the whole graph each time.
That makes it i.i.d. sampling without replacement *within* an epoch -- a strong control, but
not the exhaustive curriculum the name implies, and on a large graph it can never run out of
data. `--comprehensive-cumulative` is what creates a data-limited regime to measure in.
"""

from tests.helpers.factories import build_ring_graph
from traversals.ComprehensiveTraversal import ComprehensiveTraversal


def _graph(size):
    """A connected ring, via the shared factory rather than a hand-rolled node."""
    graph, _nodes, _edges = build_ring_graph(count=size)
    return graph


def _drain(traversal, batch_size=8, limit=200):
    """Collect until the traversal stops offering nodes."""
    collected = []
    for _ in range(limit):
        batch = traversal.traverse(batch_size)
        if not batch:
            break
        collected.extend(batch)
    return collected


def test_default_resamples_every_epoch():
    traversal = ComprehensiveTraversal(_graph(40), num_pointers=1, num_steps=10_000)
    first = _drain(traversal)
    traversal.reset_pointers()
    second = _drain(traversal)
    # Cleared visited -> the pool is full again, so a second epoch is just as productive.
    assert len(second) == len(first) > 0


def test_cumulative_exhausts_the_pool():
    traversal = ComprehensiveTraversal(_graph(40), num_pointers=1, num_steps=10_000,
                                       cumulative=True)
    first = _drain(traversal)
    traversal.reset_pointers()
    second = _drain(traversal)
    assert len(first) > 0
    # visited survived the reset, so there is nothing new left to hand out.
    assert len(second) == 0, "a cumulative traversal must run out after covering the graph"


def test_cumulative_covers_each_node_at_most_once():
    graph = _graph(40)
    traversal = ComprehensiveTraversal(graph, num_pointers=1, num_steps=10_000,
                                       cumulative=True)
    seen = [node.node_id for node in _drain(traversal)]
    assert len(seen) == len(set(seen)), "no node should be handed out twice"
    assert len(seen) <= 40


def test_default_flag_is_off_so_existing_runs_are_unchanged():
    assert ComprehensiveTraversal(_graph(8), num_pointers=1).cumulative is False
