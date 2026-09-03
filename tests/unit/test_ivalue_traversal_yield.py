"""The I-value traversal must always yield nodes, in either walk.

`BasicTrainingCapability` collects with `while True: batch = traversal.traverse(); if not
batch: break`, so the *first* empty batch ends the epoch. A traversal that returns `[]` on
step one trains the model on nothing -- and nothing downstream notices: the capability
returns zeroed metrics, validation still scores the untrained network at roughly the class
prior, and the configuration is written out as complete. `i-value-cluster-hop-subcluster` did
exactly that for three epochs in two consecutive sweeps.

That variant is now removed, along with its sibling, and the two survivors are one class with
the walk chosen from the graph. These tests pin the properties whose absence caused the
original failure: a non-empty first batch, a terminal `return`, and an outlier filter that
does not exclude every node when the variance is zero.
"""

import pytest

from tests.helpers.factories import DummyTrainer, build_two_cluster_graph
from traversals.IValueTraversal import IValueTraversal

#: Both walks, by the flag that selects them.
WALKS = (("connected", False), ("cluster_hop", True))


@pytest.fixture
def graph_with_nodes():
    graph, nodes, _edges = build_two_cluster_graph(per_cluster=8)
    return graph, nodes


@pytest.mark.parametrize("name,cluster_hop", WALKS, ids=[n for n, _ in WALKS])
def test_the_first_batch_is_never_empty(graph_with_nodes, name, cluster_hop):
    """The condition that silently zeroed an entire epoch."""
    graph, _nodes = graph_with_nodes
    traversal = IValueTraversal(graph, 1, 20, trainer=DummyTrainer(), cluster_hop=cluster_hop)
    batch = traversal.traverse()
    assert batch is not None, f"{name}: traverse() returned None"
    assert len(batch) > 0, (
        f"{name}: empty first batch; BasicTrainingCapability stops collecting there, so "
        f"the epoch would train on zero nodes"
    )


@pytest.mark.parametrize("name,cluster_hop", WALKS, ids=[n for n, _ in WALKS])
def test_no_empty_batch_before_the_step_budget_is_spent(graph_with_nodes, name, cluster_hop):
    graph, _nodes = graph_with_nodes
    traversal = IValueTraversal(graph, 1, 20, trainer=DummyTrainer(), cluster_hop=cluster_hop)
    total = 0
    while traversal.t < traversal.num_steps:
        batch = traversal.traverse()
        assert batch, f"{name}: empty batch at step {traversal.t}/{traversal.num_steps}"
        total += len(batch)
    assert total > 0


@pytest.mark.parametrize("name,cluster_hop", WALKS, ids=[n for n, _ in WALKS])
def test_traverse_returns_the_batch_it_records(graph_with_nodes, name, cluster_hop):
    graph, _nodes = graph_with_nodes
    traversal = IValueTraversal(graph, 1, 20, trainer=DummyTrainer(), cluster_hop=cluster_hop)
    batch = traversal.traverse()
    assert list(batch) == list(traversal.current_batch_nodes)


@pytest.mark.parametrize("name,cluster_hop", WALKS, ids=[n for n, _ in WALKS])
def test_both_walks_yield_comparable_amounts(graph_with_nodes, name, cluster_hop):
    """The removed subcluster variants yielded 1 node per step against ~17 for these two,
    so any comparison across that boundary was measuring training-set size."""
    graph, _nodes = graph_with_nodes
    traversal = IValueTraversal(graph, 1, 20, trainer=DummyTrainer(), cluster_hop=cluster_hop)
    total = sum(len(traversal.traverse()) for _ in range(20))
    assert total / 20 > 2, f"{name} yielded only {total / 20:.1f} nodes per step"


# -- walk selection ------------------------------------------------------------- #

@pytest.mark.parametrize("graph_type,expected", [
    ("clustered", True),
    ("clustered_subclustered", True),
    ("nonclustered", False),
    ("nonclustered_subclustered", False),
    (None, False),
])
def test_the_walk_is_detected_from_the_graph(graph_with_nodes, graph_type, expected):
    """Hopping exists because a clustered graph's groups are disjoint, so it is a property
    of the construction rather than an independently selectable strategy."""
    graph, _nodes = graph_with_nodes
    if graph_type is not None:
        graph.graph_type = graph_type
    traversal = IValueTraversal(graph, 1, 10, trainer=DummyTrainer())
    assert traversal.cluster_hop is expected


def test_an_explicit_flag_overrides_detection(graph_with_nodes):
    graph, _nodes = graph_with_nodes
    graph.graph_type = "nonclustered"
    assert IValueTraversal(graph, 1, 10, trainer=DummyTrainer(), cluster_hop=True).cluster_hop
    graph.graph_type = "clustered"
    assert not IValueTraversal(
        graph, 1, 10, trainer=DummyTrainer(), cluster_hop=False
    ).cluster_hop


# -- lazy i-values -------------------------------------------------------------- #

def test_neither_walk_pre_warms_i_values(graph_with_nodes):
    """The connected walk used to call `trainer.get_i_value` for every node in the graph,
    for every pointer, inside `reset_pointers` -- one DQN forward pass per node, repeated on
    every refresh period. That is not runnable on a large graph."""
    graph, _nodes = graph_with_nodes

    class CountingTrainer(DummyTrainer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_i_value(self, node, model_idx=0):
            self.calls += 1
            return super().get_i_value(node, model_idx)

    for cluster_hop in (False, True):
        trainer = CountingTrainer()
        IValueTraversal(graph, 2, 20, trainer=trainer, cluster_hop=cluster_hop)
        assert trainer.calls == 0, (
            f"cluster_hop={cluster_hop}: construction made {trainer.calls} I-value calls"
        )


@pytest.mark.parametrize("name,cluster_hop", WALKS, ids=[n for n, _ in WALKS])
def test_zero_variance_i_values_do_not_exclude_every_node(graph_with_nodes, name, cluster_hop):
    """The `<` vs `<=` bug that the removed subcluster variants died on, kept as a
    regression guard: with every i-value equal, a filter of the form
    `v < mean + k*std` is false for all of them including the mean itself."""
    graph, nodes = graph_with_nodes
    traversal = IValueTraversal(graph, 1, 20, trainer=DummyTrainer(), cluster_hop=cluster_hop)
    for pointer in traversal.pointers:
        pointer["i_values"] = {}

    values = [0.5] * len(nodes)
    mean = sum(values) / len(values)
    kept = [n for n, v in zip(nodes, values) if v <= mean + 2.0 * 0.0]
    assert len(kept) == len(nodes), "a point at the mean is not an outlier"
    assert traversal.traverse(), "and the traversal must still yield nodes"


def test_hop_history_is_available_in_both_walks(graph_with_nodes):
    """The visualization hooks read this unconditionally."""
    graph, _nodes = graph_with_nodes
    for cluster_hop in (False, True):
        traversal = IValueTraversal(
            graph, 1, 10, trainer=DummyTrainer(), cluster_hop=cluster_hop
        )
        assert isinstance(traversal.get_hop_i_value_history(), list)
