"""The graph-updater axis: selection, rewiring, and the pairings that make no sense.

Before this, neither updater was reachable. `NoGraphManager` was hardcoded as the training
manager, so `PerformanceGraphManager` was imported and never constructed; and worse,
nothing called `graphmanager.update_graph()` on the live path at all -- only the legacy
`Trainer.run` did -- and nothing anywhere called `track_performance`, so every node sat at
the neutral 0.5 default and no node could ever be classified weak or strong.
`GraphReductionManager` was fully wired into the epoch loop but gated on a config key that
`test_configs` never set.

Two properties matter most:

**A rewiring manager that cannot rewire must say so.** Without an I-value predictor every
node reads the neutral default, so "weak" and "strong" are meaningless and the graph is
static -- indistinguishable in the results from `--graph-manager none`. That has to be
refused at plan time and reported at run time, not silently produce a plausible row.

**`--graph-manager none` must be exactly the old behavior.** It is the default, so any
drift here changes every existing run's numbers.
"""

import numpy as np
import pytest

from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager
from test_helpers.args_utils import parse_args
from web_ui.gpu_queue_manager import GPUQueueManager


@pytest.fixture
def many_node_graph():
    """A graph big enough for quantiles to be defined (>= MIN_TRACKED_FOR_QUANTILES).

    The ring fixture has six nodes, which is below the floor the manager refuses under -- it
    would report "too few measured" rather than exercising an update.
    """
    from graphs.HyperGraph import HyperGraph
    from tests.helpers.factories import connect_ring, make_attr_nodes

    nodes = make_attr_nodes(count=40)
    connect_ring(nodes)
    return HyperGraph(nodes), nodes


@pytest.fixture
def build_args():
    """`_build_command_args` without constructing a live queue manager."""
    manager = GPUQueueManager.__new__(GPUQueueManager)
    return lambda config, **kwargs: manager._build_command_args(config, **kwargs)


# -- CLI surface ---------------------------------------------------------------- #

def test_defaults_reproduce_the_previous_behavior():
    """Every new flag must default to the state that existed before it did."""
    args = parse_args([])
    assert args.graph_manager == "none"
    assert args.reduction_enabled is False
    assert args.reduction_strategy == "none"
    assert args.restoration_strategy == "none"
    assert args.reduction_percentage == 0.0


def test_graph_manager_choices_are_constrained():
    args = parse_args(["--graph-manager", "performance"])
    assert args.graph_manager == "performance"
    with pytest.raises(SystemExit):
        parse_args(["--graph-manager", "nonsense"])


def test_reduction_strategies_match_the_manager_implementation():
    """A strategy argparse accepts but the manager cannot dispatch would fail mid-run."""
    from managers.GraphReductionManager import GraphReductionManager

    for strategy in ("max_ival", "min_ival", "mix_max_ival", "random", "none"):
        args = parse_args(["--reduction-strategy", strategy])
        assert args.reduction_strategy == strategy
        manager = GraphReductionManager(reduction_strategy=strategy)
        assert manager.reduction_strategy == strategy

    with pytest.raises(SystemExit):
        parse_args(["--reduction-strategy", "lowest_ivalue"])


@pytest.mark.parametrize("flag,config_key,value", [
    ("--graph-manager", "graph_manager", "performance"),
    ("--weak-quantile", "weak_quantile", 0.85),
    ("--strong-quantile", "strong_quantile", 0.15),
    ("--removal-fraction", "removal_fraction", 0.03),
    ("--graph-updates-per-epoch", "graph_updates_per_epoch", 6),
    ("--graph-remove-target", "graph_remove_target", "weak"),
    ("--graph-manager-sample-nodes", "graph_manager_sample_nodes", 100),
    ("--reduction-strategy", "reduction_strategy", "random"),
    ("--reduction-percentage", "reduction_percentage", 15.0),
    ("--reduction-top-percentage", "reduction_top_percentage", 5.0),
    ("--reduction-bottom-percentage", "reduction_bottom_percentage", 5.0),
    ("--reduction-interval", "reduction_interval", "every_n_steps"),
    ("--reduction-interval-steps", "reduction_interval_steps", 25),
    ("--restoration-strategy", "restoration_strategy", "random_pool"),
    ("--restoration-percentage", "restoration_percentage", 30.0),
    ("--restoration-trigger-threshold", "restoration_trigger_threshold", 0.01),
])
def test_every_updater_flag_survives_the_queues_allowlist(
    build_args, flag, config_key, value
):
    """An unrouted key is dropped silently, producing a run that ignores the setting."""
    command = build_args({config_key: value})
    assert flag in command, f"{config_key} did not reach the CLI"
    assert str(value) in command


def test_reduction_enabled_is_a_bare_flag(build_args):
    assert "--reduction-enabled" in build_args({"reduction_enabled": True})
    # False emits nothing, so the CLI default (disabled) applies.
    assert "--reduction-enabled" not in build_args({"reduction_enabled": False})


# -- NoGraphManager ------------------------------------------------------------- #

def test_no_graph_manager_accepts_the_steps_argument(ring_graph):
    """Callers advance any manager uniformly; a static graph has nothing to advance."""
    graph, _nodes, _edges = ring_graph
    manager = NoGraphManager(graph)
    assert manager.update_graph() is None
    assert manager.update_graph(steps_taken=500) is None


def test_no_graph_manager_never_mutates_the_graph(ring_graph):
    graph, nodes, _edges = ring_graph
    manager = NoGraphManager(graph)
    before = [len(node.edges) for node in nodes]
    for _ in range(10):
        manager.update_graph(steps_taken=1000)
    assert [len(node.edges) for node in nodes] == before


# -- PerformanceGraphManager ---------------------------------------------------- #

class FakePredictor:
    """Stands in for the DQN. `update_graph` only checks that one is present."""


def measured(manager, nodes, values=None):
    """Give every node a distinct I-value, so the quantiles have a real distribution."""
    for index, node in enumerate(nodes):
        value = values[index] if values else index / max(1, len(nodes) - 1)
        manager.track_performance(node, value)


def test_untracked_nodes_read_the_neutral_default(ring_graph):
    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    assert manager.tracked_count() == 0
    assert manager.get_node_avg_performance(nodes[0]) == 0.5
    # And quantiles refuse until enough nodes are measured, rather than inventing cuts.
    assert manager.quantiles() is None


def test_quantiles_bracket_the_observed_distribution(many_node_graph):
    """The old absolute 0.8/0.2 pair did not bracket the DQN output at all, so no node was
    ever classified strong and the updater changed nothing."""
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, weak_quantile=0.9, strong_quantile=0.1)
    # Values nowhere near the retired absolute thresholds.
    measured(manager, nodes, [5.0 + index * 0.01 for index in range(len(nodes))])

    cuts = manager.quantiles()
    assert cuts is not None
    strong_cut, weak_cut = cuts
    assert strong_cut < weak_cut
    assert manager.identify_strong_nodes(), "a quantile cut always selects somebody"
    assert manager.identify_weak_nodes()


def test_no_predictor_means_no_update_and_a_message(ring_graph, capsys):
    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    measured(manager, nodes)

    before = len(list(graph.get_nodes()))
    assert manager.update_graph() is None
    assert len(list(graph.get_nodes())) == before
    assert "no I-value predictor" in capsys.readouterr().out


def test_too_few_measurements_means_no_update(ring_graph, capsys):
    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes[:3])

    assert manager.update_graph() is None
    assert "measured" in capsys.readouterr().out


def test_an_update_withdraws_nodes(many_node_graph):
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)

    before = len(list(graph.get_nodes()))
    stats = manager.update_graph()
    assert stats is not None
    assert stats["nodes_removed"] > 0
    assert len(list(graph.get_nodes())) == before - stats["nodes_removed"]


def test_strong_target_withdraws_the_lowest_i_values(many_node_graph):
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05,
                                      remove_target='strong')
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)
    manager.update_graph()

    withdrawn = {node.node_id for node in manager.removed_nodes}
    lowest = {node.node_id for node in nodes[:len(withdrawn)]}
    assert withdrawn == lowest


def test_weak_target_withdraws_the_highest_i_values(many_node_graph):
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05, remove_target='weak')
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)
    manager.update_graph()

    withdrawn = {node.node_id for node in manager.removed_nodes}
    highest = {node.node_id for node in nodes[-len(withdrawn):]}
    assert withdrawn == highest


def test_removal_never_runs_away(many_node_graph):
    """A graph pruned to nothing measures the traversal on a starved graph, which is a far
    bigger effect than the pruning under study."""
    from managers.PerformanceGraphManager import MIN_GRAPH_FRACTION

    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=1.0)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)

    for _ in range(50):
        manager.update_graph()
    floor = int(manager.initial_node_count * MIN_GRAPH_FRACTION)
    assert len(list(graph.get_nodes())) >= floor


def test_withdrawal_is_reversible(many_node_graph):
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)

    before = len(list(graph.get_nodes()))
    stats = manager.update_graph()
    restored = manager.restore_nodes()
    assert restored == stats["nodes_removed"]
    assert len(list(graph.get_nodes())) == before


def test_restoration_works_after_removal(many_node_graph):
    """`HyperGraph.remove_node` left `_node_data_map` stale, so `add_node` then refused the
    node as a duplicate and restoration silently did nothing."""
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)
    manager.update_graph()
    manager.restore_nodes()

    ids = [node.node_id for node in graph.get_nodes()]
    assert len(ids) == len(set(ids)), "restoration must not duplicate a node"
    assert len(ids) == len(nodes)


def test_selection_is_deterministic(many_node_graph):
    """Worst-first with node_id as the tie-break, so it needs no RNG and cannot depend on
    PYTHONHASHSEED."""
    from tests.helpers.factories import make_attr_nodes, connect_ring
    from graphs.HyperGraph import HyperGraph

    choices = []
    for _ in range(2):
        made = make_attr_nodes(count=40)
        connect_ring(made)
        manager = PerformanceGraphManager(HyperGraph(made), removal_fraction=0.05)
        manager.set_i_value_predictor(FakePredictor())
        measured(manager, made)
        manager.update_graph()
        choices.append([node.node_id for node in manager.removed_nodes])
    assert choices[0] == choices[1]


def test_performance_storage_is_constant_per_node(ring_graph):
    """An EWMA, not a list of the last 100 observations. That list would have needed
    several gigabytes to track a million nodes; two floats each is ~200 MB."""
    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    for _ in range(500):
        manager.track_performance(nodes[0], 0.42)

    entry = manager.node_performance[nodes[0].node_id]
    assert len(entry) == 2, "one mean and one count, whatever the observation count"
    assert entry[1] == 500
    assert abs(entry[0] - 0.42) < 1e-9


def test_tracking_ignores_unusable_values(ring_graph):
    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    for bad in (float("nan"), float("inf"), None, "x"):
        manager.track_performance(nodes[0], bad)
    assert manager.tracked_count() == 0


def test_updates_per_epoch_converts_to_a_step_interval(ring_graph):
    graph, _nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph, updates_per_epoch=4)
    assert manager.steps_between_updates(500) == 125
    # Never zero, however few steps an epoch has.
    assert manager.steps_between_updates(1) >= 1


def test_retired_edge_parameters_are_accepted_and_announced(ring_graph, capsys):
    """Existing configs still construct, rather than raising on an unexpected kwarg."""
    graph, _nodes, _edges = ring_graph
    PerformanceGraphManager(
        graph, rewire_threshold=0.8, edge_removal_threshold=0.2, update_interval=200,
    )
    output = capsys.readouterr().out
    assert "ignored" in output
    assert "quantiles" in output


def test_update_history_records_what_happened(many_node_graph):
    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph, removal_fraction=0.05)
    manager.set_i_value_predictor(FakePredictor())
    measured(manager, nodes)
    manager.update_graph()

    assert len(manager.update_history) == 1
    entry = manager.update_history[0]
    for key in ("update", "tracked_nodes", "graph_nodes", "strong_cut", "weak_cut",
                "nodes_removed", "removed_total", "remove_target"):
        assert key in entry


# -- the runner's wiring helpers ------------------------------------------------ #

def test_attach_predictor_is_a_no_op_without_a_dqn_capability(ring_graph, capsys):
    """Without a DQN, get_i_value returns a random draw -- rewiring on that is noise."""
    import test_hierarchical

    graph, _nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)

    class TrainerWithoutDQN:
        capabilities = type("Capabilities", (), {"dqn_capability": None})()

    assert test_hierarchical.attach_i_value_predictor(manager, TrainerWithoutDQN()) is False
    assert manager.i_value_predictor is None
    assert "no DQN capability" in capsys.readouterr().out


def test_attach_predictor_takes_the_first_dqn(ring_graph):
    import test_hierarchical

    graph, _nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)
    predictor = FakePredictor()

    class TrainerWithDQN:
        capabilities = type(
            "Capabilities", (), {"dqn_capability": type(
                "DQNCapability", (), {"dqns": [predictor, FakePredictor()]}
            )()},
        )()

    assert test_hierarchical.attach_i_value_predictor(manager, TrainerWithDQN()) is True
    assert manager.i_value_predictor is predictor


def test_attach_predictor_ignores_a_static_manager(ring_graph):
    import test_hierarchical

    graph, _nodes, _edges = ring_graph
    assert test_hierarchical.attach_i_value_predictor(
        NoGraphManager(graph), object()
    ) is False


def test_track_graph_performance_samples_and_records(ring_graph):
    import test_hierarchical

    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)

    class Trainer:
        def get_i_value(self, node, model_idx=0):
            return 0.9

    tracked = test_hierarchical.track_graph_performance(manager, Trainer(), sample_size=2)
    assert tracked == 2
    assert len(manager.node_performance) == 2


def test_track_graph_performance_zero_means_every_node(ring_graph):
    import test_hierarchical

    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)

    class Trainer:
        def get_i_value(self, node, model_idx=0):
            return 0.9

    assert test_hierarchical.track_graph_performance(
        manager, Trainer(), sample_size=0
    ) == len(nodes)


def test_track_graph_performance_survives_a_failing_i_value(ring_graph, capsys):
    """One unreadable node must not abort the epoch."""
    import test_hierarchical

    graph, nodes, _edges = ring_graph
    manager = PerformanceGraphManager(graph)

    class Trainer:
        def __init__(self):
            self.calls = 0

        def get_i_value(self, node, model_idx=0):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("no features")
            return 0.9

    tracked = test_hierarchical.track_graph_performance(manager, Trainer(), sample_size=0)
    assert tracked == len(nodes) - 1
    assert "could not track performance" in capsys.readouterr().out


def test_track_graph_performance_ignores_a_static_manager(ring_graph):
    import test_hierarchical

    graph, _nodes, _edges = ring_graph
    assert test_hierarchical.track_graph_performance(
        NoGraphManager(graph), object(), sample_size=10
    ) == 0


# -- the tracking hook ---------------------------------------------------------- #
#
# `PerformanceGraphManager` documents training's own I-value computations as its input, and
# for one whole sweep nothing called `track_performance`: it logged "0 node(s) measured",
# never reached the quantile minimum, pruned nothing, and three cells came back with
# byte-identical record tables. The hook lives on `AdaptiveTrainer.get_i_value` because that
# is the single funnel every predicted I-value passes through.

def test_the_trainer_records_i_values_into_the_graph_manager(many_node_graph):
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    graph, nodes = many_node_graph
    manager = PerformanceGraphManager(graph)

    trainer = AdaptiveTrainer.__new__(AdaptiveTrainer)
    trainer.graphmanager = manager
    trainer.capabilities = type("Caps", (), {
        "get_i_value": staticmethod(lambda node, model_idx=0: 0.25)
    })()

    for node in nodes:
        assert trainer.get_i_value(node) == 0.25
    assert manager.tracked_count() == len(nodes), (
        "every I-value the pipeline computes must reach the manager"
    )
    assert manager.quantiles() is not None, "and that must be enough for quantiles"


def test_the_hook_is_inert_for_a_manager_that_does_not_track(ring_graph):
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    graph, nodes, _edges = ring_graph
    trainer = AdaptiveTrainer.__new__(AdaptiveTrainer)
    trainer.graphmanager = NoGraphManager(graph)
    trainer.capabilities = type("Caps", (), {
        "get_i_value": staticmethod(lambda node, model_idx=0: 0.5)
    })()
    assert trainer.get_i_value(nodes[0]) == 0.5


def test_a_failing_tracker_does_not_break_training(many_node_graph, capsys):
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    graph, nodes = many_node_graph

    class Exploding:
        def track_performance(self, node, value):
            raise RuntimeError("boom")

    trainer = AdaptiveTrainer.__new__(AdaptiveTrainer)
    trainer.graphmanager = Exploding()
    trainer.capabilities = type("Caps", (), {
        "get_i_value": staticmethod(lambda node, model_idx=0: 0.75)
    })()
    assert trainer.get_i_value(nodes[0]) == 0.75
    assert "could not record I-value" in capsys.readouterr().out
