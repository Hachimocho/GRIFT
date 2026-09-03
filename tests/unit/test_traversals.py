"""Traversal behavior and per-seed reproducibility.

These are the tests that catch the `key=lambda x: id(x)` sort in
ComprehensiveTraversal -- sorting by memory address, in the branch whose comment
claims it exists "for deterministic order during testing".
"""

import pytest

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from nodes.atrnode import AttributeNode
from nodes.Node import Node
from tests.helpers.factories import (
    DummyTrainer, build_traversal, get_traversal_classes, make_attr_nodes,
)
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal
from traversals.RandomTraversal import RandomTraversal


def build_line_graph(count=6, attribute_nodes=True):
    """A path graph 0-1-2-...-(n-1)."""
    if attribute_nodes:
        nodes = make_attr_nodes(count)
    else:
        nodes = [
            Node(node_id=str(index), split="train", data=None, edges=[], label=0)
            for index in range(count)
        ]
    for index in range(count - 1):
        edge = Edge(nodes[index], nodes[index + 1], x=None)
        nodes[index].add_edge(edge)
        nodes[index + 1].add_edge(edge)
    return HyperGraph(nodes)


def test_random_traversal_returns_graph_nodes():
    graph = build_line_graph(6)
    traversal = RandomTraversal(graph, num_pointers=2, num_steps=10)
    batch = traversal.traverse(batch_size=16)
    assert isinstance(batch, list)
    assert set(batch).issubset(set(graph.get_nodes()))


def test_comprehensive_traversal_visits_every_node():
    graph = build_line_graph(8)
    traversal = ComprehensiveTraversal(graph, num_pointers=1, num_steps=None)
    seen = []
    while True:
        batch = traversal.traverse(batch_size=3)
        if not batch:
            break
        seen.extend(batch)
    assert set(seen) == set(graph.get_nodes())


def test_ivalue_traversal_returns_attribute_nodes():
    graph = build_line_graph(7)
    traversal = IValueTraversal(graph, num_pointers=2, num_steps=5, trainer=DummyTrainer())
    batch = traversal.traverse(batch_size=8)
    assert isinstance(batch, list)
    assert all(isinstance(node, AttributeNode) for node in batch)


def test_ivalue_traversal_consults_the_trainer():
    graph = build_line_graph(6)
    trainer = DummyTrainer()
    IValueTraversal(graph, num_pointers=2, num_steps=5, trainer=trainer).traverse(batch_size=8)
    assert trainer.calls, "IValueTraversal must query the trainer for I-values"


def _graph_with_ids_decorrelated_from_allocation(count=8):
    """Build a graph where allocation order and node-id order disagree.

    Necessary to detect the `key=lambda x: id(x)` sort at all: if nodes are
    allocated in node-id order, CPython hands out ascending addresses and sorting
    by id(x) *coincidentally* matches sorting by node_id. Allocating in a
    scrambled order breaks that coincidence.
    """
    from tests.helpers.factories import make_attr_node

    permutation = [5, 2, 7, 0, 4, 1, 6, 3][:count]
    nodes_by_id = {}
    for index in permutation:  # allocation order != id order
        nodes_by_id[f"n{index}"] = make_attr_node(index)
    nodes = [nodes_by_id[f"n{index}"] for index in range(count)]
    for index in range(count - 1):
        edge = Edge(nodes[index], nodes[index + 1], x=None)
        nodes[index].add_edge(edge)
        nodes[index + 1].add_edge(edge)
    return HyperGraph(nodes)


def test_comprehensive_traversal_test_mode_order_is_by_node_id():
    """The test_mode branch's "deterministic order" must actually be deterministic.

    It sorted by `id(x)` -- a memory address -- so the order varied between
    processes and between allocations, in the one branch whose comment claims it
    exists for deterministic testing.
    """
    graph = _graph_with_ids_decorrelated_from_allocation(8)
    traversal = ComprehensiveTraversal(graph, num_pointers=1, num_steps=None)
    assert traversal.test_mode, "precondition: num_steps=None selects the test_mode branch"

    batch = traversal.traverse(batch_size=4)
    ids = [node.node_id for node in batch]
    assert ids == sorted(ids), (
        f"expected node-id order, got {ids} -- sorting by id(x) orders by memory address"
    )


#: The three Random*Warp / *NoReturn traversals override `traverse()` without a
#: `batch_size` parameter, unlike the base class and every other subclass. They
#: are unreachable from `create_traversal`, so this is latent rather than active
#: -- but `DQNCapability` calls `traversal.traverse(self.batch_size)`
#: positionally, so wiring any of them up would raise TypeError immediately.
TRAVERSALS_WITHOUT_BATCH_SIZE = {
    "RandomWarpTraversal",
    "RandomNoReturnTraversal",
    "RandomNoReturnWarpTraversal",
}


@pytest.mark.parametrize("traversal_name", sorted(get_traversal_classes()))
def test_traverse_signature_accepts_batch_size(traversal_name):
    """Pins which traversals honor the base class's `traverse(batch_size=32)`.

    `Traversal.traverse` declares `batch_size=32` and `DQNCapability.py:304`
    passes it positionally, so a traversal that omits it is not substitutable for
    its base class.
    """
    import inspect

    traversal_class = get_traversal_classes()[traversal_name]
    parameters = inspect.signature(traversal_class.traverse).parameters
    accepts = "batch_size" in parameters

    if traversal_name in TRAVERSALS_WITHOUT_BATCH_SIZE:
        assert not accepts, (
            f"{traversal_name} now accepts batch_size -- remove it from "
            "TRAVERSALS_WITHOUT_BATCH_SIZE, the Liskov violation is fixed"
        )
    else:
        assert accepts, f"{traversal_name}.traverse() must accept batch_size"


def test_every_cli_traversal_choice_is_constructible():
    """Every `--traversal-type` value argparse accepts must actually build.

    `i-value-subcluster` and `i-value-cluster-hop-subcluster` were listed in
    `choices` but had no branch in `create_traversal`, so selecting either failed
    instantly with "Unsupported traversal type".
    """
    import test_hierarchical
    from test_helpers.args_utils import parse_args

    # Pull the advertised choices straight from the parser so the two cannot drift.
    import argparse
    import unittest.mock

    captured = {}
    real_add_argument = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kwargs):
        if "--traversal-type" in args:
            captured["choices"] = kwargs.get("choices", [])
        return real_add_argument(self, *args, **kwargs)

    with unittest.mock.patch.object(argparse.ArgumentParser, "add_argument", spy):
        with unittest.mock.patch("sys.argv", ["prog"]):
            parse_args()

    choices = captured.get("choices")
    assert choices, "could not discover --traversal-type choices"

    graph = build_line_graph(4)
    failures = {}
    for choice in choices:
        try:
            test_hierarchical.create_traversal(choice, graph, trainer=DummyTrainer())
        except Exception as exc:  # noqa: BLE001 - we want the type and message
            failures[choice] = f"{type(exc).__name__}: {exc}"
    assert not failures, f"advertised traversal types that fail to construct: {failures}"


class _SelectionConfiguredTrainer(DummyTrainer):
    """A trainer exposing every selection-mechanism knob `create_traversal` must read.

    Values are all non-default, so a factory that silently drops one of them produces a
    traversal that is distinguishable from the correctly-wired one -- which is exactly the
    bug this guards against: `test_hierarchical.create_traversal` (used for every
    non-switching run) used to build `IValueTraversal` with none of these, so
    `--ivalue-candidate-pool`, `--ivalue-selection`, `--ivalue-band`, and
    `--ivalue-group-targeting` were silently inert for every single-traversal run, and
    three sweep arms meant to test three different selection mechanisms ran the identical
    one without either the sweep or the training log ever mentioning it.
    """

    def __init__(self, graph=None, fairness_selection=False):
        super().__init__()
        from trainers.capabilities.group_fairness import GroupPerformanceTracker
        from trainers.capabilities.group_targeting import GroupTargeting

        self.ivalue_candidate_pool = 64
        self.ivalue_selection_mode = "band"
        self.ivalue_selection_band = (0.2, 0.6)
        self.group_targeting = GroupTargeting(top_groups=2, enabled=True)
        self.ivalue_fairness_selection = fairness_selection
        self.fairness_tracker = GroupPerformanceTracker(enabled=True)
        # `_create_traversal` does `kwargs.get('graph', self.graphmanager.get_graph())`,
        # which evaluates the default eagerly even when `graph` is supplied -- so this
        # attribute has to exist regardless of whether the test ends up using it.
        self.graphmanager = _StaticGraphManager(graph)


class _StaticGraphManager:
    def __init__(self, graph):
        self._graph = graph

    def get_graph(self):
        return self._graph


def test_create_traversal_forwards_every_selection_knob():
    """`create_traversal` must build the same `IValueTraversal` config `_create_traversal`
    does, given the same trainer -- see `_SelectionConfiguredTrainer`'s docstring.
    """
    import test_hierarchical
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    graph = build_line_graph(6)
    trainer = _SelectionConfiguredTrainer(graph=graph)

    single_mode = test_hierarchical.create_traversal(
        "i-value", graph, trainer=trainer, bias_hop_period=7,
    )
    switching_mode = AdaptiveTrainer._create_traversal(
        trainer, "i-value", graph=graph, bias_hop_period=7,
    )

    for traversal, label in ((single_mode, "create_traversal"),
                             (switching_mode, "_create_traversal")):
        assert traversal.candidate_pool == 64, label
        assert traversal.selection_mode == "band", label
        assert traversal.selection_band == (0.2, 0.6), label
        assert traversal.group_targeting is trainer.group_targeting, label


def test_both_factories_swap_in_the_fairness_tracker_when_asked():
    """`--ivalue-fairness-selection` must swap which object fills the traversal's
    `group_targeting` slot in *both* factories, not just one -- the exact failure shape
    `test_create_traversal_forwards_every_selection_knob` already guards for the other
    four knobs.
    """
    import test_hierarchical
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    graph = build_line_graph(6)
    trainer = _SelectionConfiguredTrainer(graph=graph, fairness_selection=True)

    single_mode = test_hierarchical.create_traversal(
        "i-value", graph, trainer=trainer, bias_hop_period=7,
    )
    switching_mode = AdaptiveTrainer._create_traversal(
        trainer, "i-value", graph=graph, bias_hop_period=7,
    )

    for traversal, label in ((single_mode, "create_traversal"),
                             (switching_mode, "_create_traversal")):
        assert traversal.group_targeting is trainer.fairness_tracker, label
        assert traversal.group_targeting is not trainer.group_targeting, label


def test_create_traversal_defaults_match_ivaluetraversals_own_defaults():
    """A trainer with none of the knobs set must reproduce `IValueTraversal`'s own
    defaults, not `None` or some other falsy placeholder that changes behavior.
    """
    import test_hierarchical

    graph = build_line_graph(6)
    traversal = test_hierarchical.create_traversal(
        "i-value", graph, trainer=DummyTrainer(),
    )
    assert traversal.candidate_pool == 0
    assert traversal.selection_mode == "max"
    assert traversal.selection_band == (0.4, 0.7)
    assert traversal.group_targeting is None


@pytest.mark.parametrize("traversal_name", sorted(get_traversal_classes()))
def test_traversal_is_reproducible_under_a_fixed_seed(traversal_name):
    """Same seed -> same visit sequence, for every traversal class.

    Traversals currently draw from the process-global `random` module, so this
    only holds when nothing else consumes randomness in between. That is exactly
    the fragility the per-component RNG streams in `test_helpers.determinism`
    remove; this test pins the observable guarantee either way.
    """
    from test_helpers.determinism import configure_determinism

    traversal_class = get_traversal_classes()[traversal_name]

    def visit_sequence():
        configure_determinism(seed=1234, mode="strict", allow_multi_gpu=True)
        graph = build_line_graph(10)
        traversal = build_traversal(traversal_class, graph, num_pointers=2, num_steps=12)

        sequence = []
        for _ in range(4):
            if traversal_name in TRAVERSALS_WITHOUT_BATCH_SIZE:
                batch = traversal.traverse()
            else:
                batch = traversal.traverse(batch_size=4)
            if not batch:
                break
            sequence.append([getattr(node, "node_id", None) for node in batch])
        return sequence

    assert visit_sequence() == visit_sequence(), (
        f"{traversal_name} is not reproducible at a fixed seed"
    )


def test_traversal_pointers():
    graph = build_line_graph(6)
    traversal = RandomTraversal(graph, num_pointers=3, num_steps=10)
    traversal.reset_pointers()
    assert len(traversal.get_pointers()) == 3


@pytest.mark.parametrize("traversal_name", sorted(get_traversal_classes()))
def test_traversal_len_is_implemented_or_explicitly_absent(traversal_name):
    """Records which traversals implement `__len__`.

    `Traversal.__len__` raises NotImplementedError with "Subclass must implement
    __len__()", but only ComprehensiveTraversal, IValueTraversal, and
    IValueTraversalClusterHop actually do. So `len()` on any Random* traversal
    raises. Pinned as current behavior; the base class's contract and the
    subclasses disagree, which is worth resolving separately.
    """
    implements = {
        "ComprehensiveTraversal",
        "IValueTraversal",
        "IValueTraversalSubcluster",
        "IValueTraversalClusterHop",
        "IValueTraversalClusterHopSubcluster",
    }
    traversal_class = get_traversal_classes()[traversal_name]
    node_count = 6
    graph = build_line_graph(node_count)
    traversal = build_traversal(traversal_class, graph, num_pointers=2, num_steps=10)

    if traversal_name in implements:
        # ComprehensiveTraversal caps at the node count; the I-value ones return
        # num_steps directly.
        assert len(traversal) in (10, min(10, node_count))
    else:
        with pytest.raises(NotImplementedError):
            len(traversal)
