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


@pytest.mark.xfail(
    strict=True,
    reason="C2: ComprehensiveTraversal's test_mode branch sorts by id(x), i.e. by "
           "memory address, so its 'deterministic' order is not deterministic across "
           "processes. Fixed by sorting on node_id in group C.",
)
def test_comprehensive_traversal_test_mode_order_is_by_node_id():
    """The test_mode branch claims a deterministic order; it should be by node id."""
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
