"""Short real traversal runs over a synthetic graph.

Asserts that each traversal terminates, returns real graph nodes, and produces the
same visit sequence at a fixed seed -- including when the graph came from the edge
cache rather than a fresh build.
"""

import pytest

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from tests.helpers.factories import (
    DummyTrainer, build_traversal, get_traversal_classes, make_attr_nodes,
)

pytestmark = pytest.mark.slow

#: These three diverge from the traversal interface in three separate ways: their
#: `traverse()` takes no `batch_size`, returns None instead of a node list, and moves
#: pointers in place for the caller to read via `get_pointers()`. That is the older
#: iterator-style contract still used by `Traversal.__next__`. They are unreachable
#: from `create_traversal`, so this is latent rather than active -- but
#: `DQNCapability` calls `traversal.traverse(self.batch_size)` and
#: `BasicTrainingCapability` expects a returned batch, so wiring any of them up would
#: fail immediately.
LEGACY_POINTER_TRAVERSALS = {
    "RandomWarpTraversal", "RandomNoReturnTraversal", "RandomNoReturnWarpTraversal",
}


def build_graph(node_count=40, degree=4):
    nodes = make_attr_nodes(node_count)
    for index, node in enumerate(nodes):
        for offset in range(1, degree + 1):
            peer = nodes[(index + offset) % node_count]
            if peer is node:
                continue
            if any(peer in edge.get_nodes() for edge in node.edges):
                continue
            edge = Edge(node, peer, x=None)
            node.add_edge(edge)
            peer.add_edge(edge)
    graph = HyperGraph(nodes)
    graph.canonicalize_edge_order()
    return graph


def run_traversal(traversal, name, rounds=8, batch_size=8):
    """Drive a traversal and return its per-round visited node ids.

    Handles both interface styles: the batch-returning one, and the legacy style
    that moves pointers in place and returns None.
    """
    sequence = []
    for _ in range(rounds):
        if name in LEGACY_POINTER_TRAVERSALS:
            traversal.traverse()
            batch = [pointer["current_node"] for pointer in traversal.get_pointers()]
        else:
            batch = traversal.traverse(batch_size=batch_size)
        if not batch:
            break
        sequence.append([getattr(node, "node_id", None) for node in batch])
    return sequence


@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_traversal_terminates_and_returns_graph_nodes(name):
    graph = build_graph()
    valid = {node.node_id for node in graph.get_nodes()}
    traversal = build_traversal(
        get_traversal_classes()[name], graph, num_pointers=2, num_steps=60
    )

    sequence = run_traversal(traversal, name)
    assert sequence, f"{name} produced no batches"
    for batch in sequence:
        assert batch, f"{name} produced an empty batch mid-run"
        for node_id in batch:
            assert node_id in valid, f"{name} returned a node not in the graph"


@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_traversal_is_reproducible(name):
    from test_helpers.determinism import configure_determinism

    def once():
        configure_determinism(seed=31337, mode="strict", allow_multi_gpu=True)
        graph = build_graph()
        traversal = build_traversal(
            get_traversal_classes()[name], graph, num_pointers=2, num_steps=60
        )
        return run_traversal(traversal, name)

    assert once() == once(), f"{name} is not reproducible at a fixed seed"


@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_traversal_is_unaffected_by_unrelated_rng_consumption(name):
    """A traversal's own stream must be independent of the global RNG.

    Before per-component streams, anything that consumed global randomness first --
    a debug print, a differently-sized graph driving more I-value fallback draws --
    changed which nodes were visited.
    """
    import random

    from test_helpers.determinism import configure_determinism

    def once(drain):
        configure_determinism(seed=4242, mode="strict", allow_multi_gpu=True)
        for _ in range(drain):
            random.random()
        graph = build_graph()
        traversal = build_traversal(
            get_traversal_classes()[name], graph, num_pointers=2, num_steps=60
        )
        return run_traversal(traversal, name)

    assert once(drain=0) == once(drain=997), (
        f"{name} changed after unrelated global RNG consumption"
    )


def test_comprehensive_traversal_visits_every_node():
    graph = build_graph(node_count=24)
    traversal = get_traversal_classes()["ComprehensiveTraversal"](
        graph, num_pointers=1, num_steps=None
    )
    seen = set()
    while True:
        batch = traversal.traverse(batch_size=5)
        if not batch:
            break
        seen.update(node.node_id for node in batch)
    assert seen == {node.node_id for node in graph.get_nodes()}


@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_cached_graph_induces_the_same_traversal(tmp_path, name):
    """A warm cache must not change exploration.

    export/load previously produced a different adjacency *order* than a fresh build,
    and traversals break neighbor ties by list position -- so resuming from cache
    silently explored a different part of the graph.
    """
    from test_helpers.determinism import configure_determinism

    original = build_graph()
    path = tmp_path / "edges.csv.gz"
    original.export_edges_csv(str(path))

    from_cache = HyperGraph(make_attr_nodes(len(original.get_nodes())))
    from_cache.load_edges_from_csv(str(path))

    def once(graph):
        configure_determinism(seed=55, mode="strict", allow_multi_gpu=True)
        traversal = build_traversal(
            get_traversal_classes()[name], graph, num_pointers=2, num_steps=60
        )
        return run_traversal(traversal, name)

    assert once(original) == once(from_cache), (
        f"{name} explores differently when the graph comes from cache"
    )


@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_traverse_return_type_is_recorded(name):
    """Pins which traversals return a batch and which move pointers in place.

    Three of the eight return None. `BasicTrainingCapability` uses the returned
    batch, so those three would train on nothing if wired up -- worth having
    written down rather than rediscovered.
    """
    graph = build_graph(node_count=12)
    traversal = build_traversal(
        get_traversal_classes()[name], graph, num_pointers=2, num_steps=30
    )
    if name in LEGACY_POINTER_TRAVERSALS:
        assert traversal.traverse() is None
        assert traversal.get_pointers(), "the legacy style must still expose pointers"
    else:
        batch = traversal.traverse(batch_size=4)
        assert isinstance(batch, list)


def test_ivalue_traversal_prefers_high_value_neighbors():
    """The I-value traversal must actually use the trainer's scores.

    With one node scored far above the rest, a traversal that ignored I-values would
    reach it only by chance.
    """
    from test_helpers.determinism import configure_determinism

    configure_determinism(seed=9, mode="strict", allow_multi_gpu=True)
    graph = build_graph(node_count=20)
    target = graph.get_nodes()[7].node_id
    trainer = DummyTrainer(
        i_values={node.node_id: (5.0 if node.node_id == target else 0.01)
                  for node in graph.get_nodes()}
    )

    traversal = get_traversal_classes()["IValueTraversal"](
        graph, num_pointers=2, num_steps=40, trainer=trainer
    )
    visited = set()
    for _ in range(6):
        batch = traversal.traverse(batch_size=8)
        if not batch:
            break
        visited.update(node.node_id for node in batch)

    assert trainer.calls, "the traversal never consulted the trainer"
    assert target in visited, "the highest-I-value node was never visited"
