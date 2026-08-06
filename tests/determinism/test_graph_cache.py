"""Graph cache round-trip equivalence.

The asymmetry these cover is easy to miss and hard to debug: `export_edges_csv`
writes by iterating nodes then their edges, while `load_edges_from_csv` appends in
file order. So a cache-loaded graph and a freshly built graph agreed on the edge
*set* but disagreed on each node's adjacency *order* -- and traversals do
`random.choice(adjacent)` and argmax-over-neighbors with ties broken by list
position. Same seed, same data, different visited nodes, depending only on whether
the graph happened to come from cache.
"""

import pytest

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from tests.helpers.factories import (
    DummyTrainer, build_traversal, get_traversal_classes, make_attr_nodes,
)
from traversals.IValueTraversal import IValueTraversal


def build_graph(node_count=12, degree=3):
    """A deterministic graph where most nodes have several neighbors.

    Multiple neighbors matter: with degree 1 there is nothing for adjacency order
    to permute, so the bug would be invisible.
    """
    nodes = make_attr_nodes(node_count)
    for index, node in enumerate(nodes):
        for offset in range(1, degree + 1):
            peer = nodes[(index + offset) % node_count]
            if peer is node:
                continue
            if any(peer in (edge.get_nodes()) for edge in node.edges):
                continue
            edge = Edge(node, peer, x=None)
            node.add_edge(edge)
            peer.add_edge(edge)
    return HyperGraph(nodes)


def reload_from_cache(graph, path):
    """Export a graph's edges and rebuild it from that file with fresh nodes."""
    graph.export_edges_csv(str(path))
    fresh_nodes = make_attr_nodes(len(graph.get_nodes()))
    fresh = HyperGraph(fresh_nodes)
    fresh.load_edges_from_csv(str(path))
    return fresh


def adjacency_map(graph):
    return {
        node.node_id: [peer.node_id for peer in node.get_adjacent_nodes()]
        for node in graph.get_nodes()
    }


def test_export_then_load_preserves_the_edge_set(tmp_path):
    original = build_graph()
    restored = reload_from_cache(original, tmp_path / "edges.csv")
    assert restored.get_edge_list() == original.get_edge_list()


def test_export_then_load_preserves_adjacency_order(tmp_path):
    """The core C5 assertion: not just the same edges, the same *order*."""
    original = build_graph()
    original.canonicalize_edge_order()
    restored = reload_from_cache(original, tmp_path / "edges.csv")
    assert adjacency_map(restored) == adjacency_map(original)


def test_export_is_a_byte_stable_fixpoint(tmp_path):
    """export -> load -> export must reproduce the same file exactly."""
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    original = build_graph()
    restored = reload_from_cache(original, first_path)
    restored.export_edges_csv(str(second_path))

    assert first_path.read_bytes() == second_path.read_bytes()


def test_export_is_independent_of_insertion_order(tmp_path):
    """Two graphs with the same edges written in different orders match on disk."""
    forward_path = tmp_path / "forward.csv"
    reversed_path = tmp_path / "reversed.csv"

    forward = build_graph()
    forward.export_edges_csv(str(forward_path))

    shuffled = build_graph()
    for node in shuffled.get_nodes():
        node.edges.reverse()
    shuffled.nodes = list(reversed(shuffled.nodes))
    shuffled.export_edges_csv(str(reversed_path))

    assert forward_path.read_bytes() == reversed_path.read_bytes()


@pytest.mark.parametrize("traversal_name", sorted(get_traversal_classes()))
def test_cached_and_fresh_graphs_induce_the_same_traversal(tmp_path, traversal_name):
    """Same seed on a fresh graph and a cache-loaded graph must visit the same nodes.

    This is the user-visible consequence of the adjacency-order asymmetry, and the
    reason it matters: a run resumed from a warm cache silently explored a
    different part of the graph than the run that built it.
    """
    from test_helpers.determinism import configure_determinism

    traversal_class = get_traversal_classes()[traversal_name]
    accepts_batch_size = "batch_size" in __import__("inspect").signature(
        traversal_class.traverse
    ).parameters

    def visit_sequence(graph):
        configure_determinism(seed=4242, mode="strict", allow_multi_gpu=True)
        traversal = build_traversal(traversal_class, graph, num_pointers=2, num_steps=16)
        sequence = []
        for _ in range(4):
            batch = traversal.traverse(batch_size=4) if accepts_batch_size else traversal.traverse()
            if not batch:
                break
            sequence.append([getattr(node, "node_id", None) for node in batch])
        return sequence

    original = build_graph()
    original.canonicalize_edge_order()
    from_cache = reload_from_cache(original, tmp_path / "edges.csv")

    assert visit_sequence(original) == visit_sequence(from_cache), (
        f"{traversal_name} visits different nodes depending on whether the graph came "
        f"from cache"
    )


def test_ivalue_argmax_tie_break_is_stable_under_reordering(tmp_path):
    """Ties in the I-value argmax are broken by neighbor list position.

    `i_values.index(max(i_values))` returns the *first* maximum, so identical
    I-values resolve by adjacency order. Canonicalizing that order is what makes
    the choice reproducible.
    """
    from test_helpers.determinism import configure_determinism

    def first_choice(graph):
        configure_determinism(seed=11, mode="strict", allow_multi_gpu=True)
        # A trainer returning a constant makes every neighbor a tie, so the result
        # is determined purely by adjacency order.
        traversal = IValueTraversal(
            graph, num_pointers=1, num_steps=8,
            trainer=DummyTrainer(i_values={node.node_id: 0.5 for node in graph.get_nodes()}),
        )
        batch = traversal.traverse(batch_size=4)
        return [node.node_id for node in batch]

    original = build_graph()
    original.canonicalize_edge_order()
    baseline = first_choice(original)

    for node in original.get_nodes():
        node.edges.reverse()
    original.canonicalize_edge_order()
    assert first_choice(original) == baseline


def test_build_twice_produces_identical_graphs():
    """Two builds at the same seed must agree on edges and adjacency order."""
    from test_helpers.determinism import configure_determinism

    def build():
        configure_determinism(seed=99, mode="strict", allow_multi_gpu=True)
        graph = build_graph()
        graph.canonicalize_edge_order()
        return graph.get_edge_list(), adjacency_map(graph)

    assert build() == build()
