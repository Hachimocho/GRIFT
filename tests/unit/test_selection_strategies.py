"""Selection mode, band and the candidate pool.

Phase 0 measured i-value batches at 2.3x less diverse than i.i.d. ones (0.0385 vs 0.0872,
p10 0.0121 vs 0.0623), because the walk's argmax ranges over the current node's k-NN
neighbours -- similar faces by construction. The pool widens that choice; the band tests
whether the *middle* of the range beats the extreme. A band over ~8 correlated neighbours
would be noise, which is why the two belong together.
"""

import pytest

from tests.helpers.factories import DummyTrainer, build_ring_graph
from traversals.IValueTraversal import SELECTION_MODES, IValueTraversal


def _traversal(graph, trainer=None, **kwargs):
    return IValueTraversal(graph, num_pointers=1, num_steps=10_000,
                           trainer=trainer or DummyTrainer(), **kwargs)


def test_defaults_are_the_historical_behaviour():
    graph, _n, _e = build_ring_graph(count=12)
    walk = _traversal(graph)
    assert walk.selection_mode == "max"
    assert walk.candidate_pool == 0


@pytest.mark.parametrize("mode", SELECTION_MODES)
def test_every_mode_returns_a_candidate(mode):
    graph, nodes, _e = build_ring_graph(count=20)
    walk = _traversal(graph, selection_mode=mode)
    values = [0.1, 0.5, 0.9, 0.3]
    assert walk._pick(nodes[:4], values) in nodes[:4]


def test_max_and_min_take_the_extremes():
    graph, nodes, _e = build_ring_graph(count=20)
    values = [0.1, 0.9, 0.5, 0.3]
    assert _traversal(graph, selection_mode="max")._pick(nodes[:4], values) is nodes[1]
    assert _traversal(graph, selection_mode="min")._pick(nodes[:4], values) is nodes[0]


def test_band_stays_inside_the_requested_quantile_range():
    graph, nodes, _e = build_ring_graph(count=40)
    candidates = nodes[:10]
    values = [i / 10.0 for i in range(10)]      # already ascending
    walk = _traversal(graph, selection_mode="band", selection_band=(0.4, 0.6))
    picked = {candidates.index(walk._pick(candidates, values)) for _ in range(60)}
    # low=0.4, high=0.6 over 10 candidates -> ranks 3..5 inclusive
    assert picked <= {3, 4, 5}, picked
    assert picked, "the band must yield something"


def test_band_never_returns_the_extremes_it_excludes():
    graph, nodes, _e = build_ring_graph(count=40)
    candidates = nodes[:12]
    values = list(range(12))
    walk = _traversal(graph, selection_mode="band", selection_band=(0.45, 0.55))
    for _ in range(60):
        index = candidates.index(walk._pick(candidates, values))
        assert index not in (0, 11), "the hardest and easiest must be excluded"


def test_band_is_scale_free():
    """A rank criterion, so an estimator's output range cannot change the choice -- the
    legacy family lives in a 0.02-wide band around 0.31 while the fixed ones are unbounded."""
    graph, nodes, _e = build_ring_graph(count=40)
    candidates = nodes[:8]
    walk = _traversal(graph, selection_mode="band", selection_band=(0.3, 0.4))
    small = [0.310, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317]
    large = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    walk._rng = __import__("random").Random(7)
    first = candidates.index(walk._pick(candidates, small))
    walk._rng = __import__("random").Random(7)
    second = candidates.index(walk._pick(candidates, large))
    assert first == second


def test_single_candidate_is_returned_by_every_mode():
    graph, nodes, _e = build_ring_graph(count=8)
    for mode in SELECTION_MODES:
        assert _traversal(graph, selection_mode=mode)._pick(nodes[:1], [0.5]) is nodes[0]


def test_empty_candidates_returns_none():
    graph, _n, _e = build_ring_graph(count=8)
    assert _traversal(graph)._pick([], []) is None


@pytest.mark.parametrize("bad", [(-0.1, 0.5), (0.5, 1.5), (0.7, 0.3)])
def test_invalid_bands_are_refused(bad):
    graph, _n, _e = build_ring_graph(count=8)
    with pytest.raises(ValueError, match="selection_band"):
        _traversal(graph, selection_band=bad)


def test_unknown_mode_is_refused():
    graph, _n, _e = build_ring_graph(count=8)
    with pytest.raises(ValueError, match="unknown selection_mode"):
        _traversal(graph, selection_mode="whatever")


def test_group_targeting_restricts_the_drawn_pool():
    from trainers.capabilities.group_targeting import GroupTargeting

    graph, nodes, _e = build_ring_graph(count=60)
    # Give half the nodes one group and half another, then target only the first.
    for index, node in enumerate(nodes):
        node.attributes["Ground Truth Race"] = "A" if index % 2 == 0 else "B"
        node.attributes["Ground Truth Gender"] = "m"
        node.attributes["Ground Truth Age"] = "y"

    targeting = GroupTargeting(top_groups=1, min_observations=1, enabled=True)
    for node in nodes[::2]:
        targeting.observe(node, 0.9)       # group A looks weak
    for node in nodes[1::2]:
        targeting.observe(node, 0.1)

    walk = _traversal(graph, candidate_pool=40, group_targeting=targeting)
    drawn = walk._candidates(nodes[0:1], set())
    extras = drawn[1:]
    assert extras, "the pool must still yield candidates"
    for node in extras:
        assert node.attributes["Ground Truth Race"] == "A", \
            "only targeted groups may enter the pool"


# --------------------------------------------------------------------------- #
# midband: `band`'s smooth analogue -- a weighted distribution, not a hard cutoff
# --------------------------------------------------------------------------- #

def test_midband_favours_the_middle_over_many_draws():
    """Not a hard window like `band`: over many draws, the plateau should be picked most
    often, but the extremes must remain reachable -- exactly the "unlikely, not
    impossible" contract `_pick`'s docstring makes."""
    import collections

    graph, nodes, _e = build_ring_graph(count=20)
    candidates = nodes[:10]
    values = [i / 9.0 for i in range(10)]  # ranks 0..9 map to quantiles 0..1 exactly
    walk = _traversal(graph, selection_mode="midband", selection_band=(0.4, 0.7),
                      candidate_pool=0)
    walk._rng = __import__("random").Random(11)

    counts = collections.Counter(
        candidates.index(walk._pick(candidates, values)) for _ in range(4000)
    )
    # Quantile 0.5 (index 4-5) sits on the plateau; quantiles 0 and 1 (indices 0, 9) sit
    # outside the band entirely.
    assert counts[4] + counts[5] > counts[0] + counts[9]
    # "Unlikely, not impossible": the extremes must still appear sometimes.
    assert counts[0] > 0 and counts[9] > 0


def test_midband_is_reproducible_under_a_fixed_seed():
    graph, nodes, _e = build_ring_graph(count=20)
    candidates = nodes[:10]
    values = [i / 9.0 for i in range(10)]

    walk_a = _traversal(graph, selection_mode="midband", selection_band=(0.4, 0.7))
    walk_a._rng = __import__("random").Random(3)
    picks_a = [candidates.index(walk_a._pick(candidates, values)) for _ in range(50)]

    walk_b = _traversal(graph, selection_mode="midband", selection_band=(0.4, 0.7))
    walk_b._rng = __import__("random").Random(3)
    picks_b = [candidates.index(walk_b._pick(candidates, values)) for _ in range(50)]

    assert picks_a == picks_b


def test_midband_is_scale_free_like_band():
    """Same rationale as `band`: a rank criterion must be unaffected by the estimator's
    raw output range."""
    graph, nodes, _e = build_ring_graph(count=40)
    candidates = nodes[:8]
    walk = _traversal(graph, selection_mode="midband", selection_band=(0.3, 0.7))
    small = [0.310, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317]
    large = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    walk._rng = __import__("random").Random(7)
    first = candidates.index(walk._pick(candidates, small))
    walk._rng = __import__("random").Random(7)
    second = candidates.index(walk._pick(candidates, large))
    assert first == second


def test_midband_falls_back_to_uniform_when_ivalues_are_unusable():
    """`ivalue_weights` returns `None` when every value is non-finite -- `_pick` must not
    propagate that as a crash."""
    graph, nodes, _e = build_ring_graph(count=8)
    candidates = nodes[:4]
    walk = _traversal(graph, selection_mode="midband")
    picked = walk._pick(candidates, [float("nan")] * 4)
    assert picked in candidates
