"""`GroupPerformanceTracker`: weight or select by *realised* per-group error rate.

`GroupTargeting` (`group_targeting.py`) targets groups by mean *predicted* I-value --
this project's own gate result (`docs/ivalue_gate_result.md`) found the DQN's
predictions barely correlate with realised learning gain, so a tracker fed by actual
outcomes is a genuinely different, more direct signal. These tests pin the two
properties that make it safe to use: the multiplier is bounded and geometric about 1
(so it stays comparable to every other weight in `loss_weighting.py`), and it never
uses a sample's own label to weight itself.
"""

import math

import pytest

from trainers.capabilities.group_fairness import (
    DEFAULT_TARGET_GROUPS,
    GroupPerformanceTracker,
    fairness_weights_for_batch,
    pool_targeting_for,
)


class _Node:
    def __init__(self, node_id, gender=0, race=0, age=0):
        self.node_id = node_id
        self.attributes = {
            "Ground Truth Gender": gender,
            "Ground Truth Race": race,
            "Ground Truth Age": age,
        }


def make_group(prefix, count, gender, race, age=0):
    return [_Node(f"{prefix}{i}", gender=gender, race=race, age=age) for i in range(count)]


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

def test_disabled_tracker_is_a_complete_noop():
    tracker = GroupPerformanceTracker(enabled=False)
    node = _Node("a")
    tracker.observe(node, correct=False)
    assert tracker.multiplier(node) == 1.0
    assert tracker.is_targeted(node) is True
    assert tracker.summary()["groups_seen"] == 0


def test_a_node_with_no_group_attributes_is_ignored():
    tracker = GroupPerformanceTracker(enabled=True, min_observations=1)
    node = _Node("a")
    node.attributes = {}
    tracker.observe(node, correct=False)
    assert tracker.summary()["groups_seen"] == 0
    assert tracker.multiplier(node) == 1.0


def test_an_under_observed_group_is_not_weighted():
    """A handful of noisy samples must not be enough to boost or penalise a group."""
    tracker = GroupPerformanceTracker(enabled=True, min_observations=50)
    weak = make_group("w", 10, gender=0, race=0)
    for node in weak:
        tracker.observe(node, correct=False)
    assert tracker.multiplier(weak[0]) == 1.0
    assert tracker.is_targeted(weak[0]) is True  # nothing eligible yet -> sample normally


def test_eligible_once_min_observations_is_reached():
    tracker = GroupPerformanceTracker(enabled=True, min_observations=10, target_groups=1)
    weak = make_group("w", 10, gender=0, race=0)
    strong = make_group("s", 10, gender=1, race=0)
    for node in weak:
        tracker.observe(node, correct=False)
    for node in strong:
        tracker.observe(node, correct=True)
    assert tracker.multiplier(weak[0]) > 1.0
    assert tracker.multiplier(strong[0]) < 1.0
    assert tracker.is_targeted(weak[0]) is True
    assert tracker.is_targeted(strong[0]) is False


# --------------------------------------------------------------------------- #
# multiplier(): bounded, geometric about 1
# --------------------------------------------------------------------------- #

def test_multiplier_stays_inside_the_clip_bounds():
    tracker = GroupPerformanceTracker(enabled=True, min_observations=10)
    weak = make_group("w", 20, gender=0, race=0)
    strong = make_group("s", 20, gender=1, race=0)
    for node in weak:
        tracker.observe(node, correct=False)   # 100% error
    for node in strong:
        tracker.observe(node, correct=True)    # 0% error
    for node in weak + strong:
        assert 0.5 <= tracker.multiplier(node, clip=2.0) <= 2.0


def test_a_group_at_the_overall_mean_gets_a_multiplier_of_one():
    """Neither boosted nor penalised when it is exactly as good as average."""
    tracker = GroupPerformanceTracker(enabled=True, min_observations=10, target_groups=1)
    average = make_group("a", 20, gender=0, race=0)
    also_average = make_group("b", 20, gender=1, race=0)
    # Same error rate for both groups -> both equal the overall mean.
    for node in average + also_average:
        tracker.observe(node, correct=(hash(node.node_id) % 2 == 0))
    ratio = tracker.multiplier(average[0]) / tracker.multiplier(also_average[0])
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_worse_than_average_group_gets_more_than_one():
    tracker = GroupPerformanceTracker(enabled=True, min_observations=10, target_groups=1)
    bad = make_group("bad", 20, gender=0, race=0)
    good = make_group("good", 20, gender=1, race=0)
    ok = make_group("ok", 20, gender=0, race=1)
    for node in bad:
        tracker.observe(node, correct=False)
    for node in good:
        tracker.observe(node, correct=True)
    for node in ok:
        tracker.observe(node, correct=(hash(node.node_id) % 2 == 0))
    assert tracker.multiplier(bad[0]) > 1.0
    assert tracker.multiplier(good[0]) < 1.0


def test_clip_of_one_disables_reweighting():
    """clip=1.0 -> [1/1, 1] -> every multiplier collapses to exactly 1."""
    tracker = GroupPerformanceTracker(enabled=True, min_observations=10)
    weak = make_group("w", 20, gender=0, race=0)
    for node in weak:
        tracker.observe(node, correct=False)
    assert tracker.multiplier(weak[0], clip=1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# is_targeted(): the selection-side interface, interchangeable with GroupTargeting
# --------------------------------------------------------------------------- #

def test_is_targeted_picks_the_worst_k_groups():
    tracker = GroupPerformanceTracker(enabled=True, min_observations=5, target_groups=1)
    worst = make_group("worst", 10, gender=0, race=0)
    middle = make_group("mid", 10, gender=0, race=1)
    best = make_group("best", 10, gender=0, race=2)
    for node in worst:
        tracker.observe(node, correct=False)      # 100% error
    for node in middle:
        tracker.observe(node, correct=(hash(node.node_id) % 2 == 0))  # ~50%
    for node in best:
        tracker.observe(node, correct=True)        # 0% error
    assert tracker.is_targeted(worst[0]) is True
    assert tracker.is_targeted(best[0]) is False


def test_shares_the_interface_group_targeting_exposes():
    """`_candidates` calls `targeting.is_targeted(node)` on whatever object is in the
    traversal's `group_targeting` slot -- this must work identically for either."""
    from trainers.capabilities.group_targeting import GroupTargeting

    performance = GroupPerformanceTracker(enabled=True)
    ivalue = GroupTargeting(enabled=True)
    node = _Node("a")
    # Both must accept a node and return a bool without raising, regardless of state.
    assert isinstance(performance.is_targeted(node), bool)
    assert isinstance(ivalue.is_targeted(node), bool)


# --------------------------------------------------------------------------- #
# No self-influence: a sample's own outcome must not affect its own weight
# --------------------------------------------------------------------------- #

def test_a_batchs_weight_must_use_state_from_before_that_batch():
    """The pattern every call site follows: compute weights, *then* observe. Pinned here
    against the tracker directly, since the training loop cannot easily be unit tested.

    Needs a second, static group as a baseline: with only one group ever observed,
    `multiplier` has nothing to compare it against and always returns 1.0 regardless of
    that group's error rate, which would mask the very leak this test checks for.
    """
    tracker = GroupPerformanceTracker(enabled=True, min_observations=5, target_groups=1)
    baseline = make_group("base", 20, gender=1, race=0)
    for node in baseline:
        tracker.observe(node, correct=(hash(node.node_id) % 2 == 0))  # ~50%, fixed

    group_a = make_group("a", 5, gender=0, race=0)
    for node in group_a:
        tracker.observe(node, correct=True)   # perfect so far -> currently *better* than baseline

    # A weight computed for a *new* batch of the same group, before observing it, must
    # reflect only the prior perfect record -- not this batch's own (bad) outcome.
    new_batch = make_group("a", 5, gender=0, race=0)
    weight_before = tracker.multiplier(new_batch[0])
    for node in new_batch:
        tracker.observe(node, correct=False)  # now folded in
    weight_after = tracker.multiplier(new_batch[0])
    assert weight_before != weight_after, "the tracker must actually update"
    assert weight_before < 1.0, "used only the prior perfect record, better than baseline"
    assert weight_after > weight_before, "now reflects the batch that was just observed"


# --------------------------------------------------------------------------- #
# fairness_weights_for_batch
# --------------------------------------------------------------------------- #

def test_returns_none_when_the_tracker_is_absent_or_disabled():
    assert fairness_weights_for_batch([_Node("a")], None) is None
    assert fairness_weights_for_batch([_Node("a")], GroupPerformanceTracker(enabled=False)) is None


def test_returns_none_for_an_empty_batch():
    tracker = GroupPerformanceTracker(enabled=True)
    assert fairness_weights_for_batch([], tracker) is None


def test_returns_a_tensor_of_the_right_length():
    import torch

    tracker = GroupPerformanceTracker(enabled=True, min_observations=1, target_groups=1)
    nodes = make_group("n", 4, gender=0, race=0)
    for node in nodes:
        tracker.observe(node, correct=False)
    weights = fairness_weights_for_batch(nodes, tracker, clip=2.0)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (4,)


# --------------------------------------------------------------------------- #
# pool_targeting_for: the traversal-factory helper
# --------------------------------------------------------------------------- #

class _Trainer:
    def __init__(self, fairness_selection=False, group_targeting=None, fairness_tracker=None):
        self.ivalue_fairness_selection = fairness_selection
        self.group_targeting = group_targeting
        self.fairness_tracker = fairness_tracker


def test_defaults_to_group_targeting():
    gt, ft = object(), object()
    trainer = _Trainer(fairness_selection=False, group_targeting=gt, fairness_tracker=ft)
    assert pool_targeting_for(trainer) is gt


def test_fairness_selection_wins_when_set():
    gt, ft = object(), object()
    trainer = _Trainer(fairness_selection=True, group_targeting=gt, fairness_tracker=ft)
    assert pool_targeting_for(trainer) is ft


def test_missing_attributes_are_handled_gracefully():
    class _BareTrainer:
        pass

    assert pool_targeting_for(_BareTrainer()) is None
