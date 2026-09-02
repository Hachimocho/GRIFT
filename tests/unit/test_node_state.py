"""Per-node training state: fixed-length features, bounded values, EWMA behaviour.

The length is the load-bearing property. `DQNCapability._initialize_dqns` probes
`feature_dim` exactly once from `sample_nodes[0]`, so a vector whose length depends on
whether that particular node had been seen would size the network off an accident.
"""

import math

import pytest

from trainers.capabilities.node_state import (
    NEUTRAL_LOSS,
    NEUTRAL_PROB,
    STATE_FEATURE_COUNT,
    NodeTrainingState,
)


class _Node:
    def __init__(self, node_id):
        self.node_id = node_id


def test_unseen_and_seen_vectors_are_the_same_length():
    state = NodeTrainingState()
    unseen, seen = _Node("a"), _Node("b")
    state.observe(seen, 0.9, 0.1, epoch=0)
    assert len(state.features(unseen)) == STATE_FEATURE_COUNT
    assert len(state.features(seen)) == STATE_FEATURE_COUNT


def test_unseen_node_reports_neutral_and_unseen():
    state = NodeTrainingState()
    features = state.features(_Node("a"))
    assert features[0] == 0.0                      # seen flag
    assert features[1] == pytest.approx(NEUTRAL_PROB)
    assert features[3] == pytest.approx(0.0)       # margin
    assert features[4] == pytest.approx(0.0)       # times_seen


def test_neutral_loss_is_the_bce_of_an_uncommitted_prediction():
    assert NEUTRAL_LOSS == pytest.approx(-math.log(0.5))


def test_every_feature_is_bounded():
    state = NodeTrainingState()
    node = _Node("a")
    # Deliberately extreme: an unbounded loss must not dominate the input scale.
    state.observe(node, 1.0, 10_000.0, epoch=0)
    for value in state.features(node, epoch=10_000):
        assert 0.0 <= value <= 1.0


def test_observations_move_toward_the_new_value():
    state = NodeTrainingState()
    node = _Node("a")
    state.observe(node, 0.0, 1.0, epoch=0)
    first = state.features(node)[1]
    state.observe(node, 1.0, 1.0, epoch=0)
    second = state.features(node)[1]
    assert second > first          # EWMA moves toward the new observation
    assert second < 1.0           # but does not jump straight to it


def test_times_seen_counts_and_saturates():
    state = NodeTrainingState()
    node = _Node("a")
    for _ in range(500):
        state.observe(node, 0.5, 0.5, epoch=0)
    assert state.get(node)[2] == 500
    assert state.features(node)[4] == pytest.approx(1.0)


def test_staleness_grows_with_epoch_distance():
    state = NodeTrainingState()
    node = _Node("a")
    state.observe(node, 0.5, 0.5, epoch=2)
    fresh = state.features(node, epoch=2)[5]
    stale = state.features(node, epoch=7)[5]
    assert fresh == pytest.approx(0.0)
    assert stale > fresh


@pytest.mark.parametrize("prob, loss", [(float("nan"), 1.0), (0.5, float("inf")), (0.5, float("nan"))])
def test_non_finite_observations_are_ignored(prob, loss):
    # A diverged batch must not poison the features that steer sampling for the whole run.
    state = NodeTrainingState()
    node = _Node("a")
    state.observe(node, prob, loss)
    assert state.get(node) is None
    assert len(state) == 0


def test_a_node_without_an_id_is_ignored_rather_than_crashing():
    state = NodeTrainingState()
    state.observe(object(), 0.5, 0.5)
    assert len(state) == 0
