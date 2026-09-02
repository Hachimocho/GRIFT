"""Withdrawing nodes whose measured learning gain stays negative.

38% of trained samples show a negative measured gain -- training on them made the model worse
on themselves -- so there is a large population to filter, and on a corpus with label noise
that is a plausible way for I-values to help without being used to select. Two properties make
it safe: a single bad visit is not enough, and the total withdrawn is capped so the arm keeps
measuring "train on data that helps" rather than "train on less data".
"""

import pytest
import torch

from trainers.capabilities.DQNCapability import DEFAULT_BAN_MAX_FRACTION, DQNCapability
from trainers.capabilities.node_state import NodeTrainingState


class _Node:
    def __init__(self, node_id):
        self.node_id = node_id


def test_one_bad_visit_is_not_enough():
    state = NodeTrainingState()
    node = _Node("a")
    state.observe_gain(node, -5.0)
    assert state.is_harmful(node, min_visits=3) is False, \
        "the gain distribution straddles zero; one observation means nothing"


def test_persistently_negative_is_harmful_once_min_visits_is_reached():
    state = NodeTrainingState()
    node = _Node("a")
    for _ in range(2):
        state.observe_gain(node, -0.2)
    assert state.is_harmful(node, min_visits=3) is False
    state.observe_gain(node, -0.2)
    assert state.is_harmful(node, min_visits=3) is True


def test_a_node_that_helps_on_average_is_kept():
    state = NodeTrainingState()
    node = _Node("a")
    for gain in (0.5, -0.1, 0.3, -0.05):
        state.observe_gain(node, gain)
    assert state.is_harmful(node, min_visits=3) is False


def test_running_mean_weights_every_visit_equally():
    state = NodeTrainingState()
    node = _Node("a")
    for gain in (1.0, 0.0, -1.0, 0.0):
        state.observe_gain(node, gain)
    mean, visits = state.gain_record(node)
    assert visits == 4
    assert mean == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_gains_are_ignored(bad):
    state = NodeTrainingState()
    node = _Node("a")
    state.observe_gain(node, bad)
    assert state.gain_record(node) is None


def test_unseen_node_is_never_harmful():
    assert NodeTrainingState().is_harmful(_Node("z"), min_visits=1) is False


def _capability(min_visits=3, max_fraction=1.0, graph_size=100):
    class _Graph:
        def get_nodes(self):
            return [_Node(f"g{i}") for i in range(graph_size)]

    class _Manager:
        def get_graph(self):
            return _Graph()

    class _Trainer:
        graphmanager = _Manager()

    capability = DQNCapability.__new__(DQNCapability)
    capability.trainer = _Trainer()
    capability.node_state = NodeTrainingState()
    capability.ban_min_visits = min_visits
    capability.ban_max_fraction = max_fraction
    capability.banned_nodes = set()
    capability.bans_this_epoch = 0
    return capability


def test_harmful_nodes_are_dropped_from_the_batch():
    capability = _capability()
    good, bad = _Node("good"), _Node("bad")
    for _ in range(3):
        capability.node_state.observe_gain(bad, -0.4)
        capability.node_state.observe_gain(good, 0.4)

    kept = capability._drop_harmful([good, bad])
    assert kept == [good]
    assert "bad" in capability.banned_nodes
    assert capability.bans_this_epoch == 1


def test_a_banned_node_stays_banned_without_re_measuring():
    capability = _capability()
    bad = _Node("bad")
    for _ in range(3):
        capability.node_state.observe_gain(bad, -0.4)
    capability._drop_harmful([bad])
    assert capability._drop_harmful([bad]) == []


def test_the_ban_is_capped():
    """Past the cap the arm stops testing 'data that helps' and starts testing 'less data'."""
    capability = _capability(max_fraction=0.02, graph_size=100)   # ceiling of 2
    harmful = [_Node(f"h{i}") for i in range(10)]
    for node in harmful:
        for _ in range(3):
            capability.node_state.observe_gain(node, -0.4)

    kept = capability._drop_harmful(harmful)
    assert len(capability.banned_nodes) == 2, "the ceiling must hold"
    assert len(kept) == 8, "nodes past the cap are trained on, not silently dropped"


def test_banning_is_off_by_default_and_a_no_op():
    capability = _capability(min_visits=0)
    nodes = [_Node("a"), _Node("b")]
    assert capability._drop_harmful(nodes) is nodes


def test_default_cap_is_conservative():
    assert 0.0 < DEFAULT_BAN_MAX_FRACTION <= 0.5


def test_banning_requires_a_measured_gain():
    """Only --ivalue-reward learning_gain produces one, via the extra post-update pass."""
    class _Trainer:
        device = torch.device("cpu")
        models = []
        attribute_metadata = None
        ivalue_reward = "confidence"
        ivalue_ban_negative_gain = 3

    with pytest.raises(ValueError, match="ban-negative-gain"):
        DQNCapability(_Trainer())
