"""The I-value sign, per reward mode. This is the test that matters most in this change.

`DQNModel.predict_i_value` returns `1 - sigmoid(Q)`. Under the *confidence* reward that is
correct: Q regresses "+confidence if right, -confidence if wrong", so a high Q means the
sample is already mastered and the inversion turns it into "the model does poorly here",
which is what makes the traversal's argmax an uncertainty sampler.

Under the *learning_gain* reward a high Q already means "training on this helped", so the
inversion has to be undone. Getting that backwards produces a sampler that confidently
selects the least informative samples available and raises no error anywhere -- the run
completes, the numbers look plausible, and the method is silently sabotaged. Hence an
explicit assertion on the mapping rather than trust.
"""

import pytest
import torch

from trainers.capabilities.DQNCapability import (
    DEFAULT_IVALUE_REWARD,
    IVALUE_REWARDS,
    DQNCapability,
)


class _StubDQN:
    """Returns `1 - sigmoid(q)`, exactly as every real `predict_i_value` does."""

    def __init__(self, q):
        self.q = float(q)
        self.device = torch.device("cpu")

    def parameters(self):
        yield torch.zeros(1)

    def to(self, _device):
        return self

    def predict_i_value(self, features, embedding=None):
        return 1.0 - torch.sigmoid(torch.tensor([[self.q]]))


def _capability(reward, q):
    """Build without __init__: only the fields `get_i_value` reads are needed."""
    capability = DQNCapability.__new__(DQNCapability)
    capability.ivalue_reward = reward
    capability.dqns = [_StubDQN(q)]
    capability.embedding_dim = 4
    capability.use_state_features = False
    capability.node_state = None
    capability.current_epoch = 0
    capability.prediction_stats = {}
    capability.attribute_metadata = None
    capability.trainer = None
    capability._stats = []

    def _features(_node):
        return torch.zeros(3), torch.zeros(4)

    capability._get_dqn_features = _features
    capability.update_prediction_stats = lambda *a, **k: None
    return capability


class _Node:
    node_id = "n"
    attributes = {}
    label = 1


@pytest.mark.parametrize("q", [-4.0, -1.0, 0.0, 1.0, 4.0])
def test_confidence_reward_keeps_the_inversion(q):
    """High Q (already mastered) must yield a LOW I-value."""
    value = _capability("confidence", q).get_i_value(_Node())
    assert value == pytest.approx(1.0 - torch.sigmoid(torch.tensor(q)).item(), abs=1e-6)


@pytest.mark.parametrize("q", [-4.0, -1.0, 0.0, 1.0, 4.0])
def test_learning_gain_reward_undoes_the_inversion(q):
    """High Q (large measured gain) must yield a HIGH I-value."""
    value = _capability("learning_gain", q).get_i_value(_Node())
    assert value == pytest.approx(torch.sigmoid(torch.tensor(q)).item(), abs=1e-6)


def test_the_two_modes_are_opposites():
    # The property that actually protects against a silent sign flip.
    for q in (-3.0, -0.5, 0.5, 3.0):
        confidence = _capability("confidence", q).get_i_value(_Node())
        gain = _capability("learning_gain", q).get_i_value(_Node())
        assert confidence + gain == pytest.approx(1.0, abs=1e-6)


def test_mastered_samples_rank_below_struggling_ones_under_confidence():
    mastered = _capability("confidence", 4.0).get_i_value(_Node())   # high Q
    struggling = _capability("confidence", -4.0).get_i_value(_Node())
    assert struggling > mastered, "argmax I-value must prefer the sample the model fails on"


def test_high_gain_samples_rank_above_low_gain_ones_under_learning_gain():
    high = _capability("learning_gain", 4.0).get_i_value(_Node())
    low = _capability("learning_gain", -4.0).get_i_value(_Node())
    assert high > low, "argmax I-value must prefer the sample that taught the most"


def test_default_is_the_original_behaviour():
    assert DEFAULT_IVALUE_REWARD == "confidence"
    assert set(IVALUE_REWARDS) == {"confidence", "learning_gain"}


def test_unknown_reward_is_refused_at_construction():
    class _Trainer:
        device = torch.device("cpu")
        models = []
        attribute_metadata = None
        ivalue_reward = "whatever"

    with pytest.raises(ValueError, match="unknown ivalue_reward"):
        DQNCapability(_Trainer())
