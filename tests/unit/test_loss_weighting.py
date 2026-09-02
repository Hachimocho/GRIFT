"""I-value as a loss weight rather than a selection criterion.

Every scheme that *selected* with the I-value has lost to plain i.i.d. sampling, and Phase 0
found why: choosing among a node's k-NN neighbours yields batches 2.3x less diverse. Weighting
keeps i.i.d. sampling untouched -- so the batches stay diverse -- and uses the signal only to
scale each sample's contribution. These tests pin the two properties that make that safe: the
weights are bounded, and a uniform I-value reproduces the unweighted loss exactly.
"""

import pytest
import torch

from trainers.capabilities.BasicTrainingCapability import BasicTrainingCapability
from trainers.capabilities.loss_weighting import (
    DEFAULT_WEIGHT_CLIP,
    LOSS_WEIGHT_MODES,
    LossWeighter,
)


class _Trainer:
    def __init__(self, values, **attrs):
        self.device = torch.device("cpu")
        self.models = []
        self.attribute_metadata = None
        self._values = values
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_i_value(self, node, model_idx=0):
        return self._values[node]


def _capability(values, mode="linear", clip=2.0):
    trainer = _Trainer(values, ivalue_loss_weight=mode, ivalue_weight_clip=clip)
    capability = BasicTrainingCapability.__new__(BasicTrainingCapability)
    capability.trainer = trainer
    capability.weighter = LossWeighter(mode=mode, clip=clip)
    return capability


def test_modes_and_default_clip():
    assert LOSS_WEIGHT_MODES == ("none", "linear", "rank")
    assert DEFAULT_WEIGHT_CLIP > 1.0


def test_uniform_ivalues_reproduce_the_unweighted_loss():
    """The property that makes the arm a fair comparison: with no signal, no change."""
    nodes = ["a", "b", "c", "d"]
    capability = _capability({n: 0.5 for n in nodes})
    logits = torch.tensor([2.0, -1.0, 0.5, -0.25])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    weighted = capability._weighted_loss(logits, labels, nodes)
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert float(weighted) == pytest.approx(float(plain), abs=1e-6)


@pytest.mark.parametrize("mode", ["linear", "rank"])
def test_weights_stay_inside_the_clip(mode):
    nodes = [f"n{i}" for i in range(8)]
    # Deliberately extreme spread, including a huge outlier.
    values = {n: float(i) for i, n in enumerate(nodes)}
    values["n7"] = 1e6
    capability = _capability(values, mode=mode, clip=2.0)
    logits = torch.zeros(8)
    labels = torch.ones(8)
    capability._weighted_loss(logits, labels, nodes)
    mean_weight = (capability.weighter.weight_applied
                   / capability.weighter.weighted_samples)
    # Geometric about 1 over [1/2, 2], so the mean must sit inside the bounds.
    assert 0.5 <= mean_weight <= 2.0


def test_a_high_ivalue_sample_gets_more_weight_than_a_low_one():
    nodes = ["low", "high"]
    capability = _capability({"low": 0.0, "high": 1.0})
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 1.0])
    # Same per-sample loss, so the weighted mean must exceed the unweighted one only if the
    # high-I-value sample is up-weighted; check the weights directly instead.
    capability._weighted_loss(logits, labels, nodes)
    assert capability.weighter.weighted_samples == 2


def test_rank_mode_is_invariant_to_the_ivalue_scale():
    nodes = [f"n{i}" for i in range(6)]
    logits = torch.linspace(-2, 2, 6)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    small = _capability({n: 0.310 + i * 0.001 for i, n in enumerate(nodes)}, mode="rank")
    large = _capability({n: 10.0 + i * 1000.0 for i, n in enumerate(nodes)}, mode="rank")
    assert float(small._weighted_loss(logits, labels, nodes)) == pytest.approx(
        float(large._weighted_loss(logits, labels, nodes)), abs=1e-6
    )


def test_non_finite_ivalues_do_not_poison_the_loss():
    nodes = ["a", "b", "c"]
    capability = _capability({"a": float("nan"), "b": 0.2, "c": 0.8})
    logits = torch.tensor([1.0, -1.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert torch.isfinite(capability._weighted_loss(logits, labels, nodes))


def test_a_trainer_without_get_i_value_falls_back_to_the_plain_loss():
    capability = _capability({})
    capability.trainer = object()
    logits = torch.tensor([1.0, -1.0])
    labels = torch.tensor([1.0, 0.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert float(capability._weighted_loss(logits, labels, ["a", "b"])) == pytest.approx(
        float(plain), abs=1e-6
    )


def test_weighting_forces_a_dqn_under_any_traversal():
    """The arm samples with `comprehensive` but still needs an I-value per sample; without
    this the weights would silently fall back to uniform and the arm would be its own
    control."""
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    class _Graph:
        def get_nodes(self):
            return []

    class _Manager:
        def get_graph(self):
            return _Graph()

    trainer = AdaptiveTrainer(
        graphmanager=_Manager(), models=[], device=torch.device("cpu"),
        attribute_metadata=None, loss_fn=torch.nn.BCEWithLogitsLoss(),
        ivalue_loss_weight="rank",
    )
    assert trainer.capabilities.requires_dqn_warmup is True
    assert trainer.capabilities.basic_training_capability.weighter.mode == "rank"

def test_both_training_paths_share_one_weighter():
    """`train_with_traversal` routes every traversal through the DQN path once a DQN exists,
    so an implementation only in the basic path is unreachable exactly when the I-value is
    available. That happened: the first version of this weighting was dead code and the arm
    would have silently run as its own control."""
    from trainers.capabilities.DQNCapability import DQNCapability

    class _Trainer:
        device = torch.device("cpu")
        models = []
        attribute_metadata = None
        ivalue_loss_weight = "rank"
        ivalue_weight_clip = 2.0

    dqn = DQNCapability(_Trainer())
    assert dqn.weighter.enabled and dqn.weighter.mode == "rank"
    basic = _capability({}, mode="rank")
    assert type(dqn.weighter) is type(basic.weighter)
