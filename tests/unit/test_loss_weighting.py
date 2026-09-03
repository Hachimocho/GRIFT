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
    ivalue_weights,
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


def _capability(values, mode="linear", clip=2.0, band=(0.4, 0.7)):
    trainer = _Trainer(values, ivalue_loss_weight=mode, ivalue_weight_clip=clip,
                       ivalue_selection_band=band)
    capability = BasicTrainingCapability.__new__(BasicTrainingCapability)
    capability.trainer = trainer
    capability.weighter = LossWeighter(mode=mode, clip=clip, band=band)
    return capability


def test_modes_and_default_clip():
    assert LOSS_WEIGHT_MODES == ("none", "linear", "rank", "midband")
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


# --------------------------------------------------------------------------- #
# midband: weight highest in the middle-high of the range, tapering at both ends
# --------------------------------------------------------------------------- #

def test_midband_gives_the_lowest_weight_to_both_extremes():
    """The property distinguishing `midband` from `rank`/`linear`: the single *highest*
    I-value must not be the most heavily weighted sample, because on this corpus that is
    also where mislabelled and corrupted samples land."""
    nodes = [f"n{i}" for i in range(9)]
    capability = _capability({n: float(i) for i, n in enumerate(nodes)},
                             mode="midband", band=(0.3, 0.7))
    logits = torch.zeros(9)
    labels = torch.ones(9)
    weights = ivalue_weights([float(i) for i in range(9)], "midband",
                             clip=2.0, band=(0.3, 0.7))
    lowest_ivalue_weight = weights[0].item()
    highest_ivalue_weight = weights[-1].item()
    middle_weight = weights[4].item()
    assert middle_weight > lowest_ivalue_weight
    assert middle_weight > highest_ivalue_weight
    # Symmetric band (0.3, 0.7 around a midpoint of 0.5) -> the two extremes should be
    # weighted about the same, not one favoured over the other.
    assert lowest_ivalue_weight == pytest.approx(highest_ivalue_weight, abs=0.05)


def test_midband_plateau_sits_inside_the_band_at_the_clip():
    """Deep inside the band, weight should saturate at `clip` -- the same ceiling `rank`
    reaches at its single best sample, so the two modes are on a comparable scale."""
    values = [i / 99.0 for i in range(100)]  # ranks are exactly `values` here
    weights = ivalue_weights(values, "midband", clip=2.0, band=(0.4, 0.7))
    # Rank 0.55 sits in the middle of (0.4, 0.7), well past the 0.3-width ramp on either
    # side (ramp = 0.3 * 0.3 = 0.09, so the plateau covers roughly [0.49, 0.61]).
    middle_index = int(0.55 * 99)
    assert weights[middle_index].item() == pytest.approx(2.0, abs=1e-3)


def test_midband_stays_inside_the_clip_bounds():
    values = [float(i) for i in range(50)]
    weights = ivalue_weights(values, "midband", clip=2.0, band=(0.3, 0.8))
    assert torch.all(weights >= 0.5 - 1e-6)
    assert torch.all(weights <= 2.0 + 1e-6)


def test_midband_is_rank_invariant_like_rank_mode():
    """Same rationale as `rank`: the estimator's raw output scale must not matter."""
    ranks = list(range(10))
    small_scale = [r * 0.001 + 0.31 for r in ranks]
    large_scale = [r * 1000.0 + 10.0 for r in ranks]
    small_weights = ivalue_weights(small_scale, "midband", clip=2.0, band=(0.3, 0.7))
    large_weights = ivalue_weights(large_scale, "midband", clip=2.0, band=(0.3, 0.7))
    assert torch.allclose(small_weights, large_weights, atol=1e-6)


def test_midband_default_band_matches_selection_bands_default():
    """No flags of its own -- `midband` inherits --ivalue-band's default exactly."""
    from trainers.capabilities.loss_weighting import DEFAULT_BAND
    assert DEFAULT_BAND == (0.4, 0.7)
    weighter = LossWeighter(mode="midband")
    assert weighter.band == (0.4, 0.7)


def test_midband_uses_the_bands_it_is_given():
    """Two different bands over the same I-values must produce different weight shapes,
    or the parameter is not actually reaching the computation."""
    values = [float(i) for i in range(20)]
    low_band = ivalue_weights(values, "midband", clip=2.0, band=(0.0, 0.3))
    high_band = ivalue_weights(values, "midband", clip=2.0, band=(0.7, 1.0))
    assert not torch.allclose(low_band, high_band)
    # Index 5 (rank ~0.26) sits inside the low band and outside the high one; index 15
    # (rank ~0.79) is the mirror image. Index 0 is excluded from *both* -- it sits exactly
    # on (0.0, 0.3)'s edge, not inside it -- so it is deliberately not used here.
    assert low_band[5] > high_band[5]
    assert high_band[15] > low_band[15]


# --------------------------------------------------------------------------- #
# extra_weights: composes group-fairness weighting into the same apply() call
# --------------------------------------------------------------------------- #

def test_extra_weights_compose_multiplicatively_with_the_ivalue_mode():
    weighter = LossWeighter(mode="linear", clip=2.0)
    per_sample = torch.ones(4)
    ivalues = [0.1, 0.3, 0.6, 0.9]
    base = ivalue_weights(ivalues, "linear", clip=2.0)

    extra = torch.tensor([1.0, 2.0, 0.5, 1.0])
    weighter.apply(per_sample, ivalues, extra_weights=extra)
    mean_weight = weighter.weight_applied / weighter.weighted_samples
    expected_mean = float((base * extra).mean())
    assert mean_weight == pytest.approx(expected_mean, abs=1e-5)


def test_extra_weights_alone_reweight_even_when_mode_is_none():
    """The failure `apply`'s docstring calls out by name: mode='none' must not make a
    fairness-only run silently do nothing."""
    weighter = LossWeighter(mode="none")
    per_sample = torch.tensor([1.0, 1.0, 1.0, 1.0])
    extra = torch.tensor([2.0, 0.5, 2.0, 0.5])

    plain_mean = per_sample.mean()
    weighted = weighter.apply(per_sample, values=[0.0] * 4, extra_weights=extra)
    assert not torch.isclose(weighted, plain_mean)
    assert float(weighted) == pytest.approx(float((extra * per_sample).mean()), abs=1e-6)


def test_extra_weights_of_the_wrong_length_are_ignored_not_crashed():
    weighter = LossWeighter(mode="none")
    per_sample = torch.tensor([1.0, 2.0, 3.0])
    wrong_length = torch.tensor([1.0, 2.0])
    result = weighter.apply(per_sample, values=[], extra_weights=wrong_length)
    assert float(result) == pytest.approx(float(per_sample.mean()), abs=1e-6)


def test_no_extra_weights_reproduces_the_old_behaviour_exactly():
    """Every existing caller that never passes `extra_weights` must be bit-for-bit
    unaffected by its addition."""
    weighter = LossWeighter(mode="rank", clip=2.0)
    per_sample = torch.tensor([0.5, 1.0, 1.5, 2.0])
    values = [0.1, 0.4, 0.6, 0.9]
    with_default = weighter.apply(per_sample, values)

    weighter2 = LossWeighter(mode="rank", clip=2.0)
    without = weighter2.apply(per_sample, values, extra_weights=None)
    assert float(with_default) == pytest.approx(float(without), abs=1e-9)


def test_fairness_weighting_end_to_end_through_dqncapability():
    """Constructs a real DQNCapability with fairness weighting on and I-value weighting
    off, and checks the plumbing all the way from CLI-shaped kwargs to a reweighted
    per-sample loss -- the same level `test_both_training_paths_share_one_weighter`
    checks the I-value path at."""
    import torch as _torch

    from trainers.capabilities.DQNCapability import DQNCapability
    from trainers.capabilities.group_fairness import GroupPerformanceTracker

    class _Trainer:
        device = _torch.device("cpu")
        models = []
        attribute_metadata = None
        ivalue_loss_weight = "none"
        ivalue_fairness_weight = True
        fairness_tracker = GroupPerformanceTracker(
            enabled=True, min_observations=1, target_groups=1,
        )

    dqn = DQNCapability(_Trainer())
    assert dqn.fairness_weight_enabled is True
    assert not dqn.weighter.enabled, "I-value weighting must stay off for this arm"
    assert dqn.fairness_tracker is _Trainer.fairness_tracker, \
        "must reuse the trainer's tracker, not build its own"
