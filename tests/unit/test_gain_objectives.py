"""The two training objectives.

The target is why these exist. Realised learning gain has skew +3.96 and kurtosis +45.4,
with a median of +0.005 against a 99th percentile of +1.394, so an MSE is decided by a
handful of extremes. Selection meanwhile consumes only the *ordering* of candidates, never
the magnitude -- which is exactly what a pairwise ranking loss optimises and what makes it
immune to the tail.
"""

import pytest
import torch

from models.gain_estimator import (
    GainResidual,
    RANK_TIE_EPS,
    pairwise_ranking_loss,
    signed_log1p,
)

FEATURE_DIM = 31
CPU = torch.device("cpu")


def test_ranking_loss_is_zero_when_the_order_is_already_right():
    predictions = torch.tensor([3.0, 2.0, 1.0])
    targets = torch.tensor([3.0, 2.0, 1.0])
    perfect, pairs = pairwise_ranking_loss(predictions * 50, targets)
    wrong, _ = pairwise_ranking_loss(-predictions * 50, targets)
    assert pairs == 3
    assert float(perfect) < 1e-6
    assert float(wrong) > 1.0


def test_ranking_loss_ignores_monotone_rescaling_of_the_target():
    """The property that makes it tail-proof: only the order of the target matters."""
    predictions = torch.tensor([0.4, -0.1, 2.0, 0.7])
    targets = torch.tensor([0.01, -0.5, 1.4, 0.06])
    plain, _ = pairwise_ranking_loss(predictions, targets)
    squashed, _ = pairwise_ranking_loss(predictions, signed_log1p(targets))
    scaled, _ = pairwise_ranking_loss(predictions, targets * 1000.0)
    assert float(plain) == pytest.approx(float(squashed), abs=1e-6)
    assert float(plain) == pytest.approx(float(scaled), abs=1e-6)


def test_an_extreme_outlier_cannot_dominate_the_ranking_loss():
    predictions = torch.tensor([0.1, 0.2, 0.3])
    targets = torch.tensor([0.001, 0.002, 0.003])
    modest, _ = pairwise_ranking_loss(predictions, targets)
    # Same ordering, one target 1000x larger -- an MSE would be swamped; a rank loss is not.
    targets_with_outlier = torch.tensor([0.001, 0.002, 3.0])
    extreme, _ = pairwise_ranking_loss(predictions, targets_with_outlier)
    assert float(modest) == pytest.approx(float(extreme), abs=1e-6)


def test_ties_are_skipped_and_reported():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    identical = torch.zeros(3)
    loss, pairs = pairwise_ranking_loss(predictions, identical)
    assert pairs == 0, "all-tied targets must contribute no pairs"
    assert float(loss) == 0.0

    nearly = torch.tensor([0.0, RANK_TIE_EPS / 2, RANK_TIE_EPS / 3])
    _loss, near_pairs = pairwise_ranking_loss(predictions, nearly)
    assert near_pairs == 0, "near-ties are noise at a median gain of +0.005"


def test_signed_log1p_is_monotone_and_keeps_the_sign():
    values = torch.tensor([-5.0, -0.3, 0.0, 0.3, 5.0])
    out = signed_log1p(values)
    assert torch.all(out[1:] > out[:-1])
    assert torch.sign(out).tolist() == torch.sign(values).tolist()
    assert float(out.abs().max()) < float(values.abs().max())


def _transitions(rows=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.rand(FEATURE_DIM, generator=generator),
         torch.rand(512, generator=generator),
         float(index) * 0.01,
         index + 1)
        for index in range(rows)
    ]


@pytest.mark.parametrize("objective", ["rank", "huber"])
def test_train_step_reports_what_it_trained_on(objective):
    model = GainResidual(FEATURE_DIM, CPU, objective=objective, batch_size=8)
    metrics = model.train_step(_transitions())
    assert metrics["n_transitions"] == 16
    if objective == "rank":
        assert metrics["n_pairs"] > 0, "the count that reveals a degenerate batch"
    assert metrics["loss"] >= 0.0


@pytest.mark.parametrize("objective", ["rank", "huber"])
def test_a_step_actually_moves_the_parameters(objective):
    model = GainResidual(FEATURE_DIM, CPU, objective=objective, batch_size=8)
    before = model.residual_gate.detach().clone()
    for _ in range(5):
        model.train_step(_transitions())
    assert not torch.allclose(before, model.residual_gate.detach()), \
        "the gate must be able to open, or the residual can never contribute"


def test_all_tied_batch_does_not_crash_or_step():
    model = GainResidual(FEATURE_DIM, CPU, objective="rank", batch_size=4)
    tied = [(torch.rand(FEATURE_DIM), torch.rand(512), 0.0, i + 1) for i in range(8)]
    metrics = model.train_step(tied)
    assert metrics["n_pairs"] == 0
    assert metrics["loss"] == 0.0


def test_three_tuple_transitions_still_work():
    """The legacy capability path passes `(features, embedding, reward)` with no timestamp."""
    model = GainResidual(FEATURE_DIM, CPU, objective="rank", batch_size=4)
    legacy_shaped = [(f, e, r) for f, e, r, _step in _transitions(rows=8)]
    assert model.train_step(legacy_shaped)["n_transitions"] == 8


def test_huber_target_normalisation_tracks_the_target_scale():
    model = GainResidual(FEATURE_DIM, CPU, objective="huber", batch_size=8)
    start = float(model._target_mean)
    for _ in range(30):
        model.train_step([
            (torch.rand(FEATURE_DIM), torch.rand(512), 5.0, i + 1) for i in range(8)
        ])
    assert float(model._target_mean) != start, "the running mean must adapt"
