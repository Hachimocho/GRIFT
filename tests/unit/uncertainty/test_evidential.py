"""Evidential head and its Dirichlet KL-annealed loss."""

import pytest
import torch

from models.uncertainty.evidential import (
    BinaryEvidentialHead, EvidentialBinaryClassificationLoss, _dirichlet_kl_divergence,
)
from models.uncertainty.types import PredictionBundle


def make_head(in_features=8, dropout=0.0):
    return BinaryEvidentialHead(in_features=in_features, hidden_features=16, dropout=dropout)


def features(batch_size=5, in_features=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, in_features, generator=generator)


def test_output_keys_and_shapes():
    output = make_head()(features(5))
    assert set(output) == {"logits", "probabilities", "evidence", "alpha", "uncertainty"}
    assert output["logits"].shape == (5, 1)
    assert output["probabilities"].shape == (5, 1)
    assert output["evidence"].shape == (5, 2)
    assert output["alpha"].shape == (5, 2)
    assert output["uncertainty"]["evidential_vacuity"].shape == (5, 1)
    assert output["uncertainty"]["evidential_total_evidence"].shape == (5, 1)


def test_alpha_is_at_least_one():
    """Dirichlet concentration is evidence + 1, so it can never drop below 1."""
    output = make_head()(features(32))
    assert (output["alpha"] >= 1.0).all()


def test_evidence_is_non_negative():
    """Softplus output; negative evidence would be meaningless."""
    assert (make_head()(features(32))["evidence"] >= 0).all()


def test_alpha_sum_is_evidence_plus_two():
    output = make_head()(features(8))
    assert torch.allclose(
        output["alpha"].sum(dim=1, keepdim=True),
        output["evidence"].sum(dim=1, keepdim=True) + 2.0,
        atol=1e-6,
    )


def test_vacuity_equals_two_over_alpha_sum():
    output = make_head()(features(8))
    expected = 2.0 / output["alpha"].sum(dim=1, keepdim=True)
    assert torch.allclose(output["uncertainty"]["evidential_vacuity"], expected, atol=1e-6)


def test_vacuity_is_in_the_unit_interval():
    """With alpha >= 1 in both classes, alpha_sum >= 2, so vacuity lands in (0, 1]."""
    vacuity = make_head()(features(64))["uncertainty"]["evidential_vacuity"]
    assert (vacuity > 0).all() and (vacuity <= 1.0 + 1e-6).all()


def test_logits_and_probabilities_are_consistent():
    output = make_head()(features(8))
    assert torch.allclose(torch.sigmoid(output["logits"]), output["probabilities"], atol=1e-5)


def test_more_evidence_means_less_vacuity():
    """Vacuity must be a decreasing function of total evidence."""
    head = make_head()
    output = head(features(64))
    total_evidence = output["uncertainty"]["evidential_total_evidence"].squeeze()
    vacuity = output["uncertainty"]["evidential_vacuity"].squeeze()
    order = torch.argsort(total_evidence)
    sorted_vacuity = vacuity[order]
    assert torch.all(sorted_vacuity[:-1] >= sorted_vacuity[1:] - 1e-6)


# --------------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------------- #

def test_kl_of_uniform_dirichlet_is_zero():
    ones = torch.ones(4, 2)
    assert torch.allclose(_dirichlet_kl_divergence(ones), torch.zeros(4, 1), atol=1e-5)


def test_kl_is_non_negative_and_correctly_shaped():
    alpha = torch.tensor([[1.0, 1.0], [3.0, 1.0], [10.0, 2.0], [1.0, 7.0]])
    kl = _dirichlet_kl_divergence(alpha)
    assert kl.shape == (4, 1)
    assert (kl >= -1e-6).all()


def test_kl_grows_with_concentration():
    near_uniform = _dirichlet_kl_divergence(torch.tensor([[1.5, 1.0]]))
    concentrated = _dirichlet_kl_divergence(torch.tensor([[20.0, 1.0]]))
    assert concentrated.item() > near_uniform.item()


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def make_bundle(alpha):
    probabilities = (alpha[:, 1:2] / alpha.sum(dim=1, keepdim=True)).clamp(1e-6, 1 - 1e-6)
    return PredictionBundle(
        logits=torch.logit(probabilities), probabilities=probabilities, alpha=alpha,
    ).with_predictions()


def test_loss_requires_alpha():
    """The contract that MC dropout used to violate by dropping the field."""
    loss_fn = EvidentialBinaryClassificationLoss(annealing_steps=10)
    bundle = PredictionBundle(logits=torch.zeros(2, 1), probabilities=torch.full((2, 1), 0.5))
    with pytest.raises(ValueError, match="alpha"):
        loss_fn(bundle, torch.zeros(2, 1))


def test_loss_is_finite_and_scalar():
    loss_fn = EvidentialBinaryClassificationLoss(annealing_steps=10)
    alpha = torch.tensor([[2.0, 5.0], [7.0, 1.5]])
    loss = loss_fn(make_bundle(alpha), torch.tensor([[1.0], [0.0]]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_prefers_correct_evidence():
    loss_fn = EvidentialBinaryClassificationLoss(annealing_steps=10_000)
    labels = torch.tensor([[1.0]])
    aligned = loss_fn(make_bundle(torch.tensor([[1.0, 20.0]])), labels)
    opposed = loss_fn(make_bundle(torch.tensor([[20.0, 1.0]])), labels)
    assert aligned.item() < opposed.item()


def test_annealing_weight_ramps_then_saturates():
    """KL weight is step/annealing_steps, clamped at 1."""
    loss_fn = EvidentialBinaryClassificationLoss(annealing_steps=4)
    assert loss_fn.global_step.item() == 0

    alpha = torch.tensor([[3.0, 3.0]])
    labels = torch.tensor([[1.0]])
    for _ in range(10):
        loss_fn(make_bundle(alpha), labels)

    assert loss_fn.global_step.item() == 10
    # Two further calls past saturation must give the same loss.
    first = loss_fn(make_bundle(alpha), labels)
    second = loss_fn(make_bundle(alpha), labels)
    assert torch.allclose(first, second, atol=1e-6)


def test_annealing_steps_floor_at_one():
    assert EvidentialBinaryClassificationLoss(annealing_steps=0).annealing_steps == 1


def test_global_step_is_a_registered_buffer():
    """It must live in state_dict, or the annealing schedule resets on resume."""
    loss_fn = EvidentialBinaryClassificationLoss(annealing_steps=100)
    assert "global_step" in loss_fn.state_dict()


def test_global_step_survives_a_state_dict_roundtrip():
    original = EvidentialBinaryClassificationLoss(annealing_steps=100)
    alpha = torch.tensor([[3.0, 3.0]])
    for _ in range(7):
        original(make_bundle(alpha), torch.tensor([[1.0]]))
    assert original.global_step.item() == 7

    restored = EvidentialBinaryClassificationLoss(annealing_steps=100)
    restored.load_state_dict(original.state_dict())
    assert restored.global_step.item() == 7, (
        "the KL annealing counter must be checkpointed, or a resumed run silently "
        "restarts the schedule from zero"
    )
