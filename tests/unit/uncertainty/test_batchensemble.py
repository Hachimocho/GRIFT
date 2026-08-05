"""BatchEnsemble head.

The load-bearing test is `test_eval_variance_is_nonzero`. With the original
`torch.ones` fast-weight initialization every member computed an identical
function, so at eval -- dropout off -- the reported ensemble variance was
*exactly* 0.0 for every input. The method looked useless, and nothing in the
pipeline flagged it.
"""

import pytest
import torch

from models.uncertainty.batchensemble import BatchEnsembleBinaryHead


def make_head(in_features=8, ensemble_size=4, dropout=0.0, init_seed=1234):
    return BatchEnsembleBinaryHead(
        in_features=in_features,
        ensemble_size=ensemble_size,
        hidden_features=16,
        dropout=dropout,
        init_seed=init_seed,
    )


def features(batch_size=5, in_features=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, in_features, generator=generator)


def test_output_keys_and_shapes():
    head = make_head()
    output = head(features(5))

    assert set(output) == {"logits", "probabilities", "member_logits", "uncertainty"}
    assert output["logits"].shape == (5, 1)
    assert output["probabilities"].shape == (5, 1)
    assert output["member_logits"].shape == (5, 4, 1)
    assert output["uncertainty"]["batchensemble_variance"].shape == (5, 1)


def test_eval_variance_is_nonzero():
    """Members must disagree with dropout disabled.

    Canonical BatchEnsemble initializes the rank-1 fast weights to random +/-1
    sign vectors precisely to break this symmetry. Initializing them to ones
    makes every member identical, and identical members have zero variance.
    """
    head = make_head(dropout=0.0)
    head.eval()
    with torch.no_grad():
        output = head(features(16))

    variance = output["uncertainty"]["batchensemble_variance"]
    assert torch.isfinite(variance).all()
    assert variance.max().item() > 0.0, (
        "ensemble variance is identically zero -- the members are not diverse"
    )


def test_fast_weights_are_sign_vectors():
    head = make_head()
    for name in ("input_fast_weights", "hidden_fast_weights"):
        weights = getattr(head, name)
        unique = torch.unique(weights)
        assert set(unique.tolist()) <= {-1.0, 1.0}, f"{name} must hold only +/-1, got {unique}"
        # Rows must not all be the same, or the symmetry is unbroken anyway.
        assert not all(
            torch.equal(weights[0], weights[index]) for index in range(1, weights.shape[0])
        ), f"{name} rows are all identical"


def test_member_bias_starts_at_zero():
    """Sign vectors alone break the symmetry; the bias stays at the canonical zero."""
    assert torch.equal(make_head().member_bias, torch.zeros(4, 1))


def test_init_seed_controls_reproducibility():
    same_a = make_head(init_seed=7)
    same_b = make_head(init_seed=7)
    different = make_head(init_seed=8)

    assert torch.equal(same_a.input_fast_weights, same_b.input_fast_weights)
    assert not torch.equal(same_a.input_fast_weights, different.input_fast_weights)


def test_init_without_seed_still_produces_sign_vectors():
    head = BatchEnsembleBinaryHead(in_features=8, ensemble_size=4, hidden_features=16)
    assert set(torch.unique(head.input_fast_weights).tolist()) <= {-1.0, 1.0}


def test_single_member_degenerates_to_zero_variance():
    """With one member there is nothing to disagree with; variance must be 0, not NaN."""
    head = make_head(ensemble_size=1, dropout=0.0)
    head.eval()
    with torch.no_grad():
        output = head(features(4))
    variance = output["uncertainty"]["batchensemble_variance"]
    assert torch.equal(variance, torch.zeros_like(variance))


def test_probabilities_are_the_member_mean():
    head = make_head(dropout=0.0)
    head.eval()
    with torch.no_grad():
        output = head(features(6))
    expected = torch.sigmoid(output["member_logits"]).mean(dim=1)
    assert torch.allclose(output["probabilities"], expected, atol=1e-6)


def test_logits_are_consistent_with_probabilities():
    head = make_head(dropout=0.0)
    head.eval()
    with torch.no_grad():
        output = head(features(6))
    assert torch.allclose(
        torch.sigmoid(output["logits"]), output["probabilities"], atol=1e-5
    )


def test_gradients_reach_every_parameter():
    head = make_head(dropout=0.0)
    head(features(4))["logits"].sum().backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


@pytest.mark.parametrize("ensemble_size", [2, 3, 8])
def test_variance_scales_with_member_count(ensemble_size):
    head = make_head(ensemble_size=ensemble_size, dropout=0.0)
    head.eval()
    with torch.no_grad():
        output = head(features(8))
    assert output["member_logits"].shape[1] == ensemble_size
    assert output["uncertainty"]["batchensemble_variance"].max().item() > 0.0
