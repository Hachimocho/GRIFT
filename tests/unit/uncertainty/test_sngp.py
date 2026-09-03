"""SNGP head: precision accumulation, reset policy, and numerics.

Two bugs are covered here.

A3: ``reset_precision_matrix()`` existed but had no callers anywhere, so the
Laplace precision accumulated across every epoch of a run. ``gp_variance`` then
shrank monotonically for reasons unrelated to the data, making it incomparable
between epochs -- which is precisely what a benchmark needs it to be.

A4: the covariance was recomputed with ``torch.linalg.pinv`` on *every* forward
(the cache was invalidated by each accumulation step) and ran inside
``torch.cuda.amp.autocast``, so a 256x256 pseudo-inverse and an fp16 accumulation
into a registered buffer happened per batch.
"""

import pytest
import torch

from models.uncertainty.sngp import RandomFourierFeatures, SNGPBinaryHead


def make_head(in_features=8, rff_features=16, dropout=0.0, **kwargs):
    return SNGPBinaryHead(
        in_features=in_features,
        hidden_features=16,
        rff_features=rff_features,
        dropout=dropout,
        rff_seed=1234,
        **kwargs,
    )


def features(batch_size=5, in_features=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, in_features, generator=generator)


def test_output_keys_and_shapes():
    head = make_head()
    output = head(features(5))
    assert set(output) == {"logits", "probabilities", "gp_variance", "uncertainty"}
    assert output["logits"].shape == (5, 1)
    assert output["gp_variance"].shape == (5, 1)
    assert output["uncertainty"]["sngp_variance"].shape == (5, 1)


def test_random_fourier_features_are_bounded_and_seeded():
    first = RandomFourierFeatures(8, 16, seed=99)
    second = RandomFourierFeatures(8, 16, seed=99)
    third = RandomFourierFeatures(8, 16, seed=100)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)
    assert not torch.equal(first.weight, third.weight)

    output = first(features(4))
    # sqrt(2/D) * cos(...) is bounded by sqrt(2/D).
    assert output.abs().max().item() <= (2.0 / 16) ** 0.5 + 1e-6


def test_precision_starts_at_ridge_identity():
    head = make_head(rff_features=16, ridge_penalty=2.0)
    assert torch.equal(head.precision_matrix, 2.0 * torch.eye(16))


def test_update_precision_adds_exactly_gram_matrix():
    """Accumulation must add exactly phi^T phi for the batch's own features.

    The reference features are captured with a forward hook rather than recomputed:
    `head.hidden` is spectral-normalized, and spectral_norm refreshes its
    power-iteration estimate on every training-mode forward, so recomputing
    outside the real call would use different weights.
    """
    head = make_head(rff_features=16)
    head.train()
    baseline = head.precision_matrix.clone()

    captured = {}

    def capture(module, inputs, output):
        captured["random_features"] = output.detach().clone()

    handle = head.random_features.register_forward_hook(capture)
    try:
        head(features(7), update_precision=True)
    finally:
        handle.remove()

    observed = captured["random_features"].to(torch.float32)
    expected = baseline + observed.transpose(0, 1) @ observed
    assert torch.allclose(head.precision_matrix, expected, atol=1e-5)


def test_precision_matrix_stays_float32_under_half_input():
    """fp16 accumulation into the precision buffer is numerically unsound."""
    head = make_head(rff_features=16)
    head.train()
    head.half()
    head.precision_matrix.data = head.precision_matrix.data.float()

    head(features(4).half(), update_precision=True)
    assert head.precision_matrix.dtype == torch.float32
    assert torch.isfinite(head.precision_matrix).all()


def test_precision_matrix_is_symmetric_after_accumulation():
    head = make_head(rff_features=16)
    head.train()
    for seed in range(3):
        head(features(6, seed=seed), update_precision=True)
    matrix = head.precision_matrix
    assert torch.allclose(matrix, matrix.transpose(0, 1), atol=1e-4)


def test_reset_precision_matrix_restores_the_prior():
    head = make_head(rff_features=16, ridge_penalty=1.0)
    head.train()
    for seed in range(3):
        head(features(6, seed=seed), update_precision=True)
    assert not torch.allclose(head.precision_matrix, torch.eye(16))

    head.reset_precision_matrix()
    assert torch.equal(head.precision_matrix, torch.eye(16))
    assert head._cached_covariance is None, "the covariance cache must be dropped too"


def test_variance_shrinks_as_evidence_accumulates():
    """More observed data in a direction -> less predictive variance there."""
    head = make_head(rff_features=16)
    head.eval()
    probe = features(4, seed=42)

    with torch.no_grad():
        before = head(probe)["gp_variance"].mean().item()
    head.train()
    for seed in range(10):
        head(features(16, seed=seed), update_precision=True)
    head.eval()
    with torch.no_grad():
        after = head(probe)["gp_variance"].mean().item()

    assert after < before, f"variance did not shrink with evidence ({before} -> {after})"
    assert after > 0.0, "variance must stay strictly positive"


def test_covariance_is_factorized_at_most_once_per_forward(monkeypatch):
    """A4 regression: the inverse must not be recomputed on every forward.

    The original code invalidated its cache inside `_update_precision`, so a
    256x256 pinv ran once per training batch.
    """
    head = make_head(rff_features=16)
    head.eval()

    calls = {"count": 0}
    real_cholesky = torch.linalg.cholesky
    real_pinv = torch.linalg.pinv

    def counting_cholesky(*args, **kwargs):
        calls["count"] += 1
        return real_cholesky(*args, **kwargs)

    def counting_pinv(*args, **kwargs):
        calls["count"] += 1
        return real_pinv(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counting_cholesky)
    monkeypatch.setattr(torch.linalg, "pinv", counting_pinv)

    with torch.no_grad():
        head(features(4))
        head(features(4, seed=1))
        head(features(4, seed=2))
    assert calls["count"] == 1, (
        f"expected one factorization for three unchanged-precision forwards, got {calls['count']}"
    )


def test_accumulation_invalidates_the_cache_once(monkeypatch):
    head = make_head(rff_features=16)
    head.train()

    calls = {"count": 0}
    real_cholesky = torch.linalg.cholesky

    def counting_cholesky(*args, **kwargs):
        calls["count"] += 1
        return real_cholesky(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counting_cholesky)
    head(features(4), update_precision=True)
    head(features(4, seed=1), update_precision=True)
    assert calls["count"] == 2, "each accumulation should trigger exactly one refactorization"


def test_compute_variance_false_skips_factorization(monkeypatch):
    """Training batches that are not being summarized should skip the inverse."""
    head = make_head(rff_features=16)
    head.train()

    calls = {"count": 0}
    real_cholesky = torch.linalg.cholesky

    def counting_cholesky(*args, **kwargs):
        calls["count"] += 1
        return real_cholesky(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counting_cholesky)
    output = head(features(4), update_precision=True, compute_variance=False)
    assert calls["count"] == 0
    assert output["gp_variance"] is None
    assert "sngp_variance" not in output["uncertainty"]


def test_mean_field_logits_preserve_the_decision_boundary():
    """The mean-field correction is a positive rescaling, so 0.5 is preserved."""
    head = make_head(rff_features=16)
    head.eval()
    with torch.no_grad():
        output = head(features(16))
    gp_mean = output["logits"]
    probabilities = output["probabilities"]
    assert torch.equal(gp_mean > 0, probabilities > 0.5)


def test_variance_is_positive_and_finite():
    head = make_head(rff_features=16)
    head.eval()
    with torch.no_grad():
        variance = head(features(16))["gp_variance"]
    assert torch.isfinite(variance).all()
    assert (variance > 0).all()


@pytest.mark.parametrize("policy", ["per-epoch", "final-epoch", "never-reset"])
def test_precision_policy_is_recorded(policy):
    head = make_head(precision_policy=policy)
    assert head.precision_policy == policy


def test_invalid_precision_policy_rejected():
    with pytest.raises(ValueError, match="precision_policy"):
        make_head(precision_policy="whenever")


class TestEpochPolicies:
    """Behavior of `on_epoch_start` under each reset policy."""

    def _accumulate(self, head):
        head.train()
        head(features(6), update_precision=True)

    def test_per_epoch_resets_every_epoch(self):
        head = make_head(precision_policy="per-epoch")
        for epoch in range(3):
            head.on_epoch_start(epoch, num_epochs=3)
            assert torch.equal(head.precision_matrix, torch.eye(16)), (
                f"epoch {epoch} did not start from the prior"
            )
            assert head.should_accumulate_precision is True
            self._accumulate(head)

    def test_final_epoch_only_accumulates_last(self):
        head = make_head(precision_policy="final-epoch")
        head.on_epoch_start(0, num_epochs=3)
        assert head.should_accumulate_precision is False
        head.on_epoch_start(2, num_epochs=3)
        assert head.should_accumulate_precision is True
        assert torch.equal(head.precision_matrix, torch.eye(16))

    def test_never_reset_preserves_accumulation(self):
        head = make_head(precision_policy="never-reset")
        head.on_epoch_start(0, num_epochs=2)
        self._accumulate(head)
        accumulated = head.precision_matrix.clone()
        head.on_epoch_start(1, num_epochs=2)
        assert torch.equal(head.precision_matrix, accumulated), (
            "never-reset must reproduce the original (unreset) behavior"
        )
