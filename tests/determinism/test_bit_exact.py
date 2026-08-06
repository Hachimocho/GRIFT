"""Bit-exact reproducibility, on both CPU and GPU.

One shared test body parametrized over device, so the CPU and GPU guarantees cannot
drift apart. The CUDA variants are marked `gpu` and require `--run-gpu`.

Everything here compares for exact equality. A tolerance-based check cannot tell
"reproducible" from "close enough today", which is the whole point.

For the GPU side to pass, strict mode has to force several things beyond seeding:
deterministic algorithms, cuDNN determinism with autotuning off, and -- most easily
missed -- **TF32 off**. On Ada (L40S) TF32 is enabled by default for cuDNN
convolutions, and TF32 is not fp32, so a run that merely sets seeds will produce
subtly different numbers.
"""

import pytest
import torch

from tests.helpers.determinism import state_dict_hash

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
]
HEADS = ("none", "evidential", "batchensemble", "sngp")


def build_model(tiny_detector_name, device, head, seed):
    """Fresh model on ``device`` under a fresh strict-mode seeding."""
    from models.CNNModel import CNNModel
    from test_helpers.determinism import configure_determinism

    configure_determinism(seed=seed, mode="strict", allow_multi_gpu=True)
    model = CNNModel(
        save_path="bitexact.pth", model_name=tiny_detector_name, lr=1e-2, amsgrad=True,
        device=torch.device(device), uncertainty_head=head,
        uncertainty_dropout_rate=0.2,
    )
    model.model.model.to(device)
    if model.output_head is not None:
        model.output_head.to(device)
    return model


def train_and_evaluate(model, device, steps=10, batch_size=8):
    """Deterministic short training run; returns per-step losses and final probs."""
    generator = torch.Generator().manual_seed(1234)
    images = torch.rand(batch_size * steps, 3, 16, 16, generator=generator).to(device)
    labels = (
        torch.arange(batch_size * steps).reshape(-1, 1) % 2
    ).float().to(device)

    model.train()
    losses = []
    for step in range(steps):
        start = step * batch_size
        batch = images[start:start + batch_size]
        target = labels[start:start + batch_size]
        bundle = model.forward_with_uncertainty(batch, update_precision=True)
        loss = model.compute_loss(bundle, target)
        model.optim.zero_grad()
        loss.backward()
        model.optim.step()
        losses.append(loss.detach().cpu().clone())

    model.eval()
    with torch.no_grad():
        final = model.forward_with_uncertainty(images[:batch_size])
    return losses, final


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("head", HEADS)
def test_training_is_bit_exact(tiny_detector, device, head, assert_bit_exact):
    """Two identically-seeded runs must agree exactly: losses, weights, and outputs."""
    first_model = build_model(tiny_detector, device, head, seed=17)
    first_losses, first_final = train_and_evaluate(first_model, device)

    second_model = build_model(tiny_detector, device, head, seed=17)
    second_losses, second_final = train_and_evaluate(second_model, device)

    assert_bit_exact(first_losses, second_losses, path="losses")
    assert_bit_exact(first_final.logits.cpu(), second_final.logits.cpu(), path="logits")
    assert_bit_exact(
        first_final.probabilities.cpu(), second_final.probabilities.cpu(), path="probs"
    )
    assert state_dict_hash(first_model.model.model.state_dict()) == state_dict_hash(
        second_model.model.model.state_dict()
    ), f"{head} on {device}: final backbone weights differ"
    if first_model.output_head is not None:
        assert state_dict_hash(first_model.output_head.state_dict()) == state_dict_hash(
            second_model.output_head.state_dict()
        ), f"{head} on {device}: final head weights differ"


@pytest.mark.parametrize("device", DEVICES)
def test_different_seeds_do_diverge(tiny_detector, device):
    """Sanity check: the bit-exactness above must not be trivially true.

    If two different seeds produced identical results, the tests above would be
    passing for the wrong reason -- the seed would be having no effect at all.
    """
    first_model = build_model(tiny_detector, device, "sngp", seed=1)
    first_losses, _ = train_and_evaluate(first_model, device, steps=5)
    second_model = build_model(tiny_detector, device, "sngp", seed=2)
    second_losses, _ = train_and_evaluate(second_model, device, steps=5)

    assert not all(
        torch.equal(left, right) for left, right in zip(first_losses, second_losses)
    ), "different seeds produced identical training, so the seed is not being used"


@pytest.mark.parametrize("device", DEVICES)
def test_uncertainty_scores_are_bit_exact(tiny_detector, device, assert_bit_exact):
    """Uncertainty values, not just predictions, must reproduce exactly."""
    first_model = build_model(tiny_detector, device, "sngp", seed=23)
    _, first_final = train_and_evaluate(first_model, device, steps=6)
    second_model = build_model(tiny_detector, device, "sngp", seed=23)
    _, second_final = train_and_evaluate(second_model, device, steps=6)

    assert set(first_final.uncertainty) == set(second_final.uncertainty)
    for name in first_final.uncertainty:
        assert_bit_exact(
            first_final.uncertainty[name].cpu(),
            second_final.uncertainty[name].cpu(),
            path=f"uncertainty[{name}]",
        )


@pytest.mark.parametrize("device", DEVICES)
def test_mc_dropout_is_bit_exact(tiny_detector, device, assert_bit_exact):
    """Stochastic sampling must still reproduce under a fixed seed."""
    def sample(seed):
        model = build_model(tiny_detector, device, "none", seed=seed)
        model.mc_dropout_samples = 8
        if not model.mc_dropout_available():
            pytest.skip("no stochastic dropout available")
        model.eval()
        generator = torch.Generator().manual_seed(99)
        images = torch.rand(6, 3, 16, 16, generator=generator).to(device)
        with torch.no_grad():
            return model.forward_with_uncertainty(images, use_mc_dropout=True)

    first, second = sample(41), sample(41)
    assert_bit_exact(first.probabilities.cpu(), second.probabilities.cpu(), path="probs")
    assert_bit_exact(
        first.uncertainty["mc_dropout_variance"].cpu(),
        second.uncertainty["mc_dropout_variance"].cpu(),
        path="mc_variance",
    )


@pytest.mark.gpu
def test_strict_mode_disables_tf32_on_device():
    """TF32 is the usual reason a GPU bit-exactness check fails while CPU passes.

    It is on by default for cuDNN convolutions on Ada, and TF32 is not fp32.
    """
    from test_helpers.determinism import assert_strict_invariants, configure_determinism

    configure_determinism(seed=0, mode="strict", allow_multi_gpu=True)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert_strict_invariants("gpu tf32 check")


@pytest.mark.gpu
def test_cpu_and_cuda_agree_on_a_single_forward_pass(tiny_detector):
    """Cross-device agreement is only well-defined before differences compound.

    Compared on one forward pass with *identical* weights, CPU and CUDA should agree
    to roughly float32 precision -- they run different kernels, so not bit-for-bit,
    but nothing should move the answer materially.

    Deliberately not asserted after training: each optimizer step feeds the previous
    step's rounding differences back in, and AdamW's per-parameter normalization
    amplifies them, so post-training divergence grows without any principled bound
    (measured at ~3e-2 in probability space after only four steps). Bit-exactness is
    a within-device property; the tests above are what pin it.
    """
    cpu_model = build_model(tiny_detector, "cpu", "none", seed=5)
    cuda_model = build_model(tiny_detector, "cuda", "none", seed=5)
    # Force identical weights, so the only difference is the kernels.
    cuda_model.model.model.load_state_dict(cpu_model.model.model.state_dict())
    cuda_model.model.model.to("cuda")

    generator = torch.Generator().manual_seed(77)
    images = torch.rand(8, 3, 16, 16, generator=generator)

    cpu_model.eval()
    cuda_model.eval()
    with torch.no_grad():
        cpu_out = cpu_model.forward_with_uncertainty(images)
        cuda_out = cuda_model.forward_with_uncertainty(images.cuda())

    difference = (cpu_out.probabilities - cuda_out.probabilities.cpu()).abs().max()
    assert difference < 1e-5, (
        f"CPU and CUDA disagree by {difference:.2e} on a single forward pass with "
        f"identical weights, which is more than kernel differences explain"
    )


@pytest.mark.gpu
def test_sngp_precision_stays_float32_under_autocast(tiny_detector):
    """fp16 accumulation into the precision buffer would be numerically unsound."""
    model = build_model(tiny_detector, "cuda", "sngp", seed=3)
    model.train()
    images = torch.rand(8, 3, 16, 16).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model.forward_with_uncertainty(images, update_precision=True)

    assert model.output_head.precision_matrix.dtype == torch.float32
    assert torch.isfinite(model.output_head.precision_matrix).all()

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        bundle = model.forward_with_uncertainty(images)
    assert torch.isfinite(bundle.uncertainty["sngp_variance"]).all()
    assert (bundle.uncertainty["sngp_variance"] > 0).all()
