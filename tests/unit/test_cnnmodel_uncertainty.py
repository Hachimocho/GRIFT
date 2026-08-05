"""CNNModel's uncertainty integration: head grafting, loss dispatch, checkpoints.

The first test here is the one that matters most. A single parametrized sweep over
(uncertainty head) x (MC dropout on/off), asserting only that forward + loss +
backward complete, would have caught the evidential/MC-dropout crash on the day
it was written -- instead of it surfacing as "accuracy 0.0" through a swallowed
exception in evaluate_model.
"""

import pytest
import torch
import torch.nn as nn

from models.uncertainty import PredictionBundle

HEADS = ("none", "evidential", "batchensemble", "sngp")


# --------------------------------------------------------------------------- #
# The regression matrix
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("mc_samples", [0, 3])
def test_forward_loss_backward_for_every_head_and_mc_setting(
    cnn_model_factory, tiny_batch, tiny_labels, head, mc_samples
):
    """Every (head, mc_dropout) combination must train without raising.

    The evidential + MC-dropout cell fails before the fix: mc_dropout_predict
    rebuilds the PredictionBundle and drops `alpha`, so the evidential loss
    raises ValueError("Evidential loss requires alpha values...").
    """
    model = cnn_model_factory(
        uncertainty_head=head,
        mc_dropout_samples=mc_samples,
        uncertainty_dropout_rate=0.3,
    )
    model.train()
    images = tiny_batch(batch_size=4, size=16)
    labels = tiny_labels(4)

    bundle = model.forward_with_uncertainty(
        images, update_precision=True, use_mc_dropout=mc_samples > 1
    )
    assert isinstance(bundle, PredictionBundle)
    assert bundle.logits.shape == (4, 1)
    assert bundle.probabilities.shape == (4, 1)
    assert torch.isfinite(bundle.logits).all(), "non-finite logits"
    assert ((bundle.probabilities >= 0) & (bundle.probabilities <= 1)).all()

    loss = model.compute_loss(bundle, labels)
    assert loss.ndim == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()

    grads = [
        param.grad for param in model._parameters_for_optimization()
        if param.requires_grad and param.grad is not None
    ]
    assert grads, "no parameter received a gradient"
    assert any(grad.abs().sum() > 0 for grad in grads), "all gradients are exactly zero"


@pytest.mark.parametrize("head", HEADS)
def test_mc_dropout_preserves_head_specific_bundle_fields(cnn_model_factory, tiny_batch, head):
    """MC dropout must not drop the fields the loss dispatch depends on.

    `alpha` for evidential, `member_logits` for BatchEnsemble, `gp_variance` for
    SNGP. Rebuilding the bundle from scratch silently discarded all three.
    """
    model = cnn_model_factory(
        uncertainty_head=head, mc_dropout_samples=3, uncertainty_dropout_rate=0.3
    )
    model.eval()
    images = tiny_batch(batch_size=4, size=16)

    single = model.forward_with_uncertainty(images, use_mc_dropout=False)
    averaged = model.forward_with_uncertainty(images, use_mc_dropout=True)

    for field in ("alpha", "evidence", "member_logits", "gp_variance"):
        single_value = getattr(single, field)
        averaged_value = getattr(averaged, field)
        if single_value is None:
            continue
        assert averaged_value is not None, (
            f"{head}: MC-dropout bundle lost `{field}`, which compute_loss needs"
        )
        assert averaged_value.shape == single_value.shape


def test_evidential_with_mc_dropout_computes_a_loss(cnn_model_factory, tiny_batch, tiny_labels):
    """The specific combination that used to raise, then be swallowed."""
    model = cnn_model_factory(
        uncertainty_head="evidential", mc_dropout_samples=5, uncertainty_dropout_rate=0.4
    )
    model.train()
    bundle = model.forward_with_uncertainty(
        tiny_batch(batch_size=4, size=16), use_mc_dropout=True
    )
    assert bundle.alpha is not None
    assert (bundle.alpha >= 1.0).all(), "Dirichlet concentration must stay >= 1"
    loss = model.compute_loss(bundle, tiny_labels(4))
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Head grafting machinery
# --------------------------------------------------------------------------- #

def test_find_last_linear_on_flat_sequential(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")
    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    path, linear = model._find_last_linear(module)
    assert path == "2"
    assert linear.in_features == 8


def test_find_last_linear_on_nested_module(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(4, 8)
            self.blocks = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            self.head = nn.Linear(16, 1)

    path, linear = model._find_last_linear(Nested())
    assert path == "head"
    assert linear.in_features == 16


def test_find_last_linear_returns_none_when_absent(cnn_model_factory):
    """The squeezenetdf case: a convolutional classifier with zero nn.Linear."""
    model = cnn_model_factory(uncertainty_head="none")
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Conv2d(4, 1, 1))
    path, linear = model._find_last_linear(module)
    assert path is None and linear is None


@pytest.mark.parametrize("head", ["evidential", "batchensemble", "sngp"])
def test_graft_requires_a_linear_layer(tiny_detector_no_linear, head):
    """A real head on a Linear-free backbone must fail with a clear message."""
    from models.CNNModel import CNNModel

    with pytest.raises(ValueError, match="final linear layer"):
        CNNModel(
            save_path="x.pth", model_name=tiny_detector_no_linear, lr=1e-3, amsgrad=True,
            device=torch.device("cpu"), uncertainty_head=head,
        )


def test_no_head_on_linear_free_backbone_yields_no_features(tiny_detector_no_linear, tiny_batch):
    """Pins the silent case: head='none' plus no Linear means features stay None.

    `_find_last_linear` returns (None, None), so no forward pre-hook is
    registered and `bundle.features` is never populated. Anything downstream that
    reads features must therefore handle None -- which is exactly the shape of a
    future silent failure, so it is pinned here.
    """
    from models.CNNModel import CNNModel

    model = CNNModel(
        save_path="x.pth", model_name=tiny_detector_no_linear, lr=1e-3, amsgrad=True,
        device=torch.device("cpu"), uncertainty_head="none",
    )
    model.eval()
    bundle = model.forward_with_uncertainty(tiny_batch(batch_size=2, size=16))
    assert bundle.features is None
    assert bundle.logits.shape == (2, 1)


@pytest.mark.parametrize("head", ["evidential", "batchensemble", "sngp"])
def test_graft_replaces_the_original_linear_with_identity(cnn_model_factory, head):
    model = cnn_model_factory(uncertainty_head=head)
    parent = model.model.model
    *prefix, leaf = model.final_linear_path.split(".")
    for part in prefix:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    replaced = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
    assert isinstance(replaced, nn.Identity)
    assert model.output_head is not None


def test_replace_module_handles_digit_paths(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")
    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    model._replace_module(module, "2", nn.Identity())
    assert isinstance(module[2], nn.Identity)


def test_build_uncertainty_head_rejects_unknown(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")
    model.uncertainty_head_type = "does_not_exist"
    with pytest.raises(ValueError, match="Unsupported uncertainty head"):
        model._build_uncertainty_head(16)


def test_optimizer_covers_backbone_and_head(cnn_model_factory):
    plain = cnn_model_factory(uncertainty_head="none")
    grafted = cnn_model_factory(uncertainty_head="sngp")

    plain_count = sum(param.numel() for param in plain._parameters_for_optimization())
    grafted_count = sum(param.numel() for param in grafted._parameters_for_optimization())
    assert grafted_count > plain_count, "grafted head parameters are missing from the optimizer"

    optimizer_params = {
        id(param) for group in grafted.optim.param_groups for param in group["params"]
    }
    for param in grafted.output_head.parameters():
        assert id(param) in optimizer_params, "an uncertainty-head parameter is not being optimized"


def test_penultimate_feature_hook_does_not_leak_between_calls(cnn_model_factory, tiny_batch):
    model = cnn_model_factory(uncertainty_head="none")
    model.eval()
    first = model.forward_with_uncertainty(tiny_batch(batch_size=2, size=16))
    assert first.features is not None
    second = model.forward_with_uncertainty(tiny_batch(batch_size=3, size=16, seed=1))
    assert second.features.shape[0] == 3, "stale features leaked from the previous forward"


def test_dropout_controller_covers_backbone_and_head(cnn_model_factory):
    from models.uncertainty.mc_dropout import DROPOUT_TYPES

    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.25)
    sites = [m for m in model.dropout_controller.modules() if isinstance(m, DROPOUT_TYPES)]
    assert len(sites) >= 2, (
        "the dropout controller must see both backbone and head dropouts, or MC dropout "
        "silently samples only part of the network"
    )


def test_train_eval_modes_propagate_to_head(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="sngp")
    model.train()
    assert model.current_mode == "train"
    assert model.model.model.training and model.output_head.training
    model.eval()
    assert model.current_mode == "eval"
    assert not model.model.model.training and not model.output_head.training


# --------------------------------------------------------------------------- #
# transform()
# --------------------------------------------------------------------------- #

def test_transform_is_deterministic_in_eval_mode(cnn_model_factory):
    """Precondition for deterministic evaluation.

    evaluate_model loads images in worker threads and calls model.transform there.
    In train mode that path applies RandomHorizontalFlip / RandomRotation /
    ColorJitter / RandomAffine / RandomErasing, all drawing on the global torch
    RNG from several threads at once. In eval mode the transforms must be
    RNG-free, so two calls are bit-identical.
    """
    import numpy as np

    model = cnn_model_factory(uncertainty_head="none")
    model.eval()
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert torch.equal(model.transform(image), model.transform(image))


def test_transform_varies_in_train_mode(cnn_model_factory):
    import numpy as np

    model = cnn_model_factory(uncertainty_head="none")
    model.train()
    rng = np.random.Generator(np.random.PCG64(0))
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    outputs = [model.transform(image) for _ in range(6)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:]), (
        "train-mode augmentation should introduce variation"
    )


# --------------------------------------------------------------------------- #
# Loss dispatch
# --------------------------------------------------------------------------- #

def test_batchensemble_loss_matches_manual_per_member_bce(cnn_model_factory, tiny_batch, tiny_labels):
    """Pins the member/label alignment in the reshape.

    compute_loss does member_logits.view(-1, 1) on a [B, M, 1] tensor against
    labels.repeat(1, M).view(-1, 1). Both are sample-major so it is correct, but
    a silent transposition here would be invisible in the loss value alone.
    """
    model = cnn_model_factory(uncertainty_head="batchensemble", batchensemble_members=3)
    model.eval()
    bundle = model.forward_with_uncertainty(tiny_batch(batch_size=4, size=16))
    labels = tiny_labels(4)

    actual = model.compute_loss(bundle, labels)

    criterion = torch.nn.BCEWithLogitsLoss()
    members = bundle.member_logits  # [B, M, 1]
    manual = torch.stack([
        criterion(members[:, member_index, :], labels)
        for member_index in range(members.shape[1])
    ]).mean()
    assert torch.allclose(actual, manual, atol=1e-6), f"{actual.item()} != {manual.item()}"


def test_compute_loss_accepts_raw_logits(cnn_model_factory, tiny_labels):
    model = cnn_model_factory(uncertainty_head="none")
    logits = torch.zeros(4, 1, requires_grad=True)
    loss = model.compute_loss(logits, tiny_labels(4))
    assert torch.isfinite(loss)


def test_summarize_uncertainty_returns_floats(cnn_model_factory, tiny_batch):
    model = cnn_model_factory(uncertainty_head="sngp")
    model.eval()
    bundle = model.forward_with_uncertainty(tiny_batch(batch_size=3, size=16))
    summary = model.summarize_uncertainty(bundle)
    assert summary, "SNGP must report at least one uncertainty key"
    for name, value in summary.items():
        assert isinstance(value, float), f"{name} is {type(value).__name__}, expected float"


def test_summarize_uncertainty_on_empty_bundle(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")
    bundle = PredictionBundle(logits=torch.zeros(2, 1), probabilities=torch.full((2, 1), 0.5))
    assert model.summarize_uncertainty(bundle) == {}


# --------------------------------------------------------------------------- #
# Dead code
# --------------------------------------------------------------------------- #

def test_process_node_data_is_gone():
    """The legacy path referenced undefined `self.models` and `self.accuracy`.

    It also called `self.model(...)` directly, which with a grafted head returns
    raw features rather than logits.
    """
    from models.CNNModel import CNNModel
    assert not hasattr(CNNModel, "process_node_data")
