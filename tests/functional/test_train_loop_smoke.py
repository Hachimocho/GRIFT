"""Short real training runs, one per uncertainty head.

These assert that training *functions* -- gradients flow, the loss goes down on a
memorizable set, checkpoints round-trip, uncertainty keys appear and are finite --
and that it is reproducible. They deliberately assert nothing about accuracy,
calibration, or fairness: quality is the benchmark's job, not this suite's.

Marked `slow` because they run real forward/backward passes over real image files.
They still use the tiny synthetic detector, so no dataset and no download.
"""

import pytest
import torch

from tests.helpers.determinism import state_dict_hash

HEADS = ("none", "evidential", "batchensemble", "sngp")

pytestmark = pytest.mark.slow


def train_steps(model, image_nodes, steps=30, batch_size=8, collect_losses=True):
    """Run a handful of optimizer steps over the given nodes. Returns the losses."""
    model.train()
    losses = []
    for step in range(steps):
        start = (step * batch_size) % max(1, len(image_nodes) - batch_size + 1)
        batch_nodes = image_nodes[start:start + batch_size]
        if not batch_nodes:
            continue

        images = torch.stack([
            model.transform(node.get_data().load_data()) for node in batch_nodes
        ])
        labels = torch.tensor(
            [[float(node.get_label())] for node in batch_nodes], dtype=torch.float
        )

        bundle = model.forward_with_uncertainty(
            images, nodes=batch_nodes, update_precision=True
        )
        loss = model.compute_loss(bundle, labels)
        model.optim.zero_grad()
        loss.backward()
        model.optim.step()
        if collect_losses:
            losses.append(float(loss.detach()))
    return losses


@pytest.mark.parametrize("head", HEADS)
def test_training_reduces_loss_on_a_memorizable_set(cnn_model_factory, image_nodes, head):
    """A tiny fixed set must be learnable; if the loss does not fall, nothing works.

    Uses a deliberately generous lr and step count. BatchEnsemble needs it: its loss
    is the mean per-member BCE, and because members modulate the *shared* weights
    with +/-1 sign vectors their gradients partially cancel there, so the effective
    learning rate on the shared layers is well below the nominal one. At lr=1e-3 over
    40 steps it sits flat at ln(2) -- not broken, just far from converged.
    """
    model = cnn_model_factory(
        uncertainty_head=head, uncertainty_dropout_rate=0.1, lr=1e-2
    )
    losses = train_steps(model, image_nodes, steps=120, batch_size=4)

    assert losses, "no optimizer steps ran"
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early, f"{head}: loss did not decrease ({early:.4f} -> {late:.4f})"


@pytest.mark.parametrize("head", HEADS)
def test_gradients_reach_backbone_and_head(cnn_model_factory, image_nodes, head):
    """Both halves of the model must actually train.

    Catches a graft that leaves the head detached from the loss, and a backbone
    inadvertently frozen.
    """
    model = cnn_model_factory(
        uncertainty_head=head, uncertainty_dropout_rate=0.1, finetune=False
    )
    model.train()

    images = torch.stack([
        model.transform(node.get_data().load_data()) for node in image_nodes[:4]
    ])
    labels = torch.tensor(
        [[float(node.get_label())] for node in image_nodes[:4]], dtype=torch.float
    )
    model.compute_loss(
        model.forward_with_uncertainty(images, update_precision=True), labels
    ).backward()

    backbone_grads = [
        param.grad for param in model.model.model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert backbone_grads, f"{head}: no backbone gradients"
    assert any(grad.abs().sum() > 0 for grad in backbone_grads), (
        f"{head}: all backbone gradients are exactly zero"
    )

    if model.output_head is not None:
        head_grads = [
            param.grad for param in model.output_head.parameters()
            if param.requires_grad and param.grad is not None
        ]
        assert head_grads, f"{head}: no uncertainty-head gradients"
        assert any(grad.abs().sum() > 0 for grad in head_grads), (
            f"{head}: all uncertainty-head gradients are exactly zero"
        )


@pytest.mark.parametrize("head", HEADS)
def test_expected_uncertainty_keys_are_finite_and_nondegenerate(
    cnn_model_factory, image_nodes, head
):
    """Each head must report its own keys, with real variation where applicable."""
    expected = {
        "none": set(),
        "evidential": {"evidential_vacuity", "evidential_total_evidence"},
        "batchensemble": {"batchensemble_variance"},
        "sngp": {"sngp_variance"},
    }[head]

    model = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.1)
    train_steps(model, image_nodes, steps=10, batch_size=4)

    model.eval()
    images = torch.stack([
        model.transform(node.get_data().load_data()) for node in image_nodes
    ])
    with torch.no_grad():
        bundle = model.forward_with_uncertainty(images)

    assert expected <= set(bundle.uncertainty), (
        f"{head}: missing {expected - set(bundle.uncertainty)}"
    )
    for name in expected:
        values = bundle.uncertainty[name]
        assert torch.isfinite(values).all(), f"{name} has non-finite values"
        assert values.shape == (len(image_nodes), 1)

    # Variance-style signals must not be identically zero at eval -- that is the
    # BatchEnsemble symmetry bug's signature.
    for name in ("batchensemble_variance", "sngp_variance"):
        if name in bundle.uncertainty:
            assert bundle.uncertainty[name].max().item() > 0.0, (
                f"{name} is identically zero, so it measures nothing"
            )


@pytest.mark.parametrize("head", HEADS)
def test_mc_dropout_produces_variation_when_available(cnn_model_factory, image_nodes, head):
    model = cnn_model_factory(
        uncertainty_head=head, mc_dropout_samples=8, uncertainty_dropout_rate=0.3
    )
    if not model.mc_dropout_available():
        pytest.skip("no stochastic dropout in this configuration")

    model.eval()
    images = torch.stack([
        model.transform(node.get_data().load_data()) for node in image_nodes
    ])
    with torch.no_grad():
        bundle = model.forward_with_uncertainty(images, use_mc_dropout=True)

    variance = bundle.uncertainty["mc_dropout_variance"]
    assert variance.max().item() > 0.0, "MC dropout produced zero variance"
    mutual_information = bundle.uncertainty["mc_dropout_mutual_information"]
    assert (mutual_information >= -1e-6).all(), "mutual information must be non-negative"


@pytest.mark.parametrize("head", HEADS)
def test_checkpoint_resume_reproduces_outputs(cnn_model_factory, image_nodes, tmp_path, head):
    """Train, save, rebuild, load: identical predictions and identical state."""
    path = tmp_path / f"{head}.pth"
    model = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.0)
    train_steps(model, image_nodes, steps=12, batch_size=4)
    model.save_checkpoint(str(path))

    model.eval()
    images = torch.stack([
        model.transform(node.get_data().load_data()) for node in image_nodes
    ])
    with torch.no_grad():
        expected = model.forward_with_uncertainty(images)

    restored = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.0)
    restored.load_checkpoint(str(path))
    restored.eval()
    with torch.no_grad():
        actual = restored.forward_with_uncertainty(images)

    assert torch.equal(expected.logits, actual.logits)
    assert torch.equal(expected.probabilities, actual.probabilities)
    assert state_dict_hash(model.model.model.state_dict()) == state_dict_hash(
        restored.model.model.state_dict()
    )


@pytest.mark.parametrize("head", HEADS)
def test_training_is_reproducible_at_a_fixed_seed(cnn_model_factory, image_nodes, head):
    """Same seed twice: identical loss trajectory and identical final weights."""
    from test_helpers.determinism import configure_determinism

    def run():
        configure_determinism(seed=808, mode="strict", allow_multi_gpu=True)
        model = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.2)
        losses = train_steps(model, image_nodes, steps=15, batch_size=4)
        return losses, state_dict_hash(model.model.model.state_dict())

    first_losses, first_hash = run()
    second_losses, second_hash = run()

    assert first_losses == second_losses, f"{head}: loss trajectory is not reproducible"
    assert first_hash == second_hash, f"{head}: final weights differ between runs"


def test_epoch_hooks_reset_sngp_precision(cnn_model_factory, image_nodes):
    """The per-epoch policy must actually reset between epochs.

    Without it, precision accumulates across the whole run and gp_variance is not
    comparable epoch to epoch -- which is the comparison the benchmark needs.
    """
    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.0)
    assert model.output_head.precision_policy == "per-epoch"

    model.on_epoch_start(0, num_epochs=2)
    train_steps(model, image_nodes, steps=6, batch_size=4)
    after_first = model.output_head.precision_matrix.clone()
    assert not torch.allclose(after_first, torch.eye(after_first.shape[0]))

    model.on_epoch_end(0)
    model.on_epoch_start(1, num_epochs=2)
    assert torch.equal(
        model.output_head.precision_matrix,
        torch.eye(after_first.shape[0], dtype=torch.float32),
    ), "epoch 1 did not start from the prior"


def test_graph_uncertainty_flows_through_training(cnn_model_factory, two_cluster_graph):
    """Graph-distance scores must reach the bundle during a real training step."""
    from models.uncertainty import GraphDistanceUncertainty

    _, nodes, _ = two_cluster_graph
    methods = ("attribute_distance", "hybrid_distance", "degree_penalty")
    model = cnn_model_factory(
        uncertainty_head="none", graph_uncertainty_methods=list(methods)
    )
    model.set_graph_distance_standardizer(
        GraphDistanceUncertainty(methods=methods).fit(nodes)
    )

    model.train()
    images = torch.rand(len(nodes), 3, 16, 16)
    bundle = model.forward_with_uncertainty(images, nodes=nodes)

    for method in methods:
        assert method in bundle.uncertainty, f"{method} did not reach the bundle"
        assert bundle.uncertainty[method].shape == (len(nodes), 1)
        assert torch.isfinite(bundle.uncertainty[method]).all()


def test_graph_uncertainty_warns_once_without_a_standardizer(cnn_model_factory, ring_graph, capsys):
    """Missing statistics must be reported, not silently produce nothing."""
    _, nodes, _ = ring_graph
    model = cnn_model_factory(
        uncertainty_head="none", graph_uncertainty_methods=["attribute_distance"]
    )
    assert not model.graph_uncertainty_ready()

    model.eval()
    images = torch.rand(len(nodes), 3, 16, 16)
    model.forward_with_uncertainty(images, nodes=nodes)
    first = capsys.readouterr().out
    assert "no fitted standardizer" in first

    model.forward_with_uncertainty(images, nodes=nodes)
    assert "no fitted standardizer" not in capsys.readouterr().out, "warning should fire once"
