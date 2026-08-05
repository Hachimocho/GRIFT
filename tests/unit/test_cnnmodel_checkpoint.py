"""Checkpoint completeness and round-trip fidelity.

Every piece of auxiliary state that a resumed run needs must survive save/load.
Anything that silently resets makes two nominally identical runs diverge with no
visible symptom.
"""

import pytest
import torch

HEADS = ("none", "evidential", "batchensemble", "sngp")


@pytest.mark.parametrize("head", HEADS)
def test_checkpoint_roundtrip_reproduces_outputs(
    cnn_model_factory, tiny_batch, tiny_labels, state_dict_hash, tmp_path, head
):
    path = tmp_path / f"{head}.pth"
    original = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.0)

    # Train a step so the saved state is not just the initialization.
    original.train()
    loss = original.compute_loss(
        original.forward_with_uncertainty(
            tiny_batch(batch_size=4, size=16), update_precision=True
        ),
        tiny_labels(4),
    )
    loss.backward()
    original.optim.step()
    original.save_checkpoint(str(path))

    original.eval()
    probe = tiny_batch(batch_size=3, size=16, seed=5)
    with torch.no_grad():
        expected = original.forward_with_uncertainty(probe)

    restored = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.0)
    restored.load_checkpoint(str(path))
    restored.eval()
    with torch.no_grad():
        actual = restored.forward_with_uncertainty(probe)

    assert torch.equal(expected.logits, actual.logits), "logits differ after reload"
    assert torch.equal(expected.probabilities, actual.probabilities)
    assert state_dict_hash(original.model.model.state_dict()) == state_dict_hash(
        restored.model.model.state_dict()
    )
    if original.output_head is not None:
        assert state_dict_hash(original.output_head.state_dict()) == state_dict_hash(
            restored.output_head.state_dict()
        )


def test_evidential_annealing_counter_is_checkpointed(
    cnn_model_factory, tiny_batch, tiny_labels, tmp_path
):
    """A resumed run must not restart the KL annealing schedule from zero."""
    path = tmp_path / "evidential.pth"
    model = cnn_model_factory(uncertainty_head="evidential", uncertainty_dropout_rate=0.0)
    model.train()
    for _ in range(5):
        model.compute_loss(
            model.forward_with_uncertainty(tiny_batch(batch_size=2, size=16)),
            tiny_labels(2),
        )
    assert model.evidential_loss.global_step.item() == 5
    model.save_checkpoint(str(path))

    restored = cnn_model_factory(uncertainty_head="evidential", uncertainty_dropout_rate=0.0)
    assert restored.evidential_loss.global_step.item() == 0
    restored.load_checkpoint(str(path))
    assert restored.evidential_loss.global_step.item() == 5


def test_scheduler_state_is_checkpointed(cnn_model_factory, tmp_path):
    """ReduceLROnPlateau's patience counters must survive a resume."""
    path = tmp_path / "sched.pth"
    model = cnn_model_factory(uncertainty_head="none")
    if getattr(model, "scheduler", None) is None:
        pytest.skip("model has no LR scheduler")

    # Two steps: the first establishes `best`, the second is a bad epoch. Stopping
    # at two avoids tripping `patience`, which would reduce the LR and reset the
    # counter back to zero.
    model.scheduler.step(1.0)
    model.scheduler.step(1.0)
    saved_bad_epochs = model.scheduler.num_bad_epochs
    assert saved_bad_epochs > 0, "precondition: the scheduler should have logged a bad epoch"
    model.save_checkpoint(str(path))

    restored = cnn_model_factory(uncertainty_head="none")
    assert restored.scheduler.num_bad_epochs == 0
    restored.load_checkpoint(str(path))
    assert restored.scheduler.num_bad_epochs == saved_bad_epochs


def test_sngp_precision_matrix_is_checkpointed(cnn_model_factory, tiny_batch, tmp_path):
    """The Laplace precision is a registered buffer, so it rides in state_dict."""
    path = tmp_path / "sngp.pth"
    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.0)
    model.train()
    for seed in range(3):
        model.forward_with_uncertainty(
            tiny_batch(batch_size=4, size=16, seed=seed), update_precision=True
        )
    accumulated = model.output_head.precision_matrix.clone()
    model.save_checkpoint(str(path))

    restored = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.0)
    restored.load_checkpoint(str(path))
    assert torch.equal(restored.output_head.precision_matrix, accumulated)


def test_checkpoint_records_construction_metadata(cnn_model_factory, tmp_path):
    path = tmp_path / "meta.pth"
    model = cnn_model_factory(uncertainty_head="sngp", finetune=True)
    model.save_checkpoint(str(path))

    checkpoint = torch.load(str(path), map_location="cpu")
    assert checkpoint["format_version"] >= 2
    assert checkpoint["uncertainty_head_type"] == "sngp"
    assert checkpoint["finetune"] is True
    assert checkpoint["sngp_precision_policy"] == "per-epoch"


def test_missing_checkpoint_warns_without_raising(cnn_model_factory, tmp_path, capsys):
    model = cnn_model_factory(uncertainty_head="none")
    model.load_checkpoint(str(tmp_path / "absent.pth"))
    assert "not found" in capsys.readouterr().out


def test_legacy_checkpoint_loads_and_reports_gaps(cnn_model_factory, tmp_path, capsys):
    """A pre-v2 checkpoint must still load, but say what it could not restore."""
    path = tmp_path / "legacy.pth"
    model = cnn_model_factory(uncertainty_head="evidential", uncertainty_dropout_rate=0.0)
    torch.save(
        {
            "model_state_dict": model.model.model.state_dict(),
            "optimizer_state_dict": model.optim.state_dict(),
            "uncertainty_head_type": "evidential",
        },
        str(path),
    )

    restored = cnn_model_factory(uncertainty_head="evidential", uncertainty_dropout_rate=0.0)
    restored.load_checkpoint(str(path))
    output = capsys.readouterr().out
    assert "had no entry for" in output
    assert "evidential_loss_state_dict" in output


def test_head_type_mismatch_raises_a_clear_error(cnn_model_factory, tmp_path):
    """Checkpoints are not interchangeable across uncertainty heads.

    Grafting a head replaces the backbone's final Linear with nn.Identity, so the
    state dicts genuinely do not match. Without an explicit check this surfaced as
    an opaque `Missing key(s) in state_dict: "6.weight", "6.bias"`, which says
    nothing about the actual cause.
    """
    path = tmp_path / "mismatch.pth"
    cnn_model_factory(uncertainty_head="sngp").save_checkpoint(str(path))

    other = cnn_model_factory(uncertainty_head="none")
    with pytest.raises(ValueError, match="not interchangeable"):
        other.load_checkpoint(str(path))


def test_finetune_flag_controls_trainable_parameters(cnn_model_factory):
    """B1: `finetune=True` freezes the backbone in the real detectors.

    The tiny detector mirrors that convention, so this pins the semantics of the
    flag and of the reporting that surfaces it.
    """
    full = cnn_model_factory(uncertainty_head="none", finetune=False)
    frozen = cnn_model_factory(uncertainty_head="none", finetune=True)

    full_counts = full.parameter_counts()
    frozen_counts = frozen.parameter_counts()

    assert full_counts["trainable"] == full_counts["total"], (
        "the default must be full fine-tuning"
    )
    assert frozen_counts["trainable"] < frozen_counts["total"]
    assert frozen_counts["finetune"] is True
    assert full_counts["backbone_frozen"] is False
