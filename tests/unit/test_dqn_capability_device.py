"""The DQN training path must not be GPU-only, and must not hide its own failure.

`_preprocess_batch` ended with `torch.stack(...).cuda()`. On a CPU run that raised, the
caller's `if images is None: continue` swallowed it without advancing `nodes_processed`, and
the loop spun until the traversal exhausted its steps -- then reported a perfectly ordinary
epoch with `avg_loss: 0.0`. So `--traversal-type i-value` silently trained on *nothing*
whenever no GPU was visible, which includes every strict-determinism run that pins
`CUDA_VISIBLE_DEVICES` to a single device or to none, and every CPU test.

The two halves are tested separately: the device pin, and the swallow that hid it.
"""

import os
import re

import pytest


@pytest.fixture
def dqn_source(repo_root):
    path = os.path.join(repo_root, "trainers", "capabilities", "DQNCapability.py")
    return open(path).read()


def test_no_hardcoded_cuda_in_the_training_path(dqn_source):
    """`.cuda()` anywhere here makes the path GPU-only."""
    offenders = [
        line.strip() for line in dqn_source.splitlines()
        if re.search(r"(?<!torch)\.cuda\(\)", line) and not line.strip().startswith("#")
    ]
    assert not offenders, f"hardcoded .cuda() found: {offenders}"


def test_tensors_are_moved_to_the_capability_device(dqn_source):
    assert "torch.stack(processed_batch).to(self.device)" in dqn_source


def test_the_batch_loop_gives_up_rather_than_spinning(dqn_source):
    """A preprocess failure must not loop silently: the loop does not advance
    `nodes_processed`, so an unconditional `continue` reports a zero-loss epoch."""
    from trainers.capabilities.DQNCapability import (
        MAX_CONSECUTIVE_PREPROCESS_FAILURES,
    )

    assert MAX_CONSECUTIVE_PREPROCESS_FAILURES > 0
    assert "batches_failed" in dqn_source
    assert "MAX_CONSECUTIVE_PREPROCESS_FAILURES" in dqn_source
    # And it raises rather than returning zeroed metrics.
    section = dqn_source[dqn_source.index("batches_failed >= MAX_CONSECUTIVE"):][:400]
    assert "raise RuntimeError" in section


def test_the_counter_resets_after_a_good_batch(dqn_source):
    """Otherwise a run with scattered bad images would eventually trip the limit."""
    index = dqn_source.index("batch_labels_loaded = [float(node.get_label())")
    preceding = dqn_source[:index]
    assert "batches_failed = 0" in preceding.rsplit("continue", 1)[-1]


@pytest.mark.parametrize("device", ["cpu"])
def test_preprocess_stacks_on_the_requested_device(device, tiny_detector, image_nodes):
    """The behavioural half: a CPU capability must return CPU tensors, not raise."""
    import torch

    from trainers.capabilities.DQNCapability import DQNCapability

    # Build the capability without its full trainer: `_preprocess_batch` needs only
    # `trainer.models[0].transform` and `self.device`.
    capability = DQNCapability.__new__(DQNCapability)
    capability.device = torch.device(device)

    from models.CNNModel import CNNModel
    model = CNNModel("/tmp/unused", tiny_detector, 1e-4, True, torch.device(device))
    capability.trainer = type("T", (), {"models": [model]})()

    images, valid = capability._preprocess_batch(image_nodes[:4])
    assert images is not None, "preprocessing must succeed on CPU"
    assert images.device.type == device
    assert len(valid) == len(image_nodes[:4])
