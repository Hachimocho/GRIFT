"""The five original estimators must stay exactly as they were.

Every sweep in `run_outputs/sweeps/` was produced with them, so if their behaviour drifts the
historical baselines stop meaning anything. They are deprecated, not changed: the fixes
arrive by wrapping (`--dqn-fixes`), which leaves `models/DQNModel.py` and
`models/EnhancedDQNModels.py` untouched.
"""

import pytest
import torch

from models.gain_estimator import LEGACY_ESTIMATORS, build_estimator

FEATURE_DIM = 31
CPU = torch.device("cpu")

LEGACY_CLASSES = {
    "basic": ("models.DQNModel", "DQNModel"),
    "residual": ("models.EnhancedDQNModels", "ResidualDQNModel"),
    "attention": ("models.EnhancedDQNModels", "AttentionDQNModel"),
    "conv_embedding": ("models.EnhancedDQNModels", "ConvEmbeddingDQN"),
    "ensemble": ("models.EnhancedDQNModels", "EnsembleDQNModel"),
}


def test_all_five_names_are_registered():
    assert set(LEGACY_ESTIMATORS) == set(LEGACY_CLASSES)


def test_factory_returns_the_same_classes_the_old_dispatch_did():
    import importlib

    for name, (module_name, class_name) in LEGACY_CLASSES.items():
        expected = getattr(importlib.import_module(module_name), class_name)
        assert isinstance(build_estimator(name, FEATURE_DIM, CPU), expected)


def test_the_three_defects_are_still_present_when_unwrapped():
    """Documented, not endorsed. These are the measured causes of the +0.010 result, and
    pinning them is how we know `--dqn-fixes` is actually changing something."""
    legacy = build_estimator("basic", FEATURE_DIM, CPU)

    # 1. the embedding dominates the input
    assert legacy.compressed_embedding_dim == 64
    assert legacy.compressed_embedding_dim > FEATURE_DIM

    # 2. a full epoch of stale labels
    assert legacy.replay_buffer.maxlen == 10_000

    # 3. the output is squashed into (0, 1)
    extreme = torch.full((2, FEATURE_DIM), 1e4)
    values = legacy.predict_i_value(extreme, torch.zeros(2, 512)).reshape(-1)
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_wrapping_fixes_all_three():
    fixed = build_estimator("basic", FEATURE_DIM, CPU, apply_fixes=True)
    assert fixed.replay_buffer.maxlen < 10_000
    assert fixed.value_semantics == "informativeness"
    assert hasattr(fixed, "observe") and hasattr(fixed, "learn")
    # unbounded output
    extreme = torch.zeros(2, FEATURE_DIM)
    extreme[:, -4] = torch.tensor([-50.0, 50.0])
    values = fixed.predict_value(extreme).reshape(-1)
    assert values.min() < 0.0 or values.max() > 1.0


def test_unwrapped_legacy_has_no_model_owned_buffer():
    """So `DQNCapability` takes the legacy append-and-sample branch for them."""
    legacy = build_estimator("basic", FEATURE_DIM, CPU)
    assert not hasattr(legacy, "observe")
    assert not hasattr(legacy, "learn")


def test_deprecation_notice_is_printed_only_without_fixes(capsys):
    build_estimator("basic", FEATURE_DIM, CPU)
    assert "[deprecated]" in capsys.readouterr().out
    build_estimator("basic", FEATURE_DIM, CPU, apply_fixes=True)
    assert "[deprecated]" not in capsys.readouterr().out


def test_the_flags_reach_the_cli_and_the_queue():
    from test_helpers.args_utils import parse_args
    from web_ui.gpu_queue_manager import ARG_MAPPING, validate_config_keys

    defaults = parse_args([])
    assert defaults.dqn_model == "basic" and defaults.dqn_fixes is False, \
        "defaults must reproduce today's behaviour"
    assert parse_args(["--dqn-model", "gain_residual"]).dqn_model == "gain_residual"
    for key in ("dqn_fixes", "dqn_objective", "dqn_buffer_size", "dqn_embedding_dim"):
        assert key in ARG_MAPPING
    assert validate_config_keys({key: 1 for key in (
        "dqn_fixes", "dqn_objective", "dqn_buffer_size", "dqn_embedding_dim")}) == []


# ------------------------------------------------------- regressions from the matrix ---
@pytest.mark.parametrize("name", ["basic", "residual", "attention", "conv_embedding", "ensemble"])
def test_wrapped_legacy_receives_its_embeddings(name):
    """The bug that cost three days of GPU.

    `LegacyGainAdapter` has no `embedding_processor` of its own -- the legacy network consumes
    the embedding internally -- so inferring "does this model want embeddings?" from
    `embedding_processor is not None` handed every wrapped model `None`. `DQNModel` silently
    substitutes zeros, so `basic.fixed` ran and simply never saw an embedding; the other four
    dereference `.shape` on it and raised on every single batch.
    """
    fixed = build_estimator(name, FEATURE_DIM, CPU, apply_fixes=True)
    assert fixed.wants_embeddings is True

    features = torch.rand(4, FEATURE_DIM)
    embedding = torch.rand(4, 512)
    transitions = [(features[i], embedding[i], float(i) * 0.1, i + 1) for i in range(4)]
    # Must not raise, and must actually train.
    assert fixed.train_step(transitions) is not None


@pytest.mark.parametrize("name", ["basic", "residual", "attention", "conv_embedding", "ensemble"])
def test_wrapped_legacy_tolerates_a_missing_embedding(name):
    """The adapter materialises zeros rather than forwarding None, because the five legacy
    models disagree about what None means and that is not the adapter's to inherit."""
    fixed = build_estimator(name, FEATURE_DIM, CPU, apply_fixes=True)
    assert torch.isfinite(fixed.predict_value(torch.rand(2, FEATURE_DIM), None)).all()


def test_a_persistently_failing_batch_raises_rather_than_spinning():
    """The second half of the same incident: a failure in the batch body does not advance
    `nodes_processed`, so at --train-steps 4000000 it spins forever. 258,398 identical
    tracebacks were logged over ~3 days without completing an epoch."""
    from trainers.capabilities.DQNCapability import MAX_CONSECUTIVE_BATCH_FAILURES
    import inspect
    from trainers.capabilities import DQNCapability as module

    assert 0 < MAX_CONSECUTIVE_BATCH_FAILURES < 1000
    source = inspect.getsource(module.DQNCapability.train_with_dqn)
    assert "batches_errored" in source
    # The counter must reset on success, or a run with sporadic faults would eventually die.
    assert "batches_errored = 0" in source
    assert "MAX_CONSECUTIVE_BATCH_FAILURES" in source
    assert "raise RuntimeError" in source
