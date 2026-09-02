"""The properties that make the gain estimators work at all.

Each test here corresponds to one of the three measured failures in
`docs/ivalue_gate_result.md`. They are not incidental invariants: if the state block moves,
or the gate stops starting at zero, or the output gets bounded again, the estimator silently
reverts to the thing that scored +0.010 against a free baseline of +0.331 -- and nothing
would raise.
"""

import math

import pytest
import torch

from models.gain_estimator import (
    DEFAULT_BUFFER_SIZE,
    GAIN_ESTIMATORS,
    GainEnsemble,
    GainLinear,
    GainResidual,
    LegacyGainAdapter,
    build_estimator,
)
from trainers.capabilities.node_state import (
    STATE_FEATURE_COUNT,
    STATE_LOSS_INDEX,
    NodeTrainingState,
)

FEATURE_DIM = 31
CPU = torch.device("cpu")


def _inputs(rows=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, FEATURE_DIM, generator=generator),
            torch.rand(rows, 512, generator=generator))


# --------------------------------------------------------------------------- Fix 1 ----
@pytest.mark.parametrize("name", sorted(GAIN_ESTIMATORS))
def test_untrained_model_is_exactly_the_current_loss_baseline(name):
    """The design guarantee: every variant *starts* as the free baseline.

    `state_linear` carries +1 on the current-loss column and the residual is gated shut, so
    an untrained model's output is the loss column verbatim. This is the direct answer to a
    network that was handed the answer in its input and ignored it.
    """
    features, embedding = _inputs()
    model = GAIN_ESTIMATORS[name](FEATURE_DIM, CPU)
    predicted = model.predict_value(features, embedding).reshape(-1)
    loss_column = features[:, -STATE_FEATURE_COUNT + STATE_LOSS_INDEX]
    assert torch.allclose(predicted, loss_column, atol=1e-6)


def test_the_residual_gate_starts_shut():
    model = GainResidual(FEATURE_DIM, CPU)
    assert float(torch.tanh(model.residual_gate)) == 0.0


def test_the_state_block_is_the_trailing_slice():
    """Pins the contract with `DQNCapability._get_dqn_features`, which appends the
    model-state features *after* the static attributes. If that order ever flips, the
    loss-column initialisation would point at an unrelated demographic one-hot."""
    state = NodeTrainingState()

    class _Node:
        node_id = "n"

    state.observe(_Node(), prob=0.5, loss=2.5, epoch=0)
    block = state.features(_Node())
    assert len(block) == STATE_FEATURE_COUNT

    static = [0.0] * (FEATURE_DIM - STATE_FEATURE_COUNT)
    features = torch.tensor([static + block], dtype=torch.float32)
    model = GainLinear(FEATURE_DIM, CPU)
    # loss 2.5 / LOSS_SCALE 5.0 = 0.5, and the linear head reads exactly that column.
    assert model.predict_value(features).item() == pytest.approx(0.5, abs=1e-6)


def test_embedding_pathway_can_be_removed_entirely():
    model = GainResidual(FEATURE_DIM, CPU, embedding_dim=0, use_embedding=False)
    assert model.embedding_processor is None
    features, _ = _inputs()
    assert model.predict_value(features).shape == (4, 1)


def test_missing_embedding_does_not_crash_the_residual_model():
    model = GainResidual(FEATURE_DIM, CPU)
    features, _ = _inputs()
    assert torch.isfinite(model.predict_value(features, None)).all()


# --------------------------------------------------------------------------- Fix 3 ----
@pytest.mark.parametrize("name", sorted(GAIN_ESTIMATORS))
def test_output_is_unbounded(name):
    """No sigmoid. The old head squashed a signed, unbounded target into (0, 1)."""
    model = GAIN_ESTIMATORS[name](FEATURE_DIM, CPU)
    huge = torch.zeros(2, FEATURE_DIM)
    huge[:, -STATE_FEATURE_COUNT + STATE_LOSS_INDEX] = torch.tensor([-50.0, 50.0])
    values = model.predict_value(huge).reshape(-1)
    assert values.min() < 0.0 and values.max() > 1.0


@pytest.mark.parametrize("name", sorted(GAIN_ESTIMATORS))
def test_declares_informativeness_semantics(name):
    """So `get_i_value` passes the score through instead of inverting it."""
    assert GAIN_ESTIMATORS[name](FEATURE_DIM, CPU).value_semantics == "informativeness"


def test_unknown_objective_is_refused():
    with pytest.raises(ValueError, match="unknown objective"):
        GainResidual(FEATURE_DIM, CPU, objective="whatever")


# --------------------------------------------------------------------------- Fix 2 ----
def test_buffer_is_small_by_default():
    # The old default was 10,000 -- a full epoch of labels for a model that has since moved.
    assert GainResidual(FEATURE_DIM, CPU).replay_buffer.maxlen == DEFAULT_BUFFER_SIZE
    assert DEFAULT_BUFFER_SIZE < 10_000


def test_observe_timestamps_and_learn_needs_a_full_batch():
    model = GainResidual(FEATURE_DIM, CPU, batch_size=8)
    features, embedding = _inputs(rows=1)
    for _ in range(4):
        model.observe(features[0], embedding[0], reward=0.5)
    assert len(model.replay_buffer) == 4
    assert model.learn() is None, "must not train on a partial batch"
    for index in range(8):
        model.observe(features[0], embedding[0], reward=float(index))
    assert model.learn() is not None


def test_transitions_past_the_age_cutoff_are_never_sampled():
    model = GainResidual(FEATURE_DIM, CPU, batch_size=2, max_transition_age=5,
                         buffer_size=64)
    features, embedding = _inputs(rows=1)
    for _ in range(20):
        model.observe(features[0], embedding[0], reward=1.0)
    now = model._observations
    for transition in model.sample_transitions():
        assert now - transition[3] <= 5


def test_recency_weighting_prefers_recent_transitions():
    model = GainResidual(FEATURE_DIM, CPU, batch_size=4, buffer_size=200,
                         max_transition_age=10_000, recency_half_life=5)
    features, embedding = _inputs(rows=1)
    for _ in range(100):
        model.observe(features[0], embedding[0], reward=1.0)
    now = model._observations
    ages = []
    for _ in range(40):
        ages += [now - t[3] for t in model.sample_transitions()]
    # With a 5-observation half-life over a 100-deep buffer, draws must skew hard to recent.
    assert sum(ages) / len(ages) < 40


# ------------------------------------------------------------------- legacy adapter ---
def test_adapter_starts_at_the_baseline_too():
    """So "fixed vs original" differs by the three fixes and nothing else."""
    features, embedding = _inputs()
    fixed = build_estimator("basic", FEATURE_DIM, CPU, apply_fixes=True)
    baseline = GainLinear(FEATURE_DIM, CPU)
    assert torch.allclose(fixed.predict_value(features, embedding),
                          baseline.predict_value(features, embedding), atol=1e-6)


def test_adapter_optimises_the_legacy_parameters():
    fixed = build_estimator("basic", FEATURE_DIM, CPU, apply_fixes=True)
    optimised = {id(p) for group in fixed.optimizer.param_groups for p in group["params"]}
    assert any(id(p) in optimised for p in fixed.legacy.parameters())
    assert id(fixed.residual_gate) in optimised


@pytest.mark.parametrize("name", ["basic", "residual", "attention", "conv_embedding", "ensemble"])
def test_every_legacy_architecture_can_be_wrapped(name):
    fixed = build_estimator(name, FEATURE_DIM, CPU, apply_fixes=True)
    features, embedding = _inputs()
    assert fixed.value_semantics == "informativeness"
    assert torch.isfinite(fixed.predict_value(features, embedding)).all()


def test_unwrapped_legacy_is_untouched_and_still_inverted():
    """The originals must stay reproducible: bounded output, no `value_semantics`."""
    legacy = build_estimator("basic", FEATURE_DIM, CPU)
    features, embedding = _inputs()
    values = legacy.predict_i_value(features, embedding).reshape(-1)
    assert not hasattr(legacy, "value_semantics")
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_checkpoint_round_trip(tmp_path):
    model = GainResidual(FEATURE_DIM, CPU)
    features, embedding = _inputs()
    before = model.predict_value(features, embedding)
    path = tmp_path / "estimator.pt"
    model.save_checkpoint(str(path))

    restored = GainResidual(FEATURE_DIM, CPU)
    restored.load_checkpoint(str(path))
    assert torch.allclose(before, restored.predict_value(features, embedding), atol=1e-6)


def test_ensemble_spread_is_zero_while_the_gate_is_shut():
    model = GainEnsemble(FEATURE_DIM, CPU)
    features, embedding = _inputs()
    assert float(model.predict_spread(features, embedding).abs().max()) == 0.0


# ------------------------------------------------------------ the unlearned control ---
def test_loss_ewma_ranker_is_the_stored_signal_and_never_moves():
    """The control the estimator programme has to beat: one stored number, no learning.

    Frozen rather than reimplemented, so it is provably the same starting point the learned
    models have -- any difference in a training run comes from what they learn, not from a
    different feature or scale.
    """
    from models.gain_estimator import build_estimator

    features, embedding = _inputs()
    ranker = build_estimator("loss_ewma", FEATURE_DIM, CPU)
    loss_column = features[:, -STATE_FEATURE_COUNT + STATE_LOSS_INDEX]

    before = ranker.predict_value(features, embedding).reshape(-1).clone()
    assert torch.allclose(before, loss_column, atol=1e-6)

    for _ in range(25):
        ranker.observe(features[0], embedding[0], reward=99.0)
        ranker.learn()
        ranker.train_step([(features[i], embedding[i], float(i), i + 1) for i in range(4)])

    after = ranker.predict_value(features, embedding).reshape(-1)
    assert torch.allclose(before, after, atol=1e-9), "an unlearned control must not drift"
    assert not any(p.requires_grad for p in ranker.parameters())
    assert ranker.value_semantics == "informativeness"


def test_loss_ewma_and_untrained_gain_linear_agree():
    from models.gain_estimator import build_estimator

    features, embedding = _inputs()
    ranker = build_estimator("loss_ewma", FEATURE_DIM, CPU)
    linear = build_estimator("gain_linear", FEATURE_DIM, CPU)
    assert torch.allclose(ranker.predict_value(features, embedding),
                          linear.predict_value(features, embedding), atol=1e-7)
