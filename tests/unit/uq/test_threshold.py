"""Fitting the decision threshold on held-out data.

The failure this exists for: on the ~87%-fake AI-Face split, an undertrained model ends up
with a mean logit several units above the +1.95 the prior justifies while its class
separation is only ~1.8. Every probability then lands above 0.5, accuracy reads as the
majority-class prior, and balanced accuracy pins at exactly 0.5 -- even though the ranking
is informative (AUROC 0.67-0.80 on measured runs).

**Temperature scaling cannot repair that**, and the first test says so directly: dividing a
logit by any positive T preserves its sign, so no prediction moves across the boundary. Only
a threshold does.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.uq.threshold import (
    DEFAULT_THRESHOLD, OBJECTIVES, ThresholdError, ThresholdFit, apply_to_records,
    fit_from_records, fit_threshold, load_fit, save_fit,
)


def collapsed_scores(n=1000, prior=0.8755, separation=1.8, offset=5.2, seed=0):
    """Labels and probabilities shaped like a real collapsed run.

    Logits centred well above the prior's log-odds, with genuine class separation, so every
    probability exceeds 0.5 while the ranking still carries signal.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < prior).astype(int)
    logits = offset + np.where(labels == 1, separation / 2, -separation / 2)
    logits = logits + rng.normal(0, 1.0, size=n)
    logits = np.maximum(logits, 0.05)  # keep every logit positive: the collapse condition
    return labels, 1.0 / (1.0 + np.exp(-logits))


def balanced_accuracy(labels, probabilities, threshold):
    predictions = (probabilities > threshold).astype(int)
    positive, negative = labels == 1, labels == 0
    return 0.5 * (
        (predictions[positive] == 1).mean() + (predictions[negative] == 0).mean()
    )


# -- the reason this module exists ---------------------------------------------- #

def test_temperature_scaling_cannot_move_a_single_prediction():
    """Sign-preserving, therefore useless against majority-class collapse."""
    labels, probabilities = collapsed_scores()
    logits = np.log(probabilities / (1 - probabilities))
    assert (logits > 0).all(), "fixture must reproduce the all-positive-logit condition"

    for temperature in (0.1, 0.5, 2.0, 20.0):
        scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
        assert ((scaled > 0.5).astype(int) == (probabilities > 0.5).astype(int)).all()

    # A threshold does move them.
    fit = fit_threshold(labels, probabilities)
    moved = (probabilities > fit.threshold).astype(int)
    assert np.unique(moved).size == 2


def test_a_collapsed_run_is_recognised_and_improved():
    labels, probabilities = collapsed_scores()
    fit = fit_threshold(labels, probabilities)

    assert fit.applicable
    assert fit.collapsed_at_default
    assert fit.balanced_accuracy_at_default == pytest.approx(0.5)
    assert fit.balanced_accuracy_at_threshold > 0.6
    assert fit.improved
    assert not fit.is_default


def test_a_val_fitted_threshold_generalizes_to_held_out_data():
    """The property that makes this legitimate rather than metric-fitting."""
    val_labels, val_probabilities = collapsed_scores(seed=1)
    test_labels, test_probabilities = collapsed_scores(seed=2)

    fit = fit_threshold(val_labels, val_probabilities)
    before = balanced_accuracy(test_labels, test_probabilities, DEFAULT_THRESHOLD)
    after = balanced_accuracy(test_labels, test_probabilities, fit.threshold)

    assert before == pytest.approx(0.5)
    assert after > 0.6, "a threshold fitted on val must transfer to test"


# -- objectives ----------------------------------------------------------------- #

def test_the_accuracy_objective_trades_away_minority_class_performance():
    """Why balanced_accuracy is the default.

    At 87% prevalence the accuracy-maximizing threshold sits far toward the majority class.
    It does beat the prior when real signal exists -- so this is a trade, not a no-op -- but
    it buys that accuracy by getting the minority class wrong, which on a deepfake detector
    means missing real faces. The balanced objective gives up accuracy for a far better
    minority-class rate.
    """
    labels, probabilities = collapsed_scores()
    by_accuracy = fit_threshold(labels, probabilities, objective="accuracy")
    by_balance = fit_threshold(labels, probabilities, objective="balanced_accuracy")

    assert by_accuracy.accuracy_at_threshold > by_balance.accuracy_at_threshold
    assert by_balance.balanced_accuracy_at_threshold > (
        by_accuracy.balanced_accuracy_at_threshold + 0.05
    )
    # Specificity -- the rate of correctly identifying the minority (real) class.
    negative = labels == 0
    specificity = {
        name: ((probabilities > fit.threshold).astype(int)[negative] == 0).mean()
        for name, fit in (("accuracy", by_accuracy), ("balanced", by_balance))
    }
    assert specificity["balanced"] > specificity["accuracy"] + 0.1


def test_youden_j_and_balanced_accuracy_agree():
    """Monotonically equivalent for a fixed dataset, so they select the same point."""
    labels, probabilities = collapsed_scores()
    a = fit_threshold(labels, probabilities, objective="balanced_accuracy")
    b = fit_threshold(labels, probabilities, objective="youden_j")
    assert a.threshold == pytest.approx(b.threshold)


def test_unknown_objective_is_refused():
    labels, probabilities = collapsed_scores()
    with pytest.raises(ThresholdError, match="unknown objective"):
        fit_threshold(labels, probabilities, objective="f1")


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_produces_a_usable_fit(objective):
    labels, probabilities = collapsed_scores()
    fit = fit_threshold(labels, probabilities, objective=objective)
    assert 0.0 <= fit.threshold <= 1.0
    assert fit.objective == objective


# -- degenerate input ----------------------------------------------------------- #

def test_a_single_class_split_is_not_applicable_rather_than_an_error():
    """Callers apply `fit.threshold` unconditionally, so it must always be usable."""
    labels = np.ones(100, dtype=int)
    probabilities = np.linspace(0.6, 0.99, 100)
    fit = fit_threshold(labels, probabilities)

    assert not fit.applicable
    assert fit.threshold == DEFAULT_THRESHOLD
    assert "only class" in fit.reason


def test_empty_input_is_not_applicable():
    fit = fit_threshold([], [])
    assert not fit.applicable
    assert fit.threshold == DEFAULT_THRESHOLD


def test_non_finite_probabilities_are_dropped():
    labels, probabilities = collapsed_scores(n=200, seed=3)
    probabilities = probabilities.copy()
    probabilities[:10] = np.nan
    fit = fit_threshold(labels, probabilities)
    assert fit.n_val == 190


def test_mismatched_lengths_are_refused():
    with pytest.raises(ThresholdError, match="disagree on length"):
        fit_threshold([0, 1, 1], [0.5, 0.6])


def test_a_perfect_separation_is_found():
    labels = np.array([0] * 50 + [1] * 50)
    probabilities = np.array([0.1] * 50 + [0.9] * 50)
    fit = fit_threshold(labels, probabilities)
    assert fit.balanced_accuracy_at_threshold == pytest.approx(1.0)


def test_ties_resolve_toward_the_default():
    """A fit that gains nothing should stay recognizably near 0.5."""
    labels = np.array([0, 1] * 50)
    probabilities = np.array([0.5] * 100)  # no information at all
    fit = fit_threshold(labels, probabilities)
    assert abs(fit.threshold - DEFAULT_THRESHOLD) < 0.2


# -- records integration -------------------------------------------------------- #

def records_frame(labels, probabilities):
    predictions = (probabilities > 0.5).astype(int)
    return pd.DataFrame({
        "record_id": [f"r{i:05d}" for i in range(len(labels))],
        "label": labels,
        "prob": probabilities,
        "pred": predictions,
        "correct": (predictions == labels).astype(int),
    })


def test_fit_from_records():
    labels, probabilities = collapsed_scores(n=400, seed=4)
    fit = fit_from_records(records_frame(labels, probabilities))
    assert fit.applicable
    assert fit.n_val == 400


def test_fit_from_records_needs_the_columns():
    with pytest.raises(ThresholdError, match="prob"):
        fit_from_records(pd.DataFrame({"label": [0, 1]}))


def test_apply_to_records_rewrites_predictions_but_not_probabilities():
    """A threshold is a decision rule: every ranking metric must be invariant to it."""
    labels, probabilities = collapsed_scores(n=400, seed=5)
    frame = records_frame(labels, probabilities)
    fit = fit_from_records(frame)
    updated = apply_to_records(frame, fit)

    assert (updated["prob"].to_numpy() == frame["prob"].to_numpy()).all()
    assert not (updated["pred"].to_numpy() == frame["pred"].to_numpy()).all()
    assert (
        updated["correct"].to_numpy()
        == (updated["pred"].to_numpy() == labels).astype(int)
    ).all()
    # The operating point travels with the table, so it cannot be read as 0.5-thresholded.
    assert updated["threshold"].unique().tolist() == [pytest.approx(fit.threshold)]


def test_apply_to_records_accepts_a_bare_float():
    labels, probabilities = collapsed_scores(n=100, seed=6)
    updated = apply_to_records(records_frame(labels, probabilities), 0.9)
    assert updated["threshold"].unique().tolist() == [0.9]


def test_a_fit_round_trips_through_json(tmp_path):
    labels, probabilities = collapsed_scores(n=200, seed=7)
    fit = fit_threshold(labels, probabilities)
    path = save_fit(fit, str(tmp_path / "threshold_fit.json"))
    reloaded = load_fit(path)
    assert isinstance(reloaded, ThresholdFit)
    assert reloaded.threshold == pytest.approx(fit.threshold)
    assert reloaded.objective == fit.objective


# -- scoring integration -------------------------------------------------------- #

def test_the_scoring_layer_honours_a_cell_threshold():
    from evaluation.uq.scoring import Cell, score_cells

    labels, probabilities = collapsed_scores(n=600, seed=8)
    frame = records_frame(labels, probabilities)
    fit = fit_from_records(frame)

    def score(threshold):
        return score_cells(
            [Cell(detector="tiny", method_id="baseline_maxprob",
                  score_column="u_maxprob", frame=frame, threshold=threshold)],
            require_comparable=False,
        ).iloc[0]

    at_default, at_fitted = score(DEFAULT_THRESHOLD), score(fit.threshold)

    assert at_default["clf_balanced_accuracy"] == pytest.approx(0.5)
    assert at_fitted["clf_balanced_accuracy"] > 0.6
    # AUROC is threshold-free and must not move.
    assert at_fitted["clf_auroc"] == pytest.approx(at_default["clf_auroc"])
    # The operating point is recorded on the row.
    assert at_fitted["threshold"] == pytest.approx(fit.threshold)


def test_the_default_threshold_reproduces_the_original_numbers():
    """Backward compatibility: an omitted threshold must change nothing."""
    from evaluation.uq.metrics import discrimination_metrics

    labels, probabilities = collapsed_scores(n=300, seed=9)
    explicit, _ = discrimination_metrics(labels, probabilities, threshold=0.5)
    implicit, _ = discrimination_metrics(labels, probabilities)
    assert explicit == implicit
