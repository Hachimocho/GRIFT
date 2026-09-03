"""Metric correctness, against hand-computed values.

The literals in these fixtures are worked out in the docstrings so the assertions
are checkable by reading rather than by trusting the implementation.
"""

import numpy as np
import pytest

from evaluation.uq.metrics import (
    DEGENERATE_CONSTANT_SCORE, DEGENERATE_SINGLE_BIN, EMPTY_OOD_PARTITION,
    INSUFFICIENT_SAMPLES, MCE_NO_QUALIFYING_BIN, NAN_SCORES_DROPPED,
    SINGLE_CLASS_ERROR, SINGLE_CLASS_LABELS, accuracy_at_coverage, aupr_error,
    binary_calibration, bootstrap_ci, brier_score, discrimination_metrics,
    negative_log_likelihood, ood_detection, reliability_diagram_data,
    risk_coverage_curve, score_entropy, score_margin, score_max_probability,
    uncertainty_error_auroc,
)


# --------------------------------------------------------------------------- #
# ECE
# --------------------------------------------------------------------------- #

def test_perfectly_calibrated_predictions_have_zero_ece():
    """p == 1.0 on every correct prediction: confidence 1.0, accuracy 1.0, gap 0."""
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([1.0, 1.0, 0.0, 0.0])
    result = binary_calibration(y_true, probabilities, n_bins=5)
    assert result.ece == pytest.approx(0.0)
    assert result.mce == pytest.approx(0.0) or np.isnan(result.mce)


def test_maximally_overconfident_predictions_have_ece_one():
    """Certain and always wrong: confidence 1.0, accuracy 0.0, so ECE = MCE = 1."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    probabilities = np.where(y_true == 1, 0.0, 1.0).astype(float)
    result = binary_calibration(y_true, probabilities, n_bins=5, min_bin_count=1)
    assert result.ece == pytest.approx(1.0)
    assert result.mce == pytest.approx(1.0)


def test_ece_matches_a_hand_computed_two_bin_case():
    """10 samples, two occupied confidence bins.

    Confidences: six at 0.6, four at 0.9.
    - The 0.6 group: 3 of 6 correct -> accuracy 0.5, gap |0.5 - 0.6| = 0.1, weight 0.6
    - The 0.9 group: 4 of 4 correct -> accuracy 1.0, gap |1.0 - 0.9| = 0.1, weight 0.4
    ECE = 0.6 * 0.1 + 0.4 * 0.1 = 0.1
    """
    probabilities = np.array([0.6] * 6 + [0.9] * 4)
    # Within the 0.6 group make 3 right and 3 wrong; the 0.9 group all right.
    y_true = np.array([1, 1, 1, 0, 0, 0] + [1, 1, 1, 1])
    result = binary_calibration(
        y_true, probabilities, n_bins=10, min_bin_count=1, target="confidence"
    )
    assert result.ece == pytest.approx(0.1, abs=1e-9)
    assert result.mce == pytest.approx(0.1, abs=1e-9)
    assert result.bin_counts.sum() == 10


def test_mass_weighted_ece_differs_from_the_unweighted_bin_mean():
    """The regression guard for the bug this module exists to avoid.

    `DQNEvaluator._calculate_calibration_error` averaged per-bin gaps with equal
    weight per non-empty bin. Here 98 samples sit in one well-calibrated bin and 2
    in a badly-calibrated one:
    - bin A: 98 samples, gap 0.0, weight 0.98
    - bin B:  2 samples, gap 1.0, weight 0.02
    Mass-weighted ECE = 0.02. Unweighted bin mean = 0.5 -- a 25x overstatement of
    this model's miscalibration, in the opposite direction from the usual case.
    Either way the two are not interchangeable.
    """
    # 98 samples at confidence 1.0 and always right.
    calibrated_probs = np.ones(98)
    calibrated_labels = np.ones(98, dtype=int)
    # 2 samples at confidence 1.0 and always wrong.
    wrong_probs = np.ones(2)
    wrong_labels = np.zeros(2, dtype=int)

    probabilities = np.concatenate([calibrated_probs, wrong_probs * 0.55])
    y_true = np.concatenate([calibrated_labels, wrong_labels])
    result = binary_calibration(y_true, probabilities, n_bins=10, min_bin_count=1)

    occupied = result.bin_counts > 0
    unweighted = float(
        np.mean(np.abs(result.bin_accuracy[occupied] - result.bin_confidence[occupied]))
    )
    assert result.ece != pytest.approx(unweighted, abs=1e-6), (
        "mass-weighted ECE coincided with the unweighted bin mean, so this fixture no "
        "longer distinguishes the two definitions"
    )
    assert result.ece < unweighted, (
        f"the sparse badly-calibrated bin should carry little mass "
        f"(ece={result.ece}, unweighted={unweighted})"
    )


def test_quantile_strategy_produces_equal_mass_bins():
    rng = np.random.Generator(np.random.PCG64(0))
    probabilities = rng.uniform(0.5, 1.0, size=100)
    y_true = (rng.uniform(size=100) < probabilities).astype(int)
    result = binary_calibration(y_true, probabilities, n_bins=5, strategy="quantile")
    occupied = result.bin_counts[result.bin_counts > 0]
    assert occupied.max() - occupied.min() <= 2, f"bins are not equal-mass: {occupied}"


def test_quantile_strategy_collapses_when_all_scores_are_identical():
    y_true = np.array([1, 0, 1, 0])
    probabilities = np.full(4, 0.7)
    result = binary_calibration(y_true, probabilities, n_bins=5, strategy="quantile")
    assert DEGENERATE_SINGLE_BIN in result.status_flags
    assert DEGENERATE_CONSTANT_SCORE in result.status_flags


def test_positive_and_confidence_targets_are_distinct():
    """Named separately because the literature conflates them."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3])
    confidence = binary_calibration(y_true, probabilities, target="confidence")
    positive = binary_calibration(y_true, probabilities, target="positive")
    assert confidence.target == "confidence" and positive.target == "positive"
    assert confidence.ece != pytest.approx(positive.ece)


def test_empty_bins_are_counted_and_carry_no_weight():
    y_true = np.array([1, 1, 1, 1])
    probabilities = np.array([0.99, 0.99, 0.99, 0.99])
    result = binary_calibration(y_true, probabilities, n_bins=20, min_bin_count=1)
    assert result.n_empty_bins == 19
    assert result.bin_weight.sum() == pytest.approx(1.0)


def test_mce_requires_a_qualifying_bin():
    """An unweighted max over a one-sample bin is noise, not miscalibration."""
    y_true = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.5, 0.7])
    result = binary_calibration(y_true, probabilities, n_bins=10, min_bin_count=10)
    assert np.isnan(result.mce)
    assert MCE_NO_QUALIFYING_BIN in result.status_flags


def test_calibration_reports_single_class_labels():
    result = binary_calibration(np.ones(10, dtype=int), np.full(10, 0.8))
    assert SINGLE_CLASS_LABELS in result.status_flags


def test_calibration_with_too_few_samples():
    result = binary_calibration(np.array([1]), np.array([0.9]))
    assert INSUFFICIENT_SAMPLES in result.status_flags
    assert not result.applicable
    assert np.isnan(result.ece)


def test_reliability_diagram_data_includes_counts():
    result = reliability_diagram_data(
        np.array([1, 0, 1, 0, 1, 1]), np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.6])
    )
    assert result.bin_counts.sum() == 6
    assert result.bin_edges.size == result.bin_counts.size + 1


def test_invalid_target_and_strategy_rejected():
    with pytest.raises(ValueError, match="target"):
        binary_calibration(np.array([1, 0]), np.array([0.6, 0.4]), target="bogus")
    with pytest.raises(ValueError, match="strategy"):
        binary_calibration(np.array([1, 0]), np.array([0.6, 0.4]), strategy="bogus")


# --------------------------------------------------------------------------- #
# Brier / NLL
# --------------------------------------------------------------------------- #

def test_brier_score_hand_computed():
    """p=[0.8, 0.3], y=[1, 0] -> mean((0.8-1)^2, (0.3-0)^2) = (0.04 + 0.09)/2 = 0.065"""
    result = brier_score(np.array([1, 0]), np.array([0.8, 0.3]))
    assert result.value == pytest.approx(0.065)


def test_brier_score_is_zero_for_perfect_predictions():
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])).value == pytest.approx(0.0)


def test_nll_hand_computed():
    """p=0.5 for both samples -> -log(0.5) = 0.6931471805599453"""
    result = negative_log_likelihood(np.array([1, 0]), np.array([0.5, 0.5]))
    assert result.value == pytest.approx(0.6931471805599453, abs=1e-9)
    assert result.extra["nll_clipped_fraction"] == 0.0


def test_nll_reports_clipped_fraction():
    """Exact 0/1 probabilities must not hide behind clipping."""
    result = negative_log_likelihood(np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.5, 0.5]))
    assert result.extra["nll_clipped_fraction"] == pytest.approx(0.5)
    assert np.isfinite(result.value)


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #

def test_discrimination_metrics_on_a_separable_case():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    result, flags = discrimination_metrics(y_true, probabilities)
    assert flags == ()
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["eer"] == pytest.approx(0.0)


def test_discrimination_metrics_reports_single_class():
    result, flags = discrimination_metrics(np.ones(4, dtype=int), np.array([0.9, 0.8, 0.7, 0.6]))
    assert SINGLE_CLASS_LABELS in flags
    assert np.isnan(result["auroc"])
    assert result["accuracy"] == pytest.approx(1.0)  # accuracy is still defined


def test_discrimination_matches_the_existing_repo_metric():
    """Cross-check against models/metrics/base_metrics_class.calculate_metrics_for_train.

    That helper is torch-only and returns -1 sentinels, so it is not used at runtime,
    but agreeing with it documents that this module is not silently different.
    """
    torch = pytest.importorskip("torch")
    from models.metrics.base_metrics_class import calculate_metrics_for_train

    rng = np.random.Generator(np.random.PCG64(3))
    labels = rng.integers(0, 2, size=64)
    logits = rng.normal(size=64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    auc, eer, accuracy, average_precision = calculate_metrics_for_train(
        torch.tensor(labels), torch.tensor(logits)
    )
    result, _ = discrimination_metrics(labels, probabilities)
    assert result["auroc"] == pytest.approx(auc, abs=1e-6)
    assert result["accuracy"] == pytest.approx(accuracy, abs=1e-6)
    assert result["auprc"] == pytest.approx(average_precision, abs=1e-6)
    assert result["eer"] == pytest.approx(eer, abs=1e-6)


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #

def test_uncertainty_error_auroc_is_one_for_a_perfect_detector():
    """Uncertainty exactly ranks the errors above the correct predictions."""
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.9, 0.1])  # index 2 is wrong
    uncertainty = np.array([0.1, 0.1, 0.9, 0.2])
    result = uncertainty_error_auroc(y_true, probabilities, uncertainty)
    assert result.value == pytest.approx(1.0)
    assert result.extra["error_rate"] == pytest.approx(0.25)


def test_uncertainty_error_auroc_is_half_for_a_constant_score():
    """The flag that catches an identically-zero uncertainty signal.

    This is the symptom of the BatchEnsemble all-ones initialization and of MC
    dropout on a network whose dropout layers are all p=0. Returning 0.5 without the
    flag would look like a real (if useless) measurement.
    """
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.9, 0.1])
    result = uncertainty_error_auroc(y_true, probabilities, np.zeros(4))
    assert result.value == pytest.approx(0.5)
    assert DEGENERATE_CONSTANT_SCORE in result.status_flags


def test_uncertainty_error_auroc_when_everything_is_correct():
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    result = uncertainty_error_auroc(y_true, probabilities, np.array([0.1, 0.2, 0.3, 0.4]))
    assert SINGLE_CLASS_ERROR in result.status_flags
    assert not result.applicable
    assert np.isnan(result.value)


def test_nan_scores_are_dropped_not_imputed():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    probabilities = np.array([0.9, 0.1, 0.9, 0.1, 0.6, 0.4])
    uncertainty = np.array([0.1, 0.9, 0.8, 0.2, np.nan, np.nan])
    result = uncertainty_error_auroc(y_true, probabilities, uncertainty)
    assert NAN_SCORES_DROPPED in result.status_flags
    assert result.n == 4
    assert result.extra["n_dropped"] == 2


def test_aupr_error_reports_its_baseline():
    y_true = np.array([1, 1, 1, 0])
    probabilities = np.array([0.9, 0.9, 0.1, 0.1])  # one error
    result = aupr_error(y_true, probabilities, np.array([0.1, 0.2, 0.95, 0.3]))
    assert result.value == pytest.approx(1.0)
    assert result.extra["aupr_error_baseline"] == pytest.approx(0.25)


def test_risk_coverage_curve_shape_and_monotonicity():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.2, 0.1, 0.4, 0.6, 0.7, 0.3])
    uncertainty = np.array([0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.3, 0.3])
    result = risk_coverage_curve(y_true, probabilities, uncertainty)

    assert result.coverage.size == 8
    assert result.coverage[0] == pytest.approx(1 / 8)
    assert result.coverage[-1] == pytest.approx(1.0)
    assert 0.0 <= result.aurc <= 1.0
    assert result.eaurc >= -1e-12, "AURC cannot beat the oracle"


def test_eaurc_is_zero_for_an_oracle_ranking():
    """When uncertainty ranks errors perfectly, AURC equals the oracle's."""
    y_true = np.array([1, 1, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.9])  # last is wrong
    uncertainty = np.array([0.1, 0.2, 0.3, 0.99])
    result = risk_coverage_curve(y_true, probabilities, uncertainty)
    assert result.eaurc == pytest.approx(0.0, abs=1e-12)


def test_risk_coverage_with_no_errors():
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    result = risk_coverage_curve(y_true, probabilities, np.array([0.4, 0.3, 0.2, 0.1]))
    assert np.all(result.risk == 0.0)
    assert result.aurc == pytest.approx(0.0)
    assert result.eaurc == pytest.approx(0.0)


def test_accuracy_at_coverage_improves_with_a_useful_score():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.9])  # last is wrong
    uncertainty = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.99])
    result = accuracy_at_coverage(y_true, probabilities, uncertainty)
    assert result["accuracy_at_1"] == pytest.approx(7 / 8)
    assert result["accuracy_at_0.5"] == pytest.approx(1.0)


def test_accuracy_at_coverage_ties_are_broken_deterministically():
    y_true = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.1, 0.1, 0.9])  # last two are wrong
    uncertainty = np.full(4, 0.5)
    first = accuracy_at_coverage(y_true, probabilities, uncertainty)
    second = accuracy_at_coverage(y_true, probabilities, uncertainty)
    assert first == second


# --------------------------------------------------------------------------- #
# OOD
# --------------------------------------------------------------------------- #

def test_ood_detection_perfect_separation():
    result = ood_detection(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0]))
    assert result.auroc == pytest.approx(1.0)
    assert result.fpr_at_95_tpr == pytest.approx(0.0)
    assert result.n_id == 3 and result.n_ood == 3


def test_ood_detection_no_separation():
    rng = np.random.Generator(np.random.PCG64(1))
    scores = rng.normal(size=400)
    result = ood_detection(scores[:200], scores[200:])
    assert result.auroc == pytest.approx(0.5, abs=0.12)


def test_ood_detection_with_an_empty_partition():
    result = ood_detection(np.array([0.1, 0.2]), np.array([]))
    assert EMPTY_OOD_PARTITION in result.status_flags
    assert not result.applicable
    assert np.isnan(result.auroc)


def test_ood_detection_with_a_constant_score():
    result = ood_detection(np.zeros(5), np.zeros(5))
    assert result.auroc == pytest.approx(0.5)
    assert DEGENERATE_CONSTANT_SCORE in result.status_flags


def test_ood_detection_single_class_ood_is_fine():
    """Holding out a generator yields an all-fake OOD set; ranking still works."""
    result = ood_detection(np.array([0.1, 0.15, 0.2]), np.array([0.7, 0.75, 0.8]))
    assert result.applicable
    assert result.auroc > 0.9


# --------------------------------------------------------------------------- #
# Score transforms
# --------------------------------------------------------------------------- #

def test_score_helpers_agree_on_ordering():
    """entropy, 1-maxprob, and margin are all monotone in |p - 0.5| for a Bernoulli.

    Which is why entropy is rank-identical to max-probability and cannot be reported
    as a distinct ranking result.
    """
    probabilities = np.array([0.5, 0.6, 0.7, 0.9, 0.99])
    entropy_order = np.argsort(score_entropy(probabilities))
    maxprob_order = np.argsort(score_max_probability(probabilities))
    margin_order = np.argsort(score_margin(probabilities))
    assert np.array_equal(entropy_order, maxprob_order)
    assert np.array_equal(entropy_order, margin_order)


def test_score_entropy_is_maximal_at_one_half():
    assert score_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2), abs=1e-6)
    assert score_entropy(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.Generator(np.random.PCG64(7))
    y_true = rng.integers(0, 2, size=200)
    probabilities = np.clip(y_true + rng.normal(0, 0.4, size=200), 0.01, 0.99)

    point = brier_score(y_true, probabilities).value
    low, high = bootstrap_ci(
        lambda labels, probs: brier_score(labels, probs).value,
        y_true, probabilities, n_boot=200, seed=1,
    )
    assert low <= point <= high


def test_bootstrap_ci_is_reproducible():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.7, 0.3, 0.4])
    call = lambda: bootstrap_ci(
        lambda labels, probs: brier_score(labels, probs).value,
        y_true, probabilities, n_boot=100, seed=5,
    )
    assert call() == call()


def test_bootstrap_ci_with_too_few_samples():
    low, high = bootstrap_ci(lambda a, b: 0.0, np.array([1]), np.array([0.5]))
    assert np.isnan(low) and np.isnan(high)
