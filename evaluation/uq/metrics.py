"""Uncertainty-quantification metrics.

Pure numpy + sklearn, deliberately no torch import: the metric suite is then
usable from any environment and its tests stay fast.

Two design rules run through this module.

**Calibration is measured on probabilities; ranking is measured on scores.**
Calibration metrics (ECE, MCE, Brier, NLL) only make sense for a calibrated
probability. Several methods here produce an uncertainty *score* with no
probabilistic interpretation at all -- graph distance, for instance -- so their
calibration cells are structurally N/A rather than zero. Every result carries an
``applicable`` flag for exactly this reason; treating N/A as 0 is the most common
way a UQ comparison table becomes nonsense.

**Cross-method comparisons use rank-based metrics only.** The in-tree scores live
on wildly different scales (``evidential_vacuity`` in (0, 1], ``mc_dropout_variance``
in [0, 0.25], ``sngp_variance`` unbounded, ``hybrid_distance`` in roughly [0, 2]).
Anything used to compare methods -- AUROC, AUPR-error, AURC/E-AURC,
accuracy@coverage, OOD AUROC -- is therefore invariant to any monotone rescaling.

Degenerate inputs return an explicit ``status_flags`` entry rather than a
plausible-looking number. ``degenerate_constant_score`` in particular is what
catches an uncertainty signal that is identically zero -- the symptom of the
BatchEnsemble initialization bug and of MC dropout on a network whose dropout
layers are all ``p=0``.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from sklearn import metrics as sk_metrics
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - sklearn is a hard dependency in practice
    _SKLEARN_AVAILABLE = False

EPSILON = 1e-7

# Status flags. Any of these means "do not read the number as a measurement".
SINGLE_CLASS_LABELS = "single_class_labels"
SINGLE_CLASS_ERROR = "single_class_error"
#: The model emitted one class for every sample, so its accuracy *is* the majority-class
#: prior. Distinct from SINGLE_CLASS_LABELS (the evaluation set has one class) and from
#: DEGENERATE_CONSTANT_SCORE (the uncertainty score is constant): here the labels are both
#: present and the scores vary, so neither of those fires, and the accuracy looks fine.
SINGLE_CLASS_PREDICTIONS = "single_class_predictions"
DEGENERATE_CONSTANT_SCORE = "degenerate_constant_score"
DEGENERATE_SINGLE_BIN = "degenerate_single_bin"
INSUFFICIENT_SAMPLES = "insufficient_samples"
NAN_SCORES_DROPPED = "nan_scores_dropped"
EMPTY_OOD_PARTITION = "empty_ood_partition"
MCE_NO_QUALIFYING_BIN = "mce_no_qualifying_bin"
NOT_PROBABILISTIC = "not_probabilistic"


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationResult:
    n: int
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_weight: np.ndarray
    ece: float
    mce: float
    n_empty_bins: int
    target: str
    strategy: str
    status_flags: Tuple[str, ...] = ()
    applicable: bool = True


@dataclass(frozen=True)
class ScoreResult:
    value: float
    n: int
    status_flags: Tuple[str, ...] = ()
    applicable: bool = True
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskCoverageResult:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float
    aurc_oracle: float
    eaurc: float
    n: int
    status_flags: Tuple[str, ...] = ()
    applicable: bool = True


@dataclass(frozen=True)
class OODResult:
    auroc: float
    aupr_in: float
    aupr_out: float
    fpr_at_95_tpr: float
    n_id: int
    n_ood: int
    status_flags: Tuple[str, ...] = ()
    applicable: bool = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_arrays(*arrays):
    return tuple(np.asarray(array).reshape(-1) for array in arrays)


def _drop_nan_scores(scores, *aligned):
    """Drop rows whose score is NaN. Never imputes -- a fabricated score is worse."""
    finite = np.isfinite(scores)
    dropped = int((~finite).sum())
    return (scores[finite],) + tuple(array[finite] for array in aligned) + (dropped,)


def _confidence_and_correctness(y_true, probabilities, threshold=0.5):
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predictions = (probabilities > threshold).astype(int)
    correct = (predictions == y_true.astype(int)).astype(float)
    return confidence, correct


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def binary_calibration(
    y_true, probabilities, n_bins=15, strategy="uniform",
    min_bin_count=10, target="confidence",
):
    """Binned calibration error with explicit per-bin counts.

    ``ece`` is **mass-weighted**: ``sum_i (n_i / n) * |acc_i - conf_i|``. This is the
    definition; averaging the per-bin gaps with equal weight per non-empty bin (as
    ``sklearn.calibration.calibration_curve`` invites you to do) systematically
    understates miscalibration, because sparse tail bins get the same influence as
    a bin holding most of the data.

    ``target`` selects which reliability is being measured, and both are reported
    separately elsewhere because the literature conflates them:

    ``confidence``  top-label reliability: is ``max(p, 1-p)`` the probability of
                    being right?
    ``positive``    positive-class reliability: is ``p`` the probability of y=1?

    ``mce`` is the maximum gap over bins holding at least ``min_bin_count`` samples;
    an unweighted maximum over a one-sample bin measures noise, not calibration.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    flags = []
    n = int(y_true.size)

    if n < 2:
        empty = np.zeros(0)
        return CalibrationResult(
            n=n, bin_edges=empty, bin_counts=empty, bin_confidence=empty,
            bin_accuracy=empty, bin_weight=empty, ece=float("nan"), mce=float("nan"),
            n_empty_bins=0, target=target, strategy=strategy,
            status_flags=(INSUFFICIENT_SAMPLES,), applicable=False,
        )

    if target == "confidence":
        scores, outcomes = _confidence_and_correctness(y_true, probabilities)
    elif target == "positive":
        scores, outcomes = probabilities, y_true.astype(float)
    else:
        raise ValueError(f"target must be 'confidence' or 'positive', got {target!r}")

    if strategy == "uniform":
        low = 0.5 if target == "confidence" else 0.0
        edges = np.linspace(low, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(scores, quantiles))
        if edges.size < 2:
            # Every score identical: one bin is the honest answer.
            edges = np.array([scores[0] - EPSILON, scores[0] + EPSILON])
            flags.append(DEGENERATE_SINGLE_BIN)
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    # right=True on all but the first bin, so the top edge is inclusive.
    indices = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, edges.size - 2)

    bin_count = edges.size - 1
    counts = np.zeros(bin_count)
    confidences = np.zeros(bin_count)
    accuracies = np.zeros(bin_count)
    for index in range(bin_count):
        mask = indices == index
        counts[index] = mask.sum()
        if counts[index]:
            confidences[index] = scores[mask].mean()
            accuracies[index] = outcomes[mask].mean()

    weights = counts / n
    gaps = np.abs(accuracies - confidences)
    ece = float(np.sum(weights * gaps))

    qualifying = counts >= min_bin_count
    if qualifying.any():
        mce = float(gaps[qualifying].max())
    else:
        mce = float("nan")
        flags.append(MCE_NO_QUALIFYING_BIN)

    if np.ptp(scores) == 0:
        flags.append(DEGENERATE_CONSTANT_SCORE)
    if len(np.unique(y_true)) < 2:
        flags.append(SINGLE_CLASS_LABELS)

    return CalibrationResult(
        n=n, bin_edges=edges, bin_counts=counts, bin_confidence=confidences,
        bin_accuracy=accuracies, bin_weight=weights, ece=ece, mce=mce,
        n_empty_bins=int((counts == 0).sum()), target=target, strategy=strategy,
        status_flags=tuple(flags), applicable=True,
    )


def reliability_diagram_data(y_true, probabilities, n_bins=15, **kwargs):
    """Bins plus their counts, for plotting.

    The counts are part of the contract: a reliability diagram without its bin
    histogram hides how much data each point represents, and makes an empty-bin
    artifact look like real miscalibration.
    """
    return binary_calibration(y_true, probabilities, n_bins=n_bins, **kwargs)


def brier_score(y_true, probabilities):
    y_true, probabilities = _as_arrays(y_true, probabilities)
    if y_true.size == 0:
        return ScoreResult(float("nan"), 0, (INSUFFICIENT_SAMPLES,), applicable=False)
    value = float(np.mean((probabilities - y_true.astype(float)) ** 2))
    return ScoreResult(value, int(y_true.size))


def negative_log_likelihood(y_true, probabilities, eps=EPSILON):
    """Mean NLL, reporting how much probability mass had to be clipped.

    The clipped fraction matters: a model that outputs exactly 0 or 1 would
    otherwise produce an infinite loss silently rescued by clipping, hiding
    overconfidence behind a finite-looking number.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    if y_true.size == 0:
        return ScoreResult(float("nan"), 0, (INSUFFICIENT_SAMPLES,), applicable=False)

    clipped_mask = (probabilities <= eps) | (probabilities >= 1.0 - eps)
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    labels = y_true.astype(float)
    value = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    return ScoreResult(
        value, int(y_true.size),
        extra={"nll_clipped_fraction": float(clipped_mask.mean())},
    )


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #

def discrimination_metrics(y_true, probabilities, threshold=0.5):
    """Accuracy, balanced accuracy, AUROC, average precision, and EER.

    `threshold` is the decision boundary the thresholded metrics (accuracy, balanced
    accuracy) are taken at; AUROC, AUPRC, and EER are threshold-free and unaffected.
    Defaults to 0.5, so an omitted argument reproduces the previous numbers exactly. A
    fitted value comes from `evaluation/uq/threshold.py` and must have been fitted on
    validation data.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    labels = y_true.astype(int)
    n = int(labels.size)
    result = {
        "accuracy": float("nan"), "balanced_accuracy": float("nan"),
        "auroc": float("nan"), "auprc": float("nan"), "eer": float("nan"),
        "n": n, "n_positive": int(labels.sum()),
    }
    if n == 0:
        return result, (INSUFFICIENT_SAMPLES,)

    predictions = (probabilities > threshold).astype(int)
    result["threshold"] = float(threshold)
    result["accuracy"] = float((predictions == labels).mean())

    if len(np.unique(labels)) < 2:
        return result, (SINGLE_CLASS_LABELS,)

    positive = labels == 1
    result["balanced_accuracy"] = float(
        0.5 * ((predictions[positive] == 1).mean() + (predictions[~positive] == 0).mean())
    )

    flags = ()
    # A model that emits one class for every sample scores exactly the majority-class
    # prior, which on this dataset is a respectable-looking 0.87 and says nothing at all.
    # Labels are two-class here and the scores vary, so neither SINGLE_CLASS_LABELS nor
    # DEGENERATE_CONSTANT_SCORE fires -- the row came back `ok` with no flag, which is how
    # a collapsed classifier gets promoted as a baseline. Balanced accuracy is pinned to
    # 0.5 whenever this happens, but only if someone reads that column.
    if len(np.unique(predictions)) < 2:
        flags = (SINGLE_CLASS_PREDICTIONS,)

    if _SKLEARN_AVAILABLE:
        false_positive, true_positive, _ = sk_metrics.roc_curve(labels, probabilities)
        result["auroc"] = float(sk_metrics.auc(false_positive, true_positive))
        result["auprc"] = float(sk_metrics.average_precision_score(labels, probabilities))
        false_negative = 1 - true_positive
        result["eer"] = float(
            false_positive[np.nanargmin(np.abs(false_negative - false_positive))]
        )
    return result, flags


# --------------------------------------------------------------------------- #
# Selective prediction / ranking
# --------------------------------------------------------------------------- #

def _error_labels(y_true, probabilities, threshold=0.5):
    """Which samples the model got wrong, at the given decision threshold.

    Threaded rather than hardcoded because "error" is defined by the operating point: once
    a threshold is fitted, selective prediction and uncertainty-error ranking have to be
    measured against the mistakes the model actually makes. Defaults to 0.5, so every
    existing number is unchanged unless a threshold is passed explicitly.
    """
    predictions = (probabilities > threshold).astype(int)
    return (predictions != y_true.astype(int)).astype(int)


def uncertainty_error_auroc(y_true, probabilities, uncertainty, threshold=0.5):
    """AUROC of uncertainty as a detector of the model's own mistakes.

    Rank-based, so invariant to any monotone rescaling of ``uncertainty`` -- which is
    what makes it valid for comparing methods on incompatible scales.
    """
    y_true, probabilities, uncertainty = _as_arrays(y_true, probabilities, uncertainty)
    uncertainty, y_true, probabilities, dropped = _drop_nan_scores(
        uncertainty, y_true, probabilities
    )
    flags = [NAN_SCORES_DROPPED] if dropped else []
    n = int(uncertainty.size)

    if n < 2:
        return ScoreResult(float("nan"), n, tuple(flags + [INSUFFICIENT_SAMPLES]), False)

    errors = _error_labels(y_true, probabilities, threshold=threshold)
    if len(np.unique(errors)) < 2:
        # All correct or all wrong: nothing to discriminate.
        return ScoreResult(
            float("nan"), n, tuple(flags + [SINGLE_CLASS_ERROR]), applicable=False,
            extra={"error_rate": float(errors.mean()), "n_dropped": float(dropped)},
        )
    if np.ptp(uncertainty) == 0:
        return ScoreResult(
            0.5, n, tuple(flags + [DEGENERATE_CONSTANT_SCORE]), applicable=True,
            extra={"error_rate": float(errors.mean()), "n_dropped": float(dropped)},
        )

    value = float(sk_metrics.roc_auc_score(errors, uncertainty))
    return ScoreResult(
        value, n, tuple(flags),
        extra={"error_rate": float(errors.mean()), "n_dropped": float(dropped)},
    )


def aupr_error(y_true, probabilities, uncertainty, threshold=0.5):
    """Average precision for detecting errors, with the base rate as the baseline."""
    y_true, probabilities, uncertainty = _as_arrays(y_true, probabilities, uncertainty)
    uncertainty, y_true, probabilities, dropped = _drop_nan_scores(
        uncertainty, y_true, probabilities
    )
    flags = [NAN_SCORES_DROPPED] if dropped else []
    n = int(uncertainty.size)
    if n < 2:
        return ScoreResult(float("nan"), n, tuple(flags + [INSUFFICIENT_SAMPLES]), False)

    errors = _error_labels(y_true, probabilities, threshold=threshold)
    baseline = float(errors.mean())
    if len(np.unique(errors)) < 2:
        return ScoreResult(
            float("nan"), n, tuple(flags + [SINGLE_CLASS_ERROR]), applicable=False,
            extra={"aupr_error_baseline": baseline},
        )
    value = float(sk_metrics.average_precision_score(errors, uncertainty))
    return ScoreResult(
        value, n, tuple(flags), extra={"aupr_error_baseline": baseline}
    )


def risk_coverage_curve(y_true, probabilities, uncertainty, threshold=0.5):
    """Risk as a function of coverage, plus AURC and E-AURC.

    Samples are abstained on in decreasing order of uncertainty. ``eaurc`` is
    ``aurc`` minus the AURC of an oracle that abstains on the actual errors first.

    **Report E-AURC, not raw AURC, for cross-method or cross-detector comparison.**
    Raw AURC is dominated by the base error rate, so a stronger-but-worse-calibrated
    detector can appear better purely because it makes fewer mistakes overall.
    """
    y_true, probabilities, uncertainty = _as_arrays(y_true, probabilities, uncertainty)
    uncertainty, y_true, probabilities, dropped = _drop_nan_scores(
        uncertainty, y_true, probabilities
    )
    flags = [NAN_SCORES_DROPPED] if dropped else []
    n = int(uncertainty.size)
    if n < 2:
        empty = np.zeros(0)
        return RiskCoverageResult(
            empty, empty, float("nan"), float("nan"), float("nan"), n,
            tuple(flags + [INSUFFICIENT_SAMPLES]), applicable=False,
        )

    errors = _error_labels(y_true, probabilities, threshold=threshold)
    if np.ptp(uncertainty) == 0:
        flags.append(DEGENERATE_CONSTANT_SCORE)

    # Ties broken by original index, so the curve is deterministic.
    order = np.lexsort((np.arange(n), uncertainty))
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors[order]) / np.arange(1, n + 1)
    aurc = float(np.mean(risk))

    oracle_order = np.lexsort((np.arange(n), errors))
    oracle_risk = np.cumsum(errors[oracle_order]) / np.arange(1, n + 1)
    aurc_oracle = float(np.mean(oracle_risk))

    return RiskCoverageResult(
        coverage=coverage, risk=risk, aurc=aurc, aurc_oracle=aurc_oracle,
        eaurc=float(aurc - aurc_oracle), n=n, status_flags=tuple(flags),
    )


def accuracy_at_coverage(
    y_true, probabilities, uncertainty, coverages=(0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
    threshold=0.5,
):
    """Accuracy on the most-confident fraction of the data, per coverage level."""
    y_true, probabilities, uncertainty = _as_arrays(y_true, probabilities, uncertainty)
    uncertainty, y_true, probabilities, _ = _drop_nan_scores(
        uncertainty, y_true, probabilities
    )
    n = int(uncertainty.size)
    if n == 0:
        return {f"accuracy_at_{level:g}": float("nan") for level in coverages}

    errors = _error_labels(y_true, probabilities, threshold=threshold)
    order = np.lexsort((np.arange(n), uncertainty))
    ordered_errors = errors[order]

    result = {}
    for level in coverages:
        keep = max(1, int(round(level * n)))
        result[f"accuracy_at_{level:g}"] = float(1.0 - ordered_errors[:keep].mean())
    return result


# --------------------------------------------------------------------------- #
# Distribution shift
# --------------------------------------------------------------------------- #

def ood_detection(uncertainty_id, uncertainty_ood):
    """Can uncertainty separate in-distribution from out-of-distribution samples?

    A ranking task, so a single-class OOD partition is fine here -- which matters,
    because holding out a generator from AI-Face yields an all-fake OOD set. What is
    *not* fine is computing classification AUROC on such a set; that is refused
    elsewhere.
    """
    uncertainty_id, uncertainty_ood = _as_arrays(uncertainty_id, uncertainty_ood)
    uncertainty_id = uncertainty_id[np.isfinite(uncertainty_id)]
    uncertainty_ood = uncertainty_ood[np.isfinite(uncertainty_ood)]
    n_id, n_ood = int(uncertainty_id.size), int(uncertainty_ood.size)

    if n_id == 0 or n_ood == 0:
        return OODResult(
            float("nan"), float("nan"), float("nan"), float("nan"), n_id, n_ood,
            (EMPTY_OOD_PARTITION,), applicable=False,
        )

    scores = np.concatenate([uncertainty_id, uncertainty_ood])
    is_ood = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

    flags = []
    if np.ptp(scores) == 0:
        return OODResult(
            0.5, float("nan"), float("nan"), float("nan"), n_id, n_ood,
            (DEGENERATE_CONSTANT_SCORE,), applicable=True,
        )

    auroc = float(sk_metrics.roc_auc_score(is_ood, scores))
    aupr_out = float(sk_metrics.average_precision_score(is_ood, scores))
    aupr_in = float(sk_metrics.average_precision_score(1 - is_ood, -scores))

    # FPR at the threshold achieving 95% TPR on the OOD (positive) class.
    false_positive, true_positive, _ = sk_metrics.roc_curve(is_ood, scores)
    reached = np.searchsorted(true_positive, 0.95, side="left")
    fpr_at_95 = float(false_positive[min(reached, false_positive.size - 1)])

    return OODResult(auroc, aupr_in, aupr_out, fpr_at_95, n_id, n_ood, tuple(flags))


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #

def bootstrap_ci(function, *arrays, n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for a scalar statistic.

    Seeded, so a reported interval is reproducible. Without error bars, no
    "method A beats method B" claim is defensible.
    """
    arrays = _as_arrays(*arrays)
    n = arrays[0].size
    if n < 2:
        return (float("nan"), float("nan"))

    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(n_boot)
    for index in range(n_boot):
        picks = rng.integers(0, n, size=n)
        try:
            value = function(*(array[picks] for array in arrays))
        except Exception:
            value = np.nan
        samples[index] = value.value if isinstance(value, ScoreResult) else value

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
    )


def score_entropy(probabilities):
    """Binary predictive entropy, the zero-cost baseline uncertainty score.

    Note this is a monotone function of ``max(p, 1-p)`` for a single Bernoulli, so it
    is *rank-identical* to the max-probability baseline: identical AUROC-error,
    AURC, and accuracy@coverage. Reported as a separate method only for calibration
    purposes; see ``registry.rank_equivalent_to``.
    """
    probabilities = np.clip(np.asarray(probabilities).reshape(-1), EPSILON, 1 - EPSILON)
    return -(
        probabilities * np.log(probabilities)
        + (1 - probabilities) * np.log(1 - probabilities)
    )


def score_max_probability(probabilities):
    """Uncertainty as ``1 - max(p, 1-p)``, i.e. higher means less confident."""
    probabilities = np.asarray(probabilities).reshape(-1)
    return 1.0 - np.maximum(probabilities, 1.0 - probabilities)


def score_margin(probabilities):
    """Uncertainty as ``1 - |2p - 1|``: distance from the decision boundary."""
    probabilities = np.asarray(probabilities).reshape(-1)
    return 1.0 - np.abs(2.0 * probabilities - 1.0)
