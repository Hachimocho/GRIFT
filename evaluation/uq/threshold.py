"""Decision-threshold fitting: post-hoc choice of the operating point.

The companion to `temperature.py`, and the distinction between them matters:

**Temperature scaling cannot fix a collapsed classifier.** It divides the logit by a
positive constant, which preserves the logit's *sign* -- so if every logit is positive,
every prediction stays in one class no matter what `T` is. Calibration in the ECE sense and
the choice of operating point are separate problems, and only a threshold (equivalently, a
bias shift) addresses the second.

That is the failure this module exists for. On the corrected AI-Face split (~87% fake) an
undertrained model ends up with a mean logit several units above the +1.95 the prior
justifies, while its class separation is only ~1.8 -- so the ranking is informative and
every prediction still lands on one side of 0.5. Accuracy reads as the majority-class prior
and balanced accuracy pins at exactly 0.5. Moving the threshold recovers the signal that is
already there: balanced accuracy 0.50 -> 0.68 on measured runs, with no retraining.

**Fit on validation, never on test.** Same discipline as temperature scaling: fitting the
threshold on the split you report is not choosing an operating point, it is fitting the
metric.

**Maximize balanced accuracy, not accuracy.** Tuning for accuracy on an 87%-positive split
pushes the threshold back toward predicting one class, because that is what maximizes
accuracy. The default objective is therefore balanced accuracy; Youden's J (sensitivity +
specificity - 1) is offered and is monotonically equivalent to it for a fixed dataset, so
both select the same threshold.

The search is exhaustive over the candidate thresholds implied by the data -- every
midpoint between adjacent distinct scores -- so it is deterministic, needs no optimizer, and
cannot land in a local optimum.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

#: Objectives the fit can maximize.
OBJECTIVES = ("balanced_accuracy", "youden_j", "accuracy")

#: Cap on the number of candidate thresholds evaluated. Above this the candidates are
#: sampled at even quantiles of the score, which for a smooth objective is
#: indistinguishable from exhaustive and keeps a 400k-row val split fast.
MAX_CANDIDATES = 2000

#: Returned when a fit is impossible, so downstream code has a threshold to apply either
#: way and the record of *why* it is 0.5 lives in the fit object.
DEFAULT_THRESHOLD = 0.5


class ThresholdError(ValueError):
    """Raised when a threshold cannot be fitted from the given data."""


@dataclass(frozen=True)
class ThresholdFit:
    """A fitted decision threshold and what it was worth on the fitting split."""

    threshold: float
    objective: str
    n_val: int
    n_positive: int
    #: Objective value at 0.5 and at the fitted threshold, on the fitting split.
    score_at_default: float
    score_at_threshold: float
    balanced_accuracy_at_default: float
    balanced_accuracy_at_threshold: float
    accuracy_at_default: float
    accuracy_at_threshold: float
    #: True when every val prediction at 0.5 fell in one class -- the condition this
    #: module exists to correct, worth recording so a reader knows it applied.
    collapsed_at_default: bool
    applicable: bool = True
    reason: str = ""
    fit_split: str = "val"
    records_sha256: Optional[str] = None

    @property
    def improved(self):
        return self.score_at_threshold >= self.score_at_default - 1e-12

    @property
    def is_default(self):
        return abs(self.threshold - DEFAULT_THRESHOLD) < 1e-12

    def as_dict(self):
        return asdict(self)


def _as_arrays(labels, probabilities):
    y = np.asarray(labels).reshape(-1).astype(int)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size != p.size:
        raise ThresholdError(
            f"labels and probabilities disagree on length: {y.size} vs {p.size}"
        )
    finite = np.isfinite(p)
    return y[finite], p[finite]


def _objective_value(y, p, threshold, objective):
    predictions = (p > threshold).astype(int)
    positive = y == 1
    negative = ~positive
    if objective == "accuracy":
        return float((predictions == y).mean())

    # Both remaining objectives need per-class rates.
    sensitivity = float((predictions[positive] == 1).mean()) if positive.any() else 0.0
    specificity = float((predictions[negative] == 0).mean()) if negative.any() else 0.0
    if objective == "youden_j":
        return sensitivity + specificity - 1.0
    return 0.5 * (sensitivity + specificity)


def _candidates(p):
    """Midpoints between adjacent distinct scores, plus the default.

    Midpoints rather than the scores themselves: `p > t` with `t` equal to an observed
    score would classify that sample by a floating-point tie.
    """
    unique = np.unique(p)
    if unique.size == 0:
        return np.array([DEFAULT_THRESHOLD])
    if unique.size > MAX_CANDIDATES:
        unique = np.unique(
            np.quantile(unique, np.linspace(0.0, 1.0, MAX_CANDIDATES))
        )
    midpoints = (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else np.array([])
    # Below the minimum and above the maximum, so "everything positive" and "everything
    # negative" are both reachable -- one of them is occasionally correct.
    edges = np.array([
        np.nextafter(unique[0], -np.inf), np.nextafter(unique[-1], np.inf),
    ])
    return np.unique(np.concatenate([midpoints, edges, [DEFAULT_THRESHOLD]]))


def fit_threshold(labels, probabilities, objective="balanced_accuracy",
                  records_sha256=None, fit_split="val"):
    """Fit the decision threshold that maximizes `objective`. Returns a `ThresholdFit`.

    Never raises for degenerate input: a single-class split has no meaningful operating
    point, so the fit comes back `applicable=False` at the default threshold with a reason.
    Callers can then apply `fit.threshold` unconditionally.
    """
    if objective not in OBJECTIVES:
        raise ThresholdError(
            f"unknown objective {objective!r}; choose from {', '.join(OBJECTIVES)}"
        )

    y, p = _as_arrays(labels, probabilities)

    def described(threshold, applicable, reason):
        return ThresholdFit(
            threshold=float(threshold),
            objective=objective,
            n_val=int(y.size),
            n_positive=int(y.sum()) if y.size else 0,
            score_at_default=_safe(y, p, DEFAULT_THRESHOLD, objective),
            score_at_threshold=_safe(y, p, threshold, objective),
            balanced_accuracy_at_default=_safe(y, p, DEFAULT_THRESHOLD, "balanced_accuracy"),
            balanced_accuracy_at_threshold=_safe(y, p, threshold, "balanced_accuracy"),
            accuracy_at_default=_safe(y, p, DEFAULT_THRESHOLD, "accuracy"),
            accuracy_at_threshold=_safe(y, p, threshold, "accuracy"),
            collapsed_at_default=(
                bool(np.unique((p > DEFAULT_THRESHOLD).astype(int)).size < 2)
                if p.size else False
            ),
            applicable=applicable,
            reason=reason,
            fit_split=fit_split,
            records_sha256=records_sha256,
        )

    if y.size == 0:
        return described(DEFAULT_THRESHOLD, False, "no finite probabilities to fit on")
    if np.unique(y).size < 2:
        return described(
            DEFAULT_THRESHOLD, False,
            f"the {fit_split} split contains only class {sorted(np.unique(y).tolist())}; "
            f"an operating point cannot be chosen without both classes",
        )

    candidates = _candidates(p)
    scores = np.array([
        _objective_value(y, p, threshold, objective) for threshold in candidates
    ])
    best = int(np.argmax(scores))
    # Ties broken toward the threshold nearest 0.5, so a fit that gains nothing stays
    # recognizably close to the default rather than drifting to an arbitrary extreme.
    tied = np.flatnonzero(scores >= scores[best] - 1e-12)
    best = int(tied[np.argmin(np.abs(candidates[tied] - DEFAULT_THRESHOLD))])
    return described(candidates[best], True, "")


def _safe(y, p, threshold, objective):
    if y.size == 0:
        return float("nan")
    try:
        return _objective_value(y, p, threshold, objective)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def fit_from_records(frame, objective="balanced_accuracy", records_sha256=None,
                     fit_split="val"):
    """`fit_threshold` over a record table's `label` and `prob` columns."""
    for column in ("label", "prob"):
        if column not in frame.columns:
            raise ThresholdError(f"record table has no {column!r} column")
    return fit_threshold(
        frame["label"].to_numpy(), frame["prob"].to_numpy(),
        objective=objective, records_sha256=records_sha256, fit_split=fit_split,
    )


def apply_to_records(frame, fit):
    """Return a copy with `pred` and `correct` recomputed at the fitted threshold.

    `prob` is untouched: the threshold is a decision rule, not a recalibration, so every
    ranking metric is unchanged by construction. A new `threshold` column records the
    operating point the predictions were taken at, so a table cannot be read as though it
    were thresholded at 0.5.
    """
    threshold = fit.threshold if hasattr(fit, "threshold") else float(fit)
    updated = frame.copy()
    predictions = (updated["prob"].to_numpy(dtype=float) > threshold).astype(int)
    updated["pred"] = predictions
    if "label" in updated.columns:
        updated["correct"] = (
            predictions == updated["label"].to_numpy(dtype=int)
        ).astype(int)
    updated["threshold"] = float(threshold)
    return updated


def save_fit(fit, path):
    """Write a fit as JSON. Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(fit.as_dict(), handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return path


def load_fit(path):
    with open(path) as handle:
        payload = json.load(handle)
    known = {field for field in ThresholdFit.__dataclass_fields__}
    return ThresholdFit(**{k: v for k, v in payload.items() if k in known})


__all__ = [
    "DEFAULT_THRESHOLD", "MAX_CANDIDATES", "OBJECTIVES", "ThresholdError",
    "ThresholdFit", "apply_to_records", "fit_from_records", "fit_threshold",
    "load_fit", "save_fit",
]
