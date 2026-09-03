"""Temperature scaling: post-hoc calibration by a single scalar.

Divides the logit by a learned `T > 0` fitted to minimize NLL on a held-out split.
Fully model-agnostic -- it needs only logits, so it applies to every detector,
including the one that cannot host an uncertainty head.

Two things to be clear about:

**It cannot change any ranking metric.** Dividing a logit by a positive constant is a
monotone transform of the score, so AUROC-of-error, AUPR-error, AURC, and
accuracy@coverage are *identical* to the untempered baseline by construction. Its
entire contribution is calibration -- ECE, NLL, Brier. The registry records this via
``rank_equivalent_to`` and the report collapses the duplicate ranking columns, because
printing three identical numbers invites the reader to infer a difference that does
not exist.

**It must be fit on validation data, never on test.** Fitting on the same split you
report is not calibration, it is fitting the metric.

Optimized with ``scipy.optimize.minimize_scalar`` over ``log T`` rather than a torch
LBFGS loop: deterministic, CPU-only, no autograd state to seed, and bit-reproducible.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from evaluation.uq.metrics import EPSILON, binary_calibration, negative_log_likelihood

#: Bounds on log(T). exp(-3) ~ 0.05 (sharpening), exp(3) ~ 20 (heavy smoothing).
LOG_T_BOUNDS = (-3.0, 3.0)


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    n_val: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    converged: bool
    n_iterations: int
    fit_split: str = "val"
    records_sha256: Optional[str] = None

    @property
    def improved_calibration(self):
        return self.nll_after <= self.nll_before + 1e-9

    def as_dict(self):
        return asdict(self)


def probabilities_to_logits(probabilities, eps=EPSILON):
    """Recover logits from probabilities.

    Records store both, but the probability column is the one guaranteed present, and
    for a temperature fit the two are interchangeable up to clipping.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def apply_temperature(logits, temperature):
    """Temperature-scaled probabilities."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float).reshape(-1) / temperature))


def _nll(labels, logits, temperature):
    probabilities = np.clip(apply_temperature(logits, temperature), EPSILON, 1 - EPSILON)
    return float(
        -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    )


def fit_temperature(labels, logits=None, probabilities=None, records_sha256=None,
                    fit_split="val"):
    """Fit ``T`` by minimizing NLL on the supplied (validation) data."""
    from scipy import optimize

    labels = np.asarray(labels, dtype=float).reshape(-1)
    if logits is None:
        if probabilities is None:
            raise ValueError("fit_temperature needs either logits or probabilities")
        logits = probabilities_to_logits(probabilities)
    logits = np.asarray(logits, dtype=float).reshape(-1)

    if labels.size < 2 or len(np.unique(labels)) < 2:
        # Undefined: with one class the NLL is minimized by pushing T toward a
        # degenerate extreme. Return the identity rather than a meaningless fit.
        before = _nll(labels, logits, 1.0)
        return TemperatureFit(
            temperature=1.0, n_val=int(labels.size), nll_before=before, nll_after=before,
            ece_before=float("nan"), ece_after=float("nan"), converged=False,
            n_iterations=0, fit_split=fit_split, records_sha256=records_sha256,
        )

    # Optimize over log T so the parameter is unconstrained and T stays positive.
    result = optimize.minimize_scalar(
        lambda log_t: _nll(labels, logits, float(np.exp(log_t))),
        bounds=LOG_T_BOUNDS, method="bounded",
    )
    temperature = float(np.exp(result.x))

    before_probs = apply_temperature(logits, 1.0)
    after_probs = apply_temperature(logits, temperature)
    return TemperatureFit(
        temperature=temperature,
        n_val=int(labels.size),
        nll_before=negative_log_likelihood(labels, before_probs).value,
        nll_after=negative_log_likelihood(labels, after_probs).value,
        ece_before=binary_calibration(labels, before_probs).ece,
        ece_after=binary_calibration(labels, after_probs).ece,
        converged=bool(result.success),
        n_iterations=int(getattr(result, "nit", 0) or 0),
        fit_split=fit_split,
        records_sha256=records_sha256,
    )


def fit_from_records(val_frame, records_sha256=None):
    """Fit ``T`` from a validation record table."""
    labels = val_frame["label"].to_numpy(dtype=float)
    if "logit" in val_frame.columns and val_frame["logit"].notna().all():
        logits = val_frame["logit"].to_numpy(dtype=float)
    else:
        logits = probabilities_to_logits(val_frame["prob"].to_numpy(dtype=float))
    return fit_temperature(labels, logits=logits, records_sha256=records_sha256)


def apply_to_records(frame, fit, method_id="temperature_scaling"):
    """Derive a temperature-scaled copy of a record table.

    Adds the recalibrated probability, its max-prob uncertainty column, and the
    temperature used, so the result flows through the ordinary scoring path as just
    another method.
    """
    from evaluation.uq.metrics import score_max_probability

    scaled = frame.copy()
    if "logit" in scaled.columns and scaled["logit"].notna().all():
        logits = scaled["logit"].to_numpy(dtype=float)
    else:
        logits = probabilities_to_logits(scaled["prob"].to_numpy(dtype=float))

    probabilities = apply_temperature(logits, fit.temperature)
    scaled["prob"] = probabilities
    scaled["pred"] = (probabilities > 0.5).astype(int)
    scaled["correct"] = (scaled["pred"] == scaled["label"]).astype(int)
    scaled["u_temp_maxprob"] = score_max_probability(probabilities)
    scaled["temperature"] = fit.temperature
    scaled["method_id"] = method_id
    return scaled


def save_fit(fit, path):
    """Persist a fit so the temperature used is auditable."""
    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(fit.as_dict(), handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path


def load_fit(path):
    with open(str(path)) as handle:
        return TemperatureFit(**json.load(handle))
