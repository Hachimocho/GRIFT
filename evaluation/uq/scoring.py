"""Turn per-sample record tables into a per-method results table.

Output is **tidy/long**: one row per
``(detector, method, score_column, holdout, domain, corruption, severity)`` with the
metrics as columns. Long format means adding a metric never changes the schema, and
pivots for paper tables are a one-liner.

Three refusals are built in, because each corresponds to a way a comparison table
can look fine and be wrong:

* **Sub-99% coverage.** ``evaluate_model``'s per-batch handler can skip batches, so a
  headline number might have been computed on a fraction of the data. Coverage is
  recorded per cell and low-coverage cells are refused rather than footnoted.
* **Mixed provenance.** Cells produced under different determinism modes, evaluation
  manifests, or graph-normalization statistics are not comparable, and pooling them
  would silently average incommensurable numbers.
* **Calibration on a non-probabilistic score.** Graph-distance methods have no
  calibrated probability, so their ECE/Brier/NLL cells are N/A -- never 0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from evaluation.uq.metrics import (
    DEGENERATE_CONSTANT_SCORE, SINGLE_CLASS_LABELS, accuracy_at_coverage, aupr_error,
    binary_calibration, bootstrap_ci, brier_score, discrimination_metrics,
    negative_log_likelihood, ood_detection, risk_coverage_curve, score_entropy,
    score_margin, score_max_probability, uncertainty_error_auroc,
)
from evaluation.uq.registry import method_spec, validate_score_column

MIN_COVERAGE = 0.99

#: Uncertainty scores derivable from the probability alone, so they need no extra
#: columns in the record table.
DERIVED_SCORES = {
    "u_maxprob": score_max_probability,
    "u_entropy": score_entropy,
    "u_margin": score_margin,
    "u_temp_maxprob": score_max_probability,
}


class IncomparableCellsError(ValueError):
    """Raised when cells that must not be pooled are asked to be aggregated."""


@dataclass
class Cell:
    """One scored (detector, method, condition) combination."""

    detector: str
    method_id: str
    score_column: str
    frame: object
    holdout: str = "none"
    corruption: str = "none"
    severity: int = 0
    coverage: float = 1.0
    determinism_mode: str = "unknown"
    manifest_sha256: Optional[str] = None
    graph_norm_sha256: Optional[str] = None
    seed: Optional[int] = None
    cost_forward_passes: object = 1
    cost_training_runs: int = 0
    #: Decision threshold the thresholded metrics are taken at. 0.5 reproduces the
    #: original numbers exactly; a fitted value comes from `evaluation/uq/threshold.py`
    #: and must have been fitted on validation data, never on the split being scored.
    #: It changes accuracy, balanced accuracy, *and* the definition of an "error" -- so
    #: selective prediction and uncertainty-error ranking move with it too, which is
    #: correct: those measure the mistakes the model actually makes at its operating point.
    threshold: float = 0.5
    extra: Dict[str, object] = field(default_factory=dict)


def resolve_score(frame, score_column):
    """Get a score array for ``score_column``, deriving it from ``prob`` if possible."""
    validate_score_column(score_column)
    if score_column in frame.columns:
        return frame[score_column].to_numpy(dtype=float)
    if score_column in DERIVED_SCORES:
        return DERIVED_SCORES[score_column](frame["prob"].to_numpy(dtype=float))
    raise KeyError(
        f"score column {score_column!r} is not in the record table and cannot be "
        f"derived from `prob`; available uncertainty columns: "
        f"{sorted(column for column in frame.columns if column.startswith('u_'))}"
    )


def score_cell(cell, n_boot=0, bootstrap_seed=42):
    """Compute every applicable metric for one cell. Returns a flat dict."""
    spec = method_spec(cell.method_id)
    frame = cell.frame

    if cell.coverage < MIN_COVERAGE:
        return {
            **_identity(cell, spec),
            "status": "refused_low_coverage",
            "status_flags": f"coverage={cell.coverage:.4f}<{MIN_COVERAGE}",
            "n": len(frame),
        }

    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["prob"].to_numpy(dtype=float)
    scores = resolve_score(frame, cell.score_column)

    row = {**_identity(cell, spec), "status": "ok", "n": int(len(frame))}
    flags = set()

    discrimination, discrimination_flags = discrimination_metrics(
        labels, probabilities, threshold=cell.threshold
    )
    flags.update(discrimination_flags)
    row.update({f"clf_{name}": value for name, value in discrimination.items() if name != "n"})

    # Ranking metrics -- valid for every method, since they depend only on the
    # ordering of the score.
    error_auroc = uncertainty_error_auroc(
        labels, probabilities, scores, threshold=cell.threshold
    )
    row["auroc_error"] = error_auroc.value
    row["error_rate"] = error_auroc.extra.get("error_rate", np.nan)
    flags.update(error_auroc.status_flags)

    error_aupr = aupr_error(labels, probabilities, scores, threshold=cell.threshold)
    row["aupr_error"] = error_aupr.value
    row["aupr_error_baseline"] = error_aupr.extra.get("aupr_error_baseline", np.nan)
    flags.update(error_aupr.status_flags)

    risk_coverage = risk_coverage_curve(
        labels, probabilities, scores, threshold=cell.threshold
    )
    row["aurc"] = risk_coverage.aurc
    row["aurc_oracle"] = risk_coverage.aurc_oracle
    row["eaurc"] = risk_coverage.eaurc
    flags.update(risk_coverage.status_flags)

    row.update(accuracy_at_coverage(
        labels, probabilities, scores, threshold=cell.threshold
    ))

    # Calibration -- only meaningful for a method that produces a probability.
    if spec.produces_probabilities:
        confidence = binary_calibration(labels, probabilities, target="confidence")
        positive = binary_calibration(labels, probabilities, target="positive")
        adaptive = binary_calibration(
            labels, probabilities, target="confidence", strategy="quantile"
        )
        row.update({
            "ece_confidence": confidence.ece,
            "ece_confidence_adaptive": adaptive.ece,
            "ece_positive": positive.ece,
            "mce_confidence": confidence.mce,
            "n_empty_bins": confidence.n_empty_bins,
            "brier": brier_score(labels, probabilities).value,
        })
        nll = negative_log_likelihood(labels, probabilities)
        row["nll"] = nll.value
        row["nll_clipped_fraction"] = nll.extra.get("nll_clipped_fraction", np.nan)
        row["calibration_applicable"] = True
        flags.update(confidence.status_flags)
    else:
        # N/A, not zero: this method has no calibrated probability to be right about.
        for name in (
            "ece_confidence", "ece_confidence_adaptive", "ece_positive",
            "mce_confidence", "brier", "nll", "nll_clipped_fraction",
        ):
            row[name] = np.nan
        row["calibration_applicable"] = False
        flags.add("not_probabilistic")

    row["score_min"] = float(np.nanmin(scores)) if scores.size else np.nan
    row["score_median"] = float(np.nanmedian(scores)) if scores.size else np.nan
    row["score_max"] = float(np.nanmax(scores)) if scores.size else np.nan
    row["score_std"] = float(np.nanstd(scores)) if scores.size else np.nan

    if n_boot:
        low, high = bootstrap_ci(
            lambda label_sample, probability_sample, score_sample: uncertainty_error_auroc(
                label_sample, probability_sample, score_sample
            ).value,
            labels, probabilities, scores, n_boot=n_boot, seed=bootstrap_seed,
        )
        row["auroc_error_ci_low"] = low
        row["auroc_error_ci_high"] = high

    row["status_flags"] = ";".join(sorted(flags))
    if DEGENERATE_CONSTANT_SCORE in flags:
        # Loud, because it means the uncertainty signal is identically constant --
        # the signature of an un-diversified ensemble or zero-p MC dropout.
        row["status"] = "degenerate"
    return row


def _identity(cell, spec):
    return {
        # The source table's label. Two cells can share (detector, method_id) and still
        # be different measurements -- max-probability scored on one member's records
        # and on an ensemble's averaged records is the clearest case -- so without this
        # the tidy table has rows that are indistinguishable but not equal.
        "label": cell.extra.get("label", ""),
        "detector": cell.detector,
        "method_id": cell.method_id,
        "method_family": spec.family,
        "score_column": cell.score_column,
        "holdout": cell.holdout,
        "domain": cell.extra.get("domain", "id"),
        # Which slice of the evaluation set this row measures. "overall"/"all" for a
        # whole-set cell, otherwise a demographic dimension and one of its values (see
        # evaluation/uq/subgroups.py). Present on every row so a consumer can filter to
        # overall rows without knowing whether subgroup scoring ran.
        "subgroup_dimension": cell.extra.get("subgroup_dimension", "overall"),
        "subgroup_value": cell.extra.get("subgroup_value", "all"),
        # Observations about the *slice* rather than about the scoring, so `score_cell`
        # cannot derive them. Carried here so `subgroups.annotate_small_subgroups` can
        # fold them into status_flags -- without this, a ten-row subgroup is scored and
        # reported as though it were measurable.
        "subgroup_flags": cell.extra.get("subgroup_flags", ""),
        "corruption": cell.corruption,
        "severity": cell.severity,
        "coverage": cell.coverage,
        "determinism_mode": cell.determinism_mode,
        "manifest_sha256": cell.manifest_sha256,
        "graph_norm_sha256": cell.graph_norm_sha256,
        "seed": cell.seed,
        "model_agnostic": spec.model_agnostic,
        "produces_probabilities": spec.produces_probabilities,
        "rank_equivalent_to": spec.rank_equivalent_to,
        "cost_forward_passes": cell.cost_forward_passes,
        "cost_training_runs": cell.cost_training_runs,
        "threshold": cell.threshold,
    }


def assert_comparable(cells):
    """Refuse to pool cells whose provenance differs in a way that matters."""
    problems = []
    for attribute, label in (
        ("determinism_mode", "determinism mode"),
        ("manifest_sha256", "evaluation manifest"),
        ("graph_norm_sha256", "graph-distance normalization statistics"),
    ):
        values = {getattr(cell, attribute) for cell in cells}
        values.discard(None)
        values.discard("unknown")
        if len(values) > 1:
            problems.append(f"{label}: {sorted(values)}")
    if problems:
        raise IncomparableCellsError(
            "refusing to aggregate cells with differing provenance, which would "
            "average incommensurable numbers:\n  - " + "\n  - ".join(problems)
        )


def score_cells(cells, n_boot=0, bootstrap_seed=42, require_comparable=True):
    """Score many cells into a tidy DataFrame."""
    import pandas as pd

    cells = list(cells)
    if require_comparable and cells:
        assert_comparable(cells)
    rows = [score_cell(cell, n_boot=n_boot, bootstrap_seed=bootstrap_seed) for cell in cells]
    return pd.DataFrame(rows)


def add_skipped_rows(results, decisions):
    """Append gate skips as ``status='skipped'`` rows.

    Skips belong in the table, not just the log: a published matrix should show
    explained holes rather than silently missing rows.
    """
    import pandas as pd

    rows = []
    for decision in decisions:
        if decision.compatible:
            continue
        rows.append({
            "detector": decision.detector,
            "method_id": decision.method_id,
            "status": "skipped" if decision.severity == "skip" else "broken",
            "status_flags": ";".join(sorted(decision.missing)),
            "skip_reason": " ".join(decision.reasons),
        })
    if not rows:
        return results
    return pd.concat([results, pd.DataFrame(rows)], ignore_index=True)


def collapse_rank_equivalents(results, metric_columns=None):
    """Blank ranking metrics on methods that are rank-identical to another.

    Entropy, margin, and temperature scaling are monotone functions of
    max-probability for a Bernoulli, so their ranking metrics are identical *by
    construction*. Printing them would show duplicate columns and imply a difference
    that does not exist -- so they are replaced with a pointer to the representative,
    while their calibration metrics (the whole reason temperature scaling exists) are
    left intact.
    """
    if results.empty or "rank_equivalent_to" not in results.columns:
        return results

    metric_columns = metric_columns or [
        "auroc_error", "aupr_error", "aurc", "eaurc",
    ] + [column for column in results.columns if column.startswith("accuracy_at_")]

    collapsed = results.copy()
    mask = collapsed["rank_equivalent_to"].notna()
    for column in metric_columns:
        if column in collapsed.columns:
            collapsed.loc[mask, column] = np.nan
    collapsed.loc[mask, "ranking_note"] = (
        "= " + collapsed.loc[mask, "rank_equivalent_to"].astype(str) + " by construction"
    )
    return collapsed


def score_ood(id_frame, ood_frame, score_column, detector, method_id, **cell_kwargs):
    """OOD-detection metrics for one method.

    A ranking task, so a single-class OOD partition is fine -- which matters, because
    holding a generator out of AI-Face yields an all-fake OOD set. Classification
    metrics on such a set are meaningless and are refused by
    ``guard_single_class_classification``.
    """
    spec = method_spec(method_id)
    result = ood_detection(
        resolve_score(id_frame, score_column), resolve_score(ood_frame, score_column)
    )
    return {
        "detector": detector,
        "method_id": method_id,
        "method_family": spec.family,
        "score_column": score_column,
        "status": "ok" if result.applicable else "refused",
        "ood_auroc": result.auroc,
        "ood_aupr_in": result.aupr_in,
        "ood_aupr_out": result.aupr_out,
        "ood_fpr_at_95_tpr": result.fpr_at_95_tpr,
        "n_id": result.n_id,
        "n_ood": result.n_ood,
        "status_flags": ";".join(sorted(result.status_flags)),
        **cell_kwargs,
    }


def guard_single_class_classification(frame, context=""):
    """Refuse classification metrics on a single-class set.

    Holding out a generator produces an all-fake evaluation set. AUROC and friends are
    undefined there, and reporting them -- or silently reporting an accuracy that is
    just the class prior -- would be worse than refusing. Shifted-*classification*
    evaluation must therefore mix held-out fakes with in-distribution reals.
    """
    labels = frame["label"].to_numpy(dtype=int)
    present = set(np.unique(labels))
    if len(present) < 2:
        raise IncomparableCellsError(
            f"{context or 'this evaluation set'} contains only class {present}. "
            f"Classification metrics are undefined on a single-class set -- holding a "
            f"generator out of AI-Face yields an all-fake set, so shifted-classification "
            f"evaluation must add in-distribution reals. Use score_ood() for the "
            f"OOD-detection question instead, which is a ranking task and does not need "
            f"both classes."
        )
    return True


def pivot_for_paper(results, metric="eaurc", index="method_id", columns="detector"):
    """Wide view of one metric, for a paper table."""
    usable = results[results["status"] == "ok"] if "status" in results.columns else results
    return usable.pivot_table(index=index, columns=columns, values=metric, aggfunc="mean")
