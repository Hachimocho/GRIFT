"""Diff two scored results tables: what moved, in which direction, and is it real.

The repo could score a run and could pool several runs into one table, but nothing
computed a delta. `web_ui/test_runner.compare_runs` compares `final_accuracy` and
nothing else; `holdouts.control_id` names the control a shift comparison needs and the
delta it exists for was never implemented.

Two things make a diff trustworthy, and both are data rather than judgement:

**Direction is declared, not inferred from the sign.** Lower ECE is better and higher
accuracy is better, so a table that reports raw deltas makes the reader supply the
orientation for every column -- and a reader who gets one wrong concludes the opposite of
the truth. `metric_direction` resolves it from `HIGHER_IS_BETTER` / `LOWER_IS_BETTER`, and
`assert_metrics_classified` fails a test when a newly added metric belongs to neither, so
a new column cannot arrive silently oriented as "unknown".

**Presence is reported.** A cell in the baseline and absent from the candidate is the most
important row in the table -- something stopped producing a number -- and an outer join
plus an explicit `presence` column is the only way it survives. Filtering to rows present
in both is how a regression hides.

`paired_accuracy_test` exists because the two runs usually scored *the same samples*: the
sweep pins the seed, the node cache, and strict determinism, so baseline and candidate
differ only in code. Under those conditions McNemar on per-sample correctness answers "is
this accuracy difference real?" far more sharply than comparing two independent
accuracies, and it needs no bootstrap.
"""

from typing import Optional, Sequence

import numpy as np

#: Metrics where a larger value is better.
HIGHER_IS_BETTER = frozenset({
    # classification
    "clf_accuracy", "clf_balanced_accuracy", "clf_auroc", "clf_auprc",
    # uncertainty ranking: how well the score separates errors from correct predictions
    "auroc_error", "aupr_error",
    # selective prediction
    "accuracy_at_0.5", "accuracy_at_0.7", "accuracy_at_0.8", "accuracy_at_0.9",
    "accuracy_at_0.95", "accuracy_at_1",
    # OOD detection
    "ood_auroc", "ood_aupr_in", "ood_aupr_out",
    # fairness: the worst group's accuracy-like metrics rise when disparity narrows.
    # Suffixed forms are resolved by `metric_direction`, which strips the prefix.
})

#: Metrics where a smaller value is better.
LOWER_IS_BETTER = frozenset({
    "clf_eer", "error_rate",
    # risk-coverage. Report E-AURC rather than raw AURC (docs/uq_benchmark.md), but both
    # are oriented the same way.
    "aurc", "eaurc",
    # calibration
    "ece_confidence", "ece_confidence_adaptive", "ece_positive", "mce_confidence",
    "brier", "nll",
    # OOD detection
    "ood_fpr_at_95_tpr",
    # cost
    "cost_forward_passes", "cost_training_runs", "duration_seconds",
})

#: Columns that identify or describe a measurement rather than grading it. Deltas on
#: these are meaningless, so they are neither compared nor flagged as unclassified.
IGNORED_COLUMNS = frozenset({
    # identity / provenance
    "label", "detector", "method_id", "method_family", "score_column", "holdout",
    "domain", "corruption", "severity", "coverage", "determinism_mode",
    "manifest_sha256", "graph_norm_sha256", "seed", "model_agnostic",
    "produces_probabilities", "rank_equivalent_to", "status", "status_flags",
    "ranking_note", "skip_reason", "extra", "calibration_applicable",
    "subgroup_dimension", "subgroup_value", "subgroup_flags", "subgroup_n",
    # sweep identity, added by the sweep driver
    "cell_id", "axis", "axis_value", "run_id", "sweep_id", "suite", "tag",
    "arch", "traversal", "graph_type", "uncertainty_head", "graph_manager",
    "records_path", "git_commit", "worst_group", "flags", "metric",
    # counts and descriptive statistics: they explain a delta, they are not one. A
    # changed `n` matters, but as a provenance warning rather than a better/worse call.
    # The operating point is a setting, not a score: a higher or lower threshold is
    # neither better nor worse in itself, and the metrics taken at it already move.
    "threshold", "clf_threshold",
    "n", "n_id", "n_ood", "n_groups", "clf_n_positive", "n_empty_bins",
    "nll_clipped_fraction", "aupr_error_baseline", "aurc_oracle",
    "score_min", "score_median", "score_max", "score_std",
    "auroc_error_ci_low", "auroc_error_ci_high",
})

#: Prefixes `metric_direction` strips before looking a metric up, so the fairness
#: reductions in `subgroups.py` inherit their base metric's orientation:
#: `worst_group_clf_accuracy` is higher-is-better because `clf_accuracy` is.
INHERITING_PREFIXES = ("worst_group_", "best_group_", "overall_")

#: Prefixes that invert their base metric's orientation. A *spread* between subgroups is
#: better when it is smaller, whichever way the underlying metric points.
DISPARITY_PREFIXES = ("disparity_range_", "disparity_mad_")

#: Columns that identify one measurement across two runs. `cell_id` is the sweep's
#: identifier for a configuration; the rest distinguish measurements within it.
DEFAULT_JOIN_KEYS = (
    "cell_id", "method_id", "score_column",
    "subgroup_dimension", "subgroup_value",
    "holdout", "domain", "corruption", "severity",
)

#: Statuses that mean the cell produced no usable number.
NON_OK_STATUSES = frozenset({
    "degenerate", "refused_low_coverage", "refused", "skipped", "broken", "missing",
})

#: Below this many aligned rows a paired test says nothing, so it is refused.
MIN_PAIRED_ROWS = 30

#: `records.py`'s per-sample key, and the column the paired test aligns on.
JOIN_KEY = "record_id"


class ComparisonError(ValueError):
    """Raised when two results tables cannot be compared."""


def metric_direction(metric):
    """``'higher'``, ``'lower'``, or ``None`` for a metric that grades nothing.

    Resolves prefixed fairness reductions against their base metric, so adding a base
    metric to one of the two sets orients every derived column at once.
    """
    name = str(metric)
    if name in HIGHER_IS_BETTER:
        return "higher"
    if name in LOWER_IS_BETTER:
        return "lower"

    for prefix in DISPARITY_PREFIXES:
        if name.startswith(prefix):
            # A spread is better when smaller regardless of the base metric's direction,
            # so the base only has to be *classified*, not consulted for its sign.
            base = metric_direction(name[len(prefix):])
            return "lower" if base else None

    for prefix in INHERITING_PREFIXES:
        if name.startswith(prefix):
            return metric_direction(name[len(prefix):])

    return None


def metric_columns(results, metrics=None):
    """Comparable metric columns in `results`, in table order.

    A column is comparable when `metric_direction` orients it. Unclassified columns are
    skipped here and surfaced by `unclassified_columns`, so a diff never silently prints
    a delta whose direction nobody decided.
    """
    if metrics is not None:
        return [metric for metric in metrics if metric in results.columns]
    return [
        column for column in results.columns
        if column not in IGNORED_COLUMNS and metric_direction(column) is not None
    ]


def unclassified_columns(results):
    """Numeric columns that are neither ignored nor oriented.

    The gap a new metric falls into. `assert_metrics_classified` turns this into a test
    failure; the CLI prints it as a warning so a new column is noticed the first time it
    appears rather than the first time someone misreads it.
    """
    import pandas as pd

    found = []
    for column in results.columns:
        if column in IGNORED_COLUMNS or metric_direction(column) is not None:
            continue
        if pd.api.types.is_numeric_dtype(results[column]) and not pd.api.types.is_bool_dtype(
            results[column]
        ):
            found.append(column)
    return found


def assert_metrics_classified(results):
    """Raise if any numeric column is neither ignored nor oriented."""
    missing = unclassified_columns(results)
    if missing:
        raise ComparisonError(
            "these numeric result columns have no comparison direction: "
            f"{', '.join(missing)}.\nAdd each to HIGHER_IS_BETTER, LOWER_IS_BETTER, or "
            "IGNORED_COLUMNS in evaluation/uq/compare.py. Left unclassified, a change in "
            "them would be reported without saying whether it is an improvement."
        )


def compare(
    baseline,
    candidate,
    keys: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    rel_tolerance: float = 0.0,
    abs_tolerance: float = 0.0,
):
    """Long-format diff of two scored results tables.

    One row per (measurement, metric):

    ``presence``          ``both`` | ``added`` | ``removed``
    ``baseline``          the baseline value, NaN when absent
    ``candidate``         the candidate value, NaN when absent
    ``delta``             candidate - baseline
    ``pct_delta``         delta / |baseline| * 100, NaN when the baseline is 0 or absent
    ``direction``         ``better`` | ``worse`` | ``same`` | ``n_a``
    ``status_baseline``   / ``status_candidate`` -- so a metric that moved *because* the
                          cell went degenerate is distinguishable from one that moved
                          while both cells were healthy
    ``newly_degenerate``  True when the candidate lost a status the baseline had. This is
                          the hard-failure signal: a metric that stops being computable
                          is not a small regression.

    `abs_tolerance` / `rel_tolerance` widen ``same``: a delta inside either is not called
    a movement. Both default to 0, so exact equality is required for ``same`` -- which is
    the right default when the two runs were bit-exact by construction.
    """
    import pandas as pd

    keys = list(keys or DEFAULT_JOIN_KEYS)
    baseline = _prepare(baseline, keys, "baseline")
    candidate = _prepare(candidate, keys, "candidate")

    shared_metrics = metric_columns(baseline, metrics)
    candidate_metrics = metric_columns(candidate, metrics)
    all_metrics = [metric for metric in shared_metrics if metric in candidate_metrics]
    # Metrics only one side has: still reported, as added/removed rows, because a metric
    # that vanished is a finding.
    only_baseline = [m for m in shared_metrics if m not in candidate_metrics]
    only_candidate = [m for m in candidate_metrics if m not in shared_metrics]

    merged = baseline.merge(
        candidate, on=keys, how="outer", suffixes=("__base", "__cand"), indicator=True,
    )

    rows = []
    for record in merged.to_dict("records"):
        presence = {
            "left_only": "removed", "right_only": "added", "both": "both",
        }[record["_merge"]]
        status_baseline = record.get("status__base")
        status_candidate = record.get("status__cand")
        newly_degenerate = (
            presence != "added"
            and str(status_baseline) == "ok"
            and str(status_candidate) in NON_OK_STATUSES
        )

        identity = {key: record.get(key) for key in keys}
        identity.update(_carry_identity(record))

        for metric in all_metrics + only_baseline + only_candidate:
            baseline_value = _numeric(record.get(f"{metric}__base"))
            candidate_value = _numeric(record.get(f"{metric}__cand"))
            if np.isnan(baseline_value) and np.isnan(candidate_value):
                continue
            rows.append({
                **identity,
                "metric": metric,
                "presence": presence,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "pct_delta": _percent(baseline_value, candidate_value),
                "direction": _direction(
                    metric, baseline_value, candidate_value,
                    rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance,
                ),
                "status_baseline": status_baseline,
                "status_candidate": status_candidate,
                "newly_degenerate": newly_degenerate,
                "n_baseline": _numeric(record.get("n__base")),
                "n_candidate": _numeric(record.get("n__cand")),
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Worse first: the reason anyone runs this.
    order = {"worse": 0, "n_a": 1, "same": 2, "better": 3}
    frame = frame.assign(_order=frame["direction"].map(order).fillna(1))
    frame = frame.sort_values(
        ["_order", "metric"] + keys, kind="mergesort"
    ).drop(columns="_order").reset_index(drop=True)
    return frame


def regressions(comparison):
    """Rows where the candidate is worse. Sorted worst-first by relative movement."""
    if comparison.empty:
        return comparison
    worse = comparison[comparison["direction"] == "worse"].copy()
    if worse.empty:
        return worse
    worse["_magnitude"] = worse["pct_delta"].abs().fillna(
        worse["delta"].abs().fillna(0.0)
    )
    return worse.sort_values("_magnitude", ascending=False).drop(
        columns="_magnitude"
    ).reset_index(drop=True)


def improvements(comparison):
    """Rows where the candidate is better."""
    if comparison.empty:
        return comparison
    return comparison[comparison["direction"] == "better"].reset_index(drop=True)


def hard_failures(comparison):
    """Every row belonging to a measurement that broke rather than moved.

    A cell absent from the candidate, or one that stopped producing a number. Use
    `hard_failure_summary` for reporting and for counting: one degenerate cell produces a
    row here for every metric and every subgroup slice, so the row count is a
    multiplication of the metric grid rather than a count of things that broke.
    """
    if comparison.empty:
        return comparison
    mask = (comparison["presence"] == "removed") | comparison["newly_degenerate"]
    return comparison[mask].reset_index(drop=True)


#: Columns identifying one broken *measurement*, for collapsing `hard_failures`.
#:
#: Deliberately excludes `subgroup_dimension` / `subgroup_value`: a subgroup row is a
#: slice of the same measurement, not an independent finding, so a method that went
#: degenerate is one failure however many demographic slices it had.
FAILURE_KEYS = ("cell_id", "detector", "method_id", "score_column")


def hard_failure_summary(comparison):
    """One row per broken measurement, not per (metric, subgroup) pair.

    A single cell going degenerate blanks every metric on every demographic slice, which
    in the long table is hundreds of rows and in reality is one problem. Collapsing before
    reporting is what makes the count mean "this many things broke" -- and that count is
    what the CLI's exit code is about.

    Adds `reason` (`removed` or `newly_degenerate`), the metric and subgroup counts, and
    a truncated list of the affected metric names.
    """
    import pandas as pd

    rows = hard_failures(comparison)
    if rows.empty:
        return rows

    keys = [key for key in FAILURE_KEYS if key in rows.columns]
    if not keys:
        return rows

    summary = []
    for group_values, group in rows.groupby(keys, dropna=False):
        identity = dict(zip(keys, group_values if isinstance(group_values, tuple)
                            else (group_values,)))
        removed = (group["presence"] == "removed").any()
        metrics = sorted({str(metric) for metric in group["metric"]})
        subgroups = (
            sorted({str(value) for value in group["subgroup_value"]})
            if "subgroup_value" in group.columns else []
        )
        summary.append({
            **identity,
            "reason": "removed" if removed else "newly_degenerate",
            "status_baseline": _first(group, "status_baseline"),
            "status_candidate": _first(group, "status_candidate"),
            "metrics_affected": len(metrics),
            "subgroups_affected": len(subgroups),
            "metrics": ", ".join(metrics[:5]) + (" ..." if len(metrics) > 5 else ""),
        })

    frame = pd.DataFrame(summary)
    # Removed measurements first: a vanished cell is worse news than a degenerate one.
    return frame.sort_values(
        ["reason"] + keys, ascending=[True] + [True] * len(keys), kind="mergesort"
    ).reset_index(drop=True)


def _first(group, column):
    if column not in group.columns:
        return None
    values = group[column].dropna()
    return values.iloc[0] if len(values) else None


def summarize(comparison):
    """Counts per direction and presence, for a one-line stdout summary."""
    if comparison.empty:
        return {"rows": 0}
    counts = comparison["direction"].value_counts().to_dict()
    presence = comparison["presence"].value_counts().to_dict()
    return {
        "rows": int(len(comparison)),
        "better": int(counts.get("better", 0)),
        "worse": int(counts.get("worse", 0)),
        "same": int(counts.get("same", 0)),
        "n_a": int(counts.get("n_a", 0)),
        "added": int(presence.get("added", 0)),
        "removed": int(presence.get("removed", 0)),
        "newly_degenerate": int(comparison["newly_degenerate"].sum()),
    }


def paired_accuracy_test(baseline_frame, candidate_frame, min_rows=MIN_PAIRED_ROWS):
    """McNemar's test on per-sample correctness, aligned by `record_id`.

    Returns a dict with the discordant counts, the statistic, the p-value, and the
    aligned row count. `applicable` is False -- with a reason -- when the two tables
    overlap on too few samples to say anything.

    Intersects on `record_id` rather than calling `ensemble.align_frames`, which refuses
    a mismatch outright. That refusal is correct for averaging an ensemble, where a
    member evaluating different samples must not be folded in silently. Here the sample
    sets *may* legitimately differ -- a code change can alter which nodes load -- so the
    overlap is used and its size is reported, which makes the partial comparison visible
    instead of impossible.

    Exact binomial when the discordant count is small, chi-square with continuity
    correction otherwise; that is the standard split and it matters here because a
    bit-exact re-run has zero discordant pairs, where the chi-square form is undefined.
    """
    result = {
        "applicable": False, "reason": "", "n_aligned": 0,
        "n_baseline_only_correct": 0, "n_candidate_only_correct": 0,
        "statistic": np.nan, "p_value": np.nan,
        "baseline_accuracy": np.nan, "candidate_accuracy": np.nan,
        "method": "",
    }

    for frame, name in ((baseline_frame, "baseline"), (candidate_frame, "candidate")):
        if frame is None or len(frame) == 0:
            result["reason"] = f"{name} record table is empty"
            return result
        for column in (JOIN_KEY, "correct"):
            if column not in frame.columns:
                result["reason"] = f"{name} record table has no {column!r} column"
                return result

    left = baseline_frame[[JOIN_KEY, "correct"]].drop_duplicates(JOIN_KEY)
    right = candidate_frame[[JOIN_KEY, "correct"]].drop_duplicates(JOIN_KEY)
    merged = left.merge(right, on=JOIN_KEY, suffixes=("_base", "_cand"))

    result["n_aligned"] = int(len(merged))
    if len(merged) < min_rows:
        result["reason"] = (
            f"only {len(merged)} sample(s) appear in both tables (need {min_rows}); "
            f"the two runs did not evaluate the same set"
        )
        return result

    base_correct = merged["correct_base"].to_numpy().astype(bool)
    cand_correct = merged["correct_cand"].to_numpy().astype(bool)
    baseline_only = int(np.sum(base_correct & ~cand_correct))
    candidate_only = int(np.sum(~base_correct & cand_correct))

    result.update({
        "applicable": True,
        "n_baseline_only_correct": baseline_only,
        "n_candidate_only_correct": candidate_only,
        "baseline_accuracy": float(base_correct.mean()),
        "candidate_accuracy": float(cand_correct.mean()),
    })

    discordant = baseline_only + candidate_only
    if discordant == 0:
        # Identical per-sample outcomes. p=1 exactly, and the chi-square form would
        # divide by zero. This is the expected result of a bit-exact re-run, so it must
        # not read as an error.
        result.update({"statistic": 0.0, "p_value": 1.0, "method": "identical"})
        return result

    if discordant < 25:
        from scipy.stats import binomtest

        smaller = min(baseline_only, candidate_only)
        result.update({
            "statistic": float(smaller),
            "p_value": float(
                binomtest(smaller, discordant, 0.5, alternative="two-sided").pvalue
            ),
            "method": "exact",
        })
        return result

    from scipy.stats import chi2

    statistic = (abs(baseline_only - candidate_only) - 1.0) ** 2 / discordant
    result.update({
        "statistic": float(statistic),
        "p_value": float(chi2.sf(statistic, df=1)),
        "method": "chi2_continuity_corrected",
    })
    return result


def provenance_warnings(baseline, candidate):
    """Reasons this comparison may not be attributable to the code change.

    Not refusals. `scoring.assert_comparable` refuses mixed provenance because pooling
    would average incommensurable numbers; a *diff* of two runs is a different question,
    and the git commit differing is the entire point. But a changed seed, a changed node
    cache, or non-strict determinism all move numbers on their own, so they are reported
    at the top of the report rather than quietly ignored.
    """
    warnings = []

    for column, label, note in (
        ("seed", "seed", "the two runs saw different data order, so small deltas are noise"),
        ("determinism_mode", "determinism mode", "modes are not comparable at fine resolution"),
        ("manifest_sha256", "evaluation manifest", "the two runs scored different samples"),
    ):
        left = _unique(baseline, column)
        right = _unique(candidate, column)
        if left and right and left != right:
            warnings.append(f"{label} differs ({left} vs {right}): {note}")

    for frame, name in ((baseline, "baseline"), (candidate, "candidate")):
        modes = _unique(frame, "determinism_mode")
        if modes == {"fast"}:
            warnings.append(
                f"{name} ran with determinism=fast, where same-seed runs agree only to "
                f"about 3e-2 in probability space on GPU (docs/testing.md); deltas below "
                f"that are not attributable to any change"
            )
    return warnings


# -- internals ------------------------------------------------------------------ #


def _prepare(results, keys, side):
    """Validate and normalize one side of the comparison."""
    if results is None or len(results) == 0:
        raise ComparisonError(f"{side} results table is empty")
    missing = [key for key in keys if key not in results.columns]
    if missing:
        raise ComparisonError(
            f"{side} results table is missing join key(s) {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, results.columns))}"
        )
    frame = results.copy()
    # Join keys must compare equal across a CSV round trip, where an int becomes a str
    # in one table and stays an int in the other.
    for key in keys:
        frame[key] = frame[key].map(_key_text)
    return frame


def _key_text(value):
    import pandas as pd

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return str(int(value)) if float(value).is_integer() else repr(float(value))
    return str(value)


#: Descriptive columns carried onto every diff row when both sides agree on them, so the
#: comparison table is readable without joining back to the results.
_CARRIED = (
    "detector", "method_family", "axis", "axis_value", "arch", "traversal",
    "graph_type", "uncertainty_head", "graph_manager", "label",
)


def _carry_identity(record):
    carried = {}
    for column in _CARRIED:
        value = record.get(f"{column}__base")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = record.get(f"{column}__cand")
        if value is not None:
            carried[column] = value
    return carried


def _direction(metric, baseline_value, candidate_value, rel_tolerance, abs_tolerance):
    orientation = metric_direction(metric)
    if orientation is None:
        return "n_a"
    if np.isnan(baseline_value) or np.isnan(candidate_value):
        # One side has no number. That is presence information, not a better/worse call.
        return "n_a"

    delta = candidate_value - baseline_value
    threshold = max(
        abs_tolerance, rel_tolerance * abs(baseline_value) if rel_tolerance else 0.0
    )
    if abs(delta) <= threshold:
        return "same"
    if orientation == "higher":
        return "better" if delta > 0 else "worse"
    return "better" if delta < 0 else "worse"


def _percent(baseline_value, candidate_value):
    if np.isnan(baseline_value) or np.isnan(candidate_value) or baseline_value == 0:
        return np.nan
    return (candidate_value - baseline_value) / abs(baseline_value) * 100.0


def _numeric(value):
    import pandas as pd

    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _unique(frame, column):
    if frame is None or column not in getattr(frame, "columns", ()):
        return set()
    values = frame[column].dropna().unique()
    return {_key_text(value) for value in values if _key_text(value) not in ("", "unknown")}


__all__ = [
    "ComparisonError", "DEFAULT_JOIN_KEYS", "FAILURE_KEYS", "HIGHER_IS_BETTER",
    "IGNORED_COLUMNS", "JOIN_KEY", "LOWER_IS_BETTER", "MIN_PAIRED_ROWS",
    "NON_OK_STATUSES", "assert_metrics_classified", "compare", "hard_failure_summary",
    "hard_failures", "improvements", "metric_columns", "metric_direction",
    "paired_accuracy_test", "provenance_warnings", "regressions", "summarize",
    "unclassified_columns",
]
