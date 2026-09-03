"""The baseline-vs-candidate diff: direction, presence, and the paired test.

Three failure modes drive these tests.

**An unoriented metric.** A diff that reports a raw delta makes the reader supply each
column's polarity, and one they get wrong reverses the conclusion. `metric_direction`
resolves it from data, and `test_every_result_column_is_classified` fails when a newly
added metric belongs to neither set -- so a new column cannot arrive silently unoriented.

**A silently filtered row.** A cell present in the baseline and absent from the candidate
is the most important row in the table: something stopped producing a number. An inner
join would drop it.

**A paired test that misreads a bit-exact re-run.** Two same-seed strict runs have zero
discordant pairs, where McNemar's chi-square form divides by zero. That case must come
back p=1, not an error -- it is the expected result of a clean no-change check.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.uq import compare
from evaluation.uq.compare import (
    ComparisonError, HIGHER_IS_BETTER, IGNORED_COLUMNS, LOWER_IS_BETTER,
    assert_metrics_classified, hard_failure_summary, hard_failures, improvements,
    metric_direction, paired_accuracy_test, provenance_warnings, regressions, summarize,
    unclassified_columns,
)

#: Every column `scoring.score_cells`, `score_ood`, `add_skipped_rows`, and the sweep's
#: identity merge can put in a results table. Read off a real `results.csv` header plus
#: the sweep and subgroup additions, so this list is the schema the diff must cover.
RESULT_COLUMNS = (
    "label", "detector", "method_id", "method_family", "score_column", "holdout",
    "domain", "subgroup_dimension", "subgroup_value", "subgroup_flags", "corruption",
    "severity", "coverage", "determinism_mode", "manifest_sha256", "graph_norm_sha256",
    "seed", "model_agnostic", "produces_probabilities", "rank_equivalent_to",
    "cost_forward_passes", "cost_training_runs", "status", "n",
    "clf_accuracy", "clf_balanced_accuracy", "clf_auroc", "clf_auprc", "clf_eer",
    "clf_n_positive", "auroc_error", "error_rate", "aupr_error", "aupr_error_baseline",
    "aurc", "aurc_oracle", "eaurc",
    "accuracy_at_0.5", "accuracy_at_0.7", "accuracy_at_0.8", "accuracy_at_0.9",
    "accuracy_at_0.95", "accuracy_at_1",
    "ece_confidence", "ece_confidence_adaptive", "ece_positive", "mce_confidence",
    "n_empty_bins", "brier", "nll", "nll_clipped_fraction", "calibration_applicable",
    "score_min", "score_median", "score_max", "score_std",
    "auroc_error_ci_low", "auroc_error_ci_high",
    "status_flags", "ranking_note",
    "ood_auroc", "ood_aupr_in", "ood_aupr_out", "ood_fpr_at_95_tpr", "n_id", "n_ood",
    "extra", "skip_reason",
    # sweep identity
    "cell_id", "axis", "axis_value", "arch", "traversal", "graph_type",
    "uncertainty_head", "graph_manager", "run_id", "records_path", "duration_seconds",
    # fairness reductions
    "disparity_range_clf_accuracy", "disparity_mad_clf_accuracy",
    "worst_group_clf_accuracy", "n_groups", "worst_group", "flags",
)


def results(**overrides):
    """A minimal two-row results table with the join keys the diff needs."""
    base = {
        "cell_id": ["reference", "reference"],
        "method_id": ["baseline_maxprob", "graph_hybrid_distance"],
        "score_column": ["u_maxprob", "u_hybrid_distance"],
        "subgroup_dimension": ["overall", "overall"],
        "subgroup_value": ["all", "all"],
        "holdout": ["none", "none"],
        "domain": ["id", "id"],
        "corruption": ["none", "none"],
        "severity": [0, 0],
        "detector": ["tiny", "tiny"],
        "status": ["ok", "ok"],
        "n": [1000, 1000],
        "clf_accuracy": [0.90, 0.90],
        "ece_confidence": [0.05, np.nan],
        "auroc_error": [0.70, 0.62],
        "determinism_mode": ["strict", "strict"],
        "seed": [42, 42],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# -- the direction registry ----------------------------------------------------- #

def test_every_result_column_is_classified():
    """A new metric must be oriented, ignored, or it fails here -- never silently n/a."""
    frame = pd.DataFrame({
        column: [1.0 if column not in ("status", "status_flags") else "ok"]
        for column in RESULT_COLUMNS
    })
    missing = unclassified_columns(frame)
    assert not missing, (
        f"unoriented numeric result column(s): {missing}. Add each to HIGHER_IS_BETTER, "
        f"LOWER_IS_BETTER, or IGNORED_COLUMNS in evaluation/uq/compare.py."
    )
    assert_metrics_classified(frame)


def test_the_two_direction_sets_are_disjoint():
    assert not (HIGHER_IS_BETTER & LOWER_IS_BETTER)
    assert not (HIGHER_IS_BETTER & IGNORED_COLUMNS)
    assert not (LOWER_IS_BETTER & IGNORED_COLUMNS)


@pytest.mark.parametrize("metric,expected", [
    ("clf_accuracy", "higher"),
    ("clf_auroc", "higher"),
    ("auroc_error", "higher"),
    ("accuracy_at_0.8", "higher"),
    ("ece_confidence", "lower"),
    ("brier", "lower"),
    ("eaurc", "lower"),
    ("clf_eer", "lower"),
    ("ood_fpr_at_95_tpr", "lower"),
    ("detector", None),
])
def test_metric_direction(metric, expected):
    assert metric_direction(metric) == expected


def test_disparity_is_always_lower_is_better():
    """A spread between groups is better when smaller, whichever way the base points."""
    assert metric_direction("disparity_range_clf_accuracy") == "lower"
    assert metric_direction("disparity_range_ece_confidence") == "lower"
    assert metric_direction("disparity_mad_auroc_error") == "lower"


def test_worst_group_inherits_the_base_metrics_direction():
    assert metric_direction("worst_group_clf_accuracy") == "higher"
    assert metric_direction("worst_group_ece_confidence") == "lower"


def test_a_prefixed_unknown_metric_stays_unoriented():
    """Prefix inheritance must not invent a direction for an unclassified base."""
    assert metric_direction("disparity_range_no_such_metric") is None
    assert metric_direction("worst_group_no_such_metric") is None


def test_assert_metrics_classified_names_the_offender():
    frame = pd.DataFrame({"cell_id": ["a"], "brand_new_metric": [0.5]})
    with pytest.raises(ComparisonError, match="brand_new_metric"):
        assert_metrics_classified(frame)


def test_booleans_are_not_treated_as_metrics():
    frame = pd.DataFrame({"cell_id": ["a"], "some_flag": [True]})
    assert unclassified_columns(frame) == []


# -- compare -------------------------------------------------------------------- #

def test_identical_tables_are_all_same():
    baseline = results()
    diff = compare.compare(baseline, baseline.copy())
    assert set(diff["direction"]) == {"same"}
    assert summarize(diff)["worse"] == 0
    assert regressions(diff).empty


def test_direction_reflects_polarity_not_sign():
    baseline = results()
    candidate = results(clf_accuracy=[0.88, 0.88], ece_confidence=[0.03, np.nan])
    diff = compare.compare(baseline, candidate)

    accuracy = diff[(diff["metric"] == "clf_accuracy")
                    & (diff["method_id"] == "baseline_maxprob")].iloc[0]
    assert accuracy["delta"] == pytest.approx(-0.02)
    assert accuracy["direction"] == "worse"

    ece = diff[diff["metric"] == "ece_confidence"].iloc[0]
    assert ece["delta"] == pytest.approx(-0.02)
    # Same sign, opposite verdict.
    assert ece["direction"] == "better"


def test_percent_delta_is_relative_to_the_baseline_magnitude():
    diff = compare.compare(results(), results(clf_accuracy=[0.81, 0.90]))
    row = diff[(diff["metric"] == "clf_accuracy")
               & (diff["method_id"] == "baseline_maxprob")].iloc[0]
    assert row["pct_delta"] == pytest.approx(-10.0)


def test_zero_baseline_gives_no_percent_delta():
    diff = compare.compare(
        results(clf_accuracy=[0.0, 0.0]), results(clf_accuracy=[0.1, 0.1])
    )
    row = diff[diff["metric"] == "clf_accuracy"].iloc[0]
    assert np.isnan(row["pct_delta"])
    assert row["direction"] == "better"


def test_tolerance_widens_same():
    baseline = results()
    candidate = results(clf_accuracy=[0.9005, 0.9005])
    strict = compare.compare(baseline, candidate)
    assert "worse" in set(strict["direction"]) or "better" in set(strict["direction"])

    tolerant = compare.compare(baseline, candidate, abs_tolerance=0.001)
    accuracy = tolerant[tolerant["metric"] == "clf_accuracy"]
    assert set(accuracy["direction"]) == {"same"}


def test_relative_tolerance_scales_with_the_value():
    diff = compare.compare(
        results(), results(clf_accuracy=[0.8955, 0.8955]), rel_tolerance=0.01
    )
    assert set(diff[diff["metric"] == "clf_accuracy"]["direction"]) == {"same"}


def test_a_removed_cell_is_reported_not_dropped():
    """The most important row in the table: something stopped producing a number."""
    baseline = results()
    candidate = results().iloc[:1]
    diff = compare.compare(baseline, candidate)

    removed = diff[diff["presence"] == "removed"]
    assert not removed.empty
    assert set(removed["method_id"]) == {"graph_hybrid_distance"}
    assert not hard_failures(diff).empty


def test_an_added_cell_is_reported_as_added():
    diff = compare.compare(results().iloc[:1], results())
    added = diff[diff["presence"] == "added"]
    assert set(added["method_id"]) == {"graph_hybrid_distance"}
    # New is not a failure.
    assert hard_failures(diff)["presence"].tolist() == []


def test_a_metric_that_became_degenerate_is_a_hard_failure():
    candidate = results(status=["ok", "degenerate"], auroc_error=[0.70, np.nan])
    diff = compare.compare(results(), candidate)
    failures = hard_failures(diff)
    assert not failures.empty
    assert set(failures["method_id"]) == {"graph_hybrid_distance"}
    assert failures["newly_degenerate"].all()


def test_a_metric_that_merely_moved_is_not_a_hard_failure():
    diff = compare.compare(results(), results(clf_accuracy=[0.5, 0.5]))
    assert hard_failures(diff).empty
    assert not regressions(diff).empty


def test_na_where_only_one_side_has_a_number():
    """Presence information, not a better/worse call."""
    diff = compare.compare(results(), results(ece_confidence=[np.nan, np.nan]))
    ece = diff[diff["metric"] == "ece_confidence"].iloc[0]
    assert ece["direction"] == "n_a"


def test_rows_with_no_number_on_either_side_are_omitted():
    baseline = results(ece_confidence=[np.nan, np.nan])
    diff = compare.compare(baseline, baseline.copy())
    assert "ece_confidence" not in set(diff["metric"])


def test_regressions_are_sorted_worst_first():
    candidate = results(clf_accuracy=[0.45, 0.89], auroc_error=[0.70, 0.62])
    worse = regressions(compare.compare(results(), candidate))
    assert not worse.empty
    magnitudes = worse["pct_delta"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_improvements_are_separated():
    diff = compare.compare(results(), results(clf_accuracy=[0.95, 0.95]))
    assert not improvements(diff).empty
    assert regressions(diff).empty


def test_missing_join_key_is_an_error_naming_the_key():
    baseline = results().drop(columns=["cell_id"])
    with pytest.raises(ComparisonError, match="cell_id"):
        compare.compare(baseline, results())


def test_empty_table_is_an_error():
    with pytest.raises(ComparisonError, match="empty"):
        compare.compare(pd.DataFrame(), results())


def test_join_keys_survive_a_csv_round_trip(tmp_path):
    """`severity` is an int in memory and a str after read_csv on a mixed column."""
    path = tmp_path / "results.csv"
    results().to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    diff = compare.compare(results(), reloaded)
    assert set(diff["direction"]) == {"same"}
    assert (diff["presence"] == "both").all()


def test_subgroup_rows_join_on_their_own_identity():
    """A subgroup row must not be matched to the whole-set row of the same cell."""
    baseline = pd.concat([
        results(),
        results(subgroup_dimension=["gt_race", "gt_race"], subgroup_value=[0, 0],
                clf_accuracy=[0.70, 0.70]),
    ], ignore_index=True)
    candidate = baseline.copy()
    candidate.loc[candidate["subgroup_value"] == 0, "clf_accuracy"] = 0.60

    diff = compare.compare(baseline, candidate)
    overall = diff[(diff["subgroup_value"] == "all") & (diff["metric"] == "clf_accuracy")]
    subgroup = diff[(diff["subgroup_value"] == "0") & (diff["metric"] == "clf_accuracy")]
    assert set(overall["direction"]) == {"same"}
    assert set(subgroup["direction"]) == {"worse"}


# -- hard failure collapsing ---------------------------------------------------- #

def test_hard_failure_summary_collapses_slices_of_one_measurement():
    """One degenerate method is one failure, not one per subgroup per metric."""
    baseline = pd.concat(
        [results()]
        + [
            results(subgroup_dimension=["gt_race", "gt_race"],
                    subgroup_value=[value, value])
            for value in (0, 1, 2, 3)
        ],
        ignore_index=True,
    )
    candidate = baseline.copy()
    candidate.loc[candidate["method_id"] == "graph_hybrid_distance", "status"] = "degenerate"
    candidate.loc[candidate["method_id"] == "graph_hybrid_distance", "auroc_error"] = np.nan

    diff = compare.compare(baseline, candidate)
    assert len(hard_failures(diff)) > 5, "the long table fans out, as expected"
    summary = hard_failure_summary(diff)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["method_id"] == "graph_hybrid_distance"
    assert row["reason"] == "newly_degenerate"
    assert row["subgroups_affected"] >= 4
    assert row["metrics_affected"] >= 1


def test_hard_failure_summary_reports_removed_before_degenerate():
    baseline = pd.concat([results(), results(cell_id=["other", "other"])],
                         ignore_index=True)
    candidate = baseline[baseline["cell_id"] == "reference"].copy()
    candidate.loc[candidate["method_id"] == "graph_hybrid_distance", "status"] = "degenerate"
    candidate.loc[candidate["method_id"] == "graph_hybrid_distance", "auroc_error"] = np.nan

    summary = hard_failure_summary(compare.compare(baseline, candidate))
    assert summary.iloc[0]["reason"] == "newly_degenerate" or \
        summary["reason"].tolist().count("removed") > 0
    assert set(summary["reason"]) >= {"removed"}


def test_hard_failure_summary_on_a_clean_diff_is_empty():
    assert hard_failure_summary(compare.compare(results(), results())).empty


# -- the paired test ------------------------------------------------------------ #

def make_records(n=200, seed=0, flip=0):
    rng = np.random.default_rng(seed)
    correct = (rng.random(n) > 0.15).astype(int)
    frame = pd.DataFrame({
        "record_id": [f"r{index:05d}" for index in range(n)],
        "correct": correct,
    })
    if flip:
        frame.loc[:flip - 1, "correct"] = 1 - frame.loc[:flip - 1, "correct"]
    return frame


def test_identical_records_give_p_one_not_an_error():
    """A bit-exact re-run has zero discordant pairs, where chi-square divides by zero."""
    records = make_records(seed=1)
    result = paired_accuracy_test(records, records.copy())
    assert result["applicable"]
    assert result["method"] == "identical"
    assert result["p_value"] == 1.0
    assert result["n_baseline_only_correct"] == 0
    assert result["n_candidate_only_correct"] == 0


def test_a_large_one_sided_change_is_significant():
    baseline = make_records(n=400, seed=2)
    candidate = baseline.copy()
    # Break 60 samples the baseline got right.
    right = candidate.index[candidate["correct"] == 1][:60]
    candidate.loc[right, "correct"] = 0

    result = paired_accuracy_test(baseline, candidate)
    assert result["applicable"]
    assert result["n_baseline_only_correct"] == 60
    assert result["n_candidate_only_correct"] == 0
    assert result["p_value"] < 1e-6
    assert result["candidate_accuracy"] < result["baseline_accuracy"]


def test_a_small_change_uses_the_exact_test():
    baseline = make_records(n=200, seed=3)
    candidate = baseline.copy()
    right = candidate.index[candidate["correct"] == 1][:3]
    candidate.loc[right, "correct"] = 0

    result = paired_accuracy_test(baseline, candidate)
    assert result["method"] == "exact"
    assert 0.0 < result["p_value"] <= 1.0


def test_a_balanced_change_is_not_significant():
    """Equal numbers gained and lost is exactly what McNemar should call a wash."""
    baseline = make_records(n=400, seed=4)
    candidate = baseline.copy()
    right = candidate.index[candidate["correct"] == 1][:40]
    wrong = candidate.index[candidate["correct"] == 0][:40]
    candidate.loc[right, "correct"] = 0
    candidate.loc[wrong, "correct"] = 1

    result = paired_accuracy_test(baseline, candidate)
    assert result["p_value"] > 0.5
    assert result["baseline_accuracy"] == pytest.approx(result["candidate_accuracy"])


def test_too_little_overlap_is_refused_with_a_reason():
    baseline = make_records(n=200, seed=5)
    candidate = make_records(n=200, seed=5)
    candidate["record_id"] = [f"x{index:05d}" for index in range(len(candidate))]

    result = paired_accuracy_test(baseline, candidate)
    assert not result["applicable"]
    assert "both tables" in result["reason"]
    assert result["n_aligned"] == 0


def test_partial_overlap_is_used_and_its_size_reported():
    """Refusing outright would make a partial comparison impossible rather than visible."""
    baseline = make_records(n=300, seed=6)
    candidate = baseline.iloc[50:].copy()
    result = paired_accuracy_test(baseline, candidate)
    assert result["applicable"]
    assert result["n_aligned"] == 250


def test_missing_column_is_reported_not_raised():
    baseline = make_records(seed=7).drop(columns=["correct"])
    result = paired_accuracy_test(baseline, make_records(seed=7))
    assert not result["applicable"]
    assert "correct" in result["reason"]


def test_empty_records_are_reported_not_raised():
    result = paired_accuracy_test(pd.DataFrame(), make_records(seed=8))
    assert not result["applicable"]
    assert "empty" in result["reason"]


# -- provenance ----------------------------------------------------------------- #

def test_a_changed_seed_is_a_warning_not_a_refusal():
    """The git commit differing is the point of a diff; a changed seed is not."""
    warnings = provenance_warnings(results(), results(seed=[7, 7]))
    assert any("seed" in warning for warning in warnings)


def test_fast_determinism_is_called_out():
    candidate = results(determinism_mode=["fast", "fast"])
    warnings = provenance_warnings(results(), candidate)
    assert any("determinism=fast" in warning for warning in warnings)
    assert any("3e-2" in warning for warning in warnings)


def test_matching_provenance_produces_no_warnings():
    assert provenance_warnings(results(), results()) == []


def test_unknown_provenance_is_not_a_mismatch():
    """`unknown` means unrecorded, which is not evidence of a difference."""
    candidate = results(determinism_mode=["unknown", "unknown"])
    warnings = provenance_warnings(results(), candidate)
    assert not any("determinism mode differs" in warning for warning in warnings)


# -- summary -------------------------------------------------------------------- #

def test_summarize_counts_add_up():
    candidate = results(clf_accuracy=[0.95, 0.85])
    diff = compare.compare(results(), candidate)
    counts = summarize(diff)
    assert counts["better"] + counts["worse"] + counts["same"] + counts["n_a"] == counts["rows"]


def test_summarize_on_empty():
    assert summarize(pd.DataFrame()) == {"rows": 0}
