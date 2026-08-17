"""The comparison report's tables and figures.

`ComparisonReport` subclasses `UQReport`, so it inherits the properties that make the
benchmark's output citable. Two of those are worth re-asserting here because a comparison
is where breaking them would mislead:

**`n/a` never renders as `0.0000`.** A graph-distance method has no calibrated probability
to be right about, so a zero ECE would read as perfect calibration.

**Filenames are stable across regenerations.** A report that changes name every time it
runs cannot be diffed or referenced.

And one specific to a diff: **a zero-length bar must be distinguishable from an absent
one**, since both draw nothing.
"""

import os

import numpy as np
import pandas as pd
import pytest

from evaluation.uq import compare
from evaluation.uq.compare_report import ComparisonReport, HEADLINE_METRICS, _number


def results(**overrides):
    base = {
        "cell_id": ["reference", "reference", "arch=effnetdf"],
        "method_id": ["baseline_maxprob", "graph_hybrid_distance", "baseline_maxprob"],
        "score_column": ["u_maxprob", "u_hybrid_distance", "u_maxprob"],
        "subgroup_dimension": ["overall", "overall", "overall"],
        "subgroup_value": ["all", "all", "all"],
        "holdout": ["none"] * 3,
        "domain": ["id"] * 3,
        "corruption": ["none"] * 3,
        "severity": [0] * 3,
        "detector": ["tiny", "tiny", "effnetdf"],
        "method_family": ["logit", "graph", "logit"],
        "status": ["ok", "ok", "ok"],
        "n": [1000] * 3,
        "clf_accuracy": [0.90, 0.90, 0.85],
        "clf_auroc": [0.95, 0.95, 0.91],
        "clf_eer": [0.10, 0.10, 0.14],
        "auroc_error": [0.70, 0.62, 0.68],
        "eaurc": [0.02, 0.03, 0.025],
        "accuracy_at_0.8": [0.94, 0.93, 0.90],
        # The graph method has no calibrated probability: n/a, never 0.
        "ece_confidence": [0.05, np.nan, 0.07],
        "brier": [0.08, np.nan, 0.10],
        "nll": [0.30, np.nan, 0.35],
        "determinism_mode": ["strict"] * 3,
        "seed": [42] * 3,
        "arch": ["tiny", "tiny", "effnetdf"],
        "traversal": ["random"] * 3,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def disparity_rows():
    """Fairness reduction rows, as `subgroups.disparity_as_results` shapes them."""
    return pd.DataFrame({
        "cell_id": ["reference", "reference"],
        "method_id": ["baseline_maxprob", "baseline_maxprob"],
        "score_column": ["u_maxprob", "u_maxprob"],
        "subgroup_dimension": ["gt_race", "gt_gender"],
        "subgroup_value": ["disparity", "disparity"],
        "holdout": ["none", "none"],
        "domain": ["id", "id"],
        "corruption": ["none", "none"],
        "severity": [0, 0],
        "detector": ["tiny", "tiny"],
        "status": ["ok", "ok"],
        "disparity_range_clf_accuracy": [0.17, 0.02],
        "worst_group_clf_accuracy": [0.72, 0.88],
        "n_groups": [4, 2],
        "worst_group": [2, 0],
        "flags": ["", ""],
    })


@pytest.fixture
def baseline():
    return pd.concat([results(), disparity_rows()], ignore_index=True)


@pytest.fixture
def candidate():
    """Accuracy down, calibration up, and the race gap widened."""
    scored = results(
        clf_accuracy=[0.88, 0.88, 0.85],
        ece_confidence=[0.04, np.nan, 0.07],
    )
    fairness = disparity_rows()
    fairness.loc[0, "disparity_range_clf_accuracy"] = 0.22
    fairness.loc[0, "worst_group_clf_accuracy"] = 0.66
    return pd.concat([scored, fairness], ignore_index=True)


@pytest.fixture
def report(tmp_path, baseline, candidate):
    diff = compare.compare(baseline, candidate)
    return ComparisonReport(
        tmp_path / "report", diff, baseline=baseline, candidate=candidate,
        title="unit test",
        baseline_manifest={
            "suite": "smoke", "tag": "baseline", "determinism": "strict",
            "git": {"commit": "a" * 40, "dirty": False},
            "node_cache_sha256": "c" * 64,
        },
        candidate_manifest={
            "suite": "smoke", "tag": "candidate", "determinism": "strict",
            "git": {"commit": "b" * 40, "dirty": True},
            "node_cache_sha256": "c" * 64,
        },
    )


# -- markdown ------------------------------------------------------------------- #

def test_markdown_leads_with_the_verdict(report):
    text = open(report.write_comparison_markdown()).read()
    head = text.split("## Provenance")[0]
    assert "regression" in head.lower() or "hard failure" in head.lower()


def test_na_never_renders_as_zero(report):
    """A graph method's ECE is n/a; printing 0.0000 would read as perfect calibration."""
    text = open(report.write_comparison_markdown()).read()
    graph_lines = [
        line for line in text.splitlines() if "graph_hybrid_distance" in line
    ]
    assert graph_lines
    for line in graph_lines:
        if "ece_confidence" in line:
            assert "n/a" in line
            assert "0.0000" not in line


def test_number_formats_absent_as_na():
    assert _number(None) == "n/a"
    assert _number(float("nan")) == "n/a"
    assert _number(np.nan) == "n/a"
    assert _number(0.5) == "0.5000"
    assert _number(-0.02, signed=True) == "-0.0200"
    assert _number(0.02, signed=True) == "+0.0200"


def test_headline_has_one_row_per_cell(report, candidate):
    text = open(report.write_comparison_markdown()).read()
    headline = text.split("## Headline")[1].split("##")[0]
    for cell_id in sorted(set(candidate["cell_id"])):
        assert cell_id in headline


def test_headline_marks_direction_not_just_sign(report):
    text = open(report.write_comparison_markdown()).read()
    headline = text.split("## Headline")[1].split("##")[0]
    # Accuracy fell and ECE fell; one is a regression and one is an improvement, so both
    # marks must appear on the same row.
    assert "✗" in headline
    assert "✓" in headline


def test_provenance_names_both_commits(report):
    text = open(report.write_comparison_markdown()).read()
    section = text.split("## Provenance")[1].split("## Headline")[0]
    assert "aaaaaaaaaaaa" in section
    assert "bbbbbbbbbbbb" in section
    assert "dirty tree" in section


def test_a_missing_manifest_is_stated_not_omitted(tmp_path, baseline, candidate):
    """Silence about provenance is indistinguishable from matching provenance."""
    diff = compare.compare(baseline, candidate)
    report = ComparisonReport(tmp_path / "r", diff, baseline=baseline, candidate=candidate)
    text = open(report.write_comparison_markdown()).read()
    assert "no manifest found" in text


def test_a_rebuilt_node_cache_invalidates_the_comparison_loudly(
    tmp_path, baseline, candidate
):
    diff = compare.compare(baseline, candidate)
    report = ComparisonReport(
        tmp_path / "r", diff, baseline=baseline, candidate=candidate,
        baseline_manifest={"node_cache_sha256": "1" * 64, "git": {}},
        candidate_manifest={"node_cache_sha256": "2" * 64, "git": {}},
    )
    text = open(report.write_comparison_markdown()).read()
    assert "node cache was rebuilt" in text


def test_regressions_section_lists_the_widened_race_gap(report):
    text = open(report.write_comparison_markdown()).read()
    section = text.split("## Regressions")[1].split("## Improvements")[0]
    assert "disparity_range_clf_accuracy" in section
    assert "gt_race" in section


def test_a_clean_comparison_says_so(tmp_path, baseline):
    diff = compare.compare(baseline, baseline.copy())
    report = ComparisonReport(tmp_path / "r", diff, baseline=baseline, candidate=baseline)
    text = open(report.write_comparison_markdown()).read()
    assert "No hard failures and no regressions" in text


def test_an_empty_comparison_does_not_crash(tmp_path):
    report = ComparisonReport(tmp_path / "r", pd.DataFrame())
    text = open(report.write_comparison_markdown()).read()
    assert "No comparable rows" in text


def test_long_sections_say_how_much_was_truncated(tmp_path):
    """A silent truncation reads as "that was all of them"."""
    many = pd.concat([
        results(cell_id=[f"cell{index}"] * 3) for index in range(30)
    ], ignore_index=True)
    worse = many.copy()
    worse["clf_accuracy"] = worse["clf_accuracy"] - 0.05
    diff = compare.compare(many, worse)
    report = ComparisonReport(tmp_path / "r", diff, baseline=many, candidate=worse)
    text = open(report.write_comparison_markdown()).read()
    assert "Showing" in text and "of" in text


def test_gated_cells_are_listed_with_their_reason(tmp_path, baseline):
    candidate = baseline.copy()
    candidate.loc[0, "status"] = "skipped"
    candidate.loc[0, "skip_reason"] = "detector lacks LAST_LINEAR_GRAFT"
    diff = compare.compare(baseline, candidate)
    report = ComparisonReport(
        tmp_path / "r", diff, baseline=baseline, candidate=candidate
    )
    text = open(report.write_comparison_markdown()).read()
    assert "LAST_LINEAR_GRAFT" in text


# -- csv ------------------------------------------------------------------------ #

def test_comparison_csv_round_trips(report):
    path = report.write_comparison_csv()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(report.comparison)
    assert {"metric", "baseline", "candidate", "delta", "direction", "presence"} <= set(
        reloaded.columns
    )


def test_comparison_csv_is_byte_stable(tmp_path, baseline, candidate):
    """Two reports over the same inputs must produce identical bytes."""
    diff = compare.compare(baseline, candidate)
    first = ComparisonReport(tmp_path / "a", diff, baseline=baseline, candidate=candidate)
    second = ComparisonReport(tmp_path / "b", diff, baseline=baseline, candidate=candidate)
    assert open(first.write_comparison_csv(), "rb").read() == \
        open(second.write_comparison_csv(), "rb").read()


# -- figures -------------------------------------------------------------------- #

def test_generate_all_plots_writes_both_formats(report):
    written = report.generate_all_plots()
    assert written
    names = {os.path.basename(path) for path in written}
    assert "comparison.md" in names
    assert "comparison.csv" in names
    for stem in ("delta_discrimination", "delta_calibration"):
        assert f"{stem}.png" in names
        assert f"{stem}.pdf" in names


def test_filenames_are_stable_across_regenerations(tmp_path, baseline, candidate):
    """No timestamps: a report that renames itself cannot be diffed or cited."""
    diff = compare.compare(baseline, candidate)
    first = ComparisonReport(tmp_path / "r", diff, baseline=baseline, candidate=candidate)
    names_first = sorted(os.path.basename(path) for path in first.generate_all_plots())

    second = ComparisonReport(tmp_path / "r", diff, baseline=baseline, candidate=candidate)
    names_second = sorted(os.path.basename(path) for path in second.generate_all_plots())
    assert names_first == names_second


def test_subgroup_disparity_figure_is_written(report):
    written = report.plot_subgroup_disparity()
    assert written
    assert any(path.endswith("subgroup_disparity_clf_accuracy.png") for path in written)


def test_improvement_is_re_signed_by_polarity():
    """Right is better, whatever the metric. A raw delta would point the wrong way."""
    accuracy_drop = pd.Series({"metric": "clf_accuracy", "delta": -0.02})
    ece_drop = pd.Series({"metric": "ece_confidence", "delta": -0.02})
    assert ComparisonReport._improvement(accuracy_drop) == pytest.approx(-0.02)
    # Same delta, opposite improvement.
    assert ComparisonReport._improvement(ece_drop) == pytest.approx(0.02)


def test_improvement_of_an_unoriented_metric_is_zero():
    row = pd.Series({"metric": "detector", "delta": 5.0})
    assert ComparisonReport._improvement(row) == 0.0


def test_absent_and_degenerate_bars_are_hatched():
    """A zero-length bar is invisible; absent must not look like unchanged."""
    absent = pd.Series({"presence": "removed", "status_candidate": None})
    degenerate = pd.Series({"presence": "both", "status_candidate": "degenerate"})
    healthy = pd.Series({"presence": "both", "status_candidate": "ok"})
    assert ComparisonReport._bar_hatch(absent)
    assert ComparisonReport._bar_hatch(degenerate)
    assert ComparisonReport._bar_hatch(healthy) == ""


def test_delta_bars_pick_the_worst_method_per_cell(tmp_path, baseline):
    """Which method a bar describes must not depend on row order."""
    candidate = baseline.copy()
    # maxprob improves, the graph method collapses. The bar must show the collapse.
    candidate.loc[candidate["method_id"] == "baseline_maxprob", "auroc_error"] = 0.80
    candidate.loc[candidate["method_id"] == "graph_hybrid_distance", "auroc_error"] = 0.40

    diff = compare.compare(baseline, candidate)
    report = ComparisonReport(tmp_path / "r", diff, baseline=baseline, candidate=candidate)
    written = report.plot_delta_bars()
    assert written

    overall = report._overall_rows()
    rows = overall[(overall["cell_id"] == "reference") & (overall["metric"] == "auroc_error")]
    worst = min(
        (ComparisonReport._improvement(row), row) for _index, row in rows.iterrows()
    )[1]
    assert worst["method_id"] == "graph_hybrid_distance"


def test_headline_metrics_are_all_oriented():
    """A headline column with no direction would print a delta nobody can read."""
    for metric in HEADLINE_METRICS:
        assert compare.metric_direction(metric) is not None, metric
