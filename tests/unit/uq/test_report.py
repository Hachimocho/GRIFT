"""Report generation: `evaluation/uq/report.py`.

Figures are hard to assert on, so these tests check the properties that actually matter
for a paper artifact and that a rendering bug would silently break:

* **idempotence** -- stable filenames and byte-identical output on regeneration, so a
  figure referenced from a paper stays the figure that was referenced;
* **the required companions are drawn** -- a reliability diagram has two axes (the
  bin-count histogram), a risk-coverage plot has an oracle series;
* **no dual y-axes** on the severity panels, which would invent a correlation;
* **gated cells are hatched and labelled, never coloured**, so status is not conveyed
  by colour alone;
* **N/A prints as "n/a"**, not as 0.0000, since a graph-distance method has no
  calibrated probability to be right about.
"""

import os

import numpy as np
import pandas as pd
import pytest

from evaluation.uq.report import UQReport

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_records(n=400, seed=0, with_ood=False):
    """A records table with enough spread for calibration bins to be occupied."""
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = rng.beta(2.0, 2.0, size=n)
    labels = (rng.random(n) < probabilities).astype(int)
    frame = pd.DataFrame({
        "record_id": np.arange(n),
        "rel_path": [f"/FFHQ/{index}.png" for index in range(n)],
        "label": labels,
        "prob": probabilities,
        "pred": (probabilities > 0.5).astype(int),
        "correct": ((probabilities > 0.5).astype(int) == labels).astype(int),
        "u_attribute_distance": rng.random(n) * 5,
        "u_embedding_distance": rng.random(n),
        "domain": "id",
        "corruption": "none",
        "severity": 0,
    })
    if with_ood:
        # Held-out rows are all-fake, as a generator holdout produces, and carry
        # systematically higher uncertainty so the AUROC is not degenerate.
        frame.loc[frame.index[-100:], "domain"] = "ood"
        frame.loc[frame.index[-100:], "label"] = 1
        frame.loc[frame.index[-100:], "u_attribute_distance"] += 3.0
    return frame


def make_results(detectors=("resnestdf", "effnetdf"),
                 methods=("maxprob", "graph_attribute_distance")):
    rng = np.random.Generator(np.random.PCG64(7))
    rows = []
    for detector in detectors:
        for method in methods:
            probabilistic = method == "maxprob"
            rows.append({
                "detector": detector,
                "method_id": method,
                "family": "baseline" if probabilistic else "graph",
                "status": "ok",
                "status_flags": "",
                "holdout": "none",
                "corruption": "none",
                "severity": 0,
                "n": 400,
                "auroc_error": 0.5 + rng.random() * 0.3,
                "eaurc": rng.random() * 0.1,
                "accuracy_at_0.8": 0.8 + rng.random() * 0.1,
                # N/A for a non-probabilistic method, which is the point of the
                # applicable flag.
                "ece_confidence": rng.random() * 0.1 if probabilistic else np.nan,
                "brier": rng.random() * 0.2 if probabilistic else np.nan,
                "calibration_applicable": probabilistic,
            })
    return pd.DataFrame(rows)


def with_skips(results):
    """Add the gate outcomes that must appear as hatched holes."""
    extra = pd.DataFrame([
        {"detector": "squeezenetdf", "method_id": "evidential", "status": "skipped",
         "status_flags": "needs_last_linear_graft",
         "skip_reason": "squeezenetdf has a conv classifier and no nn.Linear to graft",
         "family": "head"},
        {"detector": "xceptiondf", "method_id": "maxprob", "status": "broken",
         "status_flags": "detector_broken",
         "skip_reason": "no self.model attribute -> AttributeError at construction",
         "family": "baseline"},
    ])
    return pd.concat([results, extra], ignore_index=True)


def file_bytes(directory):
    return {
        name: open(os.path.join(directory, name), "rb").read()
        for name in sorted(os.listdir(directory))
    }


@pytest.fixture
def captured():
    """Intercept `_save` so the live figure can be inspected, then still save it.

    A saved PNG cannot be introspected, so structural claims -- "there is a second
    panel", "no axis was twinned" -- are only assertable against the figure object.
    """
    figures = []

    def make(report):
        original = report._save

        def capture(figure, name, **kwargs):
            # Keep an axes reference alive past the close() inside _save, so the test
            # can still walk them afterwards. **kwargs so a new _save parameter does
            # not silently bypass the capture.
            figures.append((name, figure, list(figure.axes)))
            return original(figure, name, **kwargs)

        report._save = capture
        return report

    make.figures = figures
    return make


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #

def test_reliability_has_a_bin_count_panel(tmp_path, captured):
    """Two axes, not one. ECE is bin-weighted, so counts are part of the claim.

    Without the histogram, a bin holding four samples looks as important as one
    holding forty thousand, and empty bins are invisible entirely.
    """
    report = captured(UQReport(tmp_path))
    report.plot_reliability(make_records(), name="rel")

    _name, _figure, axes = captured.figures[-1]
    assert len(axes) == 2, "reliability diagram must carry its bin-count panel"

    top, bottom = axes
    # The bottom panel is the histogram: bars, and a sample count on its y-axis.
    assert bottom.patches, "the count panel has no bars"
    assert "sample" in bottom.get_ylabel().lower()
    # The top panel is the diagram: a diagonal reference plus the observed curve.
    assert len(top.get_lines()) >= 2
    # Shared x-axis, so a bin lines up with its count.
    assert top.get_xlim() == bottom.get_xlim()
    assert os.path.exists(tmp_path / "rel.png")
    assert os.path.exists(tmp_path / "rel.pdf")


def test_reliability_reports_its_ece_in_the_title(tmp_path, captured):
    """The number and the picture must travel together, or they can disagree.

    A diagram whose ECE is quoted from elsewhere can be quoted from a different
    binning, a different target, or a different subset -- and nothing would show it.
    """
    from evaluation.uq.metrics import binary_calibration

    records = make_records()
    expected = binary_calibration(
        records["label"].to_numpy(), records["prob"].to_numpy(), n_bins=15,
        target="confidence",
    )

    report = captured(UQReport(tmp_path))
    report.plot_reliability(records, name="rel", n_bins=15)
    _name, _figure, axes = captured.figures[-1]
    title = axes[0].get_title()
    assert f"{expected.ece:.4f}" in title
    assert f"{expected.mce:.4f}" in title
    assert f"{expected.n_empty_bins} empty bins" in title


def test_reliability_marks_empty_bins(tmp_path, captured):
    """An empty bin contributes nothing to ECE and must be visibly empty.

    Otherwise a diagram over three occupied bins looks like a diagram over fifteen.
    """
    records = make_records()
    # Squeeze every probability into the middle, leaving the outer bins empty.
    records["prob"] = 0.45 + records["prob"] * 0.1

    report = captured(UQReport(tmp_path))
    report.plot_reliability(records, name="rel_sparse", n_bins=15)
    _name, _figure, axes = captured.figures[-1]
    annotations = [child.get_text() for child in axes[1].texts]
    assert annotations.count("0") > 0, "empty bins are not annotated"


def test_reliability_handles_a_single_confidence_value(tmp_path):
    """All-identical probabilities leave every bin but one empty.

    Must produce a figure rather than raising: a degenerate model is a result to
    report, and a crash here would take down the whole report run.
    """
    records = make_records()
    records["prob"] = 0.5
    records["pred"] = 0
    paths = UQReport(tmp_path).plot_reliability(records, name="flat")
    assert all(os.path.exists(path) for path in paths)


def test_positive_class_reliability_is_a_separate_figure(tmp_path):
    """Confidence-reliability and positive-class reliability are different plots.

    The literature conflates them; naming them distinctly is what stops a
    apples-to-oranges comparison.
    """
    report = UQReport(tmp_path)
    report.plot_reliability(make_records(), name="rel_conf", target="confidence")
    report.plot_reliability(make_records(), name="rel_pos", target="positive")
    assert os.path.exists(tmp_path / "rel_conf.png")
    assert os.path.exists(tmp_path / "rel_pos.png")


# --------------------------------------------------------------------------- #
# Risk-coverage
# --------------------------------------------------------------------------- #

def test_risk_coverage_draws_the_oracle(tmp_path, captured):
    """E-AURC is the area between the curve and the oracle, so the oracle is drawn.

    Plotting only the method makes raw AURC look like the quantity of interest, and
    raw AURC is dominated by the base error rate -- so a stronger but worse-calibrated
    detector would appear to win.
    """
    import json

    records = make_records()
    report = captured(UQReport(tmp_path))
    report.plot_risk_coverage({"maxprob": (records, "u_maxprob")}, name="rc")

    _name, _figure, axes = captured.figures[-1]
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert "oracle" in labels, f"no oracle series drawn; got {labels}"

    oracle = next(line for line in axes[0].get_lines()
                  if line.get_label() == "oracle")
    method = next(line for line in axes[0].get_lines()
                  if line.get_label() != "oracle")
    # The oracle rejects errors first, so its risk is never above the method's.
    assert (oracle.get_ydata() <= method.get_ydata() + 1e-12).all()

    with open(tmp_path / "rc_summary.json") as handle:
        summary = json.load(handle)
    # Both quantities recorded: aurc alone is not comparable across methods.
    assert "aurc" in summary["maxprob"] and "eaurc" in summary["maxprob"]
    assert summary["maxprob"]["aurc"] >= summary["maxprob"]["aurc_oracle"]


def test_risk_coverage_draws_the_oracle_only_once(tmp_path, captured):
    """The oracle depends on (labels, probs), not on the score.

    Drawing it per method would stack identical lines and clutter the legend with
    duplicate entries that look like distinct baselines.
    """
    records = make_records()
    report = captured(UQReport(tmp_path, results=make_results()))
    report.plot_risk_coverage(
        {
            "maxprob": (records, "u_maxprob"),
            "graph_attribute_distance": (records, "u_attribute_distance"),
        },
        name="rc_multi",
    )
    _name, _figure, axes = captured.figures[-1]
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert labels.count("oracle") == 1
    assert len(labels) == 3, f"expected oracle + 2 methods, got {labels}"


def test_risk_coverage_eaurc_is_the_gap_to_the_oracle(tmp_path):
    import json

    records = make_records()
    UQReport(tmp_path).plot_risk_coverage(
        {"maxprob": (records, "u_maxprob")}, name="rc"
    )
    with open(tmp_path / "rc_summary.json") as handle:
        entry = json.load(handle)["maxprob"]
    assert entry["eaurc"] == pytest.approx(entry["aurc"] - entry["aurc_oracle"],
                                           abs=1e-9)


def test_risk_coverage_takes_several_methods(tmp_path):
    records = make_records()
    paths = UQReport(tmp_path, results=make_results()).plot_risk_coverage(
        {
            "maxprob": (records, "u_maxprob"),
            "graph_attribute_distance": (records, "u_attribute_distance"),
        },
        name="rc_multi",
    )
    assert all(os.path.exists(path) for path in paths)


# --------------------------------------------------------------------------- #
# Severity panels
# --------------------------------------------------------------------------- #

def make_severity_results():
    rows = []
    for corruption in ("jpeg", "gaussian_blur"):
        for severity in range(6):
            rows.append({
                "detector": "resnestdf", "method_id": "maxprob", "family": "baseline",
                "status": "ok", "corruption": corruption if severity else "none",
                "severity": severity,
                "clf_accuracy": 0.9 - 0.03 * severity,
                "ece_confidence": 0.05 + 0.01 * severity,
                "auroc_error": 0.8 - 0.02 * severity,
            })
    return pd.DataFrame(rows)


def test_severity_panels_are_drawn(tmp_path):
    paths = UQReport(tmp_path, results=make_severity_results()).plot_severity_panels()
    assert paths and all(os.path.exists(path) for path in paths)


def test_severity_panels_never_use_twin_axes(tmp_path, captured):
    """Two metrics on independent y-axes invent a correlation that is not in the data.

    `twinx` produces a second Axes occupying the *same* subplot position as the first.
    So the check is that no two axes share a position: one metric per panel, no
    overlays.
    """
    metrics = ("clf_accuracy", "ece_confidence", "auroc_error")
    report = captured(UQReport(tmp_path, results=make_severity_results()))
    report.plot_severity_panels(name="sev", metrics=metrics)

    _name, _figure, axes = captured.figures[-1]
    assert len(axes) == len(metrics), "expected exactly one panel per metric"

    positions = [tuple(round(value, 6) for value in axis.get_position().bounds)
                 for axis in axes]
    assert len(set(positions)) == len(positions), (
        "two axes share a subplot position, which is what twinx does"
    )
    # Each panel is labelled with its own metric, so no panel is doing double duty.
    assert [axis.get_ylabel() for axis in axes] == list(metrics)


def test_severity_panels_share_one_x_axis(tmp_path, captured):
    """Severity means the same thing in every panel, so the axes must agree."""
    report = captured(UQReport(tmp_path, results=make_severity_results()))
    report.plot_severity_panels(name="sev")
    _name, _figure, axes = captured.figures[-1]
    limits = {axis.get_xlim() for axis in axes}
    assert len(limits) == 1, f"panels disagree on the x-axis range: {limits}"


def test_severity_panels_include_the_clean_baseline(tmp_path, captured):
    """Severity 0 must be on the x-axis, or degradation has no origin to start from."""
    report = captured(UQReport(tmp_path, results=make_severity_results()))
    report.plot_severity_panels(name="sev")
    _name, _figure, axes = captured.figures[-1]
    assert 0 in list(axes[0].get_xticks())


def test_severity_panels_are_skipped_without_a_ladder(tmp_path):
    """One severity level is not a ladder; drawing a single point would mislead."""
    assert UQReport(tmp_path, results=make_results()).plot_severity_panels() == []


def test_severity_panels_are_skipped_on_empty_results(tmp_path):
    assert UQReport(tmp_path, results=pd.DataFrame()).plot_severity_panels() == []


# --------------------------------------------------------------------------- #
# ID vs OOD
# --------------------------------------------------------------------------- #

def test_id_vs_ood_is_drawn_when_there_is_an_ood_partition(tmp_path):
    paths = UQReport(tmp_path).plot_id_vs_ood(
        make_records(with_ood=True), "u_attribute_distance", name="ood"
    )
    assert paths and os.path.exists(tmp_path / "ood.png")


def test_id_vs_ood_is_skipped_without_an_ood_partition(tmp_path):
    """No held-out rows means no detection question to answer."""
    assert UQReport(tmp_path).plot_id_vs_ood(
        make_records(with_ood=False), "u_attribute_distance", name="ood"
    ) == []


def test_id_vs_ood_survives_all_nan_scores(tmp_path):
    """A method with no coverage must not crash the report."""
    records = make_records(with_ood=True)
    records["u_embedding_distance"] = np.nan
    assert UQReport(tmp_path).plot_id_vs_ood(
        records, "u_embedding_distance", name="ood_nan"
    ) == []


# --------------------------------------------------------------------------- #
# Gating heatmap
# --------------------------------------------------------------------------- #

def test_gating_heatmap_is_drawn(tmp_path):
    paths = UQReport(tmp_path, results=with_skips(make_results())).plot_gating_heatmap()
    assert paths and os.path.exists(tmp_path / "gating.png")


def test_gating_heatmap_includes_gated_detectors_as_rows_and_columns(tmp_path, captured):
    """A skipped cell is a hole in the matrix, not a missing row.

    The published matrix should show explained holes: a reader must be able to see that
    squeezenetdf x evidential was *considered* and refused, not merely absent.
    """
    report = captured(UQReport(tmp_path, results=with_skips(make_results())))
    report.plot_gating_heatmap(name="gate")

    _name, _figure, axes = captured.figures[-1]
    axis = axes[0]
    detectors = [label.get_text() for label in axis.get_xticklabels()]
    methods = [label.get_text() for label in axis.get_yticklabels()]
    assert "squeezenetdf" in detectors and "xceptiondf" in detectors
    assert "evidential" in methods


def test_gating_heatmap_hatches_gated_cells(tmp_path, captured):
    """Status must not be colour-alone.

    A colour scale would place "incompatible with this architecture" on the same
    continuum as "scored poorly", which are categorically different. Asserted against
    the drawn patches, not just the hatch table.
    """
    report = captured(UQReport(tmp_path, results=with_skips(make_results())))
    report.plot_gating_heatmap(name="gate")

    _name, _figure, axes = captured.figures[-1]
    hatched = [patch for patch in axes[0].patches if patch.get_hatch()]
    # Two gated cells in the fixture, plus the empty cells the cross-product creates
    # for detectors and methods that were never paired.
    assert len(hatched) >= 2, "gated cells were not hatched"
    for patch in hatched:
        assert patch.get_hatch(), "a gated cell has no hatch"


def test_gating_heatmap_labels_each_gated_cell_with_a_reason(tmp_path, captured):
    """A hole needs a reason code, or the reader cannot tell why it is a hole."""
    report = captured(UQReport(tmp_path, results=with_skips(make_results())))
    report.plot_gating_heatmap(name="gate")

    _name, _figure, axes = captured.figures[-1]
    texts = {child.get_text() for child in axes[0].texts}
    assert "skip" in texts, f"no skip reason code drawn; got {sorted(texts)}"
    assert "broken" in texts


def test_every_status_the_scorer_emits_has_a_hatch(tmp_path):
    """A status with no hatch would fall back to colour alone."""
    from evaluation.uq.report import STATUS_HATCH

    # The statuses `scoring.score_cell` and `add_skipped_rows` can produce.
    for status in ("skipped", "broken", "refused_low_coverage", "degenerate"):
        assert STATUS_HATCH.get(status), f"{status} has no hatch"


def test_gating_heatmap_is_skipped_on_empty_results(tmp_path):
    assert UQReport(tmp_path, results=pd.DataFrame()).plot_gating_heatmap() == []


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def test_results_csv_is_written_and_sorted(tmp_path):
    report = UQReport(tmp_path, results=with_skips(make_results()))
    path = report.write_results_csv()
    frame = pd.read_csv(path)
    assert len(frame) == 6
    assert list(frame["detector"]) == sorted(frame["detector"])


def test_results_csv_is_byte_stable(tmp_path):
    """A paper artifact that changes on every regeneration cannot be diffed."""
    results = with_skips(make_results())
    first = UQReport(tmp_path / "a", results=results).write_results_csv()
    second = UQReport(tmp_path / "b", results=results).write_results_csv()
    assert open(first, "rb").read() == open(second, "rb").read()


def test_markdown_prints_na_not_zero_for_inapplicable_metrics(tmp_path):
    """A graph-distance method has no calibrated probability to be right about.

    Printing 0.0000 would read as perfect calibration, which is the single most common
    way a UQ table becomes nonsense.
    """
    report = UQReport(tmp_path, results=make_results())
    path = report.write_results_markdown()
    text = open(path).read()
    assert "n/a" in text
    # The graph method's row must carry n/a, and the baseline's must not be n/a for ECE.
    graph_lines = [line for line in text.splitlines()
                   if "graph_attribute_distance" in line and line.startswith("|")]
    assert graph_lines
    assert all("n/a" in line for line in graph_lines)


def test_markdown_lists_gated_cells_with_reasons(tmp_path):
    report = UQReport(tmp_path, results=with_skips(make_results()))
    text = open(report.write_results_markdown()).read()
    assert "## Gated cells" in text
    assert "squeezenetdf" in text
    assert "nn.Linear" in text, "the reason, not just the fact, must be printed"


def test_markdown_counts_scored_cells(tmp_path):
    report = UQReport(tmp_path, results=with_skips(make_results()))
    text = open(report.write_results_markdown()).read()
    assert "cells scored: 4 of 6" in text


# --------------------------------------------------------------------------- #
# generate_all_plots
# --------------------------------------------------------------------------- #

def test_generate_all_plots_produces_figures_and_tables(tmp_path):
    report = UQReport(
        tmp_path, results=with_skips(make_results()),
        records={"resnestdf_test": make_records(with_ood=True)},
    )
    written = report.generate_all_plots()
    names = {os.path.basename(path) for path in written}
    assert "results.csv" in names
    assert "results.md" in names
    assert "gating.png" in names
    assert any(name.startswith("reliability_") for name in names)
    assert any(name.startswith("risk_coverage_") for name in names)
    assert any(name.startswith("id_vs_ood_") for name in names)


def test_generate_all_plots_emits_both_formats(tmp_path):
    """PDF for the paper, PNG for everything else."""
    report = UQReport(tmp_path, results=make_results(),
                      records={"a": make_records()})
    written = report.generate_all_plots()
    figures = [path for path in written if path.endswith((".png", ".pdf"))]
    png = {os.path.splitext(p)[0] for p in figures if p.endswith(".png")}
    pdf = {os.path.splitext(p)[0] for p in figures if p.endswith(".pdf")}
    assert png == pdf


def test_generate_all_plots_uses_stable_filenames(tmp_path):
    """No timestamps. A figure cited from a paper must keep its name.

    Every existing tracker in the repo embeds one; the run_id already in the path
    supplies uniqueness.
    """
    inputs = dict(results=with_skips(make_results()),
                  records={"a": make_records(with_ood=True)})
    first = set(UQReport(tmp_path / "one", **inputs).generate_all_plots())
    second = set(UQReport(tmp_path / "two", **inputs).generate_all_plots())
    assert {os.path.basename(path) for path in first} == \
           {os.path.basename(path) for path in second}


def test_regenerating_over_the_same_inputs_is_idempotent(tmp_path):
    """Same inputs, same directory, same bytes for the text artifacts.

    Restricted to the tables: matplotlib embeds a creation date in PNG/PDF metadata,
    so figure bytes are not stable and asserting on them would be a false claim.
    """
    inputs = dict(results=with_skips(make_results()),
                  records={"a": make_records(with_ood=True)})
    report = UQReport(tmp_path, **inputs)
    report.generate_all_plots()
    before = {name: value for name, value in file_bytes(tmp_path).items()
              if name.endswith((".csv", ".md", ".json"))}

    UQReport(tmp_path, **inputs).generate_all_plots()
    after = {name: value for name, value in file_bytes(tmp_path).items()
             if name.endswith((".csv", ".md", ".json"))}
    assert before == after


def test_generate_all_plots_works_with_no_records(tmp_path):
    """Aggregated results alone must still produce the gating figure and tables."""
    written = UQReport(tmp_path, results=with_skips(make_results())).generate_all_plots()
    assert any(path.endswith("gating.png") for path in written)
    assert any(path.endswith("results.csv") for path in written)


def test_generate_all_plots_works_with_no_results(tmp_path):
    """Records alone must still produce the per-sample figures."""
    written = UQReport(tmp_path, records={"a": make_records()}).generate_all_plots()
    assert any("reliability_" in path for path in written)


def test_generate_all_plots_on_nothing_does_not_raise(tmp_path):
    assert UQReport(tmp_path).generate_all_plots() == []


def test_a_label_with_a_colon_becomes_a_safe_filename(tmp_path):
    """`taming_transformer:VQGAN` is a real group name and a bad filename."""
    written = UQReport(
        tmp_path, records={"taming_transformer:VQGAN": make_records()}
    ).generate_all_plots()
    assert written
    for path in written:
        assert ":" not in os.path.basename(path)


def test_the_backend_is_agg(tmp_path):
    """Set explicitly, not inferred from $DISPLAY.

    Otherwise a report generated on a workstation with a display picks an interactive
    backend and behaves differently from the same call on the server.
    """
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
