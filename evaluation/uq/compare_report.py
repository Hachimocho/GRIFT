"""Figures and tables for a baseline-vs-candidate comparison.

Subclasses `UQReport` rather than reimplementing it, so this inherits the choices that
make the benchmark's output citable: `matplotlib.use("Agg")` before pyplot, DPI 300 in
both PNG and PDF, stable timestamp-free filenames, hatched-not-coloured gate outcomes,
and `n/a` where a metric does not apply.

Three presentation rules, each preventing a specific misreading of a diff:

* **Bars point the way they mean.** Every delta is redrawn as *improvement*, so right is
  always better whether the metric is accuracy or ECE. Plotting raw deltas side by side
  forces the reader to remember each column's orientation, and the one they misremember
  is the one they act on.
* **Absent is not zero.** A cell that disappeared between the two runs gets its own
  section and a hatch, never a bar at 0.0 -- which would read as "no change".
* **Provenance leads.** If the seed, the node cache, or the determinism mode differ, that
  is printed above the numbers, because it changes what the numbers can mean.
"""

import os

import numpy as np

from evaluation.uq.compare import (
    hard_failure_summary, improvements, metric_direction, provenance_warnings,
    regressions, summarize,
)
from evaluation.uq.report import DPI, STATUS_HATCH, UQReport, _safe_name  # noqa: F401

import matplotlib.pyplot as plt

#: Colours for the two verdicts. Distinguishable in greyscale by position (better bars
#: point right, worse point left), so colour is not carrying the meaning alone.
DIRECTION_COLORS = {
    "better": "#55a868",
    "worse": "#c44e52",
    "same": "#bbbbbb",
    "n_a": "#7f7f7f",
}

#: Metrics worth a headline row, grouped so a panel holds commensurable quantities.
#: Anything absent from the comparison is skipped rather than drawn empty.
METRIC_PANELS = (
    ("Discrimination", ("clf_accuracy", "clf_auroc", "clf_eer")),
    ("Uncertainty quality", ("auroc_error", "eaurc", "accuracy_at_0.8")),
    ("Calibration", ("ece_confidence", "brier", "nll")),
    ("Fairness", ("disparity_range_clf_accuracy", "worst_group_clf_accuracy",
                  "disparity_range_ece_confidence")),
)

#: Columns of the headline table, in order.
HEADLINE_METRICS = (
    "clf_accuracy", "ece_confidence", "auroc_error", "eaurc",
    "worst_group_clf_accuracy", "disparity_range_clf_accuracy",
)

#: At most this many cells per delta-bar panel, so labels stay readable. The rest go to
#: a second figure rather than being dropped -- see `plot_delta_bars`.
MAX_CELLS_PER_FIGURE = 20


class ComparisonReport(UQReport):
    """Writes the comparison table, the regression sections, and the delta figures."""

    def __init__(
        self,
        save_dir,
        comparison,
        baseline=None,
        candidate=None,
        paired=None,
        title=None,
        baseline_manifest=None,
        candidate_manifest=None,
    ):
        """
        Args:
            save_dir: directory for figures and tables. Created if absent.
            comparison: long diff from `compare.compare`.
            baseline / candidate: the two scored results tables, for provenance and for
                the per-cell status columns.
            paired: optional ``{cell_id: paired_accuracy_test result}``, so the headline
                table can say whether an accuracy delta is real rather than only how
                large it is.
            title: optional suffix for figure titles.
            baseline_manifest / candidate_manifest: sweep manifests, for the git commit
                and node-cache digest printed in the provenance section.
        """
        super().__init__(save_dir, results=candidate, title=title)
        self.comparison = comparison
        self.baseline = baseline
        self.candidate = candidate
        self.paired = dict(paired or {})
        self.baseline_manifest = baseline_manifest or {}
        self.candidate_manifest = candidate_manifest or {}

    # -- tables ------------------------------------------------------------- #

    def write_comparison_csv(self, name="comparison.csv"):
        """The full long diff, sorted so two reports over the same inputs match."""
        if self.comparison is None or self.comparison.empty:
            return None
        sort_columns = [
            column for column in
            ("cell_id", "method_id", "subgroup_dimension", "subgroup_value", "metric")
            if column in self.comparison.columns
        ]
        ordered = (
            self.comparison.sort_values(sort_columns)
            if sort_columns else self.comparison
        )
        path = os.path.join(self.save_dir, name)
        ordered.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        self.written.append(path)
        return path

    def write_comparison_markdown(self, name="comparison.md"):
        """The report a human reads: verdict, provenance, headline, then detail."""
        if self.comparison is None or self.comparison.empty:
            lines = ["# Comparison", "", "No comparable rows. ",
                     "Both results tables were empty, or they share no measurement."]
            return self._write_lines(name, lines)

        counts = summarize(self.comparison)
        # Collapsed to one row per broken measurement: a single degenerate cell blanks
        # every metric on every subgroup slice, which is hundreds of rows and one problem.
        failures = hard_failure_summary(self.comparison)
        worse = regressions(self.comparison)
        better = improvements(self.comparison)

        lines = ["# Comparison: baseline vs candidate", ""]
        if self.title:
            lines += [self.title, ""]

        # Verdict first. A reader who stops after one line should still learn the thing
        # that matters most.
        if not failures.empty:
            lines += [
                f"**{len(failures)} hard failure(s)**: a measurement disappeared or "
                f"stopped being computable. See *Hard failures* below.", "",
            ]
            lines += [
                "| cell | method | reason | status | metrics | subgroups |",
                "|---|---|---|---|---|---|",
            ]
            for _index, row in failures.head(30).iterrows():
                lines.append(
                    f"| {row.get('cell_id', '')} | {row.get('method_id', '')} | "
                    f"{row.get('reason', '')} | "
                    f"{row.get('status_baseline')} → {row.get('status_candidate')} | "
                    f"{row.get('metrics_affected', 0)} ({row.get('metrics', '')}) | "
                    f"{row.get('subgroups_affected', 0)} |"
                )
            if len(failures) > 30:
                lines.append(f"| ... | | | | {len(failures) - 30} more in "
                             f"comparison.csv | |")
            lines.append("")
        elif counts["worse"]:
            lines += [
                f"No hard failures. {counts['worse']} metric(s) moved the wrong way -- "
                f"reported below, not treated as errors.", "",
            ]
        else:
            lines += ["No hard failures and no regressions.", ""]

        lines += [
            f"- rows compared: {counts['rows']}",
            f"- better: {counts['better']} | worse: {counts['worse']} | "
            f"same: {counts['same']} | n/a: {counts['n_a']}",
            f"- cells added: {counts['added']} | removed: {counts['removed']} | "
            f"newly degenerate: {counts['newly_degenerate']}",
            "",
        ]

        lines += self._provenance_lines()
        lines += self._headline_lines()
        lines += self._section("Regressions", worse, limit=40)
        lines += self._section("Improvements", better, limit=20)
        lines += self._gated_lines()

        return self._write_lines(name, lines)

    # -- figures ------------------------------------------------------------ #

    def plot_delta_bars(self, name="delta", metrics=None):
        """Improvement bars per cell, one panel per metric family.

        Bars are signed by *improvement*, not by raw delta: a 0.01 drop in ECE and a 0.01
        rise in accuracy both point right. Written as one figure per `METRIC_PANELS`
        group, with cells split across numbered figures past `MAX_CELLS_PER_FIGURE` so a
        large sweep does not silently truncate.
        """
        if self.comparison is None or self.comparison.empty:
            return []

        overall = self._overall_rows()
        if overall.empty:
            return []

        written = []
        for panel_name, panel_metrics in METRIC_PANELS:
            selected = [
                metric for metric in (metrics or panel_metrics)
                if metric in set(overall["metric"])
            ]
            if not selected:
                continue
            subset = overall[overall["metric"].isin(selected)]
            cells = sorted(subset[self._cell_column()].dropna().unique())
            if not cells:
                continue

            for chunk_index in range(0, len(cells), MAX_CELLS_PER_FIGURE):
                chunk = cells[chunk_index:chunk_index + MAX_CELLS_PER_FIGURE]
                suffix = "" if len(cells) <= MAX_CELLS_PER_FIGURE else (
                    f"_{chunk_index // MAX_CELLS_PER_FIGURE + 1}"
                )
                figure = self._delta_panel(subset, chunk, selected, panel_name)
                if figure is None:
                    continue
                written += self._save(
                    figure, f"{name}_{_safe_name(panel_name.lower())}{suffix}"
                )
        return written

    def plot_subgroup_disparity(self, name="subgroup_disparity", metric="clf_accuracy"):
        """Baseline vs candidate disparity range, per subgroup dimension.

        The fairness counterpart to the delta bars: paired bars rather than signed ones,
        because the absolute level of disparity matters as much as its movement -- going
        from 0.22 to 0.20 is an improvement and still a large gap.
        """
        if self.comparison is None or self.comparison.empty:
            return []

        column = f"disparity_range_{metric}"
        rows = self.comparison[self.comparison["metric"] == column]
        if rows.empty or "subgroup_dimension" not in rows.columns:
            return []

        grouped = rows.groupby("subgroup_dimension", dropna=False)[
            ["baseline", "candidate"]
        ].max()
        grouped = grouped.dropna(how="all")
        if grouped.empty:
            return []

        positions = np.arange(len(grouped))
        width = 0.38
        figure, axis = plt.subplots(figsize=(max(5.0, 1.4 * len(grouped) + 2.5), 3.6))
        axis.bar(positions - width / 2, grouped["baseline"].to_numpy(), width,
                 label="baseline", color="#4c72b0")
        axis.bar(positions + width / 2, grouped["candidate"].to_numpy(), width,
                 label="candidate", color="#dd8452")
        axis.set_xticks(positions)
        axis.set_xticklabels(list(grouped.index), rotation=20, ha="right")
        axis.set_ylabel(f"max - min {metric}\nacross subgroups (lower is better)")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.3)
        self._titled(axis, f"Subgroup disparity in {metric}")
        return self._save(figure, f"{name}_{_safe_name(metric)}")

    # -- entry point -------------------------------------------------------- #

    def generate_all_plots(self):
        """Every comparison artifact the available inputs support. Idempotent."""
        self.written = []
        self.write_comparison_csv()
        self.write_comparison_markdown()
        self.plot_delta_bars()
        self.plot_subgroup_disparity()
        if self.candidate is not None and not self.candidate.empty:
            # The candidate's own scored table, so the report stands alone rather than
            # requiring the sweep directory alongside it.
            self.write_results_csv(self.candidate, name="candidate_results.csv")
        return sorted(self.written)

    # -- internals ---------------------------------------------------------- #

    def _cell_column(self):
        """Whichever column identifies a configuration in this comparison."""
        for column in ("cell_id", "detector", "method_id"):
            if column in self.comparison.columns:
                return column
        return self.comparison.columns[0]

    def _overall_rows(self):
        """Whole-set rows, plus the fairness reductions, excluding per-subgroup detail.

        A per-subgroup row for every group would multiply the bars by eight and bury the
        headline; the disparity reductions carry the fairness signal instead.
        """
        frame = self.comparison
        if "subgroup_value" not in frame.columns:
            return frame
        keep = frame["subgroup_value"].isin(["all", "disparity"])
        return frame[keep]

    def _delta_panel(self, subset, cells, metrics, panel_name):
        """One horizontal bar chart: improvement per (cell, metric).

        A cell has one row per uncertainty method, and for the ranking and calibration
        metrics the method is the whole point. Rather than picking an arbitrary one, this
        shows the **worst** row per (cell, metric) and names the method it came from: the
        question a comparison answers is "did anything get worse", so the bar that matters
        is the worst one, and labelling it keeps the figure honest about which method it
        describes.
        """
        cell_column = self._cell_column()
        rows = subset[subset[cell_column].isin(cells)]
        if rows.empty:
            return None

        labels, values, colors, hatches = [], [], [], []
        for cell in cells:
            for metric in metrics:
                match = rows[(rows[cell_column] == cell) & (rows["metric"] == metric)]
                if match.empty:
                    continue
                scored = [(self._improvement(row), row) for _index, row in match.iterrows()]
                improvement, row = min(scored, key=lambda pair: pair[0])
                method = str(row.get("method_id", "") or "")
                suffix = f"  [{method}]" if method and len(match) > 1 else ""
                labels.append(f"{cell} / {metric}{suffix}")
                values.append(improvement)
                colors.append(DIRECTION_COLORS.get(str(row["direction"]), "#7f7f7f"))
                hatches.append(self._bar_hatch(row))

        if not labels:
            return None

        height = max(3.0, 0.30 * len(labels) + 1.4)
        figure, axis = plt.subplots(figsize=(8.0, height))

        positions = np.arange(len(labels))
        bars = axis.barh(positions, values, color=colors, edgecolor="#333333",
                         linewidth=0.4)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

        # A zero-length bar is invisible, which makes "measured, and identical" look the
        # same as "never measured". Both get an explicit mark, and they get *different*
        # marks: a tick at zero for unchanged, an x for absent.
        for position, value, row_hatch in zip(positions, values, hatches):
            if value != 0.0:
                continue
            axis.plot(
                [0.0], [position],
                marker="x" if row_hatch else "|",
                color="#333333", markersize=7 if row_hatch else 10,
                markeredgewidth=1.4, linestyle="none",
            )

        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlabel("improvement (right is better, whatever the metric's polarity)")
        axis.grid(axis="x", alpha=0.3)
        axis.legend(handles=self._legend_handles(), fontsize=7, frameon=False,
                    loc="lower right")
        self._titled(axis, f"{panel_name}: worst method per cell")
        return figure

    @staticmethod
    def _bar_hatch(row):
        """A hatch for a bar whose number is absent or no longer trustworthy."""
        if str(row.get("presence")) != "both":
            return STATUS_HATCH.get("skipped", "///")
        status = str(row.get("status_candidate"))
        if status in ("degenerate", "refused_low_coverage", "refused", "skipped",
                      "broken"):
            return STATUS_HATCH.get(status, "\\\\\\")
        return ""

    @staticmethod
    def _legend_handles():
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        return [
            Patch(facecolor=DIRECTION_COLORS["better"], edgecolor="#333333",
                  label="better"),
            Patch(facecolor=DIRECTION_COLORS["worse"], edgecolor="#333333",
                  label="worse"),
            Line2D([0], [0], marker="|", color="#333333", linestyle="none",
                   markersize=9, label="unchanged"),
            Line2D([0], [0], marker="x", color="#333333", linestyle="none",
                   markersize=7, label="absent or degenerate"),
        ]

    @staticmethod
    def _improvement(row):
        """The delta, re-signed so positive always means better."""
        delta = row.get("delta")
        if delta is None or (isinstance(delta, float) and not np.isfinite(delta)):
            return 0.0
        orientation = metric_direction(row.get("metric"))
        if orientation == "lower":
            return -float(delta)
        if orientation == "higher":
            return float(delta)
        return 0.0

    def _provenance_lines(self):
        lines = ["## Provenance", ""]
        for label, manifest in (
            ("baseline", self.baseline_manifest), ("candidate", self.candidate_manifest),
        ):
            if not manifest:
                # Said out loud: a results.csv given by bare path has no manifest beside
                # it, so the commit and node cache behind it are unknown -- which is
                # exactly what a reader needs to be told, not left to infer from silence.
                lines.append(
                    f"- **{label}**: no manifest found, so its commit, determinism mode, "
                    f"and node cache are unknown. Pass a sweep id, or keep "
                    f"manifest.json beside the results.csv."
                )
                continue
            git = manifest.get("git", {}) or {}
            dirty = " (dirty tree)" if git.get("dirty") else ""
            lines.append(
                f"- **{label}**: suite `{manifest.get('suite', '?')}`, "
                f"tag `{manifest.get('tag') or '-'}`, "
                f"commit `{str(git.get('commit', '?'))[:12]}`{dirty}, "
                f"determinism `{manifest.get('determinism', '?')}`, "
                f"node cache `{str(manifest.get('node_cache_sha256') or '-')[:12]}`"
            )

        warnings = provenance_warnings(self.baseline, self.candidate)
        if warnings:
            lines += ["", "**Read these before the numbers:**", ""]
            lines += [f"- {warning}" for warning in warnings]

        baseline_cache = (self.baseline_manifest or {}).get("node_cache_sha256")
        candidate_cache = (self.candidate_manifest or {}).get("node_cache_sha256")
        if baseline_cache and candidate_cache and baseline_cache != candidate_cache:
            lines += [
                "",
                "- **the node cache was rebuilt between these runs**, so the two "
                "swept different samples. Deltas here are not attributable to the code "
                "change; re-establish the baseline against the current cache.",
            ]
        return lines + [""]

    def _headline_lines(self):
        """One row per cell, the metrics that matter, plus the paired-test verdict."""
        overall = self._overall_rows()
        if overall.empty:
            return []
        cell_column = self._cell_column()
        available = [
            metric for metric in HEADLINE_METRICS if metric in set(overall["metric"])
        ]
        if not available:
            return []

        header = [cell_column] + [f"Δ {metric}" for metric in available] + ["paired p"]
        lines = ["## Headline", "",
                 "Δ is candidate − baseline. An arrow marks the direction that counts "
                 "as better for that metric.", "",
                 "| " + " | ".join(header) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]

        for cell in sorted(overall[cell_column].dropna().unique()):
            rows = overall[overall[cell_column] == cell]
            cells = [str(cell)]
            for metric in available:
                match = rows[rows["metric"] == metric]
                cells.append(self._delta_text(match))
            paired = self.paired.get(str(cell), {})
            if paired.get("applicable"):
                cells.append(f"{paired['p_value']:.3g}")
            elif paired:
                cells.append("n/a")
            else:
                cells.append("-")
            lines.append("| " + " | ".join(cells) + " |")
        return lines + [""]

    @staticmethod
    def _delta_text(match):
        """`+0.0123 ↑better` style cell text, or an explicit absence marker."""
        if match.empty:
            return "-"
        row = match.iloc[0]
        if str(row["presence"]) == "removed":
            return "**gone**"
        if str(row["presence"]) == "added":
            return "*new*"
        delta = row.get("delta")
        if delta is None or (isinstance(delta, float) and not np.isfinite(delta)):
            return "n/a"
        direction = str(row["direction"])
        mark = {"better": "✓", "worse": "✗", "same": "=", "n_a": ""}.get(direction, "")
        return f"{float(delta):+.4f} {mark}".strip()

    def _section(self, heading, rows, limit=None):
        if rows is None or rows.empty:
            return [f"## {heading}", "", "None.", ""]

        lines = [f"## {heading}", ""]
        shown = rows if limit is None else rows.head(limit)
        if limit is not None and len(rows) > limit:
            # Never a silent truncation: a reader has to know the list continues.
            lines.append(
                f"Showing {len(shown)} of {len(rows)}; the rest are in comparison.csv."
            )
            lines.append("")

        columns = ["cell", "method", "subgroup", "metric", "baseline", "candidate",
                   "delta"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")

        cell_column = self._cell_column()
        for _index, row in shown.iterrows():
            entries = [
                str(row.get(cell_column, "")), str(row.get("method_id", "")),
                _subgroup_text(row), str(row.get("metric", "")),
                _number(row.get("baseline")), _number(row.get("candidate")),
                _number(row.get("delta"), signed=True),
            ]
            lines.append("| " + " | ".join(entries) + " |")
        return lines + [""]

    def _gated_lines(self):
        """Cells the candidate refused or skipped, with the reason."""
        if self.candidate is None or self.candidate.empty:
            return []
        if "status" not in self.candidate.columns:
            return []
        gated = self.candidate[
            self.candidate["status"].isin(
                ("skipped", "broken", "refused", "refused_low_coverage", "degenerate")
            )
        ]
        if gated.empty:
            return []
        lines = ["## Gated or degenerate in the candidate", ""]
        for _index, row in gated.iterrows():
            reason = row.get("skip_reason") or row.get("status_flags") or ""
            lines.append(
                f"- **{row.get('method_id')} x {row.get('detector')}** "
                f"({row.get('status')}): {reason}"
            )
        return lines + [""]

    def _write_lines(self, name, lines):
        path = os.path.join(self.save_dir, name)
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        self.written.append(path)
        return path


def _subgroup_text(row):
    """`gt_race=2`, `disparity`, or `overall` -- never a bare blank."""
    value = str(row.get("subgroup_value", "all"))
    if value in ("all", "nan", ""):
        return "overall"
    if value == "disparity":
        return f"{row.get('subgroup_dimension', '')} disparity"
    return f"{row.get('subgroup_dimension', '')}={value}"


def _number(value, signed=False):
    """Fixed-precision, with `n/a` for absent -- never `0.0000`."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.4f}" if signed else f"{number:.4f}"


__all__ = ["ComparisonReport", "DIRECTION_COLORS", "HEADLINE_METRICS", "METRIC_PANELS"]
