"""Figures and tables for the uncertainty benchmark.

Follows `BiasMetricsTracker`'s shape (`save_dir` in the constructor,
`generate_all_plots()` as one entry point) and the house savefig idiom
(`tight_layout()`, then `dpi=300, bbox_inches='tight'`, then `close()`), with two
deliberate departures:

**`matplotlib.use("Agg")` is set explicitly**, not left to `$DISPLAY` being unset. A
report generated on a workstation with a display would otherwise pick an interactive
backend and behave differently from the same call on the server.

**Filenames are stable, with no timestamp.** Every existing tracker embeds one, which
makes output non-idempotent and unciteable -- a paper cannot reference
`reliability_20260807_143022.png`. The `run_id` already in the path supplies uniqueness.

Nothing here is reusable from the repo: there is no reliability diagram, no
risk-coverage curve, and no error bars anywhere. `DQNEvaluator`'s "Calibration Error"
panel is a scalar traced over epochs, and its `sklearn.calibration_curve` ECE is the
unweighted variant that `metrics.py` deliberately replaced -- so this does not call
into it.

Four figure-design rules, each preventing a specific misreading:

* **A reliability diagram always carries its bin-count histogram.** ECE is a weighted
  average over bins; without counts, a bin holding four samples looks as important as
  one holding forty thousand, and the empty-bin problem is invisible.
* **A risk-coverage curve always draws the oracle.** E-AURC is the area *between* the
  method and the oracle. Plotting only the method makes raw AURC look like the
  quantity of interest, and raw AURC is dominated by the base error rate -- so a
  stronger-but-worse-calibrated detector wins for the wrong reason.
* **Severity panels share an x-axis and never use dual y-axes.** Two series on
  independent axes invent a visual correlation that the data does not contain.
* **Gated cells get a hatch and a reason code, never a colour.** Status must not be
  colour-alone, and "skipped because incompatible" is categorically different from
  "scored badly" -- a colour scale would put them on one continuum.
"""

import json
import os

import numpy as np

import matplotlib

# Before pyplot: setting the backend after import has no effect.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow matplotlib.use
from matplotlib.patches import Patch  # noqa: E402

DPI = 300
FORMATS = ("png", "pdf")

#: Colour per method family, so a family reads as a family across every figure.
FAMILY_COLORS = {
    "baseline": "#4c72b0",
    "posthoc": "#dd8452",
    "ensemble": "#55a868",
    "head": "#c44e52",
    "graph": "#8172b3",
    "control": "#937860",
    "unknown": "#7f7f7f",
}

#: Marker per family, so the figures survive being printed in greyscale.
FAMILY_MARKERS = {
    "baseline": "o", "posthoc": "s", "ensemble": "^", "head": "D",
    "graph": "v", "control": "x", "unknown": ".",
}

#: Hatch per gate outcome. Deliberately not a colour.
STATUS_HATCH = {
    "skipped": "///",
    "broken": "xxx",
    "refused_low_coverage": "...",
    "degenerate": "\\\\\\",
    # Not a gate outcome: the cell *was* scored, but its ranking metrics were blanked
    # by `collapse_rank_equivalents` because they are identical to the representative
    # method by construction. It still needs a mark -- left blank it renders as an
    # unexplained white hole, indistinguishable from a cell nobody tried.
    "rank_equivalent": "|||",
}

#: Reason codes drawn inside unscored cells.
STATUS_ABBREVIATION = {
    "skipped": "skip",
    "broken": "broken",
    "refused_low_coverage": "cov",
    "degenerate": "degen",
    "rank_equivalent": "= base",
    "missing": "-",
}

#: Above ~4 series a legend stops being readable, so panels facet rather than overlay.
MAX_SERIES_PER_PANEL = 4


class UQReport:
    """Generates every benchmark figure and table into ``save_dir``."""

    def __init__(self, save_dir, results=None, records=None, title=None):
        """
        Args:
            save_dir: directory for figures and tables. Created if absent.
            results: tidy results DataFrame from ``scoring.score_cells``.
            records: optional ``{label: records DataFrame}`` for the per-sample
                figures (reliability, risk-coverage, ID-vs-OOD histograms), which
                cannot be drawn from aggregated results.
            title: optional suffix for figure titles.
        """
        self.save_dir = str(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.results = results
        self.records = dict(records or {})
        self.title = title
        self.written = []

    # -- infrastructure ----------------------------------------------------- #

    def _save(self, figure, name, tight=True):
        """Save in every format under a stable name. Returns the paths written.

        ``tight=False`` for figures that set their own geometry -- an explicit
        ``gridspec_kw`` hspace, or a legend anchored outside the axes. `tight_layout`
        warns and then overrides both, which is how the reliability panels lose their
        alignment and the gating legend gets clipped. `bbox_inches='tight'` below
        already handles the outside legend.
        """
        if tight:
            figure.tight_layout()
        paths = []
        for extension in FORMATS:
            path = os.path.join(self.save_dir, f"{name}.{extension}")
            figure.savefig(path, dpi=DPI, bbox_inches="tight")
            paths.append(path)
        plt.close(figure)
        self.written.extend(paths)
        return paths

    def _titled(self, axis, text):
        axis.set_title(f"{text} -- {self.title}" if self.title else text)

    @staticmethod
    def _family(results, method_id):
        if results is None or "family" not in results.columns:
            return "unknown"
        matches = results.loc[results["method_id"] == method_id, "family"]
        return matches.iloc[0] if len(matches) else "unknown"

    def _color(self, method_id):
        return FAMILY_COLORS.get(self._family(self.results, method_id),
                                 FAMILY_COLORS["unknown"])

    def _marker(self, method_id):
        return FAMILY_MARKERS.get(self._family(self.results, method_id),
                                  FAMILY_MARKERS["unknown"])

    # -- reliability -------------------------------------------------------- #

    def plot_reliability(self, records, name="reliability", n_bins=15,
                         label="", target="confidence"):
        """Reliability diagram with its bin-count histogram beneath.

        The histogram is not decoration. ECE is `sum (n_i/n) * |acc_i - conf_i|`, so a
        bin's contribution scales with its occupancy; a diagram without counts invites
        reading a four-sample bin's large gap as a large miscalibration.
        """
        from evaluation.uq.metrics import binary_calibration

        labels = records["label"].to_numpy(dtype=int)
        probabilities = records["prob"].to_numpy(dtype=float)
        calibration = binary_calibration(
            labels, probabilities, n_bins=n_bins, target=target
        )

        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(6, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        centers = (calibration.bin_edges[:-1] + calibration.bin_edges[1:]) / 2.0
        counts = np.asarray(calibration.bin_counts, dtype=float)
        occupied = counts > 0

        top.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
                 label="perfect calibration", zorder=1)
        top.plot(np.asarray(calibration.bin_confidence)[occupied],
                 np.asarray(calibration.bin_accuracy)[occupied],
                 marker="o", color="#4c72b0", linewidth=1.5,
                 label=label or "observed", zorder=3)
        # Vertical drop lines make the signed gap per bin readable, which a bar chart
        # of |gap| would hide.
        for center, confidence, accuracy, count in zip(
            centers, calibration.bin_confidence, calibration.bin_accuracy, counts
        ):
            if count > 0:
                top.plot([confidence, confidence], [confidence, accuracy],
                         color="#c44e52", linewidth=1, alpha=0.5, zorder=2)

        top.set_ylabel("accuracy" if target == "confidence" else "observed frequency")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.legend(loc="upper left", fontsize=8)
        self._titled(
            top,
            f"Reliability ({target}) -- ECE {calibration.ece:.4f}, "
            f"MCE {calibration.mce:.4f}, {calibration.n_empty_bins} empty bins"
        )

        width = (calibration.bin_edges[1] - calibration.bin_edges[0]) * 0.9
        bottom.bar(centers, counts, width=width, color="#8c8c8c")
        bottom.set_xlabel("confidence" if target == "confidence"
                          else "predicted probability")
        bottom.set_ylabel("samples")
        bottom.set_yscale("symlog")
        # Annotate empty bins: their absence from the top panel is the whole point.
        for center, count in zip(centers, counts):
            if count == 0:
                bottom.annotate("0", (center, 0), ha="center", va="bottom",
                                fontsize=7, color="#c44e52")

        # tight=False: the explicit gridspec hspace above is the alignment that makes
        # a bin line up with its count bar, and tight_layout would recompute it.
        return self._save(figure, name, tight=False)

    # -- risk-coverage ------------------------------------------------------ #

    def plot_risk_coverage(self, curves, name="risk_coverage"):
        """Risk-coverage curves with the oracle drawn, so E-AURC is visible.

        Args:
            curves: ``{method_id: records DataFrame}`` or
                ``{method_id: (records, score_column)}``.
        """
        from evaluation.uq.metrics import risk_coverage_curve
        from evaluation.uq.scoring import resolve_score

        figure, axis = plt.subplots(figsize=(6.5, 5))

        oracle_drawn = False
        summary = {}
        for method_id, value in curves.items():
            records, score_column = value if isinstance(value, tuple) else (
                value, self._default_score_column(method_id, value)
            )
            labels = records["label"].to_numpy(dtype=int)
            probabilities = records["prob"].to_numpy(dtype=float)
            scores = resolve_score(records, score_column)
            result = risk_coverage_curve(labels, probabilities, scores)

            if not oracle_drawn:
                # The oracle is a property of (labels, probabilities), not of the
                # score, so it is identical across methods on one records table and is
                # drawn once. E-AURC is the area between each curve and this line.
                oracle = self._oracle_curve(labels, probabilities, result.coverage)
                axis.plot(result.coverage, oracle, color="grey", linestyle="--",
                          linewidth=1.2, label="oracle", zorder=1)
                oracle_drawn = True

            axis.plot(result.coverage, result.risk, linewidth=1.5,
                      color=self._color(method_id),
                      marker=self._marker(method_id), markevery=max(
                          1, len(result.coverage) // 12),
                      markersize=4, label=f"{method_id} (E-AURC {result.eaurc:.4f})")
            summary[method_id] = {"aurc": result.aurc, "eaurc": result.eaurc,
                                  "aurc_oracle": result.aurc_oracle}

        axis.set_xlabel("coverage")
        axis.set_ylabel("risk (error rate on the retained fraction)")
        axis.set_xlim(0, 1)
        axis.legend(fontsize=8, loc="upper left")
        self._titled(axis, "Risk-coverage (shaded gap to oracle = E-AURC)")

        paths = self._save(figure, name)
        self._write_json(f"{name}_summary.json", summary)
        return paths

    @staticmethod
    def _oracle_curve(labels, probabilities, coverages):
        """Risk of an oracle that rejects every error first.

        Recomputed here rather than taken from `risk_coverage_curve`'s scalar so the
        gap can be *drawn*, not merely reported.
        """
        errors = (probabilities > 0.5).astype(int) != labels
        n = errors.size
        if n == 0:
            return np.zeros_like(coverages, dtype=float)
        # Correct samples first, so every retained prefix is error-free until the
        # correct ones run out.
        ordered = np.sort(errors.astype(float))
        risks = []
        for coverage in coverages:
            keep = max(1, int(round(coverage * n)))
            risks.append(ordered[:keep].mean())
        return np.asarray(risks)

    def _default_score_column(self, method_id, records):
        from evaluation.uq.registry import method_spec

        try:
            column = method_spec(method_id).score_column
        except (KeyError, ValueError):
            column = "u_maxprob"
        return column if column in records.columns or column.startswith("u_") else "u_maxprob"

    # -- severity ----------------------------------------------------------- #

    def plot_severity_panels(self, results=None, name="severity",
                             metrics=("clf_accuracy", "ece_confidence", "auroc_error")):
        """One row of panels sharing the x-axis. Never dual axes.

        Accuracy, ECE, and AUROC-of-error live on different scales and mean different
        things; overlaying two on twin axes would let a reader see a correlation that
        is an artifact of the axis choice.
        """
        results = self.results if results is None else results
        if results is None or results.empty:
            return []

        frame = results[results["status"] == "ok"].copy()
        if "severity" not in frame.columns or frame["severity"].nunique() < 2:
            return []

        available = [metric for metric in metrics if metric in frame.columns]
        if not available:
            return []

        corruptions = sorted(
            value for value in frame["corruption"].dropna().unique() if value != "none"
        )
        figure, axes = plt.subplots(
            1, len(available), figsize=(4.2 * len(available), 4.0), sharex=True
        )
        axes = np.atleast_1d(axes)

        for axis, metric in zip(axes, available):
            for corruption in corruptions:
                subset = frame[frame["corruption"].isin([corruption, "none"])]
                grouped = subset.groupby("severity")[metric].mean().sort_index()
                if grouped.notna().sum() < 2:
                    continue
                axis.plot(grouped.index, grouped.to_numpy(), marker="o",
                          linewidth=1.5, label=corruption)
            axis.set_xlabel("severity (0 = clean)")
            axis.set_ylabel(metric)
            axis.set_xticks(sorted(frame["severity"].dropna().unique()))
            if metric == available[0]:
                axis.legend(fontsize=8)
        self._titled(axes[0], "Degradation under corruption")

        return self._save(figure, name)

    # -- ID vs OOD ---------------------------------------------------------- #

    def plot_id_vs_ood(self, records, score_column, name="id_vs_ood"):
        """Score histograms for the ID and OOD partitions, plus the detection AUROC.

        The figure that shows *why* an OOD AUROC is what it is: two overlapping
        unimodal distributions read very differently from a bimodal one produced by a
        missing-value sentinel, and the scalar cannot distinguish them.
        """
        from evaluation.uq.holdouts import partition_records
        from evaluation.uq.metrics import ood_detection
        from evaluation.uq.scoring import resolve_score

        id_frame, ood_frame = partition_records(records)
        if len(ood_frame) == 0:
            return []

        id_scores = resolve_score(id_frame, score_column)
        ood_scores = resolve_score(ood_frame, score_column)
        detection = ood_detection(id_scores, ood_scores)

        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        finite = np.concatenate([
            id_scores[np.isfinite(id_scores)], ood_scores[np.isfinite(ood_scores)]
        ])
        if finite.size == 0:
            plt.close(figure)
            return []
        bins = np.linspace(float(finite.min()), float(finite.max()), 40)

        axis.hist(id_scores, bins=bins, density=True, alpha=0.55,
                  color="#4c72b0", label=f"in-distribution (n={len(id_frame)})")
        axis.hist(ood_scores, bins=bins, density=True, alpha=0.55,
                  color="#c44e52", label=f"held-out (n={len(ood_frame)})")
        axis.set_xlabel(score_column)
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
        flags = ", ".join(detection.status_flags) if detection.status_flags else "none"
        self._titled(
            axis,
            f"ID vs held-out -- AUROC {detection.auroc:.4f}, "
            f"FPR@95TPR {detection.fpr_at_95_tpr:.4f} [flags: {flags}]"
        )
        return self._save(figure, name)

    # -- gating heatmap ----------------------------------------------------- #

    def plot_gating_heatmap(self, results=None, name="gating", metric="eaurc"):
        """method x detector, with gated cells hatched and reason-coded.

        The visual statement of the model-agnostic requirement: a published matrix
        should show explained holes, not missing rows. Status is hatch plus text, never
        colour alone -- a colour scale would place "incompatible with this
        architecture" on the same continuum as "scored poorly".
        """
        results = self.results if results is None else results
        if results is None or results.empty:
            return []

        methods = sorted(results["method_id"].dropna().unique())
        detectors = sorted(results["detector"].dropna().unique())
        if not methods or not detectors:
            return []

        values = np.full((len(methods), len(detectors)), np.nan)
        statuses = np.empty((len(methods), len(detectors)), dtype=object)
        statuses[:] = "missing"

        # sort_index, or every .loc below emits a lexsort-depth PerformanceWarning.
        indexed = results.set_index(["method_id", "detector"], drop=False).sort_index()
        for row_index, method_id in enumerate(methods):
            for column_index, detector in enumerate(detectors):
                try:
                    row = indexed.loc[(method_id, detector)]
                except KeyError:
                    continue
                if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                    row = row.iloc[0]
                status = str(row.get("status", "missing"))
                value = row.get(metric) if metric in results.columns else None
                scored = (status == "ok" and value is not None
                          and np.isfinite(value))
                if scored:
                    values[row_index, column_index] = float(value)
                elif status == "ok" and row.get("rank_equivalent_to") is not None \
                        and not _is_missing(row.get("rank_equivalent_to")):
                    # Scored, but this metric is identical to the representative
                    # method's by construction. Marked as such rather than left blank.
                    status = "rank_equivalent"
                statuses[row_index, column_index] = status

        figure, axis = plt.subplots(
            figsize=(1.5 + 1.35 * len(detectors), 1.5 + 0.55 * len(methods))
        )
        image = axis.imshow(values, cmap="viridis_r", aspect="auto")
        figure.colorbar(image, ax=axis, label=metric, fraction=0.035)

        for row_index in range(len(methods)):
            for column_index in range(len(detectors)):
                status = statuses[row_index, column_index]
                if status == "ok":
                    value = values[row_index, column_index]
                    if np.isfinite(value):
                        axis.text(column_index, row_index, f"{value:.3f}",
                                  ha="center", va="center", fontsize=7, color="white")
                    continue
                # Hatch over a neutral fill, plus an abbreviated reason. Both channels,
                # so the distinction survives greyscale printing and colour blindness.
                axis.add_patch(plt.Rectangle(
                    (column_index - 0.5, row_index - 0.5), 1, 1,
                    facecolor="#e8e8e8",
                    hatch=STATUS_HATCH.get(status, "///"),
                    edgecolor="#555555", linewidth=0.5,
                ))
                axis.text(column_index, row_index, _abbreviate(status),
                          ha="center", va="center", fontsize=6, color="#333333")

        axis.set_xticks(range(len(detectors)))
        axis.set_xticklabels(detectors, rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(len(methods)))
        axis.set_yticklabels(methods, fontsize=8)
        self._titled(axis, f"Capability gating ({metric} where scored)")

        present = [status for status in np.unique(statuses)
                   if status not in ("ok", "missing")]
        if present:
            axis.legend(
                handles=[
                    Patch(facecolor="#e8e8e8", edgecolor="#555555",
                          hatch=STATUS_HATCH.get(status, "///"), label=status)
                    for status in present
                ],
                loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=7,
                title="not scored", title_fontsize=7,
            )
        # tight=False: the legend is anchored outside the axes, which tight_layout
        # cannot account for. bbox_inches='tight' at savefig keeps it in frame.
        return self._save(figure, name, tight=False)

    # -- tables ------------------------------------------------------------- #

    def write_results_csv(self, results=None, name="results.csv"):
        """The tidy long table. One row per (detector, method, condition)."""
        results = self.results if results is None else results
        if results is None:
            return None
        path = os.path.join(self.save_dir, name)
        # Sorted and fixed-precision so two reports over the same results are
        # byte-identical -- a paper artifact that changes on every regeneration cannot
        # be diffed or trusted.
        sort_columns = [
            column for column in
            ("detector", "method_id", "label", "holdout", "corruption", "severity")
            if column in results.columns
        ]
        ordered = results.sort_values(sort_columns) if sort_columns else results
        ordered.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        self.written.append(path)
        return path

    def write_results_markdown(self, results=None, name="results.md",
                               metrics=("auroc_error", "eaurc", "accuracy_at_0.8",
                                        "ece_confidence", "brier")):
        """A readable summary, with N/A cells explicit rather than blank."""
        results = self.results if results is None else results
        if results is None or results.empty:
            return None

        available = [metric for metric in metrics if metric in results.columns]
        lines = ["# Uncertainty benchmark results", ""]
        if self.title:
            lines += [self.title, ""]

        scored = results[results["status"] == "ok"]
        lines += [
            f"- cells scored: {len(scored)} of {len(results)}",
            f"- detectors: {', '.join(sorted(results['detector'].dropna().unique()))}",
            "",
        ]

        header = ["detector", "method_id", "label"] + available + ["status"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for _index, row in results.iterrows():
            cells = [str(row.get("detector", "")), str(row.get("method_id", "")),
                     str(row.get("label", ""))]
            for metric in available:
                value = row.get(metric)
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    # "n/a" not "0.000": a graph-distance method has no calibrated
                    # probability to be right about, and a zero would read as perfect.
                    cells.append("n/a")
                else:
                    cells.append(f"{float(value):.4f}")
            cells.append(str(row.get("status", "")))
            lines.append("| " + " | ".join(cells) + " |")

        skipped = results[results["status"].isin(("skipped", "broken"))]
        if not skipped.empty:
            lines += ["", "## Gated cells", ""]
            for _index, row in skipped.iterrows():
                lines.append(
                    f"- **{row.get('method_id')} x {row.get('detector')}**: "
                    f"{row.get('skip_reason', row.get('status_flags', ''))}"
                )

        path = os.path.join(self.save_dir, name)
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        self.written.append(path)
        return path

    def _write_json(self, name, payload):
        path = os.path.join(self.save_dir, name)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        self.written.append(path)
        return path

    # -- entry point -------------------------------------------------------- #

    def generate_all_plots(self):
        """Every figure and table the available inputs support.

        Idempotent: stable filenames, sorted tables, no timestamps, so regenerating
        over unchanged inputs rewrites the same bytes.
        """
        self.written = []

        self.write_results_csv()
        self.write_results_markdown()
        self.plot_gating_heatmap()
        self.plot_severity_panels()

        for label, records in sorted(self.records.items()):
            if records is None or len(records) == 0:
                continue
            safe = _safe_name(label)
            if "prob" in records.columns and "label" in records.columns:
                self.plot_reliability(records, name=f"reliability_{safe}", label=label)
                self.plot_risk_coverage({"maxprob": (records, "u_maxprob")},
                                        name=f"risk_coverage_{safe}")
            if "domain" in records.columns and (records["domain"] == "ood").any():
                for column in _uncertainty_columns(records)[:MAX_SERIES_PER_PANEL]:
                    self.plot_id_vs_ood(
                        records, column, name=f"id_vs_ood_{safe}_{_safe_name(column)}"
                    )

        return sorted(self.written)


def _uncertainty_columns(records):
    return sorted(
        column for column in records.columns if column.startswith("u_")
    )


def _abbreviate(status):
    return STATUS_ABBREVIATION.get(status, status[:6])


def _is_missing(value):
    """NaN-safe emptiness test. pandas hands back NaN, not None, for a blank cell."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return str(value).strip() == ""


def _safe_name(text):
    """Filename-safe label. `taming_transformer:VQGAN` contains a colon."""
    safe = str(text)
    for character in (":", "/", "\\", " ", ",", "*", "?", "="):
        safe = safe.replace(character, "_")
    return safe
