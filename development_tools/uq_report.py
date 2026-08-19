#!/usr/bin/env python3
"""Score record tables and generate the benchmark report.

Offline, and separate from the training CLI on purpose. The web UI's results path is
log scraping (`_parse_final_test_metrics` plus stdout regexes) with no concept of a
sweep matrix, and report artifacts belong next to the paper, versioned -- not
regenerated on every page load.

Takes record tables written by `--uq-records` (plus any ensemble tables from
`launch_ensemble.py`), scores every applicable method, and writes figures, a tidy
`results.csv`, and a `results.md`.

    # One run.
    python development_tools/uq_report.py \
        --records run_outputs/<run_id>/<config>/records_test.csv.gz \
        --out run_outputs/reports/<run_id>

    # A comparison, with an ensemble and a corruption cell.
    python development_tools/uq_report.py \
        --records clean=run_outputs/a/c/records_test.csv.gz \
        --records jpeg3=run_outputs/b/c/records_test.csv.gz:jpeg:3 \
        --ensemble run_outputs/ensembles/<id>/records_test.csv.gz \
        --out run_outputs/reports/comparison

Each `--records` value is ``[label=]path[:corruption:severity]``. The corruption and
severity default to whatever the table itself records, so they only need stating when
overriding.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.uq.records import default_meta_path, read_records
from evaluation.uq.registry import UQ_METHODS, expand_matrix
from evaluation.uq.report import UQReport
from evaluation.uq.scoring import (
    Cell,
    add_skipped_rows,
    collapse_rank_equivalents,
    score_cells,
    score_ood,
)

#: Methods whose scores come from columns a records table may or may not carry. Each is
#: scored only when its column is present (or derivable from `prob`), so a run without
#: an uncertainty head simply contributes fewer rows rather than failing.
CANDIDATE_METHODS = (
    "baseline_maxprob", "baseline_entropy", "baseline_margin",
    "graph_attribute_distance", "graph_embedding_distance", "graph_hybrid_distance",
    "graph_degree_only",
    "mc_dropout", "evidential", "batchensemble", "sngp",
    "ivalue", "ivalue_rank",
    "deep_ensemble", "temperature_scaling",
)

#: Columns `resolve_score` can derive from `prob` alone, so they count as available even
#: when absent from the table.
DERIVABLE = ("u_maxprob", "u_entropy", "u_margin")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score UQ record tables and write the benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--records", action="append", default=[], metavar="SPEC",
                        help="[label=]path[:corruption:severity]. Repeatable.")
    parser.add_argument("--ensemble", action="append", default=[], metavar="PATH",
                        help="Aggregated ensemble table from launch_ensemble.py. "
                             "Repeatable.")
    parser.add_argument("--out", required=True, help="Report directory.")
    parser.add_argument("--detector", default=None,
                        help="Detector name. Read from each table's manifest when "
                             "omitted.")
    parser.add_argument("--title", default=None, help="Suffix for figure titles.")
    parser.add_argument("--n-boot", type=int, default=0,
                        help="Bootstrap resamples for AUROC-error CIs. 0 disables. "
                             "Seeded, so the interval is reproducible.")
    parser.add_argument("--gate-detectors", default=None,
                        help="Comma-separated detectors to include in the gating "
                             "heatmap even if no records exist for them. This is what "
                             "turns the heatmap into a statement about what was "
                             "*considered*, not merely what happened to run.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the sha256 check against each table's manifest.")
    return parser.parse_args(argv)


def parse_records_spec(spec):
    """``[label=]path[:corruption:severity]`` -> (label, path, corruption, severity)."""
    label, _, remainder = spec.partition("=")
    if not remainder:
        label, remainder = None, spec

    corruption, severity = None, None
    # rsplit from the right, and only on a suffix that looks like :name:digit -- a
    # Windows drive letter or a path containing a colon must not be mistaken for a
    # corruption spec.
    parts = remainder.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        remainder, corruption, severity = parts[0], parts[1], int(parts[2])

    if label is None:
        label = os.path.basename(os.path.dirname(remainder)) or os.path.basename(remainder)
    return label, remainder, corruption, severity


def load_manifest(path):
    meta_path = default_meta_path(path)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def available_methods(frame):
    """Methods whose primary column this table can supply."""
    columns = set(frame.columns)
    found = []
    for method_id in CANDIDATE_METHODS:
        spec = UQ_METHODS.get(method_id)
        if spec is None:
            continue
        column = spec.primary_column
        if column in columns or column in DERIVABLE:
            found.append(method_id)
    return found


def build_cells(entries, args):
    """One Cell per (table, available method). Returns (cells, records_by_label)."""
    cells, records = [], {}
    for label, path, corruption, severity in entries:
        frame = read_records(path, verify=not args.no_verify)
        manifest = load_manifest(path)
        records[label] = frame

        detector = args.detector or manifest.get("detector") or "unknown"
        # The table's own labels are authoritative unless explicitly overridden: they
        # were written by the code that applied the corruption.
        cell_corruption = corruption or _single(frame, "corruption", "none")
        cell_severity = severity if severity is not None else int(
            _single(frame, "severity", 0)
        )

        methods = available_methods(frame)
        if not methods:
            print(f"  {label}: no scoreable method columns, skipping")
            continue
        print(f"  {label}: {len(frame)} rows, detector {detector}, "
              f"corruption {cell_corruption}/{cell_severity}, "
              f"{len(methods)} method(s)")

        for method_id in methods:
            cells.append(Cell(
                detector=detector,
                method_id=method_id,
                score_column=UQ_METHODS[method_id].primary_column,
                frame=frame,
                holdout=manifest.get("holdout_id") or "none",
                corruption=cell_corruption,
                severity=cell_severity,
                coverage=float(manifest.get("coverage", 1.0)),
                determinism_mode=(manifest.get("determinism_mode")
                                  or _nested(manifest, "determinism", "mode")
                                  or "unknown"),
                manifest_sha256=manifest.get("sha256_records"),
                seed=manifest.get("seed"),
                cost_forward_passes=UQ_METHODS[method_id].cost_forward_passes,
                cost_training_runs=(
                    manifest.get("cost_training_runs")
                    or UQ_METHODS[method_id].cost_training_runs
                ),
                extra={"label": label, "records_path": path},
            ))
    return cells, records


def _observed_capabilities(records, detectors, args):
    """Per-detector gate config inferred from the tables actually present.

    The plan-time gate asks "could a run configured this way produce this method?".
    Here the run has already happened, so evidence beats prediction: an ensemble table
    carrying three members' worth of columns proves MULTI_CHECKPOINT regardless of what
    any config said.
    """
    n_members = 0
    for path in args.ensemble:
        manifest = load_manifest(path)
        n_members = max(n_members, int(manifest.get("n_members", 0)))
    if not n_members:
        return {}
    # Applied to every gated detector: the ensemble is per-detector in principle, but
    # the manifest records which one, and mixing detectors is already refused upstream.
    return {detector: {"n_members": n_members} for detector in detectors}


def _single(frame, column, default):
    """The table's value for a column that should be constant across its rows."""
    if column not in frame.columns or frame.empty:
        return default
    values = frame[column].dropna().unique()
    return values[0] if len(values) == 1 else default


def _nested(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def ood_rows(records, args):
    """OOD-detection rows for every table carrying a held-out partition.

    Reported separately from classification because a generator holdout is all-fake:
    detection is the question a single-class partition can answer, and accuracy is not.
    """
    rows = []
    for label, frame in sorted(records.items()):
        if "domain" not in frame.columns or not (frame["domain"] == "ood").any():
            continue
        from evaluation.uq.holdouts import partition_records

        id_frame, ood_frame = partition_records(frame)
        detector = args.detector or "unknown"
        for method_id in available_methods(frame):
            spec = UQ_METHODS[method_id]
            if not spec.uncertainty_columns:
                continue
            try:
                row = score_ood(
                    id_frame, ood_frame, spec.primary_column, detector, method_id,
                    extra={"label": label, "partition": "ood_detection"},
                )
            except Exception as exc:  # noqa: BLE001 - one bad method must not stop the rest
                print(f"  ood {label}/{method_id}: {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
        print(f"  {label}: OOD detection over {len(ood_frame)} held-out rows")
    return rows


def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    if not args.records and not args.ensemble:
        raise SystemExit("nothing to report: pass --records and/or --ensemble")

    entries = [parse_records_spec(spec) for spec in args.records]
    entries += [
        (f"ensemble_{os.path.basename(os.path.dirname(path))}", path, None, None)
        for path in args.ensemble
    ]

    print(f"Loading {len(entries)} record table(s):")
    cells, records = build_cells(entries, args)
    if not cells:
        raise SystemExit("no scoreable cells found")

    # require_comparable=False: this CLI deliberately pools corruption severities and
    # holdouts into one table, which is the whole point of the tidy long format. The
    # per-figure code re-splits on those columns.
    print(f"\nScoring {len(cells)} cell(s)...")
    results = score_cells(cells, n_boot=args.n_boot, require_comparable=False)
    results = collapse_rank_equivalents(results)

    detection = ood_rows(records, args)
    if detection:
        results = pd.concat([results, pd.DataFrame(detection)], ignore_index=True)

    # Gate every candidate method against every detector, so the heatmap shows what was
    # considered and refused -- explained holes rather than absent rows.
    gate_detectors = sorted(set(
        (args.gate_detectors.split(",") if args.gate_detectors else [])
        + [value for value in results["detector"].dropna().unique()]
    ))
    if gate_detectors:
        # Capabilities that a *records table* already demonstrates. The plan-time gate
        # cannot know these -- it reasons about what a training config could produce --
        # so without this the gate would mark deep_ensemble "skipped" for a detector
        # whose ensemble table is sitting right there being scored.
        gate_config = _observed_capabilities(records, gate_detectors, args)
        _runnable, decisions = expand_matrix(
            list(CANDIDATE_METHODS), gate_detectors, config=gate_config
        )
        # Drop skips for cells that were actually scored: a method that produced a row
        # is not skipped, whatever the plan-time gate concluded.
        scored_pairs = {
            (str(row.method_id), str(row.detector))
            for row in results.itertuples() if getattr(row, "status", "") == "ok"
        }
        decisions = [
            decision for decision in decisions
            if (decision.method_id, decision.detector) not in scored_pairs
        ]
        results = add_skipped_rows(results, decisions)

    counts = results["status"].value_counts().to_dict()
    print(f"\n{len(results)} result row(s): {counts}")

    report = UQReport(args.out, results=results, records=records, title=args.title)
    written = report.generate_all_plots()
    print(f"\nWrote {len(written)} file(s) to {args.out}")
    for path in written:
        print(f"  {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
