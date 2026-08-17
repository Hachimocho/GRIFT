#!/usr/bin/env python3
"""Run a matrix of configurations, score it, and diff it against a baseline.

The development loop this exists for:

    # Once, on known-good code: establish the baseline.
    python development_tools/sweep.py run --suite standard --tag baseline
    python development_tools/sweep.py promote <sweep-id>

    # After a change: run the same matrix and see what moved.
    python development_tools/sweep.py run --suite standard --compare-to baseline

    # Cheaper: only the axis you touched.
    python development_tools/sweep.py run --suite standard --only traversal \\
        --compare-to baseline

    # Free: check the matrix before spending any GPU time.
    python development_tools/sweep.py plan --suite standard

Every number comes from the per-sample record tables that `--uq-records` writes, scored
through `evaluation/uq/scoring.py` -- the same path the benchmark uses. The stdout metrics
dict is deliberately not an input: it carries batch means of raw uncertainty signals on
incomparable scales.

Cells run as subprocesses through the web UI's `GPUQueueManager`, so they inherit GPU
discovery, memory-based admission, `CUDA_VISIBLE_DEVICES` pinning, and process
monitoring, and a sweep-launched run is byte-for-byte the same kind of run as a
UI-launched one.

Exit codes: 0 clean, 1 usage or setup error, 2 hard failure -- a cell that did not
complete, produced no records, or stopped producing a metric the baseline had. Metric
movement is reported, never fatal: run-to-run movement is a finding, not a build error.
Add `--strict-gate` to fail on regressions too.
"""

import argparse
import copy
import json
import os
import secrets
import sys
import time

#: Repository root, from this file's location rather than the working directory.
#:
#: Every path below is anchored here because `GPUQueueManager` launches each cell with
#: `cwd=<repo root>` regardless of where the caller stood. Resolving these relative to the
#: caller's cwd instead split the two apart: run from `development_tools/`, the cells wrote
#: their records to `<repo>/run_outputs/<run-id>/` while the sweep looked for them under
#: `development_tools/run_outputs/<run-id>/`, found nothing, and reported three perfectly
#: good runs as having produced no records.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, REPO_ROOT)

from development_tools.sweep_suites import (
    SuiteError, expand, format_plan, load_suite, summarize as summarize_cells,
)

#: Where training runs write `determinism.json` and their record tables. Must match the
#: directory the cells themselves use, which is `run_outputs` relative to the repo root.
RUN_OUTPUTS_DIR = os.path.join(REPO_ROOT, "run_outputs")

#: Where sweeps write their artifacts.
SWEEPS_DIR = os.path.join(RUN_OUTPUTS_DIR, "sweeps")

#: Where promoted baselines live. Small enough to commit, so the git history of this
#: directory *is* the history of the project's baselines.
BASELINES_DIR = os.path.join(REPO_ROOT, "benchmarks")

#: The queue's own run metadata and logs, likewise anchored so a sweep launched from a
#: subdirectory does not create a second `web_ui/runs` tree beside itself.
RUNS_DIR = os.path.join(REPO_ROOT, "web_ui", "runs")

#: Metrics the comparison reports by default. Chosen as one per question -- can it
#: classify, is it calibrated, does its uncertainty rank errors, how much is lost to
#: selective prediction, and is any of it worse for one demographic group.
DEFAULT_METRICS = (
    "clf_accuracy", "clf_auroc", "clf_eer",
    "auroc_error", "eaurc", "accuracy_at_0.8",
    "ece_confidence", "brier", "nll",
)

#: Metrics the fairness reduction is computed over.
DISPARITY_METRICS = ("clf_accuracy", "ece_confidence", "auroc_error")

#: Methods scored from whatever columns a table happens to carry. Mirrors
#: `uq_report.CANDIDATE_METHODS`: a run without an uncertainty head contributes fewer
#: rows rather than failing.
CANDIDATE_METHODS = (
    "baseline_maxprob", "baseline_entropy", "baseline_margin",
    "graph_attribute_distance", "graph_embedding_distance", "graph_hybrid_distance",
    "graph_degree_only",
    "mc_dropout", "evidential", "batchensemble", "sngp",
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_HARD_FAILURE = 2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_matrix_options(sub):
        sub.add_argument("--suite", default="standard",
                         help="Suite name: smoke, standard, or full.")
        sub.add_argument("--suite-file", default=None,
                         help="JSON file whose keys merge over the built-in suite, so a "
                              "variant need not restate every axis.")
        sub.add_argument("--axes", default=None,
                         help="Comma-separated axes to expand, overriding the suite's.")
        sub.add_argument("--cross", default=None,
                         help="Comma-separated axes to expand factorially against each "
                              "other instead of one at a time.")
        sub.add_argument("--only", action="append", default=[],
                         help="Restrict to `axis` or `axis=value`. Repeatable. The "
                              "reference cell is always kept.")
        sub.add_argument("--allow-broken", action="store_true",
                         help="Include detectors the capability table marks broken. They "
                              "are expected to fail; this exists to test the failure path.")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         dest="overrides",
                         help="Set any run-config key on every cell, e.g. "
                              "--set fair_train=true --set num_workers=8. Repeatable. "
                              "Keys are the config names in ARG_MAPPING (underscores, not "
                              "dashes); an unknown key is refused with the list of valid "
                              "ones. This is the whole training CLI, without sweep.py "
                              "having to mirror ninety flags.")

    plan = subparsers.add_parser(
        "plan", help="Expand and gate the matrix, run nothing.")
    add_matrix_options(plan)
    plan.add_argument("--json", action="store_true",
                      help="Emit the cell list as JSON instead of a table.")

    run = subparsers.add_parser(
        "run", help="Run the matrix, score it, and write a report.")
    add_matrix_options(run)
    run.add_argument("--tag", default=None,
                     help="Human label for this sweep, recorded in the manifest.")
    run.add_argument("--sweep-id", default=None,
                     help="Resume this sweep: cells already complete are skipped. "
                          "Generated when omitted.")
    run.add_argument("--force", action="store_true",
                     help="With --sweep-id, re-run cells that already completed.")
    run.add_argument("--compare-to", default=None,
                     help="Baseline to diff against once the sweep finishes: a sweep id, "
                          "a path to a results.csv, `baseline` for this suite's promoted "
                          "baseline, or `baseline:<suite>`.")
    run.add_argument("--out", default=None,
                     help="Report directory. Defaults to <sweep dir>/report.")
    run.add_argument("--data-root", default=None,
                     help="AI-Face root, if the automatic discovery needs overriding.")
    run.add_argument("--cache-file", default=None,
                     help="Node cache every cell loads. Overrides the suite's.")
    run.add_argument("--seed", type=int, default=None,
                     help="Master seed for every cell. Overrides the suite's.")
    run.add_argument("--determinism", default=None, choices=["strict", "fast"],
                     help="Overrides the suite's `strict`. `fast` makes small deltas "
                          "unattributable; the report says so.")
    run.add_argument("--num-epochs", type=int, default=None,
                     help="Overrides the suite's epoch count.")
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--timeout-hours", type=float, default=12.0)
    run.add_argument("--launch-only", action="store_true",
                     help="Queue the cells and exit without waiting or scoring. Resume "
                          "with --sweep-id once they finish.")
    run.add_argument("--score-only", action="store_true",
                     help="Skip launching; score the cells this sweep already ran. "
                          "Requires --sweep-id.")
    run.add_argument("--n-boot", type=int, default=0,
                     help="Bootstrap resamples for AUROC-error CIs. 0 disables.")
    run.add_argument("--no-verify", action="store_true",
                     help="Skip the sha256 check on each record table.")
    run.add_argument("--strict-gate", action="store_true",
                     help="Also exit nonzero when a metric regresses, not only on hard "
                          "failures.")

    compare = subparsers.add_parser(
        "compare", help="Diff two scored results tables.")
    compare.add_argument("--baseline", required=True,
                         help="Sweep id, results.csv path, `baseline:<suite>`.")
    compare.add_argument("--candidate", required=True, help="Same forms as --baseline.")
    compare.add_argument("--out", default=None,
                         help="Report directory. Defaults to "
                              "run_outputs/reports/compare_<candidate>.")
    compare.add_argument("--title", default=None)
    compare.add_argument("--abs-tolerance", type=float, default=0.0,
                         help="Deltas within this are called `same`. 0 requires exact "
                              "equality, which is right when both runs were strict and "
                              "same-seeded.")
    compare.add_argument("--rel-tolerance", type=float, default=0.0,
                         help="As --abs-tolerance, relative to the baseline value.")
    compare.add_argument("--strict-gate", action="store_true",
                         help="Exit nonzero on regressions too, not only hard failures.")

    promote = subparsers.add_parser(
        "promote", help="Record a sweep's results as this suite's baseline.")
    promote.add_argument("sweep_id")
    promote.add_argument("--suite", default=None,
                         help="Baseline name. Read from the sweep manifest when omitted.")
    promote.add_argument("--allow-partial", action="store_true",
                         help="Promote even though some cells did not complete. The "
                               "missing cells then read as `removed` in every later diff.")

    subparsers.add_parser("list", help="List sweeps and promoted baselines.")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "plan":
            return command_plan(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "compare":
            return command_compare(args)
        if args.command == "promote":
            return command_promote(args)
        if args.command == "list":
            return command_list(args)
    except SuiteError as error:
        print(f"\nERROR: {error}")
        return EXIT_USAGE
    except SweepError as error:
        print(f"\nERROR: {error}")
        return EXIT_USAGE
    return EXIT_USAGE


class SweepError(RuntimeError):
    """Raised for a usage or setup problem the user can fix."""


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #

def build_cells(args):
    """Expand the matrix described by the CLI options."""
    suite = load_suite(args.suite, suite_file=args.suite_file)
    apply_overrides(suite, args)
    cells = expand(
        suite,
        axes=_split(getattr(args, "axes", None)),
        cross=_split(getattr(args, "cross", None)),
        only=getattr(args, "only", None),
        allow_broken=getattr(args, "allow_broken", False),
    )
    return suite, cells


def apply_overrides(suite, args):
    """Fold CLI overrides into the suite's reference cell, before expansion.

    Applied to the reference rather than to each expanded cell so an axis variant that
    deliberately sets the same key still wins.
    """
    reference = suite["reference"]
    for attribute, key in (
        ("data_root", "data_root"), ("cache_file", "cache_file"),
        ("seed", "seed"), ("num_epochs", "num_epochs"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            reference[key] = value
    determinism = getattr(args, "determinism", None)
    if determinism:
        # FORCED sets strict; an explicit override belongs in the reference *and* has to
        # survive the FORCED merge, so it is recorded on the suite for expand() to use.
        suite.setdefault("forced_overrides", {})["determinism"] = determinism
        reference["determinism"] = determinism

    # `--set` last, and into forced_overrides as well as the reference, so it beats both
    # the suite's own value and anything in FORCED. An explicit flag from the command line
    # is the most specific instruction available.
    for key, value in parse_overrides(getattr(args, "overrides", None)).items():
        reference[key] = value
        suite.setdefault("forced_overrides", {})[key] = value


def parse_overrides(entries):
    """`["fair_train=true", "num_workers=8"]` -> `{"fair_train": True, "num_workers": 8}`.

    Keys are validated against the queue's argument table, because a key it does not know
    is silently dropped on the way to the CLI -- the run would then quietly ignore the
    setting and report numbers that look real.
    """
    from web_ui.gpu_queue_manager import ARG_MAPPING, DEFAULT_ON_KEYS

    valid = set(ARG_MAPPING) | set(DEFAULT_ON_KEYS)
    overrides = {}
    for entry in entries or []:
        key, separator, raw = str(entry).partition("=")
        if not separator:
            raise SweepError(
                f"--set expects KEY=VALUE, got {entry!r}. For a boolean flag write "
                f"{key.strip() or 'key'}=true."
            )
        # `--set fair-train=true` is the natural thing to type, since the training flag is
        # spelled `--fair-train`; only the config key uses underscores. Both are accepted.
        normalized = key.strip().lstrip("-").replace("-", "_")
        if normalized not in valid:
            raise SweepError(
                f"--set key {key.strip()!r} is not a run-config key, so it would be "
                f"dropped on the way to test_hierarchical.py."
                + _did_you_mean(normalized, valid)
                + "\nThe full list is ARG_MAPPING in web_ui/gpu_queue_manager.py."
            )
        overrides[normalized] = _coerce(raw.strip())
    return overrides


def _did_you_mean(key, valid, limit=6):
    """A suggestion built from substring overlap and edit distance, or ``''``."""
    import difflib

    close = [name for name in sorted(valid) if key and (key in name or name in key)]
    close += [
        name for name in difflib.get_close_matches(key, sorted(valid), n=limit, cutoff=0.6)
        if name not in close
    ]
    if not close:
        return ""
    return f" Did you mean: {', '.join(close[:limit])}?"


def _coerce(raw):
    """`"true"` -> True, `"8"` -> 8, `"0.5"` -> 0.5, anything else stays a string.

    Types matter downstream: `_build_command_args` emits a bare flag for a True bool and
    *nothing at all* for False, so `fair_train="true"` as a string would reach argparse as
    a value for a store_true flag rather than enabling it.
    """
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def command_plan(args):
    _suite, cells = build_cells(args)
    if args.json:
        print(json.dumps([cell.to_dict() for cell in cells], indent=2, sort_keys=True))
        return EXIT_OK

    print(format_plan(cells))
    problems = _unroutable(cells)
    if problems:
        print("\nWARNING: these config keys would be dropped by the queue's argument "
              "table, so the runs would silently differ from the plan:")
        for cell_id, keys in sorted(problems.items()):
            print(f"  {cell_id}: {', '.join(keys)}")
        return EXIT_USAGE
    return EXIT_OK


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def command_run(args):
    suite, cells = build_cells(args)

    problems = _unroutable(cells)
    if problems:
        raise SweepError(
            "refusing to launch: these config keys are not in the queue's argument "
            "table, so they would be silently dropped:\n"
            + "\n".join(f"  {cell_id}: {', '.join(keys)}"
                        for cell_id, keys in sorted(problems.items()))
        )

    runnable = [cell for cell in cells if cell.runnable]
    if not runnable:
        raise SweepError("every cell in this matrix was gated; nothing to run")

    # Validated before anything is written: this used to be checked after the sweep
    # directory and manifest had been created, so a mistyped invocation left an empty
    # sweep behind for `list` to report.
    if args.score_only and not args.sweep_id:
        raise SweepError(
            "--score-only needs --sweep-id: there is nothing to score for a sweep that "
            "has not run"
        )

    sweep_id = args.sweep_id or _generate_sweep_id(args.suite)
    sweep_dir = os.path.join(SWEEPS_DIR, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    manifest_path = os.path.join(sweep_dir, "manifest.json")

    manifest = _load_json(manifest_path) or {}
    manifest.update({
        "sweep_id": sweep_id,
        "suite": args.suite,
        "tag": args.tag,
        "description": suite.get("description", ""),
        "determinism": runnable[0].config.get("determinism"),
        "seed": runnable[0].config.get("seed"),
        "git": _git_state(),
        "node_cache": runnable[0].config.get("cache_file"),
        "node_cache_sha256": _cache_digest(runnable[0].config.get("cache_file")),
        "created": manifest.get("created") or _timestamp(),
        "updated": _timestamp(),
    })
    cell_records = manifest.setdefault("cells", {})
    for cell in cells:
        entry = cell_records.setdefault(cell.cell_id, {})
        entry.update(cell.to_dict())
        entry.setdefault("status", "gated" if not cell.runnable else "pending")
    _write_json(manifest_path, manifest)

    print(f"Sweep {sweep_id}: suite {args.suite!r}"
          + (f", tag {args.tag!r}" if args.tag else ""))
    counts = summarize_cells(cells)
    print(f"  {counts['runnable']} runnable, {counts['skipped']} gated")
    if manifest["node_cache_sha256"]:
        print(f"  node cache {manifest['node_cache']} "
              f"({manifest['node_cache_sha256'][:12]})")
    print(f"  artifacts -> {sweep_dir}")

    if not args.score_only:
        pending = _pending_cells(runnable, cell_records, force=args.force)
        if not pending:
            print("\nEvery runnable cell already completed; nothing to launch. "
                  "Pass --force to re-run them.")
        else:
            _launch(pending, sweep_id, manifest_path, manifest, args)
            if args.launch_only:
                print(f"\nQueued. Score them later with:\n  python "
                      f"{os.path.relpath(__file__)} run --suite {args.suite} "
                      f"--sweep-id {sweep_id} --score-only")
                return EXIT_OK

    results, paired_inputs = _score(cells, cell_records, args)
    if results is None or results.empty:
        raise SweepError(
            "no cell produced a scoreable record table. Check "
            f"{os.path.join(sweep_dir, 'manifest.json')} for per-cell status, and "
            "web_ui/runs/<run_id>.log for the failures."
        )

    results_path = os.path.join(sweep_dir, "results.csv")
    results.to_csv(results_path, index=False, float_format="%.17g", lineterminator="\n")
    print(f"\nWrote {results_path} ({len(results)} row(s))")

    manifest["updated"] = _timestamp()
    manifest["results"] = os.path.relpath(results_path)
    _write_json(manifest_path, manifest)

    _report_own(results, args.out or os.path.join(sweep_dir, "report"), sweep_id)

    if not args.compare_to:
        _print_own_summary(results)
        return _hard_failure_exit(cell_records)

    return _compare_and_report(
        baseline_ref=args.compare_to,
        candidate_results=results,
        candidate_manifest=manifest,
        candidate_paired=paired_inputs,
        out=args.out or os.path.join(sweep_dir, "report"),
        title=f"{args.suite} / {args.tag or sweep_id}",
        strict_gate=args.strict_gate,
        default_suite=args.suite,
        cell_records=cell_records,
        no_verify=args.no_verify,
    )


def _launch(pending, sweep_id, manifest_path, manifest, args):
    """Queue the pending cells and wait for them, recording status as it changes."""
    from development_tools.queue_runs import QueueTimeout, open_manager, queue_all
    from development_tools.queue_runs import wait_for

    print(f"\nLaunching {len(pending)} cell(s)...")
    configs = [(f"{sweep_id}_{cell.cell_id}", cell.config) for cell in pending]

    with open_manager(shutdown=not args.launch_only, runs_dir=RUNS_DIR) as manager:
        run_ids = queue_all(manager, configs)
        for cell, run_id in zip(pending, run_ids):
            entry = manifest["cells"][cell.cell_id]
            entry.update({"run_id": run_id, "status": "queued",
                          "queued_at": _timestamp()})
        _write_json(manifest_path, manifest)

        if args.launch_only:
            return

        started = time.time()
        try:
            statuses = wait_for(
                manager, run_ids,
                poll_seconds=args.poll_seconds, timeout_hours=args.timeout_hours,
            )
        except QueueTimeout as timeout:
            _write_json(manifest_path, manifest)
            raise SweepError(
                f"{timeout} Resume with:\n  python {os.path.relpath(__file__)} run "
                f"--suite {args.suite} --sweep-id {sweep_id}"
            ) from timeout

        elapsed = time.time() - started
        for cell, run_id in zip(pending, run_ids):
            metadata = manager.get_run(run_id) or {}
            entry = manifest["cells"][cell.cell_id]
            entry.update({
                "status": statuses.get(run_id, "unknown"),
                "duration_seconds": _duration(metadata),
                "log_file": metadata.get("log_file"),
            })
        _write_json(manifest_path, manifest)
        print(f"  wall clock: {elapsed / 60:.1f} min")

    failed = {
        cell.cell_id: manifest["cells"][cell.cell_id]["status"]
        for cell in pending
        if manifest["cells"][cell.cell_id]["status"] != "completed"
    }
    if failed:
        print(f"\nWARNING: {len(failed)} cell(s) did not complete: {failed}")
        print("  Their logs are in web_ui/runs/<run_id>.log. Scoring continues with the "
              "cells that did, and the report reports the holes.")


def _pending_cells(runnable, cell_records, force=False):
    """Cells still needing a run. Completed cells are skipped unless `force`."""
    pending = []
    for cell in runnable:
        entry = cell_records.get(cell.cell_id, {})
        if force or entry.get("status") != "completed":
            pending.append(cell)
            continue
        records = (entry.get("records") or {}).get("test")
        if not records or not os.path.exists(records):
            # Recorded complete but the table is gone. Re-run rather than score nothing.
            pending.append(cell)
    if len(pending) < len(runnable):
        print(f"  resuming: {len(runnable) - len(pending)} cell(s) already complete")
    return pending


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _score(cells, cell_records, args):
    """Score every completed cell. Returns (results frame, {cell_id: records path}).

    Artifacts are located through each run's `determinism.json`, never by globbing: the
    fingerprint records which configuration completed, where its records landed, and
    which head produced them, so a half-finished or mismatched run cannot be folded in.
    """
    import pandas as pd

    from evaluation.uq import subgroups
    from evaluation.uq.records import read_records
    from evaluation.uq.registry import UQ_METHODS, expand_matrix
    from evaluation.uq.scoring import Cell as ScoringCell
    from evaluation.uq.scoring import (
        add_skipped_rows, collapse_rank_equivalents, score_cells,
    )

    scoring_cells = []
    identity_rows = []
    paired_inputs = {}
    digests = {}

    print("\nCollecting artifacts...")
    for cell in cells:
        entry = cell_records.get(cell.cell_id, {})
        if not cell.runnable:
            continue
        artifacts = _collect_artifacts(cell, entry)
        entry.update(artifacts)
        if not artifacts.get("records", {}).get("test"):
            print(f"  {cell.cell_id}: no test records "
                  f"({artifacts.get('collect_note', 'unknown reason')})")
            continue

        path = artifacts["records"]["test"]
        frame = read_records(path, verify=not args.no_verify)
        manifest = _load_json(_meta_path(path)) or {}
        paired_inputs[cell.cell_id] = path
        digest = manifest.get("sha256_records")
        if digest:
            digests.setdefault(str(digest), []).append(cell.cell_id)

        # The threshold the runner fitted on this cell's val records, if it did. Applied to
        # the thresholded metrics only; every ranking metric is invariant to it.
        threshold = _fitted_threshold(artifacts.get("records", {}).get("threshold_fit"))

        methods = _available_methods(frame)
        print(f"  {cell.cell_id}: {len(frame)} rows, {len(methods)} method(s)")
        if not methods:
            continue

        identity_rows.append({
            "label": cell.cell_id,
            "cell_id": cell.cell_id,
            "axis": cell.axis,
            "axis_value": cell.axis_value,
            "arch": cell.detector,
            "traversal": cell.config.get("traversal_type"),
            "graph_type": cell.config.get("graph_type"),
            "uncertainty_head": cell.config.get("uncertainty_head"),
            "graph_manager": cell.config.get("graph_manager"),
            "run_id": entry.get("run_id"),
            "records_path": path,
            "duration_seconds": entry.get("duration_seconds"),
            # `threshold` is deliberately NOT set here: `scoring._identity` already emits it
            # per row, and a second column of the same name collides on the merge below --
            # pandas renames both to threshold_x/threshold_y, which leaves neither usable
            # and both unclassified by the comparison's direction registry.
        })

        for method_id in methods:
            spec = UQ_METHODS[method_id]
            scoring_cells.append(ScoringCell(
                detector=cell.detector,
                method_id=method_id,
                score_column=spec.primary_column,
                frame=frame,
                holdout=manifest.get("holdout_id") or "none",
                corruption=str(manifest.get("corruption", {}).get("corruption", "none")
                               if isinstance(manifest.get("corruption"), dict)
                               else manifest.get("corruption", "none")),
                severity=int(manifest.get("corruption", {}).get("severity", 0)
                             if isinstance(manifest.get("corruption"), dict) else 0),
                coverage=float(manifest.get("coverage", 1.0)),
                determinism_mode=(manifest.get("determinism_mode")
                                  or (manifest.get("determinism") or {}).get("mode")
                                  or "unknown"),
                manifest_sha256=manifest.get("sha256_records"),
                seed=manifest.get("seed"),
                cost_forward_passes=spec.cost_forward_passes,
                cost_training_runs=spec.cost_training_runs,
                threshold=threshold,
                # `label` is the cell id, so scoring's identity block carries it and the
                # sweep axes can be merged back on one column.
                extra={"label": cell.cell_id},
            ))

    _warn_identical_records(digests)

    if not scoring_cells:
        return None, paired_inputs

    # Whole-set cells plus every demographic slice of each. Subgroup rows are where the
    # fairness signal lives: the record tables have always carried gt_gender/gt_race/
    # gt_age, and nothing grouped by them before.
    expanded = subgroups.expand_cells(scoring_cells)
    print(f"\nScoring {len(expanded)} cell(s) "
          f"({len(scoring_cells)} whole-set + "
          f"{len(expanded) - len(scoring_cells)} subgroup)...")

    # require_comparable=False deliberately: a sweep pools different detectors, heads, and
    # graph types into one table, which is the point of the tidy long format.
    results = score_cells(expanded, n_boot=args.n_boot, require_comparable=False)
    results = subgroups.annotate_small_subgroups(results)
    results = collapse_rank_equivalents(results)

    disparity = subgroups.disparity(results, DISPARITY_METRICS)
    if not disparity.empty:
        results = pd.concat(
            [results, subgroups.disparity_as_results(disparity)], ignore_index=True,
        )

    # Gate every candidate method against every detector present, so the table shows what
    # was considered and refused rather than merely what happened to run.
    detectors = sorted({cell.detector for cell in cells if cell.runnable})
    if detectors:
        _runnable, decisions = expand_matrix(list(CANDIDATE_METHODS), detectors)
        scored_pairs = {
            (str(row.method_id), str(row.detector))
            for row in results.itertuples() if getattr(row, "status", "") == "ok"
        }
        decisions = [
            decision for decision in decisions
            if (decision.method_id, decision.detector) not in scored_pairs
        ]
        results = add_skipped_rows(results, decisions)

    identity = pd.DataFrame(identity_rows)
    if not identity.empty:
        results = results.merge(identity, on="label", how="left")
        # Gate-skip rows have no label, so they carry no cell identity. Named explicitly
        # rather than left NaN, which would join them to nothing in the diff.
        results["cell_id"] = results["cell_id"].fillna("(gated)")

    _warn_unclassified(results)
    return results, paired_inputs


def _collect_artifacts(cell, entry):
    """Find a cell's record tables via its run's `determinism.json`."""
    run_id = entry.get("run_id")
    if not run_id:
        return {"records": {}, "collect_note": "no run id recorded"}

    fingerprint_path = os.path.join(RUN_OUTPUTS_DIR, run_id, "determinism.json")
    fingerprint = _load_json(fingerprint_path)
    if not fingerprint:
        return {"records": {}, "collect_note": f"no {fingerprint_path}"}

    blocks = fingerprint.get("results") or {}
    block = blocks.get(cell.description)
    if block is None and len(blocks) == 1:
        # One configuration per run, so an unexpected description key is still
        # unambiguous. Happens when a traversal name is normalized inside the runner.
        block = next(iter(blocks.values()))
    if block is None:
        return {
            "records": {},
            "collect_note": f"determinism.json has no results block for "
                            f"{cell.description!r} (has: {sorted(blocks)})",
        }
    if not block.get("complete"):
        return {"records": {}, "collect_note": "run did not finish its configuration"}

    return {
        # The runner wrote these relative to its own cwd, which the queue pins to the repo
        # root. Anchored here so they resolve from anywhere the sweep is invoked.
        "records": {
            split: _resolve_artifact(path)
            for split, path in (block.get("records") or {}).items()
        },
        "test_accuracy": block.get("test_accuracy"),
        "best_epoch": block.get("best_epoch"),
        "best_val_accuracy": block.get("best_val_accuracy"),
        "checkpoint": block.get("checkpoint"),
        "collect_note": "",
    }


def _available_methods(frame):
    """Methods whose primary score column this table can supply."""
    from evaluation.uq.registry import UQ_METHODS

    derivable = ("u_maxprob", "u_entropy", "u_margin")
    columns = set(frame.columns)
    found = []
    for method_id in CANDIDATE_METHODS:
        spec = UQ_METHODS.get(method_id)
        if spec is None:
            continue
        if spec.primary_column in columns or spec.primary_column in derivable:
            found.append(method_id)
    return found


def identical_record_groups(digests):
    """Cells whose record tables are byte-identical. `{digest: [cell_id, ...]}`, size > 1.

    Distinct configurations producing the same digest means at least one of the settings
    never reached training, and every metric for those cells is one measurement wearing
    several names. It has happened three ways already: a graph type that resolved to the
    same dataloader, subclustering that no traversal could read, and two graph updaters
    whose mutations landed after the epoch that produced the best checkpoint.
    """
    return {
        digest: sorted(cells)
        for digest, cells in (digests or {}).items() if len(cells) > 1
    }


def _warn_identical_records(digests):
    groups = identical_record_groups(digests)
    if not groups:
        return
    print(f"\nWARNING: {len(groups)} group(s) of cells produced byte-identical record "
          f"tables, so their settings never reached the reported numbers:")
    for digest, cells in sorted(groups.items()):
        print(f"  {digest[:12]}: {', '.join(cells)}")
    print("  Each group is one measurement under several names. Check that the differing "
          "setting reaches training, and that the best checkpoint is not from an epoch "
          "before the setting takes effect.")


def _warn_unclassified(results):
    from evaluation.uq.compare import unclassified_columns

    missing = unclassified_columns(results)
    if missing:
        print(
            "\nWARNING: these result columns have no comparison direction and will be "
            f"reported as n/a: {', '.join(missing)}.\n  Add each to HIGHER_IS_BETTER, "
            "LOWER_IS_BETTER, or IGNORED_COLUMNS in evaluation/uq/compare.py."
        )


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def command_compare(args):
    baseline, baseline_manifest = resolve_reference(args.baseline)
    candidate, candidate_manifest = resolve_reference(args.candidate)

    out = args.out or os.path.join(
        RUN_OUTPUTS_DIR, "reports", f"compare_{_slug(args.candidate)}"
    )
    return _compare_and_report(
        baseline_ref=args.baseline,
        candidate_results=candidate,
        candidate_manifest=candidate_manifest,
        candidate_paired=_paired_inputs_from(candidate, candidate_manifest),
        out=out,
        title=args.title or f"{_slug(args.candidate)} vs {_slug(args.baseline)}",
        strict_gate=args.strict_gate,
        baseline_results=baseline,
        baseline_manifest=baseline_manifest,
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
    )


def _compare_and_report(
    baseline_ref,
    candidate_results,
    candidate_manifest,
    candidate_paired,
    out,
    title,
    strict_gate,
    default_suite=None,
    cell_records=None,
    baseline_results=None,
    baseline_manifest=None,
    abs_tolerance=0.0,
    rel_tolerance=0.0,
    no_verify=False,
):
    """Diff, write the comparison report, and pick the exit code."""
    from evaluation.uq import compare as compare_module
    from evaluation.uq.compare_report import ComparisonReport

    if baseline_results is None:
        reference = baseline_ref
        if reference == "baseline":
            reference = f"baseline:{default_suite}"
        baseline_results, baseline_manifest = resolve_reference(reference)

    comparison = compare_module.compare(
        baseline_results, candidate_results,
        abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance,
    )

    paired = _paired_tests(
        _paired_inputs_from(baseline_results, baseline_manifest),
        candidate_paired,
        verify=not no_verify,
    )

    report = ComparisonReport(
        out, comparison,
        baseline=baseline_results, candidate=candidate_results, paired=paired,
        title=title,
        baseline_manifest=baseline_manifest, candidate_manifest=candidate_manifest,
    )
    written = report.generate_all_plots()

    counts = compare_module.summarize(comparison)
    # Collapsed: one degenerate cell blanks every metric on every subgroup, so the raw
    # row count would report hundreds of failures for one problem.
    failures = compare_module.hard_failure_summary(comparison)
    worse = compare_module.regressions(comparison)

    print(f"\nWrote {len(written)} file(s) to {out}")
    print(f"  better {counts['better']} | worse {counts['worse']} | "
          f"same {counts['same']} | n/a {counts['n_a']}")
    print(f"  added {counts['added']} | removed {counts['removed']} | "
          f"newly degenerate {counts['newly_degenerate']}")

    for warning in compare_module.provenance_warnings(baseline_results, candidate_results):
        print(f"  NOTE: {warning}")

    if not worse.empty:
        print("\nLargest regressions:")
        for _index, row in worse.head(8).iterrows():
            print(f"  {row.get('cell_id')} / {row.get('method_id')} / "
                  f"{row.get('metric')}: {row.get('baseline'):.4f} -> "
                  f"{row.get('candidate'):.4f}")

    exit_code = EXIT_OK
    if not failures.empty:
        print(f"\n{len(failures)} hard failure(s): a measurement disappeared or stopped "
              f"being computable.")
        for _index, row in failures.head(8).iterrows():
            print(f"  {row.get('cell_id')} / {row.get('method_id')}: "
                  f"{row.get('reason')} ({row.get('status_baseline')} -> "
                  f"{row.get('status_candidate')}, {row.get('metrics_affected')} "
                  f"metric(s) over {row.get('subgroups_affected')} slice(s))")
        print(f"  Full list: {os.path.join(out, 'comparison.md')}")
        exit_code = EXIT_HARD_FAILURE
    if cell_records is not None:
        exit_code = max(exit_code, _hard_failure_exit(cell_records, quiet=True))
    if strict_gate and not worse.empty:
        print(f"\n--strict-gate: failing on {len(worse)} regression(s).")
        exit_code = max(exit_code, EXIT_HARD_FAILURE)
    return exit_code


def _paired_tests(baseline_paths, candidate_paths, verify=True):
    """McNemar per cell, wherever both record tables are still on disk.

    Best effort by design. A promoted baseline is a small CSV committed to git; the
    hundreds of megabytes of record tables behind it are not, so on a fresh checkout the
    paired test has nothing to align and the report says `-` rather than inventing one.
    """
    from evaluation.uq.compare import paired_accuracy_test
    from evaluation.uq.records import read_records

    results = {}
    shared = sorted(set(baseline_paths) & set(candidate_paths))
    if not shared:
        return results

    print(f"\nPaired accuracy tests on {len(shared)} cell(s)...")
    for cell_id in shared:
        left, right = baseline_paths[cell_id], candidate_paths[cell_id]
        if not (left and right and os.path.exists(left) and os.path.exists(right)):
            continue
        try:
            outcome = paired_accuracy_test(
                read_records(left, verify=verify), read_records(right, verify=verify),
            )
        except Exception as error:  # noqa: BLE001 - one bad table must not stop the rest
            print(f"  {cell_id}: {type(error).__name__}: {error}")
            continue
        results[cell_id] = outcome
        if outcome["applicable"]:
            print(f"  {cell_id}: {outcome['baseline_accuracy']:.4f} -> "
                  f"{outcome['candidate_accuracy']:.4f}, p={outcome['p_value']:.3g} "
                  f"({outcome['method']}, n={outcome['n_aligned']})")
        else:
            print(f"  {cell_id}: not applicable -- {outcome['reason']}")
    return results


def _paired_inputs_from(results, manifest):
    """`{cell_id: records path}` from a manifest, falling back to the results table."""
    paths = {}
    for cell_id, entry in ((manifest or {}).get("cells") or {}).items():
        path = (entry.get("records") or {}).get("test")
        if path:
            paths[cell_id] = _resolve_artifact(path)
    if paths or results is None or "records_path" not in getattr(results, "columns", ()):
        return paths
    for row in results.dropna(subset=["records_path"]).itertuples():
        paths[str(row.cell_id)] = _resolve_artifact(str(row.records_path))
    return paths


# --------------------------------------------------------------------------- #
# promote / list
# --------------------------------------------------------------------------- #

def command_promote(args):
    sweep_dir = os.path.join(SWEEPS_DIR, args.sweep_id)
    manifest_path = os.path.join(sweep_dir, "manifest.json")
    manifest = _load_json(manifest_path)
    if not manifest:
        raise SweepError(f"no manifest at {manifest_path}")

    results_path = os.path.join(sweep_dir, "results.csv")
    if not os.path.exists(results_path):
        raise SweepError(
            f"{results_path} does not exist; score the sweep first with "
            f"`run --sweep-id {args.sweep_id} --score-only`"
        )

    incomplete = {
        cell_id: entry.get("status")
        for cell_id, entry in (manifest.get("cells") or {}).items()
        if entry.get("status") not in ("completed", "gated")
    }
    if incomplete and not args.allow_partial:
        raise SweepError(
            f"{len(incomplete)} cell(s) did not complete: "
            f"{json.dumps(incomplete, indent=2)}\n"
            "Promoting now would make each of them read as `removed` in every later "
            "diff, which looks like a regression the code did not cause. Re-run them, "
            "or pass --allow-partial."
        )

    suite = args.suite or manifest.get("suite") or "standard"
    os.makedirs(BASELINES_DIR, exist_ok=True)
    baseline_path = os.path.join(BASELINES_DIR, f"baseline_{suite}.csv")
    baseline_manifest_path = os.path.join(
        BASELINES_DIR, f"baseline_{suite}.manifest.json"
    )

    import shutil

    shutil.copyfile(results_path, baseline_path)
    # The manifest travels with the CSV so a later diff can name the commit and the node
    # cache this baseline was measured against, and warn when either has moved on.
    promoted = copy.deepcopy(manifest)
    promoted["promoted_at"] = _timestamp()
    promoted["promoted_from"] = args.sweep_id
    _write_json(baseline_manifest_path, promoted)

    print(f"Promoted {args.sweep_id} -> {baseline_path}")
    print(f"  manifest -> {baseline_manifest_path}")
    print(f"  commit {str(manifest.get('git', {}).get('commit', '?'))[:12]}"
          + (" (dirty tree)" if manifest.get("git", {}).get("dirty") else ""))
    print("\nCommit both files so this baseline travels with the code:")
    print(f"  git add {baseline_path} {baseline_manifest_path}")
    return EXIT_OK


def command_list(args):
    print("Sweeps:")
    found = False
    for name in sorted(_listdir(SWEEPS_DIR), reverse=True):
        manifest = _load_json(os.path.join(SWEEPS_DIR, name, "manifest.json")) or {}
        cells = manifest.get("cells") or {}
        done = sum(1 for entry in cells.values() if entry.get("status") == "completed")
        scored = "scored" if os.path.exists(
            os.path.join(SWEEPS_DIR, name, "results.csv")
        ) else "unscored"
        found = True
        print(f"  {name}  suite={manifest.get('suite', '?')} "
              f"tag={manifest.get('tag') or '-'} "
              f"{done}/{len(cells)} complete, {scored}")
    if not found:
        # An existing-but-empty directory printed nothing at all, which reads as output
        # that failed rather than as an empty list.
        print("  none")

    print("\nPromoted baselines:")
    found = False
    for name in sorted(_listdir(BASELINES_DIR)):
        if not name.startswith("baseline_") or not name.endswith(".csv"):
            continue
        found = True
        manifest = _load_json(
            os.path.join(BASELINES_DIR, name.replace(".csv", ".manifest.json"))
        ) or {}
        print(f"  {name}  from={manifest.get('promoted_from', '?')} "
              f"commit={str(manifest.get('git', {}).get('commit', '?'))[:12]} "
              f"at={manifest.get('promoted_at', '?')}")
    if not found:
        print("  none")
    return EXIT_OK


def _fitted_threshold(path, default=0.5):
    """The threshold from a `threshold_fit.json`, or `default`.

    An inapplicable fit -- a single-class validation split, say -- reports the default, so
    the cell is scored at 0.5 rather than at a threshold nothing justified.
    """
    fit = _load_json(_resolve_artifact(path)) if path else None
    if not fit or not fit.get("applicable", True):
        return default
    try:
        return float(fit["threshold"])
    except (KeyError, TypeError, ValueError):
        return default


def _resolve_artifact(path):
    """Make a runner-written artifact path absolute.

    Anchored to `RUN_OUTPUTS_DIR`'s parent rather than to `REPO_ROOT` directly, because the
    runner writes these relative to *its* working directory -- which is normally the repo
    root, but is redirected along with `RUN_OUTPUTS_DIR` when a test drives cells itself.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(RUN_OUTPUTS_DIR), path)


def _listdir(path):
    """Directory entries, or an empty list when the directory does not exist."""
    try:
        return os.listdir(path)
    except OSError:
        return []


def resolve_reference(reference):
    """`(results frame, manifest)` for a sweep id, a CSV path, or `baseline:<suite>`."""
    import pandas as pd

    text = str(reference)

    if text.startswith("baseline:") or text == "baseline":
        suite = text.partition(":")[2] or "standard"
        path = os.path.join(BASELINES_DIR, f"baseline_{suite}.csv")
        if not os.path.exists(path):
            raise SweepError(
                f"no promoted baseline for suite {suite!r} at {path}. Establish one "
                f"with:\n  python {os.path.relpath(__file__)} run --suite {suite} "
                f"--tag baseline\n  python {os.path.relpath(__file__)} promote <sweep-id>"
            )
        manifest = _load_json(
            os.path.join(BASELINES_DIR, f"baseline_{suite}.manifest.json")
        ) or {}
        return pd.read_csv(path), manifest

    sweep_dir = os.path.join(SWEEPS_DIR, text)
    if os.path.isdir(sweep_dir):
        path = os.path.join(sweep_dir, "results.csv")
        if not os.path.exists(path):
            raise SweepError(
                f"sweep {text} has no results.csv; score it with "
                f"`run --sweep-id {text} --score-only`"
            )
        return pd.read_csv(path), _load_json(os.path.join(sweep_dir, "manifest.json")) or {}

    if os.path.exists(text):
        manifest_path = os.path.join(os.path.dirname(text), "manifest.json")
        return pd.read_csv(text), _load_json(manifest_path) or {}

    raise SweepError(
        f"cannot resolve {reference!r}: it is not a sweep id under {SWEEPS_DIR}, a path "
        f"to a results.csv, or `baseline:<suite>`"
    )


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #

def _report_own(results, out, sweep_id):
    """The sweep's own benchmark report, independent of any comparison."""
    from evaluation.uq.report import UQReport

    report = UQReport(out, results=results, title=sweep_id)
    # Figures only: write_results_csv would duplicate the sweep's own results.csv under
    # a second name.
    written = []
    written.append(report.write_results_markdown())
    written += report.plot_gating_heatmap() or []
    print(f"Wrote the sweep's own report to {out}")
    return [path for path in written if path]


def _print_own_summary(results):
    """A short per-cell summary for a sweep with nothing to compare against."""
    overall = results[
        (results.get("subgroup_dimension") == "overall")
        & (results.get("method_id") == "baseline_maxprob")
    ] if "subgroup_dimension" in results.columns else results

    if overall.empty:
        return
    print("\nPer-cell accuracy (baseline_maxprob rows):")
    columns = [
        column for column in ("cell_id", "arch", "traversal", "clf_accuracy",
                              "clf_balanced_accuracy", "ece_confidence", "auroc_error",
                              "status")
        if column in overall.columns
    ]
    print(overall[columns].to_string(index=False))

    _warn_collapsed(overall)

    print("\nPromote this as the baseline once you are happy with it:")
    print("  python development_tools/sweep.py promote <sweep-id>")


def _warn_collapsed(overall):
    """Call out cells whose model emitted one class for every sample.

    Worth its own warning rather than a flag in a CSV column: `clf_accuracy` then equals
    the majority-class prior, which on this dataset is about 0.87 and reads as a
    respectable result. Identical accuracy across otherwise-different configurations is
    the tell, and it is exactly the state that should not be promoted as a baseline.
    """
    from evaluation.uq.metrics import SINGLE_CLASS_PREDICTIONS

    if "status_flags" not in overall.columns:
        return
    collapsed = overall[
        overall["status_flags"].fillna("").str.contains(SINGLE_CLASS_PREDICTIONS)
    ]
    if collapsed.empty:
        return

    print(f"\nWARNING: {len(collapsed)} cell(s) predicted a single class for every "
          f"sample: {', '.join(sorted(collapsed['cell_id'].astype(str).unique()))}")
    print("  Their accuracy is the majority-class prior, not a measurement, and "
          "balanced accuracy is pinned at 0.5.")
    print("  Expected for a very short run on an imbalanced split; do not promote it as "
          "a baseline. Train longer, or use a balanced split (--fair-train/--fair-test).")


def _hard_failure_exit(cell_records, quiet=False):
    """Exit code 2 when a cell did not complete or produced no records."""
    broken = {
        cell_id: entry.get("status")
        for cell_id, entry in (cell_records or {}).items()
        if entry.get("status") not in ("completed", "gated", "pending")
    }
    missing = {
        cell_id: "completed but wrote no test records"
        for cell_id, entry in (cell_records or {}).items()
        if entry.get("status") == "completed"
        and not (entry.get("records") or {}).get("test")
    }
    problems = {**broken, **missing}
    if not problems:
        return EXIT_OK
    if not quiet:
        print(f"\n{len(problems)} cell(s) failed:")
        for cell_id, reason in sorted(problems.items()):
            print(f"  {cell_id}: {reason}")
    return EXIT_HARD_FAILURE


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def _unroutable(cells):
    from web_ui.gpu_queue_manager import validate_config_keys

    problems = {}
    for cell in cells:
        keys = validate_config_keys(cell.config)
        if keys:
            problems[cell.cell_id] = keys
    return problems


def _split(text):
    if not text:
        return None
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _generate_sweep_id(suite):
    # secrets, not the seeded RNG: a sweep id must not consume from the run's randomness,
    # and two sweeps launched at the same seed are different sweeps.
    return f"sweep_{time.strftime('%Y%m%d_%H%M%S')}_{suite}_{secrets.token_hex(3)}"


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _duration(metadata):
    """Wall-clock seconds for a run, from the queue's own start/end stamps."""
    from datetime import datetime

    start, end = metadata.get("start_time"), metadata.get("end_time")
    if not (start and end):
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def _git_state():
    from test_helpers.determinism import run_fingerprint

    return (run_fingerprint().get("git") or {})


def _cache_digest(cache_file):
    """sha256 of the node cache, so a rebuilt cache invalidates a baseline visibly."""
    if not cache_file or not os.path.exists(cache_file):
        return None
    from evaluation.uq.records import sha256_of_file

    return sha256_of_file(cache_file)


def _meta_path(records_path):
    from evaluation.uq.records import default_meta_path

    return default_meta_path(records_path)


def _load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return path


def _slug(text):
    """A short, filename-safe name for a reference.

    A path collapses to its meaningful tail -- the sweep directory plus the filename --
    rather than the whole absolute path, which would otherwise become a hundred-character
    directory name and an unreadable report title.
    """
    safe = str(text)
    if os.sep in safe or safe.endswith(".csv"):
        parent = os.path.basename(os.path.dirname(safe))
        stem = os.path.splitext(os.path.basename(safe))[0]
        safe = f"{parent}_{stem}" if parent else stem
    for character in (":", "/", "\\", " ", ",", "="):
        safe = safe.replace(character, "_")
    return safe.strip("_")


if __name__ == "__main__":
    sys.exit(main())
