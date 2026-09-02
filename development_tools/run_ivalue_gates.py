#!/usr/bin/env python3
"""Run the I-value gate across every estimator configuration and print one table.

The gate asks one question: does the predicted I-value rank a sample's *realised* learning
gain better than the sample's current loss, which the training loop already has for free? The
first run of it answered no by a factor of 33 (`docs/ivalue_gate_result.md`), which is why
this matrix exists -- three architectural fixes across three new models and five wrapped
legacy ones is too many combinations to judge by argument.

Each cell is a 2-epoch run on the 40k pool, ~20 minutes, and writes its own
`ivalue_diagnostic.csv.gz`. Nothing here trains to convergence or reports accuracy: a
configuration that cannot rank learning gain has nothing to contribute to a training sweep,
and the point is to spend twenty minutes finding that out instead of a day.

Usage:
    python development_tools/run_ivalue_gates.py --gpus 0,1
    python development_tools/run_ivalue_gates.py --gpus 0 --only gain_linear
    python development_tools/run_ivalue_gates.py --collect-only     # re-print the table
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PYTHON = sys.executable
CACHE = os.path.join("node_cache", "cached_nodes_40k.pkl")

#: The fixed family, plus each legacy architecture wrapped in the fixed base and left alone.
NEW_MODELS = ("gain_linear", "gain_residual", "gain_ensemble")
LEGACY_MODELS = ("basic", "residual", "attention", "conv_embedding", "ensemble")
OBJECTIVES = ("rank", "huber")


def configurations():
    """Every cell of the matrix, as `(label, extra_args)`."""
    cells = []
    for model, objective in itertools.product(NEW_MODELS, OBJECTIVES):
        cells.append((f"{model}.{objective}",
                      ["--dqn-model", model, "--dqn-objective", objective]))
    for model in LEGACY_MODELS:
        # The original, so "did the fixes help?" has a same-architecture comparison.
        cells.append((f"{model}.original", ["--dqn-model", model]))
        cells.append((f"{model}.fixed",
                      ["--dqn-model", model, "--dqn-fixes", "--dqn-objective", "rank"]))
    return cells


def command(label, extra, epochs, nodes_per_epoch):
    return [
        PYTHON, "-u", "test_hierarchical.py",
        "--use-cached", "--cache-file", CACHE, "--cached-nodes", "10000",
        "--graph-type", "nonclustered", "--traversal-type", "i-value",
        "--architectures", "resnestdf",
        "--num-epochs", str(epochs),
        "--train-steps", "4000000",
        "--max-nodes-per-epoch", str(nodes_per_epoch),
        "--val-steps", "500", "--batch-size", "32", "--seed", "42",
        "--determinism", "strict", "--edge-construction", "knn", "--knn-neighbors", "50",
        "--preprocess-workers", "8",
        "--ivalue-reward", "learning_gain", "--ivalue-state-features", "--ivalue-diagnostic",
        "--run-id", f"gate_{label.replace('.', '_')}",
    ] + extra


def collect(labels):
    """Read every diagnostic and rank the configurations against the free baseline."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    rows = []
    for label in labels:
        run_id = f"gate_{label.replace('.', '_')}"
        import glob
        hits = glob.glob(os.path.join(REPO_ROOT, "run_outputs", run_id,
                                      "*", "ivalue_diagnostic.csv.gz"))
        if not hits:
            rows.append((label, None, None, 0))
            continue
        frame = pd.read_csv(hits[0]).dropna(subset=["predicted_ivalue", "gain"])
        if len(frame) < 50 or frame["predicted_ivalue"].std() == 0:
            rows.append((label, None, None, len(frame)))
            continue
        predicted = frame["predicted_ivalue"].to_numpy(float)
        gain = frame["gain"].to_numpy(float)
        # The bar is what the estimator could see when it *chose*: the stale per-node loss
        # EWMA. `loss_before` is produced by the forward pass on the batch being trained, so
        # ranking by it would cost a detector pass per candidate -- unattainable, and using
        # it as the bar made the gate unpassable by construction.
        if "state_loss" in frame.columns and frame["state_loss"].notna().any():
            bar_signal = frame["state_loss"].to_numpy(float)
        else:
            bar_signal = frame["loss_before"].to_numpy(float)
        rows.append((label,
                     float(stats.spearmanr(predicted, gain).statistic),
                     float(stats.spearmanr(bar_signal, gain).statistic),
                     len(frame)))

    print()
    print(f"{'configuration':28s} {'spearman':>9s} {'seen-bar':>9s} {'delta':>8s} "
          f"{'rows':>7s}  verdict")
    print("-" * 78)
    scored = [r for r in rows if r[1] is not None]
    for label, spearman, baseline, n in sorted(
        rows, key=lambda r: (r[1] is None, -(r[1] or 0))
    ):
        if spearman is None:
            print(f"{label:28s} {'--':>9s} {'--':>9s} {'--':>8s} {n:7d}  no usable diagnostic")
            continue
        delta = spearman - baseline
        verdict = "BEATS the seen signal" if delta > 0.02 else (
            "no signal at all" if spearman <= 0.05 else "no better than what it sees")
        print(f"{label:28s} {spearman:+9.4f} {baseline:+9.4f} {delta:+8.4f} {n:7d}  {verdict}")

    if scored:
        best = max(scored, key=lambda r: r[1] - r[2])
        print()
        print(f"Best margin over the selection-time signal: {best[0]} "
              f"({best[1]:+.4f} vs {best[2]:+.4f}, delta {best[1] - best[2]:+.4f})")
        if best[1] - best[2] <= 0.02:
            print("Nothing beats the signal it already receives. The honest conclusion is to "
                  "rank candidates by their stored loss EWMA directly and drop the learned "
                  "estimator from the selection path.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU ids to spread the runs over, e.g. 0,1.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--nodes-per-epoch", type=int, default=10000)
    parser.add_argument("--only", action="append", default=None,
                        help="Run just these labels (repeatable), e.g. --only gain_linear.rank")
    parser.add_argument("--collect-only", action="store_true",
                        help="Skip running; re-read the diagnostics and print the table.")
    parser.add_argument("--log-dir", default=os.path.join(REPO_ROOT, "run_outputs", "gate_logs"))
    parser.add_argument("--timeout-minutes", type=float, default=90.0,
                        help="Kill a cell that exceeds this. A 2-epoch gate takes ~20 min, so "
                             "anything near this is stuck -- and a stuck cell holds a GPU "
                             "indefinitely: one wrapped estimator that raised on every batch "
                             "logged 258,398 tracebacks over nearly three days without "
                             "finishing an epoch, because a failed batch does not advance the "
                             "progress counter. The estimator now fails fast, but a wall-clock "
                             "backstop costs nothing and catches the next surprise.")
    args = parser.parse_args(argv)

    cells = configurations()
    if args.only:
        wanted = set(args.only)
        cells = [c for c in cells if c[0] in wanted or c[0].split(".")[0] in wanted]
        if not cells:
            print(f"No configuration matched {sorted(wanted)}")
            return 2

    labels = [label for label, _ in cells]
    if args.collect_only:
        return collect(labels)

    gpus = [int(g) for g in args.gpus.replace(" ", ",").split(",") if g]
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"{len(cells)} configuration(s) over GPU(s) {gpus}; "
          f"~20 min each, {len(cells) / max(1, len(gpus)) * 20 / 60:.1f} h wall clock")

    # One process per GPU at a time: the bottleneck is a single detector training loop, and
    # two on one card would distort the timings without finishing sooner.
    running = {}
    pending = list(cells)
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            label, extra = pending.pop(0)
            log_path = os.path.join(args.log_dir, f"{label}.log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            handle = open(log_path, "w")
            process = subprocess.Popen(
                command(label, extra, args.epochs, args.nodes_per_epoch),
                cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
            running[gpu] = (label, process, handle, time.monotonic())
            print(f"  [gpu {gpu}] started {label} -> {log_path}")
        for gpu, (label, process, handle, started) in list(running.items()):
            if process.poll() is not None:
                handle.close()
                print(f"  [gpu {gpu}] {label} exited {process.returncode}")
                del running[gpu]
            elif time.monotonic() - started > args.timeout_minutes * 60:
                process.kill()
                process.wait(timeout=30)
                handle.close()
                print(f"  [gpu {gpu}] {label} KILLED after "
                      f"{args.timeout_minutes:.0f} min -- see {os.path.join(args.log_dir, label + '.log')}")
                del running[gpu]
        if running:
            time.sleep(10)

    return collect(labels)


if __name__ == "__main__":
    sys.exit(main())
