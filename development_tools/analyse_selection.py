#!/usr/bin/env python3
"""Compare what different selection strategies actually trained on.

The question this exists to answer: an i-value walk takes its argmax over the current node's
k-NN neighbours, and k-NN neighbours are similar faces *by construction*, so its batches
should contain near-duplicates where an i.i.d. batch does not. Low effective batch diversity
is a standard cause of worse SGD, and it is the leading unexamined explanation for i-value
training losing to i.i.d. sampling by ~1.3 points of balanced accuracy while a 16x better
learning-gain estimator changed nothing.

Reads `selection_diagnostic.csv.gz` from runs made with `--selection-diagnostic`.

Usage:
    python development_tools/analyse_selection.py run_outputs/<run_a> run_outputs/<run_b>
    python development_tools/analyse_selection.py --compare i-value comprehensive
"""

import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

#: A difference in mean batch diversity below this is not worth acting on: batch-to-batch
#: spread is large, so a tiny difference in means says little about gradient quality.
MEANINGFUL_DIVERSITY_GAP = 0.02


def find(target):
    if os.path.isfile(target):
        return target
    for pattern in ("selection_diagnostic.csv.gz", "*/selection_diagnostic.csv.gz"):
        hits = sorted(glob.glob(os.path.join(target, pattern)))
        if hits:
            return hits[0]
    return None


def summarise(path, label):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    usable = frame.dropna(subset=["batch_diversity"])
    if usable.empty:
        print(f"  {label}: no batch had two embedded nodes; nothing to compare")
        return None

    diversity = usable["batch_diversity"].to_numpy(float)
    row = {
        "label": label,
        "batches": len(frame),
        "diversity_mean": float(diversity.mean()),
        "diversity_sd": float(diversity.std()),
        "diversity_p10": float(np.percentile(diversity, 10)),
    }
    row["batch_size"] = float(frame["batch_size"].mean())
    for column, key in (("loss_mean", "loss"), ("frac_positive", "frac_pos"),
                        ("race_coverage", "race_cov"),
                        ("gender_coverage", "gender_cov")):
        if column in frame and frame[column].notna().any():
            row[key] = float(frame[column].dropna().mean())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="run directories, or the csv.gz files")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Shorthand: resolve run_outputs/selection_<A> and _<B>.")
    args = parser.parse_args(argv)

    targets = list(args.targets)
    if args.compare:
        targets += [os.path.join(REPO_ROOT, "run_outputs", f"selection_{name}")
                    for name in args.compare]
    if not targets:
        parser.error("give at least one run directory, or use --compare")

    rows = []
    for target in targets:
        path = find(target)
        label = os.path.basename(target.rstrip("/")).replace("selection_", "")
        if path is None:
            print(f"  {label}: no selection_diagnostic.csv.gz "
                  f"(was the run given --selection-diagnostic?)")
            continue
        summary = summarise(path, label)
        if summary:
            rows.append(summary)

    if not rows:
        return 2

    print()
    header = (f"{'run':22s} {'batches':>8s} {'bsize':>6s} {'diversity':>10s} {'sd':>7s} "
              f"{'p10':>7s}")
    extras = [k for k in ("loss", "frac_pos", "race_cov", "gender_cov") if k in rows[0]]
    header += "".join(f" {k:>9s}" for k in extras)
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r["diversity_mean"]):
        line = (f"{row['label']:22s} {row['batches']:8d} {row['batch_size']:6.1f} "
                f"{row['diversity_mean']:10.4f} {row['diversity_sd']:7.4f} "
                f"{row['diversity_p10']:7.4f}")
        line += "".join(f" {row.get(k, float('nan')):9.4f}" for k in extras)
        print(line)

    if len(rows) >= 2:
        best = max(rows, key=lambda r: r["diversity_mean"])
        worst = min(rows, key=lambda r: r["diversity_mean"])
        gap = best["diversity_mean"] - worst["diversity_mean"]
        print()
        print(f"Largest gap: {worst['label']} is {gap:+.4f} less diverse per batch than "
              f"{best['label']}")
        sizes = {round(r["batch_size"]) for r in rows}
        if len(sizes) > 1:
            print(f"NOTE: batch sizes differ ({sorted(sizes)}). For i.i.d. sampling the mean "
                  f"pairwise distance does not depend on batch size, but a walk's larger "
                  f"batch spans more steps and so looks MORE diverse -- so a gap measured "
                  f"with the walk on the larger size is conservative.")
        if gap > MEANINGFUL_DIVERSITY_GAP:
            print("VERDICT: batch diversity DIFFERS materially. Breaking the walk's locality "
                  "(--ivalue-candidate-pool) is the primary experiment; treat any result that "
                  "does not control for diversity as confounded.")
        else:
            print(f"VERDICT: batch diversity is comparable (gap <= "
                  f"{MEANINGFUL_DIVERSITY_GAP}). Locality is NOT the explanation, so the gap "
                  f"to i.i.d. lies elsewhere and the selection-strategy arms are worth "
                  f"testing on their own terms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
