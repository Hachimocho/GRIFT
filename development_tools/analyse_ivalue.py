#!/usr/bin/env python3
"""Does the predicted I-value track the learning gain it is supposed to represent?

This is a gate, not a report. A selection strategy built on an estimator that does not rank
informativeness cannot be expected to beat i.i.d. sampling, so it is worth spending two
epochs answering the question before spending a day of GPU on the downstream comparison.

Reads `ivalue_diagnostic.csv.gz`, written by a run with `--ivalue-diagnostic`. Each row pairs
the I-value the traversal saw when it *chose* a node with the per-sample loss reduction that
training on it actually produced.

Read the correlation, not the p-value: at 10,000 rows per epoch almost anything is
"significant", and what matters is whether the association is positive and large enough to
steer sampling.

Usage:
    python development_tools/analyse_ivalue.py run_outputs/<run_id>
    python development_tools/analyse_ivalue.py run_outputs/<run_id> --per-epoch
"""

import argparse
import glob
import os
import sys

#: Absolute floor: below this the estimator is not ranking learning gain at all.
ABSOLUTE_FLOOR = 0.05

#: How far the estimator must beat the best *selection-time* signal to earn its keep.
#:
#: The bar has been wrong twice, in opposite directions. First it was the absolute floor
#: alone (0.05), which would have passed an estimator losing to a trivial signal by 5x.
#: Then it was `spearman(current loss, gain)` = +0.32 -- but `loss_before` is produced by
#: the forward pass on the batch being trained, so consulting it for a *candidate* costs a
#: detector pass per candidate, which is precisely the cost the estimator exists to avoid.
#: No estimator can beat it without paying it, so that bar was unpassable by construction.
#:
#: The honest bar is `spearman(state_loss, gain)`: the stale per-node loss EWMA the estimator
#: really sees at selection time, measured at +0.132 on the 40k pool. Beating that is
#: possible -- static attributes carry difficulty signal the EWMA does not -- and it is what
#: "the learned estimator is worth its cost" actually means. `loss_before` is still reported,
#: as the (unattainable) ceiling for context.
BASELINE_MARGIN = 0.02


def find_diagnostic(target):
    """Accept a run directory, a description directory, or the file itself."""
    if os.path.isfile(target):
        return target
    for pattern in ("ivalue_diagnostic.csv.gz", "*/ivalue_diagnostic.csv.gz"):
        hits = sorted(glob.glob(os.path.join(target, pattern)))
        if hits:
            return hits[0]
    return None


def report(frame, label="all epochs"):
    """Print the association between predicted I-value and realised gain."""
    import numpy as np
    from scipy import stats

    usable = frame.dropna(subset=["predicted_ivalue", "gain"])
    if len(usable) < 50:
        print(f"  {label}: only {len(usable)} usable row(s); need >= 50 to say anything")
        return None, None

    predicted = usable["predicted_ivalue"].to_numpy(dtype=float)
    gain = usable["gain"].to_numpy(dtype=float)

    if np.std(predicted) == 0:
        print(f"  {label}: predicted I-value is constant; the estimator is not discriminating")
        return None, None

    pearson = float(np.corrcoef(predicted, gain)[0, 1])
    spearman = float(stats.spearmanr(predicted, gain).statistic)

    # Does a high predicted I-value pick out the samples that actually taught the most?
    top_decile = gain >= np.quantile(gain, 0.9)
    if top_decile.any() and not top_decile.all():
        auroc = float(stats.mannwhitneyu(
            predicted[top_decile], predicted[~top_decile], alternative="two-sided"
        ).statistic / (top_decile.sum() * (~top_decile).sum()))
    else:
        auroc = float("nan")

    print(f"  {label}: n={len(usable)}")
    print(f"    pearson(predicted, gain)   = {pearson:+.4f}")
    print(f"    spearman(predicted, gain)  = {spearman:+.4f}")
    print(f"    AUROC(ranks top-decile gain) = {auroc:.4f}   (0.5 = no signal)")
    print(f"    mean gain: top-quartile predicted = "
          f"{gain[predicted >= np.quantile(predicted, 0.75)].mean():+.5f}  vs  "
          f"bottom-quartile = {gain[predicted <= np.quantile(predicted, 0.25)].mean():+.5f}")

    # The control that decides whether the DQN is earning its keep. A sample's *current*
    # loss is free -- the training loop already has it -- and it trivially bounds how much
    # loss can be removed, so it predicts gain on its own. If the DQN cannot beat it, the
    # honest description of the method is "sample by current loss", and the network, the
    # replay buffer and the reward are all decoration.
    # What was knowable at selection time. `loss_before` is measured by the forward pass on
    # the batch being trained, so an estimator cannot consult it for a candidate without
    # paying a detector pass per candidate -- which is the cost the whole design avoids.
    # `state_loss` is the stale EWMA the estimator really sees, and it is the honest ceiling.
    selection_baseline = None
    if "state_loss" in usable.columns and usable["state_loss"].notna().any():
        state_loss = usable["state_loss"].to_numpy(dtype=float)
        if np.std(state_loss) > 0:
            selection_baseline = float(stats.spearmanr(state_loss, gain).statistic)
            print(f"    spearman(state loss, gain)   = {selection_baseline:+.4f}   "
                  f"<-- the bar: knowable at SELECTION time")

    if "loss_before" in usable.columns:
        loss_before = usable["loss_before"].to_numpy(dtype=float)
        if np.std(loss_before) > 0:
            baseline = float(stats.spearmanr(loss_before, gain).statistic)
            print(f"    spearman(current loss, gain) = {baseline:+.4f}   <-- needs a forward pass per candidate")
            bar = selection_baseline if selection_baseline is not None else baseline
            verdict = ("beats the selection-time signal"
                       if spearman > bar + BASELINE_MARGIN
                       else "does NOT beat the signal it already sees")
            print(f"    -> {verdict}")
            return spearman, bar

    return spearman, selection_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="run_outputs/<run_id>, or the csv.gz itself")
    parser.add_argument("--per-epoch", action="store_true",
                        help="Also break the association down by epoch. Useful because the "
                             "estimator may only become informative once the detector has "
                             "stopped changing quickly.")
    args = parser.parse_args(argv)

    path = find_diagnostic(args.target)
    if path is None:
        print(f"No ivalue_diagnostic.csv.gz under {args.target}. "
              f"Was the run given --ivalue-diagnostic?")
        return 2

    import pandas as pd

    frame = pd.read_csv(path)
    print(f"{path}: {len(frame)} row(s), epochs {sorted(frame['epoch'].unique())}")

    overall, baseline = report(frame)
    if args.per_epoch:
        for epoch in sorted(frame["epoch"].unique()):
            report(frame[frame["epoch"] == epoch], label=f"epoch {epoch}")

    print()
    if overall is None:
        print("VERDICT: inconclusive -- not enough usable rows, or a constant predictor.")
        return 2
    if overall <= ABSOLUTE_FLOOR:
        print(f"VERDICT: FAIL (spearman {overall:+.4f} <= {ABSOLUTE_FLOOR}). The estimator "
              f"does not rank realised learning gain at all, so a selection strategy built "
              f"on it has nothing to exploit.")
        return 1
    if baseline is not None and overall <= baseline + BASELINE_MARGIN:
        print(f"VERDICT: FAIL (spearman {overall:+.4f} does not beat the selection-time "
              f"signal {baseline:+.4f} by {BASELINE_MARGIN}). It ranks gain, but no better "
              f"than the stale loss EWMA it already receives as an input -- so the learned "
              f"part is not paying for itself. Rank by that feature directly, or fix the "
              f"estimator and re-gate.")
        return 1
    margin = f" (baseline {baseline:+.4f})" if baseline is not None else ""
    print(f"VERDICT: PASS (spearman {overall:+.4f}{margin}). The estimator ranks learning "
          f"gain better than the free baseline; the downstream comparison is worth running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
