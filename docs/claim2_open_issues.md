# Claim 2 (I-values as an uncertainty measure): open issues

**Claim under test.** Is a predicted I-value a better uncertainty signal than the
established uncertainty methods — softmax confidence, entropy, margin, MC-dropout,
evidential, BatchEnsemble, SNGP?

**Status: not yet answerable.** `benchmarks/suite_claim2_uncertainty.json` runs, scores, and
produces a clean comparison, but the comparison it produces is not yet the one the claim
needs. Recorded here for revision before any Claim 2 result is written up.

Evidence as of `run_outputs/sweeps/sweep_20260819_111405_standard_360d12` (19/19 cells,
3 epochs, 5,000-node graph, single seed) — mean AUROC-error across cells:

| method | AUROC-error |
|---|---|
| evidential | 0.855 |
| baseline_maxprob | 0.845 |
| mc_dropout | 0.843 |
| batchensemble | 0.842 |
| graph_hybrid_distance | 0.553 |
| graph_degree_only (control) | 0.529 |
| sngp | 0.513 |
| **ivalue** | **0.4997** |

## 1. `ivalue` scores at chance, and it is not a degeneracy artifact

0.4997 AUROC-error over 6 cells is exactly coin-flip: the predicted I-value carries no
information about which test samples the model gets wrong. This is *not* the usual
"constant column" failure — the column holds 4,661 distinct values, every cell came back
`status=ok`, and no `DEGENERATE_UNCERTAINTY` flag was raised. The signal is real-valued
and genuinely uninformative.

Two readings, and the suite cannot currently tell them apart:

- **The honest null.** I-value predicts *expected learning gain on the training graph*.
  That is a different quantity from *probability this test prediction is wrong*. A DQN
  trained to rank training nodes by how much they would teach the model has no reason to
  transfer to test-time error ranking. If so, Claim 2 is false as stated and the
  interesting paper is Claim 1.
- **An undertrained DQN.** 3 epochs on a 5,000-node graph is very little signal for the
  DQN, and the I-value head is only exercised on nodes the traversal actually visits.
  At 30 epochs on the full graph the estimator may become informative.

**Resolution:** re-run Claim 2 at Claim 1's scale before drawing any conclusion. If
`ivalue` is still at 0.50 with a well-trained DQN on the full dataset, that is a
publishable negative result — but it must be measured, not assumed.

## 2. The comparison is framed against the wrong baseline

`ivalue` is a *learned* confidence estimator with access to the model's own state. The
graph-distance methods (`graph_attribute_distance`, `graph_embedding_distance`,
`graph_hybrid_distance`, `graph_degree_only`) are unlearned geometric heuristics and score
0.53–0.55 — barely above chance themselves. Beating them proves nothing.

The comparison that matters is `ivalue` vs **`baseline_maxprob`** (0.845): free, one
forward pass, no extra training. Any method that does not beat max-probability is not
worth its cost. The report must state this pairing explicitly rather than presenting a
flat ranking in which `ivalue` sits beneath both the strong methods and the weak controls
without distinguishing them.

## 3. `ivalue_rank` is structurally unmeasurable and should probably be dropped

`ivalue_rank` returns NaN by design: the registry declares
`rank_equivalent_to="ivalue"`, so `collapse_rank_equivalents` removes it from every
rank-based metric (AUROC-error, E-AURC, risk-coverage) because a monotone transform cannot
change a rank statistic. It survives only to make the calibration story explicit — an
I-value is not a probability, so it has no ECE/Brier/NLL of its own.

As written it contributes one all-NaN row per cell to the results table. Either drop it
from `CANDIDATE_METHODS` or give it a real job: rank-normalise, then Platt-scale against
val error labels so it becomes a probability with a *comparable* ECE. The second option
is the one that would actually strengthen the claim.

## 4. Two strong comparators are missing from the sweep

- **`deep_ensemble`** — the standard reference point for uncertainty quality, and absent.
  It needs multiple runs per cell, which the sweep's one-run-per-cell shape cannot
  express. Route: `development_tools/launch_ensemble.py --members 3 --traversal-type
  i-value`, then fold the members in with `uq_report --ensemble`. Until this lands,
  "better than traditional uncertainty methods" is being tested against a weakened field.
- **`temperature_scaling`** — deliberately not scored inside the sweep, because it is
  fitted post-hoc on the val split by `uq_report`. Correct as an implementation choice,
  but it means the sweep's calibration numbers are all *uncalibrated*, and temperature
  scaling alone often closes most of an ECE gap. Any ECE-based Claim 2 statement needs the
  post-hoc pass run first.

## 5. Statistical strength is insufficient for a negative result

Every number above is one seed, 3 epochs, 5,000 nodes, one detector (`resnestdf`).
`compare.paired_accuracy_test` gives a paired test on *accuracy*, but there is no paired
test on AUROC-error, so "0.4997 vs 0.845" currently has no interval attached.
`metrics.bootstrap_ci` exists and is seeded — the comparison report should carry
bootstrap intervals on AUROC-error per method, and Claim 2 needs ≥3 seeds. Publishing a
null demands a tighter bound than publishing an effect.

## 6. The suite's axis is confounded

`suite_claim2_uncertainty.json` varies `uncertainty_head` across cells
(`none`/`evidential`/`batchensemble`/`sngp`) and then compares *all* methods within each
cell. So `ivalue` is measured on four differently-trained models, and its mean over 6
cells averages across architectures whose accuracy differs. For a clean
method-vs-method comparison, the model must be held fixed: score every uncertainty method
on **one** trained model per detector, and treat the head as a separate axis with its own
`ivalue` measurement rather than pooling.

## Revision checklist

- [ ] Re-run at Claim 1 scale (full dataset, 30 epochs) before interpreting 0.4997.
- [ ] Restructure the suite so uncertainty methods are compared on a fixed model.
- [ ] Name `baseline_maxprob` as *the* comparator in the report; demote graph-distance
      methods to controls.
- [ ] Add `deep_ensemble` via `launch_ensemble.py` + `uq_report --ensemble`.
- [ ] Run the post-hoc `temperature_scaling` pass before quoting any ECE.
- [ ] ≥3 seeds, with bootstrap CIs on AUROC-error.
- [ ] Decide `ivalue_rank`: drop, or Platt-scale it into a real probability.
