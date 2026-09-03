# The uncertainty benchmark

How to produce a scored, comparable uncertainty result. Four stages: build a node
cache once, run training with `--uq-records`, optionally aggregate ensembles, then
report.

The metrics dict printed to stdout is **not** a benchmark input. It carries batch
means of each raw uncertainty signal, and those live on incomparable scales, so it
cannot answer "which method is better". Everything below reads the per-sample record
tables instead.

For the *development* loop -- run a matrix across every detector, traversal, graph type,
head, and updater, then diff it against a committed baseline -- see
[dev_sweep.md](dev_sweep.md). It drives the same four stages below through one CLI and adds
the per-demographic breakdown and the run-to-run comparison that this document does not
cover.

---

## 1. Build a node cache

`AIFaceDataset.load()` constructs every node in all three splits and reads the 19 GB of
`*_quality.csv` sidecars **before** `--cached-nodes` truncates anything, so even a tiny
run pays the full load. Measured: 458s to load, 3.1s from a warm cache.

```bash
python development_tools/build_node_cache.py \
    --out node_cache/cached_nodes.pkl \
    --max-nodes-per-split 5000 --cached-nodes 2000
```

`--max-nodes-per-split` is what keeps the file small. `--cached-nodes` only sizes the
*balanced* view; without the cap the full lists hold all ~1.6M nodes (~7.5 GB).

The builder reads its own output back through `load_cached_nodes` before declaring
success. That check exists because the reader **returns `None`** for an unrecognized
shape rather than raising — so a rejected cache is indistinguishable from an absent one
in the logs, and the run silently pays the full load. It also refuses to write a cache
whose nodes carry no quality attributes, which would disable graph-distance uncertainty
for every run using it.

The pickle is generated, not committed: it embeds absolute `node_id` paths, `dill.load`
executes code from it, and it breaks on any rename to `AttributeNode`/`ImageFileData`.

---

## 2. Run training with records

```bash
python test_hierarchical.py \
    --use-cached --cache-file node_cache/cached_nodes.pkl --cached-nodes 2000 \
    --architectures resnestdf --traversal-type random --graph-type nonclustered \
    --num-epochs 10 --seed 42 --determinism strict \
    --uq-records --uq-records-splits val,test
```

Writes `run_outputs/<run_id>/<config>/records_{val,test}.csv.gz` plus a `.meta.json`
provenance sidecar, and `run_outputs/<run_id>/determinism.json`.

Keep `val` in `--uq-records-splits` unless you are certain you will not fit temperature
scaling: it must be fitted on data the reported test numbers never saw. Strict mode
forces one loader thread, so a val pass over a large split is slow.

### Distribution shift

```bash
python test_hierarchical.py ... --holdout H1_diffusion_unseen      # held-out generators
python test_hierarchical.py ... --corruption jpeg --corruption-severity 3
python test_hierarchical.py --list-holdouts                        # the tables
```

**Every holdout needs a paired `--holdout none` control on the same reduced training
set.** Holding out generators removes fake-only samples, so the class prior shifts;
without the control, "OOD degradation" is indistinguishable from "trained on less
data". The report refuses to print an OOD delta without it.

Held-out generators produce an **all-fake** OOD partition, so it supports two separate
questions: OOD *detection* (does uncertainty separate held-out from in-distribution?
single-class is fine — that is the point) and *shifted classification* (held-out fakes
plus in-distribution reals, so both classes are present and accuracy stays defined).
Classification metrics on the raw OOD partition are refused.

Severity 0 is byte-identical to clean, so the severity-0 row of a shift table *is* the
in-distribution row.

Note that graph-distance uncertainty is a static function of the graph and does not read
the image, so it is invariant under image corruption by construction. Its severity rows
are identical across the ladder, and that is correct rather than a bug.

---

## 3. Deep ensembles

```bash
python development_tools/launch_ensemble.py \
    --members 3 --arch resnestdf --seed 42 --num-epochs 10 \
    --determinism strict --use-cached --cache-file node_cache/cached_nodes.pkl \
    --records-splits test
```

Members differ in `--ensemble-member`, **not** `--seed`. Two reasons: the graph cache
key embeds the seed whenever a split has edges, so N seeds means N full graph rebuilds;
and an ensemble should differ in initialization, not in its training data, or
disagreement conflates initialization variance with curriculum variance.

Aggregation averages **probabilities, not logits** (logit averaging is a geometric mean
of odds, which is not the Bernoulli mixture's predictive mean). Members are discovered
from `determinism.json`, not by globbing, and averaging is refused when detector, head,
seed, git commit, or determinism mode differ, or when member indices collide.

`u_ens_variance` is the number to check first: **exactly zero means the members are the
same model**, and every ensemble metric is then measuring nothing.

Aggregate members that already finished:

```bash
python development_tools/launch_ensemble.py --aggregate-only --ensemble-id <id>
```

---

## 4. Report

```bash
python development_tools/uq_report.py \
    --records clean=run_outputs/<a>/<config>/records_test.csv.gz \
    --records jpeg3=run_outputs/<b>/<config>/records_test.csv.gz \
    --records H1=run_outputs/<c>/<config>/records_test.csv.gz \
    --ensemble run_outputs/ensembles/<id>/records_test.csv.gz \
    --out run_outputs/reports/<name> --detector resnestdf \
    --gate-detectors resnestdf,effnetdf,squeezenetdf,vistransformdf,xceptiondf
```

Produces `results.csv` (tidy/long, one row per detector × method × condition),
`results.md`, and PNG+PDF figures with stable filenames.

`--gate-detectors` is what turns the gating heatmap into a statement about what was
*considered* rather than what happened to run. Gated cells appear as hatched holes with
reason codes — never absent rows, and never distinguished by colour alone.

### Reading the output

- **Report E-AURC, not raw AURC.** Raw AURC is dominated by the base error rate, so a
  stronger-but-worse-calibrated detector can win for the wrong reason.
- **Calibration cells are N/A for graph-distance methods, never zero.** Those methods
  produce no calibrated probability, so there is nothing for them to be right about; a
  zero would read as perfect calibration.
- **Max-probability, entropy, margin, and temperature scaling are rank-identical.** For
  a single Bernoulli, entropy and margin are monotone in max-probability and temperature
  scaling is a monotone logit rescaling, so their AUROC-error / E-AURC /
  accuracy@coverage are equal *by construction*. The table marks them `= base` rather
  than printing duplicate columns. Temperature scaling's entire contribution is
  calibration.
- **`graph_degree_only` is the ablation control.** Graph uncertainty adds
  `penalty_weight/sqrt(degree+1)`, so without a degree-only comparison you cannot tell
  whether "graph distance predicts error" is really "low-degree nodes are harder".
  Since graph distance is the novel contribution, that is the reviewer question you
  cannot afford to be unable to answer.
- **A `degenerate_constant_score` flag means the uncertainty signal is identically
  constant** — the signature of an un-diversified ensemble or zero-`p` MC dropout.
- **Cells below 99% coverage are refused**, guarding against `evaluate_model`'s
  per-batch `except Exception: continue` computing a headline number on a fraction of
  the data.
- **`cost_forward_passes` / `cost_training_runs` travel with every row.** A table where
  ensembles win without showing 5× the training cost is misleading.

---

## Composition of the corrected dataset

Read this before interpreting any prior-sensitive metric (Brier, NLL, ECE via the base
rate). Real sources are FFHQ, IMDB-WIKI (`wiki/`), and the `real/` portions of FF++,
DFDC, DFD, and Celeb-DF-v2. `CelebA` and `casia-webface` were dropped.

| | train | val | test |
|---|---|---|---|
| rows | 562,214 | 240,333 | 286,969 |
| real | 73,592 | 31,401 | 37,958 |

**The corrected dataset is 13.1% real, against AI-Face v2's published ~24%.** The six
real sources hold 142,951 images in this copy versus v2's 400,885, so roughly 258k real
images are absent. Recovering them means re-downloading. `Target` is now a pure function
of (source, second path component) in all three splits.

The real/fake encoding is inconsistent upstream: `ff++` and `celebdf` use `real/` vs
`crop_img/`, while `dfdc` and `dfd` use `real/` vs `fake/`. The robust predicate is
`second_component == "real"`, never `!= "fake"`.

`taming_transformer:VQGAN` contains a colon, so any group id reaching a filename or a
cache key goes through `sanitize_key_component`.
