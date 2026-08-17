# The development sweep

One command to answer "did my change make anything worse?" across every model, traversal,
graph type, uncertainty head, and graph updater.

```bash
# Once, on known-good code.
python development_tools/sweep.py run --suite standard --tag baseline
python development_tools/sweep.py promote <sweep-id>

# After a change.
python development_tools/sweep.py run --suite standard --compare-to baseline
```

The comparison covers accuracy, calibration, uncertainty quality, and per-demographic
fairness, per configuration. Everything is read from the per-sample record tables that
`--uq-records` writes and scored through `evaluation/uq/scoring.py` -- the same path
`docs/uq_benchmark.md` describes. The stdout metrics dict is never an input.

---

## 1. Look before you spend GPU time

```bash
python development_tools/sweep.py plan --suite standard
```

Prints the matrix with a reason beside every cell that will not run, and exits nonzero if
any cell carries a config key the queue would drop. Free, and it catches the two mistakes
that otherwise waste an hour: a typo'd `--only` selector, and a suite whose new setting was
never routed through `ARG_MAPPING`.

## 2. Suites: one axis at a time

A full cross product of 5 usable detectors x 3 traversals x 4 graph types x 4 heads is 240
runs. A suite instead pins a **reference cell** and varies **one axis at a time**, so every
implementation of every component is exercised in `1 + sum(len(axis) - 1)` runs -- 18 for
`standard`, not 240. That is enough to answer "did my change break any traversal, detector,
head, or updater?", which is the development question. Interactions are opt-in:

```bash
python development_tools/sweep.py plan --suite standard --cross arch,traversal
```

| suite | cells | shape |
|---|---|---|
| `smoke` | 3 | 1 epoch, 200 nodes, minutes. Does anything run at all? |
| `standard` | 18 | 3 epochs, 2000 nodes. The default development check. |
| `full` | 26 | 10 epochs, 5000 nodes, plus the arch x traversal factorial. |

Axes and their variants live in `development_tools/sweep_suites.py::AXES`. Adding a
traversal or a detector means adding one entry there; the tests then check that its config
reaches the CLI and that it is either runnable or gated with a reason.

Narrow to what you touched:

```bash
python development_tools/sweep.py run --suite standard --only traversal --compare-to baseline
python development_tools/sweep.py run --suite standard --only arch=effnetdf,head=sngp
```

### The I-value traversal is one class

`--traversal-type` accepts `comprehensive`, `random`, and `i-value`. There used to be four
I-value spellings; the walk is now selected from the graph instead, because that is what it
always depended on:

- a **clustered** graph is built from disjoint race-gender groups, so a pointer cannot walk
  between them and has to hop;
- an **unclustered** graph is connected, so hopping would only discard the locality the
  I-value signal exists to exploit.

`i-value-cluster-hop` is therefore a retired spelling of `i-value` with identical behavior on
a clustered graph. The two `*-subcluster` names are retired too, but that is a **behavior
change**: Louvain-community area selection is gone. It never ran as designed -- its outlier
filter excluded every node whenever the variance was zero, `CapabilityManager` never enabled
a DQN for it so its I-values were random draws, and it yielded one node per step against ~17
for the other walks. Old names still parse and print a notice saying which case applies, so
the ~140 saved UI configs keep working.

The `traversal` axis pairs `i-value` with each graph type, which is what distinguishes the
two cells.

### Setting anything the training CLI accepts

`sweep.py` mirrors only a handful of `test_hierarchical.py`'s ~90 flags directly (`--seed`,
`--num-epochs`, `--cache-file`, `--data-root`, `--determinism`). Everything else goes
through `--set`, which applies to every cell:

```bash
# Subgroup-balanced splits, which is the lever when a short run collapses to the
# majority class.
python development_tools/sweep.py run --suite standard \
    --set fair_train=true --set fair_test=true --compare-to baseline

python development_tools/sweep.py plan --suite standard \
    --set num_workers=8 --set holdout=H1_diffusion_unseen
```

Keys are the config names from `ARG_MAPPING` (`fair_train`), and the dashed spelling of the
CLI flag works too (`fair-train`). Values are typed: `true`/`false` become booleans,
numbers become numbers, everything else stays a string -- which matters, because the queue
emits a bare flag for a True boolean and nothing at all for a False one, so a boolean
arriving as the string `"true"` would be passed as a *value* instead.

An unrouted key is refused with a suggestion rather than silently dropped:

```
ERROR: --set key 'epochs' is not a run-config key, so it would be dropped on the way
to test_hierarchical.py. Did you mean: num_epochs, switch_epochs?
```

A `--set` value beats both the suite and the forced defaults -- a command-line flag is the
most specific instruction available -- except that it still cannot turn off `uq_records`,
without which there is nothing to score.

For a change you want to keep, a suite file is the durable form:

```bash
echo '{"standard": {"reference": {"num_epochs": 1}}}' > /tmp/quick.json
python development_tools/sweep.py run --suite standard --suite-file /tmp/quick.json
```

### Checkpoint selection, and why it decides whether an axis measures anything

The best epoch is chosen by `--checkpoint-metric`, default **`auroc`**. This is not a detail:

`accuracy` was the original criterion, and on an imbalanced split it is close to useless. A
model that emits one class for every sample scores the majority-class prior — about 87% on
AI-Face — at epoch 1, and `current > best` can then never fire again. `best_epoch` freezes at
1, and **everything after epoch 1 is computed and discarded**: further training, graph
rewiring, node reduction. That is what made the `updater` axis produce record tables
byte-identical to a plain run — the mutations were real, they just landed after the epoch
that produced the checkpoint.

`balanced_accuracy` is prevalence-free but pins to *exactly* 0.5 for such a model, so it
ties rather than improving and freezes the same way. `auroc` is threshold-free and moves
whenever the ranking improves, which is why it is the default. It falls back to accuracy for
any epoch whose validation subsample is single-class, where both other metrics are undefined.

Every validation pass now reports `balanced_accuracy` and `auroc` alongside accuracy, from
`evaluation/uq/metrics.discrimination_metrics` — the same implementation the benchmark scores
with, so the per-epoch log and `results.csv` cannot drift apart. A collapsed model is called
out in the log as `NOTE (Validation): single_class_predictions`.

The criterion and its score are recorded in `determinism.json`; two runs selected on
different metrics are not comparable.

### Class imbalance: the two levers

The corrected AI-Face split is about 87% fake, and at that prior BCE is minimized
substantially by raising the output bias rather than by learning features. Measured on real
runs, the mean logit sits at +5.2 to +8.4 while the prior only justifies **+1.95** and the
class separation is only ~1.8 — so every probability lands above 0.5, accuracy reads as the
majority-class prior, and balanced accuracy pins at exactly 0.5. The ranking is nonetheless
informative (AUROC 0.67–0.80).

Note that `--fair-train` / `--fair-test` do **not** help here: they balance *demographic*
subgroups (race × gender) and leave the class prior untouched.

**`--balance-labels {none,train,all}`** attacks the prior directly.

- `train` balances only the training set; validation and test keep the population's real
  distribution, so reported numbers stay on the data as it is. Usually what you want.
- `all` balances every split, which makes a 0.5 threshold directly interpretable but stops
  measuring the deployed distribution.

Only ~13% of the corpus is real, so a balanced list is roughly twice the real count and most
fakes are discarded. The log says what was dropped, and a smaller-than-requested result is
called out rather than silently accepted.

**`--tune-threshold`** fixes the operating point instead, post hoc and for free — it reads
the record tables the run already wrote. Fitted on **val**, applied to test, with both
numbers reported:

```
--- Decision Threshold ---
  objective       : balanced_accuracy (fitted on 2000 val samples)
  threshold       : 0.993800 (default 0.5)
  NOTE            : at 0.5 the model predicted a single class for every val sample
  val balanced acc: 0.5000 -> 0.6803
  test @0.5        : accuracy 0.8755  balanced accuracy 0.5000
  test @0.9938     : accuracy 0.6060  balanced accuracy 0.6785
```

On measured runs the val-fitted threshold lands within 0.001–0.02 of the oracle, which is
what confirms the offset is systematic rather than noise. Accuracy *falls* — correctly: it
was only ever the prior.

Two things worth knowing:

- **Temperature scaling cannot do this.** Dividing a logit by any positive `T` preserves its
  sign, so no prediction moves across the boundary. Calibration and the operating point are
  separate problems; `temperature.py` addresses the first, this the second.
- **`--threshold-objective` defaults to `balanced_accuracy`, and should stay there.** Tuning
  for `accuracy` at 87% prevalence buys accuracy by getting the minority class wrong — on a
  deepfake detector, by missing real faces.

The threshold is written to `threshold_fit.json` beside the records and flows into the
sweep's `results.csv`, where it moves accuracy, balanced accuracy, *and* the definition of an
error — so selective prediction and uncertainty-error ranking move with it, which is correct:
those measure the mistakes the model actually makes at its operating point. The record table
itself is untouched, so every rank-based metric is invariant by construction.

Both are enabled in the suites' `FORCED` block for `tune_threshold` (free) and left off for
`balance_labels` (it changes which samples are trained on, so it is your call). Turn it on
for a whole sweep with `--set balance_labels=train`.

### What every suite forces on

`uq_records` with `val,test` splits (val is needed to fit temperature scaling on data the
test numbers never saw), `build_val_test_edges`, `enable_val_bias_inference`, and
`determinism: strict`.

Strict is not decoration. Same seed + same node cache + strict means the baseline and the
candidate score the *same samples in the same order*, so a delta is attributable to the code
change and a **paired** test is valid: McNemar on per-sample correctness, which is far
sharper than comparing two independent accuracies. `--determinism fast` is allowed and
stamps the report with a warning, because `docs/testing.md` measures cross-device drift at
about 3e-2 in probability space -- below that, a delta means nothing.

### Cells that are refused, and why

A gated cell is reported with its reason, never silently dropped -- a matrix with explained
holes is honest, one with missing rows is not.

- **Broken detectors.** `models/uncertainty/capabilities.py` marks seven of the twelve
  catalogued detectors `BROKEN`; the reason travels with the skip. `--allow-broken` runs them
  anyway, which is how the failure path gets tested.
- **`graph_manager=performance` with a non-I-value traversal.** Rewiring reads predicted
  I-values, and without a DQN capability `get_i_value` returns a random draw. Every node
  would read the neutral default, nothing would be rewired, and the cell would be
  indistinguishable from a static graph.
- **`reduction_strategy` in `{max_ival, min_ival, mix_max_ival}` with a non-I-value
  traversal.** `GraphReductionManager` raises on a trainer with no `get_i_value`.
  `reduction_strategy=random` works with any traversal.
- **Subclustering without `python-louvain`.** `HyperGraph.assign_louvain_subclusters` is then
  a documented no-op, so the cell would quietly run its non-subcluster fallback while
  claiming to test subclustering. It is not installed in the training env, so four
  `standard` cells are gated today; installing `python-louvain` enables them.

## 3. Running

Cells are dispatched as subprocesses through the web UI's `GPUQueueManager`, so they inherit
GPU discovery, memory-based admission, `CUDA_VISIBLE_DEVICES` pinning, and process
monitoring -- and a sweep-launched cell is the same kind of run as a UI-launched one. One
training process per GPU, so four run at a time on a four-GPU box.

```bash
python development_tools/sweep.py run --suite standard --tag my-change --compare-to baseline
```

A long sweep that dies partway resumes:

```bash
python development_tools/sweep.py run --suite standard --sweep-id <sweep-id>
```

Completed cells are skipped -- including a cell recorded complete whose record table has
since gone missing, which is re-run rather than scored as absent. `--force` re-runs
everything. `--launch-only` queues and exits; `--score-only` scores an already-run sweep.

Artifacts:

```
run_outputs/sweeps/<sweep-id>/
  manifest.json     suite, tag, git commit, node-cache sha256, and per-cell config,
                    run id, status, records paths, duration
  results.csv       tidy long scored table: whole-set rows, per-subgroup rows,
                    and the fairness reductions
  report/           the sweep's own figures and results.md
```

Cells are located through each run's `run_outputs/<run-id>/determinism.json`, never by
globbing -- the fingerprint records which configuration completed and where its records
landed, so a half-finished run cannot be folded in. This matters more than it sounds: a
configuration can fail mid-run, be caught by the per-configuration handler, and the process
still exits 0. The exit code is not evidence; the fingerprint is.

## 4. Baselines

```bash
python development_tools/sweep.py promote <sweep-id>
git add benchmarks/baseline_standard.csv benchmarks/baseline_standard.manifest.json
```

`promote` copies the scored table to `benchmarks/baseline_<suite>.csv` and its manifest
beside it. Both are small and committed, so **the git history of `benchmarks/` is the history
of the project's baselines** and any checkout can be compared against. It refuses a sweep
with incomplete cells unless `--allow-partial`, because a missing cell reads as `removed` in
every later diff -- a regression the code did not cause.

`python development_tools/sweep.py list` shows sweeps and promoted baselines with their
commits.

### What invalidates a baseline

The manifest records the git commit, the seed, the determinism mode, and the **sha256 of the
node cache**. A comparison prints these first and warns when they differ. Rebuilding the node
cache is the one that bites: the two sweeps then scored different samples, and the report
says so rather than reporting the difference as a regression.

## 5. Comparing

```bash
python development_tools/sweep.py compare \
    --baseline baseline:standard --candidate <sweep-id> --out run_outputs/reports/mine
```

`--baseline` and `--candidate` each accept a sweep id, a path to a `results.csv`, or
`baseline:<suite>`. Output is `comparison.md`, `comparison.csv`, and delta figures.

### Reading it

**Direction is declared, not inferred from the sign.** Lower ECE is better and higher
accuracy is better, so every row carries `direction` in `{better, worse, same, n_a}` resolved
from `evaluation/uq/compare.py`'s `HIGHER_IS_BETTER` / `LOWER_IS_BETTER`. The delta bars are
redrawn as *improvement*, so right is always better whatever the metric. A unit test fails if
a newly added metric belongs to neither set, so a new column cannot arrive silently
unoriented.

**`presence` is part of the answer.** A cell in the baseline and absent from the candidate is
the most important row in the table. Rows are outer-joined and marked `both` / `added` /
`removed`.

**A zero-length bar is marked.** "Measured and identical" gets a tick at zero; "absent or
degenerate" gets an x and a hatch. Without that, both draw nothing.

**Fairness travels through the same logic.** Subgroup rows are scored per `gt_gender`,
`gt_race`, `gt_age`, and `intersection`, then reduced to `disparity_range_<metric>`
(max − min across groups, the same definition `evaluate_model` reports as
`race_gender_overall_bias`), `disparity_mad_<metric>` (mean |group − overall|, matching
`race_gender_average_subgroup_bias`), and `worst_group_<metric>`. A spread is better when
smaller whatever the base metric's polarity; a worst-group value inherits its base metric's
polarity. Groups under 50 rows are **flagged, not dropped** -- dropping them would narrow the
range exactly when a group is too rare to measure, which reads as an improvement.

**The paired p-value is the one to trust for accuracy.** `identical` with p=1 is the expected
result of an unchanged re-run. A `-` means the record tables were not both on disk: a
promoted baseline is a committed CSV, and the hundreds of megabytes of tables behind it are
not committed, so on a fresh checkout there is nothing to align.

### Exit codes

`0` clean. `1` usage or setup error. `2` **hard failure** -- a cell that did not complete,
produced no records, is absent from the candidate, or stopped producing a metric the baseline
had. Metric movement is reported and never fatal: run-to-run movement is a research finding,
not a build error. `--strict-gate` fails on regressions too, for use as a pre-merge check.

A degenerate cell is counted once, not once per metric per demographic slice: a subgroup row
is a slice of the same measurement, not an independent finding.

## 6. The graph-updater axis

Both updaters were unreachable before this tool existed, and once reachable both turned out
to do nothing. The sweep is what surfaced that -- it flagged their record tables as
byte-identical to a plain run.

### `--graph-manager performance`: prune nodes by quantile

The original version rewired *edges* between nodes classified by absolute I-value
thresholds, and every update logged `42 weak / 0 strong` and `+0 / -0 edges`. Four
independent reasons, all measured:

- **The thresholds did not bracket the distribution.** "Strong" meant an I-value below 0.2
  and nothing ever fell there, so there was never a partner to rewire to.
- **The graph was too dense for edges to matter.** 823,814 edges over 1,304 nodes -- average
  degree 1,264, every node connected to ~97% of the graph. Tens of edges is ~0.01% of the
  topology, and `max_edges_per_node=10` sat so far below the real degree that the add path
  returned immediately every time.
- **It ran a handful of times**, on a step-counted interval ticked once per epoch, so only
  epochs after the best checkpoint could matter.
- **300 of 1,304 nodes were measured**, leaving the rest at the neutral default where they
  could not be classified either way.

So: thresholds are **quantiles** of the observed values (both sets non-empty by
construction, whatever scale the DQN outputs), the unit of change is the **node** rather than
the edge, updates run several times per epoch, and every node the traversal visits is
recorded.

```bash
python test_hierarchical.py --graph-manager performance --traversal-type i-value \
    --weak-quantile 0.9 --strong-quantile 0.1 --removal-fraction 0.02 \
    --graph-updates-per-epoch 4 --graph-remove-target strong \
    --use-cached --cached-nodes 2000 --determinism strict --uq-records
```

`--graph-remove-target` is the research knob: `strong` withdraws already-learned nodes
(curriculum pruning), `weak` withdraws the ones the model keeps failing on (noise pruning).
Both are on the `updater` axis. Pruning is capped at 5% of the graph per update and never
takes it below half its starting size -- a starved graph measures the traversal, not the
pruning. `restore_nodes` puts the most recent withdrawals back, so a validation drop is
recoverable.

The log now names the cuts, so a mismatch is visible immediately rather than inferred:

```
[PerformanceGraphManager] update 1: tracked 1304, cuts [0.0983, 0.8850],
    withdrew 26 strong node(s), graph now 1278 (26 withdrawn)
```

### Does tracking every node scale?

Yes, now. Two things had to change.

**Memory.** The old store was `dict[node] -> list of the last 100 I-values`, roughly 36
bytes per observation plus per-list overhead. Measured:

| nodes | old | new (EWMA) |
|---|---|---|
| 1,304 | 0.5 MB | 0.03 MB |
| 100,000 | 0.31 GiB | 0.019 GiB |
| **1,000,000** | **3.13 GiB** | **0.19 GiB** |

An exponential moving average is two numbers per node instead of a hundred -- 16x smaller,
and it drops the arbitrary 100-sample horizon in favour of smooth decay.

**Compute, which was always the real constraint.** Classifying a node needs its I-value, and
that is a DQN forward pass. A separate sampling pass over a million nodes is a million
forwards *per update*. So tracking is now opportunistic: training already computes an
I-value for every node the traversal visits, and `track_performance` folds that in at O(1).
Extra sampling is off by default (`--graph-manager-sample-nodes 0`) and exists only to seed
the quantiles faster at the start of a run.

Node removal is likewise a single batched `HyperGraph.remove_nodes` pass. Removing nodes one
at a time meant a linear index scan each -- O(N·k), which at a million nodes withdrawing 2%
would be 2e10 comparisons. That method also fixes a latent bug: `remove_node` left
`_node_data_map` stale, so `add_node` later rejected the node as a duplicate and restoration
silently did nothing.

### `--reduction-*`: withdraw a percentage per epoch

Unchanged in design, and it does work -- it removed 542 of 2,000 nodes in a real run. Its
effect was invisible only because the best checkpoint predated it, which the AUROC-based
`--checkpoint-metric` fixes.

## 7. Extending it

- **A new traversal, detector, head, or updater**: add a variant to
  `development_tools/sweep_suites.py::AXES`. If it needs a pairing the capability gate cannot
  see, add it to `axis_constraints` with a reason a user can act on.
- **A new metric**: `score_cell` picks it up automatically, but classify it in
  `evaluation/uq/compare.py` -- `HIGHER_IS_BETTER`, `LOWER_IS_BETTER`, or `IGNORED_COLUMNS`.
  `tests/unit/uq/test_compare.py::test_every_result_column_is_classified` fails until you do.
- **A new config key**: add it to `ARG_MAPPING` in `web_ui/gpu_queue_manager.py`, or it is
  dropped on the way to the CLI and the run quietly differs from the plan.
  `validate_config_keys` is what makes that a refusal instead of a surprise.

## Tests

```bash
pytest tests/unit/test_sweep_suites.py tests/unit/test_graph_updaters.py \
       tests/unit/uq/test_subgroups.py tests/unit/uq/test_compare.py \
       tests/unit/uq/test_compare_report.py

pytest --run-slow tests/functional/test_sweep_end_to_end.py
```

The functional test runs the real entrypoint over a synthetic node cache and the tiny
detector -- no dataset, no GPU -- then promotes, re-runs, and compares. Its central assertion
is that **an unchanged re-run compares clean**: every direction `same`, McNemar exactly p=1,
exit 0. If that stops holding, every real comparison is measuring noise on top of the change.

Per `docs/testing.md`, none of these assert anything about accuracy, calibration, or fairness
*quality* -- only that the machinery works.
