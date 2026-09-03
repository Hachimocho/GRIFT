# Full-scale graph construction: measured properties

Numbers from the one-time warm build of the whole AI-Face node cache
(`node_cache/cached_nodes_full.pkl`, 562,214 train / 240,333 val / 286,969 test =
1,089,516 nodes) at `--edge-construction knn --knn-neighbors 50 --seed 42`, thresholds
`quality 0.800 / symmetry 0.750 / embedding 0.700`. Recorded 2026-08-19; the caches these
numbers describe live in `graph_cache/` under `..._v3_edges.csv.gz`.

## Isolation: the open problem

**27.8% of train nodes (156,066 of 562,214) have no surviving edge after filtering and are
connected to a randomly chosen partner.** The same holds across splits and graph types:

| graph | split | nodes | edges | avg degree | isolated | isolated % |
|---|---|---|---|---|---|---|
| nonclustered | train | 562,214 | 2,346,231 | 8.35 | 156,066 | 27.8% |
| nonclustered | val | 240,333 | 887,438 | 7.39 | 70,523 | 29.3% |
| nonclustered | test | 286,969 | 1,087,115 | 7.58 | 83,207 | 29.0% |
| clustered | val | 240,333 | 521,610 | 4.34 | — | — |
| clustered | test | 286,969 | 611,895 | 4.26 | — | — |

This is **not** a full-scale artifact: the 5,000-node cache isolated 1,697 nodes (33.9%),
so the rate is a standing property of the similarity thresholds, not of dataset size. What
*is* new at scale is the absolute count -- 156k random edges in the training graph.

Why it matters, and why it does not invalidate the current comparison:

- Every arm of a sweep reads the *same* cached, seeded graph, so the random partner edges
  are byte-identical across arms. A traversal comparison over this graph is still fair.
- But roughly a quarter of the graph's connectivity is *noise* rather than similarity. Any
  claim of the form "the graph structure carries signal the traversal exploits" is weakened
  by that fraction, and a reviewer will ask. Worth fixing before the structural claims are
  written up.

Things to try, cheapest first:

1. **Lower the thresholds.** The cascade shows where the loss is (train split): 22,561,503
   candidate pairs -> 7,174,952 after quality (31.8% pass) -> 2,475,892 after symmetry
   (34.5% pass) -> 2,346,231 after embedding (94.8% pass). Symmetry at 0.750 is the
   tightest gate and the least principled -- facial symmetry similarity is a weak reason to
   believe two faces should be graph neighbours.
2. **Raise `--knn-neighbors`.** k=50 candidates yields 40.1 per node after symmetrise and
   dedup; a node whose 50 nearest neighbours all fail the filters gets nothing. k=200 costs
   4x the candidate list (the filters are ~1.5 min, so this is nearly free) but does not
   change the k-NN search cost, which is 98% of the build.
3. **Guarantee a real neighbour instead of a random one.** The isolated-node fallback
   currently picks a random partner. Connecting to the *nearest* neighbour by embedding --
   which the k-NN already computed and then discarded -- would keep the graph connected
   without injecting noise. This is the principled fix and it is cheap.
4. **Report degree distribution, not just the mean.** Average degree 8.35 hides a
   spike at degree 1 of ~156k nodes. Any figure describing the graph should show the
   histogram.

Note also that 43,191 train nodes (7.7%) have no `face_embedding` at all (92.3% quality
coverage) and receive a zero vector, so they are mutually nearest under cosine and form one
dense blob. That is a separate defect from isolation and interacts with fix 3.

## Build cost

| build | wall clock | note |
|---|---|---|
| node cache (all 1,089,516 nodes) | ~7 min load + pickle | 4.8 GB, `dataset_load_seconds` 412.3 |
| nonclustered, all 3 splits | 3 h 23 m (12,187 s) | 89.4 nodes/s; 99.6% is edge construction |
| clustered, all 3 splits | 59 m | k-NN runs per race-gender subgroup |

Two facts worth keeping:

- **The k-NN search is ~98% of the build; the filter cascade is ~1.5 min.** Optimising the
  filters is pointless. Optimising the search is the only lever.
- **The search is sub-quadratic in practice.** train (562,214) took 89.7 min of search,
  val (240,333) took 51.8 min: 2.34x the nodes for 1.73x the time. It is bound by memory
  traffic in sklearn's chunked distance computation, not by flops, so quadratic
  extrapolation over-predicts by ~20%.
- Clustered is 3.4x faster than nonclustered because per-subgroup k-NN costs
  `O(sum of m_i^2)`, far below `O(N^2)`.

Strict determinism pins `OMP_NUM_THREADS=1`, so the search is single-core. Raising the
thread count does **not** help at these shapes -- measured at n=15,000: 4.70 s on one
thread vs 5.24 s on sixteen. There is nothing to win by relaxing it.

## Traversal cost defects found while scaling (2026-08-20)

**The periodic full-graph I-value refresh (fixed).** `IValueTraversal` refreshed cached
I-values on a `predictor_update_period` timer by calling `trainer.get_i_value` for *every
node in the graph, for every pointer* -- one DQN forward pass each. `self.t` counts
`traverse()` calls, not walk steps, so at period 50 and ~313 batches per epoch it fired 6
times an epoch. Measured on the 562,214-node graph: ~6 min per refresh, ~36 min of a
38.6 min epoch, GPU at 6%, with training running in ~16 s bursts of ~850 nodes between
sweeps. The timer now clears the cache (O(1)) and `_get_i_value(fetch_on_miss=True)`
re-fetches only what the walk looks at. **Epoch time 38:40 -> 4:45, an 8.1x speedup**, same
10,016 nodes trained.

This had been half-fixed earlier: the `reset_pointers` pre-warm was removed and the class
docstring updated to say pre-warming was gone, but the periodic call survived. The three
now-unreachable methods (`update_i_values`, `_update_i_values_connected`,
`_update_i_values_hop`) were deleted rather than left in place -- an O(N) full-graph DQN
sweep that nothing calls is how it came back the first time.

**`Node.get_adjacent_nodes` (memoised).** Rebuilt a Python list from `self.edges` on every
call; `get_degree` calls it too, and `DQNCapability._get_dqn_features` takes a degree per
node, so a full-scale profile recorded **1,344,952 calls for 1,504 trained nodes** (30.5 s
of a 40.9 s window). Now memoised with the cache *validated* rather than invalidated -- on
list identity (reassignment: `node.edges = []`, `canonicalize_edge_order`) and length
(append via `add_edge`, removal via `GraphReductionManager` and `HyperGraph.remove_nodes`),
both O(1). `Edge.set_node1/set_node2/set_nodes` invalidate explicitly, being the one path
that changes a neighbour without touching either list. Verified bit-identical on a
full training run.

Worth recording honestly: this was expected to be the stall and **was not**. Wall clock at
full scale was unchanged, because the 1.34M calls came from the full-graph refresh above --
one call per node per sweep, so the cache never hit twice within a sweep. It is a real
saving that shows up once the same node is asked about repeatedly, and it was not the
bottleneck.

**`ComprehensiveTraversal` rebuilds an O(N) unvisited set per batch** (`set(node for node
in self.graph.nodes if node not in pointer['visited'])`). At 562k nodes and ~313 batches
that is ~176M membership tests per epoch, roughly 31 s -- noticeable but not a stall.
Left alone; noted here as the same class of defect.

**Image loading is not the bottleneck.** At full scale `ImageFileData.load_data` is 0.49 s
of a 40.9 s training window. `--preprocess-workers` threads the decode (worth ~7%) and
deliberately leaves the transform serial: `CNNModel.train_transforms` is five random
augmentations drawing from the *global* torch RNG, so threading it changes which
augmentation each image gets and the RNG state every later consumer inherits -- measured, it
produced a different record digest and node count. Ordering results by submission fixes
their arrangement, not a shared mutable RNG.

## The training-budget confound (fixed)

`--train-steps` bounds how far a traversal *walks*; the number of nodes actually trained on
was a hardcoded constant that **differed between the arms being compared**:
`BasicTrainingCapability` 5000, `DQNCapability` 10000. Realised over 15 epochs: random
~51,156 nodes, comprehensive 75,000, every i-value arm 150,240 -- so i-value trained on
2.0x comprehensive and 2.9x random, and any accuracy gap confounded sample *selection* with
sample *count*. The traversal half of Claim 1 was unanswerable from that sweep.

Two things were needed to make budgets equal, not just equally configured:

1. `--max-nodes-per-epoch`, applied to both capabilities, with exact truncation so the DQN
   path lands on 10000 rather than overshooting to 10016.
2. A tolerance for empty traversal batches. A walk returns `[]` at a *local* dead end --
   `RandomTraversal` gives up after 100 steps without an unvisited node -- and both
   collection loops treated the first empty batch as end-of-epoch. That is why the random
   arm trained on as few as 264 nodes against a 5000 budget while the DQN arm hit its full
   cap: the very next `traverse()` call after an empty one returned 6,302 nodes.
   `MAX_CONSECUTIVE_EMPTY_BATCHES = 50` now applies to both paths, so they give up at the
   same point.

Verified at full scale before relaunching: random, comprehensive and i-value each report
exactly 10,000 nodes per epoch.

Note that `--train-steps` still means different things per traversal -- walk steps for
`RandomTraversal`, `traverse()` calls for `IValueTraversal`. It is set to 4,000,000 in
`benchmarks/suite_claim1_matched.json` purely so it never binds; `max_nodes_per_epoch` is
the real budget. Unifying that meaning would be a good follow-up.
