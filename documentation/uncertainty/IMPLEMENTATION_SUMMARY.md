# Uncertainty Quantification Implementation Summary

This document builds on `documentation/uncertainty/README.md`. The older README
captured the original plan and early milestones; this file records the full
implementation state as it exists now, including the work that has been added
after the original plan was written.

## Current Status

Uncertainty quantification is implemented as a final test-time analysis layer.
It does not affect model training, validation-time checkpoint selection, graph
traversal, graph reduction/restoration, calibration, or bias loss computation.

The implemented pipeline can:

- select uncertainty methods from the Web UI or CLI
- collect final-test prediction records
- optionally collect train prediction records for methods that need a reference
  set
- optionally build a test-graph neighbor map for graph uncertainty
- compute and save per-node uncertainty scores
- save summary JSON files for each method
- surface uncertainty artifacts in the Web UI results pages

Supported method names are:

```text
msp,ddu,trust_score,graph
```

The CLI flag is:

```text
--uncertainty-methods msp,ddu,trust_score,graph
```

Aliases accepted by the parser include `trust`, `trust-score`,
`trust score`, `graph_uncertainty`, `graph-uq`, and `graph uq`.

## Advisor Constraint

The governing rule from the original uncertainty plan still applies:

Use or calculate uncertainty only during test time.

The implementation follows that rule by running uncertainty scoring only after:

1. normal training finishes
2. the best checkpoint is loaded for final testing
3. final test prediction records are generated

Uncertainty is not used during:

- model training
- validation-time model selection
- graph traversal
- graph reduction or restoration
- calibration during training
- bias loss computation

If uncertainty is later used to change predictions or actions, the preferred
policy from the older README is to treat highly uncertain samples as fake rather
than ignore them, because missed fake faces are the main risk.

## Implementation Flow

The current end-to-end flow is:

1. The user selects uncertainty methods in `web_ui/templates/configure.html`.
2. The UI stores selected methods as the comma-separated config value
   `uncertainty_methods`.
3. `web_ui/gpu_queue_manager.py` maps that config key to
   `--uncertainty-methods`.
4. `test_helpers/args_utils.py` defines the CLI flag.
5. `test_hierarchical.py` normalizes the selected method names with
   `parse_uncertainty_methods`.
6. Final test evaluation runs with `return_prediction_records=True` only when
   uncertainty methods were selected.
7. `save_uncertainty_test_inputs` writes final-test prediction records and an
   input summary under the run output directory.
8. If DDU or Trust Score is selected, `test_hierarchical.py` runs an additional
   train-set inference pass to collect train prediction records for fitting.
9. If graph uncertainty is selected, `test_hierarchical.py` builds a neighbor
   map from the final test graph.
10. `run_selected_uncertainty_methods` dispatches to the selected scorers in
    `evaluation/uncertainty.py`.
11. The GPU queue manager scans completed run outputs and stores discovered
    uncertainty artifacts under `results.uncertainty`.
12. `web_ui/templates/results.html` and `web_ui/templates/run_details.html`
    display uncertainty availability and method summaries.

## Files Changed For This Feature

Core implementation:

- `evaluation/uncertainty.py`
- `test_hierarchical.py`
- `test_helpers/args_utils.py`

Web UI and queue integration:

- `web_ui/templates/configure.html`
- `web_ui/gpu_queue_manager.py`
- `web_ui/templates/results.html`
- `web_ui/templates/run_details.html`

Documentation:

- `documentation/uncertainty/README.md`
- `documentation/uncertainty/IMPLEMENTATION_SUMMARY.md`

## Artifacts

All uncertainty artifacts are written below the specific config output directory:

```text
run_outputs/<run_id>/<config_description>/uncertainty/
```

The shared input artifacts are:

```text
summary.json
final_test_predictions.csv
```

`summary.json` contains:

- selected methods
- number of final-test prediction records
- path to the prediction-record CSV
- generation timestamp

`final_test_predictions.csv` contains:

- `node_id`
- `label`
- `prediction`
- `logit`
- `probability_fake`
- `probability_real`
- `confidence`
- `correct`
- `false_negative`
- `has_face_embedding`
- `edge_count`

Each scoring method also writes a method-specific score CSV and summary JSON.

## Shared Summary Policy

All implemented scoring methods use the same policy summary helper:

```text
summarize_uncertainty_as_fake_policy
```

For each method, records are sorted from most uncertain to least uncertain. The
summary then reports what would happen if the top 5%, 10%, and 20% most
uncertain test samples were forced to the fake class.

The summary includes:

- number of scored records
- original accuracy
- original false negatives
- original fake recall
- number of samples flagged as fake for each policy
- uncertainty threshold for each policy
- remaining false negatives
- false-negative reduction
- accuracy after fake override
- fake recall after fake override

This matches the project priority from the older README: uncertainty should be
judged by whether it helps identify risky test samples, especially fake faces
that would otherwise be missed.

## MSP

Status: implemented.

MSP means Maximum Softmax Probability. For the binary detector, confidence is
the larger of:

```text
probability_fake
probability_real
```

The uncertainty score is:

```text
msp_uncertainty = 1 - confidence
```

Implemented outputs:

```text
msp_scores.csv
msp_summary.json
```

`msp_scores.csv` contains:

- `node_id`
- `label`
- `prediction`
- `confidence`
- `msp_uncertainty`
- `correct`
- `false_negative`

MSP needs no train-reference pass and no graph data. It is the simplest
baseline and is fully post-hoc.

## DDU

Status: implemented as a simplified logit-space DDU.

The original plan described DDU as a feature-density method using intermediate
features, usually penultimate features. The current implementation preserves
the post-hoc density idea but uses detector logits as a lightweight proxy so it
does not require per-architecture feature hooks.

Current behavior:

1. Run train samples through the trained detector after training.
2. Split train logits by true class.
3. Fit one Gaussian to fake-class train logits and one Gaussian to real-class
   train logits.
4. Score each test logit by the maximum density under the two fitted Gaussians.
5. Rank-normalize in-distribution scores.
6. Convert density support into uncertainty with:

```text
ddu_uncertainty = 1 - rank_normalized_in_distribution_score
```

Implemented outputs:

```text
ddu_scores.csv
ddu_summary.json
```

`ddu_scores.csv` contains:

- `node_id`
- `label`
- `prediction`
- `logit`
- `confidence`
- `p_fake`
- `p_real`
- `in_dist_score`
- `ddu_uncertainty`
- `correct`
- `false_negative`

`ddu_summary.json` also records:

- fake-logit Gaussian mean and standard deviation
- real-logit Gaussian mean and standard deviation
- number of train fake records
- number of train real records

Skip behavior:

- If no train prediction records are available, DDU writes a skipped summary.
- If one class is missing from the train-reference records, DDU writes a
  skipped summary.

Important limitation:

This is not full penultimate-feature DDU. It is a pragmatic logit-space
approximation that keeps the implementation architecture-independent.

## Trust Score

Status: implemented as a simplified 1-D logit-space Trust Score.

The original plan described Trust Score as an embedding-neighborhood method.
The current implementation uses train logits as the reference embedding space.

Current behavior:

1. Run train samples through the trained detector after training.
2. Split train logits by true class.
3. For each test sample, use the predicted class as the same-class reference.
4. Find the nearest train logit in the predicted class.
5. Find the nearest train logit in the other class.
6. Compute:

```text
trust_score = distance_to_other_class / distance_to_same_class
```

Low trust means the prediction is not well supported by nearby train examples,
so uncertainty is computed as:

```text
trust_score_uncertainty = 1 - rank_normalized_trust_score
```

Implemented outputs:

```text
trust_score_scores.csv
trust_score_summary.json
```

`trust_score_scores.csv` contains:

- `node_id`
- `label`
- `prediction`
- `logit`
- `confidence`
- `d_same_class`
- `d_other_class`
- `trust_score`
- `trust_score_uncertainty`
- `correct`
- `false_negative`

Skip behavior:

- If no train prediction records are available, Trust Score writes a skipped
  summary.
- If one class is missing from the train-reference records, Trust Score writes a
  skipped summary.

Important limitation:

This is not full embedding-space Trust Score. It is a lightweight approximation
that keeps the scorer independent of model-specific feature extraction.

## Graph Uncertainty

Status: implemented as neighbor prediction consistency in the test graph.

The older README described graph uncertainty as a method based on graph density,
edge strength, sparsity, and neighbor disagreement. The current implementation
focuses on neighbor disagreement because reliable similarity metadata is not yet
guaranteed on all graph edges.

Current behavior:

1. Build a neighbor map from final test nodes and their adjacent nodes.
2. For each test node, keep neighbors that are also present in the test
   prediction records.
3. Compare the node prediction with neighbor predictions.
4. Compute neighbor homophily:

```text
neighbor_homophily = same_prediction_neighbors / valid_neighbors
```

5. Compute graph uncertainty:

```text
graph_uncertainty = 1 - neighbor_homophily
```

The scorer also records neighbor probability variance using neighbor fake
probabilities.

Implemented outputs:

```text
graph_scores.csv
graph_summary.json
```

`graph_scores.csv` contains:

- `node_id`
- `label`
- `prediction`
- `confidence`
- `neighbor_count`
- `neighbor_homophily`
- `neighbor_prob_variance`
- `graph_uncertainty`
- `is_bridge_node`
- `correct`
- `false_negative`

Skip and fallback behavior:

- If no neighbor map is available, graph uncertainty writes a skipped summary.
- If a non-bridge node has no valid scored neighbors, it receives
  `graph_uncertainty = 0.5`.
- `run_graph_uncertainty` supports a `bridge_node_ids` argument so fallback
  bridge nodes can be excluded and flagged.

Important limitation:

The current call site builds the neighbor map but does not yet pass bridge-node
IDs. The helper is ready to drop bridge nodes when those IDs are available, but
the pipeline still needs an upstream source for them.

## Graph Caveat From The Original Plan

The older README warned that some graph builders add fallback edges to force
connectivity. Those edges are not true similarity edges and should not be
treated as evidence of local sample density.

That caveat still matters. The current graph implementation avoids edge-weight
claims and only uses neighbor prediction agreement. Future graph uncertainty
work should still:

- confirm where true similarity values are stored
- distinguish similarity edges from fallback connectivity edges
- pass bridge/fallback node IDs into `run_graph_uncertainty`
- decide whether fallback-edge-dependent nodes should be skipped or reported
  separately

## Web UI Behavior

The configuration page includes an Uncertainty Methods selector with checkboxes
for:

- MSP
- DDU
- Trust Score
- Graph Uncertainty

The selected methods are saved as:

```text
uncertainty_methods
```

The configuration preview displays selected methods, or `Disabled` when none
are selected.

After a run completes, the queue manager scans:

```text
run_outputs/<run_id>/*/uncertainty/summary.json
```

For each uncertainty output it collects:

- config description
- input summary path
- selected methods
- number of prediction records
- prediction records file
- method summaries from `*_summary.json`
- all files in the uncertainty artifact directory

The results table displays a UQ badge when artifacts exist. The run details page
shows selected methods, record counts, summary values, and score-file paths.

## Completed Milestones

Milestone 1: Config Selection.

Done in commit `2e279f4` (`Add uncertainty method config selection`).

Completed work:

- Added uncertainty method checkboxes to the config page.
- Saved selected methods as a comma-separated config value.
- Added selected methods to the config preview.
- Added `--uncertainty-methods` to the parser.
- Added GPU queue pass-through for the CLI flag.

Milestone 2: Final Test Data Collection.

Done in commit `fddc88c` (`Collect final test inputs for uncertainty scoring`).

Completed work:

- Added method-name parsing and normalization.
- Added optional final-test prediction record collection.
- Saved final-test prediction records to CSV.
- Saved an uncertainty input summary JSON.
- Kept collection disabled unless uncertainty methods are selected.

Milestone 3: MSP Baseline.

Done in commit `4988187` (`Add MSP uncertainty baseline`).

Completed work:

- Added `evaluation/uncertainty.py`.
- Implemented MSP scoring from final-test confidence.
- Added per-node MSP score CSV output.
- Added MSP summary JSON output.
- Added top-uncertain-as-fake policy summaries.

Milestone 4: Results Page Display.

Done in commit `89d375b` (`Show uncertainty artifacts in run results`).

Completed work:

- Added queue-manager artifact discovery.
- Stored discovered artifacts under `results.uncertainty`.
- Added a UQ column to the results table.
- Added an uncertainty summary section to the run details page.

Milestone 5: DDU.

Done in commit `5eb7ba8`
(`Implemented DDU, Trust score, Graph Uncertainty (whether neighbor nodes agree or not)`).

Completed work:

- Added train-reference collection when DDU is selected.
- Implemented logit-space class-conditional Gaussian fitting.
- Implemented per-node DDU scores and summary output.
- Added skipped-summary behavior for missing train-reference data.

Milestone 6: Trust Score.

Done in commit `5eb7ba8`.

Completed work:

- Reused train-reference collection when Trust Score is selected.
- Implemented nearest-neighbor trust scoring in logit space.
- Implemented per-node Trust Score outputs and summary output.
- Added skipped-summary behavior for missing train-reference data.

Milestone 7: Graph Uncertainty.

Done in commit `5eb7ba8`.

Completed work:

- Added test-graph neighbor-map construction.
- Implemented neighbor-prediction-consistency uncertainty.
- Recorded neighbor count, homophily, probability variance, and bridge-node
  flags in graph score outputs.
- Added skipped-summary behavior for missing neighbor maps.

## Dependencies

DDU uses:

```text
scipy.stats.norm
```

`scipy` is already present in `requirements-test.txt` and `environment.yml`.

The uncertainty implementation also uses:

- `numpy`
- `csv`
- `json`
- `datetime`
- `pathlib`

## What Is Still Not Implemented

The current code intentionally avoids several larger changes:

- No uncertainty-aware training.
- No uncertainty-aware validation checkpoint selection.
- No calibration training.
- No subgroup ECE focus.
- No uncertainty-driven traversal.
- No uncertainty-driven graph reduction/restoration.
- No full penultimate-feature DDU.
- No full embedding-space Trust Score.
- No edge-similarity graph uncertainty.
- No guaranteed exclusion of fallback bridge edges at the call site.

## Recommended Next Work

Useful follow-up tasks are:

- Update `documentation/uncertainty/README.md` or point readers to this file as
  the latest implementation status.
- Add unit tests for `evaluation/uncertainty.py` using small synthetic
  prediction records.
- Pass bridge/fallback node IDs into graph uncertainty once graph construction
  exposes them.
- Add optional feature-hook support for true penultimate-feature DDU.
- Add optional embedding-based Trust Score when face embeddings or detector
  features are consistently available.
- Add method-comparison plots or tables in the Web UI.
- Add AUROC/AUPR for misclassification detection if the team wants a
  threshold-independent metric.

## Quick Usage

From the CLI:

```text
python test_hierarchical.py --uncertainty-methods msp,ddu,trust_score,graph
```

From the Web UI:

1. Open the run configuration page.
2. Select one or more Uncertainty Methods.
3. Start the run.
4. After completion, open the run details page and inspect the Uncertainty
   Summary section.

Artifacts will be available under:

```text
run_outputs/<run_id>/<config_description>/uncertainty/
```
