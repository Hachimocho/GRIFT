# Uncertainty Quantification Plan

This document explains the uncertainty quantification work in simple terms.
It is meant to help future contributors understand what we are adding, why we
are adding it, and what parts of the project are allowed to change right now.

## Main Idea

The project already trains a deepfake detector on graph-organized face samples.
Uncertainty quantification, or UQ, asks a different question:

> How unsure is the detector about each test face?

For now, uncertainty is only a test-time analysis tool. It should not change
training, traversal, graph reduction, calibration, or the loss function yet.
The detector should train exactly as it did before. After training finishes and
the best checkpoint is loaded, selected uncertainty methods can score the final
test samples.

If the test-time results are useful, later work can decide whether uncertainty
should affect training or decisions.

## Current Rule From Advisor

Use or calculate uncertainty only during test time.

That means:

- Train the model normally.
- Load the best checkpoint normally.
- Run final test evaluation normally.
- Collect the extra values needed for uncertainty.
- Save uncertainty scores and summaries as run artifacts.

Do not use uncertainty during:

- model training
- validation-time model selection
- graph traversal
- graph reduction or restoration
- calibration during training
- bias loss computation

## Methods We Plan To Support

### MSP

MSP means Maximum Softmax Probability. In this binary detector, we can treat the
model confidence as the larger of:

- probability of real
- probability of fake

The uncertainty score is:

```text
uncertainty = 1 - confidence
```

This is the simplest baseline. It needs no extra training.

### DDU

DDU is a feature-density uncertainty method. The rough idea is:

1. Run training samples through the trained detector.
2. Collect intermediate features, usually the penultimate features.
3. Fit a class-conditional density model on those features.
4. At test time, faces in low-density regions are considered more uncertain.

This method needs features from the trained detector, but it still runs after
training. It does not require changing the detector architecture.

### Trust Score

Trust Score is an embedding-neighborhood method. The rough idea is:

1. Store reference embeddings or features for training samples.
2. For each test sample, find nearby training samples.
3. Compare distance to the predicted class against distance to the nearest other
   class.
4. Lower trust means higher uncertainty.

This is useful because it is model-agnostic and close in spirit to the graph
uncertainty method.

### Graph Uncertainty

Graph uncertainty uses the sample-similarity graph rather than the neural model
internals. The rough idea is:

- A node with many strong similarity edges is in a dense familiar region.
- A node with weak or few edges is in a sparse region.
- A node whose neighbors disagree may be harder to classify.

This method should use graph structure and similarity, not DQN I-values.

Important distinction:

- edge similarity is static and comes from graph construction
- I-value is learned by the DQN during training
- graph uncertainty is post-hoc and calculated at test time

## Important Graph Caveat

Some graph builders add fallback edges so disconnected nodes become connected.
Those fallback edges are not true similarity edges. For graph uncertainty, we
should either:

- ignore fallback edges, or
- skip uncertainty results for nodes that depend on fallback edges

Before implementing graph uncertainty, we need to confirm how edge weights or
similarity values are stored. At the time this document was created, the edge
objects mostly stored an edge label and default traversal weight. If similarity
is not stored on edges, graph uncertainty may need to recompute similarity from
node attributes or update edge creation to preserve similarity metadata.

## UI Behavior

The configuration page now has an Uncertainty Methods selector. It works like
the DQN model selector:

- users can select zero or more uncertainty methods
- selected methods are saved as a comma-separated config value
- the queue passes that value into `test_hierarchical.py`

The config key is:

```text
uncertainty_methods
```

The command-line flag is:

```text
--uncertainty-methods
```

Example:

```text
--uncertainty-methods msp,ddu,trust_score
```

## Planned Milestones

### Milestone 1: Config Selection

Status: done.

What changed:

- Added uncertainty method checkboxes to the config page.
- Added selected methods to the config preview.
- Added `--uncertainty-methods` to the CLI parser.
- Added queue-manager pass-through for the new CLI flag.

What this does not do yet:

- It does not calculate uncertainty.
- It does not save uncertainty artifacts.
- It does not change training or testing behavior.

### Milestone 2: Final Test Data Collection

Status: done.

Goal:

Extend final test evaluation so it can return the values UQ methods need.

Needed values:

- node id
- true label
- predicted label
- model logit
- model probability
- confidence
- optional intermediate feature vector
- optional face embedding from node attributes

This should still be test-time only.

What changed:

- Added parsing for `--uncertainty-methods`.
- Added optional final-test prediction record collection.
- Saved one CSV file with per-node final-test predictions.
- Saved one JSON summary file describing the selected methods and artifact paths.

What this does not do yet:

- It does not calculate MSP, DDU, Trust Score, or graph uncertainty scores.
- It does not collect train-reference features for Trust Score or DDU yet.
- It does not change predictions, training, validation, or checkpoint selection.

Current output files:

```text
run_outputs/<run_id>/<config_description>/uncertainty/summary.json
run_outputs/<run_id>/<config_description>/uncertainty/final_test_predictions.csv
```

The CSV currently includes:

- node id
- true label
- predicted label
- model logit
- probability fake
- probability real
- confidence
- correctness
- false-negative flag
- whether the node has a face embedding
- number of graph edges attached to the node

### Milestone 3: MSP Baseline

Goal:

Implement the simplest uncertainty baseline first.

Outputs:

- per-node MSP uncertainty scores
- summary metrics
- saved JSON or CSV artifact in the run output directory

### Milestone 4: Results Page Display

Goal:

Make uncertainty artifacts visible from the run details page.

The results page should show:

- which methods ran
- where the artifact files are
- basic summary values for each method

### Milestone 5: Trust Score

Goal:

Fit a train-reference nearest-neighbor scorer and apply it to test samples.

This needs train features or embeddings plus final test sample data.

### Milestone 6: DDU

Goal:

Fit density models on train features and score test features.

This needs reliable intermediate feature extraction from the detector.

### Milestone 7: Graph Uncertainty

Goal:

Use test graph structure to score node uncertainty from local density,
edge strength, sparsity, and neighbor disagreement.

Before coding this fully:

- confirm how similarity edges are built
- confirm whether edge weights are stored
- decide how to handle fallback connectivity edges

## Suggested Artifact Layout

Each run already writes outputs under:

```text
run_outputs/<run_id>/
```

Uncertainty artifacts should live under the specific config output directory:

```text
run_outputs/<run_id>/<config_description>/uncertainty/
```

Suggested files:

```text
summary.json
per_node_scores.csv
msp_scores.csv
trust_score_scores.csv
ddu_scores.csv
graph_scores.csv
```

The exact file names can change, but they should be stable enough for the web UI
to find and display.

## Metrics To Prefer For Now

The advisor asked us to avoid focusing on subgroup ECE calibration right now.
Calibration during training may come later.

For this phase, uncertainty should be judged by whether it helps identify risky
test samples, especially fake faces that might otherwise be missed.

Useful metrics:

- false negatives among high-uncertainty samples
- fake recall after treating uncertain samples as fake
- accuracy changes if uncertain samples are handled as fake
- AUROC/AUPR for misclassification detection, if used carefully

Important advisor note:

If uncertainty is used to change predictions later, uncertain faces should be
classified as fake rather than ignored, because missed fake faces are the main
risk.

## Commit Process

This work should be committed in small milestones.

The current environment can create local commits, but it may not be able to push
to GitHub because pushing requires credentials. If push fails, the local commit
should still be kept and the user can push from their own terminal.
