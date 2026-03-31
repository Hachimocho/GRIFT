# `test_hierarchical.py` Architecture Guide

This document explains how `test_hierarchical.py` is organized, how it uses the
project modules, and how it interfaces with the Web UI (`web_ui`).

## What this file is in the system

`test_hierarchical.py` is the executable workload launched by the Web UI queue.

- `web_ui/app.py` receives "start run" requests.
- `web_ui/gpu_queue_manager.py` converts saved config into CLI flags.
- It launches `python test_hierarchical.py ...`.
- This script performs graph loading/building, training, validation, final test,
  and artifact generation.

In short: **UI schedules + monitors**, this script **does the ML work**.

## End-to-end flow

1. Parse CLI arguments (`parse_args` from `test_helpers.args_utils`).
2. Enable crash diagnostics / runtime logging.
3. Set reproducibility controls (`set_seed`, optional hashseed warning).
4. Load dataset node splits (train/val/test, full + balanced variants).
5. Build or load graph cache per split (`graph_cache/`).
6. Build graph managers for train/val/test.
7. Resolve traversal plan (single traversal or switching sequence).
8. For each test configuration:
   - Build model
   - Build trainer
   - Run epoch training + validation
   - Optionally perform graph reduction/restoration
   - Run final test inference
   - Generate plots/reports
9. Print final metrics JSON under:
   - `--- Final Test Results ---`

## Module types and how they are used

### 1) Helper modules (`test_helpers`)

- `args_utils.parse_args`
  - Defines/reads all CLI flags used by this script.
  - Contract point with Web UI config-to-CLI mapping.
- `logging_utils`
  - `set_seed`, output capture helpers, exception logging.
- `data_graph_utils`
  - Data split loading, balancing, cache utilities, threshold search helpers.

### 2) Data/graph domain modules

- Dataloaders:
  - `HierarchicalDeepfakeDataloader`
  - `UnclusteredDeepfakeDataloader`
- Graph primitives:
  - `HyperGraph`, `Edge`, node/data classes
- Purpose:
  - Construct graph objects for train/val/test splits, often from cache.

### 3) Manager modules

- `PerformanceGraphManager`
  - Wraps split graph for normal training/validation/testing usage.
- `NoGraphManager`
  - Alternative manager path when no graph traversal is needed.
- `GraphReductionManager`
  - Optional dynamic node reduction/restoration during training epochs.

### 4) Traversal modules

- `ComprehensiveTraversal`
- `RandomTraversal`
- `IValueTraversal`
- `IValueTraversalClusterHop`

Traversal mode controls how training samples/nodes are explored over graph steps.

### 5) Trainer modules

- `AdaptiveTrainer`
  - Core orchestrator used in this script.
  - Handles training and traversal switching logic.
  - Hosts one or more models and exposes training APIs consumed in epoch loop.

### 6) Model modules

- `CNNModel` for standard architecture runs.
- DQN models (`DQNModel`, enhanced variants) for I-value traversal behaviors.

## Key function map

- `_load_node_data(node, model)`
  - Load + transform one sample safely (parallel evaluation helper).
- `evaluate_model(...)`
  - Batched inference, accuracy/loss, optional bias metrics.
- `create_traversal(...)`
  - Traversal factory (comprehensive/random/i-value/cluster-hop).
- `parse_traversal_config(args)`
  - Normalizes and validates traversal-related args.
- `create_adaptive_trainer(...)`
  - Builds configured `AdaptiveTrainer`.
- `create_dqn_model(...)`
  - DQN variant factory.
- `create_model(...)`
  - Route to CNN vs DQN model path.
- `main()`
  - Full workflow orchestration.

## Frontend integration contract

The Web UI depends on these behaviors.

### 1) `--run-id`

Queue manager passes `--run-id`, and this script uses it to create:

- `run_outputs/<run_id>/...`

This keeps artifacts aligned with run metadata in `web_ui/runs/`.

### 2) Log output format

`GPUQueueManager` parses run logs to extract results. It specifically looks for
the "Final Test Results" block and bias-related markers.

If you change the text format around final metrics output, update
`GPUQueueManager._extract_results(...)` accordingly.

### 3) Cache directory and naming

This script reads/writes graph caches in project-level `graph_cache/`.
Cache filenames encode:

- dataset
- split
- graph type
- balancing mode
- thresholds
- node hash

UI cache status pages and compatibility checks assume this structure.

## Outputs written by this script

### Run-level outputs

- `run_outputs/<run_id>/<config_description>/...`
  - checkpoints
  - optional graph CSV exports
  - i-value visualizations
  - bias hop visualizations
  - bias metrics plots

### Log outputs

- stdout/stderr are captured by queue manager into:
  - `web_ui/runs/<run_id>.log`

### Metadata outputs (indirect)

- Queue manager updates:
  - `web_ui/runs/<run_id>.json`
  - includes status, timings, extracted results.

## Practical notes for future maintainers

1. Keep CLI arg names in sync across:
   - UI form fields
   - queue manager arg mapping
   - `parse_args` in helpers
2. Treat result-printing format as an API for the queue parser.
3. Prefer additive logging changes over replacing existing key markers.
4. If adding new traversal or reduction modes:
   - extend factory functions
   - document expected config keys
   - verify UI config mapping
5. Validate cache compatibility assumptions before changing cache naming logic.

