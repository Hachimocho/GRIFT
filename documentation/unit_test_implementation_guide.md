# Unit Test Implementation Guide

This guide is the practical playbook for the graph-generation unit tests in this repo.

## Goal

Build a small, deterministic test stack that validates graph generation in three layers:

1. Dataset test: sample rows become nodes correctly.
2. Dataloader test: nodes become a graph correctly.
3. Degree test: the resulting graph has an average degree inside an expected range.

That order matters. If node creation is wrong, graph-building tests are much harder to trust.

## Why We Are Using Synthetic Test Data

The production dataset classes in this repo are large, file-heavy, and tied to external CSV/image layouts. That is useful for real runs, but it is a poor fit for unit tests because it makes failures slow and noisy.

For unit tests we want:

- deterministic inputs
- no dependency on external datasets
- fast runtime
- assertions that are easy to reason about

So the tests will use small synthetic samples that still match the attributes the graph code expects:

- `split`
- `label`
- `race_*`
- `gender_*`
- `age_*`
- quality metrics
- symmetry metrics
- `face_embedding`

## Test Layout

The unit-test stack will live in `tests/`:

- `tests/graph_test_support.py`
  Reusable helpers for sample records, fake datasets, sample nodes, and graph metrics.
- `tests/test_01_dataset.py`
  Dataset-level checks for converting sample data into nodes.
- `tests/test_02_dataloaders.py`
  Graph-construction checks using the sample nodes.
- `tests/test_03_degree_thresholds.py`
  Average-degree checks on the generated train graph.

The numbering is intentional. It matches the dependency chain and keeps local runs easy to follow.

## What Each Layer Must Prove

### 1. Dataset Tests

These tests should prove that a sample input row becomes a valid `AttributeNode`.

Minimum assertions:

- correct node count
- correct node type
- correct `node_id`
- correct `split`
- correct `label`
- expected attribute keys exist
- face embedding is preserved and has the expected length

These tests should not try to validate graph edges yet.

### 2. Dataloader Tests

These tests should prove that graph builders consume the nodes correctly.

Minimum assertions:

- exactly 100 input nodes are used
- resulting graph node count matches expectations
- train graph contains edges
- validation and test graph behavior matches the loader’s current expected output

For this repo:

- `UnclusteredDeepfakeDataloader` and `HierarchicalDeepfakeDataloader` accept `preloaded_nodes`, so they are straightforward to unit test.
- `ConnectedClusteredDeepfakeDataloader` expects dataset objects with `.load()`, so tests should use a tiny fake dataset object instead of production datasets.

### 3. Degree Tests

These tests should prove that graph density stays within an expected range.

The helper formula for an undirected graph is:

`average_degree = (2 * unique_edge_count) / node_count`

Minimum assertions:

- the graph is not empty
- the graph has at least one edge
- the average degree is between a hardcoded min and max threshold

Thresholds should be chosen from deterministic fixtures, not guessed randomly.

## Shared Test Helpers

The helper module should provide:

- sample-record builders
- sample-node builders
- a fake dataset class with `.load()`
- a lightweight edge class for tests
- graph metric helpers such as unique edge count and average degree

This avoids repeating setup code across test files.

## Rules For The Test Data

To keep the suite stable:

- no random test inputs unless the seed is fixed
- use exact split counts, such as 100 train nodes for graph tests
- keep attribute values intentionally similar inside groups so loaders produce predictable edges
- keep different groups intentionally distinct so tests can reason about connectivity

## Recommended Assertions

Use assertions that describe the behavior we care about.

Good:

- `assert len(nodes) == 6`
- `assert node.split == "train"`
- `assert "face_embedding" in node.attributes`
- `assert graph.num_edges() > 0`
- `assert 4.0 <= average_degree(graph) <= 20.0`

Weak:

- `assert graph is not None`
- `assert result`

## CI Plan

The repo currently does not have a root Python test workflow.

We will add:

- a root `.github/workflows/python-tests.yml`
- a small test dependency file if needed

The workflow should follow the standard GitHub Actions Python pattern:

1. check out the repo
2. set up Python
3. install dependencies
4. run `pytest`

## Practical Development Flow

When implementing or updating the tests, use this sequence:

1. update the shared helpers first
2. run dataset tests
3. run dataloader tests
4. run degree tests
5. run the full suite

This keeps failures localized and easier to debug.

## What To Watch Out For In This Repo

- Some loader paths are optimized for large real datasets, so unit tests should avoid those code paths when a lighter entry point exists.
- Some graph builders mutate `node.edges`, so tests should use fresh nodes per test instead of sharing one mutable graph fixture everywhere.
- Any randomness in graph connection logic should be seeded or avoided in the test fixtures.

## Definition Of Success

This unit-test work is successful when:

- the suite is fast enough to run during normal development
- failures clearly identify whether node conversion, graph construction, or graph density broke
- GitHub Actions runs the same test suite automatically on pushes and pull requests

## What Was Added

The following new files were added for this unit-test work:

- `tests/README.md`
  Runbook for installing dependencies and running the suite locally.
- `tests/graph_test_support.py`
  Deterministic sample-data builders, fake dataset class, edge helper, and graph metric helpers.
- `tests/test_01_dataset.py`
  Dataset-layer unit tests.
- `tests/test_02_dataloaders.py`
  Dataloader-layer unit tests.
- `tests/test_03_degree_thresholds.py`
  Degree-threshold unit tests.
- `pytest.ini`
  Pytest discovery config.
- `requirements-test.txt`
  Python dependency list for the test suite.
- `.github/workflows/python-tests.yml`
  GitHub Actions workflow for running the tests automatically.

## What Was Changed

The following existing files were updated to make the suite runnable and reliable:

- `dataloaders/ConnectedClusteredDeepfakeDataloader.py`
  Fixed the buffer node construction so it matches the shared `Node` constructor and can complete graph generation correctly.
- `dataloaders/HierarchicalDeepfakeDataloader.py`
  Fixed an invalid duplicate `get_graph` block so the module imports correctly and returns the expected cached graph.
- `data/__init__.py`
- `dataloaders/__init__.py`
- `utils/__init__.py`
  Updated package auto-import behavior so optional modules that depend on packages like `cv2` or `wandb` do not break test collection in environments where those packages are not installed.

## Exact Testing Flow Implemented

The implemented test flow is:

1. create deterministic sample records
2. convert records into `AttributeNode` objects
3. pass those nodes into dataloaders
4. build graphs
5. compute unique-edge count and average degree
6. assert degree thresholds

This is the exact order used by the test files:

1. `tests/test_01_dataset.py`
2. `tests/test_02_dataloaders.py`
3. `tests/test_03_degree_thresholds.py`

## Current Local Verification Result

The suite was locally verified with:

```bash
python3 -m pytest
```

Result at implementation time:

- `10 passed`
