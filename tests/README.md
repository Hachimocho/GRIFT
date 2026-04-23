# Graph Unit Tests

This test suite validates graph generation in three layers:

1. dataset conversion
2. dataloader graph construction
3. graph average degree thresholds

## Files

- `tests/graph_test_support.py`
  Shared deterministic sample data, fake datasets, sample nodes, edge helpers, and graph-metric helpers.
- `tests/test_01_dataset.py`
  Verifies sample records are converted into valid `AttributeNode` objects.
- `tests/test_02_dataloaders.py`
  Verifies graph construction for the target dataloaders.
- `tests/test_03_degree_thresholds.py`
  Verifies average node degree stays inside expected hardcoded ranges.

## Install Dependencies

Use the test dependency file:

```bash
python3 -m pip install --user --break-system-packages -r requirements-test.txt
```

If your machine already has an activated virtual environment, use:

```bash
pip install -r requirements-test.txt
```

## Run The Full Suite

From the repo root:

```bash
python3 -m pytest
```

## Run One Layer At A Time

Dataset tests:

```bash
python3 -m pytest tests/test_01_dataset.py
```

Dataloader tests:

```bash
python3 -m pytest tests/test_02_dataloaders.py
```

Degree threshold tests:

```bash
python3 -m pytest tests/test_03_degree_thresholds.py
```

## Run One Specific Test

Example:

```bash
python3 -m pytest tests/test_02_dataloaders.py -k hierarchical
```

## What The Tests Check

### Dataset Layer

- sample rows become `AttributeNode` objects
- `node_id`, `split`, `label`, and threshold are preserved
- expected graph attributes are attached
- face embeddings are present and normalized

### Dataloader Layer

- 100 sample nodes are loaded into the graph path under test
- train graph contains edges
- validation and test graph behavior matches current loader behavior
- connected clustered loader includes the buffer node

### Degree Layer

- graph has edges
- average degree is computed from unique undirected edges
- average degree stays within expected hardcoded bounds

## GitHub Actions

CI is configured in:

`/.github/workflows/python-tests.yml`

GitHub Actions will:

1. check out the repo
2. set up Python
3. install `requirements-test.txt`
4. run `pytest`

## Supporting Documentation

- Implementation details and design rationale:
  [documentation/unit_test_implementation_guide.md](/home/hkb6416/GRIFT/documentation/unit_test_implementation_guide.md:1)
- Pytest configuration:
  [pytest.ini](/home/hkb6416/GRIFT/pytest.ini:1)
- Test dependencies:
  [requirements-test.txt](/home/hkb6416/GRIFT/requirements-test.txt:1)

## Which File To Start With

If you are new to this test suite, read the files in this order:

1. [README.md](/home/hkb6416/GRIFT/tests/README.md:1)
   Start here for the high-level picture and the commands to run the tests.
2. [graph_test_support.py](/home/hkb6416/GRIFT/tests/graph_test_support.py:1)
   This shows the deterministic sample records, sample nodes, fake dataset, and graph helpers that every test reuses.
3. [test_01_dataset.py](/home/hkb6416/GRIFT/tests/test_01_dataset.py:1)
   This is the first real test layer and the easiest place to learn how the sample data becomes nodes.
4. [test_02_dataloaders.py](/home/hkb6416/GRIFT/tests/test_02_dataloaders.py:1)
   Read this next to see how those nodes are passed into the graph-generation code.
5. [test_03_degree_thresholds.py](/home/hkb6416/GRIFT/tests/test_03_degree_thresholds.py:1)
   Read this last to see how graph metrics are checked after graph construction succeeds.
6. [unit_test_implementation_guide.md](/home/hkb6416/GRIFT/documentation/unit_test_implementation_guide.md:1)
   Use this after the test files if you want the full implementation summary and the list of repo changes made for the suite.
