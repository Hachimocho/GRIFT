Testing Guide

Overview
- This repository uses pytest for unit tests.
- Tests live under the `tests/` directory and are organized by feature: nodes, edges, graphs, traversals, data/managers, and dataloaders.

Environment
- One Conda environment file (`environment.yml`) supports both development and testing.
- CI installs the same environment and runs tests automatically.

Running Tests Locally
1) Create/activate the conda env:
   - conda/mamba env create -f environment.yml
   - conda activate Primary (or your env name)
2) Install pytest if missing in your local env (CI installs it explicitly):
   - mamba install -c conda-forge pytest
3) Run:
   - pytest -q

Test Design
- Generic class tests:
  - Nodes: verifies equality/hash, data accessors, adjacency via edges.
  - AttributeNode: validates attribute similarity and match behavior.
  - Edges: getters/setters and traversal weight.
  - HyperGraph: node CRUD, edge extraction, and edge list roundtrip.
- Traversals:
  - RandomTraversal: collects nodes from a simple line graph.
  - ComprehensiveTraversal: visits all nodes deterministically in test mode.
  - IValueTraversal: exercises selection with a dummy trainer; returns AttributeNodes.
- Data and Managers:
  - Data, ImageData, ImageFileData: small image IO roundtrip; NoGraphManager basics.
- Dataloaders:
  - UnclusteredDeepfakeDataloader and HierarchicalDeepfakeDataloader: build tiny graphs with fabricated AttributeNodes and very permissive thresholds.

Creating New Tests
- Prefer small, deterministic dummy inputs (e.g., tiny graphs, 2×2 images) and avoid network/dataset IO.
- Keep tests independent; use pytest fixtures and `tmp_path` for files.
- Add new tests as `tests/test_<area>.py` and follow existing style.

CI (GitHub Actions)
- Workflow at `.github/workflows/tests.yml`:
  - Creates the environment from `environment.yml` (via mamba).
  - Installs pytest.
  - Runs `pytest -q` on Ubuntu.

What You Might Need To Do
- Ensure `environment.yml` resolves on GitHub runners. If your local environment name/path is embedded, it’s fine; CI overrides the environment name.
- If tests need additional packages not in `environment.yml`, add them there (preferred) to keep a single source of truth.

Troubleshooting
- Qt/Matplotlib issues in headless CI: we set `QT_QPA_PLATFORM=offscreen` in CI. If you add new plotting code in tests, avoid showing windows or rely on non-interactive backends only.
- GPU-related code: tests avoid CUDA. Do not require GPU to run; guard such paths.

