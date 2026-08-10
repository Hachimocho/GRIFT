# Testing and reproducibility

## Running the tests

```bash
pytest                      # fast tier: CPU only, synthetic, no dataset, no network (~20s)
pytest --run-slow           # + short real training / traversal runs
pytest --run-gpu            # + CUDA, real backbones, AMP, bit-exactness on device
pytest --run-data           # + the real AI-Face dataset on disk
pytest --run-network        # + tests that download pretrained weights
pytest --run-all            # everything
```

Tiers are gated by these options plus `pytest_collection_modifyitems`, **not** by
`-m` in `addopts`. That is deliberate: a user's own `-m gpu` would silently *replace*
an `addopts` marker filter rather than add to it, so the fast tier's exclusions would
disappear with no warning. Skip reasons are printed, so `pytest -v` tells you exactly
which tier a skipped test belongs to.

Use the environment that has both torch and pytest:

```bash
/home/brg2890/miniforge3/envs/Primary/bin/python -m pytest
```

## The determinism contract

`pytest_configure` re-execs pytest once with `PYTHONHASHSEED=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and single-threaded BLAS, so a bare `pytest`
invocation is sufficient for the strict-mode assertions. Set `GRIFT_NO_REEXEC=1` to
disable it — necessary under a debugger, which does not survive `execv`.

Every test runs under `configure_determinism(mode="strict")`, which forces
deterministic algorithms, cuDNN determinism with autotuning off, **TF32 off**, single
-threaded torch, and AMP disabled.

### What bit-exactness does and does not promise

Guaranteed, and asserted by `tests/determinism/test_bit_exact.py`:

- Two runs at the same seed on the **same machine and same device** produce identical
  per-step losses, identical final weights, identical predictions, and identical
  uncertainty scores. Verified for all four uncertainty heads, on CPU and on CUDA.

Not guaranteed:

- **Across devices.** CPU and CUDA run different kernels. They agree to ~1e-6 on a
  single forward pass with identical weights, but divergence compounds through
  training: each optimizer step feeds the previous step's rounding differences back
  in and AdamW's per-parameter normalization amplifies them. Measured at ~3e-2 in
  probability space after only four steps. There is no principled tolerance for a
  post-training cross-device comparison, so none is asserted.
- **Across GPU architectures, CUDA/cuDNN versions, or torch builds.**
  `run_fingerprint()` records all of these into `run_outputs/<run_id>/determinism.json`
  so a mismatch is diagnosable rather than mysterious.

### TF32 is the usual culprit

If a GPU bit-exactness check fails while the CPU one passes, suspect TF32 first. It is
enabled by default for cuDNN convolutions on Ada (the L40S), and TF32 is not fp32.
Strict mode disables it on both the matmul and cuDNN backends and sets
`float32_matmul_precision("highest")`.

### Per-component RNG streams

Every component draws from its own stream, derived as
`blake2b(component_name | master_seed)` — see `test_helpers/determinism.py`. This is
the property that makes runs robust rather than merely seeded: previously every
traversal, replay buffer, and balancing routine drew from the process-global `random`
module, so anything that consumed randomness upstream shifted every downstream
decision. Adding a log line that called `random.random()`, or changing the graph size
(which changes how many I-value fallback draws occur), silently changed which nodes a
traversal visited.

`blake2b` rather than `hash()` because `hash()` of a `str` is PYTHONHASHSEED-dependent,
which would make sub-seeds vary between processes.

### Training runs

```bash
python test_hierarchical.py --determinism strict --seed 42 ...   # bit-exact
python test_hierarchical.py --determinism fast --seed 42 ...     # default; faster
```

`strict` additionally requires `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` to be
set before interpreter start. Neither can be set usefully from inside a running
process, so `test_hierarchical.py` re-execs itself once with the right environment.
That works regardless of launcher, which matters because `run_reproducible.sh` cannot
guarantee it — `web_ui/gpu_queue_manager.py` invokes the script directly.

The re-exec is guarded on `__name__ == "__main__"`: `execv` replaces the whole
process, so an unguarded call would kill any host that merely imports the module.
`test_helpers/bootstrap.py` additionally refuses to exec when a test runner or REPL is
loaded.

## Writing tests

- **Prefer synthetic fixtures.** `tests/helpers/factories.py` builds nodes, edges, and
  graphs deterministically without RNG. `tests/helpers/images.py` writes tiny PNGs for
  anything needing a real `ImageFileData`.
- **Use `tiny_detector`, not a real one.** It injects a fake module into `sys.modules`
  so `CNNModel`'s `importlib.import_module` finds it. Importing any real detector pulls
  in all ten DeepfakeBench detectors plus `efficientnet_pytorch`, and most of them
  fetch weights from *unpinned* upstream GitHub branches. The default tier blocks
  `torch.hub` entirely; mark a test `network` if it genuinely needs a download.
- **Demographics must be `np.int64`.** That is what pandas produces, and `np.int64` is
  not a Python `int`. An `isinstance(value, (int, float))` check silently drops it — the
  bug that made graph-distance ignore gender, race, and age entirely. The factories
  store them correctly so this class of bug stays reproducible in a unit test.
- **Compare exactly, not with a tolerance,** when the claim is reproducibility. Use
  `assert_bit_exact` / `state_dict_hash` from `tests/helpers/determinism.py`. A
  tolerance cannot distinguish "reproducible" from "close enough today".
- **Assert function, not quality.** These tests check that things run, are shaped
  correctly, are finite, reduce a loss on a memorizable set, and reproduce. They
  deliberately assert nothing about accuracy, calibration, or fairness — that is the
  uncertainty benchmark's job, and a threshold here would either be vacuous or would
  fail for legitimate research reasons.
- **Pin surprising behavior you are not fixing.** Several tests document current
  behavior with an explanation rather than asserting the ideal: `add_edges_from_list`
  testing nodes for truthiness, `get_adjacent_nodes` counting duplicate edges, the
  three traversals whose `traverse()` returns `None`. That makes the next person's
  encounter with them a read rather than a debugging session.

## Autouse hygiene

Each test gets a temp cwd (the training code scatters `logs/`, `graph_cache/`,
`run_outputs/`), a fresh seeding, restored class attributes, headless matplotlib/Qt,
and blocked network access. The class-attribute restore is simultaneously hygiene and
a regression test: the dataloaders used to mutate their class-level `hyperparameters`
dict, so building a second loader retroactively changed the first one's settings.

Note the temp cwd interacts with `resolve_ai_face_data_root`, which probes
`$CWD/ai-face` among other candidates — `data`-tier tests should pass an explicit root
rather than rely on discovery.

## CI

`.github/workflows/tests.yml` runs the **fast tier only**, on all branches. It builds
from `environment.ci.yml`, a small hand-written CPU environment, rather than
sanitizing the 340-package CUDA-pinned `environment.yml` (whose `name:` and `prefix:`
point at a path that no longer exists). A second informational job,
`ci/check_env_drift.py`, reports where the two diverge so the divergence is visible
rather than discovered later.

The GPU, slow, and data tiers are not run in CI — they need hardware and a mounted
dataset. Run them locally before merging anything that touches training, evaluation,
or the uncertainty modules:

```bash
CUDA_VISIBLE_DEVICES=0 pytest --run-all
```

## Related

[`docs/uq_benchmark.md`](uq_benchmark.md) covers the other half of this work: how to
build a node cache, run with `--uq-records`, launch and aggregate deep ensembles, apply
the two shift protocols, and generate the report — plus how to read the resulting table
without misinterpreting it.

## Environment notes

- **`pytest` was absent from every environment that had torch.** It is now in
  `environment.yml`; if a fresh env lacks it, `mamba install -n <env> pytest
  pytest-cov pytest-timeout`.
- **`python-louvain` is not installed** in the training environment, so
  `HyperGraph.assign_louvain_subclusters` is currently a silent no-op and the
  `*_subclustered` traversals run on their no-subcluster fallback paths. Tests that
  need it use `pytest.importorskip("community")`.
- **No Parquet engine exists** (`pyarrow` and `fastparquet` are both absent), which is
  why benchmark record tables are `.csv.gz`.
- **opencv is a qt6 build**, so `QT_QPA_PLATFORM=offscreen` is required in headless
  contexts. `conftest.py` sets it.
