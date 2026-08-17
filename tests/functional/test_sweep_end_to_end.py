"""The sweep driver end to end: run, score, promote, compare.

Runs the *real* training entrypoint for two configurations over the tiny synthetic node
cache and the tiny synthetic detector -- no dataset, no download, no GPU.

Cells run as subprocesses, built by the same `_build_command_args` table the queue uses,
but without constructing a `GPUQueueManager`: that would start background threads and
rewrite run metadata under `web_ui/runs/`, which a test must not touch. Subprocesses rather
than in-process calls for one specific reason -- strict determinism refuses to run with
more than one CUDA device visible, and `CUDA_VISIBLE_DEVICES` has to be set *before* torch
initializes CUDA. A queue-launched cell gets that from the launcher's child environment,
so doing the same here keeps the test on the real path.

The property worth the runtime: **an unchanged re-run must compare clean**. Same code,
same seed, strict determinism, same cache means the two runs score the same samples in the
same order, so every direction must be `same`, McNemar must be exactly p=1, and the exit
code must be 0. If that does not hold, every real comparison is measuring noise on top of
the change and no delta can be attributed.

Marked `slow`: real forward and backward passes over real image files.
"""

import json
import os

import pytest

from development_tools import sweep
from tests.helpers.node_cache import write_synthetic_node_cache

pytestmark = pytest.mark.slow

#: Two cells, the smallest matrix that still exercises axis expansion.
SUITE = {
    "description": "functional test suite",
    "reference": {
        "architectures": ["tinydetector"],
        "traversal_type": "random",
        "graph_type": "nonclustered",
        "uncertainty_head": "none",
        "graph_manager": "none",
        "seed": 7,
        "num_epochs": 1,
        "batch_size": 4,
        "cached_nodes": True,
        "cached_nodes_count": 40,
        "train_steps": 8,
        "val_steps": 8,
        "num_workers": 0,
        "val_num_workers": 0,
        "export_csv_per_run": False,
        "enable_val_bias_inference": False,
    },
    "axes": ["traversal"],
    "extra_axes": {"traversal": {"comprehensive": {"traversal_type": "comprehensive"}}},
}


@pytest.fixture
def sweep_workspace(tmp_path, monkeypatch, repo_root, tiny_detector):
    """A temp cwd with a node cache, a suite file, and the sweep dirs redirected.

    `tiny_detector` registers a `ModelOut`-shaped module in `sys.modules` plus a matching
    capability profile, so `--architectures` validation accepts it and `CNNModel`'s
    `importlib.import_module` finds it without pulling in a real backbone. The child
    processes re-register it themselves via `tests/helpers/run_tiny.py`.
    """
    monkeypatch.chdir(tmp_path)

    cache = write_synthetic_node_cache(
        tmp_path / "node_cache" / "cached_nodes.pkl",
        tmp_path / "cache_images",
        # The test split must exceed `compare.MIN_PAIRED_ROWS`, or McNemar refuses for
        # want of overlap and the "unchanged re-run is clean" check cannot be made.
        n_train=40, n_val=20, n_test=60, embedding_dim=16,
    )

    suite = json.loads(json.dumps(SUITE))
    suite["reference"]["architectures"] = [tiny_detector]
    suite["reference"]["cache_file"] = str(cache)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({"functional": suite}))

    # All three, together: the cells run with `cwd=tmp_path` here rather than at the repo
    # root, so the tree the sweep reads from has to be redirected alongside the tree it
    # writes to. `_resolve_artifact` anchors relative record paths to RUN_OUTPUTS_DIR's
    # parent, so redirecting that one constant moves artifact resolution with it.
    monkeypatch.setattr(sweep, "RUN_OUTPUTS_DIR", str(tmp_path / "run_outputs"))
    monkeypatch.setattr(sweep, "SWEEPS_DIR", str(tmp_path / "sweeps"))
    monkeypatch.setattr(sweep, "BASELINES_DIR", str(tmp_path / "benchmarks"))
    return {
        "root": tmp_path,
        "suite_file": str(suite_path),
        "cache": str(cache),
        "runner_kwargs": {
            "repo_root": repo_root,
            "cwd": tmp_path,
            "tiny_detector_name": tiny_detector,
        },
    }


def run_cells(cells, sweep_id, manifest_path, repo_root, cwd, tiny_detector_name):
    """Run each cell as a subprocess, recording status into the manifest.

    The argv comes from `GPUQueueManager._build_command_args` on an *uninitialized*
    instance -- so the flag translation under test is the production one, with no queue
    threads and no writes to `web_ui/runs/`.
    """
    import subprocess

    from web_ui.gpu_queue_manager import GPUQueueManager

    builder = GPUQueueManager.__new__(GPUQueueManager)
    manifest = json.load(open(manifest_path))

    environment = dict(os.environ)
    environment.update({
        # Before torch initializes CUDA, which is why this cannot be done in-process.
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": str(cells[0].config.get("seed", 0)),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(repo_root),
        # The tiny detector lives in sys.modules, so a child process has to install it
        # again before the runner resolves --architectures.
        "GRIFT_TEST_TINY_DETECTOR": tiny_detector_name,
    })

    for index, cell in enumerate(cells):
        run_id = f"{sweep_id}_r{index}"
        argv = builder._build_command_args(cell.config, run_id=run_id)
        # argv[:5] is `python -u -X faulthandler test_hierarchical.py`; replace the script
        # path with a shim that installs the tiny detector first.
        command = [argv[0], "-u", str(repo_root / "tests" / "helpers" / "run_tiny.py")]
        command += argv[5:]
        completed = subprocess.run(
            command, cwd=str(cwd), env=environment,
            capture_output=True, text=True, timeout=900,
        )
        assert completed.returncode == 0, (
            f"{cell.cell_id} failed (exit {completed.returncode})\n"
            f"--- stdout tail ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr tail ---\n{completed.stderr[-4000:]}"
        )
        manifest["cells"][cell.cell_id].update({
            "run_id": run_id, "status": "completed", "duration_seconds": 1.0,
        })

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def prepare_sweep(workspace, sweep_id, overrides=None):
    """Expand the matrix and write the manifest the scorer reads.

    `overrides` merge into the reference cell before expansion, so a test can vary one
    setting and have both the launched argv and the recorded manifest agree about it.
    """
    from development_tools.sweep_suites import expand, load_suite

    suite = load_suite("functional", suite_file=workspace["suite_file"])
    suite["reference"].update(overrides or {})
    cells = expand(suite)
    assert all(cell.runnable for cell in cells), [
        (cell.cell_id, cell.skip_reason) for cell in cells if not cell.runnable
    ]

    sweep_dir = os.path.join(sweep.SWEEPS_DIR, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    manifest = {
        "sweep_id": sweep_id, "suite": "functional", "tag": None,
        "determinism": "strict", "seed": 7,
        "git": {"commit": "0" * 40, "dirty": False},
        "node_cache": workspace["cache"], "node_cache_sha256": "unchanged",
        "cells": {
            cell.cell_id: {**cell.to_dict(), "status": "pending"} for cell in cells
        },
    }
    manifest_path = os.path.join(sweep_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return cells, sweep_dir, manifest_path


def score(workspace, sweep_id):
    """`sweep.py run --score-only` over an already-run sweep."""
    return sweep.main([
        "run", "--suite", "functional", "--suite-file", workspace["suite_file"],
        "--sweep-id", sweep_id, "--score-only",
    ])


def test_run_score_promote_compare(sweep_workspace):
    """The whole loop, and the invariant that an unchanged re-run compares clean."""
    cells, sweep_dir, manifest_path = prepare_sweep(sweep_workspace, "sweep_base")
    run_cells(cells, "sweep_base", manifest_path, **sweep_workspace["runner_kwargs"])

    assert score(sweep_workspace, "sweep_base") == 0

    results_path = os.path.join(sweep_dir, "results.csv")
    assert os.path.exists(results_path)

    import pandas as pd

    results = pd.read_csv(results_path)
    assert not results.empty
    # Whole-set rows and demographic slices both present: the subgroup breakdown is the
    # thing that did not exist before, so its absence must fail here.
    assert "overall" in set(results["subgroup_dimension"])
    assert len(set(results["subgroup_dimension"])) > 1
    assert set(results["cell_id"]) >= {cell.cell_id for cell in cells}
    # Every cell contributed at least one scored row.
    for cell in cells:
        scored = results[(results["cell_id"] == cell.cell_id) & (results["status"] == "ok")]
        assert not scored.empty, f"{cell.cell_id} produced no scored row"

    assert sweep.main(["promote", "sweep_base"]) == 0
    baseline_path = os.path.join(sweep.BASELINES_DIR, "baseline_functional.csv")
    assert os.path.exists(baseline_path)
    assert os.path.exists(
        os.path.join(sweep.BASELINES_DIR, "baseline_functional.manifest.json")
    )

    # The same code, again. Every direction must be `same`.
    cells2, sweep_dir2, manifest_path2 = prepare_sweep(sweep_workspace, "sweep_again")
    run_cells(cells2, "sweep_again", manifest_path2, **sweep_workspace["runner_kwargs"])
    assert score(sweep_workspace, "sweep_again") == 0

    report_dir = os.path.join(sweep_workspace["root"], "cmp")
    exit_code = sweep.main([
        "compare", "--baseline", "baseline:functional",
        "--candidate", os.path.join(sweep_dir2, "results.csv"),
        "--out", report_dir,
    ])

    comparison = pd.read_csv(os.path.join(report_dir, "comparison.csv"))
    directions = set(comparison["direction"])
    assert directions <= {"same", "n_a"}, (
        f"an unchanged re-run moved: {comparison[~comparison['direction'].isin(['same', 'n_a'])]}"
    )
    assert (comparison["presence"] == "both").all()
    assert not comparison["newly_degenerate"].any()
    assert exit_code == 0

    text = open(os.path.join(report_dir, "comparison.md")).read()
    assert "No hard failures and no regressions" in text


def test_a_real_change_is_detected_and_attributed(sweep_workspace):
    """A different seed is a real difference; the diff must find it.

    Seed rather than epoch count: with a tiny detector on forty nodes, a second epoch
    often fails to beat the first epoch's validation accuracy, so the best checkpoint --
    and therefore every reported number -- is unchanged. That is correct behavior, and it
    makes epoch count a poor probe. A different seed changes weight initialization and
    traversal order, so the reported numbers must move.
    """
    cells, sweep_dir, manifest_path = prepare_sweep(sweep_workspace, "sweep_seed7")
    run_cells(cells, "sweep_seed7", manifest_path, **sweep_workspace["runner_kwargs"])
    assert score(sweep_workspace, "sweep_seed7") == 0
    assert sweep.main(["promote", "sweep_seed7"]) == 0

    cells2, sweep_dir2, manifest_path2 = prepare_sweep(
        sweep_workspace, "sweep_seed11", overrides={"seed": 11}
    )
    run_cells(cells2, "sweep_seed11", manifest_path2, **sweep_workspace["runner_kwargs"])
    assert score(sweep_workspace, "sweep_seed11") == 0

    import pandas as pd

    report_dir = os.path.join(sweep_workspace["root"], "cmp_changed")
    sweep.main([
        "compare", "--baseline", "baseline:functional",
        "--candidate", os.path.join(sweep_dir2, "results.csv"),
        "--out", report_dir,
    ])
    comparison = pd.read_csv(os.path.join(report_dir, "comparison.csv"))
    # Something must have moved -- which way is a quality question this suite does not
    # assert, per docs/testing.md.
    assert set(comparison["direction"]) & {"better", "worse"}

    # And the report must say the seed changed, since that makes the delta unattributable
    # to a code change.
    text = open(os.path.join(report_dir, "comparison.md")).read()
    assert "seed differs" in text


def test_resume_skips_completed_cells(sweep_workspace, capsys):
    cells, sweep_dir, manifest_path = prepare_sweep(sweep_workspace, "sweep_resume")
    run_cells(cells, "sweep_resume", manifest_path, **sweep_workspace["runner_kwargs"])
    assert score(sweep_workspace, "sweep_resume") == 0

    from development_tools.sweep import _pending_cells

    manifest = json.load(open(manifest_path))
    assert _pending_cells(cells, manifest["cells"]) == []
    # --force re-runs them.
    assert len(_pending_cells(cells, manifest["cells"], force=True)) == len(cells)


def test_a_cell_that_wrote_no_records_is_a_hard_failure(sweep_workspace):
    """A completed run with no record table must not pass silently."""
    cells, sweep_dir, manifest_path = prepare_sweep(sweep_workspace, "sweep_partial")
    run_cells(cells, "sweep_partial", manifest_path, **sweep_workspace["runner_kwargs"])

    # Break one cell's artifact discovery the way a crashed configuration would.
    manifest = json.load(open(manifest_path))
    manifest["cells"][cells[1].cell_id]["run_id"] = "run_that_never_existed"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    assert score(sweep_workspace, "sweep_partial") == sweep.EXIT_HARD_FAILURE


def test_promote_refuses_an_incomplete_sweep(sweep_workspace):
    """Promoting a partial baseline makes every later diff report false regressions."""
    cells, sweep_dir, manifest_path = prepare_sweep(sweep_workspace, "sweep_incomplete")
    run_cells(cells, "sweep_incomplete", manifest_path, **sweep_workspace["runner_kwargs"])
    assert score(sweep_workspace, "sweep_incomplete") == 0

    manifest = json.load(open(manifest_path))
    manifest["cells"][cells[1].cell_id]["status"] = "failed"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    assert sweep.main(["promote", "sweep_incomplete"]) == sweep.EXIT_USAGE
    assert sweep.main(["promote", "sweep_incomplete", "--allow-partial"]) == 0
