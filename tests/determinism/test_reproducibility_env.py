"""The determinism module and the startup bootstrap."""

import os

import pytest
import torch

from test_helpers.bootstrap import _scan_argv_for_seed, _scan_argv_for_strict, required_env
from test_helpers.determinism import (
    COMPONENTS, NonDeterministicEnvironmentError, assert_strict_invariants,
    configure_determinism, get_determinism_config, is_strict, numpy_rng_for, rng_for,
    run_fingerprint, seed_for, snapshot_rng_states, restore_rng_states, swallow_or_raise,
    torch_generator_for,
)


# --------------------------------------------------------------------------- #
# Environment pinning
# --------------------------------------------------------------------------- #

def test_test_session_runs_under_strict_mode():
    """Every test runs strict, so bit-exactness assertions mean something."""
    assert is_strict()
    assert get_determinism_config().mode == "strict"


def test_strict_flags_are_set_on_both_cpu_and_cuda_paths():
    """These used to sit inside `if torch.cuda.is_available()`.

    That meant a CPU-only run exercised an entirely different configuration than a
    GPU run -- which defeats the purpose of having CPU determinism tests.
    """
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.get_num_threads() == 1


def test_tf32_is_disabled_in_strict_mode():
    """TF32 is not fp32.

    On Ada (L40S) it is enabled by default for cuDNN convolutions, and it is the
    most likely reason a naive GPU bit-exactness check fails while CPU passes.
    """
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_strict_mode_rejects_an_unpinned_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(NonDeterministicEnvironmentError, match="PYTHONHASHSEED"):
        configure_determinism(seed=1, mode="strict", allow_multi_gpu=True)


def test_strict_mode_rejects_a_random_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(NonDeterministicEnvironmentError, match="PYTHONHASHSEED"):
        configure_determinism(seed=1, mode="strict", allow_multi_gpu=True)


def test_strict_mode_rejects_a_missing_cublas_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(NonDeterministicEnvironmentError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_determinism(seed=1, mode="strict", allow_multi_gpu=True)


def test_hash_seed_need_not_equal_the_master_seed():
    """Decoupled on purpose.

    Requiring PYTHONHASHSEED == seed would make the seed unchangeable within a
    process, which breaks deep ensembles (N seeds per process), threshold grid
    search, and any test that varies the seed. Reproducibility only needs the hash
    seed pinned and recorded.
    """
    assert os.environ["PYTHONHASHSEED"] == "0"
    config = configure_determinism(seed=99999, mode="strict", allow_multi_gpu=True)
    assert config.seed == 99999
    assert config.pythonhashseed == "0"


def test_fast_mode_relaxes_the_flags():
    configure_determinism(seed=0, mode="fast", allow_multi_gpu=True)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is True
    assert not is_strict()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="strict.*fast"):
        configure_determinism(seed=0, mode="turbo")


# --------------------------------------------------------------------------- #
# Invariant enforcement
# --------------------------------------------------------------------------- #

def test_assert_strict_invariants_catches_drift():
    """A run that silently falls out of strict mode must fail, not carry on."""
    configure_determinism(seed=0, mode="strict", allow_multi_gpu=True)
    assert_strict_invariants("baseline")  # clean

    torch.backends.cudnn.benchmark = True
    try:
        with pytest.raises(NonDeterministicEnvironmentError, match="benchmark"):
            assert_strict_invariants("after tampering")
    finally:
        torch.backends.cudnn.benchmark = False


def test_assert_strict_invariants_is_inert_in_fast_mode():
    configure_determinism(seed=0, mode="fast", allow_multi_gpu=True)
    assert_strict_invariants("fast mode")  # must not raise


def test_swallow_or_raise_reraises_in_strict_mode():
    """Swallowed exceptions are the largest silent divergence amplifier.

    If run 2 swallows something run 1 did not, bit-exactness breaks with no visible
    symptom -- which is exactly how the evidential/MC-dropout crash presented as
    'accuracy 0.0'.
    """
    configure_determinism(seed=0, mode="strict", allow_multi_gpu=True)
    with pytest.raises(RuntimeError, match="strict determinism"):
        swallow_or_raise(ValueError("boom"), context="unit test")


def test_swallow_or_raise_warns_in_fast_mode(capsys):
    configure_determinism(seed=0, mode="fast", allow_multi_gpu=True)
    swallow_or_raise(ValueError("boom"), context="unit test")
    assert "boom" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Per-component RNG streams
# --------------------------------------------------------------------------- #

def test_seed_for_is_deterministic_and_component_specific():
    assert seed_for("traversal.RandomTraversal") == seed_for("traversal.RandomTraversal")
    assert seed_for("traversal.RandomTraversal") != seed_for("dqn.replay")


def test_seed_for_depends_on_the_master_seed():
    configure_determinism(seed=1, mode="strict", allow_multi_gpu=True)
    first = seed_for("dqn.replay")
    configure_determinism(seed=2, mode="strict", allow_multi_gpu=True)
    assert seed_for("dqn.replay") != first


def test_seed_for_is_stable_across_processes(subprocess_run_py):
    """blake2b, not hash(): hash() of a str is PYTHONHASHSEED-dependent.

    Sub-seeds derived from hash() would vary between processes with different hash
    seeds, so a component's stream would not be reproducible.
    """
    code = (
        "from test_helpers.determinism import configure_determinism, seed_for;"
        "configure_determinism(7, 'fast');"
        "print(seed_for('traversal.IValueTraversal'))"
    )
    first = subprocess_run_py(code, env={"PYTHONHASHSEED": "1"})
    second = subprocess_run_py(code, env={"PYTHONHASHSEED": "12345"})
    assert first.returncode == 0 and second.returncode == 0, first.stderr + second.stderr
    assert first.stdout.strip() == second.stdout.strip()


def test_streams_are_independent():
    """Draining one component's stream must not shift another's.

    The point of the whole design: previously every traversal drew from the global
    `random` module, so RNG consumption anywhere upstream changed which nodes a
    traversal visited.
    """
    baseline = [rng_for("traversal.RandomTraversal").random() for _ in range(5)]
    noisy = rng_for("dqn.replay")
    for _ in range(1000):
        noisy.random()
    assert [rng_for("traversal.RandomTraversal").random() for _ in range(5)] == baseline


def test_global_rng_consumption_does_not_shift_component_streams():
    import random as global_random

    baseline = rng_for("balance.subgroup").random()
    for _ in range(500):
        global_random.random()
    assert rng_for("balance.subgroup").random() == baseline


def test_seed_for_rejects_a_non_string_component():
    with pytest.raises(ValueError):
        seed_for(None)


def test_numpy_and_torch_generators_are_reproducible():
    first = numpy_rng_for("viz.node_sample").random(4).tolist()
    second = numpy_rng_for("viz.node_sample").random(4).tolist()
    assert first == second

    generator_a = torch_generator_for("model.sngp_rff")
    generator_b = torch_generator_for("model.sngp_rff")
    assert torch.equal(
        torch.randn(4, generator=generator_a), torch.randn(4, generator=generator_b)
    )


def test_components_registry_covers_the_documented_streams():
    for component in (
        "traversal.RandomTraversal", "ivalue.fallback", "dqn.replay",
        "balance.subgroup", "model.batchensemble_init", "model.sngp_rff", "runid",
    ):
        assert component in COMPONENTS


# --------------------------------------------------------------------------- #
# RNG snapshots
# --------------------------------------------------------------------------- #

def test_rng_state_roundtrip():
    import random as global_random

    configure_determinism(seed=5, mode="strict", allow_multi_gpu=True)
    state = snapshot_rng_states()
    expected = [global_random.random() for _ in range(3)]
    expected_torch = torch.rand(3)

    restore_rng_states(state)
    assert [global_random.random() for _ in range(3)] == expected
    assert torch.equal(torch.rand(3), expected_torch)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #

def test_run_fingerprint_records_what_affects_reproducibility():
    fingerprint = run_fingerprint({"extra": 1})
    assert fingerprint["determinism"]["mode"] in ("strict", "fast")
    assert fingerprint["env"]["PYTHONHASHSEED"] == "0"
    assert fingerprint["versions"]["torch"] == torch.__version__
    assert "deterministic_algorithms" in fingerprint["torch_flags"]
    assert "commit" in fingerprint["git"]
    assert fingerprint["extra"] == 1


def test_run_fingerprint_is_json_serializable():
    import json
    json.dumps(run_fingerprint())


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "argv,expected",
    [
        (["prog"], 42),
        (["prog", "--seed", "7"], 7),
        (["prog", "--seed=13"], 13),
        (["prog", "--seed", "notanumber"], 42),
        (["prog", "--other", "1", "--seed", "99"], 99),
    ],
)
def test_scan_argv_for_seed(argv, expected):
    assert _scan_argv_for_seed(argv) == expected


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["prog"], False),
        (["prog", "--determinism", "strict"], True),
        (["prog", "--determinism", "fast"], False),
        (["prog", "--strict-determinism"], True),
    ],
)
def test_scan_argv_for_strict(argv, expected):
    assert _scan_argv_for_strict(argv) is expected


def test_required_env_pins_threads_only_in_strict():
    assert "OMP_NUM_THREADS" not in required_env(42, strict=False)
    assert required_env(42, strict=True)["OMP_NUM_THREADS"] == "1"
    assert required_env(42, strict=False)["PYTHONHASHSEED"] == "42"


def test_bootstrap_never_replaces_a_host_process():
    """execv would kill pytest.

    Calling the bootstrap at module import time did exactly that: importing
    test_hierarchical under pytest replaced the test runner mid-run, which showed up
    as the suite silently stopping partway through with a zero exit code.
    """
    from test_helpers.bootstrap import ensure_deterministic_env
    assert ensure_deterministic_env(["prog", "--seed", "31337"]) is True


def test_importing_the_entrypoint_does_not_reexec(subprocess_run_py):
    """Importing test_hierarchical must be side-effect-free w.r.t. the process."""
    result = subprocess_run_py(
        "import os; import test_hierarchical; print('imported', os.getpid())",
        env={"PYTHONHASHSEED": "0", "GRIFT_REEXEC": None, "CUBLAS_WORKSPACE_CONFIG": None},
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
    assert "re-exec" not in result.stderr


def test_running_the_entrypoint_does_reexec(subprocess_run_py, repo_root):
    """As a script it must pin the environment, whatever the launcher."""
    import subprocess
    import sys

    env = dict(os.environ)
    for key in ("PYTHONHASHSEED", "GRIFT_REEXEC", "CUBLAS_WORKSPACE_CONFIG"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, str(repo_root / "test_hierarchical.py"), "--help"],
        capture_output=True, text=True, timeout=180, cwd=str(repo_root), env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "re-exec with PYTHONHASHSEED=42" in result.stderr
