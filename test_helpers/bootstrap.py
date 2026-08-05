"""Early-startup environment fixup, imported before torch.

``PYTHONHASHSEED`` and ``CUBLAS_WORKSPACE_CONFIG`` cannot be set usefully from
inside a running interpreter: the former is consumed at startup, the latter when
the cuBLAS handle is created. The old code set ``os.environ['CUBLAS_WORKSPACE_CONFIG']``
inside ``set_seed()``, after torch was already imported, which is fragile at best.

A wrapper script cannot fix this either, because it is unenforceable:
``run_reproducible.sh`` exists but ``web_ui/gpu_queue_manager.py`` invokes
``test_hierarchical.py`` directly, and humans type ``python test_hierarchical.py``.
So instead the entrypoint re-executes itself once with the correct environment.
That works regardless of how the process was launched.

This module must import nothing beyond the stdlib -- it runs before torch.
"""

import os
import sys

REEXEC_SENTINEL = "GRIFT_REEXEC"
OPT_OUT = "GRIFT_NO_REEXEC"
DEFAULT_SEED = 42
DEFAULT_CUBLAS = ":4096:8"


def _scan_argv_for_seed(argv, default=DEFAULT_SEED):
    """Find --seed without argparse (which is not available this early)."""
    for index, token in enumerate(argv):
        if token == "--seed" and index + 1 < len(argv):
            candidate = argv[index + 1]
            if candidate.lstrip("-").isdigit():
                return int(candidate)
        elif token.startswith("--seed="):
            candidate = token.split("=", 1)[1]
            if candidate.lstrip("-").isdigit():
                return int(candidate)
    return default


def _scan_argv_for_strict(argv):
    for index, token in enumerate(argv):
        if token == "--determinism" and index + 1 < len(argv):
            return argv[index + 1] == "strict"
        if token == "--determinism=strict" or token == "--strict-determinism":
            return True
    return False


def required_env(seed, strict):
    env = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS),
    }
    if strict:
        # Single-threaded BLAS so CPU reduction order is fixed.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    return env


def ensure_deterministic_env(argv=None):
    """Re-exec this process with reproducibility env vars set, if needed.

    Returns True if the environment was already correct (no re-exec), and does
    not return at all if it re-execs. Set ``GRIFT_NO_REEXEC=1`` to disable --
    useful under a debugger, which does not survive ``execv``.
    """
    if os.environ.get(OPT_OUT) == "1":
        return True
    if os.environ.get(REEXEC_SENTINEL) == "1":
        return True

    argv = list(sys.argv if argv is None else argv)
    seed = _scan_argv_for_seed(argv)
    strict = _scan_argv_for_strict(argv)
    wanted = required_env(seed, strict)

    if all(os.environ.get(key) == value for key, value in wanted.items()):
        # Already correct; mark so a nested call is a no-op.
        os.environ[REEXEC_SENTINEL] = "1"
        return True

    os.environ.update(wanted)
    os.environ[REEXEC_SENTINEL] = "1"

    # sys.orig_argv preserves interpreter flags (-u, -X faulthandler) that the
    # GPU queue manager passes. Rebuilding from sys.argv would drop them.
    exec_argv = list(getattr(sys, "orig_argv", [sys.executable] + argv))
    sys.stderr.write(
        f"[determinism] re-exec with PYTHONHASHSEED={wanted['PYTHONHASHSEED']} "
        f"CUBLAS_WORKSPACE_CONFIG={wanted['CUBLAS_WORKSPACE_CONFIG']}"
        + (" OMP/MKL_NUM_THREADS=1" if strict else "")
        + "\n"
    )
    sys.stderr.flush()
    os.execv(sys.executable, exec_argv)
