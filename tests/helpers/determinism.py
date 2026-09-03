"""Bit-exactness assertion helpers.

Everything here compares for *exact* equality, never ``allclose``. A tolerance-
based check cannot distinguish "reproducible" from "close enough today", which is
the whole thing these tests exist to pin down.
"""

import hashlib

import numpy as np
import torch


def state_dict_hash(state_dict):
    """A stable sha256 over a state dict's tensor bytes.

    Keys are sorted so insertion order cannot affect the digest. dtype and shape
    are folded in, so a silent dtype change is caught rather than hashed over.
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def model_hash(module):
    return state_dict_hash(module.state_dict())


def assert_bit_exact(left, right, path="value"):
    """Recursively assert exact equality across tensors, arrays, and containers."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), (
            f"{path}: tensor vs non-tensor ({type(left).__name__} vs {type(right).__name__})"
        )
        assert left.shape == right.shape, f"{path}: shape {left.shape} != {right.shape}"
        assert left.dtype == right.dtype, f"{path}: dtype {left.dtype} != {right.dtype}"
        if left.is_floating_point():
            # equal_nan semantics: NaN in the same positions counts as equal, so a
            # legitimately-NaN metric does not make the comparison useless.
            left_cpu, right_cpu = left.detach().cpu(), right.detach().cpu()
            both_nan = torch.isnan(left_cpu) & torch.isnan(right_cpu)
            assert torch.equal(left_cpu[~both_nan], right_cpu[~both_nan]), (
                f"{path}: tensors differ (max abs delta "
                f"{(left_cpu[~both_nan] - right_cpu[~both_nan]).abs().max().item() if (~both_nan).any() else 0})"
            )
        else:
            assert torch.equal(left.detach().cpu(), right.detach().cpu()), f"{path}: tensors differ"
        return

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert np.array_equal(left, right, equal_nan=np.issubdtype(np.asarray(left).dtype, np.floating)), (
            f"{path}: arrays differ"
        )
        return

    if isinstance(left, dict):
        assert isinstance(right, dict), f"{path}: dict vs {type(right).__name__}"
        assert sorted(left.keys()) == sorted(right.keys()), (
            f"{path}: keys differ ({sorted(left.keys())} vs {sorted(right.keys())})"
        )
        for key in sorted(left.keys()):
            assert_bit_exact(left[key], right[key], path=f"{path}[{key!r}]")
        return

    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)), f"{path}: sequence vs {type(right).__name__}"
        assert len(left) == len(right), f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_bit_exact(left_item, right_item, path=f"{path}[{index}]")
        return

    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return
        assert left == right, f"{path}: {left!r} != {right!r} (delta {left - right!r})"
        return

    assert left == right, f"{path}: {left!r} != {right!r}"


def run_twice(callable_, seed=0, mode="strict"):
    """Run a zero-arg callable twice, re-seeding identically before each.

    Returns both results so the caller can assert_bit_exact them.
    """
    from test_helpers.determinism import configure_determinism

    configure_determinism(seed=seed, mode=mode, allow_multi_gpu=True)
    first = callable_()
    configure_determinism(seed=seed, mode=mode, allow_multi_gpu=True)
    second = callable_()
    return first, second


def subprocess_run_py(code, env=None, python=None, timeout=180):
    """Run a snippet in a fresh interpreter with a controlled environment.

    The only way to test PYTHONHASHSEED behavior, which cannot be changed inside
    a running process.
    """
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_env = dict(os.environ)
    full_env.setdefault("PYTHONPATH", repo_root)
    if env:
        for key, value in env.items():
            if value is None:
                full_env.pop(key, None)
            else:
                full_env[key] = value

    return subprocess.run(
        [python or sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
        cwd=repo_root, env=full_env,
    )
