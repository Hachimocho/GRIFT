#!/usr/bin/env python3
"""Run `test_hierarchical.main` in a child process with a tiny detector installed.

The tiny synthetic detectors live in `sys.modules`, so they do not survive a fork/exec.
A child process therefore has to install one before the runner resolves
`--architectures`, or validation rejects the name and the run exits 1.

Used only by `tests/functional/test_sweep_end_to_end.py`, which needs subprocesses
because strict determinism requires `CUDA_VISIBLE_DEVICES` to be set before torch
initializes CUDA -- exactly the condition `GPUQueueManager` arranges for a real cell.

    GRIFT_TEST_TINY_DETECTOR=grifttiny python tests/helpers/run_tiny.py <runner args...>
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def main():
    from tests.helpers.tiny_detector import (
        register_tiny_detector, register_tiny_detector_no_linear,
    )

    requested = os.environ.get("GRIFT_TEST_TINY_DETECTOR", "")
    if requested.endswith("nolinear"):
        installed = register_tiny_detector_no_linear()
    else:
        installed = register_tiny_detector()
    if requested and installed != requested:
        raise SystemExit(
            f"asked for tiny detector {requested!r} but installed {installed!r}"
        )

    import test_hierarchical

    return test_hierarchical.main(sys.argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
