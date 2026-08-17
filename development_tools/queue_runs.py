#!/usr/bin/env python3
"""Launch runs through the web UI's GPU queue and wait for them.

Shared by `launch_ensemble.py` and `sweep.py`, which both need the same three things:
one `GPUQueueManager`, a batch of configs queued in a readable order, and a blocking
wait until every run reaches a terminal status.

The one-manager rule is not stylistic. `GPUQueueManager.__init__` calls
`reconcile_existing_runs()`, which rewrites the metadata of any run currently marked
`running` or `queued` to `failed` on the assumption that it was orphaned by a server
restart. A second live instance would therefore declare the first instance's in-flight
runs dead. Construct exactly one, via `open_manager`.
"""

import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Statuses `GPUQueueManager` uses for a run that will not progress further.
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "error", "crashed"})


class QueueTimeout(RuntimeError):
    """Raised when runs are still in flight after the caller's deadline.

    Carries the run ids so a caller can tell the user how to resume rather than
    discarding the work that is still going.
    """

    def __init__(self, message, pending):
        super().__init__(message)
        self.pending = tuple(pending)


#: Repo root, so a caller invoked from a subdirectory still finds the one run registry.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The queue's run metadata and logs. `GPUQueueManager`'s own default is the *relative*
#: `web_ui/runs`, so a caller standing anywhere but the repo root would create a second
#: registry beside itself while the cells -- which the manager launches with
#: `cwd=<repo root>` -- wrote their artifacts to the first.
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "web_ui", "runs")


@contextlib.contextmanager
def open_manager(shutdown=True, runs_dir=None):
    """Yield the single `GPUQueueManager` for this process.

    `shutdown=False` leaves the queue's threads and child processes alive after the
    block exits, for callers that queue work and intend to let it finish after they
    return. Imported lazily: the manager pulls in psutil, GPUtil, and torch, which a
    caller doing offline scoring should not have to have installed.
    """
    from web_ui.gpu_queue_manager import GPUQueueManager

    manager = GPUQueueManager(runs_dir=runs_dir or DEFAULT_RUNS_DIR)
    try:
        yield manager
    finally:
        if shutdown:
            manager.shutdown()


def validate_configs(configs):
    """Raise if any config carries a key the queue would silently drop.

    `_build_command_args` ignores unmapped keys, so a typo or a newly added flag that
    was never routed does not fail -- it produces a run that quietly differs from the
    one requested, whose numbers look perfectly real. Checked before anything launches.
    """
    from web_ui.gpu_queue_manager import validate_config_keys

    problems = {}
    for name, config in configs:
        unroutable = validate_config_keys(config)
        if unroutable:
            problems[name] = unroutable
    if problems:
        lines = [f"  {name}: {', '.join(keys)}" for name, keys in sorted(problems.items())]
        raise ValueError(
            "config keys would be dropped on the way to test_hierarchical.py:\n"
            + "\n".join(lines)
            + "\n\nAdd them to ARG_MAPPING in web_ui/gpu_queue_manager.py, or remove them."
        )


def queue_all(manager, configs, verbose=True):
    """Queue `(config_name, config)` pairs. Returns the run ids, in order.

    Priority descends across the batch so runs start in the order given, which makes
    interleaved logs readable. It does not affect any result.
    """
    validate_configs(configs)
    total = len(configs)
    run_ids = []
    for index, (config_name, config) in enumerate(configs):
        run_id = manager.queue_run(
            config_name=config_name, config=config, priority=total - index,
        )
        run_ids.append(run_id)
        if verbose:
            print(f"  queued [{index + 1}/{total}] {config_name}: {run_id}")
    return run_ids


def wait_for(manager, run_ids, poll_seconds=30.0, timeout_hours=12.0, verbose=True):
    """Block until every run reaches a terminal status.

    Returns `{run_id: status}`. Raises `QueueTimeout` on the deadline -- the runs are
    still going at that point, so the caller should report how to resume rather than
    treating them as failed.
    """
    deadline = time.time() + timeout_hours * 3600.0
    reported = {}
    while True:
        statuses = {
            run_id: (manager.get_run(run_id) or {}).get("status", "unknown")
            for run_id in run_ids
        }

        if verbose:
            for run_id, status in statuses.items():
                if reported.get(run_id) != status:
                    print(f"  {run_id}: {status}")
                    reported[run_id] = status

        if all(status in TERMINAL_STATUSES for status in statuses.values()):
            return statuses

        if time.time() >= deadline:
            pending = [
                run_id for run_id, status in statuses.items()
                if status not in TERMINAL_STATUSES
            ]
            raise QueueTimeout(
                f"Timed out after {timeout_hours}h with {len(pending)} run(s) still in "
                f"flight. They have not been stopped.",
                pending,
            )

        # Sleep no longer than the time actually left, so the deadline is honored to
        # within one status check rather than one poll interval.
        time.sleep(min(poll_seconds, max(0.0, deadline - time.time())))
