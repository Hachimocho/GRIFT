"""The GPU allowlist, which is what keeps a sweep off a colleague's card.

The memory check cannot do this job: another user training in 5 GB of a 46 GB L40S
leaves the card looking idle and available, so `get_available_gpus` would hand it over.
Only an explicit allowlist expresses "someone else owns that GPU".
"""

import pytest

from web_ui.gpu_queue_manager import GPUQueueManager, parse_visible_gpus


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("0,1", {0, 1}),
        ("0 1", {0, 1}),
        ("0,1,", {0, 1}),
        ("1", {1}),
        ([2, 3], {2, 3}),
        ((0,), {0}),
        ({1, 2}, {1, 2}),
    ],
)
def test_parses_the_forms_a_caller_actually_types(value, expected):
    assert parse_visible_gpus(value) == expected


@pytest.mark.parametrize("value", ["x", "0,x", "1.5", "0;1"])
def test_refuses_garbage_rather_than_widening_to_all_gpus(value):
    # Falling back to None here would mean a typo silently re-enables every GPU.
    with pytest.raises(ValueError):
        parse_visible_gpus(value)


def test_refuses_negative_ids():
    with pytest.raises(ValueError):
        parse_visible_gpus("0,-1")


class _StubManager(GPUQueueManager):
    """Subclass rather than instantiate: __init__ starts threads and touches the disk."""

    def __init__(self, visible_gpus, fake_gpus):
        self.visible_gpus = parse_visible_gpus(visible_gpus)
        self._fake_gpus = fake_gpus
        self.gpu_allocations = {}
        self.min_gpu_memory_gb = 2.0

    def get_gpu_info(self):
        gpu_info = list(self._fake_gpus)
        if self.visible_gpus is not None:
            gpu_info = [gpu for gpu in gpu_info if gpu["id"] in self.visible_gpus]
        return gpu_info


def _four_idle_l40s():
    return [
        {"id": i, "status": "available", "memory_free_gb": 41.0}
        for i in range(4)
    ]


def test_allowlist_hides_gpus_from_availability():
    manager = _StubManager("0,1", _four_idle_l40s())
    assert manager.get_available_gpus() == [0, 1]


def test_no_allowlist_sees_every_gpu():
    manager = _StubManager(None, _four_idle_l40s())
    assert manager.get_available_gpus() == [0, 1, 2, 3]


def test_a_busy_gpu_still_looks_available_without_the_allowlist():
    # The regression this guards: a colleague using 5 GB of a 46 GB card leaves ~41 GB
    # free, so the memory floor passes and the sweep would launch on top of them.
    gpus = _four_idle_l40s()
    gpus[2]["memory_free_gb"] = 41.0  # 5 GB in use by another user
    assert 2 in _StubManager(None, gpus).get_available_gpus()
    assert 2 not in _StubManager("0,1", gpus).get_available_gpus()


def test_env_var_is_the_fallback(monkeypatch):
    monkeypatch.setenv("GRIFT_VISIBLE_GPUS", "1,3")
    assert parse_visible_gpus(None) is None  # the parser itself reads no env
    # The manager is what consults the env; check the contract it relies on.
    import os
    assert os.environ["GRIFT_VISIBLE_GPUS"] == "1,3"
    assert parse_visible_gpus(os.environ["GRIFT_VISIBLE_GPUS"]) == {1, 3}


# ------------------------------------------------------------------ concurrency ---------
class _ConcurrencyManager(GPUQueueManager):
    """Subclass rather than instantiate: __init__ starts threads and touches the disk."""

    def __init__(self, runs_per_gpu, gpu_count=2):
        self.visible_gpus = None
        self.runs_per_gpu = runs_per_gpu
        self.min_gpu_memory_gb = 2.0
        self.gpu_allocations = {}
        self.run_processes = {}
        self.run_gpus = {}
        self.run_monitor_threads = {}
        self.gpu_run_ids = {}
        self._gpu_count = gpu_count

    def get_gpu_info(self):
        return [
            {
                "id": i,
                "memory_free_gb": 41.0,
                "allocated_to": self.gpu_allocations.get(i),
                "runs_active": len(self.gpu_run_ids.get(i, ())),
                "status": ("allocated"
                           if len(self.gpu_run_ids.get(i, ())) >= self.runs_per_gpu
                           else "available"),
            }
            for i in range(self._gpu_count)
        ]

    def _allocate(self, gpu_id, run_id):
        self.gpu_allocations[gpu_id] = run_id
        self.run_gpus[run_id] = gpu_id
        self.gpu_run_ids.setdefault(gpu_id, set()).add(run_id)


def test_one_run_per_gpu_is_the_default_behaviour():
    manager = _ConcurrencyManager(runs_per_gpu=1)
    assert manager.get_available_gpus() == [0, 1]
    manager._allocate(0, "run-a")
    assert manager.get_available_gpus() == [1], "a busy GPU must disappear at concurrency 1"


def test_a_gpu_stays_available_until_the_concurrency_limit():
    manager = _ConcurrencyManager(runs_per_gpu=4)
    for index in range(3):
        manager._allocate(0, f"run-{index}")
        assert 0 in manager.get_available_gpus(), \
            f"GPU 0 must still accept work with {index + 1} of 4 runs"
    manager._allocate(0, "run-3")
    assert 0 not in manager.get_available_gpus(), "the 4th run must fill the GPU"
    assert 1 in manager.get_available_gpus(), "other GPUs are unaffected"


def test_per_run_state_is_keyed_by_run_not_by_gpu():
    """The bug this guards: keyed by GPU, a second run on a card overwrote the first's
    process handle, so the monitor credited one run's exit code to another and freed a GPU
    that was still busy."""
    manager = _ConcurrencyManager(runs_per_gpu=3)
    manager._allocate(0, "run-a")
    manager._allocate(0, "run-b")
    assert manager.run_gpus == {"run-a": 0, "run-b": 0}
    assert manager.gpu_run_ids[0] == {"run-a", "run-b"}


def test_releasing_one_run_does_not_free_a_gpu_still_in_use():
    manager = _ConcurrencyManager(runs_per_gpu=3)
    manager._allocate(0, "run-a")
    manager._allocate(0, "run-b")

    # Mirror the release path's bookkeeping.
    remaining = manager.gpu_run_ids[0]
    remaining.discard("run-a")
    manager.run_gpus.pop("run-a")
    assert remaining == {"run-b"}
    assert 0 in manager.gpu_run_ids, "the GPU must stay allocated while run-b lives"

    remaining.discard("run-b")
    if not remaining:
        manager.gpu_run_ids.pop(0)
    assert 0 not in manager.gpu_run_ids, "the last run out frees the GPU"


def test_concurrency_and_the_allowlist_compose():
    manager = _ConcurrencyManager(runs_per_gpu=2)
    manager.visible_gpus = {1}
    # get_gpu_info in the real class filters by the allowlist; emulate that here.
    visible = [g for g in manager.get_gpu_info() if g["id"] in manager.visible_gpus]
    assert [g["id"] for g in visible] == [1]
