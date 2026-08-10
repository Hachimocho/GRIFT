"""Single source of truth for seeding and reproducibility in GRIFT.

Two modes:

``fast``   perf-oriented. cuDNN autotuning on, TF32 on, AMP allowed, threads
           untouched. Same-seed runs agree closely but not bit-exactly on GPU.
``strict`` bit-exact everywhere. Deterministic algorithms, TF32 off, single
           thread, AMP off, ordered result collection, one visible CUDA device.

Sub-seeds are *content-addressed*: every component derives its own stream from
``blake2b(component | master_seed)``. This is the important property. Before
this module, every traversal, DQN replay buffer, and balancing routine drew from
the process-global ``random`` module, so RNG consumption anywhere upstream
shifted every downstream decision -- adding a log line that happened to call
``random.random()`` would change which nodes a traversal visited. With derived
streams, components are independent by construction.

``blake2b`` rather than ``hash()`` because ``hash()`` of a str is
PYTHONHASHSEED-dependent, which would make sub-seeds vary across processes.
"""

import hashlib
import json
import os
import random
import sys

import numpy as np
import torch


class NonDeterministicEnvironmentError(RuntimeError):
    """Raised when strict mode is requested but the environment cannot honor it."""


# Environment variables that must be set before interpreter start (see bootstrap.py).
REQUIRED_STRICT_ENV = {
    "PYTHONHASHSEED": None,          # must equal str(seed); filled in at check time
    "CUBLAS_WORKSPACE_CONFIG": (":4096:8", ":16:8"),
}

# Every component that owns an RNG stream. Enumerated so a test can assert that
# no call site was missed, and so `seed_for` typos fail loudly instead of
# silently minting a fresh, plausible-looking stream.
COMPONENTS = frozenset({
    # graph construction
    "graph.isolated_node_fallback",
    "graph.louvain",
    "graph.random_node",
    "graph.display_sample",
    # traversals (per-class streams are derived as "traversal.<ClassName>")
    "traversal.RandomTraversal",
    "traversal.RandomWarpTraversal",
    "traversal.RandomNoReturnTraversal",
    "traversal.RandomNoReturnWarpTraversal",
    "traversal.ComprehensiveTraversal",
    "traversal.IValueTraversal",
    "traversal.IValueTraversalSubcluster",
    "traversal.IValueTraversalClusterHop",
    "traversal.IValueTraversalClusterHopSubcluster",
    # i-value / DQN
    "ivalue.fallback",
    "dqn.replay",
    "dqn.ivalue_predictor",
    # data
    "balance.subgroup",
    "dataset.split_shuffle",
    "dataset.debug_sample",
    "cache.node_subsample",
    # graph mutation
    "reduction.remove",
    "reduction.restore",
    # visualization
    "viz.node_sample",
    "viz.graph_layout",
    # evaluation
    "eval.val_subsample",
    "eval.train_bias_subsample",
    # model init (ensemble members derive as "model.init.member<n>")
    "model.batchensemble_init",
    "model.sngp_rff",
    # misc
    "runid",
})

_STATE = {"config": None}


class DeterminismConfig:
    """Resolved determinism settings for the current process."""

    __slots__ = ("seed", "mode", "pythonhashseed", "cublas_workspace_config", "amp_enabled")

    def __init__(self, seed, mode, pythonhashseed, cublas_workspace_config, amp_enabled):
        self.seed = seed
        self.mode = mode
        self.pythonhashseed = pythonhashseed
        self.cublas_workspace_config = cublas_workspace_config
        self.amp_enabled = amp_enabled

    @property
    def strict(self):
        return self.mode == "strict"

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return f"DeterminismConfig(seed={self.seed}, mode={self.mode!r})"


def configure_determinism(seed=42, mode="fast", allow_multi_gpu=False):
    """Seed every RNG and set the torch flags implied by ``mode``.

    Call once, as early as possible. Returns the resolved DeterminismConfig.
    """
    if mode not in ("strict", "fast"):
        raise ValueError(f"determinism mode must be 'strict' or 'fast', got {mode!r}")

    seed = int(seed)

    if mode == "strict":
        _assert_strict_env(seed, allow_multi_gpu=allow_multi_gpu)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These are NOT inside a cuda-availability guard. The old set_seed() put all
    # of them inside `if torch.cuda.is_available()`, so CPU-only runs silently
    # exercised a different code path than GPU runs -- which defeats the point of
    # having CPU determinism tests at all.
    if mode == "strict":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # TF32 is the most likely reason a naive GPU bit-exactness check fails
        # while CPU passes: on Ada (L40S) it is enabled by default for cuDNN
        # convolutions, and TF32 is not fp32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Only settable before the interop pool is initialized; harmless.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    config = DeterminismConfig(
        seed=seed,
        mode=mode,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        amp_enabled=(mode == "fast"),
    )
    _STATE["config"] = config
    return config


def seed_model_init(member=None):
    """Reseed the torch RNGs so a model's weights depend on ``member``.

    Deep ensembles need members that differ in *initialization* only. Varying
    ``--seed`` would do that, but the graph cache key embeds the seed whenever a
    split has edges, so N seeds means N full graph rebuilds -- the expensive part of
    a run, repeated for no experimental gain. It would also vary the training data
    order, which confounds "ensemble diversity" with "trained on a different
    curriculum".

    So the seed stays fixed and this shifts only the weight-init stream. Call it
    immediately before constructing the model.

    ``member=None`` is the single-run case and is a **no-op** -- reseeding from a
    derived stream would change every existing non-ensemble run's weights, which
    would silently invalidate the reference numbers already recorded.

    Returns the seed used, or None when nothing was reseeded.
    """
    if member is None:
        return None
    seed = seed_for(f"model.init.member{int(member)}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def _assert_strict_env(seed, allow_multi_gpu=False):
    problems = []

    # PYTHONHASHSEED must be *pinned to a known value*, not necessarily equal to
    # the master seed. Requiring equality would make the seed unchangeable within
    # a process, which breaks deep ensembles (N seeds per process), threshold grid
    # search, and any test that varies the seed. Reproducibility only needs the
    # hash seed to be fixed and recorded, which run_fingerprint() does.
    actual_hashseed = os.environ.get("PYTHONHASHSEED")
    if actual_hashseed is None or actual_hashseed == "random" or not actual_hashseed.isdigit():
        problems.append(
            f"PYTHONHASHSEED={actual_hashseed!r} is not pinned to a fixed integer. "
            "It cannot be set from inside Python after startup -- run via "
            "test_helpers.bootstrap.ensure_deterministic_env() or run_reproducible.sh."
        )

    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in REQUIRED_STRICT_ENV["CUBLAS_WORKSPACE_CONFIG"]:
        problems.append(
            f"CUBLAS_WORKSPACE_CONFIG={cublas!r}, expected one of "
            f"{REQUIRED_STRICT_ENV['CUBLAS_WORKSPACE_CONFIG']}. Deterministic cuBLAS "
            "GEMM requires it, and it is read when the cuBLAS handle is created."
        )

    if not allow_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        problems.append(
            f"{torch.cuda.device_count()} CUDA devices visible. Strict mode requires exactly "
            "one -- set CUDA_VISIBLE_DEVICES, or pass allow_multi_gpu=True if you accept "
            "that allocation order across devices is not controlled."
        )

    if problems:
        raise NonDeterministicEnvironmentError(
            "Cannot honor --determinism strict:\n  - " + "\n  - ".join(problems)
        )


def get_determinism_config():
    config = _STATE["config"]
    if config is None:
        raise RuntimeError(
            "configure_determinism() has not been called. Seeding must be configured "
            "before any RNG-consuming work so runs are reproducible."
        )
    return config


def is_strict():
    config = _STATE["config"]
    return config is not None and config.strict


def is_configured():
    return _STATE["config"] is not None


def amp_enabled(device=None):
    """Whether AMP autocast should be active.

    Strict mode disables AMP for three reasons: fp16 Cholesky/pinv on the SNGP
    precision matrix is numerically unsound; GradScaler's scale factor is
    dynamic, path-dependent state that is not checkpointed, so one non-finite
    gradient permanently diverges two otherwise-identical runs; and autocast's
    per-op dtype choices interact badly with deterministic algorithms.
    """
    config = _STATE["config"]
    if config is not None and config.strict:
        return False
    if device is not None and getattr(device, "type", str(device)) != "cuda":
        return False
    return torch.cuda.is_available()


def seed_for(component):
    """Derive a stable sub-seed for ``component`` from the master seed.

    Stable across processes and independent of PYTHONHASHSEED, so a component's
    stream depends only on (component name, master seed) -- never on how much
    randomness anything else happened to consume first.
    """
    if not isinstance(component, str) or not component:
        raise ValueError(f"component must be a non-empty str, got {component!r}")
    master = get_determinism_config().seed
    digest = hashlib.blake2b(
        f"{component}|{master}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2 ** 31 - 1)


def rng_for(component):
    """A private ``random.Random`` for ``component``."""
    return random.Random(seed_for(component))


def component_rng(component, fallback_seed=0):
    """``rng_for(component)``, but usable before determinism is configured.

    Library modules are constructed in contexts that may not have called
    ``configure_determinism`` (bare unit tests, notebooks, tooling). Returning a
    fixed-seed private stream there is still strictly better than drawing from the
    process-global ``random`` module, which is what these call sites did before.

    Cached per component so repeated calls in a loop keep advancing one stream
    rather than restarting it.
    """
    cache = _STATE.setdefault("component_rngs", {})
    config = _STATE["config"]
    generation = (id(config), component)
    cached = cache.get(component)
    if cached is not None and cached[0] == generation:
        return cached[1]

    stream = rng_for(component) if config is not None else random.Random(fallback_seed)
    cache[component] = (generation, stream)
    return stream


def numpy_rng_for(component):
    """A private numpy Generator for ``component``."""
    return np.random.Generator(np.random.PCG64(seed_for(component)))


def torch_generator_for(component, device="cpu"):
    """A private ``torch.Generator`` for ``component``."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_for(component))
    return generator


def snapshot_rng_states():
    """Capture every RNG state, for checkpointing mid-run."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])


def _as_byte_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(bytearray(value), dtype=torch.uint8)


def assert_strict_invariants(context=""):
    """Re-verify every strict-mode knob. No-op in fast mode.

    Called at several points during a run (after configure, after model
    creation, at each epoch start, before final eval) so that a run which
    silently fell out of strict mode -- because some library flipped a backend
    flag -- fails instead of quietly producing unreproducible numbers.
    """
    config = _STATE["config"]
    if config is None or not config.strict:
        return

    drift = []
    if not torch.are_deterministic_algorithms_enabled():
        drift.append("torch.use_deterministic_algorithms is off")
    if not torch.backends.cudnn.deterministic:
        drift.append("cudnn.deterministic is False")
    if torch.backends.cudnn.benchmark:
        drift.append("cudnn.benchmark is True")
    if torch.backends.cuda.matmul.allow_tf32:
        drift.append("cuda.matmul.allow_tf32 is True")
    if torch.backends.cudnn.allow_tf32:
        drift.append("cudnn.allow_tf32 is True")
    if torch.get_num_threads() != 1:
        drift.append(f"torch.get_num_threads()=={torch.get_num_threads()}, expected 1")
    # Compare against what was captured at configure time, not against the seed --
    # the two are intentionally decoupled (see _assert_strict_env).
    if os.environ.get("PYTHONHASHSEED") != config.pythonhashseed:
        drift.append(
            f"PYTHONHASHSEED changed from {config.pythonhashseed!r} to "
            f"{os.environ.get('PYTHONHASHSEED')!r} mid-run"
        )

    if drift:
        where = f" ({context})" if context else ""
        raise NonDeterministicEnvironmentError(
            f"Strict determinism drifted{where}:\n  - " + "\n  - ".join(drift)
        )


def swallow_or_raise(exc, context, logger=None):
    """Re-raise in strict mode; log and continue in fast mode.

    Broad ``except Exception: continue`` blocks are the single largest *silent*
    divergence amplifier in this codebase: if run 2 swallows an exception run 1
    did not, bit-exactness breaks with no visible symptom. That is exactly how
    the evidential/MC-dropout crash presented as "accuracy 0.0" instead of a
    traceback. Route every such handler through here.
    """
    message = f"[{context}] {type(exc).__name__}: {exc}"
    if is_strict():
        raise RuntimeError(
            f"{message}\n(strict determinism: exceptions are not swallowed, because a "
            f"swallowed exception on one run and not another breaks reproducibility)"
        ) from exc
    if logger is not None:
        logger.warning(message)
    else:
        print(f"WARNING: {message}", file=sys.stderr)


def run_fingerprint(extra=None):
    """A machine-readable record of everything that affects reproducibility.

    Written to ``run_outputs/<run_id>/determinism.json``. Two runs that claim to
    be identical but disagree can then be diffed instead of guessed about.
    """
    config = _STATE["config"]
    fingerprint = {
        "determinism": config.as_dict() if config is not None else None,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "torch_flags": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "num_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": _git_state(),
    }
    if extra:
        fingerprint.update(extra)
    return fingerprint


def write_run_fingerprint(path, **extra):
    """Write (or update) ``determinism.json``. Returns the fingerprint written.

    Also the deep-ensemble member manifest: a launcher discovers its members by
    reading these files rather than by globbing checkpoint names, so a member that
    was launched with a different head or a different config cannot be silently
    folded into the average.

    Called more than once per run -- first at startup so a crashed run still leaves
    its environment on disk, then again after each configuration finishes. Later keys
    merge over earlier ones rather than replacing the file, and dict-valued keys merge
    one level deep so a multi-configuration run accumulates all of its results instead
    of each config clobbering the last.
    """
    path = str(path)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            # A corrupt or truncated file is not worth failing a training run over;
            # it gets overwritten with a good one.
            existing = {}

    fingerprint = run_fingerprint()
    # Environment facts are re-measured every write (flags can drift mid-run, which
    # is precisely what assert_strict_invariants exists to catch); caller-supplied
    # fields accumulate.
    merged = dict(existing)
    merged.update(fingerprint)
    for key, value in extra.items():
        if value is None:
            continue
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            combined = dict(previous)
            combined.update(value)
            merged[key] = combined
        else:
            merged[key] = value

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return merged


def _git_state():
    import subprocess

    def _run(args):
        try:
            result = subprocess.run(
                args, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }
