"""Shared pytest configuration and fixtures for GRIFT.

Tiers
-----
Bare ``pytest`` runs the fast tier: CPU only, synthetic fixtures, no dataset, no
network. Heavier tiers are opted into explicitly::

    pytest --run-gpu      # CUDA, real backbones, AMP, bit-exactness on device
    pytest --run-slow     # short multi-epoch train / traverse smoke runs
    pytest --run-data     # the real AI-Face dataset on disk
    pytest --run-network  # torch.hub downloads
    pytest --run-all      # everything

Tiers are gated by these options plus ``pytest_collection_modifyitems``, rather
than by ``-m`` in ``addopts``: a user's own ``-m gpu`` would silently *replace*
an addopts marker filter instead of adding to it, so the fast tier's exclusions
would disappear without any warning.

Determinism
-----------
``pytest_configure`` re-execs pytest itself with ``PYTHONHASHSEED=0`` and
``CUBLAS_WORKSPACE_CONFIG`` set, so bare ``pytest`` is sufficient for the
strict-mode bit-exactness assertions. Every test then runs under
``configure_determinism(mode="strict")``.
"""

import copy
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_SEED = 0
REEXEC_SENTINEL = "GRIFT_TEST_REEXEC"

TIER_MARKERS = ("gpu", "slow", "data", "network")

# Packages whose class-level `tags` / `hyperparameters` dicts must be restored
# between tests. The dataloaders mutate `self.hyperparameters` -- a *class*
# attribute -- via `.update(kwargs)`, so without this any test that constructs a
# dataloader leaks its config into every later test.
MUTABLE_ATTR_PACKAGES = (
    "nodes", "edges", "graphs", "managers", "traversals", "dataloaders", "datasets",
)
MUTABLE_CLASS_ATTRS = ("tags", "hyperparameters")


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #

def pytest_addoption(parser):
    group = parser.getgroup("grift")
    for tier in TIER_MARKERS:
        group.addoption(
            f"--run-{tier}", action="store_true", default=False,
            help=f"run tests marked '{tier}'",
        )
    group.addoption(
        "--run-all", action="store_true", default=False,
        help="run every tier (implies all --run-* options)",
    )
    group.addoption(
        "--ai-face-root", action="store", default=None,
        help="path to the AI-Face dataset root (overrides discovery for the data tier)",
    )


def pytest_configure(config):
    """Pin the reproducibility environment, re-execing once if needed."""
    if os.environ.get(REEXEC_SENTINEL) != "1" and os.environ.get("GRIFT_NO_REEXEC") != "1":
        wanted = {
            "PYTHONHASHSEED": str(TEST_SEED),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if any(os.environ.get(key) != value for key, value in wanted.items()):
            os.environ.update(wanted)
            os.environ[REEXEC_SENTINEL] = "1"
            sys.stderr.write(
                f"[tests] re-exec with PYTHONHASHSEED={wanted['PYTHONHASHSEED']}, "
                f"CUBLAS_WORKSPACE_CONFIG={wanted['CUBLAS_WORKSPACE_CONFIG']} "
                "(set GRIFT_NO_REEXEC=1 to disable, e.g. under a debugger)\n"
            )
            sys.stderr.flush()
            os.execv(sys.executable, list(getattr(sys, "orig_argv", [sys.executable] + sys.argv)))
        os.environ[REEXEC_SENTINEL] = "1"

    # opencv here is a qt6 build, and matplotlib is imported by several modules
    # at import time; both need to be headless.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    enabled = {
        tier: config.getoption("--run-all") or config.getoption(f"--run-{tier}")
        for tier in TIER_MARKERS
    }

    cuda_available = None
    if enabled["gpu"]:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

    for item in items:
        for tier in TIER_MARKERS:
            if tier not in item.keywords:
                continue
            if not enabled[tier]:
                item.add_marker(pytest.mark.skip(
                    reason=f"'{tier}' tier not enabled; pass --run-{tier} (or --run-all)"
                ))
                break
            if tier == "gpu" and not cuda_available:
                item.add_marker(pytest.mark.skip(reason="--run-gpu given but CUDA is unavailable"))
                break


# --------------------------------------------------------------------------- #
# Session fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def _pin_torch_threads():
    """Single-threaded torch for the whole session, so CPU reduction order is fixed."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # only settable before the interop pool spins up


@pytest.fixture(scope="session")
def class_attr_baseline():
    """Deep copies of every framework class's mutable class attributes."""
    import importlib

    baseline = {}
    for package_name in MUTABLE_ATTR_PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except Exception:
            continue
        import inspect
        for _, cls in inspect.getmembers(package, inspect.isclass):
            if not getattr(cls, "__module__", "").startswith(package_name):
                continue
            for attr in MUTABLE_CLASS_ATTRS:
                value = cls.__dict__.get(attr)
                if isinstance(value, (dict, list, set)):
                    baseline[(cls, attr)] = copy.deepcopy(value)
    return baseline


@pytest.fixture(scope="session")
def cuda_device():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def ai_face_root(request):
    """The real AI-Face dataset root, or skip."""
    explicit = request.config.getoption("--ai-face-root")
    from test_helpers.data_graph_utils import resolve_ai_face_data_root
    try:
        return Path(resolve_ai_face_data_root(explicit))
    except FileNotFoundError as exc:
        pytest.skip(f"AI-Face dataset not found: {exc}")


# --------------------------------------------------------------------------- #
# Autouse hygiene
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """Run each test from a temp cwd.

    The training code writes `logs/`, `graph_cache/`, `node_cache/`,
    `run_outputs/`, and `ivalue_visualizations/` relative to cwd. Note that
    `resolve_ai_face_data_root` also probes `$CWD/ai-face` and friends, so data-tier
    tests must pass an explicit root rather than rely on discovery.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _reset_determinism():
    """Re-seed before every test so order cannot leak RNG state."""
    from test_helpers.determinism import configure_determinism
    configure_determinism(seed=TEST_SEED, mode="strict", allow_multi_gpu=True)
    yield


@pytest.fixture(autouse=True)
def _restore_class_attrs(class_attr_baseline):
    """Restore mutated class attributes after each test.

    This is simultaneously hygiene and the regression test for the dataloaders'
    class-attribute mutation: it will fail loudly if that leak reappears.
    """
    yield
    for (cls, attr), pristine in class_attr_baseline.items():
        current = cls.__dict__.get(attr)
        if isinstance(current, (dict, list, set)) and current != pristine:
            if isinstance(current, dict):
                current.clear()
                current.update(copy.deepcopy(pristine))
            elif isinstance(current, list):
                current[:] = copy.deepcopy(pristine)
            else:
                current.clear()
                current.update(copy.deepcopy(pristine))


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Fail fast on any torch.hub download outside the ``network`` tier.

    The detectors pull from unpinned upstream branches
    (``zhanghang1989/ResNeSt``, ``NVIDIA/DeepLearningExamples:torchhub``,
    ``pytorch/vision:v0.10.0``), so an accidental download is both slow and a
    silent reproducibility hazard.
    """
    if "network" in request.keywords:
        return
    try:
        import torch.hub
    except ImportError:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "network access blocked in this test tier. Use the tiny_detector fixture, "
            "or mark the test with @pytest.mark.network and pass --run-network."
        )

    monkeypatch.setattr("torch.hub.load", _blocked, raising=False)
    monkeypatch.setattr("torch.hub.load_state_dict_from_url", _blocked, raising=False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

@pytest.fixture
def make_attr_node():
    from tests.helpers.factories import make_attr_node as factory
    return factory


@pytest.fixture
def attr_nodes():
    from tests.helpers.factories import make_attr_nodes
    return make_attr_nodes(6)


@pytest.fixture
def ring_graph():
    from tests.helpers.factories import build_ring_graph
    return build_ring_graph(6)


@pytest.fixture
def two_cluster_graph():
    from tests.helpers.factories import build_two_cluster_graph
    return build_two_cluster_graph(4)


@pytest.fixture
def isolated_node_graph():
    from tests.helpers.factories import build_isolated_node_graph
    return build_isolated_node_graph()


@pytest.fixture
def dummy_trainer():
    from tests.helpers.factories import DummyTrainer
    return DummyTrainer()


@pytest.fixture
def tiny_png(tmp_path):
    from tests.helpers.images import write_tiny_png
    return write_tiny_png(tmp_path / "img.png")


@pytest.fixture
def image_nodes(tmp_path):
    from tests.helpers.images import make_image_nodes
    return make_image_nodes(tmp_path / "images", count=8, size=8)


@pytest.fixture
def tiny_detector():
    """Install the graftable tiny detector; yields its architecture name."""
    from tests.helpers.tiny_detector import register_tiny_detector, unregister_tiny_detectors
    name = register_tiny_detector()
    yield name
    unregister_tiny_detectors()


@pytest.fixture
def tiny_detector_no_linear():
    """Install the Linear-free (squeezenetdf-shaped) tiny detector."""
    from tests.helpers.tiny_detector import (
        register_tiny_detector_no_linear, unregister_tiny_detectors,
    )
    name = register_tiny_detector_no_linear()
    yield name
    unregister_tiny_detectors()


@pytest.fixture
def cnn_model_factory(tiny_detector):
    """Build a CNNModel on the tiny detector. Kwargs pass through to CNNModel."""
    import torch
    from models.CNNModel import CNNModel

    built = []

    def factory(save_path=None, uncertainty_head="none", lr=1e-3, **kwargs):
        model = CNNModel(
            save_path=str(save_path) if save_path else "tiny_model.pth",
            model_name=tiny_detector,
            lr=lr,
            amsgrad=True,
            device=torch.device("cpu"),
            uncertainty_head=uncertainty_head,
            **kwargs,
        )
        built.append(model)
        return model

    return factory


@pytest.fixture(params=["none", "evidential", "batchensemble", "sngp"])
def cnn_model(request, cnn_model_factory):
    """A CNNModel per uncertainty head, so head-agnostic tests run all four."""
    model = cnn_model_factory(uncertainty_head=request.param)
    model.uncertainty_head_param = request.param
    return model


@pytest.fixture
def tiny_batch():
    from tests.helpers.tiny_detector import tiny_batch as factory
    return factory


@pytest.fixture
def tiny_labels():
    from tests.helpers.tiny_detector import tiny_labels as factory
    return factory


@pytest.fixture
def state_dict_hash():
    from tests.helpers.determinism import state_dict_hash as helper
    return helper


@pytest.fixture
def assert_bit_exact():
    from tests.helpers.determinism import assert_bit_exact as helper
    return helper


@pytest.fixture
def run_twice():
    from tests.helpers.determinism import run_twice as helper
    return helper


@pytest.fixture
def subprocess_run_py():
    from tests.helpers.determinism import subprocess_run_py as helper
    return helper
