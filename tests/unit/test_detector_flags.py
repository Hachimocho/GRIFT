"""`pretrained` and `finetune` must do what they say.

`vistransformdf` accepted both and ignored both: two models built with opposite `pretrained`
values came out bit-identical, no ViT checkpoint was ever fetched, and `finetune=True` left
all 91,070,977 parameters trainable. It therefore trained a ViT from scratch on 2000 images
for three epochs and scored at chance (AUROC 0.50 and 0.56) in two consecutive sweeps.

The construction tests are `network`-marked because they fetch real weights; the
source-level checks below run in the fast tier so a regression is caught without a download.
"""

import ast
import inspect
import os

import pytest


@pytest.fixture
def detector_source(repo_root):
    """Return a reader for a detector's source text.

    Never imports: `models/detectors/__init__.py` eagerly imports eleven DeepfakeBench
    detectors and therefore `efficientnet_pytorch`, and `sys.modules` keeps them for the
    rest of the session -- which breaks
    `test_harness_smoke.py::test_tiny_detector_avoids_the_detector_zoo` for whatever runs
    after. Importing also binds `torch.hub.load` at that moment.
    """
    def read(name):
        return open(os.path.join(repo_root, "models", "detectors", f"{name}.py")).read()
    return read


@pytest.fixture
def effnetdf_source(repo_root):
    """The detector's source text.

    Read rather than imported: an autouse fixture chdirs each test into a temp directory,
    so the path must be anchored, and importing a detector in the fast tier binds whatever
    `torch.hub.load` is at that moment for the rest of the session.
    """
    return open(os.path.join(repo_root, "models", "detectors", "effnetdf.py")).read()


# -- source-level, no download -------------------------------------------------- #

def test_vistransformdf_uses_the_pretrained_argument(detector_source):
    source = detector_source("vistransformdf")
    assert "ViT_B_16_Weights" in source
    # The flag has to reach the constructor, not sit unused.
    assert "weights=weights" in source
    assert "if pretrained else None" in source


def test_vistransformdf_honours_finetune(detector_source):
    source = detector_source("vistransformdf")
    assert "requires_grad = False" in source, "finetune must freeze something"


def test_vistransformdf_resizes_for_the_shared_transform(detector_source):
    """The pipeline emits 255x255 and torchvision's ViT asserts its exact image_size."""
    source = detector_source("vistransformdf")
    assert "INPUT_SIZE = 224" in source
    assert "interpolate" in source


def test_effnetdf_keeps_the_backbone_pooling_by_default(effnetdf_source):
    """Max pooling inflates the activation scale 42.7x into a fresh Linear(1792, 1024)."""
    source = effnetdf_source
    assert "AdaptiveMaxPool2d" in source, "the option should remain reachable"
    assert "configuration == 'maxpool'" in source, "but it must be opt-in"


def test_effnetdf_does_not_import_distutils(effnetdf_source):
    """Removed from the stdlib in 3.12; it resolved only through setuptools' shim.

    Read from disk rather than imported: importing a detector inside the fast tier binds
    whatever `torch.hub.load` is at that moment, and the module cache then keeps it for the
    rest of the session.
    """
    source = ast.parse(effnetdf_source)
    imported = {
        alias.name
        for node in ast.walk(source)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("distutils" in str(name) for name in imported)


def test_effnetdf_resolves_torch_hub_at_call_time(effnetdf_source):
    """`from torch.hub import load` made the network block depend on import order."""
    assert "from torch.hub import load" not in effnetdf_source
    assert "torch.hub.load(" in effnetdf_source


# -- real construction ---------------------------------------------------------- #

pytestmark_network = pytest.mark.network


@pytest.mark.network
def test_pretrained_changes_the_weights():
    import torch

    from models.detectors.vistransformdf import ModelOut

    trained = ModelOut(pretrained=True, output_classes=1, classification_strategy='binary')
    random_init = ModelOut(pretrained=False, output_classes=1, classification_strategy='binary')
    a = next(trained.model.parameters()).detach().flatten()[:8]
    b = next(random_init.model.parameters()).detach().flatten()[:8]
    assert not torch.equal(a, b), "`pretrained` must change the weights"


@pytest.mark.network
def test_finetune_freezes_the_backbone():
    from models.detectors.vistransformdf import ModelOut

    full = ModelOut(pretrained=True, finetune=False, output_classes=1,
                    classification_strategy='binary')
    probe = ModelOut(pretrained=True, finetune=True, output_classes=1,
                     classification_strategy='binary')
    trainable = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable(probe) < trainable(full) / 100, (
        "finetune must leave only the head trainable"
    )


@pytest.mark.network
@pytest.mark.parametrize("name", ["vistransformdf", "effnetdf"])
def test_the_capability_table_matches_the_built_model(name):
    """`reconcile` exists because a torchvision upgrade can silently invalidate the table."""
    import importlib

    import torch
    import torch.nn as nn

    from models.uncertainty.capabilities import profile_for

    module = importlib.import_module(f"models.detectors.{name}")
    model = module.ModelOut(pretrained=True, finetune=False, output_classes=1,
                            classification_strategy='binary')
    profile = profile_for(name)

    graft = dict(model.model.named_modules()).get(profile.last_linear_path)
    assert isinstance(graft, nn.Linear), (
        f"{name}: table says the graft point is {profile.last_linear_path}"
    )
    assert graft.in_features == profile.last_linear_in_features

    sites = sum(1 for m in model.modules() if isinstance(m, nn.Dropout) and m.p > 0)
    assert sites == profile.dropout_sites_head_none

    # And it must accept what the shared transform actually produces.
    with torch.no_grad():
        assert tuple(model(torch.rand(2, 3, 255, 255)).shape) == (2, 1)
