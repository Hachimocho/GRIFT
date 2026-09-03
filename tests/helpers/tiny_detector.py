"""Tiny synthetic detectors, injected into ``sys.modules``.

``CNNModel.__init__`` resolves its backbone with
``importlib.import_module(f'models.detectors.{model_name}').ModelOut``. Importing
any real detector triggers ``models/detectors/__init__.py``, which imports all
ten DeepfakeBench detectors, which imports ``models/networks/__init__.py`` and
``efficientnet_pytorch`` -- and most real detectors then fetch pretrained weights
from *unpinned* upstream GitHub branches via ``torch.hub``.

Pre-seeding ``sys.modules`` with a fake module short-circuits that: ``importlib``
returns our entry without touching the package. All four uncertainty heads then
construct, forward, compute loss, and backward in well under a second on CPU,
with no network access.

Two shapes are provided:

``ModelOut``            ends in ``nn.Linear`` -- the graftable case, standing in
                        for effnetdf / resnestdf / swintransformdf / vistransformdf.
``ModelOutNoLinear``    ends in ``nn.Conv2d`` with zero ``nn.Linear`` anywhere --
                        reproduces squeezenetdf, where ``_find_last_linear``
                        returns ``(None, None)``.
"""

import sys
import types

import torch
import torch.nn as nn

TINY_MODULE = "models.detectors.grifttiny"
TINY_NO_LINEAR_MODULE = "models.detectors.grifttinynolinear"

TINY_FEATURE_DIM = 8


class _TinyModelOut(nn.Module):
    """Mirrors the real detectors' two-level structure.

    ``self.model`` is the inner network -- CNNModel reaches through to it for
    device placement, the dropout controller, last-Linear discovery, module
    surgery, ``.parameters()``, mode switching, and checkpointing. The five
    detectors that omit this attribute are exactly the ones that crash at
    ``CNNModel.py:54``.
    """

    def __init__(self, pretrained=False, finetune=False, exclude_top=False,
                 output_classes=1, classification_strategy="binary",
                 configuration="default", feature_dim=TINY_FEATURE_DIM):
        super().__init__()
        self.in_features = feature_dim
        self.out_features = output_classes if classification_strategy == "categorical" else 1
        # Head is a Sequential ending in Linear, matching effnetdf/resnestdf/swin:
        # Linear -> Dropout -> Linear, so the last Linear sits at a digit path and
        # exercises _replace_module's integer-index branch.
        self.model = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, self.out_features),
        )
        if finetune:
            # Deliberately mirrors the real detectors' freeze convention so tests
            # can assert on --finetune / --no-finetune behavior.
            for name, param in self.model.named_parameters():
                if "6." not in name:  # only the final Linear stays trainable
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class _TinyModelOutNoLinear(nn.Module):
    """Zero ``nn.Linear`` modules -- the squeezenetdf shape."""

    def __init__(self, pretrained=False, finetune=False, exclude_top=False,
                 output_classes=1, classification_strategy="binary",
                 configuration="default", feature_dim=TINY_FEATURE_DIM):
        super().__init__()
        self.in_features = feature_dim
        self.out_features = output_classes if classification_strategy == "categorical" else 1
        self.model = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),  # squeezenet's classifier.0 -- MC dropout works here
            nn.Conv2d(feature_dim, self.out_features, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        return self.model(x)


def _install(module_name, model_out_class, profile_factory):
    module = types.ModuleType(module_name)
    module.ModelOut = model_out_class
    module.__all__ = ["ModelOut"]
    sys.modules[module_name] = module

    # A capability profile too, so the tiny detector survives the *real* validation
    # path. `validate_architectures` rejects any name absent from DETECTOR_PROFILES, and
    # `test_hierarchical.main` calls it at startup and exits 1 -- so without this, a test
    # can construct a CNNModel on the tiny detector but cannot drive the entrypoint.
    from models.uncertainty import capabilities

    name = module_name.rsplit(".", 1)[-1]
    capabilities.DETECTOR_PROFILES[name] = profile_factory(name)
    return name


def _graftable_profile(name):
    from models.uncertainty.capabilities import (
        SUPPORTED, Capability, DetectorProfile,
    )

    return DetectorProfile(
        name=name, status=SUPPORTED,
        static_capabilities=frozenset({
            Capability.LOGITS, Capability.PROBABILITIES,
            Capability.PENULTIMATE_FEATURES, Capability.LAST_LINEAR_GRAFT,
            Capability.STOCHASTIC_DROPOUT,
        }),
        last_linear_path="model.6", last_linear_in_features=TINY_FEATURE_DIM * 2,
        # Its own space: a synthetic 16-d head must never be pooled with a real
        # detector's 1024-d one by `comparable_detector_groups`.
        penultimate_space=f"{name}_head{TINY_FEATURE_DIM * 2}_postdrop",
        backbone_embedding_dim=TINY_FEATURE_DIM,
        dropout_sites_head_none=1, requires_download=False,
        notes="Synthetic test detector. Mirrors the effnetdf/resnestdf head shape "
              "(Linear -> Dropout -> Linear) so head grafting takes the same path.",
    )


def _no_linear_profile(name):
    from models.uncertainty.capabilities import (
        LOGIT_ONLY, Capability, DetectorProfile,
    )

    return DetectorProfile(
        name=name, status=LOGIT_ONLY,
        static_capabilities=frozenset({
            Capability.LOGITS, Capability.PROBABILITIES,
            Capability.STOCHASTIC_DROPOUT,
        }),
        last_linear_path=None, last_linear_in_features=None,
        penultimate_space=None, backbone_embedding_dim=TINY_FEATURE_DIM,
        dropout_sites_head_none=1, requires_download=False,
        notes="Synthetic test detector with no nn.Linear anywhere, reproducing "
              "squeezenetdf: head grafting is impossible, logit methods still work.",
    )


def register_tiny_detector():
    """Install the graftable tiny detector; returns its ``--architectures`` name."""
    return _install(TINY_MODULE, _TinyModelOut, _graftable_profile)


def register_tiny_detector_no_linear():
    """Install the Linear-free tiny detector; returns its name."""
    return _install(
        TINY_NO_LINEAR_MODULE, _TinyModelOutNoLinear, _no_linear_profile
    )


def unregister_tiny_detectors():
    from models.uncertainty import capabilities

    for module_name in (TINY_MODULE, TINY_NO_LINEAR_MODULE):
        sys.modules.pop(module_name, None)
        capabilities.DETECTOR_PROFILES.pop(module_name.rsplit(".", 1)[-1], None)


def tiny_batch(batch_size=4, size=16, seed=0):
    """A small deterministic image batch shaped [B, 3, size, size]."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, 3, size, size, generator=generator)


def tiny_labels(batch_size=4):
    """Labels shaped [B, 1] float, as the training loop and BCEWithLogitsLoss expect."""
    return torch.tensor(
        [[float(index % 2)] for index in range(batch_size)], dtype=torch.float
    )
