"""Which uncertainty methods work with which detectors.

GRIFT's framework is model-agnostic in the sense that any ``models/detectors/*.py``
exposing a ``ModelOut`` can be selected by name. Uncertainty methods are *not*
equally agnostic, and the incompatibilities are **structural rather than
family-based** -- "is it a CNN or a transformer" predicts almost nothing here:

* ``squeezenetdf`` has a convolutional classifier and **zero** ``nn.Linear``
  modules, so there is nothing to graft an external head onto -- yet it is the
  detector where MC dropout works best out of the box (``classifier.0``, p=0.5).
* ``vistransformdf`` and ``swintransformdf`` are both transformers, but the former
  exposes a genuine 768-d CLS embedding while the latter exposes a 1024-d
  post-dropout head activation.
* ``vistransformdf`` has 37 ``nn.Dropout`` modules and every one is ``p=0.0``, so
  MC dropout on it returns *identically zero* variance -- a silently wrong
  measurement, not an error.
* Five detectors have no ``self.model`` attribute at all and raise inside
  ``CNNModel.__init__``; two more wrap a non-``nn.Module`` holder and additionally
  need a checkpoint file that is not in the repo.

Two deliberate design choices:

**An explicit table keyed by module name, not the `tags` mechanism.** The ``tags``
convention is vestigial -- its only consumers are a doc generator and a helper with
no callers (and a latent bug: it is documented to accept a list of tags but tests
``tags in obj.tags``). Reviving it would mean *building* the resolution machinery,
and flat string tags cannot express structured requirements anyway. String-name
dispatch is what the rest of the codebase already does.

**Static declarations are cross-checked against a runtime probe.** Whether MC
dropout is usable depends on ``--uncertainty-dropout-rate`` and on modules that
only exist after construction, so it cannot be answered by a table alone.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


class Capability:
    """Capabilities a model can offer, that methods can require."""

    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    PENULTIMATE_FEATURES = "penultimate_features"
    LAST_LINEAR_GRAFT = "last_linear_graft"
    STOCHASTIC_DROPOUT = "stochastic_dropout"
    MEMBER_LOGITS = "member_logits"
    EVIDENCE_ALPHA = "evidence_alpha"
    GP_VARIANCE = "gp_variance"
    NODE_ATTRIBUTES = "node_attributes"
    NODE_EMBEDDING = "node_embedding"
    GRAPH_EDGES = "graph_edges"
    MULTI_CHECKPOINT = "multi_checkpoint"
    VAL_SPLIT = "val_split"

    ALL = frozenset({
        LOGITS, PROBABILITIES, PENULTIMATE_FEATURES, LAST_LINEAR_GRAFT,
        STOCHASTIC_DROPOUT, MEMBER_LOGITS, EVIDENCE_ALPHA, GP_VARIANCE,
        NODE_ATTRIBUTES, NODE_EMBEDDING, GRAPH_EDGES, MULTI_CHECKPOINT, VAL_SPLIT,
    })


#: Detector support levels.
SUPPORTED = "supported"        # external head + MC dropout + feature methods
LOGIT_ONLY = "logit_only"      # no graftable Linear; logit-space methods only
BROKEN = "broken"              # raises during construction


@dataclass(frozen=True)
class DetectorProfile:
    """Verified facts about one ``models/detectors/<name>.py`` module."""

    name: str
    status: str
    static_capabilities: frozenset = field(default_factory=frozenset)
    last_linear_path: Optional[str] = None
    last_linear_in_features: Optional[int] = None
    penultimate_space: Optional[str] = None
    backbone_embedding_dim: Optional[int] = None
    dropout_sites_head_none: int = 0
    freezes_backbone_when_finetune: bool = False
    requires_download: bool = True
    broken_reason: Optional[str] = None
    notes: str = ""

    @property
    def usable(self):
        return self.status != BROKEN

    @property
    def can_graft_head(self):
        return Capability.LAST_LINEAR_GRAFT in self.static_capabilities


_GRAFTABLE = frozenset({
    Capability.LOGITS, Capability.PROBABILITIES, Capability.PENULTIMATE_FEATURES,
    Capability.LAST_LINEAR_GRAFT,
})
_LOGITS_ONLY = frozenset({Capability.LOGITS, Capability.PROBABILITIES})


DETECTOR_PROFILES = {
    "effnetdf": DetectorProfile(
        name="effnetdf", status=SUPPORTED, static_capabilities=_GRAFTABLE,
        last_linear_path="classifier.fc.2", last_linear_in_features=1024,
        penultimate_space="head1024_postdrop", backbone_embedding_dim=1792,
        # Two, measured: the backbone's own `classifier.dropout` plus the grafted head's
        # Dropout(0.4). Recorded as 1 before, which understated MC dropout's viability
        # here -- `registry.py` quotes this number when explaining a gate decision.
        dropout_sites_head_none=2, freezes_backbone_when_finetune=True,
        notes="NVIDIA EfficientNet-B4 via torch.hub. Head is "
              "Sequential(Linear(1792,1024), Dropout(0.4), Linear(1024,1)), so grafting "
              "on the last Linear puts the uncertainty head on a post-dropout "
              "activation rather than the backbone embedding. The backbone's native "
              "AdaptiveAvgPool2d is retained: it was previously replaced with "
              "AdaptiveMaxPool2d, which inflates the activation scale 42.7x into a "
              "freshly initialized Linear(1792, 1024) and saturates it. Pass "
              "configuration='maxpool' to restore that behavior.",
    ),
    "resnestdf": DetectorProfile(
        name="resnestdf", status=SUPPORTED, static_capabilities=_GRAFTABLE,
        last_linear_path="fc.2", last_linear_in_features=1024,
        penultimate_space="head1024_postdrop", backbone_embedding_dim=2048,
        dropout_sites_head_none=1,
        notes="ResNeSt-50 via torch.hub (unpinned upstream branch).",
    ),
    "swintransformdf": DetectorProfile(
        name="swintransformdf", status=SUPPORTED, static_capabilities=_GRAFTABLE,
        last_linear_path="head.2", last_linear_in_features=1024,
        penultimate_space="head1024_postdrop", backbone_embedding_dim=768,
        dropout_sites_head_none=1, freezes_backbone_when_finetune=True,
        notes="torchvision swin_t. Its 24 backbone nn.Dropout modules are all p=0.0 "
              "(Swin uses stochastic depth, not dropout), so the only usable dropout "
              "site is the Dropout(0.4) in the replaced head.",
    ),
    "vistransformdf": DetectorProfile(
        name="vistransformdf", status=SUPPORTED, static_capabilities=_GRAFTABLE,
        last_linear_path="heads.head", last_linear_in_features=768,
        penultimate_space="vit_cls768", backbone_embedding_dim=768,
        dropout_sites_head_none=0, requires_download=True,
        freezes_backbone_when_finetune=True,
        notes="torchvision vit_b_16 with IMAGENET1K_V1 weights. It previously built a "
              "hand-configured VisionTransformer at 255/51, where `pretrained` and "
              "`finetune` were both accepted and silently ignored -- two models built with "
              "opposite flags came out bit-identical, and all 91M parameters trained from "
              "scratch, which is why it scored at chance (AUROC 0.50-0.56) in both sweeps. "
              "The pipeline resizes to 255x255 and ViT asserts its exact image_size, so the "
              "detector interpolates to 224 in forward. Graft point, 768-d penultimate "
              "space and zero p>0 dropout sites are unchanged, so MC dropout still yields "
              "identically zero variance unless a head supplies dropout. This is the CLI "
              "default architecture.",
    ),
    "squeezenetdf": DetectorProfile(
        name="squeezenetdf", status=LOGIT_ONLY,
        static_capabilities=_LOGITS_ONLY | frozenset({Capability.STOCHASTIC_DROPOUT}),
        last_linear_path=None, last_linear_in_features=None,
        penultimate_space=None, backbone_embedding_dim=None,
        dropout_sites_head_none=1,
        notes="SqueezeNet's classifier is nn.Conv2d and the network has zero "
              "nn.Linear modules, so _find_last_linear returns (None, None) and no "
              "external head can be attached; bundle.features stays None. Its "
              "classifier.0 Dropout(0.5) does survive, so MC dropout works natively.",
    ),
}

_BROKEN_NO_INNER_MODEL = (
    "ModelOut has no `self.model` attribute, so CNNModel.__init__ raises "
    "AttributeError at `self.model.model.to(device)`."
)
for _name in ("xceptiondf", "xceptionalternate", "mesonetdf", "mesoinceptiondf"):
    DETECTOR_PROFILES[_name] = DetectorProfile(
        name=_name, status=BROKEN, broken_reason=_BROKEN_NO_INNER_MODEL,
    )

DETECTOR_PROFILES["facexray"] = DetectorProfile(
    name="facexray", status=BROKEN,
    broken_reason=_BROKEN_NO_INNER_MODEL + " It is also not a network: forward() "
                  "returns a Python list of ints, it imports face_alignment at module "
                  "import, and it appends a hardcoded absolute path to sys.path.",
)
for _name in ("dag_fdd", "daw_fdd"):
    DETECTOR_PROFILES[_name] = DetectorProfile(
        name=_name, status=BROKEN,
        broken_reason="`self.model` is a plain _BackboneHolder rather than an "
                      "nn.Module, so CNNModel's dropout_controller.add_module raises "
                      "TypeError. It also loads ./models/detectors/"
                      "xception-b5690688.pth, which is not present in the repo.",
    )


def profile_for(detector_name):
    """Profile for ``detector_name``, or None if it is not catalogued."""
    return DETECTOR_PROFILES.get(detector_name)


def supported_detectors(include_logit_only=True):
    """Detector names that can actually be trained."""
    return sorted(
        name for name, profile in DETECTOR_PROFILES.items()
        if profile.status == SUPPORTED
        or (include_logit_only and profile.status == LOGIT_ONLY)
    )


def broken_detectors():
    return sorted(
        name for name, profile in DETECTOR_PROFILES.items() if profile.status == BROKEN
    )


def describe_detector(detector_name):
    """One-line human-readable summary, for CLI validation messages."""
    profile = profile_for(detector_name)
    if profile is None:
        return f"{detector_name}: not catalogued in DETECTOR_PROFILES"
    if profile.status == BROKEN:
        return f"{detector_name}: BROKEN -- {profile.broken_reason}"
    graft = profile.last_linear_path or "no nn.Linear (logit-only)"
    return (
        f"{detector_name}: {profile.status}, graft at {graft}, "
        f"{profile.dropout_sites_head_none} usable dropout site(s) with head='none'"
    )


def validate_architectures(names, allow_broken=False):
    """Split requested architecture names into (usable, problems).

    ``problems`` maps name -> reason. Used to fail at argument-parse time rather
    than with a ModuleNotFoundError or AttributeError deep inside model
    construction.
    """
    usable, problems = [], {}
    for name in names:
        profile = profile_for(name)
        if profile is None:
            problems[name] = (
                f"unknown detector {name!r}; catalogued detectors are "
                f"{sorted(DETECTOR_PROFILES)}"
            )
        elif profile.status == BROKEN and not allow_broken:
            problems[name] = profile.broken_reason
        else:
            usable.append(name)
    return usable, problems


def probe_model_capabilities(model):
    """Capabilities observed on a *constructed* model.

    Complements the static table with the facts that can only be known at runtime:
    whether any dropout module actually randomizes, and whether the model exposes
    penultimate features.
    """
    from .mc_dropout import count_stochastic_dropout_sites

    observed = {Capability.LOGITS, Capability.PROBABILITIES}

    controller = getattr(model, "dropout_controller", None)
    if controller is not None and count_stochastic_dropout_sites(controller) > 0:
        observed.add(Capability.STOCHASTIC_DROPOUT)

    if getattr(model, "output_head", None) is not None:
        observed.add(Capability.LAST_LINEAR_GRAFT)
        head_type = getattr(model, "uncertainty_head_type", "none")
        if head_type == "evidential":
            observed.add(Capability.EVIDENCE_ALPHA)
        elif head_type == "batchensemble":
            observed.add(Capability.MEMBER_LOGITS)
        elif head_type == "sngp":
            observed.add(Capability.GP_VARIANCE)
    elif getattr(model, "final_linear_path", None) is not None:
        observed.add(Capability.LAST_LINEAR_GRAFT)
        observed.add(Capability.PENULTIMATE_FEATURES)

    if getattr(model, "graph_distance_standardizer", None) is not None:
        observed.add(Capability.NODE_ATTRIBUTES)
        observed.add(Capability.GRAPH_EDGES)
        coverage = model.graph_distance_standardizer.embedding_coverage
        if coverage is not None and coverage > 0.0:
            observed.add(Capability.NODE_EMBEDDING)

    return frozenset(observed)


def reconcile(detector_name, probed, strict=False):
    """Compare a probe against the static profile; return a list of discrepancies.

    Raises in strict mode. This is what catches a torchvision or upstream-hub
    upgrade silently invalidating the table -- e.g. if a future release gives
    VisionTransformer a non-zero default dropout.
    """
    profile = profile_for(detector_name)
    if profile is None:
        message = f"{detector_name} is not catalogued in DETECTOR_PROFILES"
        if strict:
            raise ValueError(message)
        return [message]

    discrepancies = []
    expects_dropout = profile.dropout_sites_head_none > 0
    observes_dropout = Capability.STOCHASTIC_DROPOUT in probed
    if expects_dropout != observes_dropout:
        discrepancies.append(
            f"{detector_name}: profile says dropout_sites_head_none="
            f"{profile.dropout_sites_head_none} but probe "
            f"{'found' if observes_dropout else 'found no'} stochastic dropout"
        )

    if profile.can_graft_head and Capability.LAST_LINEAR_GRAFT not in probed:
        discrepancies.append(
            f"{detector_name}: profile says it is graftable but the probe found no "
            f"nn.Linear to graft onto"
        )

    if strict and discrepancies:
        raise ValueError(
            "detector capability profile is out of date:\n  - "
            + "\n  - ".join(discrepancies)
        )
    return discrepancies
