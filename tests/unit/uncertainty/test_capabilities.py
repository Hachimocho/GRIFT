"""The detector capability table and its runtime cross-check."""

import pytest

from models.uncertainty.capabilities import (
    BROKEN, DETECTOR_PROFILES, LOGIT_ONLY, SUPPORTED, Capability, broken_detectors,
    describe_detector, probe_model_capabilities, profile_for, reconcile,
    supported_detectors, validate_architectures,
)


def test_every_ui_offered_architecture_is_catalogued():
    """The web UI offers 11 architectures and --architectures had no validation.

    Users will select the broken ones, so the table must at least know why they
    fail -- that turns a 40-line traceback into one line.
    """
    ui_offered = {
        "resnestdf", "effnetdf", "xceptiondf", "mesoinceptiondf", "mesonetdf",
        "facexray", "vistransformdf", "swintransformdf", "squeezenetdf",
        "dag_fdd", "daw_fdd",
    }
    missing = ui_offered - set(DETECTOR_PROFILES)
    assert not missing, f"architectures offered in the UI but not catalogued: {sorted(missing)}"


def test_supported_and_broken_partitions():
    assert set(supported_detectors()) == {
        "effnetdf", "resnestdf", "swintransformdf", "vistransformdf", "squeezenetdf",
    }
    assert set(supported_detectors(include_logit_only=False)) == {
        "effnetdf", "resnestdf", "swintransformdf", "vistransformdf",
    }
    assert set(broken_detectors()) == {
        "xceptiondf", "xceptionalternate", "mesonetdf", "mesoinceptiondf",
        "facexray", "dag_fdd", "daw_fdd",
    }


def test_every_broken_detector_states_a_reason():
    for name in broken_detectors():
        reason = profile_for(name).broken_reason
        assert reason and len(reason) > 20, f"{name} has no useful broken_reason"


def test_squeezenetdf_is_logit_only_but_supports_mc_dropout():
    """The clearest case of structural, not family-based, compatibility.

    It cannot host an external head (no nn.Linear at all) yet it is the one
    detector where MC dropout works without any head attached.
    """
    profile = profile_for("squeezenetdf")
    assert profile.status == LOGIT_ONLY
    assert profile.last_linear_path is None
    assert not profile.can_graft_head
    assert Capability.STOCHASTIC_DROPOUT in profile.static_capabilities
    assert profile.dropout_sites_head_none == 1


def test_vistransformdf_has_no_usable_dropout():
    """37 nn.Dropout modules, all p=0.0 -> MC dropout measures nothing."""
    profile = profile_for("vistransformdf")
    assert profile.status == SUPPORTED
    assert profile.dropout_sites_head_none == 0
    assert Capability.STOCHASTIC_DROPOUT not in profile.static_capabilities
    # It *does* download now. This asserted False while `pretrained` was accepted and
    # ignored, which is precisely the bug that had it training a 91M-parameter ViT from
    # scratch and scoring at chance.
    assert profile.requires_download is True


def test_frozen_backbone_detectors_are_flagged():
    """finetune=True freezes these two, which makes a "deep ensemble" head-only."""
    assert profile_for("effnetdf").freezes_backbone_when_finetune is True
    assert profile_for("swintransformdf").freezes_backbone_when_finetune is True
    assert profile_for("resnestdf").freezes_backbone_when_finetune is False


def test_penultimate_spaces_are_not_comparable_across_detectors():
    """Feature-space methods cannot be compared across these detectors.

    Three expose a 1024-d post-dropout head activation, one a 768-d ViT CLS token,
    and one nothing at all -- so any cross-detector feature-space number would be
    meaningless.
    """
    spaces = {
        name: profile_for(name).penultimate_space
        for name in supported_detectors()
    }
    assert spaces["vistransformdf"] != spaces["effnetdf"]
    assert spaces["squeezenetdf"] is None
    assert len({space for space in spaces.values() if space}) > 1


@pytest.mark.parametrize("name", sorted(DETECTOR_PROFILES))
def test_graftable_profiles_declare_their_graft_point(name):
    profile = profile_for(name)
    if profile.can_graft_head:
        assert profile.last_linear_path, f"{name} claims graftable but names no path"
        assert profile.last_linear_in_features, f"{name} names no in_features"
    elif profile.status != BROKEN:
        assert profile.last_linear_path is None


# --------------------------------------------------------------------------- #
# validate_architectures
# --------------------------------------------------------------------------- #

def test_validate_accepts_supported_architectures():
    usable, problems = validate_architectures(["effnetdf", "vistransformdf"])
    assert usable == ["effnetdf", "vistransformdf"]
    assert problems == {}


def test_validate_rejects_broken_architectures():
    usable, problems = validate_architectures(["effnetdf", "dag_fdd"])
    assert usable == ["effnetdf"]
    assert "dag_fdd" in problems
    assert "xception-b5690688.pth" in problems["dag_fdd"]


def test_validate_rejects_unknown_architectures():
    """A typo used to become a ModuleNotFoundError deep inside CNNModel.__init__."""
    usable, problems = validate_architectures(["efficientnet"])
    assert usable == []
    assert "unknown detector" in problems["efficientnet"]


def test_validate_can_allow_broken_explicitly():
    usable, problems = validate_architectures(["dag_fdd"], allow_broken=True)
    assert usable == ["dag_fdd"] and problems == {}


def test_describe_detector_is_informative():
    assert "BROKEN" in describe_detector("dag_fdd")
    assert "classifier.fc.2" in describe_detector("effnetdf")
    assert "logit-only" in describe_detector("squeezenetdf")
    assert "not catalogued" in describe_detector("nonexistent")


# --------------------------------------------------------------------------- #
# Runtime probe
# --------------------------------------------------------------------------- #

def test_probe_detects_a_grafted_head(cnn_model_factory):
    for head, expected in (
        ("evidential", Capability.EVIDENCE_ALPHA),
        ("batchensemble", Capability.MEMBER_LOGITS),
        ("sngp", Capability.GP_VARIANCE),
    ):
        model = cnn_model_factory(uncertainty_head=head, uncertainty_dropout_rate=0.3)
        probed = probe_model_capabilities(model)
        assert Capability.LAST_LINEAR_GRAFT in probed
        assert expected in probed
        assert Capability.STOCHASTIC_DROPOUT in probed


def test_probe_detects_absent_stochastic_dropout(cnn_model_factory):
    """dropout_rate=0 must not count as available MC dropout.

    This is the runtime half of the vistransformdf problem: the number of usable
    dropout sites depends on a user-supplied rate, so no static table can answer it.
    """
    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.0)
    # The tiny detector's own Dropout(0.3) is still present, so drop it too.
    import torch.nn as nn
    for module in model.dropout_controller.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    assert Capability.STOCHASTIC_DROPOUT not in probe_model_capabilities(model)
    assert model.mc_dropout_available() is False


def test_probe_detects_penultimate_features_without_a_head(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none")
    probed = probe_model_capabilities(model)
    assert Capability.PENULTIMATE_FEATURES in probed
    assert Capability.EVIDENCE_ALPHA not in probed


def test_probe_reports_no_graft_for_a_linear_free_backbone(tiny_detector_no_linear):
    import torch
    from models.CNNModel import CNNModel

    model = CNNModel(
        save_path="x.pth", model_name=tiny_detector_no_linear, lr=1e-3, amsgrad=True,
        device=torch.device("cpu"), uncertainty_head="none",
    )
    probed = probe_model_capabilities(model)
    assert Capability.LAST_LINEAR_GRAFT not in probed
    assert Capability.PENULTIMATE_FEATURES not in probed
    # Its Dropout(0.5) is real, so MC dropout is still available -- exactly the
    # squeezenetdf situation.
    assert Capability.STOCHASTIC_DROPOUT in probed


def test_probe_detects_graph_capabilities(cnn_model_factory, ring_graph):
    from models.uncertainty import GraphDistanceUncertainty

    _, nodes, _ = ring_graph
    model = cnn_model_factory(
        uncertainty_head="none", graph_uncertainty_methods=["attribute_distance"]
    )
    assert Capability.NODE_ATTRIBUTES not in probe_model_capabilities(model)

    model.set_graph_distance_standardizer(
        GraphDistanceUncertainty(methods=("attribute_distance",)).fit(nodes)
    )
    probed = probe_model_capabilities(model)
    assert Capability.NODE_ATTRIBUTES in probed
    assert Capability.GRAPH_EDGES in probed
    assert Capability.NODE_EMBEDDING in probed


def test_reconcile_accepts_a_matching_probe(cnn_model_factory):
    model = cnn_model_factory(uncertainty_head="none", uncertainty_dropout_rate=0.3)
    probed = probe_model_capabilities(model)
    # The tiny detector stands in for a graftable, dropout-bearing detector.
    assert reconcile("resnestdf", probed) == []


def test_reconcile_flags_a_stale_profile():
    """If a torchvision upgrade gave ViT non-zero dropout, this must be noticed."""
    probed = frozenset({
        Capability.LOGITS, Capability.PROBABILITIES,
        Capability.LAST_LINEAR_GRAFT, Capability.STOCHASTIC_DROPOUT,
    })
    discrepancies = reconcile("vistransformdf", probed)
    assert discrepancies
    assert "stochastic dropout" in discrepancies[0]


def test_reconcile_raises_in_strict_mode():
    probed = frozenset({Capability.LOGITS, Capability.STOCHASTIC_DROPOUT})
    with pytest.raises(ValueError, match="out of date"):
        reconcile("vistransformdf", probed, strict=True)


def test_reconcile_flags_an_uncatalogued_detector():
    assert reconcile("brand_new_detector", frozenset()) != []
