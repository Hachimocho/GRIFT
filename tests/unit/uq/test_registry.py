"""Method registry and capability gating.

The gate is the concrete realization of the model-agnostic requirement: an
incompatible (method, detector) pair must be skipped with a reason, never crashed
and never silently scored.
"""

import pytest

from evaluation.uq.registry import (
    FAMILY_GRAPH, FAMILY_HEAD, FAMILY_LOGIT, FORBIDDEN_SCORE_PREFIXES, UQ_METHODS,
    comparable_detector_groups, expand_matrix, gate, gate_model, method_spec,
    methods_by_family, rank_equivalence_groups, validate_score_column,
)
from models.uncertainty.capabilities import Capability, broken_detectors, supported_detectors

ALL_METHODS = sorted(UQ_METHODS)


# --------------------------------------------------------------------------- #
# Registry integrity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method_id", ALL_METHODS)
def test_primary_column_is_one_of_the_declared_columns(method_id):
    spec = method_spec(method_id)
    assert spec.primary_column in spec.uncertainty_columns


@pytest.mark.parametrize("method_id", ALL_METHODS)
def test_every_method_declares_requirements_and_a_family(method_id):
    spec = method_spec(method_id)
    assert spec.requires, f"{method_id} declares no requirements"
    assert spec.requires <= Capability.ALL, (
        f"{method_id} requires unknown capabilities "
        f"{sorted(spec.requires - Capability.ALL)}"
    )
    assert spec.family
    assert spec.display_name


@pytest.mark.parametrize("method_id", ALL_METHODS)
def test_no_method_scores_an_annotation_column(method_id):
    """AI-Face's annotation uncertainty must never be a UQ method."""
    for column in method_spec(method_id).uncertainty_columns:
        for prefix in FORBIDDEN_SCORE_PREFIXES:
            assert not column.startswith(prefix), (
                f"{method_id} registers {column!r}, which is annotation uncertainty"
            )


def test_validate_score_column_rejects_annotation_uncertainty():
    assert validate_score_column("u_sngp_variance") == "u_sngp_variance"
    with pytest.raises(ValueError, match="annotation uncertainty"):
        validate_score_column("anno_unc_race")


def test_graph_methods_are_marked_non_probabilistic():
    """Their calibration cells must be N/A, not zero."""
    for method_id in methods_by_family(FAMILY_GRAPH):
        assert method_spec(method_id).produces_probabilities is False


def test_graph_methods_are_model_agnostic():
    """They read node attributes and topology, never the network."""
    for method_id in methods_by_family(FAMILY_GRAPH):
        spec = method_spec(method_id)
        assert spec.model_agnostic is True
        assert spec.cost_forward_passes == 0


def test_head_methods_are_not_model_agnostic():
    for method_id in methods_by_family(FAMILY_HEAD):
        spec = method_spec(method_id)
        assert spec.model_agnostic is False
        assert Capability.LAST_LINEAR_GRAFT in spec.requires
        assert spec.cost_training_runs >= 1


def test_the_degree_only_control_exists():
    """Without it, 'graph distance predicts error' is confounded with node degree."""
    spec = method_spec("graph_degree_only")
    assert spec.requires == frozenset({Capability.GRAPH_EDGES})
    assert "control" in spec.display_name.lower()


def test_rank_equivalent_methods_are_declared():
    """Entropy, margin, and temperature scaling cannot differ from max-prob on ranks.

    All three are monotone functions of max(p, 1-p) for a Bernoulli, so reporting
    them as distinct ranking results would print identical columns and imply a
    difference that does not exist.
    """
    groups = rank_equivalence_groups()
    assert set(groups["baseline_maxprob"]) == {
        "baseline_entropy", "baseline_margin", "temperature_scaling",
    }


def test_rank_equivalence_is_numerically_true():
    """Verify the declaration rather than trusting it."""
    import numpy as np
    from evaluation.uq.metrics import (
        score_entropy, score_margin, score_max_probability, uncertainty_error_auroc,
    )

    rng = np.random.Generator(np.random.PCG64(5))
    y_true = rng.integers(0, 2, size=200)
    probabilities = np.clip(y_true * 0.6 + 0.2 + rng.normal(0, 0.25, size=200), 0.01, 0.99)

    baseline = uncertainty_error_auroc(
        y_true, probabilities, score_max_probability(probabilities)
    ).value
    for score in (score_entropy(probabilities), score_margin(probabilities)):
        assert uncertainty_error_auroc(y_true, probabilities, score).value == pytest.approx(
            baseline, abs=1e-12
        )
    # And temperature scaling: a monotone logit rescaling.
    logits = np.log(probabilities / (1 - probabilities))
    tempered = 1.0 / (1.0 + np.exp(-logits / 1.7))
    assert uncertainty_error_auroc(
        y_true, tempered, score_max_probability(tempered)
    ).value == pytest.approx(baseline, abs=1e-12)


def test_temperature_scaling_needs_a_val_split():
    assert Capability.VAL_SPLIT in method_spec("temperature_scaling").requires


def test_deep_ensemble_declares_its_cost():
    spec = method_spec("deep_ensemble")
    assert spec.cost_training_runs >= 3, "an ensemble table must show its training cost"
    assert Capability.MULTI_CHECKPOINT in spec.requires


def test_unknown_method_raises():
    with pytest.raises(KeyError, match="unknown UQ method"):
        method_spec("does_not_exist")


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #

def test_logit_methods_work_on_every_usable_detector():
    """Including the logit-only one -- that is what 'model-agnostic' should mean."""
    for detector in supported_detectors():
        for method_id in methods_by_family(FAMILY_LOGIT):
            decision = gate(method_id, detector)
            assert decision.compatible, decision.format()


@pytest.mark.parametrize("method_id", ["evidential", "batchensemble", "sngp"])
def test_heads_are_skipped_on_a_detector_with_no_linear(method_id):
    decision = gate(method_id, "squeezenetdf")
    assert not decision.compatible
    assert decision.severity == "skip"
    assert Capability.LAST_LINEAR_GRAFT in decision.missing
    assert "no nn.Linear" in decision.reasons[0]
    assert "Logit-space methods" in decision.reasons[0], "the reason should offer a remedy"


def test_mc_dropout_is_skipped_on_a_zero_dropout_backbone():
    """vistransformdf's 37 nn.Dropout modules are all p=0.0."""
    decision = gate("mc_dropout", "vistransformdf", config={"uncertainty_head": "none"})
    assert not decision.compatible
    assert Capability.STOCHASTIC_DROPOUT in decision.missing
    assert "identically zero variance" in decision.reasons[0]


def test_mc_dropout_is_allowed_once_a_head_supplies_dropout():
    decision = gate(
        "mc_dropout", "vistransformdf",
        config={"uncertainty_head": "sngp", "uncertainty_dropout_rate": 0.2},
    )
    assert decision.compatible, decision.format()


def test_mc_dropout_works_natively_on_squeezenetdf():
    """The detector that cannot host a head is the one with real dropout."""
    decision = gate("mc_dropout", "squeezenetdf", config={"uncertainty_head": "none"})
    assert decision.compatible, decision.format()


@pytest.mark.parametrize("detector", sorted(broken_detectors()))
@pytest.mark.parametrize("method_id", ["baseline_maxprob", "sngp"])
def test_broken_detectors_are_reported_as_broken(detector, method_id):
    decision = gate(method_id, detector)
    assert not decision.compatible
    assert decision.severity == "broken"
    assert decision.reasons and len(decision.reasons[0]) > 20


def test_uncatalogued_detector_is_reported():
    decision = gate("baseline_maxprob", "brand_new_detector")
    assert decision.severity == "broken"
    assert "not catalogued" in decision.reasons[0]


def test_graph_methods_require_an_edge_bearing_graph():
    without = gate("graph_hybrid_distance", "effnetdf", config={})
    assert not without.compatible
    assert Capability.GRAPH_EDGES in without.missing

    with_edges = gate(
        "graph_hybrid_distance", "effnetdf", config={"graph_edges_available": True}
    )
    assert with_edges.compatible, with_edges.format()


def test_embedding_distance_requires_coverage():
    low = gate(
        "graph_embedding_distance", "effnetdf",
        config={"graph_edges_available": True, "embedding_coverage": 0.1},
    )
    assert not low.compatible
    assert Capability.NODE_EMBEDDING in low.missing
    assert "10.0%" in low.reasons[0]

    high = gate(
        "graph_embedding_distance", "effnetdf",
        config={"graph_edges_available": True, "embedding_coverage": 0.95},
    )
    assert high.compatible, high.format()


def test_degree_only_control_needs_nothing_but_edges():
    decision = gate(
        "graph_degree_only", "squeezenetdf", config={"graph_edges_available": True}
    )
    assert decision.compatible, "the control must run wherever the distances do"


def test_deep_ensemble_requires_multiple_members():
    assert not gate("deep_ensemble", "effnetdf", config={"n_members": 1}).compatible
    assert gate("deep_ensemble", "effnetdf", config={"n_members": 5}).compatible


def test_gate_decision_formats_a_readable_line():
    decision = gate("sngp", "squeezenetdf")
    text = decision.format()
    assert "[UQ-GATE][SKIP]" in text
    assert "method=sngp" in text and "detector=squeezenetdf" in text
    assert "last_linear_graft" in text


# --------------------------------------------------------------------------- #
# Matrix expansion
# --------------------------------------------------------------------------- #

def test_expand_matrix_records_every_pair():
    methods = ["baseline_maxprob", "sngp", "graph_degree_only"]
    detectors = ["effnetdf", "squeezenetdf", "dag_fdd"]
    cells, decisions = expand_matrix(methods, detectors, config={
        detector: {"graph_edges_available": True} for detector in detectors
    })

    assert len(decisions) == len(methods) * len(detectors)
    assert all(cell["method_id"] in methods for cell in cells)
    # Every skip must carry a reason, so the published matrix has explained holes.
    for decision in decisions:
        if not decision.compatible:
            assert decision.reasons, decision.format()


def test_expand_matrix_golden_counts():
    """Pins the compatible/skipped split so a regression in gating is visible.

    Plan-time gating: `uncertainty_head='none'` describes the *baseline* config, and a
    head method is understood to bring its own head when the harness schedules it.
    What still has to hold is that the detector can host one at all.
    """
    methods = ["baseline_maxprob", "evidential", "sngp", "mc_dropout"]
    detectors = ["effnetdf", "vistransformdf", "squeezenetdf", "dag_fdd"]
    config = {detector: {"uncertainty_head": "none"} for detector in detectors}
    cells, decisions = expand_matrix(methods, detectors, config=config)

    compatible = {(cell["detector"], cell["method_id"]) for cell in cells}
    # dag_fdd is broken outright.
    assert not any(detector == "dag_fdd" for detector, _ in compatible)
    # squeezenetdf takes logit methods and MC dropout, but no grafted head.
    assert ("squeezenetdf", "baseline_maxprob") in compatible
    assert ("squeezenetdf", "mc_dropout") in compatible
    assert ("squeezenetdf", "evidential") not in compatible
    # vistransformdf takes heads but not bare MC dropout.
    assert ("vistransformdf", "evidential") in compatible
    assert ("vistransformdf", "mc_dropout") not in compatible

    broken = [d for d in decisions if d.severity == "broken"]
    assert len(broken) == len(methods), "every method on dag_fdd should be 'broken'"


# --------------------------------------------------------------------------- #
# Runtime gating
# --------------------------------------------------------------------------- #

def test_gate_model_uses_the_runtime_probe(cnn_model_factory):
    """The probe supplies capabilities the static table cannot know."""
    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.3)
    decision = gate_model("sngp", "resnestdf", model)
    assert decision.compatible, decision.format()


def test_gate_model_rejects_a_head_method_on_a_headless_model(cnn_model_factory):
    """Runtime gating is strict: the head's outputs must actually be observable."""
    model = cnn_model_factory(uncertainty_head="none")
    decision = gate_model("sngp", "resnestdf", model, config={"uncertainty_head": "none"})
    assert not decision.compatible
    assert Capability.GP_VARIANCE in decision.missing


def test_plan_time_gating_is_permissive_about_a_methods_own_head():
    """A head method is schedulable before its head has been trained.

    Plan-time gating decides what to *run*; requiring the head's outputs to already
    exist would make it impossible to schedule any head method at all.
    """
    planned = gate("sngp", "effnetdf", config={"uncertainty_head": "none"})
    assert planned.compatible, planned.format()


def test_plan_time_gating_still_requires_a_graftable_detector():
    """Permissive about the head, not about whether one can be attached."""
    decision = gate("sngp", "squeezenetdf", config={"uncertainty_head": "none"})
    assert not decision.compatible
    assert Capability.LAST_LINEAR_GRAFT in decision.missing


def test_strict_runtime_flag_disables_the_plan_time_allowance():
    strict = gate(
        "sngp", "effnetdf",
        config={"uncertainty_head": "none", "strict_runtime": True},
    )
    assert not strict.compatible
    assert Capability.GP_VARIANCE in strict.missing


# --------------------------------------------------------------------------- #
# Penultimate-space comparability
# --------------------------------------------------------------------------- #

def test_penultimate_spaces_split_the_detectors():
    groups = comparable_detector_groups(supported_detectors())
    assert len(groups) > 1, (
        "feature-space methods cannot be compared across all detectors, so the "
        "grouping must be non-trivial"
    )
    head_group = groups.get("head1024_postdrop", [])
    assert set(head_group) == {"effnetdf", "resnestdf", "swintransformdf"}
    assert groups.get("vit_cls768") == ["vistransformdf"]
    assert groups.get(None) == ["squeezenetdf"]


def test_head_methods_are_flagged_penultimate_sensitive():
    """So the report knows not to pool them across incomparable spaces."""
    for method_id in methods_by_family(FAMILY_HEAD):
        assert method_spec(method_id).penultimate_space_sensitive is True
    assert method_spec("baseline_maxprob").penultimate_space_sensitive is False
