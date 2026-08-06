"""Which uncertainty methods the benchmark can score, and on which detectors.

This is where the "model-agnostic framework" requirement is enforced. Each method
declares the capabilities it needs; each detector declares (and is probed for) the
capabilities it offers; an incompatible pairing is **skipped with a logged, machine-
readable reason** rather than crashed or silently scored as garbage. The published
matrix therefore shows explicit reasoned holes instead of missing rows.

Two honesty mechanisms are built in deliberately.

``rank_equivalent_to``
    For a single Bernoulli, predictive entropy is a monotone function of
    ``max(p, 1-p)``, and temperature scaling is a monotone rescaling of the logit.
    So those methods have *mathematically identical* AUROC-error, AUPR-error, AURC,
    and accuracy@coverage to the max-probability baseline. Reporting them as
    separate ranking results would show three columns of identical numbers and
    invite the reader to conclude something false. Temperature scaling's entire
    contribution is calibration (ECE/NLL/Brier), and that is where it is credited.

``graph_degree_only``
    ``degree_penalty`` alone, as the ablation control for the graph-distance methods.
    The distances are computed over a node's neighbors, so "graph distance predicts
    error" could just as easily be "low-degree nodes are harder". Without this
    control that confound is unanswerable -- and since graph distance is the novel
    contribution here, it is the first thing a reviewer will ask about.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from models.uncertainty.capabilities import (
    BROKEN, Capability, probe_model_capabilities, profile_for,
)

# Method families, used to facet reporting and plots.
FAMILY_LOGIT = "logit"
FAMILY_POSTHOC = "posthoc"
FAMILY_SAMPLING = "sampling"
FAMILY_HEAD = "head"
FAMILY_ENSEMBLE = "ensemble"
FAMILY_GRAPH = "graph"


@dataclass(frozen=True)
class UQMethodSpec:
    """One scoreable uncertainty method."""

    method_id: str
    display_name: str
    family: str
    requires: frozenset
    uncertainty_columns: Tuple[str, ...]
    primary_column: str
    produces_probabilities: bool
    model_agnostic: bool
    cost_forward_passes: object = 1
    cost_training_runs: int = 0
    needs_training_config: Dict[str, object] = field(default_factory=dict)
    penultimate_space_sensitive: bool = False
    rank_equivalent_to: Optional[str] = None
    notes: str = ""


_LOGITS = frozenset({Capability.LOGITS, Capability.PROBABILITIES})

UQ_METHODS = {
    "baseline_maxprob": UQMethodSpec(
        method_id="baseline_maxprob", display_name="Max probability",
        family=FAMILY_LOGIT, requires=_LOGITS,
        uncertainty_columns=("u_maxprob",), primary_column="u_maxprob",
        produces_probabilities=True, model_agnostic=True,
        notes="Zero-cost reference point. Any method that cannot beat this is not "
              "earning its complexity.",
    ),
    "baseline_entropy": UQMethodSpec(
        method_id="baseline_entropy", display_name="Predictive entropy",
        family=FAMILY_LOGIT, requires=_LOGITS,
        uncertainty_columns=("u_entropy",), primary_column="u_entropy",
        produces_probabilities=True, model_agnostic=True,
        rank_equivalent_to="baseline_maxprob",
        notes="Monotone in max(p, 1-p) for a Bernoulli, so all ranking metrics are "
              "identical to baseline_maxprob by construction.",
    ),
    "baseline_margin": UQMethodSpec(
        method_id="baseline_margin", display_name="Decision margin",
        family=FAMILY_LOGIT, requires=_LOGITS,
        uncertainty_columns=("u_margin",), primary_column="u_margin",
        produces_probabilities=True, model_agnostic=True,
        rank_equivalent_to="baseline_maxprob",
        notes="1 - |2p - 1|; also monotone in max(p, 1-p).",
    ),
    "temperature_scaling": UQMethodSpec(
        method_id="temperature_scaling", display_name="Temperature scaling",
        family=FAMILY_POSTHOC, requires=_LOGITS | frozenset({Capability.VAL_SPLIT}),
        uncertainty_columns=("u_temp_maxprob",), primary_column="u_temp_maxprob",
        produces_probabilities=True, model_agnostic=True,
        rank_equivalent_to="baseline_maxprob",
        notes="A single scalar fitted on val. Being a monotone logit rescaling, it "
              "cannot change any ranking metric -- its entire contribution is "
              "calibration.",
    ),
    "mc_dropout": UQMethodSpec(
        method_id="mc_dropout", display_name="MC dropout",
        family=FAMILY_SAMPLING,
        requires=_LOGITS | frozenset({Capability.STOCHASTIC_DROPOUT}),
        uncertainty_columns=(
            "u_mc_dropout_variance", "u_mc_dropout_entropy",
            "u_mc_dropout_mutual_information", "u_mc_dropout_variation_ratio",
        ),
        primary_column="u_mc_dropout_mutual_information",
        produces_probabilities=True, model_agnostic=False,
        cost_forward_passes="mc_dropout_samples",
        notes="Requires at least one nn.Dropout with p>0. vistransformdf and "
              "swintransformdf backbones have none (all p=0.0), so they need a head "
              "that supplies dropout; squeezenetdf works natively.",
    ),
    "evidential": UQMethodSpec(
        method_id="evidential", display_name="Evidential (Dirichlet)",
        family=FAMILY_HEAD,
        requires=_LOGITS | frozenset({Capability.LAST_LINEAR_GRAFT, Capability.EVIDENCE_ALPHA}),
        uncertainty_columns=("u_evidential_vacuity", "u_evidential_total_evidence"),
        primary_column="u_evidential_vacuity",
        produces_probabilities=True, model_agnostic=False, cost_training_runs=1,
        needs_training_config={"uncertainty_head": "evidential"},
        penultimate_space_sensitive=True,
    ),
    "batchensemble": UQMethodSpec(
        method_id="batchensemble", display_name="BatchEnsemble",
        family=FAMILY_HEAD,
        requires=_LOGITS | frozenset({Capability.LAST_LINEAR_GRAFT, Capability.MEMBER_LOGITS}),
        uncertainty_columns=("u_batchensemble_variance",),
        primary_column="u_batchensemble_variance",
        produces_probabilities=True, model_agnostic=False, cost_training_runs=1,
        needs_training_config={"uncertainty_head": "batchensemble"},
        penultimate_space_sensitive=True,
    ),
    "sngp": UQMethodSpec(
        method_id="sngp", display_name="SNGP",
        family=FAMILY_HEAD,
        requires=_LOGITS | frozenset({Capability.LAST_LINEAR_GRAFT, Capability.GP_VARIANCE}),
        uncertainty_columns=("u_sngp_variance",), primary_column="u_sngp_variance",
        produces_probabilities=True, model_agnostic=False, cost_training_runs=1,
        needs_training_config={"uncertainty_head": "sngp"},
        penultimate_space_sensitive=True,
        notes="Comparable across epochs only with --sngp-precision-policy per-epoch.",
    ),
    "deep_ensemble": UQMethodSpec(
        method_id="deep_ensemble", display_name="Deep ensemble",
        family=FAMILY_ENSEMBLE,
        requires=_LOGITS | frozenset({Capability.MULTI_CHECKPOINT}),
        uncertainty_columns=(
            "u_ens_variance", "u_ens_entropy", "u_ens_mutual_information",
            "u_ens_disagreement",
        ),
        primary_column="u_ens_mutual_information",
        produces_probabilities=True, model_agnostic=True,
        cost_forward_passes="n_members", cost_training_runs=5,
        notes="Averages probabilities, not logits. On effnetdf/swintransformdf with "
              "--finetune this measures head-initialization variance only, because "
              "those detectors freeze their backbone.",
    ),
    "graph_attribute_distance": UQMethodSpec(
        method_id="graph_attribute_distance", display_name="Graph attribute distance",
        family=FAMILY_GRAPH,
        requires=frozenset({Capability.NODE_ATTRIBUTES, Capability.GRAPH_EDGES}),
        uncertainty_columns=("u_attribute_distance",), primary_column="u_attribute_distance",
        produces_probabilities=False, model_agnostic=True, cost_forward_passes=0,
    ),
    "graph_embedding_distance": UQMethodSpec(
        method_id="graph_embedding_distance", display_name="Graph embedding distance",
        family=FAMILY_GRAPH,
        requires=frozenset({
            Capability.NODE_ATTRIBUTES, Capability.NODE_EMBEDDING, Capability.GRAPH_EDGES,
        }),
        uncertainty_columns=("u_embedding_distance",), primary_column="u_embedding_distance",
        produces_probabilities=False, model_agnostic=True, cost_forward_passes=0,
        notes="Needs face_embedding coverage. Missing embeddings fall back to a flat "
              "sentinel, which fabricates a bimodal score distribution.",
    ),
    "graph_hybrid_distance": UQMethodSpec(
        method_id="graph_hybrid_distance", display_name="Graph hybrid distance",
        family=FAMILY_GRAPH,
        requires=frozenset({Capability.NODE_ATTRIBUTES, Capability.GRAPH_EDGES}),
        uncertainty_columns=("u_hybrid_distance",), primary_column="u_hybrid_distance",
        produces_probabilities=False, model_agnostic=True, cost_forward_passes=0,
    ),
    "graph_degree_only": UQMethodSpec(
        method_id="graph_degree_only", display_name="Degree penalty only (control)",
        family=FAMILY_GRAPH, requires=frozenset({Capability.GRAPH_EDGES}),
        uncertainty_columns=("u_degree_penalty",), primary_column="u_degree_penalty",
        produces_probabilities=False, model_agnostic=True, cost_forward_passes=0,
        notes="Ablation control. The distance methods aggregate over a node's "
              "neighbors, so without this you cannot separate 'attributes differ from "
              "neighbors' from 'this node has few neighbors'.",
    ),
}

#: Score columns that must never be treated as predictive uncertainty.
#: AI-Face ships per-sample annotation-uncertainty scores for its demographic
#: labels. They are a property of the *labels*, not of any model, and conflating
#: them with a UQ method would be a category error -- so they are namespaced
#: `anno_` and rejected here.
FORBIDDEN_SCORE_PREFIXES = ("anno_",)


@dataclass(frozen=True)
class GateDecision:
    method_id: str
    detector: str
    compatible: bool
    severity: str  # "ok" | "skip" | "broken"
    missing: frozenset = frozenset()
    reasons: Tuple[str, ...] = ()

    def format(self):
        tag = {"ok": "OK", "skip": "SKIP", "broken": "BROKEN"}[self.severity]
        line = f"[UQ-GATE][{tag}] method={self.method_id} detector={self.detector}"
        if self.missing:
            line += f" missing={{{', '.join(sorted(self.missing))}}}"
        for reason in self.reasons:
            line += f"\n  reason: {reason}"
        return line


def method_spec(method_id):
    spec = UQ_METHODS.get(method_id)
    if spec is None:
        raise KeyError(
            f"unknown UQ method {method_id!r}; registered methods are {sorted(UQ_METHODS)}"
        )
    return spec


def methods_by_family(family):
    return sorted(
        spec.method_id for spec in UQ_METHODS.values() if spec.family == family
    )


def rank_equivalence_groups():
    """Map representative method -> methods whose ranking metrics it determines."""
    groups = {}
    for spec in UQ_METHODS.values():
        if spec.rank_equivalent_to:
            groups.setdefault(spec.rank_equivalent_to, []).append(spec.method_id)
    return {key: sorted(value) for key, value in groups.items()}


def validate_score_column(column):
    """Reject a column that must not be scored as predictive uncertainty."""
    for prefix in FORBIDDEN_SCORE_PREFIXES:
        if column.startswith(prefix):
            raise ValueError(
                f"{column!r} is annotation uncertainty (a property of the dataset's "
                f"demographic labels), not predictive model uncertainty. It must not "
                f"be registered or scored as a UQ method."
            )
    return column


def gate(method_id, detector, probed=None, config=None):
    """Decide whether ``method_id`` can be scored on ``detector``."""
    spec = method_spec(method_id)
    profile = profile_for(detector)
    config = config or {}

    if profile is None:
        return GateDecision(
            method_id, detector, False, "broken",
            reasons=(f"detector {detector!r} is not catalogued in DETECTOR_PROFILES",),
        )
    if profile.status == BROKEN:
        return GateDecision(
            method_id, detector, False, "broken", reasons=(profile.broken_reason,),
        )

    available = set(profile.static_capabilities)
    if probed is not None:
        available |= set(probed)

    # Capabilities that come from configuration rather than from the model.
    if config.get("has_val_split", True):
        available.add(Capability.VAL_SPLIT)
    if int(config.get("n_members", 0)) > 1:
        available.add(Capability.MULTI_CHECKPOINT)
    if config.get("graph_edges_available"):
        available.add(Capability.GRAPH_EDGES)
        available.add(Capability.NODE_ATTRIBUTES)
    if config.get("embedding_coverage", 0.0) >= 0.5:
        available.add(Capability.NODE_EMBEDDING)

    # A head-based method supplies its own dropout, so it satisfies the dropout
    # requirement for itself even on a backbone that has none of its own.
    head = config.get("uncertainty_head", "none")
    if head != "none" and float(config.get("uncertainty_dropout_rate", 0.0)) > 0.0:
        available.add(Capability.STOCHASTIC_DROPOUT)

    # Plan-time vs runtime gating.
    #
    # `gate()` decides whether to *schedule* a cell, which happens before the model
    # for it has been trained. A head method's own outputs (alpha, member_logits,
    # gp_variance) therefore cannot be observed yet -- they are satisfied by
    # construction, because the harness will train this cell with that head. What
    # still has to hold is that the detector can host the head at all, which is the
    # LAST_LINEAR_GRAFT requirement checked below.
    #
    # `gate_model()` passes a real capability probe and so remains the strict
    # after-the-fact check; use it to verify a trained model really produces what
    # its method claims.
    if spec.needs_training_config and not config.get("strict_runtime", False):
        provided_by_own_head = {
            Capability.EVIDENCE_ALPHA, Capability.MEMBER_LOGITS,
            Capability.GP_VARIANCE, Capability.STOCHASTIC_DROPOUT,
        }
        available |= (set(spec.requires) & provided_by_own_head)

    missing = frozenset(spec.requires) - available
    if not missing:
        return GateDecision(method_id, detector, True, "ok")

    return GateDecision(
        method_id, detector, False, "skip", missing=missing,
        reasons=(_explain(spec, profile, missing, config),),
    )


def _explain(spec, profile, missing, config):
    """Human-readable, remedy-bearing reason for a skip."""
    if Capability.LAST_LINEAR_GRAFT in missing:
        return (
            f"{profile.name} exposes no nn.Linear to graft onto "
            f"(its classifier is convolutional), so _find_last_linear returns "
            f"(None, None) and CNNModel raises ValueError. Logit-space methods "
            f"(baseline_maxprob, temperature_scaling, deep_ensemble) still apply."
        )
    if Capability.STOCHASTIC_DROPOUT in missing:
        rate = config.get("uncertainty_dropout_rate", 0.0)
        return (
            f"{profile.name} has {profile.dropout_sites_head_none} dropout module(s) "
            f"with p>0 at uncertainty_dropout_rate={rate}. MC dropout on a network "
            f"whose dropout is all p=0 returns identically zero variance, which is a "
            f"silently wrong measurement rather than an error. Remedies: attach an "
            f"uncertainty head with --uncertainty-dropout-rate>0, or use a detector "
            f"with real dropout (squeezenetdf)."
        )
    if Capability.NODE_EMBEDDING in missing:
        coverage = config.get("embedding_coverage")
        return (
            f"face_embedding coverage is "
            f"{'unknown' if coverage is None else f'{coverage:.1%}'}; below 50% the "
            f"score is dominated by the missing-value sentinel, which fabricates a "
            f"bimodal distribution that reads like signal."
        )
    if Capability.MULTI_CHECKPOINT in missing:
        return (
            f"deep ensembles need several independently-seeded checkpoints; "
            f"n_members={config.get('n_members', 0)}."
        )
    if Capability.GRAPH_EDGES in missing:
        return (
            "graph-based uncertainty needs an edge-bearing graph for this split. "
            "Ensure --build-val-test-edges is enabled and a fitted "
            "GraphDistanceUncertainty is attached."
        )
    return f"{profile.name} lacks required capabilities {sorted(missing)}"


def expand_matrix(methods, detectors, probed_map=None, config=None):
    """Expand a (methods x detectors) matrix into runnable cells plus decisions.

    Returns ``(cells, decisions)`` where cells is the compatible subset. Every pair
    appears in ``decisions``, so the report can render skips as explicit holes.
    """
    probed_map = probed_map or {}
    cells, decisions = [], []
    for detector in detectors:
        for method_id in methods:
            decision = gate(
                method_id, detector,
                probed=probed_map.get(detector), config=(config or {}).get(detector),
            )
            decisions.append(decision)
            if decision.compatible:
                cells.append({"method_id": method_id, "detector": detector})
    return cells, decisions


def gate_model(method_id, detector, model, config=None):
    """Gate against a *constructed* model, using its runtime capability probe.

    Stricter than plan-time ``gate()``: a head method's own outputs must actually be
    observable on this model, not merely promised by the training config. Use this to
    confirm a trained checkpoint really produces what its method claims before
    scoring it.
    """
    runtime_config = dict(config or {})
    runtime_config["strict_runtime"] = True
    return gate(
        method_id, detector,
        probed=probe_model_capabilities(model), config=runtime_config,
    )


def comparable_detector_groups(detectors):
    """Group detectors by penultimate space.

    Feature-space methods must not be compared across groups:
    effnetdf/resnestdf/swintransformdf expose a 1024-d post-dropout head activation,
    vistransformdf a 768-d ViT CLS token, and squeezenetdf nothing at all. A single
    cross-detector number over those would be meaningless.
    """
    groups = {}
    for detector in detectors:
        profile = profile_for(detector)
        key = profile.penultimate_space if profile else None
        groups.setdefault(key, []).append(detector)
    return {key: sorted(value) for key, value in groups.items()}
