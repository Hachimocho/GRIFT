"""Deep ensembles as a post-processor over record tables.

An ensemble is not a new inference path here. Each member is an ordinary run that
wrote `records_test.csv.gz`; this module joins those tables and emits another table of
exactly the same schema. So ensembles flow through the same scoring, metric, and
report code as every other method, and a bug in aggregation cannot masquerade as a
calibration result.

Two decisions worth stating, because both are easy to get backwards:

**Average probabilities, not logits.** Logit averaging is a geometric mean of odds.
For a Bernoulli ensemble that is the wrong quantity: it is not the predictive
distribution of the mixture, it is systematically more confident than the mixture, and
it does not equal any member's marginal. `mean(p)` is the mixture's predictive mean by
definition. The recorded `logit` is then recomputed from the averaged probability by
`logit(p̄)`, which is a monotone function of `p̄` and so leaves every rank-based metric
unchanged while keeping the column meaningful.

**Members are discovered from `determinism.json`, not by globbing.** The manifest
records the detector, the uncertainty head, the config, and the git SHA. Globbing
`run_outputs/*/records_test.csv.gz` would happily average a `sngp` member into an
`evidential` ensemble, or fold in a member from a different commit, and the result
would look like a working number.

Uncertainty for the ensemble is the standard decomposition:

    total    = H(p̄)                    predictive entropy of the mixture
    aleatoric= mean_m H(p_m)           average member entropy
    epistemic= total - aleatoric       mutual information / disagreement

plus `u_ens_variance` (variance of member probabilities), which is the direct
disagreement signal and the one to check first: it is *exactly zero* if the members
were not actually initialized differently, which is the failure BatchEnsemble had.
"""

import json
import os

import numpy as np

from evaluation.uq.records import (
    ANNOTATION_PREFIX,
    UNCERTAINTY_PREFIX,
    default_meta_path,
    read_records,
    write_records,
)

#: Columns that must agree exactly across members. A mismatch means the members did
#: not evaluate the same samples in the same order, which makes a row-wise average
#: meaningless -- and silently so, since the shapes would still line up.
JOIN_KEY = "record_id"
ALIGNMENT_COLUMNS = ("rel_path", "node_id", "label", "split")

#: Uncertainty columns this module produces.
ENSEMBLE_UNCERTAINTY = (
    "u_ens_variance",
    "u_ens_entropy",
    "u_ens_entropy_aleatoric",
    "u_ens_mutual_information",
    "u_ens_disagreement",
)

#: Fields in a member's manifest that must match across the ensemble.
COMPATIBILITY_FIELDS = ("detector", "uncertainty_head", "mc_dropout_samples")

_EPSILON = 1e-12


class EnsembleCompatibilityError(ValueError):
    """Raised when members cannot be averaged as if they were one ensemble."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

class Member:
    """One ensemble member: its manifest fields and its records path."""

    __slots__ = ("run_id", "member_index", "description", "records_path", "fingerprint")

    def __init__(self, run_id, member_index, description, records_path, fingerprint):
        self.run_id = run_id
        self.member_index = member_index
        self.description = description
        self.records_path = records_path
        self.fingerprint = fingerprint

    def field(self, name):
        """Read a compatibility field from the per-config results block."""
        results = (self.fingerprint.get("results") or {}).get(self.description, {})
        return results.get(name)

    def __repr__(self):
        return (f"Member(run_id={self.run_id!r}, member={self.member_index}, "
                f"description={self.description!r})")


def discover_members(run_outputs_dir, ensemble_id=None, description=None, split="test"):
    """Find complete ensemble members under ``run_outputs_dir``.

    Reads every `<run_id>/determinism.json` and keeps the runs that finished a
    configuration and wrote records for ``split``. Filtering by ``ensemble_id`` is
    what separates two concurrent ensembles that happen to share an output root.

    Returns members sorted by (member_index, run_id) so aggregation is order-stable
    regardless of directory iteration order.
    """
    members = []
    root = str(run_outputs_dir)
    if not os.path.isdir(root):
        return members

    for run_id in sorted(os.listdir(root)):
        fingerprint_path = os.path.join(root, run_id, "determinism.json")
        if not os.path.exists(fingerprint_path):
            continue
        try:
            with open(fingerprint_path) as handle:
                fingerprint = json.load(handle)
        except (OSError, ValueError):
            continue

        if ensemble_id is not None and fingerprint.get("ensemble_id") != ensemble_id:
            continue

        for config_name, result in sorted((fingerprint.get("results") or {}).items()):
            if description is not None and config_name != description:
                continue
            if not result.get("complete"):
                continue
            records_path = (result.get("records") or {}).get(split)
            if not records_path or not os.path.exists(records_path):
                continue
            members.append(Member(
                run_id=fingerprint.get("run_id", run_id),
                member_index=fingerprint.get("ensemble_member"),
                description=config_name,
                records_path=records_path,
                fingerprint=fingerprint,
            ))

    members.sort(key=lambda entry: (
        entry.member_index if entry.member_index is not None else -1, entry.run_id
    ))
    return members


def assert_members_compatible(members, require_distinct_members=True):
    """Refuse an ensemble whose members are not the same experiment.

    Every check here corresponds to a way the average would be wrong but plausible:
    a mixed head means averaging two different uncertainty parameterizations; a mixed
    detector means averaging two different feature spaces; duplicate member indices
    mean the *same* initialization counted twice, which shrinks apparent disagreement
    toward zero and makes the ensemble look uselessly confident.
    """
    if len(members) < 2:
        raise EnsembleCompatibilityError(
            f"an ensemble needs at least 2 members, got {len(members)}"
        )

    problems = []
    for field in COMPATIBILITY_FIELDS:
        values = {member.field(field) for member in members}
        if len(values) > 1:
            problems.append(f"{field} differs across members: {sorted(map(str, values))}")

    commits = {
        (member.fingerprint.get("git") or {}).get("commit") for member in members
    }
    if len(commits) > 1:
        problems.append(
            f"members were run at {len(commits)} different git commits: "
            f"{sorted(str(commit)[:12] for commit in commits)}"
        )

    modes = {
        ((member.fingerprint.get("determinism") or {}).get("mode"))
        for member in members
    }
    if len(modes) > 1:
        problems.append(f"members mix determinism modes: {sorted(map(str, modes))}")

    seeds = {member.fingerprint.get("seed") for member in members}
    if len(seeds) > 1:
        problems.append(
            f"members used different --seed values {sorted(map(str, seeds))}. Members "
            "should differ in --ensemble-member only: a different seed also changes "
            "the graph and the data order, so disagreement would conflate "
            "initialization variance with data-order variance"
        )

    indices = [member.member_index for member in members]
    if require_distinct_members:
        if any(index is None for index in indices):
            problems.append(
                "at least one member has no --ensemble-member index, so it cannot be "
                "distinguished from another run of the same configuration"
            )
        elif len(set(indices)) != len(indices):
            problems.append(f"duplicate --ensemble-member indices: {sorted(indices)}")

    if problems:
        raise EnsembleCompatibilityError(
            "cannot aggregate these members:\n  - " + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _binary_entropy(probabilities):
    """H(p) in nats for a Bernoulli, safe at p in {0, 1}."""
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _logit(probabilities):
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


def align_frames(frames, labels=None):
    """Check that member tables describe the same samples in the same order.

    Sorts each by ``record_id`` first, so a member written with a different row order
    still aligns -- but a member evaluating a *different sample set* is rejected
    rather than truncated to the intersection. Silently intersecting would change
    which samples the headline number is computed over, per member, invisibly.
    """
    if not frames:
        raise EnsembleCompatibilityError("no member tables to align")

    labels = labels or [f"member{index}" for index in range(len(frames))]
    ordered = [
        frame.sort_values(JOIN_KEY).reset_index(drop=True) for frame in frames
    ]

    reference, reference_label = ordered[0], labels[0]
    reference_ids = reference[JOIN_KEY].to_numpy()
    for frame, label in zip(ordered[1:], labels[1:]):
        if len(frame) != len(reference):
            raise EnsembleCompatibilityError(
                f"{label} has {len(frame)} rows but {reference_label} has "
                f"{len(reference)}; the members did not evaluate the same samples"
            )
        if not np.array_equal(frame[JOIN_KEY].to_numpy(), reference_ids):
            raise EnsembleCompatibilityError(
                f"{label} and {reference_label} disagree on record_id"
            )
        for column in ALIGNMENT_COLUMNS:
            if column not in frame.columns or column not in reference.columns:
                continue
            left = frame[column].to_numpy()
            right = reference[column].to_numpy()
            if not np.array_equal(left, right):
                mismatches = int(np.sum(left != right))
                raise EnsembleCompatibilityError(
                    f"{label} and {reference_label} disagree on {column} for "
                    f"{mismatches} of {len(reference)} rows -- the row order or the "
                    f"evaluation set differs, so averaging would mix samples"
                )
    return ordered


def aggregate_frames(frames, labels=None, method_id="deep_ensemble"):
    """Average member probabilities into one records table.

    Member-specific uncertainty columns (`u_*`) are **dropped**, not averaged: the
    mean of four members' `u_gp_variance` is not any member's variance and is not the
    ensemble's, so it would be a number with no definition. The ensemble's own
    uncertainty columns replace them.
    """
    import pandas as pd

    ordered = align_frames(frames, labels)
    probabilities = np.vstack([
        frame["prob"].to_numpy(dtype=np.float64) for frame in ordered
    ])

    mean_probability = probabilities.mean(axis=0)
    member_entropy = _binary_entropy(probabilities).mean(axis=0)
    total_entropy = _binary_entropy(mean_probability)

    result = ordered[0].copy()
    # Drop every member's own uncertainty columns; keep annotation columns, which
    # describe the *data* and are identical across members.
    for column in list(result.columns):
        if column.startswith(UNCERTAINTY_PREFIX) and not column.startswith(
            f"{UNCERTAINTY_PREFIX}ens_"
        ):
            result = result.drop(columns=[column])

    result["prob"] = mean_probability
    result["logit"] = _logit(mean_probability)
    result["pred"] = (mean_probability > 0.5).astype(np.float64)
    result["correct"] = (
        result["pred"].to_numpy() == result["label"].to_numpy()
    ).astype(np.float64)
    # Recomputed rather than averaged: the per-sample loss of the ensemble is the loss
    # of its averaged prediction, not the average of the members' losses (which is an
    # upper bound on it by Jensen).
    labels_array = result["label"].to_numpy(dtype=np.float64)
    clipped = np.clip(mean_probability, _EPSILON, 1.0 - _EPSILON)
    result["loss_sample"] = -(
        labels_array * np.log(clipped) + (1.0 - labels_array) * np.log(1.0 - clipped)
    )

    result["u_ens_variance"] = probabilities.var(axis=0, ddof=0)
    result["u_ens_entropy"] = total_entropy
    result["u_ens_entropy_aleatoric"] = member_entropy
    # Mutual information is non-negative in exact arithmetic (Jensen), but floating
    # point can produce ~-1e-17 when the members agree exactly. Clip rather than let a
    # negative "epistemic uncertainty" reach a plot.
    result["u_ens_mutual_information"] = np.maximum(
        total_entropy - member_entropy, 0.0
    )
    result["u_ens_disagreement"] = probabilities.max(axis=0) - probabilities.min(axis=0)

    result["method_id"] = method_id
    result["n_members"] = len(ordered)

    # Stable column order: originals in place, ensemble columns appended in a fixed
    # order, so two aggregations of the same members are byte-identical.
    columns = [
        column for column in result.columns
        if column not in ENSEMBLE_UNCERTAINTY and column not in ("method_id", "n_members")
    ]
    columns += list(ENSEMBLE_UNCERTAINTY) + ["method_id", "n_members"]
    return pd.DataFrame(result[columns])


def aggregate_members(members, require_distinct_members=True, verify=True,
                      method_id="deep_ensemble"):
    """Discovery -> compatibility -> aggregation. Returns ``(frame, manifest)``."""
    assert_members_compatible(members, require_distinct_members=require_distinct_members)
    frames = [read_records(member.records_path, verify=verify) for member in members]
    labels = [f"member{member.member_index}({member.run_id})" for member in members]
    frame = aggregate_frames(frames, labels=labels, method_id=method_id)

    reference = members[0]
    manifest = {
        "method_id": method_id,
        "n_members": len(members),
        "members": [
            {
                "run_id": member.run_id,
                "ensemble_member": member.member_index,
                "description": member.description,
                "records_path": member.records_path,
                "checkpoint": member.field("checkpoint"),
                "best_epoch": member.field("best_epoch"),
                "test_accuracy": member.field("test_accuracy"),
            }
            for member in members
        ],
        "detector": reference.field("detector"),
        "uncertainty_head": reference.field("uncertainty_head"),
        "ensemble_id": reference.fingerprint.get("ensemble_id"),
        "seed": reference.fingerprint.get("seed"),
        "git_commit": (reference.fingerprint.get("git") or {}).get("commit"),
        # A table where ensembles win without showing 5x the training cost is
        # misleading, so the cost travels with the result.
        "cost_training_runs": len(members),
        "cost_forward_passes": len(members),
        "mean_disagreement": float(frame["u_ens_variance"].mean()),
        "max_disagreement": float(frame["u_ens_variance"].max()),
    }
    return frame, manifest


def save_ensemble(frame, manifest, records_path):
    """Write the aggregated table plus its manifest. Returns the sha256."""
    digest = write_records(frame, records_path)
    payload = dict(manifest)
    payload["sha256_records"] = digest
    payload["records_path"] = os.path.basename(str(records_path))
    payload["n_rows"] = int(len(frame))
    # An aggregate over complete member tables is complete by construction; recorded
    # explicitly so the report's coverage guard has a value to check.
    payload["coverage"] = 1.0

    meta_path = default_meta_path(records_path)
    temporary = f"{meta_path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, meta_path)
    return digest


def annotation_columns(frame):
    """Annotation columns present, for tests asserting they are never scored."""
    return [
        column for column in frame.columns
        if column.startswith(ANNOTATION_PREFIX)
    ]
