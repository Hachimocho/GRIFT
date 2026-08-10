"""Deep-ensemble aggregation: `evaluation/uq/ensemble.py`.

The aggregator is the one place where a bug produces numbers that look entirely
reasonable. Averaging logits instead of probabilities, or intersecting mismatched
member tables instead of refusing them, yields a well-formed table with plausible
calibration -- so every guard here exists because its absence would be invisible.

No torch, no model, no GPU: members are synthetic record tables, which is what makes
these fast-tier tests.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from evaluation.uq.ensemble import (
    EnsembleCompatibilityError,
    Member,
    aggregate_frames,
    aggregate_members,
    align_frames,
    assert_members_compatible,
    discover_members,
    save_ensemble,
)
from evaluation.uq.records import (
    default_meta_path,
    read_records,
    write_manifest,
    write_records,
)

N_ROWS = 12


def make_frame(probabilities, labels=None, uncertainty=None, **overrides):
    """A record table with the columns the aggregator reads."""
    count = len(probabilities)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels is None:
        labels = np.array([index % 2 for index in range(count)], dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    # `prob` keeps its exact value (including 0.0 and 1.0 -- the aggregator has to
    # cope with those), but the derived columns are computed on a clipped copy so the
    # *fixture* never contains inf or nan. Otherwise a test would pass on a table no
    # real collector could produce.
    safe = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    frame = pd.DataFrame({
        "record_id": np.arange(count),
        "rel_path": [f"/FFHQ/{index:05d}.png" for index in range(count)],
        "node_id": [f"/root/FFHQ/{index:05d}.png" for index in range(count)],
        "source_top": "FFHQ",
        "source_group": "FFHQ",
        "split": "test",
        "domain": "id",
        "corruption": "none",
        "severity": 0,
        "graph_degree": 3,
        "label": labels,
        "pred": (probabilities > 0.5).astype(np.float64),
        "correct": ((probabilities > 0.5).astype(np.float64) == labels).astype(np.float64),
        "logit": np.log(safe / (1.0 - safe)),
        "prob": probabilities,
        "loss_sample": -(labels * np.log(safe) + (1 - labels) * np.log(1 - safe)),
        "anno_unc_gender": 0.25,
        "gt_gender": 0,
    })
    for name, values in (uncertainty or {}).items():
        frame[name] = values
    for name, value in overrides.items():
        frame[name] = value
    return frame


def linear_probs(count, start=0.05, stop=0.95):
    return np.linspace(start, stop, count)


# --------------------------------------------------------------------------- #
# Probability averaging, not logit averaging
# --------------------------------------------------------------------------- #

def test_probabilities_are_averaged_arithmetically():
    """`prob` must be the arithmetic mean -- the mixture's predictive mean.

    The literal here is the whole contract: mean(0.2, 0.8) = 0.5. Logit averaging
    would give 0.5 too for this symmetric pair, which is exactly why the asymmetric
    case below is the real test.
    """
    left = make_frame([0.2] * 4)
    right = make_frame([0.8] * 4)
    result = aggregate_frames([left, right])
    assert np.allclose(result["prob"], 0.5)


def test_logit_averaging_would_give_a_different_answer():
    """Pin the distinction with an asymmetric pair.

    p = (0.9, 0.1, 0.1) -- arithmetic mean 0.3667, geometric mean of odds 0.25.
    A regression to logit averaging would shift every probability toward 0 or 1, so
    calibration would degrade while every rank-based metric stayed identical: the
    failure would show up only in ECE, and would look like a real finding about
    ensembles rather than a bug.
    """
    frames = [make_frame([p] * 4) for p in (0.9, 0.1, 0.1)]
    result = aggregate_frames(frames)

    arithmetic = (0.9 + 0.1 + 0.1) / 3.0
    odds = [p / (1 - p) for p in (0.9, 0.1, 0.1)]
    geometric_odds = np.exp(np.mean(np.log(odds)))
    logit_mean = geometric_odds / (1 + geometric_odds)

    assert not np.isclose(arithmetic, logit_mean), "fixture must distinguish the two"
    assert np.allclose(result["prob"], arithmetic)
    assert not np.allclose(result["prob"], logit_mean)


def test_logit_column_is_the_logit_of_the_averaged_probability():
    frames = [make_frame([p] * 4) for p in (0.3, 0.6, 0.9)]
    result = aggregate_frames(frames)
    expected = np.log(result["prob"] / (1 - result["prob"]))
    assert np.allclose(result["logit"], expected)


def test_logit_column_stays_a_monotone_function_of_prob():
    """Rank-based metrics read `logit` too, so it must not reorder the samples."""
    probabilities = linear_probs(N_ROWS)
    frames = [make_frame(probabilities * scale) for scale in (0.9, 1.0)]
    result = aggregate_frames(frames)
    order_by_prob = np.argsort(result["prob"].to_numpy(), kind="stable")
    order_by_logit = np.argsort(result["logit"].to_numpy(), kind="stable")
    assert np.array_equal(order_by_prob, order_by_logit)


def test_prediction_and_correctness_are_recomputed():
    """`pred` follows the *averaged* probability, not a vote of member predictions.

    Members at 0.9, 0.45, 0.45 average to 0.6 -- the ensemble says positive while two
    of three members said negative. A majority vote over `pred` would disagree, and
    would also be a different (and worse-calibrated) estimator.
    """
    frames = [make_frame([p], labels=[1.0]) for p in (0.9, 0.45, 0.45)]
    result = aggregate_frames(frames)
    assert np.isclose(result["prob"].iloc[0], 0.6)
    assert result["pred"].iloc[0] == 1.0
    assert result["correct"].iloc[0] == 1.0


def test_loss_is_the_loss_of_the_average_not_the_average_loss():
    """By Jensen, mean(NLL) >= NLL(mean). Recording the former would overstate loss."""
    labels = np.ones(4)
    frames = [make_frame([p] * 4, labels=labels) for p in (0.2, 0.9)]
    result = aggregate_frames(frames)

    nll_of_mean = -np.log(0.55)
    mean_of_nll = (-np.log(0.2) + -np.log(0.9)) / 2.0
    assert nll_of_mean < mean_of_nll
    assert np.allclose(result["loss_sample"], nll_of_mean)


# --------------------------------------------------------------------------- #
# The uncertainty decomposition
# --------------------------------------------------------------------------- #

def test_disagreement_is_zero_when_members_are_identical():
    """The BatchEnsemble failure, in ensemble form.

    Identical members means no epistemic signal at all -- and every ensemble metric
    then measures nothing. Asserted as an exact zero so the launcher's warning has a
    defined trigger.
    """
    frame = make_frame(linear_probs(N_ROWS))
    result = aggregate_frames([frame, frame.copy()])
    assert np.all(result["u_ens_variance"] == 0.0)
    assert np.all(result["u_ens_disagreement"] == 0.0)


def test_disagreement_is_positive_when_members_differ():
    frames = [make_frame(linear_probs(N_ROWS, start, 0.95))
              for start in (0.05, 0.15, 0.30)]
    result = aggregate_frames(frames)
    assert result["u_ens_variance"].min() >= 0.0
    assert result["u_ens_variance"].max() > 0.0


def test_mutual_information_is_never_negative():
    """Total entropy >= mean member entropy by Jensen, but float error can invert it.

    A negative "epistemic uncertainty" in a plot is worse than a clipped zero: it
    reads as a bug in the method rather than in the arithmetic.
    """
    frame = make_frame(linear_probs(N_ROWS))
    result = aggregate_frames([frame, frame.copy(), frame.copy()])
    assert (result["u_ens_mutual_information"] >= 0.0).all()
    assert np.allclose(result["u_ens_mutual_information"], 0.0)


def test_entropy_decomposition_adds_up():
    frames = [make_frame(linear_probs(N_ROWS, start, 0.9))
              for start in (0.1, 0.4)]
    result = aggregate_frames(frames)
    total = result["u_ens_entropy"].to_numpy()
    aleatoric = result["u_ens_entropy_aleatoric"].to_numpy()
    mutual = result["u_ens_mutual_information"].to_numpy()
    assert np.allclose(total, aleatoric + mutual, atol=1e-12)
    assert (total >= aleatoric - 1e-12).all()


def test_entropy_is_finite_at_the_extremes():
    """p in {0, 1} must not produce inf or nan."""
    frames = [make_frame([0.0, 1.0, 0.0, 1.0]), make_frame([0.0, 1.0, 1.0, 0.0])]
    result = aggregate_frames(frames)
    for column in ("u_ens_entropy", "u_ens_entropy_aleatoric",
                   "u_ens_mutual_information", "logit", "loss_sample"):
        assert np.isfinite(result[column].to_numpy()).all(), f"{column} not finite"


def test_member_uncertainty_columns_are_dropped():
    """Averaging a member's own `u_gp_variance` produces an undefined quantity.

    The mean of four members' SNGP variances is not any member's variance and is not
    the ensemble's; keeping the column would put a number with no definition into the
    scored table.
    """
    frames = [
        make_frame([0.3] * 4, uncertainty={"u_gp_variance": 0.1, "u_vacuity": 0.2}),
        make_frame([0.7] * 4, uncertainty={"u_gp_variance": 0.9, "u_vacuity": 0.8}),
    ]
    result = aggregate_frames(frames)
    assert "u_gp_variance" not in result.columns
    assert "u_vacuity" not in result.columns
    assert "u_ens_variance" in result.columns


def test_annotation_columns_survive():
    """`anno_*` describes the *data*, is identical across members, and must persist.

    It is also never a UQ score -- the registry refuses that prefix -- so it has to be
    distinguishable from the `u_*` columns that were just dropped.
    """
    frames = [make_frame([0.3] * 4), make_frame([0.7] * 4)]
    result = aggregate_frames(frames)
    assert "anno_unc_gender" in result.columns
    assert np.allclose(result["anno_unc_gender"], 0.25)


def test_method_id_and_member_count_are_recorded():
    frames = [make_frame([0.3] * 4), make_frame([0.5] * 4), make_frame([0.7] * 4)]
    result = aggregate_frames(frames)
    assert (result["method_id"] == "deep_ensemble").all()
    assert (result["n_members"] == 3).all()


def test_column_order_is_stable():
    frames = [make_frame([0.3] * 4), make_frame([0.7] * 4)]
    first = aggregate_frames(frames)
    second = aggregate_frames([frame.copy() for frame in frames])
    assert list(first.columns) == list(second.columns)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def test_row_order_does_not_matter():
    """A member written in a different row order still aligns, via record_id."""
    probabilities = linear_probs(N_ROWS)
    left = make_frame(probabilities)
    right = make_frame(probabilities).iloc[::-1].reset_index(drop=True)
    result = aggregate_frames([left, right])
    assert np.allclose(result["prob"], probabilities)


def test_different_row_counts_are_refused():
    with pytest.raises(EnsembleCompatibilityError, match="same samples"):
        align_frames([make_frame([0.3] * 6), make_frame([0.3] * 5)])


def test_different_record_ids_are_refused():
    """Not silently intersected: that would change the evaluated set per member."""
    left = make_frame([0.3] * 6)
    right = make_frame([0.3] * 6)
    right["record_id"] = np.arange(100, 106)
    with pytest.raises(EnsembleCompatibilityError, match="record_id"):
        align_frames([left, right])


def test_mismatched_paths_at_the_same_record_id_are_refused():
    """Same ids, different images -- the worst case, since shapes still line up."""
    left = make_frame([0.3] * 6)
    right = make_frame([0.3] * 6)
    right.loc[2, "rel_path"] = "/dfdc/fake/other.png"
    with pytest.raises(EnsembleCompatibilityError, match="rel_path"):
        align_frames([left, right])


def test_mismatched_labels_are_refused():
    left = make_frame([0.3] * 6)
    right = make_frame([0.3] * 6)
    right.loc[3, "label"] = 1.0 - right.loc[3, "label"]
    with pytest.raises(EnsembleCompatibilityError, match="label"):
        align_frames([left, right])


def test_empty_input_is_refused():
    with pytest.raises(EnsembleCompatibilityError):
        align_frames([])


# --------------------------------------------------------------------------- #
# Compatibility
# --------------------------------------------------------------------------- #

def make_member(index, run_id=None, seed=42, commit="abc123", mode="strict",
                detector="resnestdf", head="none", mc_samples=0, ensemble_id="ens1",
                records_path="/tmp/records.csv.gz", description="resnestdf_random"):
    fingerprint = {
        "run_id": run_id or f"run_{index}",
        "seed": seed,
        "ensemble_member": index,
        "ensemble_id": ensemble_id,
        "determinism": {"mode": mode, "seed": seed},
        "git": {"commit": commit, "dirty": False},
        "results": {
            description: {
                "detector": detector,
                "uncertainty_head": head,
                "mc_dropout_samples": mc_samples,
                "complete": True,
                "records": {"test": records_path},
                "checkpoint": f"/ckpt/{index}.pth",
                "best_epoch": 2,
                "test_accuracy": 0.7 + index * 0.01,
            }
        },
    }
    return Member(
        run_id=fingerprint["run_id"], member_index=index, description=description,
        records_path=records_path, fingerprint=fingerprint,
    )


def test_compatible_members_pass():
    assert_members_compatible([make_member(0), make_member(1), make_member(2)]) is None


def test_a_single_member_is_refused():
    with pytest.raises(EnsembleCompatibilityError, match="at least 2"):
        assert_members_compatible([make_member(0)])


def test_mixed_uncertainty_heads_are_refused():
    """Averaging an sngp member into an evidential ensemble is not an ensemble.

    The heads parameterize uncertainty differently, so their `u_*` columns are not the
    same quantity -- and the plan calls this out as checkable precisely because the
    checkpoint records it.
    """
    members = [make_member(0, head="sngp"), make_member(1, head="evidential")]
    with pytest.raises(EnsembleCompatibilityError, match="uncertainty_head differs"):
        assert_members_compatible(members)


def test_mixed_detectors_are_refused():
    members = [make_member(0, detector="resnestdf"),
               make_member(1, detector="effnetdf")]
    with pytest.raises(EnsembleCompatibilityError, match="detector differs"):
        assert_members_compatible(members)


def test_mixed_git_commits_are_refused():
    members = [make_member(0, commit="a" * 40), make_member(1, commit="b" * 40)]
    with pytest.raises(EnsembleCompatibilityError, match="git commits"):
        assert_members_compatible(members)


def test_mixed_determinism_modes_are_refused():
    members = [make_member(0, mode="strict"), make_member(1, mode="fast")]
    with pytest.raises(EnsembleCompatibilityError, match="determinism modes"):
        assert_members_compatible(members)


def test_different_seeds_are_refused_with_the_reason():
    """Members must vary --ensemble-member, not --seed.

    A different seed also changes the graph and the data order, so the resulting
    disagreement would conflate initialization variance with data-order variance --
    and the graph cache key embeds the seed, so it would also cost N graph rebuilds.
    """
    members = [make_member(0, seed=1), make_member(1, seed=2)]
    with pytest.raises(EnsembleCompatibilityError) as info:
        assert_members_compatible(members)
    message = str(info.value)
    assert "--ensemble-member only" in message
    assert "data order" in message


def test_duplicate_member_indices_are_refused():
    """The same initialization counted twice deflates apparent disagreement."""
    members = [make_member(0), make_member(0, run_id="run_other")]
    with pytest.raises(EnsembleCompatibilityError, match="duplicate"):
        assert_members_compatible(members)


def test_a_member_without_an_index_is_refused():
    members = [make_member(0), make_member(1)]
    members[1].member_index = None
    with pytest.raises(EnsembleCompatibilityError, match="no --ensemble-member"):
        assert_members_compatible(members)


def test_indexless_members_can_be_allowed_explicitly():
    """Useful for seed-variability error bars, which are not an ensemble."""
    members = [make_member(0), make_member(1)]
    members[0].member_index = None
    members[1].member_index = None
    assert_members_compatible(members, require_distinct_members=False)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def write_run(root, run_id, member_index, ensemble_id="ens1", complete=True,
              with_records=True, probabilities=None, description="resnestdf_random",
              head="none", seed=42):
    """A run_outputs/<run_id> directory as test_hierarchical.py writes one."""
    run_dir = os.path.join(str(root), run_id)
    os.makedirs(run_dir, exist_ok=True)

    records_path = None
    if with_records:
        records_path = os.path.join(run_dir, "records_test.csv.gz")
        frame = make_frame(probabilities if probabilities is not None
                           else linear_probs(N_ROWS))
        digest = write_records(frame, records_path)
        # The sidecar is what makes the sha256 check possible. Writing the table
        # without it would silently turn `verify=True` into a no-op, so a tampering
        # test would pass for the wrong reason.
        write_manifest(default_meta_path(records_path), records_path, digest)

    fingerprint = {
        "run_id": run_id,
        "seed": seed,
        "ensemble_member": member_index,
        "ensemble_id": ensemble_id,
        "determinism": {"mode": "strict", "seed": seed},
        "git": {"commit": "c" * 40, "dirty": False},
        "results": {
            description: {
                "detector": "resnestdf",
                "uncertainty_head": head,
                "mc_dropout_samples": 0,
                "complete": complete,
                "records": {"test": records_path} if records_path else {},
                "checkpoint": os.path.join(run_dir, "best.pth"),
                "best_epoch": 2,
                "test_accuracy": 0.7,
            }
        },
    }
    with open(os.path.join(run_dir, "determinism.json"), "w") as handle:
        json.dump(fingerprint, handle)
    return run_dir


def test_discovery_finds_complete_members(tmp_path):
    root = tmp_path / "run_outputs"
    for index in range(3):
        write_run(root, f"run_{index}", index)
    members = discover_members(root, ensemble_id="ens1")
    assert [member.member_index for member in members] == [0, 1, 2]


def test_discovery_is_ordered_by_member_index(tmp_path):
    """Directory iteration order must not decide which member is "first"."""
    root = tmp_path / "run_outputs"
    write_run(root, "zzz_run", 0)
    write_run(root, "aaa_run", 1)
    members = discover_members(root, ensemble_id="ens1")
    assert [member.member_index for member in members] == [0, 1]
    assert members[0].run_id == "zzz_run"


def test_discovery_skips_incomplete_runs(tmp_path):
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0)
    write_run(root, "run_1", 1, complete=False)
    members = discover_members(root, ensemble_id="ens1")
    assert [member.member_index for member in members] == [0]


def test_discovery_skips_runs_with_no_records(tmp_path):
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0)
    write_run(root, "run_1", 1, with_records=False)
    assert len(discover_members(root, ensemble_id="ens1")) == 1


def test_discovery_skips_records_that_no_longer_exist(tmp_path):
    """A manifest pointing at a deleted table must not become a member."""
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0)
    run_dir = write_run(root, "run_1", 1)
    os.remove(os.path.join(run_dir, "records_test.csv.gz"))
    assert len(discover_members(root, ensemble_id="ens1")) == 1


def test_discovery_filters_by_ensemble_id(tmp_path):
    """Two ensembles sharing an output root must not be merged."""
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0, ensemble_id="ens1")
    write_run(root, "run_1", 1, ensemble_id="ens1")
    write_run(root, "run_2", 0, ensemble_id="ens2")
    assert len(discover_members(root, ensemble_id="ens1")) == 2
    assert len(discover_members(root, ensemble_id="ens2")) == 1


def test_discovery_ignores_unrelated_directories(tmp_path):
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0)
    (root / "not_a_run").mkdir()
    (root / "junk").mkdir()
    (root / "junk" / "determinism.json").write_text("{ not json")
    assert len(discover_members(root, ensemble_id="ens1")) == 1


def test_discovery_on_a_missing_directory_is_empty(tmp_path):
    assert discover_members(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_aggregate_members_round_trips_through_disk(tmp_path):
    root = tmp_path / "run_outputs"
    offsets = (0.0, 0.05, -0.05)
    for index, offset in enumerate(offsets):
        write_run(root, f"run_{index}", index,
                  probabilities=np.clip(linear_probs(N_ROWS) + offset, 0.01, 0.99))

    members = discover_members(root, ensemble_id="ens1")
    frame, manifest = aggregate_members(members)

    assert len(frame) == N_ROWS
    assert manifest["n_members"] == 3
    assert manifest["detector"] == "resnestdf"
    assert manifest["cost_training_runs"] == 3
    assert manifest["mean_disagreement"] > 0.0
    assert [entry["ensemble_member"] for entry in manifest["members"]] == [0, 1, 2]


def test_saved_ensemble_reads_back_and_verifies(tmp_path):
    root = tmp_path / "run_outputs"
    for index in range(2):
        write_run(root, f"run_{index}", index,
                  probabilities=linear_probs(N_ROWS, 0.05 + index * 0.1, 0.9))

    members = discover_members(root, ensemble_id="ens1")
    frame, manifest = aggregate_members(members)
    out = tmp_path / "ensemble" / "records_test.csv.gz"
    digest = save_ensemble(frame, manifest, out)

    # read_records verifies the sha256 against the sidecar by default, so this also
    # asserts save_ensemble wrote a manifest the reader accepts.
    reloaded = read_records(str(out))
    assert len(reloaded) == N_ROWS
    assert np.allclose(reloaded["prob"], frame["prob"])
    assert np.allclose(reloaded["u_ens_variance"], frame["u_ens_variance"])

    with open(str(out).replace(".csv.gz", ".meta.json")) as handle:
        saved = json.load(handle)
    assert saved["sha256_records"] == digest
    assert saved["n_rows"] == N_ROWS
    assert saved["coverage"] == 1.0


def test_saving_the_same_ensemble_twice_is_byte_identical(tmp_path):
    """A benchmark artifact must be reproducible, gzip mtime included."""
    root = tmp_path / "run_outputs"
    for index in range(2):
        write_run(root, f"run_{index}", index,
                  probabilities=linear_probs(N_ROWS, 0.05 + index * 0.1, 0.9))
    members = discover_members(root, ensemble_id="ens1")

    digests = []
    for run in range(2):
        frame, manifest = aggregate_members(members)
        digests.append(save_ensemble(frame, manifest, tmp_path / f"out{run}.csv.gz"))
    assert digests[0] == digests[1]


def test_aggregation_refuses_a_tampered_member_table(tmp_path):
    """The sha256 check is what makes a records path trustworthy."""
    root = tmp_path / "run_outputs"
    for index in range(2):
        write_run(root, f"run_{index}", index)

    tampered = os.path.join(str(root), "run_1", "records_test.csv.gz")
    with open(tampered, "ab") as handle:
        handle.write(b"\x00")

    members = discover_members(root, ensemble_id="ens1")
    with pytest.raises(ValueError, match="does not match its manifest"):
        aggregate_members(members)


def test_aggregation_refuses_mixed_heads_end_to_end(tmp_path):
    root = tmp_path / "run_outputs"
    write_run(root, "run_0", 0, head="sngp")
    write_run(root, "run_1", 1, head="evidential")
    members = discover_members(root, ensemble_id="ens1")
    with pytest.raises(EnsembleCompatibilityError, match="uncertainty_head"):
        aggregate_members(members)
