"""Scoring records into a per-method results table.

The refusals matter as much as the numbers: each one corresponds to a way a
comparison table can look perfectly reasonable while being wrong.
"""

import numpy as np
import pytest

from evaluation.uq.scoring import (
    MIN_COVERAGE, Cell, IncomparableCellsError, add_skipped_rows, assert_comparable,
    collapse_rank_equivalents, guard_single_class_classification, pivot_for_paper,
    resolve_score, score_cell, score_cells, score_ood,
)


def make_frame(n=200, seed=0, single_class=False, constant_score=False):
    """A record table with an informative uncertainty column."""
    import pandas as pd

    rng = np.random.Generator(np.random.PCG64(seed))
    labels = np.ones(n, dtype=int) if single_class else rng.integers(0, 2, size=n)
    probabilities = np.clip(labels * 0.6 + 0.2 + rng.normal(0, 0.22, size=n), 0.01, 0.99)
    errors = (probabilities > 0.5).astype(int) != labels
    scores = (
        np.full(n, 0.3) if constant_score
        else 0.1 + errors * 0.35 + rng.uniform(0, 0.25, size=n)
    )
    return pd.DataFrame({
        "record_id": np.arange(n),
        "label": labels,
        "prob": probabilities,
        "pred": (probabilities > 0.5).astype(int),
        "correct": (~errors).astype(int),
        "u_sngp_variance": scores,
        "u_hybrid_distance": scores * 7.0 + 3.0,   # same ordering, different scale
        "source_group": np.where(np.arange(n) % 4 == 0, "ProGAN", "CelebA"),
        "domain": "id",
    })


def make_cell(frame=None, method_id="sngp", score_column="u_sngp_variance", **kwargs):
    return Cell(
        detector=kwargs.pop("detector", "effnetdf"),
        method_id=method_id,
        score_column=score_column,
        frame=frame if frame is not None else make_frame(),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Score resolution
# --------------------------------------------------------------------------- #

def test_resolve_score_reads_an_existing_column():
    frame = make_frame()
    assert np.array_equal(
        resolve_score(frame, "u_sngp_variance"), frame["u_sngp_variance"].to_numpy()
    )


def test_resolve_score_derives_baselines_from_probability():
    """The baselines need no extra columns, so they work on any record table."""
    frame = make_frame()
    for column in ("u_maxprob", "u_entropy", "u_margin"):
        scores = resolve_score(frame, column)
        assert scores.shape == (len(frame),)
        assert np.isfinite(scores).all()


def test_resolve_score_rejects_annotation_uncertainty():
    frame = make_frame()
    frame["anno_unc_race"] = 0.3
    with pytest.raises(ValueError, match="annotation uncertainty"):
        resolve_score(frame, "anno_unc_race")


def test_resolve_score_reports_a_missing_column_usefully():
    with pytest.raises(KeyError, match="available uncertainty columns"):
        resolve_score(make_frame(), "u_does_not_exist")


# --------------------------------------------------------------------------- #
# Scoring one cell
# --------------------------------------------------------------------------- #

def test_score_cell_produces_ranking_and_calibration_metrics():
    row = score_cell(make_cell())
    assert row["status"] == "ok"
    assert row["detector"] == "effnetdf" and row["method_id"] == "sngp"
    for name in ("auroc_error", "aupr_error", "aurc", "eaurc", "accuracy_at_0.9"):
        assert np.isfinite(row[name]), f"{name} is not finite"
    for name in ("ece_confidence", "ece_positive", "brier", "nll"):
        assert np.isfinite(row[name]), f"{name} is not finite"
    assert row["calibration_applicable"] is True


def test_score_cell_reports_cost():
    """A table where ensembles win must show what they cost."""
    row = score_cell(make_cell(method_id="deep_ensemble", score_column="u_sngp_variance",
                               cost_training_runs=5, cost_forward_passes=5))
    assert row["cost_training_runs"] == 5
    assert row["cost_forward_passes"] == 5


def test_graph_methods_get_na_calibration_not_zero():
    """The most common way a UQ table becomes nonsense.

    A graph distance has no calibrated probability, so its ECE is undefined. Filling
    it with 0 would make it look perfectly calibrated -- better than every real
    method.
    """
    row = score_cell(make_cell(
        method_id="graph_hybrid_distance", score_column="u_hybrid_distance"
    ))
    assert row["calibration_applicable"] is False
    for name in ("ece_confidence", "brier", "nll", "mce_confidence"):
        assert np.isnan(row[name]), f"{name} should be N/A, got {row[name]}"
    assert "not_probabilistic" in row["status_flags"]
    # Ranking metrics are still valid, since they only use the ordering.
    assert np.isfinite(row["auroc_error"])
    assert np.isfinite(row["eaurc"])


def test_scale_does_not_change_ranking_metrics():
    """Two columns with identical orderings but different scales must tie."""
    frame = make_frame()
    bounded = score_cell(make_cell(frame=frame, score_column="u_sngp_variance"))
    unbounded = score_cell(make_cell(
        frame=frame, method_id="graph_hybrid_distance", score_column="u_hybrid_distance"
    ))
    for name in ("auroc_error", "aupr_error", "aurc", "eaurc"):
        assert bounded[name] == pytest.approx(unbounded[name], abs=1e-12), name


def test_low_coverage_cell_is_refused():
    """A headline number computed on part of the data must not be published."""
    row = score_cell(make_cell(coverage=0.80))
    assert row["status"] == "refused_low_coverage"
    assert "coverage=0.8" in row["status_flags"]
    assert "auroc_error" not in row


def test_full_coverage_is_accepted():
    assert score_cell(make_cell(coverage=1.0))["status"] == "ok"
    assert score_cell(make_cell(coverage=MIN_COVERAGE))["status"] == "ok"


def test_constant_score_is_flagged_degenerate():
    """An identically-constant uncertainty signal measures nothing.

    The signature of un-diversified ensemble members or zero-p MC dropout.
    """
    row = score_cell(make_cell(frame=make_frame(constant_score=True)))
    assert row["status"] == "degenerate"
    assert "degenerate_constant_score" in row["status_flags"]
    assert row["auroc_error"] == pytest.approx(0.5)


def test_single_class_labels_are_flagged():
    row = score_cell(make_cell(frame=make_frame(single_class=True)))
    assert "single_class" in row["status_flags"]
    assert np.isnan(row["clf_auroc"])


def test_bootstrap_ci_is_added_when_requested():
    row = score_cell(make_cell(), n_boot=80, bootstrap_seed=3)
    assert row["auroc_error_ci_low"] <= row["auroc_error"] <= row["auroc_error_ci_high"]


def test_score_cell_is_reproducible():
    cell = make_cell()
    first = score_cell(cell, n_boot=50, bootstrap_seed=9)
    second = score_cell(cell, n_boot=50, bootstrap_seed=9)
    assert first == second


# --------------------------------------------------------------------------- #
# Provenance guards
# --------------------------------------------------------------------------- #

def test_mixed_determinism_modes_are_refused():
    cells = [make_cell(determinism_mode="strict"), make_cell(determinism_mode="fast")]
    with pytest.raises(IncomparableCellsError, match="determinism mode"):
        assert_comparable(cells)


def test_mixed_manifests_are_refused():
    """Different evaluation manifests mean different samples were scored."""
    cells = [make_cell(manifest_sha256="aaa"), make_cell(manifest_sha256="bbb")]
    with pytest.raises(IncomparableCellsError, match="evaluation manifest"):
        assert_comparable(cells)


def test_mixed_graph_normalization_is_refused():
    """Graph-distance values fitted on different statistics are not comparable."""
    cells = [make_cell(graph_norm_sha256="aaa"), make_cell(graph_norm_sha256="bbb")]
    with pytest.raises(IncomparableCellsError, match="normalization"):
        assert_comparable(cells)


def test_matching_provenance_is_allowed():
    cells = [
        make_cell(determinism_mode="strict", manifest_sha256="m1", graph_norm_sha256="g1"),
        make_cell(determinism_mode="strict", manifest_sha256="m1", graph_norm_sha256="g1"),
    ]
    assert_comparable(cells)  # must not raise


def test_unknown_provenance_does_not_block():
    """Absent metadata must not be treated as a conflict."""
    assert_comparable([make_cell(), make_cell(determinism_mode="strict")])


def test_score_cells_enforces_comparability():
    cells = [make_cell(determinism_mode="strict"), make_cell(determinism_mode="fast")]
    with pytest.raises(IncomparableCellsError):
        score_cells(cells)
    # Explicit opt-out is available for exploratory use.
    assert len(score_cells(cells, require_comparable=False)) == 2


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #

def test_score_cells_returns_a_tidy_frame():
    frame = make_frame()
    cells = [
        make_cell(frame=frame, method_id="sngp", score_column="u_sngp_variance"),
        make_cell(frame=frame, method_id="baseline_maxprob", score_column="u_maxprob"),
        make_cell(frame=frame, method_id="graph_hybrid_distance",
                  score_column="u_hybrid_distance"),
    ]
    results = score_cells(cells)
    assert len(results) == 3
    assert set(results["method_id"]) == {"sngp", "baseline_maxprob", "graph_hybrid_distance"}
    assert "method_family" in results.columns


def test_skipped_pairs_appear_as_rows():
    """A published matrix should show explained holes, not missing rows."""
    from evaluation.uq.registry import expand_matrix

    results = score_cells([make_cell()])
    _, decisions = expand_matrix(["sngp", "baseline_maxprob"], ["squeezenetdf", "dag_fdd"])
    combined = add_skipped_rows(results, decisions)

    skipped = combined[combined["status"].isin({"skipped", "broken"})]
    assert not skipped.empty
    assert skipped["skip_reason"].str.len().min() > 20, "every skip must explain itself"
    assert set(skipped["status"]) <= {"skipped", "broken"}


def test_rank_equivalents_are_collapsed_not_duplicated():
    """Entropy must not print a second identical ranking column.

    It is a monotone function of max-probability, so identical numbers there imply a
    difference that does not exist. Its calibration metrics stay, since those can
    genuinely differ.
    """
    frame = make_frame()
    results = score_cells([
        make_cell(frame=frame, method_id="baseline_maxprob", score_column="u_maxprob"),
        make_cell(frame=frame, method_id="baseline_entropy", score_column="u_entropy"),
    ])
    # Before collapsing, they are numerically identical -- which is the point.
    by_method = results.set_index("method_id")
    assert by_method.loc["baseline_maxprob", "auroc_error"] == pytest.approx(
        by_method.loc["baseline_entropy", "auroc_error"], abs=1e-12
    )

    collapsed = collapse_rank_equivalents(results).set_index("method_id")
    assert np.isnan(collapsed.loc["baseline_entropy", "auroc_error"])
    assert "by construction" in collapsed.loc["baseline_entropy", "ranking_note"]
    assert np.isfinite(collapsed.loc["baseline_maxprob", "auroc_error"])
    # Calibration is untouched -- that is where a post-hoc method earns its place.
    assert np.isfinite(collapsed.loc["baseline_entropy", "ece_confidence"])


def test_pivot_for_paper_shapes_a_table():
    frame = make_frame()
    cells = [
        make_cell(frame=frame, detector="effnetdf", method_id="sngp"),
        make_cell(frame=frame, detector="resnestdf", method_id="sngp"),
    ]
    table = pivot_for_paper(score_cells(cells), metric="eaurc")
    assert list(table.index) == ["sngp"]
    assert set(table.columns) == {"effnetdf", "resnestdf"}


# --------------------------------------------------------------------------- #
# OOD and single-class guards
# --------------------------------------------------------------------------- #

def test_score_ood_on_an_all_fake_holdout():
    """Holding a generator out yields a single-class set; OOD ranking still works."""
    id_frame = make_frame(n=150, seed=1)
    ood_frame = make_frame(n=60, seed=2, single_class=True)
    ood_frame["u_sngp_variance"] = ood_frame["u_sngp_variance"] + 0.5  # shifted upward

    row = score_ood(id_frame, ood_frame, "u_sngp_variance", "effnetdf", "sngp")
    assert row["status"] == "ok"
    assert row["ood_auroc"] > 0.8
    assert row["n_id"] == 150 and row["n_ood"] == 60


def test_score_ood_refuses_an_empty_partition():
    import pandas as pd

    empty = make_frame(n=0)
    row = score_ood(make_frame(), empty, "u_sngp_variance", "effnetdf", "sngp")
    assert row["status"] == "refused"
    assert "empty_ood_partition" in row["status_flags"]


def test_classification_on_a_single_class_set_is_refused():
    """The trap that makes a held-out-generator result meaningless.

    Reporting AUROC on an all-fake set is undefined, and reporting an accuracy that is
    just the class prior is worse than refusing. Shifted-classification evaluation has
    to mix held-out fakes with in-distribution reals.
    """
    with pytest.raises(IncomparableCellsError, match="only class"):
        guard_single_class_classification(
            make_frame(single_class=True), context="H1_diffusion_unseen holdout"
        )


def test_classification_guard_passes_on_a_mixed_set():
    assert guard_single_class_classification(make_frame()) is True


def test_guard_message_names_the_remedy():
    try:
        guard_single_class_classification(make_frame(single_class=True))
    except IncomparableCellsError as error:
        assert "in-distribution reals" in str(error)
        assert "score_ood()" in str(error)
    else:
        pytest.fail("expected a refusal")
