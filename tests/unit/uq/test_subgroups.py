"""Per-subgroup scoring and the disparity reduction.

The record tables have carried `gt_gender`, `gt_age`, `gt_race`, and `intersection` since
schema version 1, and nothing grouped by them: every metric in `scoring.py` was computed
on the whole evaluation set, and the only per-subgroup number anywhere was accuracy,
computed inside `evaluate_model` over race x gender.

So the property that matters most is **commensurability**: `disparity_range` must equal
the max-minus-min that `evaluate_model` already reports as `race_gender_overall_bias`, and
`disparity_mad` must equal its `race_gender_average_subgroup_bias`. If they drift, the
legacy log line and the new table disagree about the same quantity.

Second: a small group is **flagged, not dropped**. Dropping it would narrow the disparity
range exactly when a group is too rare to measure, which reads as an improvement.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.uq import subgroups
from evaluation.uq.scoring import Cell, score_cells

RNG = np.random.default_rng(20260810)


def make_records(n=400, groups=(0, 1, 2, 3), gender=(0, 1), seed=0):
    """A synthetic record table with the columns subgroup scoring reads.

    Accuracy is deliberately made to vary by race group, so a disparity is present to
    measure rather than an artifact of noise.
    """
    rng = np.random.default_rng(seed)
    race = rng.choice(groups, size=n)
    labels = rng.integers(0, 2, size=n)
    # Group 0 is the hard group: its probabilities sit near the boundary.
    confidence = np.where(race == 0, 0.55, 0.9)
    probability = np.where(labels == 1, confidence, 1.0 - confidence)
    probability = np.clip(probability + rng.normal(0, 0.03, size=n), 0.01, 0.99)
    prediction = (probability >= 0.5).astype(int)
    return pd.DataFrame({
        "record_id": [f"r{index:05d}" for index in range(n)],
        "label": labels,
        "pred": prediction,
        "correct": (prediction == labels).astype(int),
        "prob": probability,
        "logit": np.log(probability / (1 - probability)),
        "gt_gender": rng.choice(gender, size=n),
        "gt_race": race,
        "gt_age": rng.choice([0, 1, 2, 3], size=n),
        "intersection": rng.integers(0, 8, size=n),
    })


def base_cell(frame, **kwargs):
    return Cell(
        detector="tiny", method_id="baseline_maxprob", score_column="u_maxprob",
        frame=frame, extra={"label": "cell"}, **kwargs,
    )


def scored(frame, min_rows=subgroups.MIN_SUBGROUP_ROWS, dimensions=None):
    cells = subgroups.expand_cells(
        [base_cell(frame)],
        dimensions=dimensions or subgroups.SUBGROUP_DIMENSIONS,
        min_rows=min_rows,
    )
    results = score_cells(cells, require_comparable=False)
    return subgroups.annotate_small_subgroups(results)


# -- slicing -------------------------------------------------------------------- #

def test_available_dimensions_needs_two_values():
    frame = make_records(seed=1)
    assert "gt_race" in subgroups.available_dimensions(frame)

    constant = frame.assign(gt_gender=0)
    assert "gt_gender" not in subgroups.available_dimensions(constant)


def test_absent_dimension_is_not_an_error():
    frame = make_records(seed=2).drop(columns=["gt_age"])
    assert "gt_age" not in subgroups.available_dimensions(frame)


def test_slices_partition_the_rows_with_a_known_value():
    frame = make_records(seed=3)
    slices = [
        (dimension, value, subframe)
        for dimension, value, subframe, _flags in subgroups.subgroup_frames(frame)
        if dimension == "gt_race"
    ]
    total = sum(len(subframe) for _dimension, _value, subframe in slices)
    assert total == len(frame)


def test_null_demographics_belong_to_no_group():
    """A sample whose race is unknown must not be counted in any race group."""
    frame = make_records(seed=4)
    frame.loc[:19, "gt_race"] = np.nan
    slices = [
        subframe for dimension, _value, subframe, _flags
        in subgroups.subgroup_frames(frame) if dimension == "gt_race"
    ]
    assert sum(len(subframe) for subframe in slices) == len(frame) - 20


def test_expand_cells_keeps_the_overall_cell_first():
    """Disparity is defined relative to the whole-set value, so it must be present."""
    frame = make_records(seed=5)
    cells = subgroups.expand_cells([base_cell(frame)])
    assert cells[0].extra.get("subgroup_dimension") is None
    assert len(cells) > 1
    for cell in cells[1:]:
        assert cell.extra["subgroup_dimension"] in subgroups.SUBGROUP_DIMENSIONS


def test_subgroup_cells_inherit_provenance():
    """Coverage and digests describe the run, not the slice, so they carry over."""
    frame = make_records(seed=6)
    cell = base_cell(frame, coverage=0.995, determinism_mode="strict",
                     manifest_sha256="abc", seed=42)
    for clone in subgroups.subgroup_cells(cell):
        assert clone.coverage == 0.995
        assert clone.determinism_mode == "strict"
        assert clone.manifest_sha256 == "abc"
        assert clone.seed == 42


def test_scored_rows_carry_the_subgroup_identity():
    results = scored(make_records(seed=7))
    assert set(results["subgroup_dimension"]) >= {"overall", "gt_race", "gt_gender"}
    overall = results[results["subgroup_dimension"] == "overall"]
    assert len(overall) == 1
    assert set(overall["subgroup_value"]) == {"all"}


def test_subgroup_values_are_plain_scalars():
    """numpy ints would render as `np.int64(2)` in the CSV and break the diff's join."""
    results = scored(make_records(seed=8))
    for value in results["subgroup_value"]:
        assert isinstance(value, (str, int)), f"{value!r} is {type(value)}"


# -- small groups --------------------------------------------------------------- #

def test_small_subgroups_are_flagged_not_dropped():
    frame = make_records(n=300, seed=9)
    # Make one race group rare.
    frame.loc[frame["gt_race"] == 2, "gt_race"] = 1
    frame.loc[:9, "gt_race"] = 2

    results = scored(frame, min_rows=50)
    rare = results[
        (results["subgroup_dimension"] == "gt_race") & (results["subgroup_value"] == 2)
    ]
    assert len(rare) == 1, "the rare group must still be scored"
    assert subgroups.SMALL_SUBGROUP in str(rare.iloc[0]["status_flags"])


def test_annotate_preserves_existing_status_flags():
    frame = make_records(n=120, seed=10)
    results = scored(frame, min_rows=1000)  # everything is "small"
    for flags in results[results["subgroup_dimension"] != "overall"]["status_flags"]:
        assert subgroups.SMALL_SUBGROUP in str(flags)


def test_single_class_subgroup_is_flagged_by_the_existing_scorer():
    """Degeneracy handling is inherited, not reimplemented."""
    frame = make_records(n=200, seed=11)
    frame.loc[frame["gt_gender"] == 1, "label"] = 1
    frame.loc[frame["gt_gender"] == 1, "correct"] = (
        frame.loc[frame["gt_gender"] == 1, "pred"] == 1
    ).astype(int)

    results = scored(frame)
    row = results[
        (results["subgroup_dimension"] == "gt_gender") & (results["subgroup_value"] == 1)
    ].iloc[0]
    assert "single_class" in str(row["status_flags"])


# -- disparity ------------------------------------------------------------------ #

def test_disparity_range_matches_max_minus_min():
    """The definition `evaluate_model` reports as race_gender_overall_bias."""
    results = scored(make_records(seed=12))
    frame = subgroups.disparity(results, ["clf_accuracy"])
    row = frame[
        (frame["subgroup_dimension"] == "gt_race") & (frame["metric"] == "clf_accuracy")
    ].iloc[0]

    groups = results[results["subgroup_dimension"] == "gt_race"]["clf_accuracy"]
    assert row["disparity_range"] == pytest.approx(groups.max() - groups.min())


def test_disparity_mad_matches_mean_absolute_deviation_from_overall():
    """The definition `evaluate_model` reports as race_gender_average_subgroup_bias."""
    results = scored(make_records(seed=13))
    frame = subgroups.disparity(results, ["clf_accuracy"])
    row = frame[
        (frame["subgroup_dimension"] == "gt_race") & (frame["metric"] == "clf_accuracy")
    ].iloc[0]

    overall = float(
        results[results["subgroup_dimension"] == "overall"]["clf_accuracy"].iloc[0]
    )
    groups = results[results["subgroup_dimension"] == "gt_race"]["clf_accuracy"].to_numpy()
    assert row["disparity_mad"] == pytest.approx(np.mean(np.abs(groups - overall)))
    assert row["overall_value"] == pytest.approx(overall)


def test_worst_group_is_oriented_by_the_metric_direction():
    """Lowest accuracy is worst; *highest* ECE is worst. Direction is not the sign."""
    results = scored(make_records(seed=14))
    frame = subgroups.disparity(results, ["clf_accuracy", "ece_confidence"])

    accuracy = frame[
        (frame["metric"] == "clf_accuracy") & (frame["subgroup_dimension"] == "gt_race")
    ].iloc[0]
    groups = results[results["subgroup_dimension"] == "gt_race"]
    assert accuracy["worst_value"] == pytest.approx(groups["clf_accuracy"].min())

    ece = frame[
        (frame["metric"] == "ece_confidence")
        & (frame["subgroup_dimension"] == "gt_race")
    ].iloc[0]
    assert ece["worst_value"] == pytest.approx(groups["ece_confidence"].max())


def test_worst_group_names_the_group():
    results = scored(make_records(seed=15))
    frame = subgroups.disparity(results, ["clf_accuracy"])
    row = frame[
        (frame["subgroup_dimension"] == "gt_race") & (frame["metric"] == "clf_accuracy")
    ].iloc[0]

    groups = results[results["subgroup_dimension"] == "gt_race"]
    expected = groups.loc[groups["clf_accuracy"].idxmin(), "subgroup_value"]
    assert row["worst_group"] == expected
    # The synthetic data makes group 0 the hard one, so this also checks the fixture.
    assert row["worst_group"] == 0


def test_disparity_does_not_mix_methods():
    """Two methods' subgroup values must not be pooled into one range."""
    frame = make_records(seed=16)
    cells = subgroups.expand_cells([
        base_cell(frame),
        Cell(detector="tiny", method_id="baseline_entropy", score_column="u_entropy",
             frame=frame, extra={"label": "cell"}),
    ])
    results = subgroups.annotate_small_subgroups(
        score_cells(cells, require_comparable=False)
    )
    disparity = subgroups.disparity(results, ["auroc_error"])
    for method in ("baseline_maxprob", "baseline_entropy"):
        rows = disparity[disparity["method_id"] == method]
        assert not rows.empty
        assert set(rows["subgroup_dimension"]) <= set(subgroups.SUBGROUP_DIMENSIONS)


def test_small_groups_are_excluded_from_the_range_but_flagged():
    frame = make_records(n=300, seed=17)
    frame.loc[frame["gt_race"] == 3, "gt_race"] = 1
    frame.loc[:4, "gt_race"] = 3

    results = scored(frame, min_rows=50)
    disparity = subgroups.disparity(results, ["clf_accuracy"])
    row = disparity[
        (disparity["subgroup_dimension"] == "gt_race")
        & (disparity["metric"] == "clf_accuracy")
    ].iloc[0]

    assert subgroups.SMALL_SUBGROUP in str(row["flags"])
    assert row["worst_group"] != 3, "the five-row group must not set the worst value"
    assert row["n_groups"] >= 3


def test_include_small_widens_the_range():
    frame = make_records(n=300, seed=18)
    frame.loc[frame["gt_race"] == 3, "gt_race"] = 1
    frame.loc[:4, "gt_race"] = 3
    results = scored(frame, min_rows=50)

    conservative = subgroups.disparity(results, ["clf_accuracy"], include_small=False)
    inclusive = subgroups.disparity(results, ["clf_accuracy"], include_small=True)
    key = ("gt_race", "clf_accuracy")

    def value(frame_):
        row = frame_[
            (frame_["subgroup_dimension"] == key[0]) & (frame_["metric"] == key[1])
        ].iloc[0]
        return row["disparity_range"], row["n_groups"]

    conservative_range, _ = value(conservative)
    inclusive_range, _ = value(inclusive)
    assert inclusive_range >= conservative_range


def test_disparity_on_an_empty_frame_is_empty():
    assert subgroups.disparity(pd.DataFrame(), ["clf_accuracy"]).empty


def test_disparity_requires_subgroup_columns():
    results = pd.DataFrame({"clf_accuracy": [0.9], "detector": ["tiny"]})
    with pytest.raises(subgroups.SubgroupError, match="subgroup_dimension"):
        subgroups.disparity(results, ["clf_accuracy"])


def test_disparity_ignores_metrics_the_table_lacks():
    results = scored(make_records(seed=19))
    frame = subgroups.disparity(results, ["clf_accuracy", "no_such_metric"])
    assert set(frame["metric"]) == {"clf_accuracy"}


# -- reshaping for the diff ----------------------------------------------------- #

def test_disparity_as_results_produces_diffable_columns():
    """Fairness must travel through the same comparison logic as accuracy."""
    from evaluation.uq.compare import metric_direction

    results = scored(make_records(seed=20))
    disparity = subgroups.disparity(results, ["clf_accuracy", "ece_confidence"])
    reshaped = subgroups.disparity_as_results(disparity)

    assert "disparity_range_clf_accuracy" in reshaped.columns
    assert "worst_group_clf_accuracy" in reshaped.columns
    assert set(reshaped["subgroup_value"]) == {"disparity"}
    assert set(reshaped["status"]) == {"ok"}

    # A spread is better when smaller, whatever the base metric's polarity.
    assert metric_direction("disparity_range_clf_accuracy") == "lower"
    assert metric_direction("disparity_range_ece_confidence") == "lower"
    # A worst-group value inherits its base metric's polarity.
    assert metric_direction("worst_group_clf_accuracy") == "higher"
    assert metric_direction("worst_group_ece_confidence") == "lower"


def test_disparity_as_results_keeps_one_row_per_dimension():
    results = scored(make_records(seed=21))
    disparity = subgroups.disparity(results, ["clf_accuracy", "ece_confidence"])
    reshaped = subgroups.disparity_as_results(disparity)
    counts = reshaped.groupby(["method_id", "subgroup_dimension"]).size()
    assert (counts == 1).all(), "one row per (method, dimension), not one per metric"


def test_disparity_as_results_on_empty_input():
    assert subgroups.disparity_as_results(pd.DataFrame()).empty
    assert subgroups.disparity_as_results(None).empty
