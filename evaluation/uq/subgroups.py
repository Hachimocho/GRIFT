"""Score a record table per demographic subgroup, and summarize the disparity.

The record tables have carried `gt_gender`, `gt_age`, `gt_race`, and `intersection`
since schema version 1, and nothing ever grouped by them: `scoring.py`, `report.py`, and
`uq_report.py` compute every metric on the whole evaluation set. The only per-subgroup
number the codebase produced was accuracy, computed inside `evaluate_model` over
race x gender and reported as a max-min range.

So this module adds no new metric mathematics. It slices the frame and hands the slices
to the existing `score_cells`, which means a subgroup gets the *same* treatment as the
whole set -- calibration, selective prediction, E-AURC, ranking -- and the same
degeneracy handling. A subgroup with one class present comes back flagged
`single_class_labels`, not with a plausible-looking AUROC.

Two rules are worth stating because getting either wrong produces a fairness number that
is confidently wrong:

* **Small groups are flagged, never dropped.** A subgroup below `MIN_SUBGROUP_ROWS` is
  still scored and still appears, carrying `small_subgroup` in `status_flags`. Dropping
  it would make the disparity range shrink exactly when a group is too rare to measure,
  which reads as an improvement.
* **Disparity is computed over rows that are actually comparable.** `disparity` groups by
  everything that identifies a measurement (detector, method, score column, condition)
  and reduces only across `subgroup_value` -- never across methods or conditions.

Demographic values are integer codes as the dataset supplies them (`0`/`1` for gender,
`0`-`3` for race and age); this module does not name them, because the mapping lives in
the dataset and inventing labels here would be a second, drifting source of truth.
"""

from typing import Dict, Optional

import numpy as np

from evaluation.uq.records import DEMOGRAPHIC_COLUMNS

#: Dimensions worth slicing by default. `gt_age` is included -- `evaluate_model`'s bias
#: block deliberately restricted itself to gender and race, so age disparity has never
#: been measured. `intersection` is the dataset's own precomputed race x gender code.
SUBGROUP_DIMENSIONS = ("gt_gender", "gt_race", "gt_age", "intersection")

#: Below this many rows a subgroup's metrics are too noisy to compare, so the row is
#: emitted with a flag rather than treated as a measurement. Not a drop: see module
#: docstring.
MIN_SUBGROUP_ROWS = 50

#: `status_flags` marker for a subgroup under `MIN_SUBGROUP_ROWS`.
SMALL_SUBGROUP = "small_subgroup"

#: Columns identifying a measurement. Disparity reduces across `subgroup_value` within
#: each distinct combination of these, so two methods' subgroup accuracies are never
#: mixed into one range.
DISPARITY_GROUP_KEYS = (
    "label", "detector", "method_id", "score_column",
    "holdout", "domain", "corruption", "severity", "subgroup_dimension",
)

#: Value used for whole-set rows, matching `scoring._identity`'s defaults.
OVERALL_DIMENSION = "overall"
OVERALL_VALUE = "all"


class SubgroupError(ValueError):
    """Raised when a frame cannot be sliced by the requested dimension."""


def available_dimensions(frame, dimensions=SUBGROUP_DIMENSIONS):
    """Dimensions present in `frame` with at least two distinct non-null values.

    A dimension with one value cannot express disparity, and one with none is simply
    absent from this table -- neither is an error, both are excluded.
    """
    found = []
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        if frame[dimension].dropna().nunique() >= 2:
            found.append(dimension)
    return found


def subgroup_frames(frame, dimensions=SUBGROUP_DIMENSIONS, min_rows=MIN_SUBGROUP_ROWS):
    """Yield `(dimension, value, subframe, flags)` for every subgroup in `frame`.

    Rows with a null demographic code are excluded from that dimension's slices -- a
    sample whose gender is unknown belongs to no gender group -- so the slices of one
    dimension need not sum to the whole frame. `flags` is a tuple, carrying
    `SMALL_SUBGROUP` when the slice is under `min_rows`.
    """
    for dimension in available_dimensions(frame, dimensions):
        column = frame[dimension]
        for value in sorted(column.dropna().unique()):
            subframe = frame[column == value]
            if subframe.empty:
                continue
            flags = (SMALL_SUBGROUP,) if len(subframe) < min_rows else ()
            yield dimension, _plain(value), subframe, flags


def subgroup_cells(
    base_cell, dimensions=SUBGROUP_DIMENSIONS, min_rows=MIN_SUBGROUP_ROWS,
):
    """Clone `base_cell` once per subgroup of its frame.

    The clones differ only in `frame` and in the `subgroup_dimension` /
    `subgroup_value` / `subgroup_flags` entries of `extra`, which `scoring._identity`
    lifts into columns. Provenance (coverage, manifest digest, determinism mode, seed)
    is inherited unchanged, because it describes the run rather than the slice.

    Returns a list, so a caller can count the cells before scoring them.
    """
    import dataclasses

    cells = []
    for dimension, value, subframe, flags in subgroup_frames(
        base_cell.frame, dimensions=dimensions, min_rows=min_rows
    ):
        extra = dict(base_cell.extra)
        extra.update({
            "subgroup_dimension": dimension,
            "subgroup_value": value,
            "subgroup_flags": ";".join(flags),
            "subgroup_n": int(len(subframe)),
        })
        cells.append(dataclasses.replace(base_cell, frame=subframe, extra=extra))
    return cells


def expand_cells(cells, dimensions=SUBGROUP_DIMENSIONS, min_rows=MIN_SUBGROUP_ROWS):
    """`cells` plus every subgroup slice of each. Whole-set rows come first.

    The overall cell is kept: disparity is defined relative to it, and a per-subgroup
    table without the whole-set number cannot answer "did overall accuracy move, or did
    one group's?".
    """
    expanded = []
    for cell in cells:
        expanded.append(cell)
        expanded.extend(
            subgroup_cells(cell, dimensions=dimensions, min_rows=min_rows)
        )
    return expanded


def annotate_small_subgroups(results):
    """Fold `subgroup_flags` into `status_flags` on a scored results frame.

    `score_cell` builds `status_flags` from what it observed while scoring and cannot
    know a slice was small, so the flag is merged here rather than lost.
    """
    if results.empty or "subgroup_flags" not in results.columns:
        return results

    results = results.copy()
    existing = results["status_flags"].fillna("")
    extra = results["subgroup_flags"].fillna("")
    results["status_flags"] = [
        ";".join(sorted({part for part in f"{left};{right}".split(";") if part}))
        for left, right in zip(existing, extra)
    ]
    return results


def disparity(
    results,
    metrics,
    group_keys=DISPARITY_GROUP_KEYS,
    include_small=False,
):
    """Per-metric disparity across subgroup values. Returns a tidy DataFrame.

    One row per (measurement, dimension, metric) with:

    ``disparity_range``   max - min across subgroup values. This is the definition
                          `evaluate_model` already reports as `race_gender_overall_bias`,
                          kept identical so the two are commensurable.
    ``disparity_mad``     mean |subgroup - overall| , matching
                          `race_gender_average_subgroup_bias`. Requires the whole-set
                          row; NaN without it.
    ``worst_value``       the metric's worst subgroup value, oriented by
                          `compare.HIGHER_IS_BETTER` / `LOWER_IS_BETTER`. NaN for a
                          metric in neither set, since "worst" is then undefined.
    ``worst_group``       which subgroup that was -- the number a fairness regression is
                          actually about.
    ``n_groups``          how many subgroups contributed.
    ``flags``             `small_subgroup` when any contributing group was under
                          `MIN_SUBGROUP_ROWS`, so a narrow range on thin groups is not
                          mistaken for parity.

    `include_small=False` still *counts* small groups in `n_groups` and flags them; it
    excludes them from the range and the worst-group pick, which is the conservative
    reading. Pass True to include them.
    """
    import pandas as pd

    from evaluation.uq.compare import metric_direction

    if results.empty:
        return pd.DataFrame()

    metrics = [metric for metric in metrics if metric in results.columns]
    if not metrics:
        return pd.DataFrame()

    keys = [key for key in group_keys if key in results.columns]
    if "subgroup_dimension" not in keys or "subgroup_value" not in results.columns:
        raise SubgroupError(
            "results must carry subgroup_dimension and subgroup_value columns; score "
            "with subgroups.expand_cells() before calling disparity()"
        )

    overall = results[results["subgroup_dimension"] == OVERALL_DIMENSION]
    subgroups = results[results["subgroup_dimension"] != OVERALL_DIMENSION]
    if subgroups.empty:
        return pd.DataFrame()

    # Overall value looked up on the measurement keys minus subgroup_dimension, since
    # the whole-set row's dimension is "overall" by definition.
    measurement_keys = [key for key in keys if key != "subgroup_dimension"]

    rows = []
    for group_values, group in subgroups.groupby(keys, dropna=False):
        identity = dict(zip(keys, _as_tuple(group_values)))
        small = _has_small(group)
        usable = group if include_small else group[~group.apply(_is_small, axis=1)]
        if usable.empty:
            # Every contributing group was too small. Report the row so the hole is
            # visible, rather than omitting the dimension entirely.
            usable = group.iloc[0:0]

        overall_row = _lookup_overall(overall, identity, measurement_keys)

        for metric in metrics:
            values = pd.to_numeric(usable[metric], errors="coerce").dropna()
            direction = metric_direction(metric)
            overall_value = (
                _numeric(overall_row.get(metric)) if overall_row is not None else np.nan
            )

            if values.empty:
                rows.append({
                    **identity, "metric": metric, "n_groups": int(len(group)),
                    "disparity_range": np.nan, "disparity_mad": np.nan,
                    "worst_value": np.nan, "worst_group": None,
                    "best_value": np.nan, "overall_value": overall_value,
                    "flags": SMALL_SUBGROUP if small else "",
                })
                continue

            worst_group, worst_value, best_value = _worst(
                usable, metric, values, direction
            )
            rows.append({
                **identity,
                "metric": metric,
                "n_groups": int(len(group)),
                "disparity_range": float(values.max() - values.min()),
                "disparity_mad": (
                    float(np.mean(np.abs(values.to_numpy() - overall_value)))
                    if not np.isnan(overall_value) else np.nan
                ),
                "worst_value": worst_value,
                "worst_group": worst_group,
                "best_value": best_value,
                "overall_value": overall_value,
                "flags": SMALL_SUBGROUP if small else "",
            })

    return pd.DataFrame(rows)


def disparity_as_results(disparity_frame):
    """Reshape `disparity` output to look like scored results, for a uniform diff.

    `compare.compare` joins two tidy results tables on measurement keys and metric
    columns. Emitting disparity in the same shape -- one row per measurement, with
    `disparity_range_<metric>` and `worst_value_<metric>` as columns and
    `subgroup_dimension` retained -- lets fairness movement travel through exactly the
    same comparison and direction logic as accuracy, instead of a parallel code path.
    """
    import pandas as pd

    if disparity_frame is None or disparity_frame.empty:
        return pd.DataFrame()

    identity = [
        column for column in disparity_frame.columns
        if column not in {
            "metric", "n_groups", "disparity_range", "disparity_mad",
            "worst_value", "worst_group", "best_value", "overall_value", "flags",
        }
    ]

    frames = []
    for metric, group in disparity_frame.groupby("metric", dropna=False):
        wide = group[identity + ["n_groups", "worst_group", "flags"]].copy()
        wide[f"disparity_range_{metric}"] = group["disparity_range"].to_numpy()
        wide[f"disparity_mad_{metric}"] = group["disparity_mad"].to_numpy()
        wide[f"worst_group_{metric}"] = group["worst_value"].to_numpy()
        frames.append(wide)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame.drop(columns=["n_groups", "worst_group", "flags"], errors="ignore"),
            on=identity, how="outer",
        )
    merged["subgroup_value"] = "disparity"
    merged["status"] = "ok"
    return merged


# -- internals ------------------------------------------------------------------ #


def _worst(frame, metric, values, direction):
    """(group, worst value, best value) for `metric` under `direction`."""
    import pandas as pd

    series = pd.to_numeric(frame[metric], errors="coerce")
    if direction == "higher":
        index = series.idxmin()
        return _plain(frame.loc[index, "subgroup_value"]), float(values.min()), float(values.max())
    if direction == "lower":
        index = series.idxmax()
        return _plain(frame.loc[index, "subgroup_value"]), float(values.max()), float(values.min())
    # Neither better nor worse: a range is still meaningful, a "worst" is not.
    return None, np.nan, np.nan


def _lookup_overall(overall, identity, measurement_keys) -> Optional[Dict]:
    """The whole-set row matching `identity` on `measurement_keys`, or None."""
    if overall.empty:
        return None
    mask = np.ones(len(overall), dtype=bool)
    for key in measurement_keys:
        if key not in identity:
            continue
        mask &= (overall[key] == identity[key]).to_numpy()
    matched = overall[mask]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _is_small(row):
    flags = str(row.get("status_flags") or "")
    return SMALL_SUBGROUP in flags.split(";")


def _has_small(frame):
    return bool(frame.apply(_is_small, axis=1).any())


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def _plain(value):
    """A JSON- and CSV-friendly scalar. numpy ints would otherwise render as `np.int64`."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


__all__ = [
    "DEMOGRAPHIC_COLUMNS", "DISPARITY_GROUP_KEYS", "MIN_SUBGROUP_ROWS",
    "OVERALL_DIMENSION", "OVERALL_VALUE", "SMALL_SUBGROUP", "SUBGROUP_DIMENSIONS",
    "SubgroupError", "annotate_small_subgroups", "available_dimensions", "disparity",
    "disparity_as_results", "expand_cells", "subgroup_cells", "subgroup_frames",
]
