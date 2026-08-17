"""The sweep's identity merge must not collide with the scorer's own columns.

`_score` merges a per-cell identity frame onto the scored results on `label`. Any column
present in both is renamed by pandas to `<name>_x` / `<name>_y`, which leaves neither
usable -- and because the renamed pair is absent from `compare.IGNORED_COLUMNS`, both then
surface as unclassified numeric metrics in every comparison. It happened with `threshold`.
"""

import pandas as pd
import pytest

from development_tools import sweep
from evaluation.uq.compare import IGNORED_COLUMNS, metric_direction
from evaluation.uq.scoring import Cell, score_cells


def scored_columns():
    """The columns `score_cells` emits, from a minimal real scoring pass."""
    frame = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(40)],
        "label": [0, 1] * 20,
        "prob": [0.2, 0.8] * 20,
        "pred": [0, 1] * 20,
        "correct": [1] * 40,
    })
    return set(score_cells(
        [Cell(detector="tiny", method_id="baseline_maxprob",
              score_column="u_maxprob", frame=frame, extra={"label": "cell"})],
        require_comparable=False,
    ).columns)


#: The keys `_score` puts in its identity frame. Kept as a literal so a new one added
#: there without checking against the scorer fails this test.
IDENTITY_KEYS = {
    "label", "cell_id", "axis", "axis_value", "arch", "traversal", "graph_type",
    "uncertainty_head", "graph_manager", "run_id", "records_path", "duration_seconds",
}


def test_identity_keys_do_not_collide_with_scored_columns():
    overlap = (IDENTITY_KEYS & scored_columns()) - {"label"}
    assert not overlap, (
        f"the sweep's identity frame reuses scored column name(s) {sorted(overlap)}; the "
        f"merge would rename both to _x/_y and neither would be usable"
    )


def test_the_identity_frame_in_the_source_matches_this_list():
    """Guards the literal above against drift in `_score`."""
    import inspect

    source = inspect.getsource(sweep._score)
    start = source.index("identity_rows.append({")
    block = source[start:source.index("})", start)]
    found = {
        line.split('"')[1] for line in block.splitlines()
        if line.strip().startswith('"') and '":' in line
    }
    assert found == IDENTITY_KEYS, (
        f"_score's identity keys {sorted(found)} differ from the tested set "
        f"{sorted(IDENTITY_KEYS)}"
    )


def test_threshold_is_emitted_once_and_is_ignored_by_the_comparison():
    """It comes from the scorer, and it is a setting rather than a score."""
    columns = scored_columns()
    assert "threshold" in columns
    assert "threshold" not in IDENTITY_KEYS
    for name in ("threshold", "clf_threshold"):
        assert name in IGNORED_COLUMNS
        assert metric_direction(name) is None


@pytest.mark.parametrize("suffixed", ["threshold_x", "threshold_y"])
def test_a_collided_column_would_be_caught_as_unclassified(suffixed):
    """The symptom that made the original collision visible."""
    from evaluation.uq.compare import unclassified_columns

    frame = pd.DataFrame({"cell_id": ["a"], suffixed: [0.99]})
    assert unclassified_columns(frame) == [suffixed]
