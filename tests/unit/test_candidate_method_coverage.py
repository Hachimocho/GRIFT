"""The registry and the tools' candidate lists must not drift apart.

`score_cells` will score any method it is handed, but `sweep.py` and `uq_report.py` each
carry a hardcoded `CANDIDATE_METHODS` tuple naming which ones to try. Registering a method
without adding it there means the column is written to every record table and then never
scored -- which is exactly what happened to `ivalue`: 4,661 distinct values sitting in the
records of all 19 cells of a real sweep, absent from the results table, and the research
question it exists to answer silently unanswered.
"""

from development_tools.sweep import CANDIDATE_METHODS as SWEEP_METHODS
from development_tools.uq_report import CANDIDATE_METHODS as REPORT_METHODS
from evaluation.uq.registry import UQ_METHODS

#: Registry methods a per-cell sweep cannot produce, and why.
#:
#: `deep_ensemble` needs several same-config runs aggregated by `launch_ensemble.py`, which
#: is not expressible as one cell. `temperature_scaling` is fitted on val *after* a run and
#: applied by `uq_report`, so it appears when that fit is supplied rather than from a cell's
#: own columns.
INTENTIONALLY_ABSENT_FROM_SWEEP = frozenset({"deep_ensemble", "temperature_scaling"})


def test_every_candidate_is_a_real_method():
    """A typo here would silently drop a method rather than failing."""
    for name, methods in (("sweep", SWEEP_METHODS), ("report", REPORT_METHODS)):
        unknown = [m for m in methods if m not in UQ_METHODS]
        assert not unknown, f"{name}.CANDIDATE_METHODS names unregistered method(s): {unknown}"


def test_every_registered_method_is_scored_somewhere_or_explicitly_excluded():
    missing = sorted(set(UQ_METHODS) - set(SWEEP_METHODS) - INTENTIONALLY_ABSENT_FROM_SWEEP)
    assert not missing, (
        f"registered but never scored by the sweep: {missing}. Add them to "
        f"development_tools/sweep.py::CANDIDATE_METHODS, or to "
        f"INTENTIONALLY_ABSENT_FROM_SWEEP here with the reason."
    )


def test_the_report_covers_at_least_what_the_sweep_does():
    """They are separate lists; the report is the paper artifact, so it must not be narrower."""
    missing = sorted(set(SWEEP_METHODS) - set(REPORT_METHODS))
    assert not missing, f"uq_report scores fewer methods than the sweep: {missing}"


def test_the_ivalue_methods_are_scored():
    """The whole point of Claim 2: without these the question has no data behind it."""
    for method in ("ivalue", "ivalue_rank"):
        assert method in UQ_METHODS
        assert method in SWEEP_METHODS
        assert method in REPORT_METHODS


def test_no_candidate_list_has_duplicates():
    for name, methods in (("sweep", SWEEP_METHODS), ("report", REPORT_METHODS)):
        assert len(methods) == len(set(methods)), f"{name}.CANDIDATE_METHODS has duplicates"
