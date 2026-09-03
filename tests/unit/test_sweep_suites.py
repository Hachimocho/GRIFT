"""Matrix expansion for the development sweep, and the gates that prune it.

Two classes of failure these exist for.

**A silently dropped config key.** `GPUQueueManager._build_command_args` ignores anything
absent from `ARG_MAPPING`, so a sweep cell whose config carries an unrouted key launches a
run that quietly differs from the plan and reports numbers that look real. Every cell of
every suite is checked against the table.

**A cell that runs but measures nothing.** Rewiring without an I-value predictor reads a
neutral default for every node and changes nothing; a subcluster traversal without
python-louvain silently runs its non-subcluster fallback. Both produce a plausible row for
a configuration that was never actually exercised, which is worse than an empty one.

Nothing here launches a process or constructs a GPUQueueManager.
"""

import json
import os

import pytest

from development_tools import sweep_suites
from development_tools.sweep_suites import (
    AXES, FORCED, SUITES, SuiteError, axis_constraints, expand, format_plan, load_suite,
    summarize,
)
from web_ui.gpu_queue_manager import validate_config_keys

SUITE_NAMES = tuple(SUITES)


@pytest.fixture
def no_louvain(monkeypatch):
    """Force the louvain probe to report absent, whatever the environment has."""
    monkeypatch.setattr(sweep_suites, "_louvain_available", lambda: False)


@pytest.fixture
def with_louvain(monkeypatch):
    """Force the louvain probe to report present, so subcluster cells are not gated."""
    monkeypatch.setattr(sweep_suites, "_louvain_available", lambda: True)


# -- config routing ------------------------------------------------------------- #

@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_every_cell_config_reaches_the_cli(suite_name):
    """No cell may carry a key the queue would drop on the way to the CLI."""
    cells = expand(load_suite(suite_name))
    for cell in cells:
        unroutable = validate_config_keys(cell.config)
        assert not unroutable, (
            f"{suite_name}/{cell.cell_id} carries key(s) {unroutable} that are not in "
            f"ARG_MAPPING, so they would be silently dropped and the run would differ "
            f"from the plan"
        )


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_records_are_always_requested(suite_name):
    """Without records there is nothing to score: the sweep's only input is the tables."""
    for cell in expand(load_suite(suite_name)):
        assert cell.config["uq_records"] is True
        # val too, or temperature scaling has to be fitted on the test split.
        assert "val" in cell.config["uq_records_splits"]


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_forced_settings_survive_a_suite_that_disagrees(suite_name):
    """A suite cannot turn off what the comparison depends on."""
    suite = load_suite(suite_name)
    suite["reference"]["uq_records"] = False
    suite["reference"]["determinism"] = "fast"
    for cell in expand(suite):
        assert cell.config["uq_records"] is True
        assert cell.config["determinism"] == "strict"


def test_forced_overrides_let_an_explicit_flag_win():
    """`--determinism fast` must beat the forced default; nothing else may."""
    suite = load_suite("smoke")
    suite["forced_overrides"] = {"determinism": "fast"}
    for cell in expand(suite):
        assert cell.config["determinism"] == "fast"
        assert cell.config["uq_records"] is True


# -- expansion shape ------------------------------------------------------------ #

def test_one_axis_at_a_time_is_additive_not_multiplicative(with_louvain):
    """The whole point of the design: a sum of axis sizes, not their product."""
    suite = load_suite("standard")
    cells = expand(suite)
    expected = 1 + sum(len(AXES[axis]) for axis in suite["axes"])
    assert len(cells) == expected
    # And the product it is avoiding is much larger.
    product = 1
    for axis in suite["axes"]:
        product *= len(AXES[axis]) + 1
    assert product > 5 * expected


def test_reference_cell_is_always_present():
    """Variants are variants *of* something; without the reference there is no baseline."""
    for suite_name in SUITE_NAMES:
        cells = expand(load_suite(suite_name))
        assert cells[0].cell_id == "reference"
        assert cells[0].axis == "reference"


def test_cell_ids_are_unique(with_louvain):
    for suite_name in SUITE_NAMES:
        ids = [cell.cell_id for cell in expand(load_suite(suite_name))]
        assert len(ids) == len(set(ids)), f"{suite_name} has duplicate cell ids"


def test_cross_expands_factorially(with_louvain):
    suite = load_suite("standard")
    cells = expand(suite, axes=["arch", "traversal"], cross=["arch", "traversal"])
    # Reference plus every (arch, traversal) pair.
    assert len(cells) == 1 + len(AXES["arch"]) * len(AXES["traversal"])
    assert any("+" in cell.cell_id for cell in cells)


def test_cross_leaves_other_axes_one_at_a_time(with_louvain):
    suite = load_suite("standard")
    cells = expand(suite, cross=["arch", "traversal"])
    expected = (
        1
        + len(AXES["arch"]) * len(AXES["traversal"])
        + sum(len(AXES[axis]) for axis in ("graph", "head", "updater"))
    )
    assert len(cells) == expected


def test_only_filters_by_axis_and_by_value(with_louvain):
    suite = load_suite("standard")
    by_axis = expand(suite, only=["traversal"])
    assert {cell.axis for cell in by_axis} == {"reference", "traversal"}

    by_value = expand(suite, only=["traversal=i-value"])
    assert [cell.cell_id for cell in by_value] == ["reference", "traversal=i-value"]


def test_only_accepts_a_comma_list():
    cells = expand(load_suite("standard"), only=["arch,head"])
    assert {cell.axis for cell in cells} == {"reference", "arch", "head"}


def test_only_refuses_a_selector_that_matches_nothing():
    """A typo must fail loudly: silently reducing coverage is the failure mode."""
    with pytest.raises(SuiteError, match="matched nothing"):
        expand(load_suite("standard"), only=["traversal=no-such-traversal"])


def test_unknown_axis_is_an_error():
    with pytest.raises(SuiteError, match="unknown axis"):
        expand(load_suite("standard"), axes=["nonexistent"])


def test_unknown_suite_lists_the_available_ones():
    with pytest.raises(SuiteError, match="standard"):
        load_suite("no-such-suite")


def test_description_matches_the_runner_naming():
    """The sweep locates records by this name, so it must match `test_configs`."""
    for cell in expand(load_suite("standard")):
        assert cell.description == f"{cell.detector}_{cell.config['traversal_type']}"


# -- gating --------------------------------------------------------------------- #

def test_broken_detectors_are_gated_with_their_reason():
    suite = load_suite("smoke")
    suite["extra_axes"] = {"arch": {"xceptiondf": {"architectures": ["xceptiondf"]}}}
    suite["axes"] = ["arch"]
    cells = expand(suite)
    broken = [cell for cell in cells if cell.cell_id == "arch=xceptiondf"]
    assert len(broken) == 1
    assert not broken[0].runnable
    assert "xceptiondf" in broken[0].skip_reason


def test_allow_broken_lets_the_failure_path_be_tested():
    suite = load_suite("smoke")
    suite["extra_axes"] = {"arch": {"xceptiondf": {"architectures": ["xceptiondf"]}}}
    suite["axes"] = ["arch"]
    cells = expand(suite, allow_broken=True)
    broken = [cell for cell in cells if cell.cell_id == "arch=xceptiondf"][0]
    assert broken.runnable


def test_performance_updater_needs_an_ivalue_traversal():
    reason = axis_constraints({
        "graph_manager": "performance", "traversal_type": "random",
    })
    assert reason and "i-value" in reason


def test_performance_updater_is_allowed_with_an_ivalue_traversal():
    assert axis_constraints({
        "graph_manager": "performance", "traversal_type": "i-value",
    }) is None


@pytest.mark.parametrize("strategy", ("max_ival", "min_ival", "mix_max_ival"))
def test_ivalue_reduction_needs_an_ivalue_traversal(strategy):
    reason = axis_constraints({
        "reduction_enabled": True, "reduction_strategy": strategy,
        "reduction_percentage": 10.0, "traversal_type": "comprehensive",
    })
    assert reason and "get_i_value" in reason


def test_random_reduction_works_with_any_traversal():
    assert axis_constraints({
        "reduction_enabled": True, "reduction_strategy": "random",
        "reduction_percentage": 10.0, "traversal_type": "comprehensive",
    }) is None


def test_reduction_that_removes_nothing_is_gated():
    assert axis_constraints({
        "reduction_enabled": True, "reduction_strategy": "none",
        "reduction_percentage": 10.0, "traversal_type": "random",
    })
    assert axis_constraints({
        "reduction_enabled": True, "reduction_strategy": "random",
        "reduction_percentage": 0.0, "traversal_type": "random",
    })


def test_every_subclustered_graph_type_is_now_inert(with_louvain):
    """The two traversals that read `graph.subclusters` were removed, so subcluster
    assignment can no longer change any run: it writes node attributes and leaves edges
    alone, and nothing reads those attributes. Every `*_subclustered` graph type is
    therefore refused rather than silently producing a cell identical to its plain twin --
    which is what three real sweep cells did."""
    assert sweep_suites.SUBCLUSTER_TRAVERSALS == frozenset()
    for graph_type in ("clustered_subclustered", "nonclustered_subclustered"):
        reason = axis_constraints({
            "graph_type": graph_type, "traversal_type": "i-value",
        })
        assert reason and "identical" in reason
        assert "removed" in reason


@pytest.mark.parametrize("graph_type", ("clustered", "nonclustered"))
def test_plain_graph_types_are_allowed(with_louvain, graph_type):
    assert axis_constraints({
        "graph_type": graph_type, "traversal_type": "i-value",
    }) is None


def test_subclustering_is_gated_without_louvain(no_louvain):
    """A no-op subcluster assignment must not produce a row claiming to test it.

    Reached only when the inertness check above does not already refuse the cell, which is
    why it asks for a traversal that would read subclusters if one existed.
    """
    monkey = frozenset({"i-value"})
    original = sweep_suites.SUBCLUSTER_TRAVERSALS
    sweep_suites.SUBCLUSTER_TRAVERSALS = monkey
    try:
        reason = axis_constraints({
            "graph_type": "nonclustered_subclustered", "traversal_type": "i-value",
        })
    finally:
        sweep_suites.SUBCLUSTER_TRAVERSALS = original
    assert reason and "python-louvain" in reason


# -- identical-record detection -------------------------------------------------- #
#
# The safeguard for the general class of failure this axis work uncovered: two cells that
# differ in configuration but produce the same record digest are one measurement wearing
# two names, and every metric for both is the same number.

def test_identical_record_groups_finds_duplicates():
    from development_tools.sweep import identical_record_groups

    groups = identical_record_groups({
        "aaa": ["traversal=i-value", "updater=performance", "updater=reduce_max_ival"],
        "bbb": ["reference"],
        "ccc": ["updater=reduce_random", "updater=reduce_restore"],
    })
    assert set(groups) == {"aaa", "ccc"}
    assert groups["aaa"] == [
        "traversal=i-value", "updater=performance", "updater=reduce_max_ival"
    ]
    # A digest belonging to one cell is not a finding.
    assert "bbb" not in groups


def test_identical_record_groups_on_distinct_digests():
    from development_tools.sweep import identical_record_groups

    assert identical_record_groups({"a": ["x"], "b": ["y"]}) == {}
    assert identical_record_groups({}) == {}
    assert identical_record_groups(None) == {}


def test_no_axis_cell_pairs_a_subclustered_graph_with_a_blind_traversal(with_louvain):
    """Stated over the suites rather than over one hand-built config."""
    for suite_name in SUITE_NAMES:
        for cell in expand(load_suite(suite_name)):
            if not cell.runnable:
                continue
            assert not str(cell.config["graph_type"]).endswith("_subclustered")


def test_every_updater_variant_is_either_runnable_or_explained(with_louvain):
    """No updater cell may be gated for a reason the user cannot act on."""
    cells = expand(load_suite("standard"), axes=["updater"])
    for cell in cells:
        if cell.axis != "updater":
            continue
        assert cell.runnable, f"{cell.cell_id} gated: {cell.skip_reason}"


def test_gated_cells_are_kept_with_a_reason_not_dropped():
    """A matrix with explained holes is honest; one with missing rows is not."""
    suite = load_suite("standard")
    suite["extra_axes"] = {"graph": {
        "nonclustered_subclustered": {"graph_type": "nonclustered_subclustered"},
    }}
    suite["axes"] = ["graph"]
    cells = expand(suite)
    gated = [cell for cell in cells if not cell.runnable]
    assert gated, "expected the inert subclustered cell to be gated"
    for cell in gated:
        assert cell.skip_reason
        assert cell.cell_id in format_plan(cells)


# -- suite files ---------------------------------------------------------------- #

def test_suite_file_merges_over_the_builtin(tmp_path):
    """An override changes one key without restating the axes."""
    path = tmp_path / "variant.json"
    path.write_text(json.dumps({"smoke": {"reference": {"num_epochs": 7}}}))
    suite = load_suite("smoke", suite_file=str(path))
    assert suite["reference"]["num_epochs"] == 7
    # Untouched keys survive the merge.
    assert suite["reference"]["architectures"] == SUITES["smoke"]["reference"]["architectures"]
    assert suite["axes"] == SUITES["smoke"]["axes"]


def test_suite_file_accepts_a_bare_suite_dict(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"reference": {"num_epochs": 2}, "axes": ["head"]}))
    suite = load_suite("smoke", suite_file=str(path))
    assert suite["reference"]["num_epochs"] == 2
    assert suite["axes"] == ["head"]


def test_axis_limit_caps_variants():
    cells = expand(load_suite("smoke"))
    per_axis = [cell for cell in cells if cell.axis == "traversal"]
    assert len(per_axis) == SUITES["smoke"]["axis_limit"]


# -- reporting helpers ---------------------------------------------------------- #

def test_summarize_counts_match_the_cells(no_louvain):
    cells = expand(load_suite("standard"))
    counts = summarize(cells)
    assert counts["total"] == len(cells)
    assert counts["runnable"] + counts["skipped"] == counts["total"]
    assert counts["runnable"] == sum(1 for cell in cells if cell.runnable)


def test_to_dict_round_trips_through_json():
    """The manifest is JSON, so a cell must serialize without a custom encoder."""
    for cell in expand(load_suite("standard")):
        json.loads(json.dumps(cell.to_dict()))


def test_forced_keys_are_all_routable():
    """FORCED is applied to every cell, so an unrouted key there breaks every suite."""
    assert not validate_config_keys(dict(FORCED))


# -- path resolution ------------------------------------------------------------ #
#
# `GPUQueueManager` launches every cell with `cwd=<repo root>`, whatever directory the
# caller stood in. When the sweep resolved its own paths against the caller's cwd instead,
# running it from `development_tools/` sent the cells' records to `<repo>/run_outputs/` while
# the sweep looked under `development_tools/run_outputs/` -- three successful runs reported
# as having produced no records, plus a stray directory tree beside the script.

def test_sweep_paths_are_absolute():
    from development_tools import sweep

    for name in ("REPO_ROOT", "RUN_OUTPUTS_DIR", "SWEEPS_DIR", "BASELINES_DIR", "RUNS_DIR"):
        value = getattr(sweep, name)
        assert os.path.isabs(value), f"sweep.{name} is cwd-relative: {value!r}"


def test_sweep_paths_do_not_move_with_the_working_directory(tmp_path, monkeypatch):
    import importlib

    from development_tools import sweep

    before = (sweep.RUN_OUTPUTS_DIR, sweep.SWEEPS_DIR, sweep.BASELINES_DIR)
    monkeypatch.chdir(tmp_path)
    reloaded = importlib.reload(sweep)
    try:
        after = (
            reloaded.RUN_OUTPUTS_DIR, reloaded.SWEEPS_DIR, reloaded.BASELINES_DIR
        )
        assert before == after
    finally:
        # Leave the module as the rest of the session expects to find it.
        monkeypatch.undo()
        importlib.reload(sweep)


def test_sweep_run_outputs_matches_the_queues_launch_directory():
    """The sweep must read from the same tree the cells write to."""
    from development_tools import sweep
    from web_ui import gpu_queue_manager

    launch_cwd = os.path.dirname(
        os.path.dirname(os.path.abspath(gpu_queue_manager.__file__))
    )
    assert sweep.RUN_OUTPUTS_DIR == os.path.join(launch_cwd, "run_outputs")


def test_queue_runs_default_runs_dir_is_absolute():
    from development_tools import queue_runs

    assert os.path.isabs(queue_runs.DEFAULT_RUNS_DIR)


# -- --set passthrough ---------------------------------------------------------- #
#
# `sweep.py` deliberately mirrors only a handful of the training CLI's ~90 flags. `--set`
# is how the rest are reachable without the two surfaces drifting apart, and it validates
# against ARG_MAPPING so an unrouted key is refused rather than silently dropped.

def test_set_parses_types_not_just_strings():
    """`_build_command_args` emits a bare flag for True and nothing for False, so a
    boolean arriving as the string "true" would be passed as a *value* instead."""
    from development_tools.sweep import parse_overrides

    parsed = parse_overrides([
        "fair_train=true", "fair_test=False", "num_workers=8",
        "removal_fraction=0.03", "traversal_type=random", "data_root=none",
    ])
    assert parsed == {
        "fair_train": True, "fair_test": False, "num_workers": 8,
        "removal_fraction": 0.03, "traversal_type": "random", "data_root": None,
    }
    assert isinstance(parsed["num_workers"], int)
    assert isinstance(parsed["removal_fraction"], float)


def test_set_accepts_the_dashed_spelling():
    """The training flag is `--fair-train`; only the config key uses underscores."""
    from development_tools.sweep import parse_overrides

    assert parse_overrides(["fair-train=true"]) == {"fair_train": True}
    assert parse_overrides(["--fair-train=true"]) == {"fair_train": True}


def test_set_refuses_an_unrouted_key():
    from development_tools.sweep import SweepError, parse_overrides

    with pytest.raises(SweepError, match="not a run-config key"):
        parse_overrides(["no_such_setting=1"])


def test_set_suggests_a_near_miss():
    from development_tools.sweep import SweepError, parse_overrides

    with pytest.raises(SweepError, match="num_epochs"):
        parse_overrides(["epochs=3"])
    with pytest.raises(SweepError, match="fair_train"):
        parse_overrides(["fair_trian=true"])


def test_set_requires_a_value():
    from development_tools.sweep import SweepError, parse_overrides

    with pytest.raises(SweepError, match="KEY=VALUE"):
        parse_overrides(["fair_train"])


def test_set_reaches_every_cell_and_beats_forced():
    """A command-line flag is the most specific instruction; it must win over FORCED."""
    from development_tools.sweep import apply_overrides

    suite = load_suite("standard")

    class Args:
        overrides = ["fair_train=true", "determinism=fast"]
        data_root = cache_file = seed = num_epochs = determinism = None

    apply_overrides(suite, Args())
    cells = expand(suite)
    assert cells, "expansion produced no cells"
    for cell in cells:
        assert cell.config["fair_train"] is True
        assert cell.config["determinism"] == "fast"
        # And it still cannot break what the comparison depends on.
        assert cell.config["uq_records"] is True


def test_set_values_are_all_routable():
    """Whatever --set accepts must survive the queue's allowlist by construction."""
    from development_tools.sweep import parse_overrides

    parsed = parse_overrides(["fair_train=true", "num_workers=8", "holdout=none"])
    assert not validate_config_keys(parsed)


def test_relative_artifact_paths_resolve_against_the_repo_root():
    """determinism.json records `run_outputs/...` relative to the runner's cwd."""
    from development_tools.sweep import REPO_ROOT, _resolve_artifact

    resolved = _resolve_artifact("run_outputs/run_x/cfg/records_test.csv.gz")
    assert resolved == os.path.join(
        REPO_ROOT, "run_outputs/run_x/cfg/records_test.csv.gz"
    )
    # An absolute path is already anchored and must be left alone.
    assert _resolve_artifact("/tmp/records.csv.gz") == "/tmp/records.csv.gz"
    assert _resolve_artifact(None) is None
