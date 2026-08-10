"""The ensemble launcher's config, and the queue manager's flag allowlist.

`GPUQueueManager._build_command_args` maps config keys to CLI flags through an explicit
`arg_mapping` dict and **silently drops anything unlisted**. That is the failure mode
these tests exist for: an ensemble launched with `determinism='strict'` ran non-strict
because `determinism` was absent from the table, with no warning anywhere. A missing
entry produces a run that looks correct and is not reproducible.

Nothing here launches a process or constructs a GPUQueueManager: `_build_command_args`
is called on an uninitialized instance, so no background threads start and no run
metadata is touched.
"""

import pytest

from development_tools.launch_ensemble import member_config, parse_args
from web_ui.gpu_queue_manager import GPUQueueManager

#: Config keys the launcher emits that must survive the allowlist. Every one of these
#: changes the numbers or the reproducibility of a member.
REQUIRED_FLAGS = {
    "determinism": "--determinism",
    "ensemble_member": "--ensemble-member",
    "ensemble_id": "--ensemble-id",
    "uq_records": "--uq-records",
    "uq_records_splits": "--uq-records-splits",
    "uncertainty_head": "--uncertainty-head",
    "seed": "--seed",
}


@pytest.fixture
def build_args():
    """`_build_command_args` without constructing a live queue manager."""
    manager = GPUQueueManager.__new__(GPUQueueManager)
    return lambda config, **kwargs: manager._build_command_args(config, **kwargs)


def launcher_args(*extra):
    return parse_args(["--ensemble-id", "ens_test", *extra])


# --------------------------------------------------------------------------- #
# The allowlist
# --------------------------------------------------------------------------- #

def test_every_flag_the_launcher_relies_on_survives(build_args):
    """The regression test for the silent-drop bug."""
    config = member_config(launcher_args("--determinism", "strict"), 1)
    command = build_args(config)
    missing = [
        flag for key, flag in REQUIRED_FLAGS.items()
        if key in config and flag not in command
    ]
    assert not missing, (
        f"{missing} were dropped by arg_mapping. Add them to "
        f"GPUQueueManager._build_command_args or queue-launched runs will silently "
        f"ignore them."
    )


def test_determinism_reaches_the_command_line_with_its_value(build_args):
    command = build_args(member_config(launcher_args("--determinism", "strict"), 0))
    assert command[command.index("--determinism") + 1] == "strict"


def test_member_index_reaches_the_command_line(build_args):
    command = build_args(member_config(launcher_args(), 2))
    assert command[command.index("--ensemble-member") + 1] == "2"


def test_member_zero_is_not_dropped(build_args):
    """`0` is falsy but is a real member index, not an absent one.

    A mapping that tested truthiness rather than presence would drop it, silently
    turning an N-member ensemble into N-1 members plus one un-indexed run -- which
    `assert_members_compatible` would then refuse for a confusing reason.
    """
    command = build_args(member_config(launcher_args(), 0))
    assert "--ensemble-member" in command
    assert command[command.index("--ensemble-member") + 1] == "0"


def test_uq_records_is_emitted_as_a_bare_flag(build_args):
    command = build_args(member_config(launcher_args(), 0))
    index = command.index("--uq-records")
    # store_true: a value after it would be parsed as a positional and rejected.
    assert index == len(command) - 1 or command[index + 1].startswith("--")


def test_val_records_are_requested(build_args):
    """Temperature scaling must be fitted on val, so val records are not optional."""
    command = build_args(member_config(launcher_args(), 0))
    splits = command[command.index("--uq-records-splits") + 1]
    assert "val" in splits and "test" in splits


def test_the_negation_of_an_on_by_default_flag_is_emitted(build_args):
    """`False` in arg_mapping emits *nothing*, which leaves the default in place.

    `--build-val-test-edges` and `--graph-distance-robust-stats` both default to on, so
    turning them off requires the explicit negation flag.
    """
    command = build_args({"build_val_test_edges": False,
                          "graph_distance_robust_stats": False})
    assert "--no-build-val-test-edges" in command
    assert "--no-graph-distance-robust-stats" in command


def test_on_by_default_flags_emit_nothing_when_left_on(build_args):
    command = build_args({"build_val_test_edges": True,
                          "graph_distance_robust_stats": True})
    assert "--no-build-val-test-edges" not in command
    assert "--no-graph-distance-robust-stats" not in command


def test_every_cli_flag_has_a_route_through_the_queue(build_args):
    """No CLI flag should be unreachable from a queue-launched run.

    Reflects over `args_utils`'s parser rather than a hand-maintained list, so a new
    flag fails here until it is either mapped or explicitly listed below as
    intentionally launcher-only. That is the point: the failure mode is a flag that
    appears to work and is discarded.
    """
    import re

    from test_helpers import args_utils
    from web_ui import gpu_queue_manager

    with open(args_utils.__file__) as handle:
        source = handle.read()
    cli_flags = set(re.findall(r"add_argument\('(--[a-z0-9-]+)'", source))

    with open(gpu_queue_manager.__file__) as handle:
        manager_source = handle.read()
    mapped = set(re.findall(r'"(--[a-z0-9-]+)"', manager_source))
    appended = set(re.findall(r'args\.append\("(--[a-z0-9-]+)"\)', manager_source))
    extended = set(re.findall(r'args\.extend\(\["(--[a-z0-9-]+)"', manager_source))

    #: Flags that legitimately have no config-key route.
    exempt = {
        # Supplied by the queue itself, from the run id it generated.
        "--run-id",
        # A shorthand for --determinism strict, which *is* mapped.
        "--strict-determinism",
        # Prints the holdout and corruption tables and exits. Queueing a run that
        # does nothing but print would be a bug, not a feature.
        "--list-holdouts",
    }
    # Negations of on-by-default flags: the queue emits the --no- form instead, so the
    # positive flag never needs a mapping.
    negations = {flag for flag in cli_flags if flag.startswith("--no-")}
    positives_with_negations = {
        flag for flag in cli_flags
        if f"--no-{flag[2:]}" in negations
    }

    unreachable = (
        cli_flags - mapped - appended - extended - exempt - negations
        - positives_with_negations
    )
    assert not unreachable, (
        f"{sorted(unreachable)} cannot be set from a queued run. Add them to "
        f"GPUQueueManager._build_command_args's arg_mapping, or to `exempt` above "
        f"with a reason."
    )


def test_a_list_valued_flag_is_comma_joined(build_args):
    """`['resnestdf']` must reach the CLI as `resnestdf`, not as a Python repr.

    The regression: only `graph_uncertainty_methods` was special-cased, so a list
    `architectures` -- which is how both the web UI and this launcher pass it -- arrived
    as the literal string "['resnestdf']" and failed architecture validation. Every
    queue-launched multi-architecture run was affected.
    """
    command = build_args({"architectures": ["resnestdf", "effnetdf"]})
    value = command[command.index("--architectures") + 1]
    assert value == "resnestdf,effnetdf"
    assert "[" not in value and "'" not in value


def test_the_launcher_config_survives_stringification(build_args):
    """End to end: what the launcher builds must produce a usable command line."""
    command = build_args(member_config(launcher_args("--arch", "resnestdf"), 0))
    value = command[command.index("--architectures") + 1]
    assert value == "resnestdf"


@pytest.mark.parametrize("key,flag", [
    ("architectures", "--architectures"),
    ("graph_uncertainty_methods", "--graph-uncertainty-methods"),
    ("traversal_sequence", "--traversal-sequence"),
])
def test_every_list_valued_flag_is_joined(build_args, key, flag):
    command = build_args({key: ["a", "b"]})
    assert command[command.index(flag) + 1] == "a,b"


def test_a_none_value_is_omitted_rather_than_stringified(build_args):
    """`str(None)` is "None", which argparse accepts as a literal for a str flag.

    So a config carrying an unset optional would silently set `--holdout None` rather
    than leaving the default -- and "None" is not a valid holdout id.
    """
    command = build_args({"holdout": None, "data_root": None})
    assert "--holdout" not in command
    assert "None" not in command


def test_use_cached_is_not_duplicated(build_args):
    """It is both an arg_mapping entry and had a separate append; one flag is enough."""
    command = build_args({"cached_nodes": True})
    assert command.count("--use-cached") == 1


def test_unknown_keys_are_still_dropped(build_args):
    """Documenting the behavior rather than endorsing it.

    The allowlist is a real safety property -- an arbitrary config key cannot inject
    a flag -- but it is also why every new CLI flag needs a matching entry.
    """
    command = build_args({"totally_made_up_option": "value"})
    assert "--totally-made-up-option" not in command
    assert "value" not in command


# --------------------------------------------------------------------------- #
# The member config
# --------------------------------------------------------------------------- #

def test_the_seed_is_identical_across_members():
    """The core design decision: members differ in initialization, not in data.

    Varying --seed would also change the graph (the cache key embeds it, so each
    member would rebuild it) and the data order, which confounds initialization
    variance with data-order variance.
    """
    args = launcher_args("--seed", "7")
    configs = [member_config(args, index) for index in range(3)]
    assert {config["seed"] for config in configs} == {7}
    assert [config["ensemble_member"] for config in configs] == [0, 1, 2]


def test_all_members_share_one_ensemble_id():
    args = launcher_args()
    configs = [member_config(args, index) for index in range(3)]
    assert {config["ensemble_id"] for config in configs} == {"ens_test"}


def test_members_share_the_detector_and_head():
    args = launcher_args("--arch", "effnetdf", "--uncertainty-head", "sngp")
    configs = [member_config(args, index) for index in range(2)]
    for config in configs:
        assert config["architectures"] == ["effnetdf"]
        assert config["uncertainty_head"] == "sngp"


def test_test_split_edges_are_requested():
    """Graph-distance methods need test edges, and the ensemble is scored on test."""
    assert member_config(launcher_args(), 0)["build_val_test_edges"] is True


def test_determinism_defaults_to_strict():
    """An ensemble is a paper artifact: a non-reproducible member cannot be redone."""
    assert launcher_args().determinism == "strict"


def test_the_node_cache_is_threaded_through_when_requested():
    args = launcher_args("--use-cached", "--cache-file", "node_cache/x.pkl",
                         "--cached-nodes", "500")
    config = member_config(args, 0)
    assert config["cached_nodes"] is True
    assert config["cache_file"] == "node_cache/x.pkl"
    assert config["cached_nodes_count"] == 500


def test_optional_knobs_are_absent_rather_than_none():
    """A None in the config would be emitted as the literal string "None"."""
    config = member_config(launcher_args(), 0)
    for key in ("data_root", "train_steps", "val_steps", "cached_nodes_count"):
        assert key not in config
    assert None not in config.values()
