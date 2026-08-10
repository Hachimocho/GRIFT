#!/usr/bin/env python3
"""Launch a deep ensemble and aggregate its members.

Members differ in **initialization only**: same `--seed`, same graph, same data order,
different `--ensemble-member`. That matters for two reasons. Experimentally, an
ensemble is supposed to measure initialization variance, so varying the training data
alongside it confounds the measurement. Practically, the graph cache key embeds the
seed whenever a split has edges, so N differently-seeded members would each rebuild the
train graph from scratch -- N times the expensive part of a run, for a worse experiment.

Runs are dispatched through `GPUQueueManager` in library mode, which already handles
GPU discovery, memory-based admission, `CUDA_VISIBLE_DEVICES` pinning, process
monitoring, and metadata. Exactly one manager is constructed: its
`reconcile_existing_runs()` rewrites in-flight run metadata, so a second live instance
would corrupt the first's bookkeeping.

Two phases, usable separately:

    # Launch, wait, then aggregate.
    python development_tools/launch_ensemble.py --members 3 --arch resnestdf \
        --num-epochs 3 --use-cached --cache-file node_cache/cached_nodes.pkl

    # Aggregate members that already finished.
    python development_tools/launch_ensemble.py --aggregate-only \
        --ensemble-id ens_20260807_a1b2c3d4

Aggregation refuses to average members whose detector, uncertainty head, seed, git
commit, or determinism mode differ -- see `evaluation/uq/ensemble.py`.
"""

import argparse
import json
import os
import secrets
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.uq.ensemble import (
    EnsembleCompatibilityError,
    aggregate_members,
    discover_members,
    save_ensemble,
)

#: Statuses `GPUQueueManager` uses for a run that will not progress further.
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "error", "crashed"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch a deep ensemble through the GPU queue and aggregate it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--members", type=int, default=3,
                        help="Number of ensemble members.")
    parser.add_argument("--ensemble-id", default=None,
                        help="Shared identifier. Generated if omitted; required with "
                             "--aggregate-only.")
    parser.add_argument("--arch", default="resnestdf",
                        help="Detector architecture. All members must share it.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed, held FIXED across members on purpose.")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--traversal-type", default="random")
    parser.add_argument("--uncertainty-head", default="none",
                        choices=["none", "evidential", "batchensemble", "sngp"])
    parser.add_argument("--mc-dropout-samples", type=int, default=0)
    parser.add_argument("--determinism", default="strict", choices=["strict", "fast"],
                        help="strict by default here: an ensemble is a paper artifact, "
                             "and non-reproducible members cannot be re-derived.")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--use-cached", action="store_true",
                        help="Load nodes from --cache-file. Strongly recommended: "
                             "without it every member pays the full dataset load.")
    parser.add_argument("--cache-file", default="node_cache/cached_nodes.pkl")
    parser.add_argument("--cached-nodes", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--graph-type", default="clustered")
    parser.add_argument("--run-outputs", default="run_outputs",
                        help="Where members write determinism.json and records.")
    parser.add_argument("--out", default=None,
                        help="Aggregated records path. Defaults to "
                             "run_outputs/ensembles/<ensemble-id>/records_test.csv.gz")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which member record table to aggregate.")
    parser.add_argument("--records-splits", default="val,test",
                        help="Splits each member records. Keep val unless you are "
                             "certain you will not fit temperature scaling: it has to "
                             "be fitted on data the reported test numbers never saw, "
                             "and strict mode forces one loader thread, so a val pass "
                             "over a large split is slow.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip launching; aggregate existing members.")
    parser.add_argument("--launch-only", action="store_true",
                        help="Launch and exit without waiting or aggregating.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Skip the sha256 check on member record tables.")
    return parser.parse_args(argv)


def member_config(args, member_index):
    """The run config for one member.

    Keys must match `GPUQueueManager._build_command_args`'s `arg_mapping`, which
    **silently drops anything unlisted** -- that is why `--determinism` was absent
    from queue-launched runs until it was added to that table.
    """
    config = {
        "architectures": [args.arch],
        "traversal_type": args.traversal_type,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "graph_type": args.graph_type,
        "determinism": args.determinism,
        "uncertainty_head": args.uncertainty_head,
        "mc_dropout_samples": args.mc_dropout_samples,
        # Records are the point: without them there is nothing to aggregate.
        "uq_records": True,
        "uq_records_splits": args.records_splits,
        "ensemble_member": member_index,
        "ensemble_id": args.ensemble_id,
        # Graph methods need test-split edges, and the ensemble is evaluated on test.
        "build_val_test_edges": True,
    }
    if args.use_cached:
        config["cached_nodes"] = True
        config["cache_file"] = args.cache_file
    if args.cached_nodes is not None:
        config["cached_nodes_count"] = args.cached_nodes
    if args.data_root:
        config["data_root"] = args.data_root
    if args.train_steps is not None:
        config["train_steps"] = args.train_steps
    if args.val_steps is not None:
        config["val_steps"] = args.val_steps
    return config


def launch(args):
    """Queue every member. Returns the list of run ids."""
    from web_ui.gpu_queue_manager import GPUQueueManager

    # One manager only: reconcile_existing_runs() rewrites in-flight metadata, so a
    # second instance would clobber the first's bookkeeping.
    manager = GPUQueueManager()
    run_ids = []
    try:
        for member_index in range(args.members):
            config = member_config(args, member_index)
            run_id = manager.queue_run(
                config_name=f"{args.ensemble_id}_m{member_index}",
                config=config,
                # Descending priority so members start in index order, which makes the
                # logs readable. It does not affect the result.
                priority=args.members - member_index,
            )
            run_ids.append(run_id)
            print(f"  queued member {member_index}: {run_id}")

        if args.launch_only:
            return run_ids, manager
        wait_for(manager, run_ids, args)
        return run_ids, manager
    finally:
        if args.launch_only:
            # Leave the queue running: the caller wants the members to proceed after
            # this process exits.
            pass
        else:
            manager.shutdown()


def wait_for(manager, run_ids, args):
    """Block until every run reaches a terminal status, or the timeout elapses."""
    deadline = time.time() + args.timeout_hours * 3600.0
    reported = {}
    while time.time() < deadline:
        statuses = {}
        for run_id in run_ids:
            metadata = manager.get_run(run_id) or {}
            statuses[run_id] = metadata.get("status", "unknown")

        for run_id, status in statuses.items():
            if reported.get(run_id) != status:
                print(f"  {run_id}: {status}")
                reported[run_id] = status

        if all(status in TERMINAL_STATUSES for status in statuses.values()):
            failed = {
                run_id: status for run_id, status in statuses.items()
                if status != "completed"
            }
            if failed:
                print(f"\nWARNING: {len(failed)} member(s) did not complete: {failed}")
                print("Aggregation will refuse to proceed unless enough members wrote "
                      "records; check web_ui/runs/<run_id> for the logs.")
            return statuses
        time.sleep(args.poll_seconds)

    raise SystemExit(
        f"Timed out after {args.timeout_hours}h waiting for members. The runs are "
        f"still going; re-run with --aggregate-only --ensemble-id {args.ensemble_id} "
        f"once they finish."
    )


def aggregate(args):
    """Discover, aggregate, and write. Returns the output path."""
    members = discover_members(
        args.run_outputs, ensemble_id=args.ensemble_id, split=args.split
    )
    print(f"\nDiscovered {len(members)} complete member(s) for "
          f"ensemble {args.ensemble_id!r}:")
    for member in members:
        print(f"  member {member.member_index}: {member.run_id} "
              f"-> {member.records_path}")

    if len(members) < 2:
        raise SystemExit(
            f"Need at least 2 members to aggregate, found {len(members)}. Each member "
            f"must have finished a configuration and written records for the "
            f"'{args.split}' split (pass --uq-records)."
        )

    try:
        frame, manifest = aggregate_members(
            members, verify=not args.allow_unverified
        )
    except EnsembleCompatibilityError as error:
        raise SystemExit(f"\n{error}") from error

    out = args.out or os.path.join(
        args.run_outputs, "ensembles", args.ensemble_id, f"records_{args.split}.csv.gz"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    digest = save_ensemble(frame, manifest, out)

    print(f"\nWrote {out}")
    print(f"  rows: {len(frame)}, members: {manifest['n_members']}, "
          f"sha256 {digest[:12]}")
    print(f"  mean member disagreement (variance of p): "
          f"{manifest['mean_disagreement']:.3e}")
    # Zero disagreement means the members are the same model. That is not a small
    # calibration issue -- the ensemble has no epistemic signal at all, and every
    # ensemble-specific metric is measuring nothing. It is exactly the failure
    # BatchEnsemble had with identical fast-weight initialization.
    if manifest["mean_disagreement"] == 0.0:
        print("  WARNING: members agree EXACTLY on every sample. Check that "
              "--ensemble-member actually varied and that seed_model_init ran before "
              "model construction.")
    accuracies = [
        entry["test_accuracy"] for entry in manifest["members"]
        if entry["test_accuracy"] is not None
    ]
    if accuracies:
        print(f"  member test accuracies: "
              f"{', '.join(f'{value:.4f}' for value in accuracies)}")
    return out


def main(argv=None):
    args = parse_args(argv)

    if args.aggregate_only:
        if not args.ensemble_id:
            raise SystemExit("--aggregate-only requires --ensemble-id")
        aggregate(args)
        return 0

    if not args.ensemble_id:
        # secrets, not random: this id must not consume from the seeded RNG, and it
        # must not be reproducible (two ensembles launched at the same seed are
        # different ensembles).
        args.ensemble_id = f"ens_{time.strftime('%Y%m%d')}_{secrets.token_hex(4)}"
    print(f"Ensemble {args.ensemble_id}: {args.members} members on {args.arch}, "
          f"seed {args.seed} (fixed), determinism {args.determinism}")

    if args.use_cached and not os.path.exists(args.cache_file):
        raise SystemExit(
            f"--use-cached given but {args.cache_file} does not exist. Build it with "
            f"development_tools/build_node_cache.py, or drop --use-cached and accept "
            f"a full dataset load per member."
        )

    run_ids, _manager = launch(args)

    launch_record = {
        "ensemble_id": args.ensemble_id,
        "members": args.members,
        "run_ids": run_ids,
        "arch": args.arch,
        "seed": args.seed,
        "determinism": args.determinism,
        "uncertainty_head": args.uncertainty_head,
    }
    record_dir = os.path.join(args.run_outputs, "ensembles", args.ensemble_id)
    os.makedirs(record_dir, exist_ok=True)
    with open(os.path.join(record_dir, "launch.json"), "w") as handle:
        json.dump(launch_record, handle, indent=2, sort_keys=True)

    if args.launch_only:
        print(f"\nLaunched. Aggregate later with:\n  python "
              f"{os.path.relpath(__file__)} --aggregate-only "
              f"--ensemble-id {args.ensemble_id}")
        return 0

    aggregate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
