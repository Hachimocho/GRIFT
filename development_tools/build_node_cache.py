#!/usr/bin/env python3
"""Build a node cache so real-data runs start in seconds instead of minutes.

`AIFaceDataset.load()` constructs every node in all three splits and reads the
`*_quality.csv` sidecars (19 GB) *before* `--cached-nodes` truncates anything, so even
a nominally tiny run pays the full load. Once a cache exists, `--use-cached` returns
from `load_and_prepare_data_splits` before the dataset is ever constructed.

The cache is written by the **production** writer, `save_cached_nodes`, rather than by
a hand-rolled pickle. That is deliberate: `load_cached_nodes` accepts three historical
formats and rejects anything else *silently* -- returning `None`, which the caller
reads as "no cache" and falls back to a full load. `web_ui/app.py` produced exactly
such a rejected cache for as long as it built the dict itself. Routing every writer
through one function is what keeps the format from drifting again.

Size: a 512-float64 `face_embedding` is ~4.7 KB, so a node costs ~5 KB and a full
1.6M-node cache would be ~7.5 GB. `--max-nodes-per-split` caps the *full* lists, which
is what you want for anything short of a production run -- `--cached-nodes` only sizes
the balanced view, so without the cap even a 2k-node run writes all 1.6M nodes to disk.

Usage:
    # A small cache for testing and short runs (~2k nodes/split, ~30 MB).
    python development_tools/build_node_cache.py --max-nodes-per-split 2000

    # The full thing, for real experiments.
    python development_tools/build_node_cache.py --out node_cache/cached_nodes.pkl
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ImageFileData import ImageFileData
from datasets.AIFaceDataset import AIFaceDataset
from nodes.atrnode import AttributeNode
from test_helpers.data_graph_utils import (
    load_cached_nodes,
    resolve_ai_face_data_root,
    save_cached_nodes,
)
from test_helpers.determinism import configure_determinism, rng_for

SPLITS = ("train", "val", "test")

#: Attributes that only exist if the quality-sidecar join worked. A cache built without
#: them silently disables graph-distance uncertainty for every run that uses it, which
#: is worth refusing rather than discovering three experiments later.
QUALITY_ATTRIBUTES = ("blur", "brightness", "contrast", "face_embedding")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a node cache via the production save_cached_nodes writer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default=os.path.join("node_cache", "cached_nodes.pkl"),
        help="Cache file to write. This is what --cache-file should point at.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="AI-Face root. Defaults to resolve_ai_face_data_root's search order.",
    )
    parser.add_argument(
        "--cached-nodes", type=int, default=1000,
        help="Size of the *balanced* view stored alongside each full split.",
    )
    parser.add_argument(
        "--max-nodes-per-split", type=int, default=None,
        help="Cap the full lists at this many nodes per split (deterministic "
             "subsample). Omit to cache every node -- ~7.5 GB on the real dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seeds the subsample, so the same flags always yield the same cache.",
    )
    parser.add_argument(
        "--atr-threshold", type=int, default=2,
        help="AttributeNode threshold, matching --atr-threshold on the training CLI.",
    )
    parser.add_argument(
        "--allow-missing-quality", action="store_true",
        help="Write the cache even if nodes carry no quality attributes. Only useful "
             "against a dataset root with no *_quality.csv sidecars.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing cache file.",
    )
    return parser.parse_args(argv)


def subsample(nodes, limit, split):
    """Deterministically reduce `nodes` to `limit` entries.

    Sorted first so the result cannot depend on the dataset's emission order, then
    sampled from a per-split RNG stream so it is independent both of every other
    random decision in the process and of the order the splits are handled in.
    """
    if limit is None or len(nodes) <= limit:
        return list(nodes)
    ordered = sorted(nodes, key=lambda node: node.node_id)
    stream = rng_for(f"cache.node_subsample.{split}")
    chosen = stream.sample(range(len(ordered)), limit)
    # Keep them in node_id order rather than sample order: a cache is a set, and
    # sorted output makes two caches diffable.
    return [ordered[index] for index in sorted(chosen)]


def quality_coverage(nodes):
    """Fraction of nodes carrying every attribute the sidecar join supplies."""
    if not nodes:
        return 0.0
    complete = sum(
        1 for node in nodes
        if all(name in node.attributes for name in QUALITY_ATTRIBUTES)
    )
    return complete / len(nodes)


def main(argv=None):
    args = parse_args(argv)

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} already exists. Pass --force to overwrite.")
        return 1

    # The subsample draws from a named stream, so it needs the seeding machinery
    # configured. `fast` is enough -- nothing here touches CUDA or a model.
    configure_determinism(seed=args.seed, mode="fast")

    data_root = resolve_ai_face_data_root(args.data_root)
    print(f"Dataset root: {data_root}")

    started = time.time()
    dataset = AIFaceDataset(
        data_root, ImageFileData, {}, AttributeNode,
        {"threshold": args.atr_threshold},
    )
    all_nodes = dataset.load()
    load_seconds = time.time() - started
    print(f"Loaded {len(all_nodes)} nodes in {load_seconds:.1f}s")

    by_split = {
        split: [node for node in all_nodes if node.split == split] for split in SPLITS
    }
    for split in SPLITS:
        available = len(by_split[split])
        by_split[split] = subsample(by_split[split], args.max_nodes_per_split, split)
        print(f"  {split}: {available} available -> {len(by_split[split])} cached")

    empty = [split for split in SPLITS if not by_split[split]]
    if empty:
        raise SystemExit(f"No nodes for split(s) {empty}; refusing to write a cache.")

    coverage = {split: quality_coverage(by_split[split]) for split in SPLITS}
    for split in SPLITS:
        print(f"  {split}: quality-attribute coverage {coverage[split]:.1%}")
    worst = min(coverage.values())
    if worst == 0.0 and not args.allow_missing_quality:
        raise SystemExit(
            "Nodes carry none of "
            f"{QUALITY_ATTRIBUTES} -- the quality-sidecar join produced nothing, so "
            "graph-distance uncertainty would have no inputs on every run that uses "
            "this cache. Check that *_quality.csv exist under the dataset root, or "
            "pass --allow-missing-quality if that is intended."
        )

    # target_num_nodes sizes the balanced view; it cannot exceed the smallest split.
    target = min(args.cached_nodes, min(len(by_split[split]) for split in SPLITS))
    if target != args.cached_nodes:
        print(f"Clamping balanced target {args.cached_nodes} -> {target} "
              f"(smallest cached split)")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_cached_nodes(
        by_split["train"], by_split["val"], by_split["test"], out,
        target_num_nodes=target,
    )

    # Read it back through the production reader: writing a cache the loader rejects
    # is the exact failure this script exists to prevent, and it is silent.
    for split in SPLITS:
        for balanced in (False, True):
            loaded = load_cached_nodes(out, split, balanced=balanced)
            if not loaded:
                raise SystemExit(
                    f"load_cached_nodes rejected the cache we just wrote "
                    f"(split={split}, balanced={balanced})."
                )

    size_mb = os.path.getsize(out) / 1e6
    manifest = {
        "cache_file": out,
        "data_root": data_root,
        "seed": args.seed,
        "cached_nodes": target,
        "max_nodes_per_split": args.max_nodes_per_split,
        "atr_threshold": args.atr_threshold,
        "counts": {split: len(by_split[split]) for split in SPLITS},
        "quality_coverage": coverage,
        "size_mb": round(size_mb, 2),
        "dataset_load_seconds": round(load_seconds, 1),
    }
    manifest_path = out + ".meta.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"\nWrote {out} ({size_mb:.1f} MB) and {manifest_path}")
    print(f"Use it with: --use-cached --cache-file {out} --cached-nodes {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
