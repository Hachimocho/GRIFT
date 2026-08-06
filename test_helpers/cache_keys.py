"""Graph-cache key construction, in one place.

The key was previously built inline in ``test_hierarchical.py`` and *predicted*
separately in ``data_graph_utils.check_graph_cache_compatibility``, so the two could
drift -- the UI could report a cache hit while the pipeline computed a different key
and rebuilt.

More importantly the key omitted several inputs that materially change the produced
graph, so genuinely different graphs shared one cache entry:

``--seed``
    Edge construction *is* seed-dependent: any node that ends up isolated gets a
    randomly chosen partner. Two runs at different seeds therefore build different
    graphs but previously read the same cache file, so the fallback edges came from
    whichever seed happened to populate the cache first.
``sparse_mode`` / ``sparse_k_neighbors`` / ``sparse_subgroup_threshold``
    Switching between sparse k-NN and full ``combinations`` produces a completely
    different edge set.
``assign_subclusters``, ``age_split_threshold``
    Change the subgrouping the edges are built within.
``build_val_test_edges``
    Whether val/test have edges at all.

``EDGE_BUILD_VERSION`` is a manual epoch marker: bump it whenever the *meaning* of
the edge-building output changes, so stale caches are not silently reused. It was
bumped to 2 when adjacency ordering became canonical and the edge-export
orientation bug was fixed -- caches written before that are both differently
ordered and missing roughly half their edges.
"""

import hashlib

#: Bump when edge construction or serialization changes semantics.
EDGE_BUILD_VERSION = 2

#: Dataloader hyperparameters that change the graph and therefore belong in the key.
GRAPH_SHAPING_HYPERPARAMETERS = (
    "sparse_mode",
    "sparse_k_neighbors",
    "sparse_subgroup_threshold",
    "assign_subclusters",
    "age_split_threshold",
)


def node_set_hash(nodes, length=12):
    """Content hash over the node ids in a split.

    Sorted first, so the digest depends only on set membership -- not on node order
    and not on PYTHONHASHSEED.
    """
    node_ids = sorted(str(getattr(node, "node_id", node)) for node in nodes)
    return hashlib.md5("|".join(node_ids).encode()).hexdigest()[:length]


def _format_flag(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def graph_shaping_fingerprint(args=None, hyperparameters=None):
    """Short digest of every setting that changes the built graph."""
    parts = []
    source = dict(hyperparameters or {})
    for name in GRAPH_SHAPING_HYPERPARAMETERS:
        value = source.get(name)
        if value is None and args is not None:
            value = getattr(args, name, None)
        parts.append(f"{name}={_format_flag(value)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:6]


def graph_cache_key(
    *,
    dataset_name,
    split_name,
    graph_type,
    balancing_suffix,
    nodes,
    quality_threshold,
    symmetry_threshold,
    embedding_threshold,
    seed,
    build_val_test_edges=True,
    hyperparameters=None,
    args=None,
    holdout_id=None,
):
    """Build the cache basename for one split.

    Returns a string without an extension; callers append ``_graph.pkl`` or
    ``_edges.csv.gz``.
    """
    has_edges = split_name == "train" or bool(build_val_test_edges)
    edge_mode = "full_edges" if has_edges else "node_only"
    shaping = graph_shaping_fingerprint(args=args, hyperparameters=hyperparameters)

    # The seed only affects the graph when edges are actually built (via the
    # isolated-node partner fallback), so a node-only graph stays seed-independent
    # and can be shared across seeds.
    seed_part = f"seed{int(seed)}" if has_edges else "seedNA"

    return (
        f"{dataset_name}"
        f"_{split_name}"
        f"_{graph_type}"
        f"_{balancing_suffix}"
        f"_nodes_{len(nodes)}"
        f"_q{quality_threshold:.3f}"
        f"_s{symmetry_threshold:.3f}"
        f"_e{embedding_threshold:.3f}"
        f"_hash{node_set_hash(nodes)}"
        f"_mode{edge_mode}"
        f"_{seed_part}"
        f"_shape{shaping}"
        f"_ho{sanitize_key_component(holdout_id or 'none')}"
        f"_v{EDGE_BUILD_VERSION}"
    )


def sanitize_key_component(value):
    """Make a value safe for a filename.

    Needed because source-group ids come from dataset subfolder names, and one of
    them -- ``taming_transformer:VQGAN`` -- contains a colon.
    """
    text = str(value)
    for character in (":", "/", "\\", " ", ",", "*", "?"):
        text = text.replace(character, "_")
    return text


def cache_filenames(cache_dir, cache_base):
    """The pickle and edge-CSV paths for a cache base name."""
    import os

    return {
        "pickle": os.path.join(cache_dir, f"{cache_base}_graph.pkl"),
        "edges_csv": os.path.join(cache_dir, f"{cache_base}_edges.csv.gz"),
    }
