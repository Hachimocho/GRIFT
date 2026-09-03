"""Graph cache key construction.

The key must change whenever anything that changes the built graph changes.
Omissions here are silent: two materially different graphs share one cache entry,
and the second run trains on the first run's graph without any indication.
"""

import pytest

from test_helpers.cache_keys import (
    EDGE_BUILD_VERSION, cache_filenames, graph_cache_key, graph_shaping_fingerprint,
    node_set_hash, sanitize_key_component,
)
from tests.helpers.factories import make_attr_nodes

BASE = dict(
    dataset_name="ai-face",
    split_name="train",
    graph_type="clustered",
    balancing_suffix="full",
    quality_threshold=0.5,
    symmetry_threshold=0.3,
    embedding_threshold=0.7,
    seed=42,
    build_val_test_edges=True,
    hyperparameters={
        "sparse_mode": True, "sparse_k_neighbors": 20,
        "sparse_subgroup_threshold": 5000, "assign_subclusters": False,
        "age_split_threshold": 1000,
    },
)


def key(nodes=None, **overrides):
    settings = dict(BASE)
    settings.update(overrides)
    settings["nodes"] = nodes if nodes is not None else make_attr_nodes(8)
    return graph_cache_key(**settings)


def test_key_is_stable_for_identical_inputs():
    nodes = make_attr_nodes(8)
    assert key(nodes) == key(nodes)


def test_node_hash_ignores_order():
    """Depends on membership only, not on node ordering or PYTHONHASHSEED."""
    nodes = make_attr_nodes(8)
    assert node_set_hash(nodes) == node_set_hash(list(reversed(nodes)))


def test_node_hash_changes_with_membership():
    assert node_set_hash(make_attr_nodes(8)) != node_set_hash(make_attr_nodes(9))


def test_key_includes_the_version():
    assert key().endswith(f"_v{EDGE_BUILD_VERSION}")


@pytest.mark.parametrize(
    "override",
    [
        {"dataset_name": "other"},
        {"split_name": "val"},
        {"graph_type": "nonclustered"},
        {"balancing_suffix": "balanced"},
        {"quality_threshold": 0.6},
        {"symmetry_threshold": 0.4},
        {"embedding_threshold": 0.8},
        {"holdout_id": "H1_diffusion_unseen"},
    ],
)
def test_key_is_sensitive_to_each_setting(override):
    nodes = make_attr_nodes(8)
    assert key(nodes) != key(nodes, **override), f"key ignores {list(override)[0]}"


def test_key_is_sensitive_to_the_seed():
    """Edge construction is seed-dependent via the isolated-node partner fallback.

    Without the seed in the key, two runs at different seeds built different graphs
    but read the same cache file, so the fallback edges came from whichever seed
    populated the cache first.
    """
    nodes = make_attr_nodes(8)
    assert key(nodes, seed=1) != key(nodes, seed=2)


@pytest.mark.parametrize(
    "hyperparameter,value",
    [
        ("sparse_mode", False),
        ("sparse_k_neighbors", 40),
        ("sparse_subgroup_threshold", 1000),
        ("assign_subclusters", True),
        ("age_split_threshold", 500),
    ],
)
def test_key_is_sensitive_to_graph_shaping_hyperparameters(hyperparameter, value):
    """These change the edge set outright but were absent from the old key.

    Switching sparse_mode alone replaces sparse k-NN edge generation with full
    `combinations`, i.e. a completely different graph under an identical key.
    """
    nodes = make_attr_nodes(8)
    changed = dict(BASE["hyperparameters"])
    changed[hyperparameter] = value
    assert key(nodes) != key(nodes, hyperparameters=changed), (
        f"key ignores {hyperparameter}"
    )


def test_node_only_graphs_share_a_key_across_seeds():
    """A graph with no edges has no seed dependence, so it can be shared."""
    nodes = make_attr_nodes(8)
    first = key(nodes, split_name="val", build_val_test_edges=False, seed=1)
    second = key(nodes, split_name="val", build_val_test_edges=False, seed=2)
    assert first == second
    assert "seedNA" in first
    assert "modenode_only" in first


def test_edge_policy_is_in_the_key():
    nodes = make_attr_nodes(8)
    with_edges = key(nodes, split_name="val", build_val_test_edges=True)
    without = key(nodes, split_name="val", build_val_test_edges=False)
    assert "modefull_edges" in with_edges
    assert "modenode_only" in without


def test_train_split_always_has_edges():
    """--no-build-val-test-edges must not disable train edges."""
    nodes = make_attr_nodes(8)
    assert "modefull_edges" in key(nodes, split_name="train", build_val_test_edges=False)


def test_shaping_fingerprint_is_stable_and_short():
    fingerprint = graph_shaping_fingerprint(hyperparameters=BASE["hyperparameters"])
    assert fingerprint == graph_shaping_fingerprint(hyperparameters=BASE["hyperparameters"])
    assert len(fingerprint) == 6


def test_shaping_fingerprint_tolerates_missing_values():
    assert graph_shaping_fingerprint(hyperparameters={})


def test_sanitize_key_component_handles_the_colon_source():
    """One AI-Face source folder is literally named `taming_transformer:VQGAN`."""
    assert ":" not in sanitize_key_component("taming_transformer:VQGAN")
    assert sanitize_key_component("taming_transformer:VQGAN") == "taming_transformer_VQGAN"
    assert "/" not in sanitize_key_component("celebdf/crop_img")


def test_cache_filenames_derive_from_the_base():
    paths = cache_filenames("graph_cache", "base")
    assert paths["pickle"].endswith("base_graph.pkl")
    assert paths["edges_csv"].endswith("base_edges.csv.gz")


def test_key_is_filesystem_safe():
    generated = key(holdout_id="taming_transformer:VQGAN")
    for character in (":", "/", "\\", " ", "*", "?"):
        assert character not in generated, f"key contains {character!r}"


def test_find_existing_graph_caches_parses_the_current_format(tmp_path):
    """The discovery regex must understand keys the builder actually writes."""
    from test_helpers.data_graph_utils import find_existing_graph_caches

    nodes = make_attr_nodes(8)
    cache_base = key(nodes)
    (tmp_path / f"{cache_base}_edges.csv.gz").write_bytes(b"")

    found = find_existing_graph_caches(str(tmp_path))
    assert found, f"regex failed to parse {cache_base}_edges.csv.gz"
    entry = next(iter(found.values()))
    assert entry["split"] == "train"
    assert entry["node_count"] == len(nodes)
    assert entry["cache_version"] == EDGE_BUILD_VERSION
    assert entry["stale"] is False


def test_find_existing_graph_caches_flags_legacy_files(tmp_path):
    """Pre-version filenames must parse and be reported as stale, not ignored.

    Caches written before the edge-export fix are missing roughly half their edges,
    so silently reusing one would train on a materially sparser graph.
    """
    from test_helpers.data_graph_utils import find_existing_graph_caches

    legacy = (
        "ai-face_train_clustered_full_nodes_100_q0.500_s0.300_e0.700"
        "_hashabc12345_modefull_edges_edges.csv.gz"
    )
    (tmp_path / legacy).write_bytes(b"")

    found = find_existing_graph_caches(str(tmp_path))
    assert found, "legacy filename no longer parses"
    entry = next(iter(found.values()))
    assert entry["cache_version"] == 1
    assert entry["stale"] is True
