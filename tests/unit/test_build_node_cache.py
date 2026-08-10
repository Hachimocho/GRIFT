"""`development_tools/build_node_cache.py` against the tiny dataset fixture.

The builder exists so real-data runs skip a 10-minute dataset load, and its one real
requirement is that what it writes is what `load_cached_nodes` accepts. It enforces
that itself by reading the cache back through the production reader before declaring
success, so these tests mostly check the guards around that: the refusals, the
determinism of the subsample, and that the tiny fixture root is realistic enough for
the real `AIFaceDataset` to load.

Run against `tiny_ai_face_root` rather than the real dataset, so this is a fast-tier
test with no `--run-data` gate.
"""

import json
import os

import pytest

from development_tools.build_node_cache import main, subsample
from test_helpers.data_graph_utils import load_cached_nodes
from tests.helpers.factories import make_attr_nodes


def build(root, out, *extra):
    """Run the builder's main() with the tiny root, returning its exit code."""
    return main([
        "--data-root", str(root), "--out", str(out),
        "--allow-missing-quality", *extra,
    ])


# --------------------------------------------------------------------------- #
# The fixture root itself
# --------------------------------------------------------------------------- #

def test_the_real_dataset_loader_accepts_the_tiny_root(tiny_ai_face_root):
    """If AIFaceDataset cannot load the fixture, nothing built on it means anything."""
    from data.ImageFileData import ImageFileData
    from datasets.AIFaceDataset import AIFaceDataset
    from nodes.atrnode import AttributeNode

    dataset = AIFaceDataset(
        str(tiny_ai_face_root), ImageFileData, {}, AttributeNode, {"threshold": 2}
    )
    nodes = dataset.load()
    assert len(nodes) == 24
    assert {node.split for node in nodes} == {"train", "val", "test"}
    assert {node.label for node in nodes} == {0, 1}
    # Base-manifest attributes only -- no sidecars were written.
    attributes = nodes[0].attributes
    assert "Ground Truth Gender" in attributes
    assert "blur" not in attributes


def test_a_root_with_sidecars_joins(tmp_path):
    """With sidecars present, the quality attributes must reach the nodes.

    The fixture writes sidecar `image_path` values against a *dead* root, matching the
    real dataset, so this exercises the path normalization rather than bypassing it.
    """
    from data.ImageFileData import ImageFileData
    from datasets.AIFaceDataset import AIFaceDataset
    from nodes.atrnode import AttributeNode
    from tests.helpers.node_cache import write_tiny_ai_face_root

    root = write_tiny_ai_face_root(tmp_path / "with_quality", with_quality=True)
    dataset = AIFaceDataset(
        root, ImageFileData, {}, AttributeNode, {"threshold": 2}
    )
    nodes = dataset.load()
    assert nodes, "loader returned nothing"
    for node in nodes:
        assert "blur" in node.attributes
        assert "face_embedding" in node.attributes
        assert node.attributes["face_embedding"].size == 16


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #

def test_it_writes_a_cache_the_reader_accepts(tiny_ai_face_root, tmp_path):
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    assert build(tiny_ai_face_root, out, "--cached-nodes", "4") == 0

    for split, expected in (("train", 12), ("val", 6), ("test", 6)):
        assert len(load_cached_nodes(str(out), split)) == expected
        assert len(load_cached_nodes(str(out), split, balanced=True)) == 4


def test_it_writes_a_manifest(tiny_ai_face_root, tmp_path):
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    build(tiny_ai_face_root, out, "--cached-nodes", "4")

    with open(str(out) + ".meta.json") as handle:
        manifest = json.load(handle)
    assert manifest["counts"] == {"train": 12, "val": 6, "test": 6}
    assert manifest["cached_nodes"] == 4
    assert manifest["data_root"] == str(tiny_ai_face_root)
    assert manifest["size_mb"] > 0
    # No sidecars in the fixture, so this must report the truth rather than a default.
    assert manifest["quality_coverage"] == {"train": 0.0, "val": 0.0, "test": 0.0}


def test_it_clamps_a_balanced_target_larger_than_the_smallest_split(
    tiny_ai_face_root, tmp_path, capsys
):
    """`--cached-nodes 500` on a 6-node split must not produce an unloadable cache."""
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    assert build(tiny_ai_face_root, out, "--cached-nodes", "500") == 0
    assert "Clamping balanced target" in capsys.readouterr().out
    assert len(load_cached_nodes(str(out), "val", balanced=True)) == 6


def test_it_caps_the_full_lists(tiny_ai_face_root, tmp_path):
    """--max-nodes-per-split is what keeps a small cache small.

    Without it the *full* lists hold every node -- ~7.5 GB on the real dataset -- since
    --cached-nodes only sizes the balanced view.
    """
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    assert build(tiny_ai_face_root, out, "--cached-nodes", "3",
                 "--max-nodes-per-split", "5") == 0
    for split in ("train", "val", "test"):
        assert len(load_cached_nodes(str(out), split)) == min(
            5, {"train": 12, "val": 6, "test": 6}[split]
        )


def test_it_refuses_to_overwrite_without_force(tiny_ai_face_root, tmp_path):
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    assert build(tiny_ai_face_root, out, "--cached-nodes", "4") == 0
    before = os.path.getmtime(out)
    assert build(tiny_ai_face_root, out, "--cached-nodes", "4") == 1
    assert os.path.getmtime(out) == before
    assert build(tiny_ai_face_root, out, "--cached-nodes", "4", "--force") == 0


def test_it_refuses_a_cache_with_no_quality_attributes(tiny_ai_face_root, tmp_path):
    """Without --allow-missing-quality, a sidecar-less build must fail loudly.

    A cache built from a broken join disables graph-distance uncertainty for every run
    that uses it, and nothing downstream would say so -- the method just returns its
    missing-value sentinels.
    """
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    with pytest.raises(SystemExit, match="quality-sidecar join"):
        main(["--data-root", str(tiny_ai_face_root), "--out", str(out),
              "--cached-nodes", "4"])
    assert not os.path.exists(out)


def test_a_root_with_sidecars_needs_no_override(tmp_path):
    from tests.helpers.node_cache import write_tiny_ai_face_root

    root = write_tiny_ai_face_root(tmp_path / "with_quality", with_quality=True)
    out = tmp_path / "node_cache" / "cached_nodes.pkl"
    assert main(["--data-root", root, "--out", str(out), "--cached-nodes", "4"]) == 0

    with open(str(out) + ".meta.json") as handle:
        manifest = json.load(handle)
    assert manifest["quality_coverage"] == {"train": 1.0, "val": 1.0, "test": 1.0}

    nodes = load_cached_nodes(str(out), "train")
    assert all("face_embedding" in node.attributes for node in nodes)


# --------------------------------------------------------------------------- #
# The subsample
# --------------------------------------------------------------------------- #

def test_subsample_is_a_noop_below_the_limit():
    nodes = make_attr_nodes(5)
    assert subsample(nodes, 10, "train") == nodes
    assert subsample(nodes, None, "train") == nodes


def test_subsample_is_deterministic_and_order_independent():
    """Same nodes, same result -- regardless of the order they arrive in.

    The dataset's emission order is not something the cache should depend on, so the
    subsample sorts first. Without that, a reordered manifest would silently change
    which nodes a "reproducible" cache contains.
    """
    from test_helpers.determinism import configure_determinism

    configure_determinism(seed=7, mode="fast")
    nodes = make_attr_nodes(20)
    first = [node.node_id for node in subsample(nodes, 8, "train")]
    second = [node.node_id for node in subsample(list(reversed(nodes)), 8, "train")]
    assert first == second
    assert first == sorted(first), "output should be in node_id order"


def test_subsample_streams_differ_per_split():
    """Splits must not select the same positions, or the cache is correlated."""
    from test_helpers.determinism import configure_determinism

    configure_determinism(seed=7, mode="fast")
    nodes = make_attr_nodes(40)
    train = [node.node_id for node in subsample(nodes, 10, "train")]
    val = [node.node_id for node in subsample(nodes, 10, "val")]
    assert train != val


def test_subsample_follows_the_master_seed():
    from test_helpers.determinism import configure_determinism

    nodes = make_attr_nodes(30)
    configure_determinism(seed=1, mode="fast")
    first = [node.node_id for node in subsample(nodes, 10, "train")]
    configure_determinism(seed=2, mode="fast")
    second = [node.node_id for node in subsample(nodes, 10, "train")]
    assert first != second
