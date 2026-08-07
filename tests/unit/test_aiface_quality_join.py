"""The quality-sidecar join in AIFaceDataset.

Every continuous attribute graph-distance uncertainty reads -- blur, brightness,
contrast, compression, symmetry_*, emotion_*, face_embedding -- comes from the
`*_quality.csv` sidecars, not the base manifest. The sidecars record `image_path`
against whatever root the extraction script ran under, which for the real dataset is
no longer where the data lives. Base attributes get re-keyed to the current root, so
if the quality paths are used verbatim the two maps share no keys and *every* quality
attribute is silently dropped.

These tests build a tiny dataset whose sidecar deliberately references a dead root,
so the failure is reproducible without the 19 GB of real sidecars.
"""

import ast
import csv
import os

import numpy as np
import pytest

from datasets.AIFaceDataset import AIFaceDataset

DEAD_ROOT = "/home/somebody/old/location/ai-face"

BASE_COLUMNS = [
    "Image Path", "Uncertainty Score Gender", "Uncertainty Score Age",
    "Uncertainty Score Race", "Ground Truth Gender", "Ground Truth Age",
    "Ground Truth Race", "Intersection", "Target",
]
QUALITY_COLUMNS = [
    "image_id", "face_embedding", "quality_metrics", "symmetry", "emotion_scores",
    "error", "image_path", "_debug",
]

#: Relative paths mirroring the real layout: a flat source, an identity-folder source,
#: and a video source with real/ and fake/ subdirectories.
SAMPLE_PATHS = [
    "/FFHQ/00000.png",
    "/FFHQ/00001.png",
    "/casia-webface/000123/0.jpg",
    "/casia-webface/000456/0.jpg",   # same basename as the line above, different image
    "/dfdc/real/clip_0.png",
    "/dfdc/fake/clip_0.png",         # same basename as the line above, different label
]


def embedding_repr(seed, dim=8):
    """A numpy-repr embedding string, as the real sidecars store it.

    Whitespace-separated inside brackets -- not JSON, and not comma-separated.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.random(dim)
    return "[" + " ".join(f"{value:.6f}" for value in values) + "]"


@pytest.fixture
def dataset_root(tmp_path):
    """A tiny dataset whose quality sidecar points at a *dead* root."""
    root = tmp_path / "ai-face"
    root.mkdir()

    with open(root / "train.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BASE_COLUMNS)
        for index, path in enumerate(SAMPLE_PATHS):
            writer.writerow([
                path, 0.2, 0.2, 0.2,
                index % 2, index % 3, index % 4, 1,
                0 if "/real/" in path or path.startswith("/FFHQ") else 1,
            ])

    with open(root / "train_quality.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(QUALITY_COLUMNS)
        for index, path in enumerate(SAMPLE_PATHS):
            writer.writerow([
                os.path.basename(path),
                embedding_repr(index),
                repr({"blur_score": 100.0 + index * 10, "brightness": 120.0,
                      "contrast": 40.0, "compression_score": 0.5}),
                repr({"overall_symmetry": 0.8, "eye": 0.7}),
                repr({"happy": 0.9, "neutral": 0.1}),
                "",
                # The dead root: this is the whole point of the fixture.
                f"{DEAD_ROOT}{path}",
                "",
            ])
    return root


@pytest.fixture
def dataset(dataset_root):
    instance = AIFaceDataset.__new__(AIFaceDataset)
    instance.data_root = str(dataset_root)
    return instance


# --------------------------------------------------------------------------- #
# The normalizer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "given",
    [
        f"{DEAD_ROOT}/dfdc/real/clip_0.png",   # dead absolute
        "/dfdc/real/clip_0.png",               # manifest-relative
        "dfdc/real/clip_0.png",                # bare relative
        f"{DEAD_ROOT}\\dfdc\\real\\clip_0.png",  # windows separators
    ],
)
def test_paths_are_rewritten_onto_the_current_root(dataset, dataset_root, given):
    expected = os.path.join(str(dataset_root), "dfdc/real/clip_0.png")
    assert dataset._to_current_root(given, {"dfdc"}) == expected


def test_a_path_already_under_the_current_root_is_preserved(dataset, dataset_root):
    already = os.path.join(str(dataset_root), "FFHQ/00000.png")
    assert dataset._to_current_root(already, {"FFHQ"}) == already


def test_an_unrecognized_source_yields_none(dataset):
    """Better to report a row as unmatched than to guess where it belongs."""
    assert dataset._to_current_root("/elsewhere/Unknown/x.png", {"FFHQ"}) is None


def test_empty_input_yields_none(dataset):
    assert dataset._to_current_root("", {"FFHQ"}) is None
    assert dataset._to_current_root(None, {"FFHQ"}) is None


def test_normalization_is_not_basename_based(dataset, dataset_root):
    """Two different images share a basename and must not collapse together.

    Basenames are nowhere near unique in the real dataset -- 216,267 collisions in
    train alone, with `50.jpg` appearing 3,991 times across identity folders. A
    basename join would attach one identity's attributes to thousands of unrelated
    images, which is worse than no join at all because it would look like it worked.
    """
    sources = {"casia-webface"}
    first = dataset._to_current_root(f"{DEAD_ROOT}/casia-webface/000123/0.jpg", sources)
    second = dataset._to_current_root(f"{DEAD_ROOT}/casia-webface/000456/0.jpg", sources)
    assert first != second
    assert first.endswith("000123/0.jpg") and second.endswith("000456/0.jpg")


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #

def test_quality_attributes_reach_the_attribute_map(dataset, dataset_root, capsys):
    """The regression test: a dead-root sidecar must still merge."""
    attributes = dataset._load_additional_attributes("train")

    assert len(attributes) == len(SAMPLE_PATHS), (
        f"expected one entry per image, got {len(attributes)} -- extra entries mean the "
        f"quality rows landed under keys of their own instead of merging"
    )

    for path in SAMPLE_PATHS:
        key = os.path.join(str(dataset_root), path.lstrip("/"))
        assert key in attributes, f"{key} missing from the attribute map"
        entry = attributes[key]
        # Base columns.
        assert "Target" in entry and "Ground Truth Gender" in entry
        # Quality columns -- these are what were being dropped.
        assert "blur" in entry, "blur did not survive the join"
        assert "brightness" in entry and "contrast" in entry and "compression" in entry
        assert "symmetry_overall" in entry
        assert any(name.startswith("emotion_") for name in entry)
        assert "face_embedding" in entry
        assert isinstance(entry["face_embedding"], np.ndarray)
        assert entry["face_embedding"].size == 8
        assert not np.allclose(entry["face_embedding"], 0)

    assert "100.0% joined" in capsys.readouterr().out


def test_join_reports_its_rate(dataset, capsys):
    dataset._load_additional_attributes("train")
    output = capsys.readouterr().out
    assert "Quality attribute join for train" in output
    assert "merged" in output and "unmatched" in output


def test_a_collapsed_join_raises_instead_of_degrading_silently(dataset_root):
    """A 0% join must fail loudly.

    This is not hypothetical: it is exactly what the real dataset did for as long as
    the sidecar paths were used verbatim, and the only visible symptom was that
    graph-distance uncertainty returned its missing-value sentinels for every node.
    """
    # Rewrite the sidecar so no path contains a recognizable source folder.
    rows = []
    with open(dataset_root / "train_quality.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["image_path"] = "/nowhere/unrecognized/" + row["image_id"]
            rows.append(row)
    with open(dataset_root / "train_quality.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    instance = AIFaceDataset.__new__(AIFaceDataset)
    instance.data_root = str(dataset_root)
    with pytest.raises(ValueError, match="matched only"):
        instance._load_additional_attributes("train")


def test_the_error_message_names_the_consequence(dataset_root):
    """The message should say what breaks, not just that a number was low."""
    rows = []
    with open(dataset_root / "train_quality.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            row["image_path"] = "/nowhere/unrecognized/" + row["image_id"]
            rows.append(row)
    with open(dataset_root / "train_quality.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    instance = AIFaceDataset.__new__(AIFaceDataset)
    instance.data_root = str(dataset_root)
    try:
        instance._load_additional_attributes("train")
    except ValueError as error:
        message = str(error)
        assert "face_embedding" in message
        assert "graph-distance" in message
        assert "example unmatched paths" in message
    else:
        pytest.fail("expected a ValueError")


def test_missing_sidecar_is_fine(dataset, dataset_root):
    """No sidecar means base attributes only -- and skips the multi-GB read."""
    os.remove(dataset_root / "train_quality.csv")
    attributes = dataset._load_additional_attributes("train")
    assert len(attributes) == len(SAMPLE_PATHS)
    entry = next(iter(attributes.values()))
    assert "Target" in entry
    assert "blur" not in entry


def test_graph_distance_has_signal_on_joined_attributes(dataset, dataset_root):
    """The point of the whole fix.

    With the join working, graph-distance sees real continuous attributes and a real
    embedding, so its distances vary between nodes instead of collapsing onto the
    isolated-node sentinel.
    """
    from edges.Edge import Edge
    from models.uncertainty import GraphDistanceUncertainty
    from nodes.atrnode import AttributeNode

    attributes = dataset._load_additional_attributes("train")
    nodes = [
        AttributeNode(key, "train", None, [], int(entry.get("Target", 0)), dict(entry), 50)
        for key, entry in sorted(attributes.items())
    ]
    for index in range(len(nodes) - 1):
        edge = Edge(nodes[index], nodes[index + 1], x=None)
        nodes[index].add_edge(edge)
        nodes[index + 1].add_edge(edge)

    methods = ("attribute_distance", "embedding_distance", "hybrid_distance")
    scores = GraphDistanceUncertainty(methods=methods).fit(nodes).compute(nodes)

    for method in methods:
        values = scores[method].squeeze(1)
        assert np.isfinite(values.numpy()).all(), f"{method} produced non-finite values"
        assert values.max().item() > 0.0, f"{method} is identically zero"
        # 1.0 is the isolated/undefined sentinel; every node here has a neighbour and
        # a usable embedding, so nothing should be falling back to it.
        assert not np.allclose(values.numpy(), 1.0), (
            f"{method} collapsed onto the missing-value sentinel"
        )
