"""Build node caches for tests and short runs.

`AIFaceDataset.load()` builds every node for all three splits and reads the
`*_quality.csv` sidecars before `--cached-nodes` truncates anything, so a nominally
tiny run takes over ten minutes. A warm node cache skips that entirely: with
`--use-cached` and an existing `--cache-file`, `load_and_prepare_data_splits` returns
before ever constructing the dataset.

Two ways in, both routed through the *production* writer
(`test_helpers.data_graph_utils.save_cached_nodes`) so the on-disk format cannot drift
from what `load_cached_nodes` accepts:

`write_synthetic_node_cache`
    Nodes built from `tests.helpers.images.make_image_nodes` -- real `ImageFileData`
    over real (tiny) PNGs, with the full attribute surface a dataset node carries.
    Sub-second, no dataset needed.

`write_tiny_ai_face_root`
    A miniature dataset directory (real 9-column manifests + tiny PNGs, deliberately
    *no* sidecars) that the real `AIFaceDataset` can load. Slower, but it is the only
    thing that exercises the loader itself.

The pickle is generated rather than committed: it embeds absolute, root-specific
`node_id` paths, `dill.load` executes code from the file, and it breaks on any rename
to `AttributeNode` or `ImageFileData`. Generation is fast enough that committing buys
nothing.
"""

import csv
import os

import numpy as np

BASE_COLUMNS = [
    "Image Path", "Uncertainty Score Gender", "Uncertainty Score Age",
    "Uncertainty Score Race", "Ground Truth Gender", "Ground Truth Age",
    "Ground Truth Race", "Intersection", "Target",
]

#: Source folders to spread synthetic images across, mirroring the real layout: two
#: exclusively-real sources and a video source whose second component carries the
#: class. Kept consistent with the corrected manifests so `parse_source_group` and the
#: holdout logic see realistic group ids.
SYNTHETIC_LAYOUT = [
    ("FFHQ", None, 0),
    ("wiki", "00", 0),
    ("dfdc", "real", 0),
    ("dfdc", "fake", 1),
    ("ProGAN", "zip000", 1),
    ("StableDiffusion1.5", None, 1),
]


def _relative_path(index):
    """A dataset-relative image path, cycling through the synthetic layout."""
    source, second, label = SYNTHETIC_LAYOUT[index % len(SYNTHETIC_LAYOUT)]
    name = f"{index:06d}.png"
    if second:
        return f"/{source}/{second}/{name}", label
    return f"/{source}/{name}", label


def write_synthetic_node_cache(
    cache_path, image_dir, n_train=600, n_val=200, n_test=200,
    embedding_dim=512, size=8, target_num_nodes=None,
):
    """Write a node cache from synthetic nodes. Returns the cache path.

    ``embedding_dim`` defaults to 512 to match what the real sidecars produce, so the
    cache is representative in size (~4.7 KB per node) as well as in shape.
    """
    from test_helpers.data_graph_utils import save_cached_nodes

    splits = {}
    offset = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        splits[split] = _build_nodes(
            image_dir=os.path.join(str(image_dir), split),
            count=count,
            index_offset=offset,
            split=split,
            embedding_dim=embedding_dim,
            size=size,
        )
        # Offset per split so nodes at the same position in different splits do not
        # end up with identical attributes (make_attributes is a pure function of the
        # index).
        offset += count

    if target_num_nodes is None:
        target_num_nodes = min(len(nodes) for nodes in splits.values())

    # save_cached_nodes calls os.makedirs(os.path.dirname(cache_file)), which raises
    # FileNotFoundError on a bare filename -- so hand it an absolute path.
    cache_path = os.path.abspath(str(cache_path))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_cached_nodes(
        splits["train"], splits["val"], splits["test"],
        cache_path, target_num_nodes=target_num_nodes,
    )
    return cache_path


def _build_nodes(image_dir, count, index_offset, split, embedding_dim, size):
    """Nodes with real image files and a realistic attribute surface."""
    from nodes.atrnode import AttributeNode

    from .factories import make_attributes
    from .images import write_tiny_png

    os.makedirs(image_dir, exist_ok=True)
    from data.ImageFileData import ImageFileData

    nodes = []
    for position in range(count):
        index = index_offset + position
        relative, label = _relative_path(index)
        # Keep the source structure in the on-disk layout too, so anything that parses
        # the node id for a source group sees something realistic.
        path = os.path.join(image_dir, relative.lstrip("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_tiny_png(path, size=size, seed=index)

        attributes = make_attributes(index, embedding_dim=embedding_dim)
        # Columns a real node carries from the base manifest but which
        # make_attributes does not emit.
        attributes["Target"] = label
        attributes["subset"] = split
        attributes["Intersection"] = 1 + (index % 6)
        for name in ("Gender", "Age", "Race"):
            attributes[f"Uncertainty Score {name}"] = round(0.2 + (index % 5) / 50.0, 4)

        nodes.append(
            AttributeNode(path, split, ImageFileData(path), [], label, attributes, 50)
        )
    return nodes


def write_tiny_ai_face_root(root, n_train=12, n_val=6, n_test=6, size=8,
                            with_quality=False):
    """Write a miniature AI-Face dataset directory. Returns the root path.

    Omits the `*_quality.csv` sidecars by default, which is what makes it cheap: the
    `os.path.exists` guards in `_load_additional_attributes` skip the entire
    multi-gigabyte quality path when they are absent.

    Pass ``with_quality=True`` to also emit sidecars. Their `image_path` is written
    against a *deliberately different* root, so the loader's path normalization is
    exercised rather than bypassed -- that is the condition the real dataset is in.
    """
    from .images import write_tiny_png

    root = str(root)
    os.makedirs(root, exist_ok=True)

    offset = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        rows = []
        for position in range(count):
            index = offset + position
            relative, label = _relative_path(index)
            path = os.path.join(root, relative.lstrip("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_tiny_png(path, size=size, seed=index)
            rows.append((relative, label, index))
        offset += count

        with open(os.path.join(root, f"{split}.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(BASE_COLUMNS)
            for relative, label, index in rows:
                writer.writerow([
                    relative,
                    round(0.2 + (index % 5) / 50.0, 4),
                    round(0.2 + (index % 4) / 50.0, 4),
                    round(0.2 + (index % 3) / 50.0, 4),
                    index % 2, index % 3, index % 4, 1 + (index % 6),
                    label,
                ])

        if with_quality:
            _write_quality_sidecar(root, split, rows)

    return root


def _write_quality_sidecar(root, split, rows, dead_root="/elsewhere/old/ai-face"):
    """Sidecar keyed against a different root, as the real dataset's are."""
    columns = [
        "image_id", "face_embedding", "quality_metrics", "symmetry",
        "emotion_scores", "error", "image_path", "_debug",
    ]
    with open(os.path.join(root, f"{split}_quality.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for relative, _label, index in rows:
            rng = np.random.Generator(np.random.PCG64(index))
            embedding = " ".join(f"{value:.6f}" for value in rng.random(16))
            writer.writerow([
                os.path.basename(relative),
                f"[{embedding}]",
                repr({
                    "blur_score": 40.0 + (index % 37) * 9.5,
                    "brightness": 90.0 + (index % 11) * 4.0,
                    "contrast": 25.0 + (index % 7) * 3.0,
                    "compression_score": round((index % 9) / 10.0, 3),
                }),
                repr({
                    "overall_symmetry": round(0.5 + (index % 5) / 10.0, 3),
                    "eye": round(0.4 + (index % 6) / 10.0, 3),
                    "mouth": round(0.3 + (index % 7) / 10.0, 3),
                    "nose": round(0.6 + (index % 4) / 10.0, 3),
                }),
                repr({
                    "happy": round((index % 10) / 10.0, 2),
                    "neutral": round(1.0 - (index % 10) / 10.0, 2),
                }),
                "",
                f"{dead_root}{relative}",
                "",
            ])
