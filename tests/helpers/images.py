"""Tiny on-disk images for tests that need a real ``ImageFileData``.

``data/ImageFileData.py`` asserts ``os.path.isfile`` at construction and checks
the extension against ``["jpg", "jpeg", "png"]`` with a naive
``split('.')[-1]``, so the extension must be lowercase. ``load_data()`` is
``cv2.imread``, so cv2 is also the natural writer.
"""

import numpy as np


def write_tiny_png(path, size=8, seed=0):
    """Write a deterministic ``size x size`` BGR PNG. Returns the path as str."""
    import cv2

    rng = np.random.Generator(np.random.PCG64(seed))
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path = str(path)
    assert cv2.imwrite(path, image), f"cv2.imwrite failed for {path}"
    return path


def write_tiny_pngs(directory, count=8, size=8):
    """Write ``count`` distinct PNGs into ``directory``. Returns their paths."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        write_tiny_png(directory / f"img_{index:03d}.png", size=size, seed=index)
        for index in range(count)
    ]


def make_image_nodes(directory, count=8, size=8, split="train"):
    """AttributeNodes whose ``.data`` is a real ``ImageFileData``.

    Lets training and eval loops run end-to-end with no dataset -- the loader,
    ``model.transform``, batching, and the loss all execute for real.
    """
    from data.ImageFileData import ImageFileData

    from .factories import make_attributes
    from nodes.atrnode import AttributeNode

    paths = write_tiny_pngs(directory, count=count, size=size)
    nodes = []
    for index, path in enumerate(paths):
        nodes.append(
            AttributeNode(
                node_id=path,  # real nodes use the image path as their id
                split=split,
                data=ImageFileData(path),
                edges=[],
                label=int(index % 2),
                attributes=make_attributes(index),
                threshold=50,
            )
        )
    return nodes
