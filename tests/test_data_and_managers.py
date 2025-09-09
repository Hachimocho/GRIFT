import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")
from data.Data import Data
from data.ImageData import ImageData
from data.ImageFileData import ImageFileData
from graphs.HyperGraph import HyperGraph
from managers.NoGraphManager import NoGraphManager
from nodes.Node import Node


def test_data_load_and_set():
    d = Data(indata={"a": 1})
    assert d.load_data() == {"a": 1}
    d.set_data([1, 2])
    assert d.load_data() == [1, 2]


def test_imagefiledata_roundtrip(tmp_path):
    # Create a tiny image file
    p = tmp_path / "img.png"
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv2.imwrite(str(p), img)

    fdata = ImageFileData(str(p))
    loaded = fdata.load_data()
    assert loaded is not None and loaded.shape == (2, 2, 3)

    # ImageData loads the image into memory at construction
    idata = ImageData(str(p))
    arr = idata.load_data()
    assert arr is not None and arr.shape == (2, 2, 3)


def test_no_graph_manager_basic():
    n1 = Node(node_id="1", split="train", data=None, edges=[], label=0)
    hg = HyperGraph([n1])
    gm = NoGraphManager(hg)
    assert gm.get_graph() is hg
    gm.update_graph()  # should not raise
    hg2 = HyperGraph([])
    gm.set_graph(hg2)
    assert gm.get_graph() is hg2

