"""Data wrappers and graph managers."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from data.Data import Data
from data.ImageData import ImageData
from data.ImageFileData import ImageFileData
from graphs.HyperGraph import HyperGraph
from managers.GraphManager import GraphManager
from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager
from nodes.Node import Node


def test_data_load_and_set():
    data = Data(indata={"a": 1})
    assert data.load_data() == {"a": 1}
    data.set_data([1, 2])
    assert data.load_data() == [1, 2]


def test_imagefiledata_lazy_roundtrip(tiny_png):
    """ImageFileData defers the read to load_data(), keeping node memory small."""
    file_data = ImageFileData(tiny_png)
    loaded = file_data.load_data()
    assert loaded is not None and loaded.shape == (8, 8, 3)


def test_imagedata_eager_roundtrip(tiny_png):
    """ImageData reads at construction."""
    array = ImageData(tiny_png).load_data()
    assert array is not None and array.shape == (8, 8, 3)


def test_imagefiledata_rejects_missing_file(tmp_path):
    with pytest.raises(AssertionError):
        ImageFileData(str(tmp_path / "does_not_exist.png"))


def test_imagefiledata_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "img.bmp"
    assert cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(AssertionError):
        ImageFileData(str(path))


def test_imagefiledata_extension_check_is_case_sensitive(tmp_path):
    """Pins that an uppercase extension is rejected.

    The check is `indata.split('.')[-1] in ["jpg", "jpeg", "png"]`, so `IMG.PNG`
    fails even though cv2 would read it happily.
    """
    path = tmp_path / "IMG.PNG"
    assert cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(AssertionError):
        ImageFileData(str(path))


def test_graph_manager_interface(ring_graph):
    graph, _, _ = ring_graph

    class Dummy(GraphManager):
        def update_graph(self):
            pass

    manager = Dummy(graph)
    assert manager.get_graph() is graph
    manager.set_graph(graph)
    assert manager.get_graph() is graph


def test_no_graph_manager_is_inert():
    node = Node(node_id="1", split="train", data=None, edges=[], label=0)
    graph = HyperGraph([node])
    manager = NoGraphManager(graph)
    assert manager.get_graph() is graph
    manager.update_graph()  # must not raise

    replacement = HyperGraph([])
    manager.set_graph(replacement)
    assert manager.get_graph() is replacement


def test_performance_graph_manager_basic_operations(ring_graph):
    graph, nodes, _ = ring_graph
    manager = PerformanceGraphManager(graph, update_interval=1)

    manager.update_graph()  # no predictor configured -> early exit, must not raise
    for node in nodes:
        manager.track_performance(node, 0.1)

    strong = manager.identify_strong_nodes()
    assert isinstance(strong, list)
