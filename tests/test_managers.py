import pytest

from managers.GraphManager import GraphManager
from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager

from .conftest import build_dummy_graph


def test_graph_manager_interface():
    graph, *_ = build_dummy_graph(5)

    class Dummy(GraphManager):
        def update_graph(self):
            pass

    gm = Dummy(graph)
    assert gm.get_graph() is graph
    gm.set_graph(graph)


def test_no_graph_manager_does_nothing():
    graph, *_ = build_dummy_graph(5)
    gm = NoGraphManager(graph)
    gm.update_graph()  # should not raise
    assert gm.get_graph() is graph


def test_performance_graph_manager_basic_operations():
    graph, nodes, _ = build_dummy_graph(6)
    pgm = PerformanceGraphManager(graph, update_interval=1)
    # No predictor; update_graph early exits
    pgm.update_graph()
    # Track some fake performance and ensure helper methods return lists
    for n in nodes:
        pgm.track_performance(n, 0.1)
    strong = pgm.identify_strong_nodes()
    assert isinstance(strong, list)

