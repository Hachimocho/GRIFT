"""Smoke tests for the test harness itself.

If these fail, every other failure in the suite is suspect.
"""

import os

import pytest


def test_repo_root_on_syspath(repo_root):
    assert (repo_root / "models" / "uncertainty" / "types.py").is_file()


def test_reproducibility_env_is_pinned():
    """pytest_configure must have pinned the env (re-execing if necessary)."""
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")


def test_determinism_configured_strict():
    from test_helpers.determinism import get_determinism_config, is_strict
    assert is_strict()
    assert get_determinism_config().seed == 0


def test_torch_threads_pinned():
    import torch
    assert torch.get_num_threads() == 1


def test_cwd_is_tmp(tmp_path):
    assert os.getcwd() == str(tmp_path)


def test_network_is_blocked():
    import torch.hub
    with pytest.raises(RuntimeError, match="network access blocked"):
        torch.hub.load("some/repo", "some_model")


def test_factories_build_a_graph(ring_graph):
    graph, nodes, edges = ring_graph
    assert len(nodes) == 6
    assert len(edges) == 6
    assert len(graph.get_nodes()) == 6
    # Ring topology: every node has exactly two neighbors.
    for node in nodes:
        assert len(node.get_adjacent_nodes()) == 2


def test_two_cluster_graph_has_a_bridge(two_cluster_graph):
    graph, nodes, edges = two_cluster_graph
    degrees = sorted(len(node.get_adjacent_nodes()) for node in nodes)
    # Two 4-cliques (degree 3 each) plus one bridge raising two nodes to degree 4.
    assert degrees == [3, 3, 3, 3, 3, 3, 4, 4]


def test_demographics_are_numpy_ints(attr_nodes):
    """Pins the dtype that makes the graph_distance isinstance bug reproducible."""
    import numpy as np
    node = attr_nodes[0]
    value = node.attributes["Ground Truth Gender"]
    assert isinstance(value, np.integer)
    assert not isinstance(value, int), (
        "np.int64 must not be a Python int here -- that distinction is exactly what "
        "graph_distance.py's isinstance check gets wrong"
    )


def test_tiny_detector_avoids_the_detector_zoo(tiny_detector):
    """The tiny detector must not drag in the real detector package."""
    import importlib
    import sys

    module = importlib.import_module(f"models.detectors.{tiny_detector}")
    assert hasattr(module, "ModelOut")
    assert "efficientnet_pytorch" not in sys.modules, (
        "importing the tiny detector pulled in the real detector zoo"
    )


def test_tiny_detector_forward(tiny_detector, tiny_batch):
    import importlib
    import torch

    model_out = importlib.import_module(f"models.detectors.{tiny_detector}").ModelOut(
        pretrained=False, finetune=False, output_classes=1, classification_strategy="binary"
    )
    with torch.no_grad():
        output = model_out(tiny_batch(batch_size=3, size=16))
    assert output.shape == (3, 1)


def test_tiny_detector_no_linear_has_no_linear(tiny_detector_no_linear):
    import importlib
    import torch.nn as nn

    model_out = importlib.import_module(
        f"models.detectors.{tiny_detector_no_linear}"
    ).ModelOut()
    linears = [m for m in model_out.modules() if isinstance(m, nn.Linear)]
    assert linears == [], "this fixture must reproduce squeezenetdf's zero-Linear shape"
    dropouts = [m for m in model_out.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    assert len(dropouts) == 1, "and it must keep one p>0 dropout, like squeezenet's classifier.0"


def test_seed_for_is_stable_and_independent():
    from test_helpers.determinism import rng_for, seed_for

    assert seed_for("traversal.RandomTraversal") == seed_for("traversal.RandomTraversal")
    assert seed_for("traversal.RandomTraversal") != seed_for("traversal.RandomWarpTraversal")

    # Draining one component's stream must not affect another's.
    first = rng_for("dqn.replay")
    baseline = [rng_for("traversal.RandomTraversal").random() for _ in range(3)]
    for _ in range(100):
        first.random()
    after = [rng_for("traversal.RandomTraversal").random() for _ in range(3)]
    assert baseline == after
