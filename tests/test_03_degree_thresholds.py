import random

from dataloaders.ConnectedClusteredDeepfakeDataloader import ConnectedClusteredDeepfakeDataloader
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from tests.graph_test_support import (
    FakeDataset,
    SampleEdge,
    average_degree,
    make_sample_nodes,
    make_split_nodes,
    unique_edge_count,
)


def test_unclustered_train_graph_average_degree_stays_in_expected_range():
    nodes = make_split_nodes(train=100, val=8, test=8)
    dataloader = UnclusteredDeepfakeDataloader(
        datasets=[],
        edge_class=SampleEdge,
        quality_threshold=0.95,
        symmetry_threshold=0.95,
        embedding_threshold=0.80,
        silent_mode=True,
    )

    train_graph, _, _ = dataloader.load(preloaded_nodes=nodes)
    avg_degree = average_degree(train_graph)

    assert unique_edge_count(train_graph) > 0
    assert 20.0 <= avg_degree <= 28.0, f"Unexpected unclustered average degree: {avg_degree}"


def test_hierarchical_train_graph_average_degree_stays_in_expected_range():
    nodes = make_split_nodes(train=100, val=8, test=8)
    dataloader = HierarchicalDeepfakeDataloader(
        datasets=[],
        edge_class=SampleEdge,
        quality_threshold=0.95,
        symmetry_threshold=0.95,
        embedding_threshold=0.80,
        silent_mode=True,
        assign_subclusters=False,
    )

    train_graph, _, _ = dataloader.load(preloaded_nodes=nodes)
    avg_degree = average_degree(train_graph)

    assert unique_edge_count(train_graph) > 0
    assert 20.0 <= avg_degree <= 28.0, f"Unexpected hierarchical average degree: {avg_degree}"


def test_connected_clustered_graph_average_degree_stays_in_expected_range():
    random.seed(42)
    nodes = make_sample_nodes(total=100, split="train", num_groups=1)
    dataloader = ConnectedClusteredDeepfakeDataloader(
        datasets=[FakeDataset(nodes)],
        edge_class=SampleEdge,
        buffer_connect_chance=0.0,
    )

    graph = dataloader.load()
    avg_degree = average_degree(graph)

    assert unique_edge_count(graph) > 0
    assert 95.0 <= avg_degree <= 99.0, f"Unexpected connected average degree: {avg_degree}"
