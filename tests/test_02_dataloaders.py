import random

from dataloaders.ConnectedClusteredDeepfakeDataloader import ConnectedClusteredDeepfakeDataloader
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from tests.graph_test_support import FakeDataset, SampleEdge, make_sample_nodes, make_split_nodes, unique_edge_count


def test_unclustered_dataloader_builds_train_graph_from_preloaded_nodes():
    nodes = make_split_nodes(train=100, val=8, test=8)
    dataloader = UnclusteredDeepfakeDataloader(
        datasets=[],
        edge_class=SampleEdge,
        quality_threshold=0.95,
        symmetry_threshold=0.95,
        embedding_threshold=0.80,
        silent_mode=True,
    )

    train_graph, val_graph, test_graph = dataloader.load(preloaded_nodes=nodes)

    assert len(train_graph.get_nodes()) == 100
    assert len(val_graph.get_nodes()) == 8
    assert len(test_graph.get_nodes()) == 8
    assert unique_edge_count(train_graph) > 0, "Train graph should contain graph edges"
    assert unique_edge_count(val_graph) == 0, "Validation graph is expected to be node-only"
    assert unique_edge_count(test_graph) == 0, "Test graph is expected to be node-only"


def test_hierarchical_dataloader_builds_train_graph_from_preloaded_nodes():
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

    train_graph, val_graph, test_graph = dataloader.load(preloaded_nodes=nodes)

    assert len(train_graph.get_nodes()) == 100
    assert len(val_graph.get_nodes()) == 8
    assert len(test_graph.get_nodes()) == 8
    assert unique_edge_count(train_graph) > 0, "Train graph should contain graph edges"
    assert unique_edge_count(val_graph) == 0
    assert unique_edge_count(test_graph) == 0


def test_connected_clustered_dataloader_loads_100_sample_nodes_into_graph():
    random.seed(42)
    nodes = make_sample_nodes(total=100, split="train", num_groups=1)
    dataset = FakeDataset(nodes)
    dataloader = ConnectedClusteredDeepfakeDataloader(
        datasets=[dataset],
        edge_class=SampleEdge,
        buffer_connect_chance=0.0,
    )

    graph = dataloader.load()
    node_ids = {node.node_id for node in graph.get_nodes()}

    assert len(graph.get_nodes()) == 101, "Expected 100 sample nodes plus the buffer node"
    assert len(node_ids.intersection({node.node_id for node in nodes})) == 100
    assert "buffer-train" in node_ids
    assert unique_edge_count(graph) > 0
