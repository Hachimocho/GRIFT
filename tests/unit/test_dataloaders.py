"""Dataloader graph construction, driven entirely off synthetic nodes.

`DummyDataset` supplies nodes directly, so these never touch `AIFaceDataset`,
CSVs, or images. `test_mode=True, silent_mode=True` plus permissive thresholds
keep them fast.
"""

import copy

import pytest

from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from edges.Edge import Edge
from tests.helpers.factories import DummyDataset, make_attr_nodes

DATALOADER_CLASSES = (HierarchicalDeepfakeDataloader, UnclusteredDeepfakeDataloader)
PERMISSIVE = dict(
    test_mode=True,
    silent_mode=True,
    quality_threshold=0.0,
    symmetry_threshold=0.0,
    embedding_threshold=-1.0,
)


def build_dataloader(dataloader_class, node_count=6, **overrides):
    nodes = make_attr_nodes(node_count)
    kwargs = dict(PERMISSIVE)
    kwargs.update(overrides)
    return dataloader_class([DummyDataset(nodes)], Edge, **kwargs), nodes


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_builds_three_graphs(dataloader_class):
    dataloader, nodes = build_dataloader(dataloader_class)
    train_graph, val_graph, test_graph = dataloader.load()

    assert len(train_graph.get_nodes()) > 0
    assert isinstance(train_graph.num_edges(), int)
    # All fixture nodes are split='train', so val/test are empty but must exist.
    for graph in (val_graph, test_graph):
        assert graph is not None
        assert isinstance(graph.num_edges(), int)


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_graph_contains_only_supplied_nodes(dataloader_class):
    dataloader, nodes = build_dataloader(dataloader_class)
    train_graph, _, _ = dataloader.load()
    supplied = {node.node_id for node in nodes}
    assert {node.node_id for node in train_graph.get_nodes()} <= supplied


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_load_is_reproducible(dataloader_class):
    """Same seed -> same edge set.

    Edge construction is seed-dependent via the isolated-node fallback, which
    picks a random partner for any node that ended up with no edges.
    """
    from test_helpers.determinism import configure_determinism

    def edge_set():
        configure_determinism(seed=7, mode="strict", allow_multi_gpu=True)
        dataloader, _ = build_dataloader(dataloader_class)
        train_graph, _, _ = dataloader.load()
        return sorted(train_graph.get_edge_list())

    assert edge_set() == edge_set()


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_construction_does_not_mutate_class_attribute(dataloader_class):
    """Constructing a dataloader must not write to the class-level dict.

    Regression test for the class-attribute leak: `hyperparameters` is a class
    attribute and was mutated via `.update(kwargs)`, so every dataloader in a
    process shared one config dict. `run_threshold_grid_search` builds one loader
    per grid point, so each silently inherited the previous point's thresholds --
    meaning past `--search` results did not measure what they reported.
    """
    pristine = copy.deepcopy(dataloader_class.hyperparameters)
    build_dataloader(dataloader_class, quality_threshold=0.123456)
    assert dataloader_class.hyperparameters == pristine, (
        "constructing a dataloader mutated the class-level hyperparameters dict"
    )


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_instances_do_not_share_config(dataloader_class):
    """Two dataloaders built with different settings must stay independent.

    Before the fix this failed outright: building the second loader with
    quality_threshold=0.75 retroactively changed the first loader's value from
    0.25 to 0.75.
    """
    first, _ = build_dataloader(dataloader_class, quality_threshold=0.25)
    second, _ = build_dataloader(dataloader_class, quality_threshold=0.75)

    assert second.hyperparameters["quality_threshold"] == 0.75
    assert first.hyperparameters["quality_threshold"] == 0.25, (
        "building a second dataloader changed the first one's configuration"
    )


@pytest.mark.parametrize("dataloader_class", DATALOADER_CLASSES, ids=lambda c: c.__name__)
def test_dataloader_defaults_are_present_after_kwargs_override(dataloader_class):
    """Overriding one setting must not drop the other defaults."""
    dataloader, _ = build_dataloader(dataloader_class, quality_threshold=0.4)
    assert dataloader.hyperparameters["quality_threshold"] == 0.4
    for key in dataloader_class.hyperparameters:
        assert key in dataloader.hyperparameters, f"default {key!r} was lost"
