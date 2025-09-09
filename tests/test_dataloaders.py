import numpy as np
import pytest
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from nodes.atrnode import AttributeNode
from edges.Edge import Edge


class DummyDataset:
    def __init__(self, nodes):
        self._nodes = nodes

    def load(self):
        return list(self._nodes)


def build_attr_nodes(count=6):
    nodes = []
    # Alternate gender and race to create small subgroups
    for i in range(count):
        attrs = {
            'race_black': (i % 2 == 0),
            'race_white': (i % 2 == 1),
            'gender_male': (i % 2 == 0),
            'gender_female': (i % 2 == 1),
            'blur': float(i),
            'brightness': float(i),
            'contrast': float(i),
            'compression': float(i),
            'symmetry_overall': 0.8,
            'face_embedding': np.ones(8, dtype=float),
        }
        split = 'train' if i < count else 'val'
        nodes.append(AttributeNode(str(i), split, None, [], int(i % 2), attrs, threshold=50))
    return nodes


def test_unclustered_dataloader_small_graph():
    nodes = build_attr_nodes(6)
    ds = DummyDataset(nodes)
    dl = UnclusteredDeepfakeDataloader([ds], Edge, test_mode=True, silent_mode=True, quality_threshold=0.0, symmetry_threshold=0.0, embedding_threshold=-1.0)
    train_g, val_g, test_g = dl.load()
    # Train graph should contain the training nodes
    assert len(train_g.get_nodes()) > 0
    # Edge count should be non-negative and well-defined
    assert isinstance(train_g.num_edges(), int)


def test_hierarchical_dataloader_small_graph():
    nodes = build_attr_nodes(6)
    ds = DummyDataset(nodes)
    dl = HierarchicalDeepfakeDataloader([ds], Edge, test_mode=True, silent_mode=True, quality_threshold=0.0, symmetry_threshold=0.0, embedding_threshold=-1.0)
    train_g, val_g, test_g = dl.load()
    assert len(train_g.get_nodes()) > 0
    assert isinstance(train_g.num_edges(), int)

