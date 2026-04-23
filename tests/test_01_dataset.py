import numpy as np

from nodes.atrnode import AttributeNode
from tests.graph_test_support import SampleAttributeDataset, make_sample_records


def test_sample_dataset_converts_records_into_attribute_nodes():
    records = make_sample_records(total=6, split="train", num_groups=3)
    dataset = SampleAttributeDataset(records, embedding_length=8, threshold=75)

    nodes = dataset.load()

    assert len(nodes) == 6, "Expected one node per sample record"
    assert all(isinstance(node, AttributeNode) for node in nodes), "Dataset should emit AttributeNode instances"


def test_sample_dataset_preserves_split_label_and_identity_fields():
    record = make_sample_records(total=1, split="val", num_groups=1)[0]
    dataset = SampleAttributeDataset([record], embedding_length=8, threshold=82)

    node = dataset.load()[0]

    assert node.node_id == record.node_id
    assert node.split == "val"
    assert node.label == record.label
    assert node.threshold == 82
    assert node.attributes["Target"] == record.label
    assert node.attributes["subset"] == "val"


def test_sample_dataset_adds_expected_graph_attributes():
    record = make_sample_records(total=1, split="test", num_groups=1)[0]
    dataset = SampleAttributeDataset([record], embedding_length=8)

    node = dataset.load()[0]

    assert node.attributes[f"gender_{record.gender}"] is True
    assert node.attributes[f"race_{record.race}"] is True
    assert node.attributes[f"age_{record.age}"] is True
    assert "blur" in node.attributes
    assert "brightness" in node.attributes
    assert "contrast" in node.attributes
    assert "compression" in node.attributes
    assert "symmetry_eye" in node.attributes
    assert "symmetry_mouth" in node.attributes
    assert "symmetry_nose" in node.attributes
    assert "symmetry_overall" in node.attributes


def test_sample_dataset_creates_stable_face_embeddings():
    record = make_sample_records(total=1, split="train", num_groups=1)[0]
    dataset = SampleAttributeDataset([record], embedding_length=8)

    node = dataset.load()[0]
    embedding = node.attributes["face_embedding"]

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (8,)
    assert np.isclose(np.linalg.norm(embedding), 1.0), "Expected normalized face embeddings"
