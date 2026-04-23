from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.Data import Data
from edges.Edge import Edge
from nodes.atrnode import AttributeNode


@dataclass(frozen=True)
class SampleRecord:
    node_id: str
    split: str
    label: int
    gender: str
    race: str
    age: str
    group_id: int
    sample_index: int


class SampleEdge(Edge):
    """Flexible edge used by tests and by loaders with inconsistent edge constructors."""

    def __init__(self, node1, node2, x=None, traversal_weight=1):
        super().__init__(node1, node2, x, traversal_weight=traversal_weight)


class FakeDataset:
    """Small dataset stub for dataloaders that expect a .load() method."""

    def __init__(self, nodes):
        self._nodes = nodes

    def load(self):
        return self._nodes


class SampleAttributeDataset:
    """
    Lightweight dataset adapter used only in tests.

    It converts deterministic sample records into AttributeNode objects without
    depending on external CSVs or image files.
    """

    def __init__(self, records, embedding_length=8, threshold=80):
        self.records = records
        self.embedding_length = embedding_length
        self.threshold = threshold

    def load(self):
        return [record_to_node(record, self.embedding_length, self.threshold) for record in self.records]


def make_sample_records(total=100, split="train", num_groups=4):
    records = []
    for index in range(total):
        group_id = index % num_groups
        label = group_id % 2
        gender = "female" if group_id % 2 == 0 else "male"
        race = "asian" if group_id < (num_groups / 2) else "white"
        age = "young" if group_id % 3 != 0 else "adult"
        records.append(
            SampleRecord(
                node_id=f"{split}-node-{index:03d}",
                split=split,
                label=label,
                gender=gender,
                race=race,
                age=age,
                group_id=group_id,
                sample_index=index,
            )
        )
    return records


def make_split_records(train=100, val=12, test=12):
    records = []
    records.extend(make_sample_records(train, split="train"))
    records.extend(make_sample_records(val, split="val"))
    records.extend(make_sample_records(test, split="test"))
    return records


def build_face_embedding(group_id, sample_index, embedding_length=8):
    base = np.zeros(embedding_length, dtype=np.float32)
    base[group_id % embedding_length] = 1.0
    base[(group_id + 1) % embedding_length] = 0.5
    base[(sample_index + 2) % embedding_length] += 0.01
    norm = np.linalg.norm(base)
    return base if norm == 0 else (base / norm).astype(np.float32)


def record_to_node(record, embedding_length=8, threshold=80):
    group_offset = record.group_id * 0.01
    attributes = {
        f"gender_{record.gender}": True,
        f"race_{record.race}": True,
        f"age_{record.age}": True,
        "blur": 0.20 + group_offset,
        "brightness": 0.40 + group_offset,
        "contrast": 0.60 + group_offset,
        "compression": 0.80 + group_offset,
        "symmetry_eye": 0.91 + group_offset,
        "symmetry_mouth": 0.89 + group_offset,
        "symmetry_nose": 0.92 + group_offset,
        "symmetry_overall": 0.90 + group_offset,
        "emotion_happy": 0.85 if record.label == 0 else 0.15,
        "emotion_neutral": 0.15 if record.label == 0 else 0.85,
        "face_embedding": build_face_embedding(
            group_id=record.group_id,
            sample_index=record.sample_index,
            embedding_length=embedding_length,
        ),
        "Target": record.label,
        "subset": record.split,
        "group_id": record.group_id,
    }
    return AttributeNode(
        record.node_id,
        record.split,
        Data(record.node_id),
        [],
        record.label,
        attributes,
        threshold,
    )


def make_sample_nodes(total=100, split="train", num_groups=4, embedding_length=8):
    records = make_sample_records(total=total, split=split, num_groups=num_groups)
    return [record_to_node(record, embedding_length=embedding_length) for record in records]


def make_split_nodes(train=100, val=12, test=12, num_groups=4, embedding_length=8):
    records = make_split_records(train=train, val=val, test=test)
    return [
        record_to_node(record, embedding_length=embedding_length)
        for record in records
    ]


def unique_edge_count(graph):
    return len(graph.get_edge_list())


def average_degree(graph):
    node_count = len(graph.get_nodes())
    if node_count == 0:
        return 0.0
    return (2 * unique_edge_count(graph)) / node_count
