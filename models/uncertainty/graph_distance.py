import math
from typing import Dict, Iterable, Optional

import numpy as np
import torch


CONTINUOUS_ATTRIBUTES = [
    "blur",
    "brightness",
    "contrast",
    "compression",
    "symmetry_eye",
    "symmetry_mouth",
    "symmetry_nose",
    "symmetry_overall",
    "emotion_angry",
    "emotion_disgust",
    "emotion_fear",
    "emotion_happy",
    "emotion_sad",
    "emotion_surprise",
    "emotion_neutral",
]

CATEGORICAL_ATTRIBUTES = [
    "Ground Truth Gender",
    "Ground Truth Race",
    "Ground Truth Age",
]


def _degree_penalty(node, penalty_weight=1.0):
    degree = len(node.get_adjacent_nodes()) if hasattr(node, "get_adjacent_nodes") else 0
    return penalty_weight / math.sqrt(float(degree) + 1.0)


def _build_attribute_vector(node):
    attrs = getattr(node, "attributes", {}) or {}
    vector = []
    for attr_name in CONTINUOUS_ATTRIBUTES:
        try:
            vector.append(float(attrs.get(attr_name, 0.0)))
        except (TypeError, ValueError):
            vector.append(0.0)

    for attr_name in CATEGORICAL_ATTRIBUTES:
        value = attrs.get(attr_name, None)
        vector.append(float(value) if isinstance(value, (int, float)) else 0.0)

    return np.asarray(vector, dtype=np.float32)


def _embedding_distance(node, neighbor):
    attrs = getattr(node, "attributes", {}) or {}
    neighbor_attrs = getattr(neighbor, "attributes", {}) or {}
    emb_a = attrs.get("face_embedding")
    emb_b = neighbor_attrs.get("face_embedding")
    if emb_a is None or emb_b is None:
        return None

    emb_a = np.asarray(emb_a, dtype=np.float32)
    emb_b = np.asarray(emb_b, dtype=np.float32)
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return None
    cosine_similarity = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return 1.0 - cosine_similarity


def _attribute_distance(node, neighbor):
    vector_a = _build_attribute_vector(node)
    vector_b = _build_attribute_vector(neighbor)
    denom = float(np.linalg.norm(vector_a) + np.linalg.norm(vector_b) + 1e-6)
    return float(np.linalg.norm(vector_a - vector_b) / denom)


def _hybrid_distance(node, neighbor):
    distances = []
    attribute_distance = _attribute_distance(node, neighbor)
    if attribute_distance is not None:
        distances.append(attribute_distance)

    embedding_distance = _embedding_distance(node, neighbor)
    if embedding_distance is not None:
        distances.append(embedding_distance)

    if not distances:
        return None
    return float(np.mean(distances))


DISTANCE_METHODS = {
    "attribute_distance": _attribute_distance,
    "embedding_distance": _embedding_distance,
    "hybrid_distance": _hybrid_distance,
}


def compute_graph_uncertainty(node, methods, penalty_weight=1.0):
    neighbors = node.get_adjacent_nodes() if hasattr(node, "get_adjacent_nodes") else []
    penalty = _degree_penalty(node, penalty_weight=penalty_weight)
    if not neighbors:
        return {method_name: float(1.0 + penalty) for method_name in methods}

    uncertainty = {}
    for method_name in methods:
        method = DISTANCE_METHODS.get(method_name)
        if method is None:
            continue

        distances = []
        for neighbor in neighbors:
            distance = method(node, neighbor)
            if distance is not None and not math.isnan(distance):
                distances.append(distance)

        if not distances:
            uncertainty[method_name] = float(1.0 + penalty)
        else:
            uncertainty[method_name] = float(np.mean(distances) + penalty)

    return uncertainty


def compute_batch_graph_uncertainty(nodes, methods, penalty_weight=1.0):
    per_node_values = {method_name: [] for method_name in methods}
    for node in nodes:
        node_uncertainty = compute_graph_uncertainty(node, methods, penalty_weight=penalty_weight)
        for method_name, value in node_uncertainty.items():
            per_node_values.setdefault(method_name, []).append(value)

    tensor_uncertainty = {}
    for method_name, values in per_node_values.items():
        if values:
            tensor_uncertainty[method_name] = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    return tensor_uncertainty
