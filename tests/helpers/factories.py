"""Deterministic builders for nodes, edges, and graphs.

Adapted from the abandoned `cursor/develop-generic-unit-tests-*` branch, with
three staleness fixes:

* attributes use the key ``face_embedding`` (not ``embedding``), so
  ``AttributeNode.compute_similarity`` and the graph-distance uncertainty
  methods actually exercise their embedding paths;
* demographics are stored as ``np.int64``, matching what
  ``AIFaceDataset`` produces from pandas -- this is what makes the
  ``isinstance(value, (int, float))`` bug in ``graph_distance.py`` reproducible
  in a unit test rather than only against the real dataset;
* traversal discovery reflects over ``Traversal`` subclasses instead of a
  hardcoded allowlist that silently omitted half of them.
"""

import inspect

import numpy as np

from edges.Edge import Edge
from graphs.HyperGraph import HyperGraph
from nodes.atrnode import AttributeNode
from nodes.Node import Node

# Continuous attributes that graph_distance.CONTINUOUS_ATTRIBUTES reads. Note the
# deliberate scale mismatch: blur/brightness/contrast/compression are unbounded
# (real values run into the hundreds) while symmetry_* and emotion_* are in
# [0, 1]. Fixtures reproduce that spread so the normalization fix is testable.
QUALITY_ATTRS = ("blur", "brightness", "contrast", "compression")
SYMMETRY_ATTRS = ("symmetry_eye", "symmetry_mouth", "symmetry_nose", "symmetry_overall")
EMOTION_ATTRS = (
    "emotion_angry", "emotion_disgust", "emotion_fear", "emotion_happy",
    "emotion_sad", "emotion_surprise", "emotion_neutral",
)
DEMOGRAPHIC_ATTRS = ("Ground Truth Gender", "Ground Truth Race", "Ground Truth Age")


def make_attributes(index, embedding_dim=8, embedding=True, demographics=True):
    """Attributes shaped like a real AI-Face node's.

    Values are a deterministic function of ``index`` -- no RNG -- so fixtures are
    reproducible without depending on seeding order.
    """
    attrs = {}

    # Unbounded quality metrics. blur spans a wide range on purpose.
    attrs["blur"] = float(10 + index * 37)
    attrs["brightness"] = float(80 + index * 11)
    attrs["contrast"] = float(30 + index * 5)
    attrs["compression"] = float(index % 7)

    for offset, name in enumerate(SYMMETRY_ATTRS):
        attrs[name] = round(((index * 7 + offset * 13) % 100) / 100.0, 4)
    for offset, name in enumerate(EMOTION_ATTRS):
        attrs[name] = round(((index * 11 + offset * 17) % 100) / 100.0, 4)

    if demographics:
        # np.int64, exactly as pandas hands them over. Python's `int` and
        # np.int64 are NOT the same for isinstance checks.
        attrs["Ground Truth Gender"] = np.int64(index % 2)
        attrs["Ground Truth Race"] = np.int64(index % 4)
        attrs["Ground Truth Age"] = np.int64(index % 3)

    # Boolean one-hots, which the dataloaders' _group_by_categorical requires.
    attrs[f"gender_{'male' if index % 2 == 0 else 'female'}"] = True
    attrs[f"race_{['black', 'white', 'asian', 'indian'][index % 4]}"] = True
    attrs[f"age_{['young', 'middle', 'senior'][index % 3]}"] = True

    if embedding:
        vector = np.asarray(
            [np.sin(index + offset + 1) for offset in range(embedding_dim)],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        attrs["face_embedding"] = (vector / norm).astype(np.float32)

    attrs["Target"] = int(index % 2)
    return attrs


def make_attr_node(index, split="train", label=None, threshold=50, data=None, **kwargs):
    """One AttributeNode with realistic attributes."""
    return AttributeNode(
        node_id=f"n{index}",
        split=split,
        data=data,
        edges=[],
        label=int(index % 2) if label is None else int(label),
        attributes=make_attributes(index, **kwargs),
        threshold=threshold,
    )


def make_attr_nodes(count=6, split="train", **kwargs):
    return [make_attr_node(index, split=split, **kwargs) for index in range(count)]


def connect_ring(nodes):
    """Ring topology: every node has degree exactly 2 (degree 1 if count == 2)."""
    edges = []
    total = len(nodes)
    if total < 2:
        return edges
    for index in range(total):
        left = nodes[index]
        right = nodes[(index + 1) % total]
        if total == 2 and index == 1:
            break  # avoid a duplicate edge between the same pair
        edge = Edge(left, right, x={"w": 1.0})
        left.add_edge(edge)
        right.add_edge(edge)
        edges.append(edge)
    return edges


def connect_clique(nodes):
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edge = Edge(nodes[i], nodes[j], x={"w": 1.0})
            nodes[i].add_edge(edge)
            nodes[j].add_edge(edge)
            edges.append(edge)
    return edges


def build_ring_graph(count=6, **kwargs):
    nodes = make_attr_nodes(count, **kwargs)
    edges = connect_ring(nodes)
    return HyperGraph(nodes), nodes, edges


def build_two_cluster_graph(per_cluster=4, **kwargs):
    """Two cliques joined by a single bridge edge.

    Exercises Louvain partitioning, k-hop expansion, the degree penalty (bridge
    nodes have higher degree), and subgroup grouping in one fixture.
    """
    nodes = make_attr_nodes(per_cluster * 2, **kwargs)
    left, right = nodes[:per_cluster], nodes[per_cluster:]
    edges = connect_clique(left) + connect_clique(right)
    bridge = Edge(left[-1], right[0], x={"w": 1.0})
    left[-1].add_edge(bridge)
    right[0].add_edge(bridge)
    edges.append(bridge)
    return HyperGraph(nodes), nodes, edges


def build_isolated_node_graph(**kwargs):
    """A single edgeless node -- the `1.0 + penalty` branch of graph uncertainty."""
    nodes = make_attr_nodes(1, **kwargs)
    return HyperGraph(nodes), nodes, []


def iter_package_classes(package, base=None):
    """Reflect over a GRIFT package's exported classes.

    The packages' ``__init__`` files auto-import their modules, so
    ``inspect.getmembers`` sees every concrete class.
    """
    found = {}
    for name, obj in inspect.getmembers(package, inspect.isclass):
        if not getattr(obj, "__module__", "").startswith(package.__name__ + "."):
            continue
        if base is not None:
            try:
                if not issubclass(obj, base):
                    continue
            except TypeError:
                continue
        found[name] = obj
    return found


def get_node_classes():
    import nodes as nodes_pkg
    return iter_package_classes(nodes_pkg, Node)


def get_edge_classes():
    import edges as edges_pkg
    return iter_package_classes(edges_pkg, Edge)


def get_traversal_classes(include_base=False):
    """Every Traversal subclass, discovered by reflection.

    The upstream branch hardcoded a four-name allowlist, which silently skipped
    RandomWarpTraversal, RandomNoReturnTraversal, RandomNoReturnWarpTraversal,
    and the single IValueTraversal (which picks its own walk from the graph).
    """
    import traversals as traversals_pkg
    from traversals.Traversal import Traversal

    classes = iter_package_classes(traversals_pkg, Traversal)
    if not include_base:
        classes.pop("Traversal", None)
    return classes


class DummyDataset:
    """Minimal Dataset stand-in: just enough for the dataloaders' load().

    Bypasses AIFaceDataset entirely, so dataloader tests need no CSVs, no images,
    and no disk I/O.
    """

    tags = ["deepfakes"]
    hyperparameters = {"parameters": {}}

    def __init__(self, nodes):
        self._nodes = list(nodes)

    def load(self):
        return list(self._nodes)


def build_traversal(traversal_class, graph, num_pointers=2, num_steps=10, trainer=None, **overrides):
    """Construct any traversal by introspecting its ``__init__``.

    The traversal constructors are inconsistent about which knobs are required
    positionally (``RandomWarpTraversal`` demands ``warp_chance``,
    ``RandomNoReturnWarpTraversal`` demands both ``return_delay`` and
    ``warp_chance``, the I-value ones default everything), so tests that sweep
    over all of them need this. Unlike the version on the abandoned branch, an
    unfillable parameter raises instead of being swallowed by a bare ``except``
    -- otherwise a test can silently verify nothing.
    """
    defaults = {
        "graph": graph,
        "num_pointers": num_pointers,
        "num_steps": num_steps,
        "trainer": trainer if trainer is not None else DummyTrainer(),
        "return_delay": 10,
        "warp_chance": 0.005,
        "predictor_update_period": 50,
        "bias_hop_period": 100,
        "pessimistic_i_value": 1.0,
        "outlier_std": 2.0,
        "softmax_temp": 0.5,
    }
    defaults.update(overrides)

    signature = inspect.signature(traversal_class.__init__)
    kwargs = {}
    unfillable = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        if name in defaults:
            kwargs[name] = defaults[name]
        elif parameter.default is inspect.Parameter.empty:
            unfillable.append(name)

    if unfillable:
        raise TypeError(
            f"{traversal_class.__name__}.__init__ requires parameter(s) {unfillable} that "
            f"build_traversal has no default for; add one to tests/helpers/factories.py"
        )
    return traversal_class(**kwargs)


class DummyTrainer:
    """Stand-in for the trainer interface IValueTraversal needs.

    Returns a deterministic I-value derived from the node id, so traversal tests
    have a defined argmax and do not depend on the DQN.
    """

    def __init__(self, i_values=None):
        self._i_values = i_values or {}
        self.calls = []

    def get_i_value(self, node, model_index=0):
        self.calls.append(getattr(node, "node_id", None))
        node_id = getattr(node, "node_id", "")
        if node_id in self._i_values:
            return self._i_values[node_id]
        digits = "".join(char for char in str(node_id) if char.isdigit())
        return (int(digits) % 10) / 10.0 if digits else 0.5
