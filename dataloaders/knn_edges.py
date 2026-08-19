"""Candidate-edge generation for graph construction.

The graph was built by enumerating **every pair** of nodes into a Python list and then
filtering by similarity:

    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

That is O(N^2) in both time and memory, and the memory is spent *before* any filtering, so
raising the similarity thresholds does not help. Measured costs of that one line:

    nodes      candidate pairs     RAM for the list
    5,000      1.25e7              0.7 GiB
    50,000     1.25e9             74.5 GiB
    1,600,000  1.28e12         76,294 GiB

The full AI-Face corpus is ~1.6M nodes, so all-pairs needs ~76 TiB of RAM to enumerate
candidates on a 541 GiB machine. There is a second, independent wall: at the project's
default thresholds the surviving graph is extremely dense -- a measured 1,304-node split
produced 823,814 edges, **97.0% of all possible pairs**, average degree 1,264 -- so even a
lazy enumeration would leave ~1.24e12 edges to store.

k-NN construction removes both walls. Each node keeps its `k` nearest neighbours by cosine
distance over the face embedding, so candidates are O(kN) and the graph is sparse by
construction: at k=50 the full corpus is ~8e7 edges rather than 1.24e12, and average degree
is 50 rather than 1,264. Similarity filtering still runs afterwards, unchanged, so an edge
means the same thing it always did -- there are simply no longer N^2 chances to make one.

A sparse graph is also what the graph-topology parts of this project need in order to mean
anything. At degree 1,264 every node is adjacent to 97% of the graph, which is why the
performance updater's rewiring was measured at ~0.01% of the topology and why the
graph-distance uncertainty methods scored barely above their degree-only control.

`NearestNeighbors` rather than an ANN index: sklearn is already a dependency, faiss is not,
and exact k-NN keeps the construction reproducible. Output is sorted, so the candidate list
does not depend on dict ordering or PYTHONHASHSEED.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Construction modes. `all_pairs` is the original behaviour, kept so an existing result can
#: be reproduced; it cannot scale past a few tens of thousands of nodes.
EDGE_CONSTRUCTION_MODES = ("knn", "all_pairs")

#: Neighbours per node under `knn`. 50 keeps the graph connected while leaving the degree
#: distribution far below the ~1,264 that made topology-based methods inert.
DEFAULT_KNN_NEIGHBOURS = 50

#: Node attribute holding the face embedding, and its width when one is missing.
EMBEDDING_KEY = "face_embedding"
EMBEDDING_DIM = 512

#: Above this many nodes, all-pairs is refused rather than attempted -- the list alone would
#: be ~75 GiB at 50,000 nodes, and an OOM kill mid-build is worse than an early error.
ALL_PAIRS_NODE_LIMIT = 30_000


class EdgeConstructionError(ValueError):
    """Raised when candidate edges cannot be generated as requested."""


def embedding_matrix(nodes, indices=None, key=EMBEDDING_KEY, dim=EMBEDDING_DIM):
    """Float32 matrix of embeddings for `indices` (default: all nodes).

    A node with no usable embedding contributes a zero row rather than being dropped, so the
    row order matches `indices` exactly and the caller's index mapping stays valid.
    """
    indices = range(len(nodes)) if indices is None else indices
    rows, missing = [], 0
    for index in indices:
        value = getattr(nodes[index], "attributes", {}).get(key)
        if isinstance(value, np.ndarray) and value.size:
            rows.append(value.astype(np.float32, copy=False).reshape(-1))
        else:
            rows.append(np.zeros(dim, dtype=np.float32))
            missing += 1
    if not rows:
        return np.zeros((0, dim), dtype=np.float32), 0
    width = max(row.shape[0] for row in rows)
    padded = [
        row if row.shape[0] == width
        else np.pad(row, (0, width - row.shape[0]))
        for row in rows
    ]
    return np.asarray(padded, dtype=np.float32), missing


def knn_candidate_edges(nodes, k=DEFAULT_KNN_NEIGHBOURS, indices=None, metric="cosine"):
    """`k` nearest neighbours per node, as sorted unique `(i, j)` index pairs with `i < j`.

    `indices` restricts the search to a subset (a race-gender group, say) and the returned
    pairs are in the caller's original index space.

    Symmetrised: if `j` is among `i`'s neighbours the undirected edge is kept, so a node's
    realised degree can exceed `k`. That is deliberate -- dropping the asymmetric half would
    make the graph depend on which endpoint was queried.
    """
    from sklearn.neighbors import NearestNeighbors

    indices = list(range(len(nodes))) if indices is None else list(indices)
    count = len(indices)
    if count <= 1:
        return []

    neighbours = max(1, min(int(k), count - 1))
    matrix, missing = embedding_matrix(nodes, indices)
    if missing:
        logger.warning(
            "%d of %d nodes have no %s; they get a zero embedding and will cluster "
            "together, which is visible as one dense component rather than as an error.",
            missing, count, EMBEDDING_KEY,
        )

    try:
        index = NearestNeighbors(
            n_neighbors=neighbours + 1, algorithm="auto", metric=metric, n_jobs=-1,
        )
        index.fit(matrix)
        _distances, found = index.kneighbors(matrix, return_distance=True)
    except Exception as error:
        raise EdgeConstructionError(
            f"k-NN search over {count} nodes failed: {error}"
        ) from error

    pairs = set()
    for local_source, row in enumerate(found):
        source = indices[local_source]
        for local_target in row:
            target = indices[int(local_target)]
            if target == source:
                continue
            pairs.add((source, target) if source < target else (target, source))
    # Sorted: a set's iteration order is not stable across processes.
    return sorted(pairs)


def all_pairs_candidate_edges(n_nodes, limit=ALL_PAIRS_NODE_LIMIT):
    """Every `(i, j)` pair with `i < j`. Refuses above `limit`."""
    if n_nodes > limit:
        pairs = n_nodes * (n_nodes - 1) // 2
        raise EdgeConstructionError(
            f"all_pairs on {n_nodes:,} nodes would enumerate {pairs:,} candidate pairs "
            f"(~{pairs * 64 / 2 ** 30:.0f} GiB just for the list) before any filtering. "
            f"Refusing rather than being OOM-killed mid-build. Use "
            f"--edge-construction knn, which is O(k*N)."
        )
    return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]


def candidate_edges(nodes, mode="knn", k=DEFAULT_KNN_NEIGHBOURS, indices=None):
    """Candidate edges for `mode`. Returns sorted `(i, j)` pairs in `nodes` index space."""
    if mode not in EDGE_CONSTRUCTION_MODES:
        raise EdgeConstructionError(
            f"unknown edge construction {mode!r}; choose from "
            f"{', '.join(EDGE_CONSTRUCTION_MODES)}"
        )
    if mode == "knn":
        return knn_candidate_edges(nodes, k=k, indices=indices)

    if indices is None:
        return all_pairs_candidate_edges(len(nodes))
    indices = list(indices)
    if len(indices) > ALL_PAIRS_NODE_LIMIT:
        all_pairs_candidate_edges(len(indices))  # raises with the explanation
    return [
        (a, b) if a < b else (b, a)
        for position, a in enumerate(indices) for b in indices[position + 1:]
    ]


__all__ = [
    "ALL_PAIRS_NODE_LIMIT", "DEFAULT_KNN_NEIGHBOURS", "EDGE_CONSTRUCTION_MODES",
    "EMBEDDING_DIM", "EMBEDDING_KEY", "EdgeConstructionError",
    "all_pairs_candidate_edges", "candidate_edges", "embedding_matrix",
    "knn_candidate_edges",
]
