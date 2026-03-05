# HyperGraph Class

## Overview

`HyperGraph` is the core graph data structure that stores nodes and provides graph operations. It serves as the foundation for all graph-based operations in the framework.

## Class Definition

```python
class HyperGraph:
    def __init__(self, nodes: list)
```

## Key Methods

### Node Management

- **`get_node(index)`**: Get node by index
- **`get_nodes()`**: Get all nodes
- **`add_node(node)`**: Add a node to the graph
- **`remove_node(index)`**: Remove a node by index
- **`set_node(index, node)`**: Replace a node at index
- **`get_random_node()`**: Get a random node

### Graph Operations

- **`k_hop_subgraph(node, k, duplicates=False)`**: Get k-hop subgraph around a node
- **`k_hop_list(node, k, duplicates=False)`**: Get k-hop ordered list of nodes
- **`get_edge_list()`**: Extract list of unique edges as tuples
- **`add_edges_from_list(edge_list)`**: Add edges from a list of node ID pairs

### Properties

- **`nodes`**: List of all nodes in the graph
- **`_node_data_map`**: Dictionary mapping node_id -> node for fast lookup
- **`subclusters`**: Optional mapping of node_id -> subcluster_id

## Usage Example

```python
from graphs.HyperGraph import HyperGraph
from nodes.atrnode import AttributeNode

# Create nodes
nodes = [AttributeNode(...) for _ in range(100)]

# Create graph
graph = HyperGraph(nodes)

# Access nodes
node = graph.get_node(0)
all_nodes = graph.get_nodes()

# Graph operations
subgraph = graph.k_hop_subgraph(node, k=2)
edge_list = graph.get_edge_list()
```

## Internal Structure

The graph maintains:
- A list of nodes for iteration
- A dictionary mapping node IDs to nodes for O(1) lookup
- Optional subcluster assignments for hierarchical organization

## Notes

- Node IDs must be unique within a graph
- Edge operations rely on nodes having `edges` attribute or `get_neighbors()` method
- The graph does not enforce edge consistency - edges are stored on nodes
