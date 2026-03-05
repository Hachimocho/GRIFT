# Edge Class

## Overview

Edges connect nodes in the graph. They store connection information and traversal weights.

## Class Definition

```python
class Edge:
    def __init__(self, node1, node2, x, traversal_weight=1)
```

## Attributes

- **`node1`**: First node in the edge
- **`node2`**: Second node in the edge
- **`x`**: Edge data/metadata
- **`traversal_weight`**: Weight for traversal algorithms (default: 1)

## Key Methods

### Node Access
- **`get_node1()`**: Get first node
- **`get_node2()`**: Get second node
- **`get_nodes()`**: Get both nodes as tuple
- **`set_node1(node)`**: Set first node
- **`set_node2(node)`**: Set second node
- **`set_nodes(node1, node2)`**: Set both nodes

### Data Access
- **`get_data()`**: Get edge data
- **`set_data(x)`**: Set edge data

### Traversal Weight
- **`get_traversal_weight()`**: Get traversal weight
- **`set_traversal_weight(w)`**: Set traversal weight

## Usage Example

```python
from edges.Edge import Edge
from nodes.atrnode import AttributeNode

# Create nodes
node1 = AttributeNode(...)
node2 = AttributeNode(...)

# Create edge
edge = Edge(
    node1=node1,
    node2=node2,
    x={"similarity": 0.95, "type": "intra_demographic"},
    traversal_weight=1.0
)

# Access edge properties
nodes = edge.get_nodes()
weight = edge.get_traversal_weight()
data = edge.get_data()

# Add edge to nodes
node1.add_edge(edge)
node2.add_edge(edge)
```

## Edge Types

Edges can represent different types of connections:

- **Intra-demographic**: Connections within the same demographic group
- **Inter-demographic**: Connections across demographic groups
- **Quality-based**: Connections based on quality metric similarity
- **Embedding-based**: Connections based on face embedding similarity

## Traversal Weights

Traversal weights influence how traversals move through the graph:
- Higher weights = more likely to traverse
- Can be updated dynamically during training
- Used by I-value traversals for weighted selection

## Notes

- Edges are bidirectional (stored on both nodes)
- Edge data can store any metadata (similarity scores, connection types, etc.)
- Traversal weights can be modified by managers for dynamic graph rewiring
