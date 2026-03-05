# Node Classes

## Overview

Nodes represent individual data points in the graph. They store data, labels, attributes, and connections to other nodes.

## Base Node Class

### `Node`

Base class for all nodes.

**Key Attributes:**
- `node_id`: Unique identifier
- `split`: Data split (train/val/test)
- `data`: Data object (e.g., ImageFileData)
- `edges`: List of Edge objects
- `label`: Ground truth label

**Key Methods:**
- `get_data()`: Get the data object
- `get_label()`: Get the label
- `get_neighbors()`: Get connected nodes
- `add_edge(edge)`: Add an edge
- `remove_edge(edge)`: Remove an edge

## AttributeNode

Extended node class for nodes with demographic and quality attributes.

**Additional Attributes:**
- `attributes`: Dictionary of attribute name -> value
- `threshold`: Threshold for attribute matching

**Key Methods:**
- `match(other)`: Check if two nodes match based on attributes
- `compute_similarity(other, attr_name, val1, val2)`: Compute similarity between attribute values
- `add_attribute(name, value)`: Add an attribute
- `remove_attribute(name)`: Remove an attribute

**Attribute Types:**
- **Demographic**: `race_*`, `gender_*`, `age_*`
- **Quality Metrics**: `blur_*`, `brightness_*`, `contrast_*`, `compression_*`
- **Symmetry**: `symmetry_*`
- **Embeddings**: Face embeddings as numpy arrays

## Usage Example

```python
from nodes.atrnode import AttributeNode
from data.ImageFileData import ImageFileData

# Create data
data = ImageFileData("path/to/image.jpg")

# Create attributes
attributes = {
    'race_asian': True,
    'gender_female': True,
    'blur_score': 0.85,
    'face_embedding': np.array([...])
}

# Create node
node = AttributeNode(
    node_id="node_001",
    split="train",
    data=data,
    edges=[],
    label=1,  # 1 = deepfake, 0 = real
    attributes=attributes,
    threshold=80  # 80% attribute match threshold
)

# Check node matching
other_node = AttributeNode(...)
if node.match(other_node):
    print("Nodes match based on attributes")
```

## Node Lifecycle

1. **Creation**: Nodes are created by dataloaders from raw data
2. **Connection**: Edges are added during graph construction
3. **Traversal**: Nodes are visited by traversal strategies
4. **Training**: Node data is used for model training
5. **Modification**: Nodes may be removed/restored by managers

## Notes

- Node IDs should be unique and stable across runs
- Attributes are used for graph construction and bias analysis
- Edge connections determine graph topology and traversal paths
