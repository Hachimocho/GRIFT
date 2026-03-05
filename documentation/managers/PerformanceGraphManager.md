# PerformanceGraphManager

## Overview

`PerformanceGraphManager` dynamically rewires the graph based on model performance using I-value predictions. It identifies weak and strong nodes and adjusts graph structure to improve training.

## Class Definition

```python
class PerformanceGraphManager(GraphManager):
    def __init__(self, graph, rewire_threshold=0.8, edge_removal_threshold=0.2, 
                 max_edges_per_node=10, update_interval=200)
```

## Parameters

- **`graph`**: HyperGraph instance to manage
- **`rewire_threshold`**: I-value threshold above which nodes are considered weak (default: 0.8)
- **`edge_removal_threshold`**: I-value threshold below which nodes are considered strong (default: 0.2)
- **`max_edges_per_node`**: Maximum number of edges per node (default: 10)
- **`update_interval`**: Number of steps between graph updates (default: 200)

## Key Methods

### `set_i_value_predictor(predictor)`
Set the DQN model for I-value prediction.

**Parameters:**
- `predictor`: DQNModel instance

### `track_performance(node, i_value)`
Track I-value for a node (called during training).

**Parameters:**
- `node`: Node object
- `i_value`: Predicted I-value (float)

### `identify_weak_nodes() -> List`
Identify nodes with consistently high I-values (poor performance).

**Returns:** List of weak node objects

### `identify_strong_nodes() -> List`
Identify nodes with consistently low I-values (good performance).

**Returns:** List of strong node objects

### `update_graph()`
Update graph structure based on tracked performance.

**Process:**
1. Identify weak and strong nodes
2. Add edges from weak nodes to strong nodes
3. Remove excess edges from strong nodes
4. Maintain graph connectivity

## Rewiring Strategy

### For Weak Nodes
- Add edges to strong nodes to improve connectivity
- Helps weak nodes learn from well-performing samples
- Increases exposure to informative neighbors

### For Strong Nodes
- Remove excess edges to prevent over-connection
- Maintains graph sparsity
- Prevents strong nodes from dominating traversal

## Usage Example

```python
from managers.PerformanceGraphManager import PerformanceGraphManager
from models.DQNModel import DQNModel

# Initialize manager
graph_manager = PerformanceGraphManager(
    graph=graph,
    rewire_threshold=0.8,
    edge_removal_threshold=0.2,
    max_edges_per_node=10,
    update_interval=200
)

# Set I-value predictor
dqn = DQNModel(...)
graph_manager.set_i_value_predictor(dqn)

# In training loop
for step in range(num_steps):
    # Track performance
    for node in batch_nodes:
        i_value = trainer.get_i_value(node)
        graph_manager.track_performance(node, i_value)
    
    # Update graph periodically
    if step % graph_manager.update_interval == 0:
        graph_manager.update_graph()
```

## Performance Tracking

The manager maintains:
- **`node_performance`**: Dictionary mapping node -> list of I-values
- **History**: Last 100 I-values per node
- **Averages**: Computed on-demand for node classification

## Notes

- Requires I-value predictor (DQN model)
- Graph updates are periodic to avoid overhead
- Weak/strong classification is based on average I-value
- Graph connectivity is maintained during rewiring
