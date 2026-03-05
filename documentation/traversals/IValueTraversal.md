# IValueTraversal

## Overview

`IValueTraversal` uses learned I-values to guide exploration toward high-utility samples. It greedily selects the highest I-value neighbor at each step.

## Class Definition

```python
class IValueTraversal(Traversal):
    def __init__(self, graph, num_pointers, num_steps, trainer=None, 
                 return_delay=10, warp_chance=0.005, predictor_update_period=50)
```

## Parameters

- **`graph`**: HyperGraph instance to traverse
- **`num_pointers`**: Number of concurrent pointers (usually 1)
- **`num_steps`**: Number of steps to take
- **`trainer`**: Trainer instance (for I-value access)
- **`return_delay`**: Steps before allowing return to visited nodes (default: 10)
- **`warp_chance`**: Probability of teleporting to random node (default: 0.005)
- **`predictor_update_period`**: Steps between DQN updates (default: 50)

## Traversal Strategy

At each step:
1. Get current node's neighbors
2. Compute I-values for all neighbors (via trainer)
3. Select neighbor with highest I-value
4. Move pointer to selected neighbor
5. Optionally warp to random node (based on `warp_chance`)

## I-Value Computation

I-values are obtained via:
1. Trainer's DQN model (if available)
2. Cached I-values (if previously computed)
3. Pessimistic default (if neither available)

## Usage Example

```python
from traversals.IValueTraversal import IValueTraversal
from trainers.AdaptiveTrainer import AdaptiveTrainer

# Create traversal
traversal = IValueTraversal(
    graph=graph,
    num_pointers=1,
    num_steps=1000,
    trainer=trainer,
    return_delay=10,
    warp_chance=0.005
)

# Use in training
trainer.set_traversal(traversal, "i-value")

# Traverse
for batch_nodes in traversal:
    # Process batch
    pass
```

## Key Features

- **Greedy Selection**: Always selects highest I-value neighbor
- **Bias Mitigation**: Naturally focuses on underrepresented groups (higher uncertainty)
- **Exploration**: Warp chance prevents getting stuck
- **Efficiency**: Caches I-values for performance

## Notes

- Requires trainer with I-value capability
- I-values are learned during training
- Works best with clustered graphs
- May get trapped in single cluster (use ClusterHop variant for clustered graphs)
