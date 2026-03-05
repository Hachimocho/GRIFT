# RandomTraversal

## Overview

`RandomTraversal` performs a Markov chain random walk, selecting uniformly from neighbors at each step. It respects graph structure without intelligent exploration.

## Class Definition

```python
class RandomTraversal(Traversal):
    def __init__(self, graph, num_pointers, num_steps)
```

## Parameters

- **`graph`**: HyperGraph instance to traverse
- **`num_pointers`**: Number of concurrent pointers (usually 1)
- **`num_steps`**: Number of steps to take

## Traversal Strategy

At each step:
1. Get current node's neighbors
2. Select uniformly at random from neighbors
3. Move pointer to selected neighbor
4. Small teleportation probability prevents getting trapped

## Use Cases

- **Baseline**: Graph-aware baseline without I-value guidance
- **Exploration**: Random exploration of graph structure
- **Comparison**: Compare against I-value guided traversal

## Usage Example

```python
from traversals.RandomTraversal import RandomTraversal

# Create traversal
traversal = RandomTraversal(
    graph=graph,
    num_pointers=1,
    num_steps=1000
)

# Use in training
trainer.set_traversal(traversal, "random")
```

## Key Features

- **Graph-Aware**: Respects graph structure
- **Random**: Uniform neighbor selection
- **Simple**: No complex logic
- **Baseline**: Graph-aware baseline

## Notes

- Stays within well-connected regions
- No bias mitigation
- No I-value guidance
- Natural clustering behavior
