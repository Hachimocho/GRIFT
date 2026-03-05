# ComprehensiveTraversal

## Overview

`ComprehensiveTraversal` visits every node exactly once per epoch in arbitrary order. This is functionally equivalent to standard i.i.d. batch training with random shuffling.

## Class Definition

```python
class ComprehensiveTraversal(Traversal):
    def __init__(self, graph, num_pointers, num_steps=None)
```

## Parameters

- **`graph`**: HyperGraph instance to traverse
- **`num_pointers`**: Number of concurrent pointers (usually 1)
- **`num_steps`**: Maximum number of nodes to visit. If None, visits all nodes.

## Traversal Strategy

1. Shuffle all nodes
2. Visit each node exactly once
3. No graph structure consideration
4. No I-value guidance

## Use Cases

- **Baseline Comparison**: Compare against standard training
- **Complete Coverage**: Ensure all data is seen
- **Simple Training**: No complex traversal logic

## Usage Example

```python
from traversals.ComprehensiveTraversal import ComprehensiveTraversal

# Create traversal
traversal = ComprehensiveTraversal(
    graph=graph,
    num_pointers=1,
    num_steps=None  # Visit all nodes
)

# Use in training
trainer.set_traversal(traversal, "comprehensive")
```

## Key Features

- **Deterministic**: Visits every node once
- **Simple**: No complex logic
- **Baseline**: Standard training equivalent
- **Complete**: Guarantees full data coverage

## Notes

- Ignores graph structure completely
- No bias mitigation
- No I-value guidance
- Useful for baseline experiments
