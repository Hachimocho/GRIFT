# NoGraphManager

## Overview

`NoGraphManager` is a static graph wrapper that doesn't modify the graph structure. Use this for static environments where graph modifications are not desired.

## Class Definition

```python
class NoGraphManager(GraphManager):
    def __init__(self, graph)
```

## Parameters

- **`graph`**: HyperGraph instance to wrap

## Key Methods

### `update_graph()`
Dummy update function that does nothing.

## Usage

```python
from managers.NoGraphManager import NoGraphManager

# Wrap graph without modifications
graph_manager = NoGraphManager(graph)

# Graph remains unchanged
graph_manager.update_graph()  # No-op
```

## Use Cases

- Baseline experiments without graph modifications
- Static graph training
- Comparison with dynamic graph managers
- Simple training setups

## Notes

- Graph structure remains unchanged
- No performance overhead
- Useful for control experiments
