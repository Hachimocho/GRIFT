# GraphReductionManager API Reference

## Overview

`GraphReductionManager` manages dynamic graph reduction and restoration strategies during training. It supports multiple reduction strategies for removing nodes from the training graph and restoration strategies for adding nodes back when validation performance drops.

## Class Definition

```python
class GraphReductionManager:
    def __init__(self, 
                 reduction_strategy: str = "none",
                 reduction_percentage: float = 0.0,
                 reduction_top_percentage: float = 0.0,
                 reduction_bottom_percentage: float = 0.0,
                 reduction_interval: str = "end_of_epoch",
                 reduction_interval_steps: int = 100,
                 restoration_strategy: str = "none",
                 restoration_percentage: float = 50.0,
                 restoration_trigger_threshold: float = 0.0)
```

## Parameters

### Reduction Parameters

- **reduction_strategy** (`str`): Strategy for node removal
  - `"none"`: No reduction
  - `"max_ival"`: Remove top X% nodes by I-value
  - `"min_ival"`: Remove bottom Y% nodes by I-value
  - `"mix_max_ival"`: Remove top X% + bottom Y% (mutually exclusive)
  - `"random"`: Remove Z% randomly (baseline)

- **reduction_percentage** (`float`): Percentage of nodes to remove (0-100)

- **reduction_top_percentage** (`float`): Top percentage for mix_max strategy (0-100)

- **reduction_bottom_percentage** (`float`): Bottom percentage for mix_max strategy (0-100)

- **reduction_interval** (`str`): When to perform reduction
  - `"end_of_epoch"`: At the end of each epoch
  - `"every_n_steps"`: Every N training steps

- **reduction_interval_steps** (`int`): Number of steps between reductions (if interval is "every_n_steps")

### Restoration Parameters

- **restoration_strategy** (`str`): Strategy for node restoration
  - `"none"`: No restoration
  - `"random_pool"`: Restore random selection from removed nodes pool
  - `"targeted"`: Restore nodes with I-values closest to average
  - `"reversion"`: Restore previous epoch's removed nodes

- **restoration_percentage** (`float`): Percentage of removed nodes to restore (0-100)

- **restoration_trigger_threshold** (`float`): Minimum validation accuracy drop to trigger restoration (default: 0.0 = any drop)

## Key Methods

### `reduce_graph(graph, trainer, epoch=0, step=0) -> Tuple[List, Dict]`

Execute graph reduction based on configured strategy.

**Parameters:**
- `graph`: HyperGraph instance to reduce
- `trainer`: Trainer instance (for I-value access)
- `epoch`: Current epoch number
- `step`: Current step number

**Returns:**
- Tuple of (removed_nodes_list, removal_stats_dict)

**Example:**
```python
removed_nodes, stats = reduction_manager.reduce_graph(
    train_manager.graph, trainer, epoch=5, step=500
)
```

### `restore_nodes(graph, trainer, current_val_acc, best_val_acc) -> Tuple[List, Dict]`

Execute node restoration based on configured strategy.

**Parameters:**
- `graph`: HyperGraph instance to restore nodes to
- `trainer`: Trainer instance (for I-value access if needed)
- `current_val_acc`: Current validation accuracy
- `best_val_acc`: Best validation accuracy seen so far

**Returns:**
- Tuple of (restored_nodes_list, restoration_stats_dict)

**Example:**
```python
restored_nodes, stats = reduction_manager.restore_nodes(
    train_manager.graph, trainer, current_val_acc=0.85, best_val_acc=0.90
)
```

### `check_restoration_trigger(current_val_acc, best_val_acc) -> bool`

Check if restoration should be triggered based on validation performance drop.

**Parameters:**
- `current_val_acc`: Current validation accuracy
- `best_val_acc`: Best validation accuracy seen so far

**Returns:**
- `True` if restoration should be triggered

### `should_reduce(current_step, epoch) -> bool`

Check if reduction should be performed at this point (for "every_n_steps" interval).

**Parameters:**
- `current_step`: Current training step
- `epoch`: Current epoch

**Returns:**
- `True` if reduction should be performed

### `store_epoch_state(epoch, removed_nodes)`

Store epoch state for reversion strategy.

**Parameters:**
- `epoch`: Epoch number
- `removed_nodes`: List of nodes removed in this epoch

### `get_removed_nodes() -> List`

Get current pool of removed nodes.

**Returns:**
- List of removed node objects

### `get_stats() -> Dict`

Get reduction/restoration statistics.

**Returns:**
- Dictionary with reduction stats, removed nodes count, and epochs with reductions

## State Management

The manager maintains several internal data structures:

- **removed_nodes_pool**: List of removed node objects
- **removed_nodes_ivalues**: Dictionary mapping node_id -> i_value
- **epoch_removal_history**: Dictionary mapping epoch -> list of removed nodes

## Error Handling

- Non-random reduction strategies require I-value capability in the trainer
- If I-values are unavailable but needed, a warning is logged and reduction is skipped
- Graph integrity is maintained (at least one node remains after reduction)

## Usage Example

```python
from managers.GraphReductionManager import GraphReductionManager

# Initialize manager
reduction_manager = GraphReductionManager(
    reduction_strategy='max_ival',
    reduction_percentage=10.0,
    reduction_interval='end_of_epoch',
    restoration_strategy='random_pool',
    restoration_percentage=50.0,
    restoration_trigger_threshold=0.01
)

# In training loop
for epoch in range(num_epochs):
    # ... training ...
    
    # Reduce at end of epoch
    if reduction_manager.reduction_interval == 'end_of_epoch':
        removed_nodes, stats = reduction_manager.reduce_graph(
            graph, trainer, epoch, epoch * steps_per_epoch
        )
    
    # ... validation ...
    
    # Check for restoration
    if reduction_manager.check_restoration_trigger(current_val_acc, best_val_acc):
        restored_nodes, stats = reduction_manager.restore_nodes(
            graph, trainer, current_val_acc, best_val_acc
        )
```
