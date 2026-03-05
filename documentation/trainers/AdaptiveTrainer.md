# AdaptiveTrainer

## Overview

`AdaptiveTrainer` is the unified trainer architecture that supports multiple traversal strategies with dynamic capability switching. It uses composition and strategy patterns to enable flexible training configurations.

## Class Definition

```python
class AdaptiveTrainer(Trainer):
    def __init__(self, graphmanager, models, device, attribute_metadata=None, 
                 loss_fn=None, attribute_weights=None, bias_group_weights=None, **kwargs)
```

## Parameters

- **`graphmanager`**: GraphManager instance wrapping the training graph
- **`models`**: List of model instances (e.g., CNNModel)
- **`device`**: PyTorch device ('cuda' or 'cpu')
- **`attribute_metadata`**: Optional metadata about node attributes
- **`loss_fn`**: Loss function (required)
- **`attribute_weights`**: Optional weights for different attributes in bias loss
- **`bias_group_weights`**: Optional weights for demographic groups
- **`dqn_model_type`**: DQN model variant ('basic', 'residual', etc.)

## Key Methods

### Traversal Management

#### `set_traversal(traversal_instance, traversal_type)`
Set the traversal method to use for training.

**Parameters:**
- `traversal_instance`: Traversal object (e.g., IValueTraversal)
- `traversal_type`: String identifier ("i-value", "random", "comprehensive")

**Example:**
```python
trainer.set_traversal(i_value_traversal, "i-value")
```

#### `switch_traversal(new_traversal_type, **traversal_kwargs)`
Switch to a different traversal method during training.

**Parameters:**
- `new_traversal_type`: New traversal type string
- `**traversal_kwargs`: Arguments for creating the new traversal

**Example:**
```python
trainer.switch_traversal("random", num_pointers=1, num_steps=500)
```

#### `_create_traversal(traversal_type, **kwargs)`
Factory method that automatically selects the correct traversal variant.

Automatically handles:
- I-value traversal variant selection based on graph type
- Subclustering detection
- Cluster-hop variant selection

### Training

#### `train(epoch=None)`
Execute training using the current traversal method.

**Returns:** Training metrics dictionary

**Example:**
```python
metrics = trainer.train(epoch=5)
```

### I-Value Access

#### `get_i_value(node, model_idx=0)`
Get I-value prediction for a node.

**Parameters:**
- `node`: Node object
- `model_idx`: Model index (for multi-model setups)

**Returns:** I-value (float)

### Capability Management

#### `save_capability_checkpoints(base_path)`
Save checkpoints for all enabled capabilities.

#### `load_capability_checkpoints(base_path)`
Load checkpoints for all enabled capabilities.

### Logging

#### `log_metrics(metrics)`
Log training metrics to file and console.

**Parameters:**
- `metrics`: Dictionary of metric name -> value

#### `get_current_traversal_info()`
Get information about current traversal configuration.

**Returns:** Dictionary with traversal type, class, and enabled capabilities

## Capabilities

The trainer uses a `CapabilityManager` to provide modular functionality:

### IValueCapability
- I-value estimation via DQN
- DQN training and updates
- I-value prediction for nodes

### BiasCapability
- Bias-aware loss functions
- Demographic performance tracking
- Attribute-weighted loss computation

### VisualizationCapability
- Training progress visualization
- I-value plots
- Bias metric tracking

## Usage Example

```python
from trainers.AdaptiveTrainer import AdaptiveTrainer
from managers.GraphReductionManager import GraphReductionManager
from models.CNNModel import CNNModel
from traversals.IValueTraversal import IValueTraversal
import torch

# Initialize components
graph = HyperGraph(nodes)
graph_manager = GraphReductionManager(graph, ...)
model = CNNModel(...)
trainer = AdaptiveTrainer(
    graphmanager=graph_manager,
    models=[model],
    device='cuda',
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    attribute_metadata=attribute_metadata,
    bias_loss_weight=0.1
)

# Create and set traversal
traversal = IValueTraversal(graph, num_pointers=1, num_steps=1000, trainer=trainer)
trainer.set_traversal(traversal, "i-value")

# Training loop
for epoch in range(num_epochs):
    # Train
    train_metrics = trainer.train(epoch)
    
    # Evaluate
    val_metrics = evaluate_model(...)
    
    # Log
    trainer.log_metrics({**train_metrics, **val_metrics})
    
    # Optional: Switch traversal
    if epoch == 10:
        trainer.switch_traversal("random", num_pointers=1, num_steps=500)
```

## Traversal Variant Selection

The trainer automatically selects the correct I-value traversal variant:

- **`IValueTraversal`**: Standard I-value traversal
- **`IValueTraversalClusterHop`**: I-value with periodic cluster hops
- **`IValueTraversalSubcluster`**: I-value with subcluster awareness
- **`IValueTraversalClusterHopSubcluster`**: Combined cluster-hop and subcluster

Selection is based on:
- Graph type (clustered vs unclustered)
- Subclustering availability
- Traversal configuration

## State Management

- Traversal state can be transferred when switching traversals
- Capability checkpoints can be saved/loaded
- Metrics are logged to JSON files

## Notes

- Loss function is required (no default)
- Capabilities are automatically configured based on traversal type
- I-value traversals require IValueCapability to be enabled
- Bias-aware training requires BiasCapability and attribute metadata
