# Quick Reference Guide

A quick reference for common tasks and patterns in the HyperGraph framework.

## Common Patterns

### Basic Training Setup

```python
from graphs.HyperGraph import HyperGraph
from managers.GraphReductionManager import GraphReductionManager
from trainers.AdaptiveTrainer import AdaptiveTrainer
from models.CNNModel import CNNModel
from traversals.IValueTraversal import IValueTraversal
import torch

# 1. Load graph
graph = dataloader.load()

# 2. Create manager
graph_manager = GraphReductionManager(graph, ...)

# 3. Create model
model = CNNModel(save_path="./checkpoints", model_name="effnetdf", 
                 lr=0.001, amsgrad=True, device='cuda')

# 4. Create trainer
trainer = AdaptiveTrainer(
    graphmanager=graph_manager,
    models=[model],
    device='cuda',
    loss_fn=torch.nn.BCEWithLogitsLoss()
)

# 5. Create and set traversal
traversal = IValueTraversal(graph, num_pointers=1, num_steps=1000, trainer=trainer)
trainer.set_traversal(traversal, "i-value")

# 6. Training loop
for epoch in range(num_epochs):
    trainer.train(epoch)
    # ... evaluation ...
```

### Graph Reduction Setup

```python
from managers.GraphReductionManager import GraphReductionManager

# Create reduction manager
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
    if epoch > 0:  # Skip first epoch
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

### Traversal Switching

```python
# Start with random traversal
trainer.set_traversal(random_traversal, "random")

# Switch to I-value traversal mid-training
for epoch in range(num_epochs):
    trainer.train(epoch)
    
    if epoch == 10:
        trainer.switch_traversal("i-value", num_pointers=1, num_steps=1000)
```

### I-Value Access

```python
# Get I-value for a node
i_value = trainer.get_i_value(node)

# Use in custom logic
if i_value > threshold:
    # Process high-value node
    pass
```

### Model Evaluation

```python
def evaluate_model(model, nodes, loss_fn, device='cuda'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for node in nodes:
            data = node.get_data().load_data()
            label = node.get_label()
            
            # Transform and predict
            input_tensor = model.transform(data).unsqueeze(0).to(device)
            output = model(input_tensor)
            
            # Compute loss and accuracy
            label_tensor = torch.tensor([[label]], dtype=torch.float32).to(device)
            loss = loss_fn(output, label_tensor)
            total_loss += loss.item()
            
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
```

## Common Configurations

### I-Value Traversal

```python
traversal = IValueTraversal(
    graph=graph,
    num_pointers=1,
    num_steps=1000,
    trainer=trainer,
    return_delay=10,
    warp_chance=0.005,
    predictor_update_period=50
)
```

### Graph Reduction Manager

```python
manager = GraphReductionManager(
    reduction_strategy='max_ival',  # or 'min_ival', 'mix_max_ival', 'random'
    reduction_percentage=10.0,
    reduction_interval='end_of_epoch',  # or 'every_n_steps'
    restoration_strategy='random_pool',  # or 'targeted', 'reversion'
    restoration_percentage=50.0
)
```

### CNN Model

```python
model = CNNModel(
    save_path="./checkpoints",
    model_name="effnetdf",  # or 'resnestdf', 'mesonetdf', etc.
    lr=0.001,
    amsgrad=True,
    device='cuda'
)
```

## File Locations

- **Main Training Script**: `test_hierarchical.py`
- **Graph Class**: `graphs/HyperGraph.py`
- **Node Class**: `nodes/atrnode.py`
- **Edge Class**: `edges/Edge.py`
- **Trainer**: `trainers/AdaptiveTrainer.py`
- **Traversals**: `traversals/`
- **Models**: `models/`
- **Dataloaders**: `dataloaders/`
- **Managers**: `managers/`

## Common Issues

### I-Value Not Available
- Ensure trainer has I-value capability enabled
- Check that DQN model is initialized
- Verify traversal has trainer reference

### Graph Reduction Not Working
- Check reduction strategy is not "none"
- Verify reduction percentage > 0
- Ensure trainer has I-value access for non-random strategies

### Traversal Getting Stuck
- Increase `warp_chance` for random warps
- Use ClusterHop variant for clustered graphs
- Check graph connectivity

### Model Not Training
- Verify loss function is provided
- Check device placement (CPU vs GPU)
- Ensure model is in training mode

## Debugging Tips

1. **Enable Logging**: Set `silent_mode=False` in dataloaders
2. **Check Graph Size**: `len(graph)` to verify nodes loaded
3. **Inspect I-Values**: `trainer.get_i_value(node)` to check predictions
4. **Monitor Metrics**: Use `trainer.log_metrics()` to track training
5. **Visualize Graph**: Use `utils.visualize.visualize_graph()` for structure
