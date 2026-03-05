# Trainer Classes

Trainers orchestrate the training loop, model updates, and evaluation. They coordinate between traversals, models, and managers.

## Available Trainers

### AdaptiveTrainer
Unified trainer supporting multiple traversal strategies with dynamic capability switching.

**Key Features:**
- Supports single-traversal and switch-traversal modes
- Automatic capability management (I-value, bias tracking, etc.)
- Dynamic traversal switching during training
- Comprehensive metrics logging

**Documentation:** [AdaptiveTrainer.md](AdaptiveTrainer.md)

### Trainer (Base Class)
Abstract base class defining the trainer interface.

**Key Features:**
- Basic training loop structure
- Model and optimizer management
- Traversal integration

**Documentation:** [Trainer.md](Trainer.md)

### DeepfakeTrainer
Legacy trainer for deepfake detection tasks.

### DeepfakeAttributeTrainer
Legacy trainer with attribute-aware training.

## Capabilities

Trainers use a capability system for modular functionality:

- **IValueCapability**: I-value estimation and DQN training
- **BiasCapability**: Bias-aware loss functions and metrics
- **VisualizationCapability**: Training visualization and tracking

## Usage Pattern

```python
from trainers.AdaptiveTrainer import AdaptiveTrainer
from managers.GraphReductionManager import GraphReductionManager
from models.CNNModel import CNNModel

# Initialize components
graph_manager = GraphReductionManager(...)
model = CNNModel(...)
trainer = AdaptiveTrainer(
    graphmanager=graph_manager,
    models=[model],
    device='cuda',
    loss_fn=torch.nn.BCEWithLogitsLoss()
)

# Set traversal
trainer.set_traversal(traversal_instance, "i-value")

# Train
for epoch in range(num_epochs):
    trainer.train(epoch)
```

## Training Modes

### Single-Traversal Mode
Use one traversal method throughout training:
```python
trainer.set_traversal(traversal, "i-value")
for epoch in range(num_epochs):
    trainer.train(epoch)
```

### Switch-Traversal Mode
Switch between traversal methods during training:
```python
trainer.set_traversal(initial_traversal, "random")
for epoch in range(num_epochs):
    trainer.train(epoch)
    if epoch == 10:
        trainer.switch_traversal("i-value", num_pointers=1, num_steps=1000)
```
