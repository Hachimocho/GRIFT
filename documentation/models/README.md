# Model Classes

Models define the neural network architectures used for deepfake detection and I-value prediction.

## Available Models

### CNNModel
CNN-based deepfake detection classifier. Supports multiple architectures via detector modules.

**Supported Architectures:**
- EfficientNetDF
- ResNeStDF
- MesoNetDF
- SqueezeNetDF
- VisionTransformerDF
- SwinTransformerDF

**Documentation:** [CNNModel.md](CNNModel.md)

### DQNModel
Deep Q-Network for I-value prediction. Predicts information values for nodes based on features and embeddings.

**Documentation:** [DQNModel.md](DQNModel.md)

### EnhancedDQNModels
Extended DQN variants with residual connections and other improvements.

## Base Model Class

All models inherit from `Model` base class, which provides:
- Save/load functionality
- Checkpoint management
- Common interface

## Model Architecture

### Detection Models
Located in `models/detectors/`:
- Base detector interface
- Architecture-specific implementations
- Pretrained weight loading

### Loss Functions
Located in `models/loss/`:
- Standard loss functions (BCE, CrossEntropy)
- Bias-aware loss functions
- Custom loss implementations

### Metrics
Located in `models/metrics/`:
- Accuracy, F1, AUROC
- Bias metrics
- Demographic performance tracking

## Usage Pattern

```python
from models.CNNModel import CNNModel

# Initialize model
model = CNNModel(
    save_path="./checkpoints",
    model_name="effnetdf",
    lr=0.001,
    amsgrad=True,
    device='cuda'
)

# Forward pass
output = model(input_tensor)

# Training mode
model.train()
loss = model.loss(output, labels)
loss.backward()
model.optim.step()

# Evaluation mode
model.eval()
with torch.no_grad():
    predictions = model(input_tensor)
```

## Model Selection

Models are selected based on:
- Task requirements (detection vs I-value prediction)
- Architecture preferences
- Performance requirements
- Hardware constraints

## Notes

- Models automatically handle device placement (CPU/GPU)
- Transform pipelines are built into CNN models
- DQN models require feature extraction from nodes
- Checkpoints are saved automatically during training
