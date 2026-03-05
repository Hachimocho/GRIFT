# DQNModel

## Overview

`DQNModel` is a Deep Q-Network for predicting I-values (information values) for nodes. It learns to estimate how informative a sample is for improving model performance.

## Class Definition

```python
class DQNModel(nn.Module):
    def __init__(self, feature_dim, device, embedding_dim=512, compressed_embedding_dim=64)
```

## Parameters

- **`feature_dim`**: Dimension of node feature vectors
- **`device`**: PyTorch device ('cuda' or 'cpu')
- **`embedding_dim`**: Dimension of face embeddings (default: 512)
- **`compressed_embedding_dim`**: Compressed embedding dimension (default: 64)

## Architecture

The DQN consists of:

1. **Embedding Processor**: Compresses face embeddings
   - Input: `embedding_dim` (512)
   - Output: `compressed_embedding_dim` (64)
   - Architecture: Linear(512, 128) -> ReLU -> Linear(128, 64) -> ReLU

2. **Main Network**: Predicts Q-value from features
   - Input: `feature_dim + compressed_embedding_dim`
   - Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Q-value

## Key Methods

### Forward Pass

```python
def forward(self, node_features, node_embedding=None)
```

**Parameters:**
- `node_features`: Tensor of node features [batch_size, feature_dim]
- `node_embedding`: Optional tensor of face embeddings [batch_size, embedding_dim]

**Returns:** Predicted Q-value [batch_size, 1]

### Training

```python
def train_step(self, transitions)
```

**Parameters:**
- `transitions`: List of (state_features, state_embedding, reward) tuples

**Returns:** Loss value (float)

**Training Process:**
1. Unpack batch of transitions
2. Forward pass to get predicted Q-values
3. Compute MSE loss: `loss = MSE(predicted_q, reward)`
4. Backpropagate and update weights

### Checkpoint Management

```python
def save_checkpoint(self, filepath)
def load_checkpoint(self, filepath)
```

Save/load model and optimizer state.

## DQN Parameters

- **Replay Buffer**: Size 10,000 transitions
- **Batch Size**: 32
- **Gamma**: 0.99 (discount factor)
- **Learning Rate**: 0.001 (Adam optimizer)

## I-Value Prediction

I-values represent how informative a sample is:
- **High I-value**: Model is uncertain or incorrect on this sample
- **Low I-value**: Model performs well on this sample

The DQN learns to predict I-values based on:
- Node features (demographics, quality metrics)
- Face embeddings (visual similarity)
- Model performance history

## Usage Example

```python
from models.DQNModel import DQNModel
import torch

# Initialize DQN
dqn = DQNModel(
    feature_dim=50,  # Number of node features
    device='cuda',
    embedding_dim=512,
    compressed_embedding_dim=64
)

# Prepare features
node_features = torch.randn(32, 50).to('cuda')
node_embeddings = torch.randn(32, 512).to('cuda')

# Predict I-values
i_values = dqn(node_features, node_embeddings)

# Training
transitions = [
    (features, embedding, reward)
    for features, embedding, reward in zip(...)
]
loss = dqn.train_step(transitions)
```

## Integration with Trainers

The DQN is managed by `IValueCapability` in `AdaptiveTrainer`:

```python
# Trainer automatically manages DQN
trainer = AdaptiveTrainer(...)
trainer.set_traversal(i_value_traversal, "i-value")

# I-values are accessed via trainer
i_value = trainer.get_i_value(node)
```

## Notes

- Embeddings are optional - model handles None gracefully
- Features and embeddings are automatically moved to the correct device
- Replay buffer stores recent transitions for training
- I-values are cached in traversals for efficiency
