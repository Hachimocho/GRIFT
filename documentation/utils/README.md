# Utility Modules

Utility modules provide helper functions and tools for various tasks in the HyperGraph framework.

## Available Utilities

### DQNIValuePredictor
DQN-based I-value prediction utility. Wraps DQN model for easy I-value computation.

**Key Features:**
- Node feature extraction
- I-value prediction
- Batch processing support

### RandomIValuePredictor
Random I-value predictor for baseline comparisons.

### attribute_utils
Utilities for working with node attributes:
- Attribute extraction
- Attribute normalization
- Demographic grouping

### visualize
Graph and training visualization utilities:
- Graph structure visualization
- I-value plots
- Bias metric visualization

### parallel_processor
Parallel processing utilities for data loading and processing.

### profiler
Performance profiling utilities for identifying bottlenecks.

### WandbArtifactUtils
Weights & Biases integration utilities for experiment tracking.

### SSIMGeneration
Structural Similarity Index (SSIM) computation for image quality metrics.

### import_utils
Dynamic import utilities for loading modules and classes.

## Usage Examples

### I-Value Prediction

```python
from utils.DQNIValuePredictor import DQNIValuePredictor
from models.DQNModel import DQNModel

# Initialize predictor
dqn = DQNModel(...)
predictor = DQNIValuePredictor(dqn)

# Predict I-value
i_value = predictor.predict(node)
```

### Visualization

```python
from utils.visualize import visualize_graph

# Visualize graph structure
visualize_graph(graph, output_path="graph.png")
```

### Attribute Utilities

```python
from utils.attribute_utils import extract_demographics

# Extract demographic attributes
demographics = extract_demographics(node)
```

## Notes

- Utilities are modular and can be used independently
- Most utilities handle errors gracefully
- Performance utilities are optional and can be disabled
