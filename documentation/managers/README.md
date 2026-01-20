# Manager Classes

Manager classes handle graph modifications and state management during training.

## Available Managers

### GraphReductionManager

Manages dynamic graph reduction and restoration strategies during training.

**Key Features:**
- Multiple reduction strategies (Max/Min/Mix-Max I-value, Random)
- Multiple restoration strategies (Random Pool, Targeted, Reversion)
- State tracking for removed nodes and I-values
- Epoch-based history for reversion strategy

**Documentation:** [GraphReductionManager.md](GraphReductionManager.md)

### PerformanceGraphManager

Dynamically rewires the graph based on model performance using I-value predictions.

### NoGraphManager

Static graph wrapper that doesn't modify the graph structure.

## Usage

Managers are typically initialized during training setup and integrated into the training loop:

```python
from managers.GraphReductionManager import GraphReductionManager

reduction_manager = GraphReductionManager(
    reduction_strategy='max_ival',
    reduction_percentage=10.0,
    restoration_strategy='random_pool',
    restoration_percentage=50.0
)
```
