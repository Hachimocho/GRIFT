# Traversal Strategies

Traversals define how pointers move through the graph during training. They determine which nodes are visited and in what order.

## Available Traversals

### ComprehensiveTraversal
Visits every node exactly once per epoch in arbitrary order. Functionally equivalent to standard i.i.d. batch training.

**Use Case:** Baseline comparison, ensures all data is seen

**Documentation:** [ComprehensiveTraversal.md](ComprehensiveTraversal.md)

### RandomTraversal
Performs a Markov chain random walk, selecting uniformly from neighbors at each step.

**Use Case:** Baseline that respects graph structure without intelligent exploration

**Documentation:** [RandomTraversal.md](RandomTraversal.md)

### IValueTraversal
Uses learned I-values to guide exploration toward high-utility samples. Greedily selects the highest I-value neighbor.

**Use Case:** Bias-aware training, focusing on informative samples

**Documentation:** [IValueTraversal.md](IValueTraversal.md)

### IValueTraversalClusterHop
Combines I-value guidance with periodic demographic targeting. Every T steps, hops to the highest I-value node in the worst-performing demographic subgroup.

**Use Case:** Clustered graphs where pure I-value traversal may get trapped

**Documentation:** [IValueTraversalClusterHop.md](IValueTraversalClusterHop.md)

### RandomWarpTraversal
Random traversal with occasional teleportation to random nodes.

### RandomNoReturnTraversal
Random traversal that avoids revisiting nodes.

## Base Traversal Class

All traversals inherit from `Traversal` base class, which provides:

- Iterator protocol (`__iter__`, `__next__`)
- Pointer management
- State transfer for switching traversals
- Trainer integration

## Traversal Selection

The `AdaptiveTrainer` automatically selects the correct traversal variant:

```python
# Automatically selects IValueTraversalClusterHop for clustered graphs
trainer.set_traversal(traversal, "i-value")
```

Selection factors:
- Graph type (clustered vs unclustered)
- Subclustering availability
- Traversal configuration

## Usage Pattern

```python
from traversals.IValueTraversal import IValueTraversal
from trainers.AdaptiveTrainer import AdaptiveTrainer

# Create traversal
traversal = IValueTraversal(
    graph=graph,
    num_pointers=1,
    num_steps=1000,
    trainer=trainer
)

# Use in trainer
trainer.set_traversal(traversal, "i-value")

# Traverse
for batch_nodes in traversal:
    # Process batch
    pass
```

## Traversal Parameters

Common parameters across traversals:

- **`graph`**: HyperGraph instance to traverse
- **`num_pointers`**: Number of concurrent pointers (usually 1)
- **`num_steps`**: Number of steps to take
- **`trainer`**: Trainer instance (for I-value access)

## State Transfer

Traversals support state transfer for seamless switching:

```python
# Get state from current traversal
state = traversal1.get_state()

# Transfer to new traversal
traversal2.set_state(state)
```

State includes:
- Pointer positions
- Step count
- Visited nodes
- Traversal-specific metadata
