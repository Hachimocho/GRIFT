# HyperGraph Documentation

Welcome to the HyperGraph documentation. This documentation is organized to mirror the codebase structure for easy navigation.

## Overview

HyperGraph is a research framework for graph-based training of deepfake detection models with bias-aware capabilities. The framework uses I-value estimation via Deep Q-Networks (DQN) to identify and correct demographic bias in real-time.

## Documentation Structure

### Core Documentation

- **[Architecture Overview](architecture.md)** - System architecture and design principles
- **[Core Components](core/README.md)** - HyperGraph, Nodes, and Edges
  - [HyperGraph](core/HyperGraph.md) - Core graph data structure
  - [Nodes](core/Nodes.md) - Node classes and attributes
  - [Edges](core/Edges.md) - Edge classes and connections

### Component Documentation

#### Managers
- **[Managers Overview](managers/README.md)** - Graph management classes
  - [GraphReductionManager](managers/GraphReductionManager.md) - Dynamic graph reduction and restoration
  - [PerformanceGraphManager](managers/PerformanceGraphManager.md) - Performance-based graph rewiring
  - [NoGraphManager](managers/NoGraphManager.md) - Static graph wrapper

#### Trainers
- **[Trainers Overview](trainers/README.md)** - Training orchestration classes
  - [AdaptiveTrainer](trainers/AdaptiveTrainer.md) - Unified trainer with capability management
  - [Trainer](trainers/Trainer.md) - Base trainer class

#### Traversals
- **[Traversals Overview](traversals/README.md)** - Graph traversal strategies
  - ComprehensiveTraversal - Visit every node once
  - RandomTraversal - Random walk through graph
  - IValueTraversal - I-value guided exploration
  - IValueTraversalClusterHop - I-value with demographic hops

#### Models
- **[Models Overview](models/README.md)** - Neural network architectures
  - [CNNModel](models/CNNModel.md) - CNN-based deepfake detection
  - [DQNModel](models/DQNModel.md) - Deep Q-Network for I-value prediction

#### Dataloaders
- **[Dataloaders Overview](dataloaders/README.md)** - Data loading and graph construction
  - HierarchicalDeepfakeDataloader - Hierarchical graph construction
  - ClusteredDeepfakeDataloader - Clustered graph construction
  - UnclusteredDeepfakeDataloader - Simple graph construction

#### Utilities
- **[Utilities Overview](utils/README.md)** - Helper functions and tools
  - DQNIValuePredictor - I-value prediction utilities
  - Visualization utilities
  - Attribute utilities

#### Evaluation
- **[Evaluation Overview](evaluation/README.md)** - Model evaluation tools
  - DQNEvaluator - DQN model evaluation
  - Bias metrics
  - Performance analysis

### Feature Documentation

- **[Graph Reduction & Restoration](features/graph_reduction_restoration.md)** - Dynamic graph modification strategies
- **[Web UI](web_ui/README.md)** - Web-based configuration and results interface
  - [Configuration](web_ui/configuration.md) - Experiment configuration
  - [Graph Reduction/Restoration UI](web_ui/graph_reduction_restoration.md) - UI-specific reduction/restoration docs

## Quick Start Guide

### For New Developers

1. **Start with Architecture**: Read [Architecture Overview](architecture.md) to understand the system design
2. **Understand Core Components**: Review [Core Components](core/README.md) to learn about graphs, nodes, and edges
3. **Learn Training Flow**: Read [AdaptiveTrainer](trainers/AdaptiveTrainer.md) to understand the training process
4. **Explore Traversals**: Check [Traversals Overview](traversals/README.md) to see how data is sampled
5. **Review Examples**: Look at `test_hierarchical.py` for a complete training example

### For Users

1. **Configuration**: Start with [Web UI Configuration](web_ui/configuration.md) to set up experiments
2. **Graph Construction**: Review [Dataloaders Overview](dataloaders/README.md) to understand graph building
3. **Training**: See [AdaptiveTrainer](trainers/AdaptiveTrainer.md) for training setup
4. **Results**: Check [Evaluation Overview](evaluation/README.md) for understanding metrics

## Key Concepts

### I-Values (Information Values)
I-values estimate how informative a sample is for improving model performance. They are learned via DQN and guide traversal toward high-uncertainty samples and underrepresented groups.

**Learn More**: [DQNModel](models/DQNModel.md), [IValueTraversal](traversals/README.md)

### Graph Reduction & Restoration
Dynamic graph modification strategies that temporarily remove nodes to focus training, then restore them based on validation performance.

**Learn More**: [GraphReductionManager](managers/GraphReductionManager.md), [Graph Reduction & Restoration](features/graph_reduction_restoration.md)

### Hierarchical Graph Construction
Graphs are organized hierarchically by demographic attributes (race-gender combinations) with quality-filtered connections.

**Learn More**: [HierarchicalDeepfakeDataloader](dataloaders/README.md)

### Traversal Strategies
Different methods for moving through the graph during training, from random walks to I-value guided exploration.

**Learn More**: [Traversals Overview](traversals/README.md)

## API Reference

### Core Classes

- **HyperGraph**: `graphs/HyperGraph.py`
- **AttributeNode**: `nodes/atrnode.py`
- **Edge**: `edges/Edge.py`

### Managers

- **GraphReductionManager**: `managers/GraphReductionManager.py`
- **PerformanceGraphManager**: `managers/PerformanceGraphManager.py`
- **NoGraphManager**: `managers/NoGraphManager.py`

### Trainers

- **AdaptiveTrainer**: `trainers/AdaptiveTrainer.py`
- **Trainer**: `trainers/Trainer.py`

### Traversals

- **IValueTraversal**: `traversals/IValueTraversal.py`
- **ComprehensiveTraversal**: `traversals/ComprehensiveTraversal.py`
- **RandomTraversal**: `traversals/RandomTraversal.py`

### Models

- **CNNModel**: `models/CNNModel.py`
- **DQNModel**: `models/DQNModel.py`

### Dataloaders

- **HierarchicalDeepfakeDataloader**: `dataloaders/HierarchicalDeepfakeDataloader.py`

## Extension Points

The framework is designed for extensibility:

- **Custom Traversals**: Inherit from `Traversal` base class
- **Custom Managers**: Inherit from `GraphManager` base class
- **Custom Trainers**: Inherit from `Trainer` or `AdaptiveTrainer`
- **Custom Models**: Implement `Model` interface
- **Custom Dataloaders**: Inherit from `Dataloader` base class

## Getting Help

- **Code Examples**: See `test_hierarchical.py` for a complete training example
- **Component Details**: Check individual component documentation pages
- **Architecture Questions**: Review [Architecture Overview](architecture.md)

## Contributing

When adding new components:

1. Document the component in the appropriate section
2. Include usage examples
3. Document parameters and return values
4. Update this README with links to new documentation
