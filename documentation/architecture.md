# HyperGraph Architecture

## Overview

HyperGraph is a research framework for graph-based training of deepfake detection models with bias-aware capabilities. The framework uses I-value estimation via Deep Q-Networks (DQN) to identify and correct demographic bias in real-time.

## Core Design Principles

1. **Graph-Based Learning**: Data points become nodes in a graph structure, relationships become edges, and learning becomes traversal through meaningful neighborhoods.
2. **Hierarchical Structure**: Graphs are organized hierarchically by demographic attributes (race-gender combinations) with quality-filtered connections.
3. **I-Value Guided Exploration**: Uses DQN models to estimate information values (I-values) that guide traversal toward informative samples.
4. **Adaptive Training**: Supports dynamic switching between traversal strategies and graph modification strategies during training.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Dataloader  │───▶│  HyperGraph  │───▶│   Manager    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Nodes     │    │    Edges     │    │   Traversal  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                               │
│         │                    │                    │          │
│         └────────────────────┼────────────────────┘          │
│                              │                                │
│                              ▼                                │
│                      ┌──────────────┐                         │
│                      │   Trainer    │                         │
│                      └──────────────┘                         │
│                              │                                │
│                              ▼                                │
│                      ┌──────────────┐                         │
│                      │    Model     │                         │
│                      │  (CNN/DQN)   │                         │
│                      └──────────────┘                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Data Layer

- **Dataloaders**: Load and preprocess datasets, construct graph structures
- **Datasets**: Interface for loading raw data (images, labels, attributes)
- **Nodes**: Represent individual data points with attributes and metadata
- **Edges**: Connect nodes based on similarity, demographics, or quality metrics

### 2. Graph Layer

- **HyperGraph**: Core graph data structure storing nodes and edges
- **Graph Managers**: Modify graph structure during training (reduction, restoration, rewiring)

### 3. Traversal Layer

- **Traversals**: Define how pointers move through the graph
- **Strategies**: Random, Comprehensive, I-value guided, Cluster-hop

### 4. Training Layer

- **Trainers**: Orchestrate training loops, model updates, evaluation
- **AdaptiveTrainer**: Unified trainer supporting multiple traversal strategies
- **Capabilities**: Modular components for I-value estimation, bias tracking, etc.

### 5. Model Layer

- **CNN Models**: Deepfake detection classifiers
- **DQN Models**: I-value prediction networks
- **Loss Functions**: Standard and bias-aware loss functions
- **Metrics**: Accuracy, F1, AUROC, bias metrics

### 6. Evaluation Layer

- **Evaluators**: Model performance assessment
- **Bias Tracking**: Demographic performance analysis
- **Visualization**: I-value plots, bias metrics, graph visualizations

## Data Flow

1. **Initialization**:
   - Dataloader loads datasets and creates nodes
   - Graph is constructed with hierarchical structure
   - Manager wraps graph for dynamic modifications

2. **Training Loop**:
   - Traversal selects next batch of nodes
   - Trainer processes nodes through model
   - Loss is computed and backpropagated
   - I-values are updated based on model performance
   - Manager optionally modifies graph structure

3. **Evaluation**:
   - Validation/test traversals visit nodes
   - Metrics are computed per demographic group
   - Results are logged and visualized

## Key Concepts

### I-Values (Information Values)

I-values estimate how informative a sample is for improving model performance. They are learned via DQN and guide traversal toward:
- High-uncertainty samples
- Misclassified samples
- Underrepresented demographic groups

### Graph Reduction & Restoration

Dynamic graph modification strategies:
- **Reduction**: Temporarily remove nodes to focus training
- **Restoration**: Add nodes back based on various strategies
- Used for curriculum learning and bias mitigation

### Hierarchical Graph Construction

1. **Level 1**: Group nodes by race-gender combinations
2. **Level 2**: Create fully-connected subgraphs within groups
3. **Level 3**: Apply quality/similarity thresholds for cross-group edges

### Traversal Strategies

- **Comprehensive**: Visit every node once (baseline)
- **Random**: Random walk through graph
- **I-Value**: Greedy selection of highest I-value neighbors
- **Cluster-Hop**: I-value traversal with periodic demographic hops

## Extension Points

The framework is designed for extensibility:

- **Custom Traversals**: Inherit from `Traversal` base class
- **Custom Managers**: Inherit from `GraphManager` base class
- **Custom Trainers**: Inherit from `Trainer` or `AdaptiveTrainer`
- **Custom Models**: Implement `Model` interface
- **Custom Dataloaders**: Inherit from `Dataloader` base class

## Performance Considerations

- **Caching**: Node data and graph structures are cached for efficiency
- **Parallel Processing**: Multi-threaded data loading and processing
- **Sparse Graphs**: Large graphs use sparse representations
- **GPU Acceleration**: Models run on CUDA when available
