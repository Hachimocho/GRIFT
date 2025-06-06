# HyperGraph

This is a research project focused on optimizing machine learning model training using graphs.
The general idea is as follows:
- We take a dataset and create a graph where nodes are data points and edges are connections between data points.
- We then train a model on the graph by traversing across it, stopping at nodes to gather data and train the model followed by moving to another node by following an edge.
- This training process is repeated for several epochs, and the model is deployed on validation and testing nodes to evaluate its performance.

This training process is a generic implementation, and can be used for any dataset and model.
In fact, it is designed to be as general as possible, using a module-based approach.

- Data: The data is represented by a class called Data, used to manage data points and their attributes as well as dynamically loading data from a source.
- Node: The node is represented by a class called Node, used to store Data objects and their connections to other nodes.
- Edge: The edge is represented by a class called Edge, used to store connections between nodes.
- Graph: The graph is represented by a class called HyperGraph, used to store nodes and edges and to perform certain operations such as getting subgraphs.
- Manager: The manager is represented by a class called GraphManager, used to store and periodically update the graph.
- Model: The model is represented by a class called Model, used to train, validate, and test the model.
- Traversal: The traversal is represented by a class called Traversal, used to traverse models across the graph.
- Dataset: The dataset is represented by a class called Dataset, used to load data from a source and create nodes.
- Dataloader: The dataloader is represented by a class called Dataloader, used to load datasets and create graphs.
- Trainer: The trainer is represented by a class called Trainer, which combines the manager, traversals, and models into a single training, validation, and testing process.

The framework is designed to be modular and extensible, allowing for easy integration of new data, node, edge, graph, manager, model, traversal, dataset, dataloader, and trainer classes.
This makes it easy to experiment with different data sources, models, and traversal methods without having to modify the core framework.

We also use a novel method called I-value estimation to monitor and reduce model bias as follows (subject to change as the methods are further developed):

1. Initialize DQN to predict Q values for nodes based on their attributes
2. Traverse primary model to nodes and generate predictions
3. Use prediction correctness as reward signal for DQN
4. DQN predicts Q values for nearby nodes to guide traversal
5. Calculate I values as 1-Q for exploration
6. Use DQN weights and prediction patterns to measure and correct both
   inter-attribute and intra-attribute bias

## NEW: Adaptive Trainer Architecture

The framework now includes a refactored AdaptiveTrainer that supports:
- **Dynamic traversal switching** during training
- **Unified trainer interface** that works with all traversal types
- **Automatic capability management** for DQN and bias features
- **State transfer** between different traversal methods

### Usage Examples

#### Single Traversal Mode (Default)
```bash
# Use comprehensive traversal throughout training
python test_hierarchical.py --traversal-type comprehensive --num-epochs 20

# Use I-value traversal with cluster hopping
python test_hierarchical.py --traversal-type i-value-cluster-hop --bias_hop_period 2 --num-epochs 20
```

#### Dynamic Traversal Switching
```bash
# Switch from comprehensive -> i-value -> i-value-cluster-hop at epochs 10 and 20
python test_hierarchical.py --enable-traversal-switching \
    --traversal-sequence "comprehensive,i-value,i-value-cluster-hop" \
    --switch-epochs "10,20" --num-epochs 30 --bias_hop_period 2

# Custom switching sequence
python test_hierarchical.py --enable-traversal-switching \
    --traversal-sequence "random,comprehensive,i-value" \
    --switch-epochs "5,15" --num-epochs 25
```

#### Architecture Testing
```bash
# Test multiple architectures with single traversal
python test_hierarchical.py --architectures "vistransformdf,effnetdf" \
    --traversal-type comprehensive

# Test all traversal types for comparison
python test_hierarchical.py --test-all-traversals \
    --architectures "vistransformdf" --num-epochs 10
```

#### Legacy Mode (Backward Compatibility)
```bash
# Use original trainer classes for comparison
python test_hierarchical.py --trainer-mode legacy \
    --traversal-type i-value-cluster-hop --bias_hop_period 2
```

### Command Line Options

**Trainer Configuration:**
- `--trainer-mode`: Choose 'adaptive' (new) or 'legacy' (original) trainer architecture
- `--traversal-type`: Single traversal method (comprehensive, random, i-value, i-value-cluster-hop)
- `--enable-traversal-switching`: Enable dynamic traversal switching during training
- `--traversal-sequence`: Comma-separated sequence of traversals for switching
- `--switch-epochs`: Comma-separated epochs at which to switch traversals
- `--architectures`: Comma-separated list of CNN architectures to test
- `--test-all-traversals`: Test all traversal types individually for comparison

**Original Options:**
- `--test`: Run in test mode with limited nodes, good for low-compute machines
- `--visualize`: Generate graph visualization csvs for use in cosmograph
- `--quality-threshold`: Set the quality threshold edge construction in the graph
- `--symmetry-threshold`: Set the symmetry threshold edge construction in the graph
- `--embedding-threshold`: Set the embedding threshold edge construction in the graph
- `--cache-nodes`: Cache nodes so node loading can be skipped next time
- `--use-cached`: Use previously cached nodes instead of loading from dataset
- `--num-epochs`: Number of training epochs
- `--bias_hop_period`: Period for bias hop in cluster hop traversals
- `--fair-train`: Use subgroup-balanced training set for graph construction
- `--fair-test`: Use subgroup-balanced validation/test sets for graph construction

## Setup and Running

This is a test build. To run the test:
1. Get the anaconda or mamba package managers running on your system
2. Build the environment using the environment.yml file
3. Get the AI-face dataset unpacked on your system (https://github.com/Purdue-M2/AI-Face-FairnessBench)
4. Update the dataset path to your AI-face dataset root
5. Generate quality csvs using the regenerate_quality_csvs.sh script, or request them from me (they're too large for the git repo)
6. Run the current test build at test_hierarchical.py with desired options

Example command: 
```bash
python test_hierarchical.py --cache-nodes --cached-nodes 10000 --use-cached \
    --quality-threshold .7 --symmetry-threshold .5 --embedding-threshold .9 \
    --num-epochs 15 --bias_hop_period 2 --fair-train --fair-test \
    --enable-traversal-switching --traversal-sequence "comprehensive,i-value-cluster-hop" \
    --switch-epochs "10" --architectures "vistransformdf"
```

Expected outputs:
1. Cached node file in node_cache folder
2. Cached edge file in graph_cache folder
3. Best model checkpoint in saved_models folder
4. Training, validation, and testing logs in logs folder
5. Any requested visualization csvs in the base directory
6. Any requested threshold search results csv in the logs directory + plots in logs/search_plots

## Testing the Refactored Architecture

To test the new architecture:
```bash
# Quick component test
python test_refactoring.py

# Full demonstration with switching
python test_adaptive_trainer.py

# Production usage with new options
python test_hierarchical.py --enable-traversal-switching --num-epochs 20
```
