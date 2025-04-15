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

This is a test build. To run the test:
1. Get the anaconda or mamba package managers running on your system
2. Build the environment using the environment.yml file
3. Get the AI-face dataset unpacked on your system (https://github.com/Purdue-M2/AI-Face-FairnessBench)
4. Update the dataset path to your AI-face dataset root
5. Generate quality csvs using the regenerate_quality_csvs.sh script, or request them from me (they're too large for the git repo)
6. Run the current test build at test_hierarchical.py. The test script supports several options:
6a. --test: Run in test mode with limited nodes, good for low-compute machines
6b. --visualize: Generate graph visualization csvs for use in cosmograph
6c. --show: Deprecated due to switching to cosmograph
6d. --quality-threshold: Set the quality threshold edge construction in the graph
6e. --symmetry-threshold: Set the symmetry threshold edge construction in the graph
6f. --embedding-threshold: Set the embedding threshold edge construction in the graph
6g. --cache-nodes: Cache nodes so node loading can be skipped next time
6h. --cache-full: Deprecated due to node caching changes
6i. --use-cached: Use previously cached nodes instead of loading from dataset
6j. --use-full-cache: Load the full dataset from cache instead of the subset
6k. --cached-nodes: Number of nodes to cache per split when not using full cache
6l. --cache-file: Path to cache file for saving/loading nodes
6m. --search: Run grid search over threshold combinations
6n. --search-split: Split to use for grid search (default: train)
6o. --quality-steps: Number of steps for quality threshold grid search (default: 5)
6p. --symmetry-steps: Number of steps for symmetry threshold grid search (default: 5)
6q. --embedding-steps: Number of steps for embedding threshold grid search (default: 5)
6r. --search-results: File to save search results to (default: threshold_search_results.csv)