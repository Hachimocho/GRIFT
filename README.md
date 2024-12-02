# HyperGraph

This is a research project focused on optimizing machine learning model training using graphs.
The general idea is as follows:
- We take a dataset and create a graph where nodes are data points and edges are connections between data points.
- We then train a model on the graph by traversing across it, stopping at nodes to gather data and train the model followed by moving to another node by following an edge.
- This training process is repeated for several epochs, and the model is deployed on validation and testing nodes to evaluate its performance.

This training process is a generic implementation, and can be used for any dataset and model.
In fact, it is designed to be as general as possible, using a module-based approach

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

Each module has a set of hyperparameters that can be tuned to optimize the performance of the framework, as well as a set of tags that can be used to identify what other modules can be used with it.

We also use integration with wandb to optimize the training process in several ways. It allows for distriubed training using a unique lock system that allows systems to claim sweeps to run, as well as doing hyperparameter optimization for each module. 

The module combinations themselves are also optimized using ACE, the Adaptive Combination Engine. This is designed to find the best combination of modules for a given task, by scoring tested combination of modules based on their performance on a validation set and using these scores to predict the performance of unseen combinations.

This is a work in progress. The goals for the project are, in the following order:
1. Finish the test.py implementation to allow for easy testing of the framework
2. Finish implementing IValueTraversal to allow for intelligent traversal of the graph based on the information content of nodes
3. Finish the ACE implementation to allow for easy optimization of the module combinations
4. Finish the training, agent, and control server implementations to allow for easy distributed training of models
5. Complete comprehensive tests to ensure that the framework is working correctly
6. Update all modules with correct tags so that they are compatible with the framework
7. Complete the documentation for the framework
8. Add automated tests so that new modules can be automatically tested and verified to work with all modules that their tags allow for
9. Follow up work in implementing more modules and improving both parts of the optimization process.