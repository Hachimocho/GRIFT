import os
from nodes import *
from edges import *

class Dataset():
    """
    Takes a data path, node and data classes, and loads the data into node format.
    Every dataset must do the following:
    - Create a list of nodes containing the relevant data
    - Create a list of masks for training, validation, and testing
    - Implement a load() method which loads the data into the nodes and returns the node list
    """
    tags = ["none"]
    hyperparameters: dict | None = None
    
    def __init__(self, data_root, data_class, data_args, node_class, node_args):
        assert os.path.isdir(data_root)
        self.data_root = data_root
        self.data_class = data_class
        self.node_class = node_class
        self.data_args = data_args
        self.node_args = node_args
        self.nodes = []
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        
    def load(self):
        #self.nodes = ...
        #return self.nodes
        raise NotImplementedError("Overwrite this!")
    
    def __len__(self):
        return len(self.nodes)
