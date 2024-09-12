import torch
import random
from skimage import io
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
import os
import glob
import sys
import tqdm
from itertools import combinations
import csv
from math import comb
import networkx as nx
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import re
from collections import Counter
from sklearn import preprocessing
import json
import copy
import networkx as nx
from networkx import Graph
from graphs.HyperGraph import HyperGraph
from models.CNNModel import CNNModel
from datasets.DeepfakeDataset import DeepfakeDataset
import wandb
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from datasets import *
from dataloaders import *
from managers import *

class Trainer():
    """
    Base class for pointer/agent based traversal and training on Hypergraphs.
    """
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    def __init__(self, trainer_config, full_config):
        print("Starting trainer.")
        self.config = full_config
        self.num_epochs = trainer_config["epochs"]
        # Dynamically load requested datasets
        datasets = []
        for dataset_name, dataset_args in full_config["datasets"].items():
            print("Loading dataset " + dataset_name)
            datasets.append(globals()[dataset_name](dataset_args))
        print("Finished loading datasets. Now creating graph.")
        # Dynamically load graph using requested dataloader
        self.graph = globals()[next(iter(full_config["dataloader"]))](full_config["dataloader"][next(iter(full_config["dataloader"]))]).get_graph()
        print("Finished HyperGraph creation. Moving to GraphManager initialization.")  
        # Dynamically load graph manager if present
        if "graphmanager" in full_config.keys():
            self.graph = globals()[next(iter(full_config["graphmanager"]))](full_config["graphmanager"][next(iter(full_config["graphmanager"]))])
            print("Graphmanager successfully initialized. Now loading traversals.")
        else:
            print("Skipped, no GraphManager used in config. Now loading traversals.")
        # Dynamically load requested traversals
        self.train_traversal = globals()[next(iter(full_config["traversals"]["train"]))()](full_config["traversals"]["train"][next(iter(full_config["traversals"]["train"]))], full_config["models"]["num_models"])
        self.test_traversal = globals()[next(iter(full_config["traversals"]["test"]))()](full_config["traversals"]["test"][next(iter(full_config["traversals"]["test"]))], full_config["models"]["num_models"])
        print("Traversals loaded, now initializing pointers and models.")
        # Dynamically load requested models
        self.models = [globals()[next(iter(full_config["models"]["model_list"]))](full_config["models"]["model_list"][next(iter(full_config["models"]["model_list"]))]) for _ in range(full_config["models"]["num_models"])]
        print("Models loaded, trainer startup finished.")
        
    def run():
        print("Running trainer.")
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            self.train()
            self.test()

    def process_node_data(self):
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        raise NotImplementedError("Overwrite this!")
        
    def test(self):
        raise NotImplementedError("Overwrite this!")