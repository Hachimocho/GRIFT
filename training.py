import torch
from torch_geometric.data import Data
import random
from torch_geometric.utils import k_hop_subgraph, to_undirected
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
from torch_geometric.utils.convert import to_networkx
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

from models.CNNModel import CNNModel
from DeepfakeDataset import DeepfakeDataset
import wandb
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from utils.tag_list_updater import update_tag_list
from trainers import *

# Set random seed for consistent results (need to test)
random.seed(785134785632495632479853246798)

# W&B setup
wandb.login(key="8e2ea87ef9c3afd51f009eaf850d7b22e935fa1e")

"""
Config explanation:
datasets: Full list of datasets to load into a single HyperGraph. Must be a valid Dataset class.
    nodes: List of node types to be used for the dataset. Must be valid Node classes.
    node_selection_method: How nodes are picked from the "nodes" list.
        "exact": Only the one given node type will be used. Errors if more than one node type is given.
        TODO: add more node selection methods
    args (optional): Extra arguments for the given dataset.
    
dataloader: One given dataloader to create the HyperGraph.
    edges: List of edge types to be used in the HyperGraph. Must be valid Edge classes.
    edge_selection_method: How edges are added using the "edges" list.
        "exact": Only the one given edge type will be used. Errors if more than one edge type is given.
        TODO: add more node selection methods
    args (optional): Extra arguments for the given dataloader.

models: Full list of models to be trained. Must be valid Model classes. (TODO: Add data compatibility check here)

"""

wandb.init(
    # set the wandb project where this run will be logged
    project="DeepEARL",

    # track hyperparameters and run metadata
    config={
    "trainer": {
        "DeepfakeAttributeTrainer": {
            "epochs": 5,
            "args": {
                "key_attributes": "gender"
            }
        },
    },
    "datasets": {
        "CDFDataset": {
            "nodes": ["AttributeNode"],
            "node_selection_method": "exact",
            "args": {
                "frames_per_video": 15
            }
        }
    },
    "dataloader": {
        "DeepfakeDataLoader": {
            "edges": ["AttributeEdge"],
            "edge_selection_method": "exact"
        }
    },
    "models": {
        "num_models": 2,
        "model_selection_method": "exact",
        "model_list": {
            "CNNModel": {
                "hops_to_analyze": 0,
                "learning_rate": 0.001,
                "args": {
                    "model_name": "resnestdf"
                }
            }
        },
    },
    "traversals": {
        "train": {
            "AttributeWarpTraversal": {
                "args": {
                    "warp_threshold": 0.01,
                    "steps_per_epoch": 100,
                    "timesteps_before_return_allowed": 3,
                },
            },
        },
        "test": {
            "AttributeBoringTraversal": {
                "args": {
                    "warp_threshold": 0.01,
                    "steps_per_epoch": 100,
                    "timesteps_before_return_allowed": 3,
                },
            },
        },
    }
    }
)

# Several asserts to make sure config is in correct format
assert len(wandb.config["trainer"]) == 1
assert len(wandb.config["dataloader"]) == 1
assert len(wandb.config["datasets"]) >= 1
assert len(wandb.config["models"]["model_list"]) >= 1
assert len(wandb.config["traversals"]) >= 1
assert ("graphmanager" not in wandb.config.keys()) or (len(wandb.config["graphmanager"] == 1))

# Update tag list
update_tag_list()

# Start up trainer
trainer = globals()[next(iter(wandb.config["trainer"]))](wandb.config["trainer"][next(iter(wandb.config["trainer"]))], wandb.config)
trainer.run()
trainer.test()
