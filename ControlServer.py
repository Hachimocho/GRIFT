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
# from DeepfakeDataset import DeepfakeDataset
import wandb
from utils.import_utils import import_classes_from_directory, get_classes_from_module, get_tagged_classes_from_module
from utils.tag_list_updater import update_tag_list
from trainers import *

import socketserver
import pkgutil
import yaml
import json
import socket
import itertools
from utils.AceV1 import AceV1

# Set up config
"""
The test set is hardcoded in the config to ensure the correct problem is being tested.
Training sets and graph generation is included in meta-optimizations using the tagging system.
Two separate sets of allowed tags are used: data and model.
Data tags are used for data selection, loading, and graph generation and management. (Everything except Model)
Model tags are used for model selection to ensure the model can actually be used with the input format of the data. (Model)
This will be deprecated in the future in favor of input conversion modules that allow all models to be used with all data.
"""

"""
Super general overview of the project (see docs for more info):
Load all the possible module combinations
Test a few and score them based on validation accuracy
Use scores to predict success of other configs
Test combinations with high predicted success
Repeat
Get a few of the highest-accuracy models
Test on test data and record results
"""

meta_config = {
    "data_config": {
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
        }
    },
    "max_time": 86400, # 24 hours
    "optimizations_per_sweep": 10,
    "sweeps_between_meta_optimizations": 10,
    "epochs_per_run": 10,
    "epsilon": 0.99,
    "epsilon_mult": 0.995,
    "epsilon_min": 0.05,
    "time_factor": 0.25,
    "allowed_data_tags": ["deepfakes", "any"], # Only deepfake data can be used for training
    "allowed_model_tags": ["image", "any"], # Only models capable of processing image data can be used
    "sweep_config": {
        "name": "sweepdemo",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {},
        "early_terminate": {
            "type": "hyperband", 
            "s": 2, 
            "eta": 3, 
            "max_iter": 27, 
        }
    }
}

# Set up W&B
USERNAME = 'wrightlab'
# Don't use '|' in project id.
PROJECT_ID = 'DeepEARLTesting'



class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    
    def __init__(self):
        super().__init__()  
        self.served_sweeps = 0
        self.ace = AceV1()
        self.epsilon = meta_config["epsilon"]
        self.epsilon_mult = meta_config["epsilon_mult"]
        self.epsilon_min = meta_config["epsilon_min"]
        
    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} requested sweep.".format(self.client_address[0]))
        if random.random() < self.server.epsilon:
            sweep_id, sweep_config = random.choice(self.server.sweeps)
        else:
            sweep_id, sweep_config = self.server.sweeps.pop(0)
        self.server.epsilon = max(self.server.epsilon * self.server.epsilon_mult, self.server.epsilon_min)
        print("Ordering sweep: " + sweep_id)
        # just send back the same data, but upper-cased
        self.request.sendall(bytes(USERNAME + "/" + PROJECT_ID + "/" + sweep_id + "|" + PROJECT_ID + "|" + json.dumps(sweep_config), 'utf-8'))
        if len(self.server.sweeps) == 0:
            print("All sweeps completed, shutting down.")
            sys.exit()
        served_sweeps += 1
        if served_sweeps % meta_config["sweeps_between_meta_optimizations"] == 0:
            meta_optimize()
            
    def meta_optimize(self):
        print("Beginning meta-optimization.")
        api = wandb.Api()
        sweep_runs = api.runs(path=USERNAME + "/" + PROJECT_ID, filters={"state": "finished"})
        scores = {run.name: run.summary["score"] for run in sweep_runs}
        self.ace.meta_optimize(scores, self.server.sweeps)
        
class MyTCPServer(socketserver.TCPServer):
    sweeps = []
    
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate = True) -> None:
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.setup()
        
    
    def setup(self) -> None:
        """
        Setup the server by generating all possible combinations of modules that have the requested tags.

        This function does the following:
        1. Set random seed for consistent results.
        2. Update the tag list.
        3. Get all valid combinations of modules that have the requested tags.
        4. Print all available modules.
        5. Generate all possible combinations of modules.
        6. Iterate through all possible combinations and create a sweep configuration for each one.
        7. Add the sweep to the list of sweeps.
        """ 

        
        # Set random seed for consistent results (need to test)
        random.seed(785134785632495632479853246798)

        # Update tag list
        update_tag_list()
        
        # Get all valid combinations of modules that have the requested tags
        def get_tagged_classes_from_module_any(module_name, tags):
            return [name for name, obj in inspect.getmembers(sys.modules[module_name]) if inspect.isclass(obj) and any(tag in obj.tags for tag in tags)]
        available_data_modules = get_tagged_classes_from_module_any('data', meta_config['allowed_data_tags'])
        available_node_modules = get_tagged_classes_from_module_any('nodes', meta_config['allowed_data_tags'])
        available_edge_modules = get_tagged_classes_from_module_any('edges', meta_config['allowed_data_tags'])
        available_dataloader_modules = get_tagged_classes_from_module_any('dataloaders', meta_config['allowed_data_tags'])
        available_dataset_modules = get_tagged_classes_from_module_any('datasets', meta_config['allowed_data_tags'])
        available_graph_modules = get_tagged_classes_from_module_any('graphs', meta_config['allowed_data_tags'])
        available_manager_modules = get_tagged_classes_from_module_any('managers', meta_config['allowed_data_tags'])
        available_trainer_modules = get_tagged_classes_from_module_any('trainers', meta_config['allowed_model_tags'])
        available_traversal_modules = get_tagged_classes_from_module_any('traversals', meta_config['allowed_data_tags'])
        available_model_modules = get_tagged_classes_from_module_any('models', meta_config['allowed_model_tags'])
        
        
        
        print("Available modules:")
        for module_type, modules in {
            "Data": available_data_modules,
            "Node": available_node_modules,
            "Edge": available_edge_modules,
            "Dataloader": available_dataloader_modules,
            "Dataset": available_dataset_modules,
            "Graph": available_graph_modules,
            "Manager": available_manager_modules,
            "Model": available_model_modules,
            "Trainer": available_trainer_modules,
            "Traversal": available_traversal_modules
        }.items():
            print(f"  {module_type}:")
            for module in modules:
                print(f"    - {module}")
                
        sweep_configuration = meta_config["sweep_config"]
        
        """
        Quick overview of how sweep autogeneration works:
        Each module has a list of tags which defines what situations the module can be used in.
        Each module also has a list of hyperparameters which can be tuned.
        Not only that, but which modules are used is a hyperparameter as well.

        To resolve this, we make a call to the Advanced Correlation Engine (ACE) to sort all possible configurations.
        Currently, ACE uses a simple NN to score each possible configuration based on past results in accuracy and time.
        We then serve sweeps either randomly or from the top of the list based on epsilon (e).
        This should continue until acceptable accuracy is achieved or a certain amount of time has passed.
        """
        self.sweeps = []
        # Iterate through all possible combinations of modules
        all_module_combinations = list(itertools.product(*[available_data_modules, available_dataloader_modules, available_dataset_modules, available_graph_modules, available_manager_modules, available_model_modules, available_trainer_modules, available_traversal_modules]))
        for combo in tqdm(all_module_combinations, desc="Generating sweep combinations"):
            # Might need to implement this after loading from modules
            # for subdict in sweep_config["parameters"]:
            #     for key, val in sweep_config["parameters"][subdict].items():
            #         try:
            #             conv = float(val)
            #             sweep_config["parameters"][subdict][key] = conv
            #         except:
            #             pass
            
            # Set name and epochs
            sweep_config["name"] = "_".join(combo)
            sweep_config["parameters"]["name"] = "_".join(combo)
            sweep_config["parameters"]["epochs"] = {"value": int(meta_config["epochs_per_run"])}

            # Set module hyperparameters
            for module in combo:
                sweep_config["parameters"][module] = {"value": module.hyperparameters}

            # Add the sweep to the list of sweeps
            self.sweeps.append([wandb.sweep(sweep=sweep_config, project=PROJECT_ID), sweep_config])
            
        # TEMP: Kill after testing
        sys.exit()

if __name__ == "__main__":
    # Find the hostname
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    HOST = s.getsockname()[0]
    s.close()
    # Set port directly
    PORT = 9998
    # Create the server, binding to the given address on the given port.
    try:
        with MyTCPServer((HOST, PORT), MyTCPHandler) as server:
            # Activate the server; this will keep running until you
            # interrupt the program with Ctrl-C
            
            print("Starting Server with IP: " + HOST + " and port: " + str(PORT))
            server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down due to user input.")
