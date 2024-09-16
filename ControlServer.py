
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
from utils.WandbArtifactUtils import save_tag, load_tag
from trainers import *
from data import *
from models import *
from graphs import *
from edges import *
from nodes import *
from managers import *
from dataloaders import *
from datasets import *
from trainers import *
from traversals import *

import socketserver
import pkgutil
import yaml
import json
import socket
import itertools
import paramiko
import getpass

# Use this to specify ACE version
from utils.AceV1 import ACE

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



def upload_sweeps(sweep_list, batch_tag, project):
    uploaded_sweeps = []
    for sweep_config in sweep_list:
        # Add the batch tag and status tags to the sweep configuration
        sweep_config['parameters']["tags"] = {"value": [batch_tag, 'unclaimed', 'unfinished']}
        
        # Ensure gpu_usage is specified
        if 'gpu_usage' not in sweep_config['parameters'].keys():
            raise ValueError("Each sweep must specify 'gpu_usage' in GB")
        
        # Create a new sweep
        sweep_id = wandb.sweep(sweep_config, project=project)
        
        # Add the sweep_id to the list of uploaded sweeps
        uploaded_sweeps.append(sweep_id)
        
        print(f"Uploaded sweep: {sweep_id} with batch tag: {batch_tag}")
    
    return uploaded_sweeps

def check_batch_completion(entity, project_name, batch_tag):
    api = wandb.Api()
    project = [p for p in api.projects(entity) if p.name == project_name][0]
    active_sweeps = project.sweeps()
    all_completed = True
    for sweep in active_sweeps:
        if batch_tag in sweep.config["parameters"]["tags"]["value"] and 'unfinished' in sweep.config["parameters"]["tags"]["value"]:
            all_completed = False
            break
    
    return all_completed

def run_completion_code(entity, project, batch_tag, epsilon, epsilon_mult, epsilon_min, ace, meta_config):
    served_sweeps = 0
    
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
    available_trainer_modules = get_tagged_classes_from_module_any('trainers', meta_config['allowed_data_tags'])
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
            
    sweep_config = meta_config["sweep_config"]
    
    """
    Quick overview of how sweep autogeneration works:
    Each module has a list of tags which defines what situations the module can be used in.
    Each module also has a list of hyperparameters which can be tuned.
    Not only that, but which modules are used is a hyperparameter as well.

    
    To resolve this, we make a call to the Advanced Correlation Engine (ACE) to sort all possible configurations.
    Currently, ACE uses simple gaussian regression to score each possible configuration based on past results in accuracy and time.
    We then serve sweeps either randomly or from the top of the list based on epsilon (e).
    This should continue until acceptable accuracy is achieved or a certain amount of time has passed.
    """
    sweeps = []
    t = time.time()
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
        # print("_".join(combo))
        # print(type("_".join(combo)))
        # #sweep_config["name"] = "_".join(combo)
        # sweep_config["parameters"]["name"] = {"value": "_".join(combo)},
        sweep_config["parameters"]["epochs"] = {"value": int(meta_config["epochs_per_run"])}
    
        # Set module hyperparameters
        for module in combo:
            try:
                if globals()[module].hyperparameters is not None:
                    sweep_config["parameters"][module] = globals()[module].hyperparameters
            except AttributeError:
                pass
        # TODO: Add GPU usage estimation based on model and batch size
        sweep_config["parameters"]["gpu_usage"] = {"value": 0}
        
        # Add the sweep to the list of sweeps
        sweeps.append([sweep_config, "_".join(combo)])
    
    print(f"Number of sweeps: {len(sweeps)}")
    print(f"Maximum number of runs: {len(sweeps) * meta_config['sweep_config']['early_terminate']['max_iter']}")
    
    # Upload first batch of sweeps
    pending_sweeps = []
    for _ in range(meta_config["sweeps_between_meta_optimizations"]):
        if random.random() < epsilon:
            sweep_config, sweep_combo = random.choice(sweeps)
        else:
            sweep_config, sweep_combo = sweeps.pop(0)
        pending_sweeps.append(sweep_config)
        epsilon = max(epsilon * epsilon_mult, epsilon_min)
        if len(sweeps) == 0:
            print("All sweeps completed, starting final testing runs.")
            # TODO: Implement final testing runs
            # TODO: Implement client shutdown
            sys.exit()
        served_sweeps += 1
        if (time.time() - t) > meta_config["max_time"]:
            print("Maximum time reached, starting final testing runs.")
            # TODO: Implement final testing runs
            # TODO: Implement client shutdown
            sys.exit()
    upload_sweeps(pending_sweeps, batch_tag, project)
    
    while True:
        while not check_batch_completion(USERNAME, project, batch_tag):
            print("Waiting for all sweeps to complete...")
            time.sleep(300)  # Check every 5 minutes
        
        print("All sweeps in the batch have completed!")
        pending_sweeps = []
        for _ in range(meta_config["sweeps_between_meta_optimizations"]):
            if random.random() < epsilon:
                sweep_config, sweep_combo = random.choice(sweeps)
            else:
                sweep_config, sweep_combo = sweeps.pop(0)
            pending_sweeps.append(sweep_config)
            epsilon = max(epsilon * epsilon_mult, epsilon_min)
            if len(sweeps) == 0:
                print("All sweeps completed, starting final testing runs.")
                # TODO: Implement final testing runs
                # TODO: Implement client shutdown
                sys.exit()
            served_sweeps += 1
            if (time.time() - t) > meta_config["max_time"]:
                print("Maximum time reached, starting final testing runs.")
                # TODO: Implement final testing runs
                # TODO: Implement client shutdown
                sys.exit()
        upload_sweeps(pending_sweeps, batch_tag, project)
        print("Beginning meta-optimization.")

        # Initialize the WandB API
        api = wandb.Api()
        
        # Fetch all runs in the given project
        runs = api.runs(path=entity + "/" + project)

        # Filter out everything but the highest-scoring run for each sweep
        highest_scoring_runs = {}
        for run in runs:
            sweep_id = run.sweep.id
            if sweep_id not in highest_scoring_runs or run.history()["score"].iloc[-1] > highest_scoring_runs[sweep_id][1]:
                highest_scoring_runs[sweep_id] = (run.history()["config"].iloc[-1], run.history()["score"].iloc[-1])

        # Create a dictionary where each key (run.history()["config"].iloc[-1]) has one value (run.history()["score"].iloc[-1])
        scores = {config: score for config, (_, score) in highest_scoring_runs.items()}
        
        # Reorder sweeps using meta-optimization
        sweeps = ace.meta_optimize(sweeps, scores)

if __name__ == "__main__":
    file = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
    USERNAME = file["WANDB_USERNAME"]
    PROJECT_ID = file["PROJECT_NAME"]
    meta_config = file["META_CONFIG"]
        
    # Use random tag for data sorting
    TAG = random.randrange(2**32 - 1)

    # Login to wandb
    with open("key.txt") as f:
        api_key = f.readline()
        wandb.login(key=api_key)

    # Handler setup
    ace = ACE()
    epsilon = meta_config["epsilon"]
    epsilon_mult = meta_config["epsilon_mult"]
    epsilon_min = meta_config["epsilon_min"]
    run_completion_code(USERNAME, PROJECT_ID, TAG, epsilon, epsilon_mult, epsilon_min, ace, meta_config)