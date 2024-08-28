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
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from utils.tag_list_updater import update_tag_list
from trainers import *

import socketserver
import pkgutil
import yaml
import json
import socket

# Set random seed for consistent results (need to test)
random.seed(785134785632495632479853246798)

# Set up W&B
USERNAME = 'wrightlab'
# Don't use '|' in project id.
PROJECT_ID = 'DeepEARLTesting'


"""
Quick overview of how sweep autogeneration works:
Each module has a list of tags which defines what situations the module can be used in.
Each module also has a list of hyperparameters which can be tuned.
Because a lot of the hyperparameters are floats, we can't completely brute-force them.
Instead, we need to use a grid search.



"""

# Use tags to autogenerate sweeps
sweep_config = {
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




class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
          
    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} requested sweep.".format(self.client_address[0]))
        sweep_id, sweep_config = self.server.sweeps.pop()
        print("Ordering sweep: " + sweep_id)
        # just send back the same data, but upper-cased
        self.request.sendall(bytes(USERNAME + "/" + PROJECT_ID + "/" + sweep_id + "|" + PROJECT_ID + "|" + json.dumps(sweep_config), 'utf-8'))
        if len(self.server.sweeps) == 0:
            print("All sweeps completed, shutting down.")
            sys.exit()
        
class MyTCPServer(socketserver.TCPServer):
    sweeps = []
    
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate = True) -> None:
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.setup()
        
    
    def setup(self) -> None:
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
        
        sweep_configuration = {
        "name": "sweepdemo",
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "validation_loss"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1},
            "batch_size": {"values": [16, 32, 64]},
            "epochs": {"values": [5, 10, 15]},
            "optimizer": {"values": ["adam", "sgd"]},
        },
}
        
        self.sweeps = []
        # Start wandb sweeps.
        for model in models:
            for env in envs:
                with open(os.path.join(PATH_TO_MODELS, model, model + '.yaml'), "r") as streamfile:
                    sweep_config = yaml.load(stream=streamfile, Loader=yaml.BaseLoader)
                for subdict in sweep_config["parameters"]:
                    for key, val in sweep_config["parameters"][subdict].items():
                        try:
                            conv = float(val)
                            sweep_config["parameters"][subdict][key] = conv
                        except:
                            pass
                sweep_config["name"] = model + "_" + env
                sweep_config["parameters"]["name"] = {"value": model + "_" + env}
                sweep_config["parameters"]["epochs"] = {"value": ENV_EPOCHS}
                sweep_config["parameters"]["mem_size"] = {"value": MEM_SIZE}
                if env in WRAPPED_ATARI_ENVS:
                    sweep_config["parameters"]["atari_wrappers_enabled"] = {"value": True}
                else:
                    sweep_config["parameters"]["atari_wrappers_enabled"] = {"value": False}
                
                self.sweeps.append([wandb.sweep(sweep=sweep_config, project=PROJECT_ID), sweep_config])
        
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