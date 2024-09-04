import torch
from torch_geometric.data import Data
import random
from torch_geometric.utils import k_hop_subgraph, to_undirected, subgraph, to_networkx, from_networkx, shuffle_node
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
import json
import copy
import networkx as nx
from networkx import Graph
from graphs.HyperGraph import HyperGraph
from models.CNNModel import CNNModel
from datasets.DeepfakeDataset import DeepfakeDataset
import wandb
from trainers.Trainer import Trainer

class DeepfakeAttributeTrainer(Trainer):
    """
    Base class for pointer/agent based traversal and training on Hypergraphs.
    """
    tags = ["deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    def __init__(self, dataloader, rng_threshold, num_pointers, num_steps=1000, X=5, traversal_method='random_delay_repeat', K=1, val_test_traversal_method="boring_comprehensive", key_attributes=[]):
        self.dataloader = dataloader
        self.rng_threshold = rng_threshold
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.X = X
        self.train_traversal_method = str.lower(traversal_method)
        self.val_test_traversal_method = str.lower(val_test_traversal_method)
        self.K = K
        self.key_attributes = key_attributes

    @staticmethod
    def available_methods():
        # TODO
        pass

    def traverse_graph(self, mode="train"):
        """ Moves one or more pointers around the given graph using some traversal method.
        Args:
            data (Data): The dataloader. Use Deepfakedataloader.py
            process_node_data (func): The function that takes the node data
            rng_threshold (float): Probability of moving to a random node instead of an adjacent node, 0 to 1
            num_pointers (int): Number of pointers to place and move on the graph
            num_steps (int, optional): Number of timesteps to do. May be stopped early if using a comprehensive method. Defaults to 1000.
            X (int, optional): How many timesteps a node should be inaccessible after visiting, for supported traversal methods. Defaults to 5.
            traversal_method (str): What traversal method to use. Check available_methods() for a list. Defaults to 'random_delay_repeat'.

        Raises:
            ValueError: If an invalid traversal method is given.
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be one of train, val, and test")
        
        elif mode == "train":
            num_steps = self.num_steps
            traversal_method = self.train_traversal_method
            main_data = self.dataloader.graph
            data = self.dataloader.train_graph
            num_nodes = data.num_nodes
            train_nodes_indexes = [i for i, val in enumerate(self.dataloader.train_mask) if val]
            videos = [video for i, video in enumerate(self.dataloader.videos) if i in train_nodes_indexes]

        elif mode == "val":
            num_steps = self.num_steps
            traversal_method = self.val_test_traversal_method
            main_data = self.dataloader.graph
            data = self.dataloader.val_graph
            num_nodes = data.num_nodes
            val_nodes_indexes = [i for i, val in enumerate(self.dataloader.val_mask) if val]
            videos = [video for i, video in enumerate(self.dataloader.videos) if i in val_nodes_indexes]
           
            
        elif mode == "test":
            traversal_method = self.val_test_traversal_method
            num_steps = -1
            main_data = self.dataloader.graph
            data = self.dataloader.test_graph
            num_nodes = data.num_nodes
            test_nodes_indexes = [i for i, val in enumerate(self.dataloader.test_mask) if val]
            videos = [video for i, video in enumerate(self.dataloader.videos) if i in test_nodes_indexes]
        t = 0
        done = False
        
        
        # Move pointers to adjacent nodes not visited in the last X steps
        if traversal_method == 'random_delay_repeat':
            # Startup pointers
            pointers = [{'current_node': random.randint(0, num_nodes), 'last_visited': [-self.X for _ in range(num_nodes)], 'ever_visited': [0 for _ in range(num_nodes)]} for _ in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else True):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    last_visited = pointer['last_visited']

                    # Get the indices of the adjacent nodes
                    adj_nodes = data.edge_index[1][data.edge_index[0] == current_node].tolist()

                    # Filter out the nodes that were visited in the last X timesteps
                    adj_nodes = [node for node in adj_nodes if t - last_visited[node] > self.X]

                    # If there are no adjacent nodes or the RNG call is below the threshold,
                    # move the pointer to a random not recently visited node
                    if not adj_nodes or random.random() < self.rng_threshold:
                        not_recently_visited_nodes = [node for node in range(num_nodes) if t - last_visited[node] > self.X]
                        if not_recently_visited_nodes:
                            current_node = random.choice(not_recently_visited_nodes)
                        else:
                            continue
                    else:
                        # Randomly select an adjacent node
                        current_node = random.choice(adj_nodes)

                    # Get the nodes X hops away from the current node
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    # Process the data of the nodes X hops away
                    for node in nodes_X_hops_away.tolist():
                        node_list.append(data.x[node])
                        labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    pointer['current_node'] = current_node
                    pointer['last_visited'][current_node] = t
        
        # Move pointers to never-before visited nodes, adjacent if possible. Stop once all nodes have been visited once.
        elif traversal_method == 'random_never_repeat_comprehensive':
            # Startup pointers
            pointers = [{'current_node': random.randint(0, num_nodes), 'last_visited': [-self.X for _ in range(num_nodes)], 'ever_visited': [0 for _ in range(num_nodes)]} for _ in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else True):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    ever_visited = pointer['ever_visited']

                    # Get the indices of the adjacent nodes
                    adj_nodes = data.edge_index[1][data.edge_index[0] == current_node].tolist()

                    # Filter out the nodes that were visited in the last X timesteps
                    adj_nodes = [node for node in adj_nodes if ever_visited[node] == 0]

                    # If there are no adjacent nodes or the RNG call is below the threshold,
                    # move the pointer to a random not recently visited node
                    if not adj_nodes or random.random() < self.rng_threshold:
                        not_visited_nodes = [node for node in range(num_nodes) if ever_visited[node] == 0]
                        if not_visited_nodes:
                            current_node = random.choice(not_visited_nodes)
                        else:
                            return
                    else:
                        # Randomly select an adjacent node
                        current_node = random.choice(adj_nodes)

                    # Get the nodes X hops away from the current node
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    # Process the data of the nodes X hops away
                    for node in nodes_X_hops_away.tolist():
                        node_list.append(data.x[node])
                        labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    pointer['current_node'] = current_node
                    pointer['ever_visited'][current_node] = 1
                    
        elif traversal_method == 'boring_comprehensive':
            # print(data.num_nodes)
            # print(num_nodes)
            # sys.exit()
            # Overwrite pointers for boring traversal
            pointers = [{'current_node': 0} for _ in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else not done):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    
                    # Get the nodes X hops away from the current node
                    # print("Getting adjacent nodes.")
                    # print(current_node)
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    
                    # Process the data of the nodes X hops away
                    for node in nodes_X_hops_away.tolist():
                        node_list.append(data.x[node])
                        labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    if (current_node < (num_nodes - 1)):
                        pointer['current_node'] = current_node + 1
                    else:
                        # Read through all nodes, break.
                        #print("breaking")
                        if all(i == (num_nodes - 1) for i in [pointer["current_node"] for pointer in pointers]):
                            done = True
                        break
                if done:
                    break
            
        elif traversal_method == "attribute_delay_repeat":
            # Startup pointers
            num_attributes = len(data.x[0][1])
            attribute_set = [[key_attribute_index, key_attribute_value] for key_attribute_index, key_attribute_value in zip([i for i in self.key_attributes for _ in range(2)], [[0, 2][j % 2] for j in range(len(self.key_attributes) * 2)])]
            pointers = [{'key_attributes': attribute_set[i], 'current_node': random.choice([num for num in range(num_nodes) if int(data.x[num][1][attribute_set[i][0]]) != int(attribute_set[i][1])]), 'last_visited': [-self.X for _ in range(num_nodes)], 'ever_visited': [0 for _ in range(num_nodes)]} for i in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else True):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    last_visited = pointer['last_visited']

                    # Get the indices of the adjacent nodes
                    adj_nodes = data.edge_index[1][data.edge_index[0] == current_node].tolist()

                    # Filter out the nodes that were visited in the last X timesteps
                    adj_nodes = [node for node in adj_nodes if t - last_visited[node] > self.X]
                    
                    # Filter out the nodes that don't have the correct key attribute
                    adj_nodes = [node for node in adj_nodes if int(data.x[node][1][pointer["key_attributes"][0]]) != int(pointer["key_attributes"][1])]

                    # If there are no adjacent nodes or the RNG call is below the threshold,
                    # move the pointer to a random not recently visited node
                    if not adj_nodes or random.random() < self.rng_threshold:
                        not_recently_visited_nodes = [node for node in range(num_nodes) if t - last_visited[node] > self.X]
                        if not_recently_visited_nodes:
                            current_node = random.choice(not_recently_visited_nodes)
                        else:
                            continue
                    else:
                        # Randomly select an adjacent node
                        current_node = random.choice(adj_nodes)

                    # Get the nodes X hops away from the current node
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    # Process the data of the nodes X hops away
                    for node in nodes_X_hops_away.tolist():
                        node_list.append(data.x[node])
                        labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    pointer['current_node'] = current_node
                    pointer['last_visited'][current_node] = t
            
        elif traversal_method == "attribute_boring_comprehensive":
            num_attributes = len(data.x[0][1])
            attribute_set = [[key_attribute_index, key_attribute_value] for key_attribute_index, key_attribute_value in zip([i for i in self.key_attributes for _ in range(2)], [[0, 2][j % 2] for j in range(len(self.key_attributes) * 2)])]
            pointers = [{'key_attributes': attribute_set[i], 'current_node': 0} for i in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else not done):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    
                    # Get the nodes X hops away from the current node
                    # print("Getting adjacent nodes.")
                    # print(current_node)
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    
                    # Process the data of the nodes X hops away which have the correct key attribute
                    for node in nodes_X_hops_away.tolist():
                        if int(data.x[node][1][pointer["key_attributes"][0]]) != int(pointer["key_attributes"][1]):
                            node_list.append(data.x[node])
                            labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    if (current_node < (num_nodes - 1)):
                        pointer['current_node'] = current_node + 1
                    else:
                        # Read through all nodes, break.
                        #print("breaking")
                        if all(i == (num_nodes - 1) for i in [pointer["current_node"] for pointer in pointers]):
                            done = True
                        break
                if done:
                    break
                
        else:
            raise ValueError("Invalid traversal method.")
                
        

    def process_node_data(self, node_data, pointer_id, labels, mode):
        # Do something with node_data and pointer_id
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        #print("Training started.")
        self.traverse_graph(mode="train")
        #print("Training finished.")
        
    def validate(self):
        #print("Validation started.")
        self.traverse_graph(mode="val")
        #print("Validation finished.")
        
    def test(self):
        #print("Testing started.")
        self.traverse_graph(mode="test")
        #print("Testing finished.")
        
    def run():
        print("Running trainer.")
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            self.train()
            self.validate()
        
# import torch
# from torch_geometric.data import Data
# import random
# from torch_geometric.utils import k_hop_subgraph, to_undirected
# from skimage import io
# from skimage.metrics import structural_similarity
# from skimage.color import rgb2gray
# import os
# import glob
# import sys
# import tqdm
# from itertools import combinations
# import csv
# from math import comb
# from torch_geometric.utils.convert import to_networkx
# import networkx as nx
# import matplotlib.pyplot as plt
# import cv2
# from tqdm import tqdm
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
# import time
# import numpy as np
# import re
# from collections import Counter
# from sklearn import preprocessing



# """
# This is what a generic
# """

# # Label key (0 for not present, 1 for unknown, 2 for present):
# # male,young,middle_aged,senior,asian,white,black,shiny_skin,bald,wavy_hair,receding_hairline,bangs,black_hair,blond_hair,gray_hair,no_beard,mustache,goatee,oval_face,square_face,double_chain,chubby,obstructed_forehead,fully_visible_forehead,brown_eyes,bags_under_eyes,bushy_eyebrows,arched_eyebrows,mouth_closed,smiling,big_lips,big_nose,pointy_nose,heavy_makeup,wearing_hat,wearing_necktie,wearing_lipstick,no_eyewear,eyeglasses,attractive
# ATTRIBUTES = "male,young,middle_aged,senior,asian,white,black,shiny_skin,bald,wavy_hair,receding_hairline,bangs,black_hair,blond_hair,gray_hair,no_beard,mustache,goatee,oval_face,square_face,double_chain,chubby,obstructed_forehead,fully_visible_forehead,brown_eyes,bags_under_eyes,bushy_eyebrows,arched_eyebrows,mouth_closed,smiling,big_lips,big_nose,pointy_nose,heavy_makeup,wearing_hat,wearing_necktie,wearing_lipstick,no_eyewear,eyeglasses,attractive".split(',')

# ATTRIBUTE_DICT = {i : val for i, val in enumerate(ATTRIBUTES)}

# CHOSEN_ATTRIBUTES = {
#     "race": [ATTRIBUTES.index("white"), ATTRIBUTES.index("black"), ATTRIBUTES.index("asian")],
#     "gender": [ATTRIBUTES.index("male")],
#     "age": [ATTRIBUTES.index("young"), ATTRIBUTES.index("middle_aged"), ATTRIBUTES.index("senior")],
#     "none": []
# }

# try:
#     key_attributes = CHOSEN_ATTRIBUTES[wandb.config["key_attributes"]]
# except Exception as _:
#     print("Invalid key_attributes selection.")
#     sys.exit()
# try:
#     assert (wandb.config["num_models"] % (len(key_attributes)) * 2) == 0
    
# except Exception as _:
#     print("Invalid number of models for the selected key attributes. Number of models must be divsible by number of key attributes * 2.")
#     sys.exit()

# dataset = DeepfakeDataset(dataset_root='/home/brg2890/major/preprocessed', attribute_root='/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/attributes', splits_root = "/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets", datasets=wandb.config["datasets"], auto_threshold=False, string_threshold=38)
# model = CNNModel(wandb.config["model"], wandb.config["frames_per_video"], dataset, wandb.config["warp_threshold"], wandb.config["num_models"], wandb.config["steps_per_epoch"], wandb.config["timesteps_before_return_allowed"], wandb.config["train_traversal_method"], wandb.config["hops_to_analyze"], wandb.config["val_test_traversal_method"], key_attributes, ATTRIBUTE_DICT)
# for epoch in tqdm(range(wandb.config["epochs"])):
#     model.train()
#     model.validate()
# model.test()