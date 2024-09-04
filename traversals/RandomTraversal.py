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
from traversals.Traversal import Traversal

class RandomTraversal(Traversal):
    """
    Abstract class, defines a method through which pointers located on nodes can move to other nodes in the graph.
    """
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }

    def __init__(self, graph, num_pointers, num_steps):
        """
        Initialize a RandomTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int): The number of steps to take each pointer. If negative, will move pointers indefinitely.
        """
        self.pointers = [{'current_node': graph.get_random_node()} for _ in range(num_pointers)]
        self.num_steps = num_steps
        self.graph = graph
    
    def traverse(self):
        t = 0
        while(t < self.num_steps if self.num_steps > 0 else True):
            t += 1
            for i, pointer in enumerate(self.pointers):
                current_node = pointer['current_node']

                # Get the indices of the adjacent nodes
                adj_nodes = data.edge_index[1][data.edge_index[0] == current_node].tolist()

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

