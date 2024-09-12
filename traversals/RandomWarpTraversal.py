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
from traversals.Traversal import Traversal

class RandomWarpTraversal(Traversal):
    """
    Traverses the graph using randomly moving pointers.
    """
    tags = ["any"]
    
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "warp_chance": {"distribution": "uniform", "min": 0.0, "max": 0.999}
        }
    }

    def __init__(self, graph, num_pointers, num_steps, warp_chance):
        """
        Initialize a RandomTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int): The number of steps to take each pointer. If negative, will move pointers indefinitely.
        """
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.graph = graph
        self.t = 0
        self.warp_chance = warp_chance
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        self.pointers = [{'current_node': self.graph.get_random_node()} for _ in range(self.num_pointers)]
    
    def traverse(self):
        if self.t > self.num_steps:
            raise RuntimeError("Maximum number of steps exceeded.")
        
        self.t += 1
        for i, pointer in enumerate(self.pointers):
            # Get the indices of the adjacent nodes
            adj_nodes = pointer['current_node'].get_adjacent_nodes()

            # If there are no adjacent nodes or the warp triggers,
            # move the pointer to a random node
            if adj_nodes and (random.random() > self.warp_chance):
                # Randomly select an adjacent node
                pointer['current_node'] = random.choice(adj_nodes)
            else:
                pointer['current_node'] = self.graph.get_random_node()

