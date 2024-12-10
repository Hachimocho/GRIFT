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

class ComprehensiveTraversal(Traversal):
    """
    Traverses the graph randomly using a single pointer, visiting each node once.
    """
    tags = ["any"]
    hyperparameters: dict | None = None

    def __init__(self, graph, num_pointers, num_steps=None):
        """
        Initialize a ComprehensiveTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int, optional): Maximum number of nodes to visit. If None, visits all nodes.
        """
        self.num_pointers = num_pointers
        self.graph = graph
        self.num_steps = num_steps if num_steps is not None else len(graph.get_nodes())
        if num_steps is not None:
            self.test_mode = False
        else:
            self.test_mode = True
        self.steps_taken = 0
        self.reset_pointers()
    
    def __len__(self):
        """Return the target number of nodes to visit."""
        return min(self.num_steps, len(self.graph.get_nodes()))
    
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        """Reset traversal state, including pointers, visited sets, and steps counter."""
        self.pointers = [{'current_node': self.graph.get_random_node(), 'visited': set()} for _ in range(self.num_pointers)]
        self.steps_taken = 0  # Reset steps counter
    
    def traverse(self, batch_size=32):
        """
        Traverse the graph and return the next batch of nodes.
        
        Args:
            batch_size (int): Number of nodes to return per batch
            
        Returns:
            list: List of nodes or empty list if all nodes have been visited or max steps reached
        """
        if self.steps_taken >= self.num_steps:
            return []
            
        # Get all unvisited nodes across all pointers
        all_unvisited = set()
        for pointer in self.pointers:
            unvisited = set(node for node in self.graph.nodes if node not in pointer['visited'])
            all_unvisited.update(unvisited)
        
        if not all_unvisited:
            return []
            
        # Select up to batch_size nodes randomly from unvisited set
        nodes_to_visit = min(batch_size, len(all_unvisited))
        
        if nodes_to_visit > 0:
            
            if self.test_mode:
                # Convert to list and sort by node ID for deterministic order during testing
                unvisited_list = sorted(list(all_unvisited), key=lambda x: id(x))
                batch_nodes = unvisited_list[:nodes_to_visit]
            else:
                # Convert to list and randomly sample
                unvisited_list = list(all_unvisited)
                batch_nodes = random.sample(unvisited_list, nodes_to_visit)
            
            # Mark nodes as visited in all pointers
            for node in batch_nodes:
                for pointer in self.pointers:
                    pointer['visited'].add(node)
                    pointer['current_node'] = node
            
            self.steps_taken += nodes_to_visit
            return batch_nodes
            
        return []