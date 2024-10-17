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
    Traverses the graph randomly using a single pointer, visiting each node once..
    """
    tags = ["any"]
    hyperparameters: dict | None = None

    def __init__(self, graph):
        """
        Initialize a RandomTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int): The number of steps to take each pointer. If negative, will move pointers indefinitely.
        """
        self.num_pointers = 1
        self.graph = graph
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        self.pointers = [{'current_node': self.graph.get_random_node(), 'visited': set()} for _ in range(self.num_pointers)]
    
    def traverse(self):
        for i, pointer in enumerate(self.pointers):
            unvisited = [node for node in self.graph.nodes if node not in pointer['visited']]
            if unvisited:
                pointer['current_node'] = random.choice(unvisited)
                pointer['visited'].add(pointer['current_node'])
            else:
                raise RuntimeError("All nodes have been visited, cannot continue traversal.")