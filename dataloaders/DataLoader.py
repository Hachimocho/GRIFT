import torch
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
from os.path import dirname, basename, isfile, join
import glob
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from nodes import *
from edges import *

class DataLoader():
    """
    Takes a Data object and loads it into a HyperGraph.
    """
    
    def __init__(self, data_root):
        assert os.path.isdir(data_root)
        self.data_root = data_root
        



        # Function to get list of classes from a module
        

        # Get lists of available classes
        available_node_types = get_classes_from_module('nodes')
        available_edge_types = get_classes_from_module('edges')

        print("Available node types:", available_node_types)
        print("Available edge types:", available_edge_types)
        nodes = []
        edges = []
        for node_type in available_node_types:
            print("Node type detected: " + node_type)
            nodes.append(globals()[node_type](0))
        for edge_type in available_edge_types:
            print("Edge type detected: " + edge_type)
            edges.append(globals()[edge_type](0))
        self.data = self.__load__()
        
    def __load__(self):
        pass
    
