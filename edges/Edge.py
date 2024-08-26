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

class Edge():
    """
    Base edge class for connecting nodes.
    All edges must have a set of tags to denote what data types/sets they can be used with.
    Bad tags could break things, so please don't do that.
    """
    tags = ["all"]
    
    def __init__(self, node1, node2, x, traversal_weight=1):
        self.node1 = node1
        self.node2 = node2
        self.x = x
        self.traversal_weight = traversal_weight
        
    def set_node1(self, node):
        self.node1 = node
        
    def set_node2(self, node):
        self.node2 = node
        
    def set_nodes(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        
    def get_node1(self):
        return self.node1
    
    def get_node2(self):
        return self.node2
    
    def get_nodes(self):
        return self.node1, self.node2
    
    def set_data(self, x):
        self.x = x
        
    def get_data(self):
        return self.x
    
    def set_traversal_weight(self, w):
        self.traversal_weight = w
        
    def get_traversal_weight(self):
        return traversal_weight
    
    
        