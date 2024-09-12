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

class Node():
    """
    Each node class must have a set of tags which matches what data types and/or datasets it can be used with.
    Invalid tags might cause bad things, so don't do that.
    """ 
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, data, edges: list):
        self.data = data
        self.edges = edges
        
    def match(self, other):
        if isinstance(other, Node):
            return True
        else:
            return False
    
    def __len__(self):
        return len(self.x)
    
    def get_data(self):
        return self.x
    
    def set_data(self, x):
        self.x = x
        
    def get_adjacent_nodes(self):
        adjacent_nodes = []
        for edge in self.edges:
            for node in edge.get_nodes():
                if node != self:
                    adjacent_nodes.append(node)
        return adjacent_nodes
        
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.data == other.data
        else:
            return False
        
    def __hash__(self):
        return hash(self.data)