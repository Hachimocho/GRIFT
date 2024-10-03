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
from nodes.Node import Node

class RandomNode(Node):
    """
    Each node class must have a set of tags which matches what data types and/or datasets it can be used with.
    Invalid tags might cause bad things, so don't do that.
    """ 
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "match_chance": {"distribution": "uniform", "min": 0, "max": 1}
        }
    }
    
    def __init__(self, split, data, edges, label, match_chance):
        self.split = split
        self.data = data
        self.edges = edges
        self.label = label
        self.match_chance = match_chance
        
    def match(self, other):
        if isinstance(other, Node):
            if random.random() <= self.match_chance:
                return True
        return False