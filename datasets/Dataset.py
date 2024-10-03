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
from os.path import dirname, basename, isfile, join
import glob
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from nodes import *
from edges import *

class Dataset():
    """
    Takes a data path, node and data classes, and loads the data into node format.
    Every dataset must do the following:
    - Create a list of nodes containing the relevant data
    - Create a list of masks for training, validation, and testing
    - Implement a load() method which loads the data into the nodes and returns the node list
    """
    tags = ["none"]
    hyperparameters = None
    
    def __init__(self, data_root, data_class, data_args, node_class, node_args):
        assert os.path.isdir(data_root)
        self.data_root = data_root
        self.data_class = data_class
        self.node_class = node_class
        self.data_args = data_args
        self.node_args = node_args
        self.nodes = []
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        
    def load(self):
        #self.nodes = ...
        #return self.nodes
        raise NotImplementedError("Overwrite this!")
    
    def __len__(self):
        return len(self.nodes)
