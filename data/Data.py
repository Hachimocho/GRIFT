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

class Data():
    """
    Takes data from some source and converts it into a DataLoader-compatible format.
    Mostly in place so that wonky data type conversions can be implemented.
    Must have a tags attribute so other modules can define compatability with it.
    """
    
    self.tags = ["all"]
    
    def __init__(self, indata):
        self.data = indata
        
    def load_data(self):
        return self.data
    
    def set_data(self, indata):
        self.data = indata