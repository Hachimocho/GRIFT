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
from data import *

class ImageFileData(Data):
    """
    Takes image data from a file and loads it for usage upon request.
    Low RAM overhead, high runtime impact.
    """
    
    tags = [["image", "file"], "deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    supported_extensions = ["jpg", "jpeg", "png"]
    
    def __init__(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        super().__init__(indata)
        
    def set_data(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        super().set_data(indata)
        
    def load_data(self):
        return cv2.imread(self.data)