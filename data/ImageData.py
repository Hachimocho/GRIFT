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

class ImageData(Data):
    """
    Takes image data from a file and stores it for usage.
    High RAM overhead, low runtime impact.
    """
    
    self.tags = [["image", "file"]]
    self.supported_extensions = ["jpg", "jpeg", "png"]
    
    def __init__(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().__init__(image)
        
    def set_data(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().set_data(image)