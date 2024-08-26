import torch
from torch_geometric.data import Data
import random
from torch_geometric.utils import k_hop_subgraph, to_undirected
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

class Model():
    """
    The base class for hypergraph-compatible models.
    All subclasses must implement the train/val/test methods, as well as providing a
        list of tags used for selecting what kinds of data the model can process.
    Incorrect tagging could lead to unsupported data being fed into the model, so don't do that.
    """
    tags = ["none"]
    
    def train(self, batch):
        raise NotImplementedError()
    
    def validate(self, batch):
        raise NotImplementedError()
    
    def test(self, batch):
        raise NotImplementedError()