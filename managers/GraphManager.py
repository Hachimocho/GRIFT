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
from graphs.HyperGraph import HyperGraph

class GraphManager(HyperGraph):
    """
    Defines a HyperGraph which changes over time. May use traversals, model performance, or just time to adjust the graph.
    Most commonly used to represent a specific environment or to perform automatic data augmentation.
    """
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }