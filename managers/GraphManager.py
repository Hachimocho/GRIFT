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
    Abstract class, overwrite for actual usage.
    """
    tags = ["none"]
    # No hyperparameters, since this class should never be used without subclassing.
    hyperparameters = None
    def __init__(self, graph):
        """
        Initialize a GraphManager object.

        Args:
            graph (HyperGraph): The graph to manage.
        """
        self.graph = graph

    def set_graph(self, graph):
        """
        Set the graph managed by the GraphManager object.

        Args:
            graph (HyperGraph): The graph to manage.
        """
        self.graph = graph

    def get_graph(self):
        """
        Get the graph managed by the GraphManager object.

        Returns:
            HyperGraph: The graph managed by the GraphManager object.
        """
        return self.graph
    
    def update_graph(self):
        """
        Update the graph managed by the GraphManager object.
        """
        raise NotImplementedError("Subclass must implement update_graph()")
    