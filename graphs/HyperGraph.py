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

"""
Planning/TODO/brainstorming:

Multigraphs need to be level-agnostic: a multigraph functions the same at level 4 as level 400 (except for level 0)
Level 0 multigraphs are NetworkX Graphs that store data directly.

Level 1+ multigraphs could be NetworkX Graphs that store other multigraphs (which are NetworkX Graphs themselves)
or they could be containers which store other multigraphs and NetworkX Graphs (miss out on extra functions, but would those work anyway?)

Any multigraph can be initialized with pointers that move according to some specified method

Pointers can traverse between multigraphs if user-approved



----- How to make L0 networks functional with existing NetworkX infrastructure ---

???

----------------------------------------------------------------------------------


Do leaves need to be different from non-leaves?
One MultiGraph class for all levels, just leave in and genericize the traversal and training functions?
Try and report back
Leaf is just MultiGraph with no subgraphs?
That means each MultiGraph can hold data and/or graphs
2 node types - Graph and Data
That's just two graph types again lol

Solution - Use NetworkX graphs all the way, store edges normally but use string-based edge attributes.
Handle everything else pointer-side: traversal and learning will access edge attributes and treat them differently, even though NetworkX
    sees them the same way
    
Generic dataloader formatted as follows:

0. Get base directory
1. Find all files in base directory except edges.json
2. Load all files as nodes into top-level graph
3. Connect nodes with edges in edges.json if present
4. Find all directories in base directory with same name (-extensions?) as a file in base directory
5. Repeat 1-4 with each new directory, loading result as graph attatched to matching node
6. Once done, you have a single graph where each node is a file + (optional) a graph, and that graph has the same property

Generic trainer/traverser formatted as follows:

0. Get graph created by dataloader
1. Initialize n_0 pointers with random position
2. For each pointer, if it is in a node with a graph, initialize n_1 pointers with random position within that graph (or n_default if not specified)
3. Repeat 2 until graphs all the way down are filled
4. Do training on that node (undefined in base/abstract implementation, needs to be overwritten)
5. Move to adjacent node
6. Repeat 2-5 for num_steps
7. Repeat 1-6 (or 2-6?) with validation
8. Repeat 1-7 (or 2-7?) for num_epochs
9. Repeat 1-6 with testing
(NOTE: Only works for low-n hypergraphs. Will need level-moving pointers or some other solution for high-n graphs due to overhead.)

HyperGraph

init: Get 
basic utility functions: add and remove data, get all or specific data
"""

class HyperGraph(Graph):
    """
    This is an abstract agent-based multigraph dataset class.
    It provides several basic functions for management and traversal of data graphs.
    Remember to  overwrite the indicated functions when subclassing.
    """
    
    def __init__(self, data: list, edges = None):
        # for graph in graphs:
        #     try:
        #         assert (isinstance(graph, MultiGraph) or isinstance(graph, nx.Graph))
        #     except Exception as e:
        #         raise Exception("Only MultiGraph (or subclasses) and NetworkX Graphs (or subclasses) can be used as graphs.")
        self.graph = Graph()
        self.data = graphs
        self.edges = edges
        
    def __len__(self):
        return len(self.graphs)
    
    def get_graph(self, index):
        if index > (len(self.graphs) + 1):
            raise Exception("Invalid index for get_graph.")
        return self.graphs[index]
    
    def get_graphs(self):
        return self.graphs
    
    def set_graph(self, index, graph):
        if index > (len(self.graphs) + 1):
            raise Exception("Invalid index for set_graph.")
        self.graphs[index] = graph
        
    def remove_graph(self, index):
        if index > (len(self.graphs) + 1):
            raise Exception("Invalid index for remove_graph.")
        self.graphs.pop(index)
        
    def add_graph(self, graph):
        self.graphs.append(graph)