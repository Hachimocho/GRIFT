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
from dataloaders.Dataloader import Dataloader

class UnclusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = None

    def load(self):
        # TODO: Add check for disconnected clusters, set local file roots, add support for other datasets
        start = time.time()
        node_list = []
        
        # Load datasets
        for dataset in self.datasets:
            node_list.extend(dataset.load())
        
        print("# of nodes:")
        print(len(node_list))
        
        matched = 0
        matched_set = []
        unmatched_set = list(range(len(node_list)))
        
        # Iterate over all pairs of nodes
        for i, j in tqdm(combinations(range(len(node_list)), 2), total= sum(1 for _ in combinations(range(len(node_list)), 2)), desc="Building graph..."):
            if node_list[i].match(node_list[j]):
                matched += 1
                edge = self.edge_class(node_list[i], node_list[j])
                node_list[i].add_edge(edge)
                node_list[j].add_edge(edge)
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(node_list)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                edge = self.edge_class(node_list[i], node_list[j])
                node_list[i].add_edge(edge)
                node_list[j].add_edge(edge)
                unmatched_set.remove(i)
                matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                matched += 1
        
        graph = HyperGraph(node_list)
        print("Graph generation finished in " + str(time.time() - start) + "s.")
        
        return graph