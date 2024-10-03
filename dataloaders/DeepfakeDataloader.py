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

class DeepfakeDataset(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = {
        "parameters": {
            "datasets": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }

    def load(self):
        # TODO: Add check for disconnected clusters, set local file roots, add support for other datasets
        # Note: In current implementation, main graph may not be equal to the sum of the training/validation/testing graphs. This is because
        #   the edges fixing disconnected nodes need to be rebuilt for each graph, since otherwise a node could be re-disconnected by removal of
        #   the node in a different split that it was connected to. Shouldn't be a big issue, but can look into alternatives later.
        start = time.time()
        node_list = []
        
        # Load datasets
        for dataset in self.datasets:
            node_list.extend(dataset.load())
        
        #Initialize edge index and edge attribute lists
        edge_index = []
        edge_attr = []
        print("# of nodes:")
        print(len(node_list))
        
        matched = 0
        matched_set = []
        unmatched_set = list(range(len(videos)))

        if auto_threshold:
            # Report average matching string vars
            print("Average string vars: " + str(result))

        # Build training graph
        train_nodes_indexes = [i for i, val in enumerate(train_mask) if val]
        train_videos = [item for i, item in enumerate(videos) if i in train_nodes_indexes]
        train_labels = [item for i, item in enumerate(y) if i in train_nodes_indexes]
        # Shuffle graph
        n = len(train_videos)
        order = list(range(n))
        random.shuffle(order)
        train_videos = [train_videos[i] for i in order]
        train_labels = [train_labels[i] for i in order]
        train_edge_attr = []
        train_num_nodes = sum(1 for val in train_mask if val)
        train_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(train_videos)))
        print("# of train videos: ")
        print(len(train_videos))
        
        # Iterate over all pairs of training videos
        for i, j in tqdm(combinations(range(len(train_videos)), 2), total= sum(1 for _ in combinations(range(len(train_videos)), 2)), desc="Building training graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[train_videos[i][0]], attributes[train_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                train_edge_index.append([i, j])
                train_edge_index.append([j, i])
                train_edge_attr.append(num_matching)
                train_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(train_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[train_videos[i][0]], attributes[train_videos[j][0]]))
                train_edge_index.append([i, j])
                train_edge_index.append([j, i])
                train_edge_attr.append(num_matching)
                train_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
                
        # Build validation graph
        val_nodes_indexes = [i for i, val in enumerate(val_mask) if val]
        val_videos = [item for i, item in enumerate(videos) if i in val_nodes_indexes]
        val_labels = [item for i, item in enumerate(y) if i in val_nodes_indexes]
        n = len(val_videos)
        order = list(range(n))
        random.shuffle(order)
        val_videos = [val_videos[i] for i in order]
        val_labels = [val_labels[i] for i in order]
        val_edge_attr = []
        val_num_nodes = sum(1 for val in val_mask if val)
        val_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(val_videos)))
        # Iterate over all pairs of validation videos
        for i, j in tqdm(combinations(range(len(val_videos)), 2), total= sum(1 for _ in combinations(range(len(val_videos)), 2)), desc="Building validation graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[val_videos[i][0]], attributes[val_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                val_edge_index.append([i, j])
                val_edge_index.append([j, i])
                val_edge_attr.append(num_matching)
                val_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(val_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[val_videos[i][0]], attributes[val_videos[j][0]]))
                val_edge_index.append([i, j])
                val_edge_index.append([j, i])
                val_edge_attr.append(num_matching)
                val_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
                
        # Build testing graph
        test_nodes_indexes = [i for i, val in enumerate(test_mask) if val]
        test_videos = [item for i, item in enumerate(videos) if i in test_nodes_indexes]
        test_labels = [item for i, item in enumerate(y) if i in test_nodes_indexes]
        n = len(test_videos)
        order = list(range(n))
        random.shuffle(order)
        test_videos = [test_videos[i] for i in order]
        test_labels = [test_labels[i] for i in order]
        test_edge_attr = []
        test_num_nodes = sum(1 for val in test_mask if val)
        test_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(test_videos)))
        # Iterate over all pairs of testing videos
        for i, j in tqdm(combinations(range(len(test_videos)), 2), total= sum(1 for _ in combinations(range(len(test_videos)), 2)), desc="Building testing graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[test_videos[i][0]], attributes[test_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                test_edge_index.append([i, j])
                test_edge_index.append([j, i])
                test_edge_attr.append(num_matching)
                test_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(test_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[test_videos[i][0]], attributes[test_videos[j][0]]))
                test_edge_index.append([i, j])
                test_edge_index.append([j, i])
                test_edge_attr.append(num_matching)
                test_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
        
        
        # Convert lists to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create a PyTorch Geometric graph (currently deprecated due to splitting into train, val, test graphs)
        # data = Data(x=videos, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, num_nodes=len(videos)) 
        
        # Do data shuffling
        # n = len(train_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # train_videos = [train_videos[i] for i in order]
        # train_labels = [train_labels[i] for i in order]
        # train_edge_index = [train_edge_index[i] for i in order]
        # train_edge_attr = [train_edge_attr[i] for i in order]
        
        # n = len(val_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # val_videos = [val_videos[i] for i in order]
        # val_edge_index = [val_edge_index[i] for i in order]
        # val_edge_attr = [val_edge_attr[i] for i in order]
        # val_labels = [val_labels[i] for i in order]
        
        # n = len(test_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # test_videos = [test_videos[i] for i in order]
        # test_edge_index = [test_edge_index[i] for i in order]
        # test_edge_attr = [test_edge_attr[i] for i in order]
        # test_labels = [test_labels[i] for i in order]
        
        data_train = Data(x=train_videos, y=train_labels, edge_index=torch.tensor(train_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(train_edge_attr, dtype=torch.float), num_nodes = train_num_nodes)
        
        data_val = Data(x=val_videos, y=val_labels, edge_index=torch.tensor(val_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(val_edge_attr, dtype=torch.float), num_nodes = val_num_nodes)
        
        data_test = Data(x=test_videos, y=test_labels, edge_index=torch.tensor(test_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(test_edge_attr, dtype=torch.float), num_nodes = test_num_nodes)
        
        for data in [data_train, data_val, data_test]:
            if data.has_isolated_nodes():
                raise ValueError("Invalid graph: isolated nodes detected.")
            if data.is_directed():
                raise ValueError("Invalid graph: directionality present.")
        
        print("Graph generation finished in " + str(time.time() - start) + "s.")
        
        return data, data_train, data_val, data_test, videos, train_mask, val_mask, test_mask

if __name__ == "__main__":
    graph, encoder = DeepfakeDataset().generate_graph(file_root='/home/brg2890/major/preprocessed/FaceForensics++_All/FaceForensics++_Graph', attribute_root='/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/A-FF++.csv', threshold=38)

    print(graph.num_nodes)


    print(graph.num_edges)


    print(graph.num_node_features)


    print(graph.has_isolated_nodes())


    print(graph.has_self_loops())


    print(graph.is_directed())

    # Call the function with your graph and processing function
    traverse_graph(graph, process_node_data, X=1, rng_threshold=0.01, num_pointers=3)
    
    #print('edge_attr' in graph)

    print(graph.num_nodes)


    print(graph.num_edges)


    print(graph.num_node_features)


    print(graph.has_isolated_nodes())


    print(graph.has_self_loops())


    print(graph.is_directed())

    # Transfer data object to GPU.
    device = torch.device('cuda')
    graph = graph.to(device)
    print("Cuda test finished.")
