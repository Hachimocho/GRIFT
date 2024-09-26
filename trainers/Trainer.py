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
from graphs.HyperGraph import HyperGraph
from models.CNNModel import CNNModel
from datasets.DeepfakeDataset import DeepfakeDataset
import wandb
from utils.import_utils import import_classes_from_directory, get_classes_from_module
from datasets import *
from dataloaders import *
from managers import *

class Trainer():
    """
    Base class for pointer/agent based traversal and training on Hypergraphs.
    """
    tags = ["none"]
    hyperparameters = None
    def __init__(self, graphmanager, model, traversal, test_traversal):
        self.graphmanager = graphmanager
        self.model = model
        self.traversal = traversal
        self.test_traversal = test_traversal
        
    def run(self):
        print("Running trainer.")
        t = time.time()
        best_acc = 0
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            avg_train_acc, train_loss = self.train()
            avg_val_acc, val_loss = self.val()
            self.graphmanager.update_graph()
            wandb.log({"epoch": epoch, "train_acc": avg_train_acc, "val_acc": avg_val_acc})
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                self.model.save_checkpoint()
            else:
                self.model.load_checkpoint()
        wandb.log({"best_acc": best_acc})
        wandb.log({"time": time.time() - t})
        
    def test_run(self):
        print("Test run!")
        t = time.time()
        best_acc = 0
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            avg_train_acc, train_loss = self.train()
            avg_val_acc, val_loss = self.val()
            self.graphmanager.update_graph()
            wandb.log({"epoch": epoch, "train_acc": avg_train_acc, "val_acc": avg_val_acc})
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                self.model.save_checkpoint()
            else:
                self.model.load_checkpoint()
        avg_test_acc = self.test()
        wandb.log({"test_acc": avg_test_acc})
        wandb.log({"time": time.time() - t})

    def process_node_data(self):
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        raise NotImplementedError("Overwrite this!")
        
    def val(self):
        raise NotImplementedError("Overwrite this!")
    
    def test(self):
        raise NotImplementedError("Overwrite this!")