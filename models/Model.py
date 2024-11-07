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

class Model():
    """
    The base class for hypergraph-compatible models.
    All subclasses must implement the train/val/test methods, as well as providing a
        list of tags used for selecting what kinds of data the model can process.
    Incorrect tagging could lead to unsupported data being fed into the model, so don't do that.
    """
    tags = ["none"]
    
    def __init__(self, save_path):
        self.save_path = save_path
        self.stored_accuracy = []
        self.stored_loss = []
        self.best_acc = 0
    
    def train(self):
        raise NotImplementedError()
    
    def validate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()
    
    def save_checkpoint(self):
        raise NotImplementedError()
    
    def load_checkpoint(self):
        raise NotImplementedError()