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
from nodes.Node import Node

class AttributeNode(Node):
    # Node tags:
    tags = ["attributes", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, path, attributes, labels, threshold):
        self.x = path
        assert len(attributes) == len(labels)
        self.attributes = {}
        for atr, label in zip(attributes, labels):
            self.attributes[label] = atr
        assert (type(threshold) == int) and (threshold >= 0)
        self.threshold = threshold
        
    def match(self, other: Node):
        # Currently uses threshold averaging if there is a dispute
        if type(Node) != AttributeNode:
            raise ValueError("Matches only supported between AttributeNodes.")
        if self.threshold != other.threshold:
            threshold = round((self.threshold + other.threshold) / 2)
        else:
            threshold = self.threshold
        matching = 0
        for label, attribute in self.attributes:
            if (label in other.attributes.keys()) and (other.attributes[label] == attribute):
                matching += 1
        if matching >= threshold:
            return True
        else:
            return False
        
    def __len__(self):
        return len(self.attributes)
    
    def add_attribute(self, attribute, label):
        self.attributes[label] = attribute
        
    def remove_attribute(self, label):
        if label in self.attributes.keys():
            del self.attributes[label]
        else:
            raise ValueError("Cannot remove nonexistent attribute.")
    
    
        
        