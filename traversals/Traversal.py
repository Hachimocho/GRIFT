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

class Traversal():
    """
    Abstract class, defines a method through which pointers located on nodes can move to other nodes in the graph.
    """
    tags = ["none"]
    # No hyperparameters, since this class should never be used without subclassing.
    
    def traverse(self, graph):
        raise NotImplementedError("Subclass must implement traverse()")
    
    def get_pointers(self):
        raise NotImplementedError("Subclass must implement get_pointers()")
    
    def reset_pointers(self):
        raise NotImplementedError("Subclass must implement reset_pointers()")