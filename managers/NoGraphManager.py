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
from managers.GraphManager import GraphManager

class NoGraphManager(GraphManager):
    """
    A GraphManager that does nothing. Use for any static environment.
    """
    tags = ["any"]
    hyperparameters = None
    
    def update_graph(self):
        """
        Dummy update function.
        """
        pass
    