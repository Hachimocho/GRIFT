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
from os.path import dirname, basename, isfile, join
import glob
from utils.import_utils import import_classes_from_directory, get_classes_from_module, load_class_from_globals
from nodes import *
from edges import *
from graphs import HyperGraph

class Dataloader():
    """
    Takes a bunch of datasets and loads them into a HyperGraph.
    """
    tags = ["none"]
    
    def load(self, datasets) -> HyperGraph:
        raise NotImplementedError("Overwrite this!")
    
