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
from data.Data import Data

class ImageData(Data):
    """
    Takes image data from a file and stores it for usage.
    High RAM overhead, low runtime impact.
    """
    
    tags = [["image", "file"]]
    supported_extensions = ["jpg", "jpeg", "png"]
    
    def __init__(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().__init__(image)
        
    def set_data(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().set_data(image)