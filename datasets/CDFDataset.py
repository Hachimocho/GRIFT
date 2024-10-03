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
from datasets.Dataset import Dataset

class CDFDataset(Dataset):
    tags = ["deepfakes"]
    hyperparameters = None

    def load(self):
        data_root = self.data_root
        
        for class_folder in ["fake", "real"]:
            if class_folder == "fake":
                sub_folders = ["Celeb-synthesis"]
            elif class_folder == "real":
                sub_folders = ["Celeb-real", "YouTube-real"]
            for sub_folder in sub_folders:
                for label_folder in ["train", "val", "test"]:
                    folder_path = os.path.join(data_root, class_folder, sub_folder, label_folder)
                    for video in os.listdir(folder_path):
                        video_path = os.path.join(folder_path, video)
                        self.nodes.append(self.node_class(label_folder, self.data_class(video_path, **self.data_args), [], 0 if class_folder == "real" else 1, **self.node_args))
        
        return self.nodes