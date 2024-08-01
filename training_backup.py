import torch
from torch_geometric.data import Data
import random
from torch_geometric.utils import k_hop_subgraph, to_undirected
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
from torch_geometric.utils.convert import to_networkx
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

from models.CNNModel import CNNModel
from DeepfakeDataset import DeepfakeDataset
import wandb

# Set random seed for consistent results (need to test)
random.seed(785134785632495632479853246798)

# Label key (0 for not present, 1 for unknown, 2 for present):
# male,young,middle_aged,senior,asian,white,black,shiny_skin,bald,wavy_hair,receding_hairline,bangs,black_hair,blond_hair,gray_hair,no_beard,mustache,goatee,oval_face,square_face,double_chain,chubby,obstructed_forehead,fully_visible_forehead,brown_eyes,bags_under_eyes,bushy_eyebrows,arched_eyebrows,mouth_closed,smiling,big_lips,big_nose,pointy_nose,heavy_makeup,wearing_hat,wearing_necktie,wearing_lipstick,no_eyewear,eyeglasses,attractive
ATTRIBUTES = "male,young,middle_aged,senior,asian,white,black,shiny_skin,bald,wavy_hair,receding_hairline,bangs,black_hair,blond_hair,gray_hair,no_beard,mustache,goatee,oval_face,square_face,double_chain,chubby,obstructed_forehead,fully_visible_forehead,brown_eyes,bags_under_eyes,bushy_eyebrows,arched_eyebrows,mouth_closed,smiling,big_lips,big_nose,pointy_nose,heavy_makeup,wearing_hat,wearing_necktie,wearing_lipstick,no_eyewear,eyeglasses,attractive".split(',')

ATTRIBUTE_DICT = {i : val for i, val in enumerate(ATTRIBUTES)}

CHOSEN_ATTRIBUTES = {
    "race": [ATTRIBUTES.index("white"), ATTRIBUTES.index("black"), ATTRIBUTES.index("asian")],
    "gender": [ATTRIBUTES.index("male")],
    "age": [ATTRIBUTES.index("young"), ATTRIBUTES.index("middle_aged"), ATTRIBUTES.index("senior")],
    "none": []
}

# W&B setup
wandb.login(key="8e2ea87ef9c3afd51f009eaf850d7b22e935fa1e")

wandb.init(
    # set the wandb project where this run will be logged
    project="DeepEARL",

    # track hyperparameters and run metadata
    config={
    "model": "resnestdf_all",
    "learning_rate": 0.001,
    "epochs": 5,
    "frames_per_video": 15,
    "warp_threshold": 0.01,
    "num_models": 2,
    "steps_per_epoch": 100,
    "timesteps_before_return_allowed": 3,
    "hops_to_analyze": 0,
    "train_traversal_method": "attribute_delay_repeat",
    "val_test_traversal_method": "attribute_boring_comprehensive",
    "key_attributes": "gender",
    "datasets": ["CDF"]
    }
)
# ["FF", "DFD", "DF1", "CDF"]
# Choose key attributes
try:
    key_attributes = CHOSEN_ATTRIBUTES[wandb.config["key_attributes"]]
except Exception as _:
    print("Invalid key_attributes selection.")
    sys.exit()
try:
    assert (wandb.config["num_models"] % (len(key_attributes)) * 2) == 0
except Exception as _:
    print("Invalid number of models for the selected key attributes. Number of models must be divsible by number of key attributes * 2.")
    sys.exit()

dataset = DeepfakeDataset(dataset_root='/home/brg2890/major/preprocessed', attribute_root='/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/attributes', splits_root = "/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets", datasets=wandb.config["datasets"], auto_threshold=False, string_threshold=38)
model = CNNModel(wandb.config["model"], wandb.config["frames_per_video"], dataset, wandb.config["warp_threshold"], wandb.config["num_models"], wandb.config["steps_per_epoch"], wandb.config["timesteps_before_return_allowed"], wandb.config["train_traversal_method"], wandb.config["hops_to_analyze"], wandb.config["val_test_traversal_method"], key_attributes, ATTRIBUTE_DICT)
for epoch in tqdm(range(wandb.config["epochs"])):
    model.train()
    model.validate()
model.test()