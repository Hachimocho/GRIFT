import torch
import torch.nn as nn
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
from models.DeepfakeModel import DeepfakeModel
from torchvision import transforms
import importlib
import wandb
import torchmetrics


class CNNModel(DeepfakeModel):
    tags = ["cnn", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    def __init__(self, model_name, frames_per_video, dataset, rng_threshold, num_pointers, num_steps=100, X=5, traversal_method='random_delay_repeat', K=0, val_test_traversal_method="boring_comprehensive", key_attributes=[], attribute_dict={}):
        super().__init__(dataset, rng_threshold, num_pointers, num_steps, X, traversal_method, K, val_test_traversal_method, key_attributes)
        self.frames_per_video = frames_per_video
        ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
        self.models = [ActiveModel(output_classes=2, classification_strategy='binary') for _ in range(self.num_pointers)]
        for model in self.models:
            model.model.cuda()
        self.loss = nn.BCEWithLogitsLoss()
        self.optims = [torch.optim.Adam(modelout.model.parameters()) for modelout in self.models]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Resize((255, 255)) # Resize to 255x255

        ])
        self.stored_loss = [[] for _ in range(self.num_pointers)]
        self.stored_accuracy = [[] for _ in range(self.num_pointers)]
        self.stored_mode = 'train'
        self.accuracy = torchmetrics.classification.BinaryAccuracy().cuda()
        self.best_acc = [0 for _ in range(self.num_pointers)]
        attribute_set = [[key_attribute_index, key_attribute_value] for key_attribute_index, key_attribute_value in zip([i for i in key_attributes for _ in range(2)], [[0, 2][j % 2] for j in range(len(key_attributes) * 2)])]
        self.model_names = [model_name + "_" + str(i) for i in range(self.num_pointers)] if (len(attribute_dict.items()) == 0) else [model_name + "_" + attribute_dict[attribute_set[i][0]] + "_" + str(i) for i in range(self.num_pointers)]
    def process_node_data(self, node_list, pointer_id, labels, mode):
        #print("New node batch")
        
        # Check if mode changed, and do logging and model swaps if so
        if self.stored_mode != mode:
            for i, loss_array in enumerate(self.stored_loss):
                wandb.log({mode + "_loss_" + str(i): sum(loss_array) / len(loss_array)})
            for i, acc_array in enumerate(self.stored_accuracy):
                wandb.log({mode + "_acc_" + str(i): sum(acc_array) / len(acc_array)})
                # Check to see if model needs to be reloaded
                if mode != "val" and ((sum(acc_array) / len(acc_array)) < self.best_acc[i]):
                    self.models[i].model.load_state_dict(torch.load("/home/brg2890/major/bryce_python_workspace/GraphWork/saved_models/" + self.model_names[i] + ".pt"))
                # Save model if new best acc found
                elif mode != "val" and ((sum(acc_array) / len(acc_array)) >= self.best_acc[i]):
                    self.best_acc[i] = sum(acc_array) / len(acc_array)
                    torch.save(self.models[i].model.state_dict(), "/home/brg2890/major/bryce_python_workspace/GraphWork/saved_models/" + self.model_names[i] + ".pt")
                    
            self.stored_loss = [[] for _ in range(self.num_pointers)]
            self.stored_accuracy = [[] for _ in range(self.num_pointers)]
            
        for i, full_node in enumerate(node_list):
            node, attributes = full_node
            frames = random.choices([x for x in os.listdir(node) if x.endswith(".jpeg")], k=self.frames_per_video)
            batch = [self.transform(cv2.cvtColor(cv2.imread(node + "/" + frame), cv2.COLOR_BGR2RGB)) for frame in frames]
            y = [labels[i] for _ in range(self.frames_per_video)]

            # Train on input data
            if mode == "train":
                for model in self.models:
                    model.train()
                y_hat = self.models[pointer_id](torch.stack(batch).cuda())
                y = torch.tensor(y).unsqueeze(1).cuda()
                loss = self.loss(y_hat, y.float())
                loss.backward()
                self.optims[pointer_id].step()
                self.optims[pointer_id].zero_grad()
                
                
                # update and log
                # self.train_acc.update(y_hat, y)
                # self.train_f1.update(y_hat, y)
                # self.train_auroc.update(y_hat, y)
                # self.log_dict({
                #     "train_loss": loss, "train_acc": self.train_acc,
                #     "train_f1": self.train_f1, "train_auroc": self.train_auroc
                # }, on_epoch=True, on_step=False)  # sync_dist=True on multigpu
                #print(loss)
                
            # Perform validation
            elif mode == "val":
                for model in self.models:
                    model.eval()
                y_hat = self.models[pointer_id](torch.stack(batch).cuda())
                y = torch.tensor(y).unsqueeze(1).cuda()
                loss = self.loss(y_hat, y.float())
           
            # Run testing 
            elif mode == "test":
                for model in self.models:
                    model.eval()
                y_hat = self.models[pointer_id](torch.stack(batch).cuda())
                y = torch.tensor(y).unsqueeze(1).cuda()
                loss = self.loss(y_hat, y.float())
                
                
            # Should never occur due to checks in DeepfakeModel traverse_graph code.
            else:
                raise ValueError("Invalid mode, this should not occur!")
            
            self.stored_loss[pointer_id].append(loss.detach())
            acc = self.accuracy(y_hat, y)
            self.stored_accuracy[pointer_id].append(acc.detach())
            
        self.stored_mode = mode
            
        #print(node_list)
        #sys.exit()
        # print(videos)
        # print(videos[node_list[0]])
        # Load video from memory
        #reader = cv2.VideoCapture(videos[node_data])
        