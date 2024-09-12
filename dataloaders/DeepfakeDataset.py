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

class DeepfakeDataset():
    tags = ["deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    def __init__(self, dataset_root, attribute_root, splits_root, datasets, auto_threshold=False, string_threshold=30):
        self.datasets = datasets
        self.string_threshold = string_threshold
        self.graph, self.train_graph, self.val_graph, self.test_graph, self.videos, self.train_mask, self.val_mask, self.test_mask = self.generate_graph(dataset_root, attribute_root, splits_root, string_threshold, auto_threshold)
        
    def __len__(self):
        return len(self.videos)
    
    @staticmethod
    def calculate_running_average():
        total = 0.0
        count = 0
        average = 0.0

        while True:
            number = yield average
            count += 1
            total += number
            average = total / count

    @staticmethod
    def most_common_numbers(arr):
        # Initialize a list of Counters, one for each index
        counters = [Counter() for _ in range(len(arr[0]))]

        # Count the occurrences of each number at each index
        for s in arr:
            for i, num in enumerate(s):
                counters[i][num] += 1

        # Find the most common number at each index
        result = [counter.most_common(1)[0][0] for counter in counters]

        # Return the result as a string
        return ''.join(result)

    def generate_graph(self, file_root, attribute_root, splits_root, threshold, auto_threshold):
        # TODO: Add check for disconnected clusters, set local file roots, add support for other datasets
        # Note: In current implementation, main graph may not be equal to the sum of the training/validation/testing graphs. This is because
        #   the edges fixing disconnected nodes need to be rebuilt for each graph, since otherwise a node could be re-disconnected by removal of
        #   the node in a different split that it was connected to. Shouldn't be a big issue, but can look into alternatives later.
        start = time.time()
        attributes = {}
        videos = []
        y = []
        train_mask = []
        val_mask = []
        test_mask = []
        if "FF" in self.datasets:
            # Load train, val, test splits
            # TODO: Figure out why relative pathing isn't working here during testing
            FF_TRAIN_SPLIT = splits_root + "/faceforensics/train.json"
            FF_VAL_SPLIT = splits_root + "/faceforensics/val.json"
            FF_TEST_SPLIT = splits_root + "/faceforensics/test.json"

            with open(FF_TRAIN_SPLIT) as split:
                train_splits = list(json.load(split))
            with open(FF_VAL_SPLIT) as split:
                val_splits = list(json.load(split))
            with open(FF_TEST_SPLIT) as split:
                test_splits = list(json.load(split))
                
            with open(attribute_root + "/A-FF++.csv", newline='') as csvfile:
                i = 0
                attribute_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                video = ''
                temp_attributes = []
                for row in tqdm(attribute_reader, desc="Loading attribute file..."):
                    if i == 0:
                        i += 1
                        continue
                    filepath = row[1].replace('FaceForensics++', file_root).replace('manipulated_sequences', 'fake').replace('original_sequences', 'real').replace('/raw/face_images', '').replace('frame', '0-').replace('.png', '.jpeg')
                    # Fix for slightly different folder name formatting in Deepfakes
                    if filepath.count("Deepfakes") > 0:
                        filepath = '/' + '/'.join(filepath.split('/')[1:-2]) + "/" + filepath.split('/')[-2] + "/" + filepath.split('/')[-1]
                    else:
                        filepath = '/' + '/'.join(filepath.split('/')[1:-2]) + "/" + filepath.split('/')[-2] + ".mp4/" + filepath.split('/')[-1]
                    # Check to make sure preprocessed video exists, may not due to poor lighting or angles causing detectors to fail
                    if os.path.isfile(filepath):
                        new_video = filepath.split('/')[-3] + "/" + filepath.split('/')[-2]
                        if (video != '') and (video != new_video):
                            # Update train/val/test masks
                            if "youtube" in video:
                                if any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in train_splits):
                                    train_mask.append(True)
                                    val_mask.append(False)
                                    test_mask.append(False)
                                elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in val_splits):
                                    train_mask.append(False)
                                    val_mask.append(True)
                                    test_mask.append(False)
                                elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in test_splits):
                                    train_mask.append(False)
                                    val_mask.append(False)
                                    test_mask.append(True)
                                else:
                                    raise OSError("PANIC")
                            else:
                                if (video.split('/')[-1].split('.')[0].split('_') in train_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in train_splits):
                                    train_mask.append(True)
                                    val_mask.append(False)
                                    test_mask.append(False)
                                elif (video.split('/')[-1].split('.')[0].split('_') in val_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in val_splits):
                                    train_mask.append(False)
                                    val_mask.append(True)
                                    test_mask.append(False)
                                elif (video.split('/')[-1].split('.')[0].split('_') in test_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in test_splits):
                                    train_mask.append(False)
                                    val_mask.append(False)
                                    test_mask.append(True)
                                else:
                                    raise OSError("PANIC")
                            
                            # Update labels list
                            if video.split('/')[0].lower() == "youtube":
                                y.append(0)
                            else:
                                y.append(1)
                                
                            # Update video attributes and videos lists
                            attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                            videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                            temp_attributes = []
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                            
                        elif ((len(temp_attributes) > 0) and (video == new_video)) or (video == ''):
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                        
                        else:
                            print("Attribute error.")
                            print(video)
                            print(new_video)
                            print(temp_attributes)
                            sys.exit()
                
                # Complete attributes for last video               
                if len(temp_attributes) > 0:
                    attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                    videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                    if video.split('/')[0].lower() == "youtube":
                        y.append(0)
                    else:
                        y.append(1)
                    if "youtube" in video:
                        if any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in train_splits):
                            train_mask.append(True)
                            val_mask.append(False)
                            test_mask.append(False)
                        elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in val_splits):
                            train_mask.append(False)
                            val_mask.append(True)
                            test_mask.append(False)
                        elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in test_splits):
                            train_mask.append(False)
                            val_mask.append(False)
                            test_mask.append(True)
                        else:
                            raise OSError("PANIC")
                    else:
                        if (video.split('/')[-1].split('.')[0].split('_') in train_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in train_splits):
                            train_mask.append(True)
                            val_mask.append(False)
                            test_mask.append(False)
                        elif (video.split('/')[-1].split('.')[0].split('_') in val_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in val_splits):
                            train_mask.append(False)
                            val_mask.append(True)
                            test_mask.append(False)
                        elif (video.split('/')[-1].split('.')[0].split('_') in test_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in test_splits):
                            train_mask.append(False)
                            val_mask.append(False)
                            test_mask.append(True)
                        else:
                            raise OSError("PANIC")
         
        if "DFD" in self.datasets:
            # Load splits
            # TODO: Figure out why relative pathing isn't working here during testing
            DFD_TRAIN_SPLIT = splits_root + "/DFD/train.json"
            DFD_VAL_SPLIT = splits_root + "/DFD/val.json"
            DFD_TEST_SPLIT = splits_root + "/DFD/test.json"

            with open(DFD_TRAIN_SPLIT) as split:
                train_splits = list(json.load(split))
            with open(DFD_VAL_SPLIT) as split:
                val_splits = list(json.load(split))
            with open(DFD_TEST_SPLIT) as split:
                test_splits = list(json.load(split))
                
            with open(attribute_root + "/A-DFD.csv", newline='') as csvfile:
                i = 0
                attribute_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                video = ''
                temp_attributes = []
                for row in tqdm(attribute_reader, desc="Loading attribute file..."):
                    if i == 0:
                        i += 1
                        continue
                    filepath = row[1].replace('FaceForensics++', file_root + "/DeepFakeDetection_Graph").replace('manipulated_sequences', 'fake').replace('original_sequences', 'real').replace('/raw/face_images', '').replace("/DeepFakeDetection", '').replace('frame', '0-').replace('.png', '.jpeg')
                    # Fix for slightly different folder name formatting in Deepfakes
                    # if filepath.count("Deepfakes") > 0:
                    #     filepath = '/' + '/'.join(filepath.split('/')[1:-2]) + "/" + filepath.split('/')[-2] + "/" + filepath.split('/')[-1]
                    # else:
                    filepath = '/' + '/'.join(filepath.split('/')[1:-2]) + "/" + filepath.split('/')[-2] + ".mp4/" + filepath.split('/')[-1]
                    # Check to make sure preprocessed video exists, may not due to poor lighting or angles causing detectors to fail
                    if os.path.isfile(filepath):
                        new_video = filepath.split('/')[-3] + "/" + filepath.split('/')[-2]
                        if (video != '') and (video != new_video):
                            # Update train/val/test masks
                            if "actors" in video:
                                if any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in train_splits):
                                    train_mask.append(True)
                                    val_mask.append(False)
                                    test_mask.append(False)
                                elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in val_splits):
                                    train_mask.append(False)
                                    val_mask.append(True)
                                    test_mask.append(False)
                                elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in test_splits):
                                    train_mask.append(False)
                                    val_mask.append(False)
                                    test_mask.append(True)
                                else:
                                    raise OSError("PANIC")
                            else:
                                if (video.split('/')[-1].split('.')[0].split('_') in train_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in train_splits):
                                    train_mask.append(True)
                                    val_mask.append(False)
                                    test_mask.append(False)
                                elif (video.split('/')[-1].split('.')[0].split('_') in val_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in val_splits):
                                    train_mask.append(False)
                                    val_mask.append(True)
                                    test_mask.append(False)
                                elif (video.split('/')[-1].split('.')[0].split('_') in test_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in test_splits):
                                    train_mask.append(False)
                                    val_mask.append(False)
                                    test_mask.append(True)
                                else:
                                    raise OSError("PANIC")
                            
                            # Update labels list
                            if video.split('/')[0].lower() == "youtube":
                                y.append(0)
                            else:
                                y.append(1)
                                
                            # Update video attributes and videos lists
                            attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                            videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                            temp_attributes = []
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                            
                        elif ((len(temp_attributes) > 0) and (video == new_video)) or (video == ''):
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                        
                        else:
                            print("Attribute error.")
                            print(video)
                            print(new_video)
                            print(temp_attributes)
                            sys.exit()
                
                # Complete attributes for last video               
                if len(temp_attributes) > 0:
                    attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                    videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                    if video.split('/')[0].lower() == "youtube":
                        y.append(0)
                    else:
                        y.append(1)
                    if "youtube" in video:
                        if any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in train_splits):
                            train_mask.append(True)
                            val_mask.append(False)
                            test_mask.append(False)
                        elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in val_splits):
                            train_mask.append(False)
                            val_mask.append(True)
                            test_mask.append(False)
                        elif any(video.split('/')[-1].split('.')[0] in sub_array for sub_array in test_splits):
                            train_mask.append(False)
                            val_mask.append(False)
                            test_mask.append(True)
                        else:
                            raise OSError("PANIC")
                    else:
                        if (video.split('/')[-1].split('.')[0].split('_') in train_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in train_splits):
                            train_mask.append(True)
                            val_mask.append(False)
                            test_mask.append(False)
                        elif (video.split('/')[-1].split('.')[0].split('_') in val_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in val_splits):
                            train_mask.append(False)
                            val_mask.append(True)
                            test_mask.append(False)
                        elif (video.split('/')[-1].split('.')[0].split('_') in test_splits) or (list(video.split('/')[-1].split('.')[0].split('_').__reversed__()) in test_splits):
                            train_mask.append(False)
                            val_mask.append(False)
                            test_mask.append(True)
                        else:
                            raise OSError("PANIC")
       
        if "DFDC" in self.datasets:
            raise ValueError("DFDC not supported due to unusable annotations.")
        
        if "CDF" in self.datasets:
            # Load train, val, test splits
            # TODO: Figure out why relative pathing isn't working here during testing
            CDF_TRAIN_SPLIT = splits_root + "/celebdf/train.json"
            CDF_VAL_SPLIT = splits_root + "/celebdf/val.json"
            CDF_TEST_SPLIT = splits_root + "/celebdf/test.json"

            with open(CDF_TRAIN_SPLIT) as split:
                train_splits = list(json.load(split))
            with open(CDF_VAL_SPLIT) as split:
                val_splits = list(json.load(split))
            with open(CDF_TEST_SPLIT) as split:
                test_splits = list(json.load(split))
                
            with open(attribute_root + "/A-Celeb-DF.csv", newline='') as csvfile:
                i = 0
                attribute_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                video = ''
                temp_attributes = []
                for row in tqdm(attribute_reader, desc="Loading attribute file..."):
                    if i == 0:
                        i += 1
                        continue
                    filepath = row[1].replace('Celeb-DF-v2_faces_single', file_root + "/CelebDF_Graph").replace('frame', '0-').replace('.png', '.jpeg')
                    # Fix for slightly different folder name formatting in Deepfakes
                    filepath = '/' + '/'.join(filepath.split('/')[1:-2]) + "/" + filepath.split('/')[-2] + ".mp4/" + filepath.split('/')[-1]
                    # Check to make sure preprocessed video exists, may not due to poor lighting or angles causing detectors to fail
                    if os.path.isfile(filepath):
                        new_video = filepath.split('/')[-3] + "/" + filepath.split('/')[-2]
                        if (video != '') and (video != new_video):
                            # Update train/val/test masks
                            if (video in train_splits):
                                train_mask.append(True)
                                val_mask.append(False)
                                test_mask.append(False)
                            elif (video in val_splits):
                                train_mask.append(False)
                                val_mask.append(True)
                                test_mask.append(False)
                            elif (video in test_splits):
                                train_mask.append(False)
                                val_mask.append(False)
                                test_mask.append(True)
                            else:
                                raise OSError("PANIC")
                            
                            # Update labels list
                            if video.split('/')[0].split('-')[-1] == "real":
                                y.append(0)
                            else:
                                y.append(1)
                                
                            # Update video attributes and videos lists
                            attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                            videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                            temp_attributes = []
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                            
                        elif ((len(temp_attributes) > 0) and (video == new_video)) or (video == ''):
                            temp_attributes.append("".join(str(int(attribute) + 1) for attribute in row[2:]))
                            video = new_video
                        
                        else:
                            print("Attribute error.")
                            print(video)
                            print(new_video)
                            print(temp_attributes)
                            sys.exit()
                            
                    #sys.exit()
                
                # Complete attributes for last video               
                if len(temp_attributes) > 0:
                    attributes['/'.join(filepath.split('/')[0:-1])] = self.most_common_numbers(temp_attributes)
                    videos.append(['/'.join(filepath.split('/')[0:-1]), self.most_common_numbers(temp_attributes)])
                    if (video in train_splits):
                        train_mask.append(True)
                        val_mask.append(False)
                        test_mask.append(False)
                    elif (video in val_splits):
                        train_mask.append(False)
                        val_mask.append(True)
                        test_mask.append(False)
                    elif (video in test_splits):
                        train_mask.append(False)
                        val_mask.append(False)
                        test_mask.append(True)
                    else:
                        raise OSError("PANIC")
                    
                    # Update labels list
                    if video.split('/')[0].split('-')[-1] == "real":
                        y.append(0)
                    else:
                        y.append(1)
        
        if "DF1" in self.datasets:
            pass
                
        #Initialize edge index and edge attribute lists
        edge_index = []
        edge_attr = []
        print("# of videos:")
        print(len(videos))
        
        # Start continuous generator for calculating average matching string vars
        avg = self.calculate_running_average()
        next(avg)  # This is necessary to start the generator
        
        # Label encoding is ridiculously slow, no thx
        # le = preprocessing.LabelEncoder()
        # le.fit(videos)
        matched = 0
        matched_set = []
        unmatched_set = list(range(len(videos)))
        
        # Set auto threshold if enabled
        if auto_threshold:
            for i, j in tqdm(combinations(range(len(videos)), 2), total= sum(1 for _ in combinations(range(len(videos)), 2)), desc="Calculating automatic edge threshold..."):
                # Calculate the number of matching variables
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[videos[i][0]], attributes[videos[j][0]]))
                
                # Update average
                result = avg.send(num_matching)
            threshold = round(result)
        
        # # Iterate over all pairs of videos (deprecated due to splitting into train, val, and test graphs)
        # for i, j in tqdm(combinations(range(len(videos)), 2), total= sum(1 for _ in combinations(range(len(videos)), 2)), desc="Building initial graph..."):
        #     # Calculate the number of matching variables
        #     num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[videos[i]], attributes[videos[j]]))
            
        #     # If the number of matching variables is above a certain threshold, add an edge
        #     if num_matching > threshold:
        #         matched += 1
        #         edge_index.append([i, j])
        #         edge_index.append([j, i])
        #         edge_attr.append(num_matching)
        #         edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
        #         # Update arrays of matched and unmatched nodes
        #         if i in unmatched_set:
        #             unmatched_set.remove(i)
        #             matched_set.append(i)
        #         if j in unmatched_set:
        #             unmatched_set.remove(j)
        #             matched_set.append(j)

        if auto_threshold:
            # Report average matching string vars
            print("Average string vars: " + str(result))
        
        #print("Possible edges: " + str(sum(1 for _ in combinations(range(len(videos)), 2)) * 2))
        #print("First edges: " + str(matched * 2))
        
        # Add semi-random edges to disconnected nodes so every node is reachable
        # for i in tqdm(range(len(videos)), desc="Fixing disconnected nodes..."):
        #     if i in unmatched_set:
        #         j = random.choice(matched_set)
        #         num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[videos[i]], attributes[videos[j]]))
        #         edge_index.append([i, j])
        #         edge_index.append([j, i])
        #         edge_attr.append(num_matching)
        #         edge_attr.append(num_matching) 
        #         unmatched_set.remove(i)
        #         matched_set.append(i)
        #         matched += 1
        
        #print("Final edges: " + str(matched * 2))       
        # assert len(unmatched_set) == 0

        # Build training graph
        train_nodes_indexes = [i for i, val in enumerate(train_mask) if val]
        train_videos = [item for i, item in enumerate(videos) if i in train_nodes_indexes]
        train_labels = [item for i, item in enumerate(y) if i in train_nodes_indexes]
        # Shuffle graph
        n = len(train_videos)
        order = list(range(n))
        random.shuffle(order)
        train_videos = [train_videos[i] for i in order]
        train_labels = [train_labels[i] for i in order]
        train_edge_attr = []
        train_num_nodes = sum(1 for val in train_mask if val)
        train_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(train_videos)))
        print("# of train videos: ")
        print(len(train_videos))
        
        # Iterate over all pairs of training videos
        for i, j in tqdm(combinations(range(len(train_videos)), 2), total= sum(1 for _ in combinations(range(len(train_videos)), 2)), desc="Building training graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[train_videos[i][0]], attributes[train_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                train_edge_index.append([i, j])
                train_edge_index.append([j, i])
                train_edge_attr.append(num_matching)
                train_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(train_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[train_videos[i][0]], attributes[train_videos[j][0]]))
                train_edge_index.append([i, j])
                train_edge_index.append([j, i])
                train_edge_attr.append(num_matching)
                train_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
                
        # Build validation graph
        val_nodes_indexes = [i for i, val in enumerate(val_mask) if val]
        val_videos = [item for i, item in enumerate(videos) if i in val_nodes_indexes]
        val_labels = [item for i, item in enumerate(y) if i in val_nodes_indexes]
        n = len(val_videos)
        order = list(range(n))
        random.shuffle(order)
        val_videos = [val_videos[i] for i in order]
        val_labels = [val_labels[i] for i in order]
        val_edge_attr = []
        val_num_nodes = sum(1 for val in val_mask if val)
        val_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(val_videos)))
        # Iterate over all pairs of validation videos
        for i, j in tqdm(combinations(range(len(val_videos)), 2), total= sum(1 for _ in combinations(range(len(val_videos)), 2)), desc="Building validation graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[val_videos[i][0]], attributes[val_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                val_edge_index.append([i, j])
                val_edge_index.append([j, i])
                val_edge_attr.append(num_matching)
                val_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(val_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[val_videos[i][0]], attributes[val_videos[j][0]]))
                val_edge_index.append([i, j])
                val_edge_index.append([j, i])
                val_edge_attr.append(num_matching)
                val_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
                
        # Build testing graph
        test_nodes_indexes = [i for i, val in enumerate(test_mask) if val]
        test_videos = [item for i, item in enumerate(videos) if i in test_nodes_indexes]
        test_labels = [item for i, item in enumerate(y) if i in test_nodes_indexes]
        n = len(test_videos)
        order = list(range(n))
        random.shuffle(order)
        test_videos = [test_videos[i] for i in order]
        test_labels = [test_labels[i] for i in order]
        test_edge_attr = []
        test_num_nodes = sum(1 for val in test_mask if val)
        test_edge_index = []
        matched_set = []
        unmatched_set = list(range(len(test_videos)))
        # Iterate over all pairs of testing videos
        for i, j in tqdm(combinations(range(len(test_videos)), 2), total= sum(1 for _ in combinations(range(len(test_videos)), 2)), desc="Building testing graph..."):
            # Calculate the number of matching variables
            num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[test_videos[i][0]], attributes[test_videos[j][0]]))
            
            # If the number of matching variables is above a certain threshold, add an edge
            if num_matching > threshold:
                matched += 1
                test_edge_index.append([i, j])
                test_edge_index.append([j, i])
                test_edge_attr.append(num_matching)
                test_edge_attr.append(num_matching)  # Add the same attribute for the reverse edge
                # Update arrays of matched and unmatched nodes
                if i in unmatched_set:
                    unmatched_set.remove(i)
                    matched_set.append(i)
                if j in unmatched_set:
                    unmatched_set.remove(j)
                    matched_set.append(j)
                    
        # Add semi-random edges to disconnected nodes so every node is reachable
        for i in tqdm(range(len(test_videos)), desc="Fixing disconnected nodes..."):
            if i in unmatched_set:
                j = random.choice(matched_set)
                num_matching = sum(c1 == c2 for c1, c2 in zip(attributes[test_videos[i][0]], attributes[test_videos[j][0]]))
                test_edge_index.append([i, j])
                test_edge_index.append([j, i])
                test_edge_attr.append(num_matching)
                test_edge_attr.append(num_matching) 
                unmatched_set.remove(i)
                matched_set.append(i)
                matched += 1
        
        
        # Convert lists to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create a PyTorch Geometric graph (currently deprecated due to splitting into train, val, test graphs)
        # data = Data(x=videos, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, num_nodes=len(videos)) 
        
        # Do data shuffling
        # n = len(train_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # train_videos = [train_videos[i] for i in order]
        # train_labels = [train_labels[i] for i in order]
        # train_edge_index = [train_edge_index[i] for i in order]
        # train_edge_attr = [train_edge_attr[i] for i in order]
        
        # n = len(val_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # val_videos = [val_videos[i] for i in order]
        # val_edge_index = [val_edge_index[i] for i in order]
        # val_edge_attr = [val_edge_attr[i] for i in order]
        # val_labels = [val_labels[i] for i in order]
        
        # n = len(test_videos)
        # order = list(range(n))
        # random.shuffle(order)
        # test_videos = [test_videos[i] for i in order]
        # test_edge_index = [test_edge_index[i] for i in order]
        # test_edge_attr = [test_edge_attr[i] for i in order]
        # test_labels = [test_labels[i] for i in order]
        
        data_train = Data(x=train_videos, y=train_labels, edge_index=torch.tensor(train_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(train_edge_attr, dtype=torch.float), num_nodes = train_num_nodes)
        
        data_val = Data(x=val_videos, y=val_labels, edge_index=torch.tensor(val_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(val_edge_attr, dtype=torch.float), num_nodes = val_num_nodes)
        
        data_test = Data(x=test_videos, y=test_labels, edge_index=torch.tensor(test_edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(test_edge_attr, dtype=torch.float), num_nodes = test_num_nodes)
        
        for data in [data_train, data_val, data_test]:
            if data.has_isolated_nodes():
                raise ValueError("Invalid graph: isolated nodes detected.")
            if data.is_directed():
                raise ValueError("Invalid graph: directionality present.")
        
        print("Graph generation finished in " + str(time.time() - start) + "s.")
        
        return data, data_train, data_val, data_test, videos, train_mask, val_mask, test_mask

if __name__ == "__main__":
    graph, encoder = DeepfakeDataset().generate_graph(file_root='/home/brg2890/major/preprocessed/FaceForensics++_All/FaceForensics++_Graph', attribute_root='/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/A-FF++.csv', threshold=38)

    print(graph.num_nodes)


    print(graph.num_edges)


    print(graph.num_node_features)


    print(graph.has_isolated_nodes())


    print(graph.has_self_loops())


    print(graph.is_directed())

    # Call the function with your graph and processing function
    traverse_graph(graph, process_node_data, X=1, rng_threshold=0.01, num_pointers=3)
    
    #print('edge_attr' in graph)

    print(graph.num_nodes)


    print(graph.num_edges)


    print(graph.num_node_features)


    print(graph.has_isolated_nodes())


    print(graph.has_self_loops())


    print(graph.is_directed())

    # Transfer data object to GPU.
    device = torch.device('cuda')
    graph = graph.to(device)
    print("Cuda test finished.")
