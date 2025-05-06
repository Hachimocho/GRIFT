"""
Test script for the Hierarchical Deepfake Dataloader

This script tests the new hierarchical graph construction approach which:
1. Groups nodes by categorical attributes (race-gender combinations)
2. Creates fully-connected subgraphs within each group
3. Applies threshold-based filtering for quality metrics, symmetry, embeddings, etc.
"""
import time
import argparse
import os
import cv2
from collections import defaultdict
import sys
import dill
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import logging  # Add missing import for logging module
import traceback # Add traceback import
import json # Add json import
import matplotlib.pyplot as plt # Added
import random # Add import
from collections import defaultdict # Add import

# Add a null handler for silencing logging
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from edges.Edge import Edge
from nodes.atrnode import AttributeNode
from graphs.HyperGraph import HyperGraph
from data.ImageFileData import ImageFileData

# Imports for model training/testing
import sys
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from trainers.ExperimentTrainer import ExperimentTrainer
from trainers.IValueTrainer import IValueTrainer
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal
from traversals.IValueTraversalClusterHop import IValueTraversalClusterHop # Added Import
from traversals.RandomTraversal import RandomTraversal
from models.CNNModel import CNNModel
from edges.Edge import Edge
import copy
import torch
import time
import random

import torch
import torch.nn as nn
import numpy as np
import traceback
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader # Make sure DataLoader is imported
import os # Added for path operations
import sys # Added for exception logging
from contextlib import contextmanager
import io

def evaluate_model(model, nodes_to_evaluate, loss_fn, batch_size, bias_loss_fn=None, device='cuda', desc="Evaluating", attribute_metadata=None): 
    """Evaluates the model on the provided nodes, calculates standard metrics,
       and optionally calculates bias metrics based on categorical attributes.
    """
    model.eval() # Ensure model is in evaluation mode
    model.model.to(device)

    total_loss = 0.0
    total_bias_loss = 0.0
    correct_predictions = 0
    total_nodes_processed = 0
    nodes_in_dataset = len(nodes_to_evaluate)
    num_batches = (nodes_in_dataset + batch_size - 1) // batch_size

    print(f"\nRunning inference for {desc} (Dataset Size: {nodes_in_dataset}, Batch Size: {batch_size})...")

    all_predictions = []
    all_labels = []
    subgroup_stats = defaultdict(lambda: {'count': 0, 'correct': 0})
    categorical_attrs = []
    if attribute_metadata:
        categorical_attrs = [
            attr for attr in attribute_metadata if attr.get('type') == 'categorical'
        ]
        if not categorical_attrs:
            print("Warning: attribute_metadata provided, but no categorical attributes found for bias calculation.")
            attribute_metadata = None # Disable bias calculation if no categorical attrs
        else:
            print(f"Bias calculation enabled for attributes: {[attr['name'] for attr in categorical_attrs]}")

    with torch.no_grad(): # Ensure no gradients are calculated during evaluation
        for i in tqdm(range(num_batches), desc=f"Inferring {desc}", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, nodes_in_dataset)
            batch_nodes = nodes_to_evaluate[start_idx:end_idx]

            batch_images_loaded = []
            batch_labels_loaded = []
            batch_nodes_loaded = [] # Keep track of nodes successfully loaded in batch

            # Load data for the current batch
            for node in batch_nodes:
                try:
                    node_data = node.get_data()
                    if node_data:
                        img = node_data.load_data()
                        label = node.get_label()
                        if img is not None and label is not None:
                            # Apply transformations using the model's internal method
                            # (assumes model.current_mode is set to 'eval' correctly)
                            img_tensor = model.transform(img)
                            batch_images_loaded.append(img_tensor)
                            batch_labels_loaded.append(float(label))
                            batch_nodes_loaded.append(node) # Add node if data loaded
                        else:
                            # print(f"DEBUG: Img or Label is None for node {node.node_id}")
                            pass
                    else:
                        # print(f"DEBUG: node.get_data() returned None for node {node.node_id}")
                        pass
                except Exception as e_load:
                    print(f"ERROR loading data for node {getattr(node, 'node_id', 'N/A')} in {desc}: {e_load}")
                    continue # Skip node on error

            # Skip batch if no data was successfully loaded
            if not batch_images_loaded:
                # print(f"DEBUG: Skipping empty batch {i} in {desc}")
                continue

            # Stack loaded data into tensors
            try:
                batch_images_tensor = torch.stack(batch_images_loaded).to(device)
                batch_labels_tensor = torch.tensor(batch_labels_loaded, dtype=torch.float).unsqueeze(1).to(device) # Ensure [batch, 1]
            except Exception as e_stack:
                print(f"ERROR stacking tensors for batch {i} in {desc}: {e_stack}")
                continue # Skip batch if stacking fails

            # Perform inference
            try:                
                outputs = model(batch_images_tensor)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                correct = (preds == batch_labels_tensor).sum().item()
                correct_predictions += correct
                current_batch_size = batch_labels_tensor.size(0)
                total_nodes_processed += current_batch_size

                loss = loss_fn(outputs, batch_labels_tensor)
                total_loss += loss.item() * current_batch_size # Accumulate total loss scaled by batch size

                if bias_loss_fn and batch_nodes_loaded: # Use successfully loaded nodes
                    try:
                        # Ensure bias_loss_fn can handle the list of node objects
                        bias_loss_val = bias_loss_fn(outputs, batch_labels_tensor, batch_nodes_loaded)
                        total_bias_loss += bias_loss_val.item() * current_batch_size # Accumulate total bias loss
                    except Exception as e_bias:
                        print(f"\nWarning: Error calculating bias loss for batch in {desc}: {e_bias}")
                        total_bias_loss += 0.0 # Add 0 on error for this batch
            except Exception as e_inf:
                 print(f"\nError during model inference or loss calculation in {desc}: {e_inf}")
                 # Potentially skip batch or handle error appropriately
                 continue # Skip batch on inference error

            # --- Store Predictions and Labels for Metrics --- 
            predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5
            current_labels = batch_labels_tensor.cpu().numpy().astype(int)
            all_predictions.extend(predictions.astype(int))
            all_labels.extend(current_labels)

            # --- Associate predictions/labels with nodes for bias calc ---
            node_results = {}
            if attribute_metadata and categorical_attrs: # Check again in case it was disabled
                 for i, node in enumerate(batch_nodes_loaded):
                      node_results[node.node_id] = {
                           'prediction': predictions[i],
                           'label': current_labels[i],
                           'node': node # Store the node object itself
                      }

            # --- Update Subgroup Stats for Bias Calculation ---
            if attribute_metadata and categorical_attrs: # Check again in case it was disabled
                # Iterate through the results we just stored for the successfully processed batch
                for node_id, result in node_results.items():
                    node = result['node']
                    prediction = result['prediction']
                    label = result['label']
                    try:
                        subgroup_key_parts = []
                        valid_node = True
                        for cat_attr in categorical_attrs:
                            attr_name = cat_attr['name']
                            # Check if node has attributes and the specific attribute
                            if not hasattr(node, 'attributes') or node.attributes is None or attr_name not in node.attributes:
                                valid_node = False
                                break
                            attr_value = node.attributes[attr_name]
                            subgroup_key_parts.append(f"{attr_name}_{attr_value}")

                        if valid_node:
                            subgroup_key = "_".join(subgroup_key_parts)
                            subgroup_stats[subgroup_key]['count'] += 1
                            # Compare prediction and label for this specific node
                            if prediction == label:
                                subgroup_stats[subgroup_key]['correct'] += 1
                    except Exception as e_subgroup:
                         print(f"Warning: Error processing node {getattr(node, 'node_id', 'N/A')} for bias subgroup stats: {e_subgroup}")

    # --- Final Metrics Calculation ---
    # Nodes skipped is estimated as total in dataset minus those successfully processed
    skipped_loading = nodes_in_dataset - total_nodes_processed

    final_metrics = {}
    if total_nodes_processed > 0:
        final_metrics['accuracy'] = (correct_predictions / total_nodes_processed) * 100
        final_metrics['average_loss'] = total_loss / total_nodes_processed # Average loss per successfully processed sample
    else:
        final_metrics['accuracy'] = 0.0
        final_metrics['average_loss'] = float('nan')

    if bias_loss_fn:
         if total_nodes_processed > 0:
             final_metrics['average_bias_loss'] = total_bias_loss / total_nodes_processed # Average bias loss per successfully processed sample
         else:
             final_metrics['average_bias_loss'] = float('nan')

    final_metrics['total_nodes_in_dataset'] = nodes_in_dataset
    final_metrics['total_nodes_skipped_loading'] = skipped_loading
    final_metrics['total_predictions_made'] = total_nodes_processed # Nodes that contributed to metrics

    print(f"\n{desc} Results: Accuracy={final_metrics.get('accuracy', 0.0):.2f}%, Avg Loss={final_metrics.get('average_loss', float('nan')):.4f}", end='')
    if bias_loss_fn and 'average_bias_loss' in final_metrics:
         print(f", Avg Bias Loss={final_metrics.get('average_bias_loss', float('nan')):.4f}", end='')
    # Adjust print statement to reflect dataset size and predictions made
    print(f" (Dataset Size: {nodes_in_dataset}, Skipped Loading: {skipped_loading}, Predictions Made: {total_nodes_processed})")

    # --- Bias Calculation Setup --- 
    bias_metrics = {}
    if attribute_metadata and categorical_attrs: # Check again in case it was disabled
        print(f"Bias calculation enabled for attributes: {[attr['name'] for attr in categorical_attrs]}")

    # --- Calculate Bias Metrics --- (Only if enabled and stats collected)
    if attribute_metadata and categorical_attrs and subgroup_stats:
        subgroup_accuracies = {}
        min_subgroup_acc = 1.0
        max_subgroup_acc = 0.0
        total_subgroup_abs_diff = 0.0
        num_subgroups = 0

        print("\n--- Bias Analysis --- ")
        per_attribute_stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'correct': 0})) # Define BEFORE subgroup loop
        overall_accuracy = final_metrics['accuracy'] / 100 # Convert to decimal
        for key, stats in sorted(subgroup_stats.items()):
            count = stats['count']
            correct = stats['correct']
            if count > 0:
                accuracy = correct / count
                subgroup_accuracies[key] = accuracy
                print(f"  Subgroup '{key}': Accuracy = {accuracy:.4f} (Count: {count})")
                min_subgroup_acc = min(min_subgroup_acc, accuracy)
                max_subgroup_acc = max(max_subgroup_acc, accuracy)
                total_subgroup_abs_diff += abs(accuracy - overall_accuracy) # Use decimal overall_accuracy
                num_subgroups += 1
                # Populate per_attribute_stats INSIDE subgroup loop
                key_parts = key.split('_')
                i = 0
                while i < len(key_parts):
                    attr_name = key_parts[i]
                    attr_value = key_parts[i+1]
                    per_attribute_stats[attr_name][attr_value]['count'] += count
                    per_attribute_stats[attr_name][attr_value]['correct'] += correct
                    i += 2 # Move to the next attribute-value pair
            else:
                print(f"  Subgroup '{key}': Accuracy = N/A (Count: 0)")
                subgroup_accuracies[key] = None

        overall_bias = max_subgroup_acc - min_subgroup_acc if num_subgroups > 0 else 0
        average_subgroup_bias = (total_subgroup_abs_diff / num_subgroups) if num_subgroups > 0 else 0
        bias_metrics['subgroup_accuracies'] = subgroup_accuracies
        bias_metrics['overall_bias'] = overall_bias
        bias_metrics['average_subgroup_bias'] = average_subgroup_bias # Add average subgroup bias

        print(f"Overall Bias (Max Acc Diff across subgroups): {overall_bias:.4f}")
        print(f"Average Subgroup Bias (Avg Abs Diff from Overall Acc): {average_subgroup_bias:.4f}") # Print average subgroup bias

        # Calculate per-attribute bias
        per_attribute_accuracies = defaultdict(dict)
        per_attribute_bias = {}
        total_attribute_bias = 0.0
        num_attributes = 0

        print("\nPer-Attribute Analysis:")
        for attr_name, value_stats in sorted(per_attribute_stats.items()):
            print(f"  Attribute '{attr_name}':")
            min_attr_acc = 1.0
            max_attr_acc = 0.0
            value_count = 0
            for value, stats in sorted(value_stats.items()):
                count = stats['count']
                correct = stats['correct']
                if count > 0:
                    accuracy = correct / count
                    per_attribute_accuracies[attr_name][value] = accuracy
                    print(f"    Value '{value}': Accuracy = {accuracy:.4f} (Count: {count})")
                    min_attr_acc = min(min_attr_acc, accuracy)
                    max_attr_acc = max(max_attr_acc, accuracy)
                    value_count += 1
                else:
                    print(f"    Value '{value}': Accuracy = N/A (Count: 0)")
                    per_attribute_accuracies[attr_name][value] = None
            
            attr_bias = max_attr_acc - min_attr_acc if value_count > 0 else 0
            per_attribute_bias[attr_name] = attr_bias
            total_attribute_bias += attr_bias
            num_attributes += 1
            print(f"    Bias for '{attr_name}' (Max Acc Diff): {attr_bias:.4f}")

        average_attribute_bias = (total_attribute_bias / num_attributes) if num_attributes > 0 else 0
        bias_metrics['per_attribute_accuracies'] = per_attribute_accuracies
        bias_metrics['per_attribute_bias'] = per_attribute_bias
        bias_metrics['average_attribute_bias'] = average_attribute_bias # Add average attribute bias
        print(f"Average Attribute Bias (Avg Max Acc Diff): {average_attribute_bias:.4f}") # Print average attribute bias

    # --- Return Results --- 
    final_metrics['bias_metrics'] = bias_metrics # Include bias metrics
    return final_metrics

@contextmanager
def capture_output(filename):
    """Capture all stdout and stderr output to a file while still printing to terminal"""
    class TeeStream:
        def __init__(self, stdout, logfile):
            self.stdout = stdout
            self.logfile = logfile
            
        def write(self, message):
            self.stdout.write(message)
            self.logfile.write(message)
            
        def flush(self):
            self.stdout.flush()
            self.logfile.flush()
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logpath = log_dir / filename
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        with open(logpath, 'w') as logfile:
            tee_stdout = TeeStream(old_stdout, logfile)
            tee_stderr = TeeStream(old_stderr, logfile)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            yield logpath
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def log_exception(logfile, exc_type, exc_value, exc_traceback):
    """Log an exception with its traceback to both stdout and the log file"""
    exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print('\n' + '=' * 80)
    print('Exception occurred:')
    print(exc_text)
    print('=' * 80)
    
    with open(logfile, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('Exception occurred:\n')
        f.write(exc_text)
        f.write('=' * 80 + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Test the hierarchical graph construction approach')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited nodes')
    parser.add_argument('--visualize', action='store_true', help='Generate graph visualizations')
    parser.add_argument('--show', action='store_true', help='Show visualizations (requires --visualize)')
    parser.add_argument('--quality-threshold', type=float, default=0.8, 
                        help='Similarity threshold for quality metrics (default: 0.8)')
    parser.add_argument('--symmetry-threshold', type=float, default=0.75, 
                        help='Similarity threshold for facial symmetry (default: 0.75)')
    parser.add_argument('--embedding-threshold', type=float, default=0.7, 
                        help='Similarity threshold for face embeddings (default: 0.7)')
    
    # Node caching options
    parser.add_argument('--cache-nodes', action='store_true', 
                        help='Save loaded nodes to cache file for faster testing')
    parser.add_argument('--cache-full', action='store_true',
                        help='Cache the entire dataset instead of just a subset (use with --cache-nodes)')
    parser.add_argument('--use-cached', action='store_true', 
                        help='Use previously cached nodes instead of loading from dataset')
    parser.add_argument('--use-full-cache', action='store_true',
                        help='Load the full dataset from cache instead of the subset (use with --use-cached)')
    parser.add_argument('--cached-nodes', type=int, default=1000, 
                        help='Number of nodes to cache per split when not using full cache (default: 1000)')
    parser.add_argument('--cache-file', type=str, default='node_cache/cached_nodes.pkl', 
                        help='Filename for caching/loading nodes (relative to script execution dir)')

    # Grid search options
    parser.add_argument('--search', action='store_true',
                        help='Run grid search over threshold combinations')
    parser.add_argument('--search-split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Split to use for grid search (default: train)')
    parser.add_argument('--quality-steps', type=int, default=5,
                        help='Number of steps for quality threshold grid search (default: 5)')
    parser.add_argument('--symmetry-steps', type=int, default=5,
                        help='Number of steps for symmetry threshold grid search (default: 5)')
    parser.add_argument('--embedding-steps', type=int, default=5,
                        help='Number of steps for embedding threshold grid search (default: 5)')
    parser.add_argument('--search-results', type=str, default='threshold_search_results.csv',
                        help='File to save search results to (default: threshold_search_results.csv)')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for training and evaluation (default: 100)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for DataLoader (default: 4)')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--bias_loss_weight', type=float, default=0.00,
                        help='Weight for bias loss (default: 0.00)')
    parser.add_argument('--bias_hop_period', type=int, default=100,
                        help='Period for bias hop (default: 100)')
    parser.add_argument('--load-last-checkpoint', action='store_true',
                        help='Load the last best checkpoint if validation accuracy decreases.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--fair-train', action='store_true', help='Use subgroup-balanced training set for graph construction')
    parser.add_argument('--fair-test', action='store_true', help='Use subgroup-balanced validation/test sets for graph construction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') # Add seed argument

    return parser.parse_args()

def balance_nodes_by_subgroup(nodes, target_num_nodes, attributes_to_balance=['Ground Truth Race', 'Ground Truth Gender']):
    """Balances nodes across subgroups to reach a target total number.

    Args:
        nodes: List of nodes to balance.
        target_num_nodes: The desired total number of nodes in the balanced list.
        attributes_to_balance: List of attribute keys to define subgroups.

    Returns:
        A list of nodes balanced across subgroups, totaling target_num_nodes.

    Raises:
        ValueError: If balancing is not possible because one or more subgroups
                    are too small to provide the required number of samples.
    """
    if not nodes or target_num_nodes <= 0:
        print(f"Warning: Cannot balance empty node list or with target_num_nodes={target_num_nodes}. Returning empty list.")
        return []

    subgroups = defaultdict(list)
    for node in nodes:
        subgroup_key = tuple(node.attributes.get(attr, 'Unknown') for attr in attributes_to_balance)
        subgroups[subgroup_key].append(node)

    num_subgroups = len(subgroups)
    if num_subgroups == 0:
        print("Warning: No subgroups found for balancing. Returning original list (or empty if target > original size).")
        # Return original list only if its size matches target, otherwise it's impossible
        return nodes if len(nodes) == target_num_nodes else []

    nodes_per_subgroup = target_num_nodes // num_subgroups
    remainder = target_num_nodes % num_subgroups

    print(f"Balancing to {target_num_nodes} nodes across {num_subgroups} subgroups.")
    print(f"Base nodes per subgroup: {nodes_per_subgroup}, Remainder: {remainder}")

    balanced_nodes = []
    subgroup_keys = list(subgroups.keys())
    random.shuffle(subgroup_keys) # Shuffle keys to randomly distribute remainder

    for i, subgroup_key in enumerate(subgroup_keys):
        group_nodes = subgroups[subgroup_key]
        required_size = nodes_per_subgroup + (1 if i < remainder else 0)

        if len(group_nodes) < required_size:
            raise ValueError(
                f"Cannot balance to {target_num_nodes} nodes. Subgroup {subgroup_key} "
                f"has only {len(group_nodes)} nodes, but requires {required_size}."
            )

        if required_size > 0:
            sampled_nodes = random.sample(group_nodes, required_size)
            balanced_nodes.extend(sampled_nodes)

    random.shuffle(balanced_nodes) # Shuffle the final list
    print(f"Total nodes after balancing: {len(balanced_nodes)} (Target: {target_num_nodes})")
    if len(balanced_nodes) != target_num_nodes:
         print(f"WARNING: Final balanced node count ({len(balanced_nodes)}) does not match target ({target_num_nodes})!") # Should not happen

    return balanced_nodes

def save_cached_nodes(train_nodes, val_nodes, test_nodes, cache_file, target_num_nodes):
    """Balances each node list to target_num_nodes and saves full/balanced versions per split."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Balancing node lists for caching to target size {target_num_nodes}...")

    cache_data = {}
    for split_name, nodes_list in [('train', train_nodes), ('val', val_nodes), ('test', test_nodes)]:
        print(f"  Processing {split_name} split ({len(nodes_list)} nodes)")
        try:
            balanced_list = balance_nodes_by_subgroup(nodes_list, target_num_nodes=target_num_nodes)
            cache_data[split_name] = {
                'full': nodes_list,
                'balanced': balanced_list
            }
            print(f"    Full: {len(nodes_list)}, Balanced: {len(balanced_list)}")
        except ValueError as e:
             print(f"    ERROR balancing {split_name} split: {e}. Skipping balancing for this split in cache.")
             # Store full list as balanced if balancing failed
             cache_data[split_name] = {
                 'full': nodes_list,
                 'balanced': nodes_list # Fallback to full if balancing fails
             }

    with open(cache_file, 'wb') as f:
        dill.dump(cache_data, f)

def load_cached_nodes(cache_file, split_name, balanced=False):
    """Loads nodes for a specific split from cache, optionally the balanced set."""
    if os.path.exists(cache_file):
        print(f"Attempting to load {split_name} nodes from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = dill.load(f)
            print(f"  Cache file loaded. Type: {type(cache_data)}. Checking structure...")

            # Check NEW format (dict keyed by split_name, containing dicts with 'full'/'balanced')
            if isinstance(cache_data, dict) and \
               split_name in cache_data and \
               isinstance(cache_data.get(split_name), dict) and \
               'full' in cache_data[split_name] and \
               'balanced' in cache_data[split_name]:
                print(f"  -> Detected NEW cache format for split '{split_name}'.")
                nodes_to_return = cache_data[split_name]['balanced'] if balanced else cache_data[split_name]['full']
                load_type = 'Balanced' if balanced else 'Full'
                print(f"     {load_type} nodes ({len(nodes_to_return)}) loaded successfully.")
                return nodes_to_return
            else:
                print(f"  -> Did not match NEW format for split '{split_name}'. Checking older formats...")
                # Provide details if it looked like a dict but failed the checks
                if isinstance(cache_data, dict):
                    if split_name not in cache_data:
                        print(f"     Reason: Split key '{split_name}' not found in top-level dict.")
                    elif not isinstance(cache_data.get(split_name), dict):
                        print(f"     Reason: Value for key '{split_name}' is not a dict (Type: {type(cache_data.get(split_name))}).")
                    elif 'full' not in cache_data.get(split_name, {}):
                        print(f"     Reason: Key 'full' not found within dict for split '{split_name}'.")
                    elif 'balanced' not in cache_data.get(split_name, {}):
                        print(f"     Reason: Key 'balanced' not found within dict for split '{split_name}'.")

            # Check intermediate OLD format (dict with 'full'/'balanced' directly)
            if isinstance(cache_data, dict) and 'full' in cache_data and 'balanced' in cache_data:
                 print(f"  -> Detected OLD cache format (dict without splits). Loading overall 'full' set as fallback for '{split_name}'.")
                 nodes_to_return = cache_data['balanced'] if balanced else cache_data['full']
                 if balanced:
                      print(f"     Warning: Requested balanced set, returning from overall balanced set ({len(nodes_to_return)} nodes). This might not be split-specific.")
                 else:
                      print(f"     Returning overall full set ({len(nodes_to_return)} nodes). This might not be split-specific.")
                 return nodes_to_return

            # Check very OLD format (just a list)
            elif isinstance(cache_data, list):
                 print(f"  -> Detected VERY OLD cache format (list). Loading as full set for '{split_name}'.")
                 if balanced:
                     print("     Warning: Cannot load balanced set from list format. Returning full list.")
                 print(f"     Returning full list ({len(cache_data)} nodes)." )
                 return cache_data

            # Unrecognized
            print(f"  -> Cache file structure is unrecognized. Ignoring cache for split '{split_name}'.")
            return None

        except (dill.UnpicklingError, EOFError, KeyError, AttributeError) as e:
            print(f"Error loading/parsing cache file {cache_file} for split '{split_name}': {e}. Ignoring cache.")
            return None
    else:
        print(f"Cache file {cache_file} not found for split '{split_name}'.")
        return None

def run_threshold_grid_search(nodes, edge_class, split_name, quality_steps, symmetry_steps, embedding_steps):
    """Run grid search over threshold parameters and log results"""
    # Create search grid
    quality_thresholds = np.linspace(.5, .9, quality_steps)
    symmetry_thresholds = np.linspace(.5, .9, symmetry_steps)
    embedding_thresholds = np.linspace(.9, .999, embedding_steps)
    
    # Create results dataframe
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/threshold_search_{timestamp}.csv"
    
    total_combinations = len(quality_thresholds) * len(symmetry_thresholds) * len(embedding_thresholds)
    print(f"\nRunning grid search with {total_combinations} threshold combinations...")
    
    # Create a single progress bar for all combinations
    combinations = list(product(quality_thresholds, symmetry_thresholds, embedding_thresholds))
    progress_bar = tqdm(total=len(combinations), desc="Running grid search")

    for q_thresh, s_thresh, e_thresh in combinations:
        # Round thresholds for cleaner reporting
        q_thresh_str = round(q_thresh, 2)
        s_thresh_str = round(s_thresh, 2)
        e_thresh_str = round(e_thresh, 2)
        
        # Update progress bar with detailed description
        progress_bar.set_description(
            f"Combination {progress_bar.n+1}/{len(combinations)} - Testing Q:{q_thresh_str} S:{s_thresh_str} E:{e_thresh_str}"
        )
        progress_bar.update(0)  # Force refresh without incrementing
        
        # Create dataloader with current thresholds
        dataloader = HierarchicalDeepfakeDataloader(
            datasets=[], 
            edge_class=edge_class,
            test_mode=False,  # Don't limit nodes
            visualize=False,  # Don't create visualizations during search
            show_viz=False,
            quality_threshold=q_thresh_str,
            symmetry_threshold=s_thresh_str,
            embedding_threshold=e_thresh_str,
            silent_mode=True  # Disable internal progress bars and logging during grid search
        )
        
        # Capture and silence ALL output (stdout, stderr, and logging)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        null_output = open(os.devnull, 'w')
        sys.stdout = null_output
        sys.stderr = null_output

        # Save the original handlers for ALL loggers
        original_handlers = {}
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            original_handlers[logger_name] = list(logger.handlers)
            logger.handlers = [NullHandler()]
            
        # Also handle the root logger
        root_logger = logging.getLogger()
        original_root_handlers = list(root_logger.handlers)
        root_logger.handlers = [NullHandler()]
        
        # Disable tqdm progress bars
        original_tqdm = tqdm.__init__
        def silent_tqdm__init__(*args, **kwargs):
            kwargs['disable'] = True
            return original_tqdm(*args, **kwargs)
        tqdm.__init__ = silent_tqdm__init__
        
        try:
            # Call _build_graph_standard directly to get the count
            graph, num_edges_after_filter = dataloader._build_graph_standard(nodes, split_name)

            # Check if fallback was triggered (using the info stored on the graph)
            fallback_triggered = getattr(graph, 'fallback_triggered', False)
            fallback_nodes_count = getattr(graph, 'fallback_nodes_count', 0)
            fallback_pct = (fallback_nodes_count / len(nodes) * 100) if len(nodes) > 0 else 0
            
            # Calculate metrics on the graph after construction (including fallback connections)
            all_nodes_in_graph = graph.get_nodes()
            total_edges = 0
            node_degrees = [0] * len(all_nodes_in_graph)
            
            # Count degrees using the graph's adjacency list
            for i, node in enumerate(all_nodes_in_graph):
                node_degrees[i] = len(node.get_adjacent_nodes())

            total_edges = sum(node_degrees) // 2  # Divide by 2 since each edge is counted twice
            avg_degree = sum(node_degrees) / len(all_nodes_in_graph) if all_nodes_in_graph else 0
            
            # Store the node count for this test
            node_count = len(all_nodes_in_graph)
            
            # Save results with detailed information about the filtering and fallback
            results.append({
                'quality_threshold': q_thresh_str,
                'symmetry_threshold': s_thresh_str,
                'embedding_threshold': e_thresh_str,
                'average_degree': avg_degree,
                'total_edges': total_edges,
                'num_edges_after_filter': num_edges_after_filter, # Use direct count
                'fallback_triggered': fallback_triggered,
                'fallback_pct': fallback_pct
            })
            
            # Write current result to CSV file (append mode)
            if len(results) == 1:
                # Create header if this is the first result
                pd.DataFrame([results[0]]).to_csv(log_file, index=False)
            else:
                # Append without header for subsequent results
                pd.DataFrame([results[-1]]).to_csv(log_file, mode='a', header=False, index=False)
                
            # Update progress bar with result
            progress_bar.set_postfix(avg_degree=f"{avg_degree:.2f}", total_edges=total_edges)
            
        except Exception as e:
            # Restore output streams TEMPORARILY to print the error
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print(f"\n--- ERROR ENCOUNTERED during grid search for thresholds: Q={q_thresh_str}, S={s_thresh_str}, E={e_thresh_str} ---")
            traceback.print_exc() # Print the full traceback
            print("--------------------------------------------------------------------------------")
            print("Stopping grid search due to error.")
            # Restore suppressors just in case, although we will exit
            sys.stdout = null_output
            sys.stderr = null_output
            # Re-raise the exception to halt the script
            raise e 
            # Optionally, append error and continue:
            # results.append({
            #     'quality_threshold': q_thresh_str,
            #     'symmetry_threshold': s_thresh_str,
            #     'embedding_threshold': e_thresh_str,
            #     'average_degree': 0,
            #     'total_edges': 0,
            #     'fallback_triggered': False,
            #     'fallback_pct': 0,
            #     'num_edges_after_filter': 0,
            #     'error': f"{e.__class__.__name__}: {e}"
            # })
            # # Suppress output again before continuing loop
            # sys.stdout = null_output
            # sys.stderr = null_output

        finally:
            # Ensure output streams and loggers are restored
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Restore all logger handlers
            for logger_name, handlers in original_handlers.items():
                logging.getLogger(logger_name).handlers = handlers
            logging.getLogger().handlers = original_root_handlers
            
            # Restore tqdm
            tqdm.__init__ = original_tqdm
            
            # Make sure to close the null output file
            null_output.close()
            
        # Update progress bar
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def visualize_search_results(results_df, output_prefix):
    """Create visualizations of search results"""
    os.makedirs('logs/search_plots', exist_ok=True)
    
    # 1. 3D scatter plot of all parameters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        results_df['quality_threshold'],
        results_df['symmetry_threshold'],
        results_df['embedding_threshold'],
        c=results_df['average_degree'],
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    
    ax.set_xlabel('Quality Threshold')
    ax.set_ylabel('Symmetry Threshold')
    ax.set_zlabel('Embedding Threshold')
    ax.set_title('Impact of Thresholds on Average Degree')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Degree')
    
    plt.tight_layout()
    plt.savefig(f'logs/search_plots/{output_prefix}_3d_plot.png')
    plt.close()
    
    # 2. Heat maps for each pair of thresholds
    param_pairs = [
        ('quality_threshold', 'symmetry_threshold', 'embedding_threshold'),
        ('quality_threshold', 'embedding_threshold', 'symmetry_threshold'),
        ('symmetry_threshold', 'embedding_threshold', 'quality_threshold')
    ]
    
    for x_param, y_param, z_param in param_pairs:
        # Create pivot table
        unique_z_values = sorted(results_df[z_param].unique())
        
        # Create subplots for each value of z_param
        fig, axes = plt.subplots(
            nrows=1, 
            ncols=len(unique_z_values), 
            figsize=(5 * len(unique_z_values), 5),
            sharey=True
        )
        
        if len(unique_z_values) == 1:
            axes = [axes]  # Ensure axes is iterable
            
        for i, z_value in enumerate(unique_z_values):
            # Filter data for this z value
            filtered_data = results_df[results_df[z_param] == z_value]
            
            # Create pivot table
            pivot_data = filtered_data.pivot_table(
                index=y_param,
                columns=x_param,
                values='average_degree',
                aggfunc='mean'
            )
            
            # Plot heatmap
            im = axes[i].imshow(pivot_data, cmap='viridis', aspect='auto', origin='lower')
            
            # Configure axes
            axes[i].set_title(f'{z_param}={z_value}')
            axes[i].set_xlabel(x_param)
            if i == 0:
                axes[i].set_ylabel(y_param)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Average Degree')
        
        plt.tight_layout()
        plt.savefig(f'logs/search_plots/{output_prefix}_{x_param}_{y_param}_heatmap.png')
        plt.close()
    
    # 3. Line plots showing individual parameter effects
    params = ['quality_threshold', 'symmetry_threshold', 'embedding_threshold']
    
    for param in params:
        # Group by current parameter and calculate mean degree
        grouped_data = results_df.groupby(param)['average_degree'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            grouped_data[param],
            grouped_data['mean'],
            yerr=grouped_data['std'],
            marker='o',
            linestyle='-',
            capsize=5
        )
        
        plt.xlabel(param)
        plt.ylabel('Average Degree')
        plt.title(f'Effect of {param} on Average Degree')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'logs/search_plots/{output_prefix}_{param}_effect.png')
        plt.close()
    
    print(f"Visualizations saved to logs/search_plots/{output_prefix}_*.png")

def plot_subgroup_i_values(history, output_filename):
    """Plots the average I-value for each subgroup over hop instances."""
    if not history:
        print("No hop history recorded, skipping I-value plot.")
        return

    try:
        # Convert history (list of dicts) to DataFrame
        records = []
        for hop_index, hop_data in enumerate(history):
            for subgroup, avg_ivalue in hop_data.items():
                # Convert tuple subgroup key to string for easier handling if needed
                subgroup_str = str(subgroup) 
                records.append({
                    'HopInstance': hop_index,
                    'Subgroup': subgroup_str,
                    'AvgIValue': avg_ivalue
                })
        
        if not records:
            print("No valid records found in hop history.")
            return
            
        df = pd.DataFrame(records)

        plt.figure(figsize=(15, 8))
        
        # Plot lines for each subgroup
        for subgroup in df['Subgroup'].unique():
            subgroup_df = df[df['Subgroup'] == subgroup]
            plt.plot(subgroup_df['HopInstance'], subgroup_df['AvgIValue'], marker='o', linestyle='-', label=subgroup)

        plt.xlabel('Hop Instance Index')
        plt.ylabel('Average I-Value')
        plt.title('Average I-Value per Subgroup Over Bias Hops')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        plt.savefig(output_filename)
        print(f"Saved subgroup I-value plot to {output_filename}")
        plt.close() # Close the figure to free memory

    except Exception as e:
        print(f"Error generating subgroup I-value plot: {e}")
        import traceback
        traceback.print_exc()

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # Configure CUDA for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) # Enforce deterministic algorithms
    print(f"Random seed set to {seed}")

def main():
    args = parse_args() # Parse args first

    # --- Check for PYTHONHASHSEED --- 
    if 'PYTHONHASHSEED' not in os.environ:
        print("\nWarning: PYTHONHASHSEED environment variable not set.")
        print("         For full reproducibility, set it before running the script, e.g.:")
        print("         export PYTHONHASHSEED=42")
        print("         Or prefix the command: PYTHONHASHSEED=42 python ...\n")
    else:
        print(f"Using PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")

    set_seed(args.seed) # Use args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the primary loss function
    criterion = nn.BCEWithLogitsLoss().to(device)
    print(f"Primary loss function defined: {criterion.__class__.__name__}")

    # --- Force num_workers=0 for reproducibility --- 
    if args.num_workers != 0:
        print(f"Warning: Forcing num_workers=0 (was {args.num_workers}) for reproducibility.")
        args.num_workers = 0

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = Path("logs") / f"test_run_{timestamp}.log"

    
    data_root = "/home/brg2890/major/datasets/ai-face"
    print("Detected arguments:")
    print(args)
    print(f"Bias hop period: {args.bias_hop_period}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/hierarchical_test_{timestamp}.log"
    print(f"Starting test run, logging to: {log_file}")

    # Set up attribute metadata for I-value traversal
    attribute_metadata = [
            {
                'name': 'Ground Truth Gender',
                'type': 'categorical',
                'possible_values': [0, 1]  
            },
            {
                'name': 'Ground Truth Race',
                'type': 'categorical',
                'possible_values': [0, 1, 2, 3]  
            },
            {
                'name': 'Ground Truth Age',
                'type': 'categorical',
                'possible_values': [0, 1, 2, 3]  
            },
            {
                'name': 'blur',
                'type': 'continuous'
            },
            {
                'name': 'brightness',
                'type': 'continuous'
            },
            {
                'name': 'contrast',
                'type': 'continuous'
            },
            {
                'name': 'compression',
                'type': 'continuous'
            },
            {
                'name': 'symmetry_eye',
                'type': 'continuous'
            },
            {
                'name': 'symmetry_mouth',
                'type': 'continuous'
            },
            {
                'name': 'symmetry_nose',
                'type': 'continuous'
            },
            {
                'name': 'symmetry_overall',
                'type': 'continuous'
            },
            {
                'name': 'emotion_angry',
                'type': 'continuous'
            },
            {
                'name': 'emotion_disgust',
                'type': 'continuous'
            },
            {
                'name': 'emotion_fear',
                'type': 'continuous'
            },
            {
                'name': 'emotion_happy',
                'type': 'continuous'
            },
            {
                'name': 'emotion_sad',
                'type': 'continuous'
            },
            {
                'name': 'emotion_surprise',
                'type': 'continuous'
            },
            {
                'name': 'emotion_neutral',
                'type': 'continuous'
            },
            {
                'name': 'face_embedding',
                'type': 'continuous'
            }
        ]
    
    # Set up a dataloader for loading datasets
    edge_class = Edge
    train_nodes, val_nodes, test_nodes = None, None, None
    train_nodes_full, val_nodes_full, test_nodes_full = None, None, None # Keep track of full sets for caching

    # --- Node Loading --- 
    node_loading_start = time.time()
    
    if args.use_cached:
        print(f"Attempting to load nodes from cache: {args.cache_file}")
        # Load potentially balanced sets based on flags
        train_nodes = load_cached_nodes(args.cache_file, 'train', balanced=args.fair_train)
        val_nodes = load_cached_nodes(args.cache_file, 'val', balanced=args.fair_test)
        test_nodes = load_cached_nodes(args.cache_file, 'test', balanced=args.fair_test)

        if train_nodes is None or val_nodes is None or test_nodes is None:
            print("Failed to load one or more splits from cache. Will attempt direct loading.")
            args.use_cached = False # Force direct load if cache failed
            train_nodes, val_nodes, test_nodes = None, None, None # Reset
        else:
            print("Successfully loaded nodes from cache.")
            # If loaded from cache, we don't necessarily have the 'full' versions unless
            # we load them separately or the cache format changes. For now, assume the
            # loaded lists are sufficient for downstream use, but caching below won't work correctly.
            # If needed, we could add logic here to load the 'full' versions too.
            # For simplicity, we set the *_full vars to the potentially balanced lists here,
            # acknowledging that re-caching might not save the original full sets.
            train_nodes_full = train_nodes
            val_nodes_full = val_nodes
            test_nodes_full = test_nodes


    if not args.use_cached:
        # Load nodes directly from datasets
        print("Loading nodes directly from dataset...")
        # Initialize the AIFaceDataset with correct parameters (using positional arguments)
        dataset = AIFaceDataset(data_root, ImageFileData, {}, AttributeNode, {"threshold": 2})
        
        # Load all nodes directly from the dataset (avoid using dataloader.load() which would load again)
        print("Loading nodes from dataset...")
        all_nodes = dataset.load()
            
        # Create node lists for each split
        print("Separating nodes by split...")
        train_nodes_full = [node for node in all_nodes if node.split == 'train']
        val_nodes_full = [node for node in all_nodes if node.split == 'val']
        test_nodes_full = [node for node in all_nodes if node.split == 'test']
        print(f"  Train: {len(train_nodes_full)}, Val: {len(val_nodes_full)}, Test: {len(test_nodes_full)}")

        # Cache the full nodes if requested
        if args.cache_nodes:
            print(f"Caching full node lists to {args.cache_file}...")
            # *** Pass the FULL lists to save_cached_nodes ***
            save_cached_nodes(train_nodes_full, val_nodes_full, test_nodes_full, args.cache_file, target_num_nodes=args.cached_nodes)

        # Apply balancing based on flags to get the lists used for graph building
        print("Applying balancing based on flags for graph construction...")
        train_nodes = balance_nodes_by_subgroup(train_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_train else train_nodes_full
        val_nodes = balance_nodes_by_subgroup(val_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_test else val_nodes_full
        test_nodes = balance_nodes_by_subgroup(test_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_test else test_nodes_full
        print(f"  Final Train Nodes used for graph: {len(train_nodes)} ({'Balanced' if args.fair_train else 'Full'})")
        print(f"  Final Val Nodes used for graph: {len(val_nodes)} ({'Balanced' if args.fair_test else 'Full'})")
        print(f"  Final Test Nodes used for graph: {len(test_nodes)} ({'Balanced' if args.fair_test else 'Full'})")

    node_loading_time = time.time() - node_loading_start
    print(f"Node loading/balancing time: {node_loading_time:.2f} seconds")
    
    graph_cache_dir = "graph_cache"
    os.makedirs(graph_cache_dir, exist_ok=True)

    # Use the potentially balanced node lists (train_nodes, val_nodes, test_nodes)
    # for graph construction below.

    # Determine cache filename suffix based on whether balanced nodes were used for graph construction
    train_suffix = "balanced" if args.fair_train else "full"
    val_suffix = "balanced" if args.fair_test else "full"
    test_suffix = "balanced" if args.fair_test else "full"

    q_thresh_str = f"{args.quality_threshold:.3f}"
    s_thresh_str = f"{args.symmetry_threshold:.3f}"
    e_thresh_str = f"{args.embedding_threshold:.3f}"

    for split_name, nodes_to_use, suffix in [
        ('train', train_nodes, train_suffix),
        ('val', val_nodes, val_suffix),
        ('test', test_nodes, test_suffix)
    ]:
        # Extract dataset name from data_root path (Corrected)
        dataset_name = os.path.basename(os.path.normpath(data_root)) if data_root else "unknown_dataset"
        
        cache_filename = os.path.join(
            graph_cache_dir,
            # Include the balancing status in the cache filename
            f"{dataset_name}_{split_name}_{suffix}_nodes_{len(nodes_to_use)}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_graph.pkl"
        )

        # Check/Load Graph Cache
        graph = None
        loaded_from_cache = False

        if os.path.exists(cache_filename):
            try:
                print(f"\nFound edge cache file: {cache_filename}. Attempting to load.")
                # 1. Load Nodes (ensure nodes are loaded for the split)
                split_nodes = train_nodes_full if split_name == 'train' else val_nodes_full if split_name == 'val' else test_nodes_full
                if not split_nodes:
                    raise ValueError(f"Nodes for split '{split_name}' not found or loaded.")
                
                # 2. Load Edge List
                with open(cache_filename, 'rb') as f:
                    edge_list = dill.load(f)
                    
                # 3. Reconstruct Graph
                print(f"Creating graph shell for {split_name} with {len(split_nodes)} nodes.")
                graph = HyperGraph(split_nodes) 
                print(f"Adding {len(edge_list)} edges from cache...")
                graph.add_edges_from_list(edge_list)
                
                print(f"Successfully loaded and reconstructed {split_name} graph from edge cache.")
                loaded_from_cache = True
            except Exception as e:
                print(f"\nError loading/reconstructing {split_name} graph from edge cache {cache_filename}: {e}. Regenerating.")
                graph = None # Ensure regeneration if loading fails

        # --- Build Graph if not loaded from cache --- 
        if not loaded_from_cache:
            # Ensure nodes are available
            split_nodes = train_nodes_full if split_name == 'train' else val_nodes_full if split_name == 'val' else test_nodes_full
            if not split_nodes:
                 print(f"Error: Nodes for split '{split_name}' not available for building graph.")
                 continue # Or handle error appropriately
                 
            print(f"\nBuilding graph for {split_name} split ({len(split_nodes)} nodes)... No suitable cache found or --use-cached=False.")
            # Use the dataloader to build the graph
            # Assuming dataloader.build_graph returns the graph object directly now
            # If it still returns a tuple, adjust accordingly (e.g., graph = dataloader.build_graph(...)[0] )
            dataloader = HierarchicalDeepfakeDataloader(
                datasets=[], 
                edge_class=edge_class,
                test_mode=False,  # Don't limit nodes
                visualize=False,  # Don't create visualizations during search
                show_viz=False,
                quality_threshold=args.quality_threshold,
                symmetry_threshold=args.symmetry_threshold,
                embedding_threshold=args.embedding_threshold,
                silent_mode=True  # Disable internal progress bars and logging during grid search
            )
            graph_build_result = dataloader._build_graph_standard(nodes_to_use, split_name) if split_name == 'train' else HyperGraph(nodes_to_use)
            
            # Handle potential tuple return from build_graph_standard
            if isinstance(graph_build_result, tuple):
                 graph = graph_build_result[0] 
                 # Potentially handle other elements in the tuple if needed
            else:
                 graph = graph_build_result
            
            # --- Save Edge List to Cache --- 
            if graph: # Only save if graph build was successful
                try:
                    print(f"Extracting edge list for {split_name} graph...")
                    edge_list_to_save = graph.get_edge_list()
                    print(f"Saving {len(edge_list_to_save)} edges for {split_name} graph to cache: {cache_filename}")
                    with open(cache_filename, 'wb') as f:
                        dill.dump(edge_list_to_save, f) # Save the list, no recurse needed
                    print(f"Saved {split_name} edge list to cache.")
                except Exception as e:
                    print(f"Error extracting or saving {split_name} edge list to cache file {cache_filename}: {e}")
            else:
                print(f"Skipping cache save for {split_name} due to build failure.")

        # --- Store Graph --- 
        # This part assumes 'graph' holds the final HyperGraph object, either loaded or built
        if graph:
             print(f"[Debug] Type of graph object for {split_name} before assignment: {type(graph)}")
             if split_name == 'train':
                 train_graph = graph # Assign the graph object
             elif split_name == 'val':
                 val_graph = graph
             else:
                 test_graph = graph
        else:
            print(f"Error: Failed to load or build graph for {split_name}. Skipping assignment.")
            # Decide how to handle this - exit, continue, assign None? 
            # Assigning None might cause issues later if not checked
            if split_name == 'train': train_graph = None
            elif split_name == 'val': val_graph = None
            else: test_graph = None 

    # Check if all graphs were loaded/built successfully before proceeding
    if train_graph is None or val_graph is None or test_graph is None:
        print("\nError: One or more graphs could not be loaded or built. Exiting.")
        sys.exit(1)
        
    graph_construction_time = time.time() - node_loading_start

    # Performance Reporting & Validation
    print("\nPerformance:")
    print(f"Total time: {graph_construction_time:.2f} seconds")
    print(f"  - Node loading: {node_loading_time:.2f} seconds ({node_loading_time/graph_construction_time*100:.1f}%)")
    print(f"  - Graph construction: {(graph_construction_time - node_loading_time):.2f} seconds ({(graph_construction_time - node_loading_time)/graph_construction_time*100:.1f}%)")

    # Validate graph objects
    if not train_graph or not val_graph or not test_graph:
        print("\nError: One or more graphs failed to build or load. Cannot proceed with validation.")
        return
    if not train_graph.nodes or not val_graph.nodes or not test_graph.nodes:
        print("\nError: One or more graphs have no nodes. Cannot proceed with validation.")
        return

    total_nodes = (len(train_graph.get_nodes()) + 
                   len(val_graph.get_nodes()) + 
                   len(test_graph.get_nodes()))
    
    print(f"Processed {total_nodes} nodes")
    print(f"Overall processing speed: {total_nodes / graph_construction_time:.2f} nodes/second")
    print(f"Graph construction speed: {total_nodes / (graph_construction_time - node_loading_time):.2f} nodes/second")
    
    # Count total edges
    train_edges = sum(len(node.edges) for node in train_graph.get_nodes()) // 2
    val_edges = sum(len(node.edges) for node in val_graph.get_nodes()) // 2
    test_edges = sum(len(node.edges) for node in test_graph.get_nodes()) // 2
    total_edges = train_edges + val_edges + test_edges
    
    print(f"Created {total_edges} total edges")
    print(f"Edge creation speed: {total_edges / (graph_construction_time - node_loading_time):.2f} edges/second")
    
    # Print average degree (edges per node)
    print(f"Average degree: {(total_edges * 2) / len(train_graph.get_nodes()):.2f}")

    #print(f"Node example: {list(train_graph.get_nodes())[0]}")
    #sys.exit()

    with capture_output(logfile.name) as logpath:
        print(f"Starting test run, logging to: {logfile}")
    
        # Create graph managers for each split
        train_manager = NoGraphManager(train_graph)
        val_manager = NoGraphManager(val_graph)
        test_manager = NoGraphManager(test_graph)

        # --- Get Nodes from Graphs for DataLoaders ---
        val_nodes_from_graph = val_manager.graph.get_nodes()
        test_nodes_from_graph = test_manager.graph.get_nodes()
        print(f"Retrieved {len(val_nodes_from_graph)} validation nodes and {len(test_nodes_from_graph)} test nodes from graph managers.")

        # ===============================
        # Model/Trainer Setup
        # ===============================
        # Define architectures to test
        cnn_architectures = [
            "swintransformdf",
            "resnestdf", 
            "effnetdf",
            #"mesonetdf",
            #"squeezenetdf",
            #"vistransformdf",
            
        ]

        random.seed(13247987501)
        
        # Define traversal types to compare
        #traversal_types = ["i-value-cluster-hop", "i-value", "comprehensive", "random"] # Added 'i-value-cluster-hop'
        traversal_types = ["comprehensive", "i-value-cluster-hop"]
        #traversal_types = ["comprehensive"]

        
        # Test each architecture with both traversal types
        for arch in cnn_architectures:
            print(f"\n{'='*80}")
            print(f"Testing {arch} architecture")
            print(f"{'='*80}\n")
            
            for traversal_type in traversal_types:
                print(f"\n{'-'*40}")
                print(f"Using {traversal_type} traversal")
                print(f"{'-'*40}\n")
                
                try:

                    # Create traversals for training
                    # Use more pointers and adjust steps based on graph sizes
                    train_size = len(train_manager.graph.get_nodes())
                    val_size = len(val_manager.graph.get_nodes())
                    test_size = len(test_manager.graph.get_nodes())
                    
                    print(f"\nGraph sizes:")
                    print(f"Train: {train_size} nodes")
                    print(f"Val: {val_size} nodes")
                    print(f"Test: {test_size} nodes")
                    
                    # Calculate appropriate number of steps
                    train_steps = 1000
                    val_steps = 1000
                    test_steps = None  # Use None to visit all test nodes
                    
                    # Create Traversal instances
                    if traversal_type == "comprehensive":
                        train_traversal = ComprehensiveTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    elif traversal_type == "random":
                        train_traversal = RandomTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    elif traversal_type == "i-value":
                        # Use the trainer's method to get the configured IValueTraversal
                        train_traversal = IValueTraversal(
                            graph=train_manager.graph,
                            num_pointers=1,
                            num_steps=train_steps
                        )
                    elif traversal_type == "i-value-cluster-hop":
                        # Instantiate the cluster hop traversal, passing the trainer
                        print(f"Bias hop period: {args.bias_hop_period}")
                        train_traversal = IValueTraversalClusterHop(
                             graph=train_manager.graph, 
                             num_pointers=1, 
                             num_steps=train_steps,
                             bias_hop_period=args.bias_hop_period
                        )
                        print(f"train_traversal.bias_hop_period: {train_traversal.bias_hop_period}")
                    else:
                        raise ValueError(f"Unsupported traversal type for training: {traversal_type}")

                    # Create model with adjusted learning rate
                    model = CNNModel(
                        f"/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/{arch}_{traversal_type}_{timestamp}.pt",
                        arch,
                        1e-4, # WAS: 0.001 - Reduced LR for debugging
                        True,
                        device=device  # Pass the device
                    )
                    
                    # --- Early Stopping & Best Model Checkpoint --- 
                    best_val_accuracy = 0.0
                    epochs_no_improve = 0
                    patience = 10 # Example patience
                    best_model_checkpoint_path = f"checkpoints/{arch}_{traversal_type}_best.pth"
                    os.makedirs(os.path.dirname(best_model_checkpoint_path), exist_ok=True) # Ensure dir exists

                    # Find the section starting around line 1311
                    if traversal_type == "i-value" or traversal_type == "i-value-cluster-hop":
                        # 1. Create IValueTrainer *without* train_traversal
                        trainer = IValueTrainer(
                            graphmanager=train_manager,
                            models=[model], # Use model
                            device=device,
                            attribute_metadata=attribute_metadata,
                            use_bias_loss_in_training=False, # Example
                            bias_loss_weight=args.bias_loss_weight,
                            loss_fn=criterion,
                            train_traversal=train_traversal # Explicitly pass None
                        )
                    
                        # 2. Create the specific IValueTraversal needed
                        if traversal_type == "i-value":
                            # This assumes IValueTraversal also needs the trainer or its components
                            # If IValueTraversal doesn't need the trainer, adjust its __init__ and this call
                            # For now, assuming it might need the trainer similar to cluster hop
                            train_traversal = IValueTraversal( # Replace with actual IValueTraversal class if different
                                graph=train_manager.graph,
                                num_pointers=1,
                                num_steps=train_steps,
                                trainer=trainer # Pass the trainer instance
                            )
                        elif traversal_type == "i-value-cluster-hop":
                            train_traversal = IValueTraversalClusterHop(
                                graph=train_manager.graph,
                                num_pointers=1,
                                num_steps=train_steps,
                                trainer=trainer, # Pass the trainer instance
                                bias_hop_period=args.bias_hop_period
                            )
                    
                        # 3. Set the traversal back on the trainer
                        trainer.set_train_traversal(train_traversal)

                    else:  # random or comprehensive traversal
                        # Example ExperimentTrainer call:
                            trainer = ExperimentTrainer(
                            graphmanager=train_manager,
                            train_traversal=train_traversal,
                            models=[model], # Wrap the primary CNN model in a list
                            device=device,
                            traversal_type=traversal_type,
                            attribute_metadata=attribute_metadata, # Pass metadata if needed by init
                            loss_fn=criterion # <-- ADD THIS
                        )
                    
                    
                        
                    # val_traversal = ComprehensiveTraversal(val_manager.graph, num_pointers=1, num_steps=val_steps)
                    # test_traversal = ComprehensiveTraversal(test_manager.graph, num_pointers=1, num_steps=test_steps)
                    
                    print(f"\nTraversal settings:")
                    print(f"Train: {train_steps} steps with 1 pointers")
                    print(f"Val: {val_steps} steps with 1 pointers")
                    print(f"Test: All nodes with 1 pointers")
                    
                    # Update trainer with correct traversals
                    trainer.train_traversal = train_traversal
                    # trainer.val_traversal = val_traversal # Removed: Using val_loader now
                    # trainer.test_traversal = test_traversal # Removed: Using test_loader now
                    
                    try:
                        print(f"Training {arch} with {traversal_type} traversal...")
                        for epoch in range(args.num_epochs):
                            print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
                            train_start_time = time.time()
                            train_distribution = None # Initialize distribution as None
                            train_metrics, train_distribution = trainer.train(epoch) if isinstance(trainer, ExperimentTrainer) else trainer.train() # Pass epoch

                            print(f"  Train Metrics: {train_metrics}")
                            
                            # --- Print Training Attribute Distribution --- 
                            if train_distribution: # Check if distribution was returned and is not None
                                print("  Training Attribute Distribution for this Epoch:")
                                # Use json.dumps for pretty printing the nested defaultdict
                                print(json.dumps(train_distribution, indent=4))
                            else:
                                print("  No attribute distribution tracked or returned for this trainer type.")
                            # ---------------------------------------------

                            # --- Validation Step --- 
                            if val_nodes_from_graph:
                                model = trainer.models[0] if trainer.models else None
                                if not model:
                                    print(f"ERROR: No model found in trainer for {arch} with {traversal_type}. Skipping.")
                                    continue

                                model.eval() # Set model to evaluation mode
                                val_metrics = evaluate_model(
                                    model=model,
                                    nodes_to_evaluate=random.sample(val_nodes_from_graph, min(len(val_nodes_from_graph), val_steps)),
                                    loss_fn=criterion,
                                    batch_size=args.batch_size,
                                    bias_loss_fn=trainer.bias_loss if isinstance(trainer, IValueTrainer) else None,
                                    device=device,
                                    desc="Validation",
                                    attribute_metadata=attribute_metadata
                                )
                                
                                current_val_accuracy = val_metrics.get('accuracy', 0.0)

                                if current_val_accuracy > best_val_accuracy:
                                    best_val_accuracy = current_val_accuracy
                                    best_epoch = epoch + 1
                                    # Save primary model checkpoint
                                    model.save_checkpoint(best_model_checkpoint_path)
                                    # Save DQN checkpoint(s) if using IValueTrainer
                                    if isinstance(trainer, IValueTrainer) and trainer.dqns:
                                        for i, dqn_model in enumerate(trainer.dqns):
                                            dqn_checkpoint_path = best_model_checkpoint_path.replace('.pt', f'_dqn{i}.pt')
                                            dqn_model.save_checkpoint(dqn_checkpoint_path)
                                    print(f"New best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}. Model(s) saved to {best_model_checkpoint_path} (and DQN paths if applicable)")
                                elif args.load_last_checkpoint and epoch > 0: # Check if we need to load the previous best
                                    print(f"Validation accuracy did not improve. Current: {val_metrics['accuracy']:.4f}, Best: {best_val_accuracy:.4f}. Loading checkpoint from epoch {best_epoch}.")
                                    if os.path.exists(best_model_checkpoint_path):
                                        # Load primary model checkpoint
                                        model.load_checkpoint(best_model_checkpoint_path)
                                        # Load DQN checkpoint(s) if using IValueTrainer
                                        if isinstance(trainer, IValueTrainer) and trainer.dqns:
                                            for i, dqn_model in enumerate(trainer.dqns):
                                                dqn_checkpoint_path = best_model_checkpoint_path.replace('.pt', f'_dqn{i}.pt')
                                                if os.path.exists(dqn_checkpoint_path):
                                                    dqn_model.load_checkpoint(dqn_checkpoint_path)
                                                else:
                                                    print(f"Warning: DQN Checkpoint {dqn_checkpoint_path} not found, cannot reload DQN {i}.")
                                    else:
                                        print(f"Warning: Checkpoint {best_model_checkpoint_path} not found, cannot reload primary model.")
                                else:
                                    print(f"Validation accuracy did not improve from {best_val_accuracy:.4f} (best epoch {best_epoch})")
 
                        # Load best model for final testing
                        if args.num_epochs > 0:
                            #print(f"\nLoading best model from epoch {best_epoch} for final testing (Val Acc: {best_val_accuracy:.4f}) from {best_model_checkpoint_path}")
                            if os.path.exists(best_model_checkpoint_path):
                                try:
                                    # Load primary model checkpoint
                                    model.load_checkpoint(best_model_checkpoint_path)
                                    model.eval() # Ensure model is in eval mode for testing
                                    print("Best model loaded successfully.")
                                    
                                    print(f"\nRunning final evaluation on Test Set...")
                                    test_metrics = evaluate_model(
                                        model=model,
                                        nodes_to_evaluate=test_nodes_from_graph,
                                        loss_fn=criterion,
                                        batch_size=args.batch_size,
                                        bias_loss_fn=trainer.bias_loss if isinstance(trainer, IValueTrainer) else None,
                                        device=device,
                                        desc="Final Test Evaluation",
                                        attribute_metadata=attribute_metadata
                                    )
                                    print("\n--- Final Test Results --- ")
                                    print(json.dumps(test_metrics, indent=2))
                                    # Optional: Save test_metrics to a results file
                                
                                except Exception as e_load:
                                     print(f"ERROR loading best model checkpoint: {e_load}")
                                     log_exception(logfile, *sys.exc_info())
                                     test_metrics = {"error": "Failed to load best model"}
                            else:
                                print(f"ERROR: Best model checkpoint not found at {best_model_checkpoint_path}. Testing with the last state model.")
                                model.eval()
                                test_metrics = evaluate_model(
                                    model=model,
                                    nodes_to_evaluate=test_nodes_from_graph,
                                    loss_fn=criterion,
                                    batch_size=args.batch_size,
                                    bias_loss_fn=trainer.bias_loss if isinstance(trainer, IValueTrainer) else None,
                                    device=device,
                                    desc="Final Test Set (Last State)",
                                    attribute_metadata=attribute_metadata
                                )
                                print("\n--- Final Test Results (Last State Model) --- ")
                                print(json.dumps(test_metrics, indent=2))

                    except Exception as e_inner_loop:
                        print(f"\nError during training/evaluation loop for {arch} with {traversal_type}: {str(e_inner_loop)}")
                        log_exception(logfile, *sys.exc_info())
                        continue # Continue with the next configuration

                    # --- Cleanup Old Test Call --- 
                    # print(f"Testing {arch} with {traversal_type} traversal...")
                    # test_metrics = trainer.test() # Removed

                    # --- Calculate and Save Final I-Values (Keep or remove based on needs) --- 
                        final_i_values = {}
                        node_data_list = []
                        test_graph_nodes = test_manager.graph.get_nodes() # Get test nodes
                        
                        if isinstance(trainer, IValueTrainer):
                            print("Calculating final I-values for test nodes...")
                        # This assumes get_all_final_i_values exists and works correctly
                        # final_i_values = trainer.get_all_final_i_values(graph_split='test') 
                        # Placeholder: 
                        final_i_values = {}
                        print(f"Placeholder: Calculated I-values for {len(final_i_values)} nodes.")
                            
                        # Gather data for all test nodes
                        print("Gathering node data for output CSV...")
                        for node in tqdm(test_graph_nodes, desc="Processing test nodes for CSV"): 
                            node_info = {'node_id': node.node_id}
                            # Add attributes if they exist
                            if hasattr(node, 'attributes') and isinstance(node.attributes, dict):
                                node_info.update(node.attributes)
                            # Add ground truth if available
                            node_info['ground_truth'] = node.label if hasattr(node, 'label') else 'N/A'
                            # Add predicted label if available from test_metrics? Requires mapping node_id to prediction
                            # node_info['predicted_label'] = ... # This needs linking test_metrics output to nodes
                            # Add I-value if available
                            node_info['i_value'] = final_i_values.get(node.node_id, 'N/A')
                            node_data_list.append(node_info)
                            
                    # Save detailed results to CSV
                        if node_data_list:
                            output_csv_path = os.path.join(args.log_dir if args.log_dir else '.', f"detailed_results_{arch}_{traversal_type}.csv")
                            try:
                                df_results = pd.DataFrame(node_data_list)
                                df_results.to_csv(output_csv_path, index=False)
                                print(f"Detailed node results saved to {output_csv_path}")
                            except Exception as e_csv:
                                print(f"Error saving detailed results CSV: {e_csv}")
                        else:
                            print("No node data collected for CSV output.")

                except Exception as e:
                    log_exception(logfile, *sys.exc_info())
                    print(f"\nOuter Error setting up {arch} with {traversal_type}: {str(e)}")
                    continue  # Continue with next configuration

                # --- Plotting Subgroup I-Values (After Training Loop) --- 
                if hasattr(trainer, 'train_traversal') and hasattr(trainer.train_traversal, 'get_hop_i_value_history'):
                    hop_history = trainer.train_traversal.get_hop_i_value_history()
                    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = os.path.join(args.log_dir, f"{run_timestamp}_subgroup_i_values.png")
                    plot_subgroup_i_values(hop_history, plot_filename)
                else:
                    print("Trainer or traversal does not support hop history retrieval.")
                # ----------------------------------------------------------
            
            
    print("\nDone!")
    
    if logpath:
        print(f"Output captured in: {logpath}")

if __name__ == "__main__":
    main()
