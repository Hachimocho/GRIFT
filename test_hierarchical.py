"""
Test script for the Hierarchical Deepfake Dataloader

This script tests the new hierarchical graph construction approach which:
1. Groups nodes by categorical attributes (race-gender combinations)
2. Creates fully-connected subgraphs within each group
3. Applies threshold-based filtering for quality metrics, symmetry, embeddings, etc.
"""
import time
import os
import cv2
from collections import defaultdict
import sys
import logging  
import traceback 
import json 
import random 
import torch
import torch.nn as nn
from datetime import datetime
import dill

# Import utilities from the new helper module
from test_helpers.logging_utils import NullHandler, capture_output, log_exception, set_seed
from test_helpers.args_utils import parse_args
from test_helpers.data_graph_utils import (
    balance_nodes_by_subgroup, save_cached_nodes, load_cached_nodes,
    run_threshold_grid_search, visualize_search_results, plot_subgroup_i_values,
    load_and_prepare_data_splits # Added import for the new function
)

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
from traversals.IValueTraversalClusterHop import IValueTraversalClusterHop 
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
from torch.utils.data import Dataset, DataLoader 
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

    # Define the Edge class to be used for graph construction
    edge_class = Edge

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
    
    # Call the new helper function to load and prepare data splits
    train_nodes, val_nodes, test_nodes, \
    train_nodes_full, val_nodes_full, test_nodes_full, \
    node_loading_time = load_and_prepare_data_splits(args, data_root)
    
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
        
    graph_construction_time = time.time() - node_loading_time

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
