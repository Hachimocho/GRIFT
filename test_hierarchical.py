"""
Test script for the Hierarchical Deepfake Dataloader

This script tests the new hierarchical graph construction approach which:
1. Groups nodes by categorical attributes (race-gender combinations)
2. Creates fully-connected subgraphs within each group
3. Applies threshold-based filtering for quality metrics, symmetry, embeddings, etc.

Updated to support the new AdaptiveTrainer architecture with:
- Single-traversal mode: Use one traversal method throughout training
- Switch-traversal mode: Switch between different traversal methods during training
"""
# Reproducibility bootstrap MUST run before torch/numpy are imported.
# PYTHONHASHSEED is consumed at interpreter startup and CUBLAS_WORKSPACE_CONFIG when
# the cuBLAS handle is created, so neither can be set usefully once those libraries
# are loaded. This re-execs the process once with the right environment if needed,
# which works regardless of launcher -- run_reproducible.sh cannot guarantee it
# because web_ui/gpu_queue_manager.py invokes this script directly.
#
# Guarded on __name__ == "__main__": re-execing replaces the *whole process*, so
# doing it on import would kill any host that merely imports this module (pytest,
# the web UI, tooling) rather than running it as a script.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
if __name__ == "__main__":
    from test_helpers.bootstrap import ensure_deterministic_env as _ensure_deterministic_env
    _ensure_deterministic_env()

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
import numpy as np
import argparse
import faulthandler
import signal
import resource
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utilities from the new helper module
from test_helpers.logging_utils import NullHandler, capture_output, log_exception
from test_helpers.determinism import (
    assert_strict_invariants, configure_determinism, is_strict, rng_for, run_fingerprint,
    swallow_or_raise,
)
from test_helpers.cache_keys import cache_filenames, graph_cache_key
from test_helpers.args_utils import parse_args
from test_helpers.data_graph_utils import (
    balance_nodes_by_subgroup, save_cached_nodes, load_cached_nodes,
    run_threshold_grid_search, visualize_search_results, plot_subgroup_i_values,
    load_and_prepare_data_splits, check_graph_cache_compatibility, resolve_ai_face_data_root
)

# Import visualization modules
from trainers.IValueVisualizationTracker import IValueVisualizationTracker
from trainers.BiasHopVisualizer import BiasHopVisualizer
from trainers.BiasMetricsTracker import BiasMetricsTracker

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

# Import AdaptiveTrainer (unified trainer architecture)
from trainers.AdaptiveTrainer import AdaptiveTrainer

from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager
from managers.GraphReductionManager import GraphReductionManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal, IValueTraversalSubcluster
from traversals.IValueTraversalClusterHop import (
    IValueTraversalClusterHop, IValueTraversalClusterHopSubcluster,
)
from traversals.RandomTraversal import RandomTraversal
from models.CNNModel import CNNModel
from models.uncertainty import GraphDistanceUncertainty, PredictionBundle
from models.uncertainty.capabilities import (
    describe_detector, supported_detectors, validate_architectures,
)
from evaluation.uq.records import PredictionRecordCollector, RecordCollectionError
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

def _load_node_data(node, model):
    """Helper function to load a single node's data. Used for parallel loading."""
    try:
        node_data = node.get_data()
        if node_data:
            img = node_data.load_data()
            label = node.get_label()
            if img is not None and label is not None:
                # Apply transformations using the model's internal method
                img_tensor = model.transform(img)
                
                # Handle tuple labels safely
                try:
                    if isinstance(label, (tuple, list)):
                        label_value = float(label[0])
                    else:
                        label_value = float(label)
                    return (img_tensor, label_value, node)
                except (ValueError, TypeError, IndexError) as e:
                    return None
        return None
    except Exception as e_load:
        return None

def evaluate_model(model, nodes_to_evaluate, loss_fn, batch_size, bias_loss_fn=None, device='cuda', desc="Evaluating", attribute_metadata=None, num_workers=4, *, record_collector=None):
    """Evaluates the model on the provided nodes, calculates standard metrics,
       and optionally calculates bias metrics based on categorical attributes.

    Args:
        num_workers: Number of parallel workers for image loading (default: 4)
        record_collector: Optional PredictionRecordCollector. When supplied, one row
            per sample is accumulated (probability, logit, per-sample uncertainty,
            source group, demographics) for the uncertainty benchmark. Without it
            this function behaves exactly as before -- the continuous scores are
            thresholded and discarded, so calibration and selective prediction are
            not recoverable afterwards.
    """
    model.eval() # Ensure model is in evaluation mode
    model.model.to(device)

    # Verify eval() actually took effect, rather than assuming it did.
    #
    # `_load_node_data` calls model.transform() from worker threads, and CNNModel's
    # transform dispatches on `current_mode`: in train mode it applies
    # RandomHorizontalFlip / RandomRotation / ColorJitter / RandomAffine /
    # RandomErasing, all drawing on the global torch RNG from several threads at
    # once. A model whose eval() fails to clear that mode would therefore randomize
    # its own evaluation, and silently -- the metrics would simply be noisier.
    current_mode = getattr(model, 'current_mode', 'eval')
    if current_mode != 'eval':
        raise RuntimeError(
            f"evaluate_model called model.eval() but current_mode is still "
            f"{current_mode!r}. Train-mode transforms are stochastic and run in worker "
            f"threads, so evaluating in this state would randomize the results."
        )

    total_loss = 0.0
    total_bias_loss = 0.0
    correct_predictions = 0
    total_nodes_processed = 0
    nodes_in_dataset = len(nodes_to_evaluate)
    num_batches = (nodes_in_dataset + batch_size - 1) // batch_size

    print(f"\nRunning inference for {desc} (Dataset Size: {nodes_in_dataset}, Batch Size: {batch_size}, Workers: {num_workers})...")

    all_predictions = []
    all_labels = []
    uncertainty_sums = defaultdict(float)
    uncertainty_counts = defaultdict(int)
    batches_failed = 0
    first_inference_error = None
    if record_collector is not None:
        record_collector.note_requested(nodes_in_dataset)
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

            # Load data for the current batch in parallel, but collect the results in
            # *submission* order rather than completion order. `as_completed` made the
            # within-batch row order vary run to run, which changes the floating-point
            # reduction order in the loss and metrics and so produces small,
            # irreproducible drift. Collecting by submission keeps the 4-8x I/O
            # parallelism that this executor exists for while making the batch
            # deterministic -- and it is what lets the benchmark assign stable
            # per-sample record ids.
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_load_node_data, node, model) for node in batch_nodes]
                for future in futures:
                    result = future.result()
                    if result is not None:
                        img_tensor, label_value, node = result
                        batch_images_loaded.append(img_tensor)
                        batch_labels_loaded.append(label_value)
                        batch_nodes_loaded.append(node)

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
                prediction_bundle = None
                if hasattr(model, 'forward_with_uncertainty'):
                    prediction_bundle = model.forward_with_uncertainty(
                        batch_images_tensor,
                        nodes=batch_nodes_loaded,
                        use_mc_dropout=getattr(model, 'mc_dropout_samples', 0) > 1,
                    )
                    outputs = prediction_bundle.logits
                    probabilities = prediction_bundle.probabilities
                    preds = prediction_bundle.predictions
                    uncertainty_summary = getattr(model, 'summarize_uncertainty', lambda _: {})(prediction_bundle)
                    for name, value in uncertainty_summary.items():
                        uncertainty_sums[name] += float(value) * len(batch_nodes_loaded)
                        uncertainty_counts[name] += len(batch_nodes_loaded)
                else:
                    outputs = model(batch_images_tensor)

                    # Safety check: Handle unexpected output types
                    if isinstance(outputs, tuple):
                        print(f"WARNING: Model returned tuple instead of tensor: {type(outputs)}, length: {len(outputs)}")
                        if len(outputs) > 0 and hasattr(outputs[0], 'size'):
                            print(f"Using first element of tuple: {outputs[0].shape}")
                            outputs = outputs[0]
                        else:
                            print(f"ERROR: Cannot extract valid tensor from tuple: {[type(x) for x in outputs]}")
                            continue
                    elif not hasattr(outputs, 'size'):
                        print(f"WARNING: Model output has no .size() method: {type(outputs)}")
                        continue

                    probabilities = torch.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()

                correct = (preds == batch_labels_tensor).sum().item()
                correct_predictions += correct
                current_batch_size = batch_labels_tensor.size(0)
                total_nodes_processed += current_batch_size

                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(
                        prediction_bundle if prediction_bundle is not None else outputs,
                        batch_labels_tensor,
                        base_criterion=loss_fn,
                    )
                else:
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
            except RecordCollectionError:
                # Never swallowed: a bug in record collection must not be reported as
                # an inference failure, and must not quietly reduce coverage.
                raise
            except Exception as e_inf:
                 print(f"\nError during model inference or loss calculation in {desc}: {e_inf}")
                 if record_collector is not None:
                     record_collector.note_batch_failure(e_inf)
                 batches_failed += 1
                 if first_inference_error is None:
                     first_inference_error = f"{type(e_inf).__name__}: {e_inf}"
                 # Clear GPU cache on error to prevent memory buildup
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                 swallow_or_raise(e_inf, f"evaluate_model[{desc}] batch {i}")
                 continue # Skip batch on inference error

            # --- Per-sample records for the uncertainty benchmark ---
            # Collected here because this is the only point where the prediction
            # bundle, the loaded nodes, and the labels are all simultaneously live.
            if record_collector is not None:
                record_collector.add_batch(
                    batch_nodes_loaded, batch_labels_tensor, prediction_bundle
                    if prediction_bundle is not None
                    else PredictionBundle(logits=outputs, probabilities=probabilities).with_predictions(),
                    batch_index=i,
                )

            # --- Store Predictions and Labels for Metrics ---
            predictions = probabilities.cpu().numpy() > 0.5
            current_labels = batch_labels_tensor.cpu().numpy().astype(int)
            all_predictions.extend(predictions.astype(int))
            all_labels.extend(current_labels)
            
            # Clear GPU cache periodically to prevent memory buildup
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Associate predictions/labels with nodes for bias calc ---
            node_results = {}
            if attribute_metadata and categorical_attrs: # Check again in case it was disabled
                 for node_idx, node in enumerate(batch_nodes_loaded):
                      node_results[node.node_id] = {
                           'prediction': predictions[node_idx],
                           'label': current_labels[node_idx],
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
                        # Filter categorical attributes to only include race and gender for subgroup construction
                        race_gender_attrs = [attr for attr in categorical_attrs 
                                           if attr['name'] in ['Ground Truth Gender', 'Ground Truth Race']]
                        
                        for cat_attr in race_gender_attrs:
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
        # Raise rather than report accuracy 0.0. Reporting zero made a total failure
        # indistinguishable from a genuinely terrible model -- which is exactly how
        # the evidential/MC-dropout crash presented for as long as it did: every
        # batch raised, was swallowed by the handler above, and the run cheerfully
        # printed "Accuracy=0.00%".
        raise RuntimeError(
            f"evaluate_model({desc}) processed 0 of {nodes_in_dataset} nodes. "
            f"{batches_failed} batch(es) failed"
            + (f"; first error was {first_inference_error}" if first_inference_error else "")
            + ". Refusing to report accuracy 0.0, which would be indistinguishable "
              "from a model that simply predicts badly."
        )

    if batches_failed:
        coverage = total_nodes_processed / max(1, nodes_in_dataset)
        print(
            f"\nWARNING: {batches_failed} batch(es) failed during {desc}; metrics "
            f"cover {coverage:.1%} of the requested nodes. First error: "
            f"{first_inference_error}"
        )
    final_metrics['batches_failed'] = batches_failed
    final_metrics['coverage'] = total_nodes_processed / max(1, nodes_in_dataset)

    if bias_loss_fn:
         if total_nodes_processed > 0:
             final_metrics['average_bias_loss'] = total_bias_loss / total_nodes_processed # Average bias loss per successfully processed sample
         else:
             final_metrics['average_bias_loss'] = float('nan')

    if record_collector is not None:
        # A small, JSON-safe summary only. The rows themselves go to disk: this dict is
        # json.dumps'd to stdout and scraped by the GPU queue manager, so embedding a
        # 400k-row table here would bloat every log and break the parser.
        final_metrics['records'] = record_collector.summary()

    if uncertainty_sums:
        final_metrics['uncertainty_summary'] = {
            name: uncertainty_sums[name] / max(1, uncertainty_counts[name])
            for name in uncertainty_sums
        }

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
        race_gender_attrs = [attr['name'] for attr in categorical_attrs 
                           if attr['name'] in ['Ground Truth Gender', 'Ground Truth Race']]
        print(f"Bias calculation enabled for race-gender subgroups using attributes: {race_gender_attrs}")

    # --- Calculate Bias Metrics --- (Only if enabled and stats collected)
    if attribute_metadata and categorical_attrs and subgroup_stats:
        subgroup_accuracies = {}
        min_subgroup_acc = 1.0
        max_subgroup_acc = 0.0
        total_subgroup_abs_diff = 0.0
        num_subgroups = 0

        print("\n--- Race-Gender Subgroup Bias Analysis ---")
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
        bias_metrics['race_gender_subgroup_accuracies'] = subgroup_accuracies
        bias_metrics['race_gender_overall_bias'] = overall_bias
        bias_metrics['race_gender_average_subgroup_bias'] = average_subgroup_bias # Add average subgroup bias

        print(f"Race-Gender Overall Bias (Max Acc Diff across subgroups): {overall_bias:.4f}")
        print(f"Race-Gender Average Subgroup Bias (Avg Abs Diff from Overall Acc): {average_subgroup_bias:.4f}") # Print average subgroup bias

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

def create_traversal(traversal_type, graph, num_pointers=1, num_steps=1000, trainer=None, **kwargs):
    """Create a traversal instance based on type and parameters."""
    if traversal_type == "comprehensive":
        return ComprehensiveTraversal(graph, num_pointers=num_pointers, num_steps=num_steps)
    elif traversal_type == "random":
        return RandomTraversal(graph, num_pointers=num_pointers, num_steps=num_steps)
    elif traversal_type == "i-value":
        return IValueTraversal(
            graph=graph,
            num_pointers=num_pointers,
            num_steps=num_steps,
            trainer=trainer
        )
    elif traversal_type == "i-value-cluster-hop":
        bias_hop_period = kwargs.get('bias_hop_period', 100)
        return IValueTraversalClusterHop(
            graph=graph,
            num_pointers=num_pointers,
            num_steps=num_steps,
            trainer=trainer,
            bias_hop_period=bias_hop_period
        )
    elif traversal_type == "i-value-subcluster":
        # These two were advertised in --traversal-type's `choices` but had no
        # branch here, so selecting either raised "Unsupported traversal type"
        # immediately. Note both rely on graph.subclusters, which is populated by
        # Louvain -- and if python-louvain ("community") is not installed,
        # HyperGraph.assign_louvain_subclusters is a silent no-op and these fall
        # back to their no-subcluster paths.
        return IValueTraversalSubcluster(
            graph=graph,
            num_pointers=num_pointers,
            num_steps=num_steps,
            trainer=trainer,
            outlier_std=kwargs.get('outlier_std', 2.0),
            softmax_temp=kwargs.get('softmax_temp', 0.5),
        )
    elif traversal_type == "i-value-cluster-hop-subcluster":
        return IValueTraversalClusterHopSubcluster(
            graph=graph,
            num_pointers=num_pointers,
            num_steps=num_steps,
            trainer=trainer,
            bias_hop_period=kwargs.get('bias_hop_period', 100),
            outlier_std=kwargs.get('outlier_std', 2.0),
            softmax_temp=kwargs.get('softmax_temp', 0.5),
        )
    else:
        raise ValueError(f"Unsupported traversal type: {traversal_type}")

def parse_traversal_config(args):
    """Parse traversal configuration from command line arguments."""
    config = {
        'trainer_mode': getattr(args, 'trainer_mode', 'adaptive'),  # Default to 'adaptive' if not set
        'single_traversal': args.traversal_type,
        'enable_switching': args.enable_traversal_switching,
        'architectures': [arch.strip() for arch in args.architectures.split(',')],
        'test_all_traversals': args.test_all_traversals,
        'disconnected_switching': getattr(args, 'disconnected_switching', False)
    }
    
    if args.enable_traversal_switching:
        # Parse traversal sequence
        config['traversal_sequence'] = [t.strip() for t in args.traversal_sequence.split(',')]
        
        # Parse switch epochs
        try:
            config['switch_epochs'] = [int(e.strip()) for e in args.switch_epochs.split(',')]
        except ValueError:
            raise ValueError(f"Invalid switch epochs format: {args.switch_epochs}. Expected comma-separated integers.")
            
        # Validate sequence and epochs match
        if len(config['switch_epochs']) != len(config['traversal_sequence']) - 1:
            raise ValueError(f"Number of switch epochs ({len(config['switch_epochs'])}) must be one less than traversal sequence length ({len(config['traversal_sequence'])})")
    
    return config

def create_adaptive_trainer(train_manager, model, device, attribute_metadata, criterion, args):
    """Create and configure an AdaptiveTrainer instance."""
    trainer = AdaptiveTrainer(
        graphmanager=train_manager,
        models=[model],
        device=device,
        attribute_metadata=attribute_metadata,
        loss_fn=criterion,
        bias_loss_weight=args.bias_loss_weight,
        dqn_model_type=getattr(args, 'dqn_model', 'basic')
    )
    return trainer

def create_dqn_model(model_type, feature_dim, device, embedding_dim=512, **kwargs):
    """Create a DQN model instance based on type and parameters."""
    if model_type == "basic":
        from models.DQNModel import DQNModel
        return DQNModel(feature_dim, device, embedding_dim=embedding_dim)
    elif model_type == "residual":
        from models.EnhancedDQNModels import ResidualDQNModel
        return ResidualDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    elif model_type == "attention":
        from models.EnhancedDQNModels import AttentionDQNModel
        return AttentionDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    elif model_type == "conv_embedding":
        from models.EnhancedDQNModels import ConvEmbeddingDQN
        return ConvEmbeddingDQN(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    elif model_type == "ensemble":
        from models.EnhancedDQNModels import EnsembleDQNModel
        return EnsembleDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported DQN model type: {model_type}")

def create_model(arch, save_path, device, dqn_model_type="basic", **kwargs):
    """Create either a CNN model or DQN model based on architecture."""
    if arch.startswith("dqn_"):
        # Extract feature dimension from kwargs or use default
        feature_dim = kwargs.get('feature_dim', 128)  # Default feature dimension
        return create_dqn_model(dqn_model_type, feature_dim, device, **kwargs)
    else:
        # Create CNN model as before
        return CNNModel(
            save_path,
            arch,
            1e-4,
            True,
            device,
            uncertainty_head=kwargs.get('uncertainty_head', 'none'),
            mc_dropout_samples=kwargs.get('mc_dropout_samples', 0),
            batchensemble_members=kwargs.get('batchensemble_members', 4),
            sngp_hidden_dim=kwargs.get('sngp_hidden_dim', 256),
            sngp_rff_dim=kwargs.get('sngp_rff_dim', 256),
            uncertainty_dropout_rate=kwargs.get('uncertainty_dropout_rate', 0.2),
            graph_uncertainty_methods=kwargs.get('graph_uncertainty_methods', []),
            graph_degree_penalty_weight=kwargs.get('graph_degree_penalty_weight', 1.0),
            uncertainty_train_frequency=kwargs.get('uncertainty_train_frequency', 10),
            sngp_precision_policy=kwargs.get('sngp_precision_policy', 'per-epoch'),
            finetune=kwargs.get('finetune', False),
        )

def main():
    args = parse_args() # Parse args first
    graph_uncertainty_methods = [
        method.strip() for method in getattr(args, 'graph_uncertainty_methods', '').split(',') if method.strip()
    ]

    # Validate architectures up front. `--architectures` is a free-form string with
    # no argparse `choices`, so a typo previously surfaced as a ModuleNotFoundError
    # deep inside CNNModel.__init__, and seven of the eleven architectures the web UI
    # offers crash during construction for reasons the capability table records.
    requested_architectures = [
        name.strip() for name in getattr(args, 'architectures', '').split(',') if name.strip()
    ]
    usable_architectures, architecture_problems = validate_architectures(requested_architectures)
    if architecture_problems:
        print("\nERROR: unusable architecture(s) requested:")
        for name, reason in architecture_problems.items():
            print(f"  - {name}: {reason}")
        print(f"\n  Usable architectures: {', '.join(supported_detectors())}")
        for name in supported_detectors():
            print(f"    {describe_detector(name)}")
        sys.exit(1)
    # Enable faulthandler for hard crashes
    try:
        faulthandler.enable()
        print("Faulthandler enabled for crash diagnostics.")
    except Exception as e:
        print(f"Warning: Could not enable faulthandler: {e}")

    # Install signal handlers to log abrupt termination
    def _signal_handler(signum, frame):
        print(f"\n[Signal] Received signal {signum}. Potential abrupt termination.")
        faulthandler.dump_traceback()
    try:
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, _signal_handler)
    except Exception as e:
        print(f"Warning: Could not set signal handlers: {e}")

    # Log memory and CPU info at startup
    try:
        vm = psutil.virtual_memory()
        print(f"System memory: total={vm.total/1e9:.2f}GB, available={vm.available/1e9:.2f}GB, used={vm.used/1e9:.2f}GB")
        print(f"CPU count: {psutil.cpu_count(logical=True)}")
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Process address space limits: soft={soft}, hard={hard}")
    except Exception as e:
        print(f"Warning: Could not query system resources: {e}")

    # --- Reproducibility -------------------------------------------------- #
    # `ensure_deterministic_env()` at import time has already re-exec'd this
    # process if PYTHONHASHSEED or CUBLAS_WORKSPACE_CONFIG needed setting, so by
    # here the environment is correct regardless of how the run was launched
    # (shell, run_reproducible.sh, or the GPU queue's direct subprocess call).
    determinism_config = configure_determinism(args.seed, args.determinism)
    print(
        f"Determinism: mode={determinism_config.mode}, seed={determinism_config.seed}, "
        f"PYTHONHASHSEED={determinism_config.pythonhashseed}, "
        f"CUBLAS_WORKSPACE_CONFIG={determinism_config.cublas_workspace_config}"
    )
    if determinism_config.strict:
        print(
            "  strict mode: deterministic algorithms on, TF32 off, single-threaded, "
            "AMP disabled, swallowed exceptions re-raised"
        )
        if getattr(args, 'val_num_workers', 0) not in (0, 1):
            print(
                f"  strict mode: forcing --val-num-workers {args.val_num_workers} -> 1 "
                "(results are collected in submission order, so parallelism is safe, "
                "but a single worker removes the remaining thread-scheduling variance)"
            )
            args.val_num_workers = 1
    # GPU override: optionally force a single GPU via env and torch device
    try:
        if getattr(args, 'gpu_override', False):
            gpu_id = int(getattr(args, 'gpu_id', 0))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[GPU] Single-GPU override enabled. Forcing CUDA_VISIBLE_DEVICES={gpu_id}")
            if torch.cuda.is_available():
                try:
                    # Set device 0 of the now-restricted visible devices
                    torch.cuda.set_device(0)
                    print(f"[GPU] torch.cuda.set_device(0) successful (maps to physical GPU {gpu_id})")
                except Exception as e:
                    print(f"[GPU][Warning] Failed torch.cuda.set_device(0): {e}")
            else:
                print("[GPU][Warning] CUDA not available after override; falling back to CPU")
    except Exception as e:
        print(f"[GPU][Error] Failed to apply GPU override: {e}")

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
    os.makedirs("logs", exist_ok=True)
    logfile = f"hierarchical_test_{timestamp}.log"   
    data_root = resolve_ai_face_data_root(args.data_root)

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
    
    pipeline_start_time = time.time()

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

    # Select dataloader based on graph type
    if args.graph_type == 'nonclustered':
        print(f"Using UnclusteredDeepfakeDataloader for non-clustered graph construction")
        dataloader_class = UnclusteredDeepfakeDataloader
        graph_type_str = 'nonclustered'
    else:
        print(f"Using HierarchicalDeepfakeDataloader for clustered graph construction")
        dataloader_class = HierarchicalDeepfakeDataloader
        graph_type_str = 'clustered'

    for split_name, nodes_to_use, suffix in [
        ('train', train_nodes, train_suffix),
        ('val', val_nodes, val_suffix),
        ('test', test_nodes, test_suffix)
    ]:
        # Extract dataset name from data_root path (Corrected)
        dataset_name = os.path.basename(os.path.normpath(data_root)) if data_root else "unknown_dataset"
        
        # Cache key comes from the shared builder so it cannot drift from the
        # compatibility check the UI uses. It covers --seed, the sparse/subcluster
        # settings, and an edge-build version, all of which change the graph but were
        # previously absent -- meaning materially different graphs shared one entry.
        cache_base = graph_cache_key(
            dataset_name=dataset_name,
            split_name=split_name,
            graph_type=graph_type_str,
            balancing_suffix=suffix,
            nodes=nodes_to_use,
            quality_threshold=args.quality_threshold,
            symmetry_threshold=args.symmetry_threshold,
            embedding_threshold=args.embedding_threshold,
            seed=args.seed,
            build_val_test_edges=getattr(args, 'build_val_test_edges', True),
            hyperparameters=dataloader_class.hyperparameters,
            args=args,
        )
        _cache_paths = cache_filenames(graph_cache_dir, cache_base)
        pickle_cache_filename = _cache_paths['pickle']
        edges_csv_filename = _cache_paths['edges_csv']

        # Check/Load Graph Cache
        graph = None
        loaded_from_cache = False

        if os.path.exists(edges_csv_filename) or os.path.exists(pickle_cache_filename):
            try:
                # 1. Load Nodes (ensure nodes are loaded for the split)
                split_nodes = train_nodes_full if split_name == 'train' else val_nodes_full if split_name == 'val' else test_nodes_full
                if not split_nodes:
                    raise ValueError(f"Nodes for split '{split_name}' not found or loaded.")

                # Prefer streaming CSV cache if present
                if os.path.exists(edges_csv_filename):
                    print(f"\nFound streaming edge cache: {edges_csv_filename}. Attempting to load.")
                    print(f"Creating graph shell for {split_name} with {len(nodes_to_use)} nodes.")
                    graph = HyperGraph(nodes_to_use)
                    added = graph.load_edges_from_csv(edges_csv_filename)
                    print(f"Loaded {added} edges from CSV cache for {split_name}.")
                    loaded_from_cache = True
                else:
                    print(f"\nFound edge cache file: {pickle_cache_filename}. Attempting to load.")
                    # 2. Load Edge List (legacy pickle)
                    with open(pickle_cache_filename, 'rb') as f:
                        edge_list = dill.load(f)

                    # 3. Create node ID set for validation
                    nodes_to_use_ids = set(node.node_id for node in nodes_to_use)
                    print(f"Nodes to use for {split_name}: {len(nodes_to_use_ids)} unique node IDs")

                    # 4. Validate edge compatibility (only for pickle; CSV is streamed)
                    edge_node_ids = set()
                    for id1, id2 in edge_list:
                        edge_node_ids.add(id1)
                        edge_node_ids.add(id2)

                    missing_nodes = edge_node_ids - nodes_to_use_ids
                    print(f"Edge cache contains {len(edge_list)} edges referencing {len(edge_node_ids)} unique nodes")
                    print(f"Missing nodes in current node set: {len(missing_nodes)} ({len(missing_nodes)/len(edge_node_ids)*100:.1f}% of edge nodes)")

                    if len(missing_nodes) > len(edge_node_ids) * 0.1:  # More than 10% missing
                        print(f"WARNING: Cache incompatible - too many missing nodes ({len(missing_nodes)}). Regenerating graph.")
                        graph = None  # Force regeneration
                    else:
                        # 5. Reconstruct Graph - Use nodes_to_use to match cache creation
                        print(f"Creating graph shell for {split_name} with {len(nodes_to_use)} nodes.")
                        graph = HyperGraph(nodes_to_use)
                        print(f"Adding {len(edge_list)} edges from legacy pickle cache...")
                        graph.add_edges_from_list(edge_list)

                        print(f"Successfully loaded and reconstructed {split_name} graph from edge cache.")
                        loaded_from_cache = True

            except Exception as e:
                which = edges_csv_filename if os.path.exists(edges_csv_filename) else pickle_cache_filename
                print(f"\nError loading/reconstructing {split_name} graph from edge cache {which}: {e}. Regenerating.")
                import traceback
                traceback.print_exc()
                graph = None # Ensure regeneration if loading fails

        # --- Build Graph if not loaded from cache --- 
        if not loaded_from_cache:
            # Ensure nodes are available
            split_nodes = train_nodes_full if split_name == 'train' else val_nodes_full if split_name == 'val' else test_nodes_full
            if not split_nodes:
                 print(f"Error: Nodes for split '{split_name}' not available for building graph.")
                 continue # Or handle error appropriately
                 
            print(f"\nBuilding graph for {split_name} split ({len(split_nodes)} nodes)... No suitable cache found or --use-cached=False.")
            # Use the selected dataloader to build the graph
            dataloader = dataloader_class(
                datasets=[], 
                edge_class=edge_class,
                test_mode=False,  # Don't limit nodes
                visualize=False,  # Don't create visualizations during search
                show_viz=False,
                quality_threshold=args.quality_threshold,
                symmetry_threshold=args.symmetry_threshold,
                embedding_threshold=args.embedding_threshold,
                build_val_test_edges=getattr(args, 'build_val_test_edges', True),
                silent_mode=True  # Disable internal progress bars and logging during grid search
            )
            should_build_edges = split_name == 'train' or getattr(args, 'build_val_test_edges', True)
            if should_build_edges:
                graph_build_result = dataloader._build_graph_standard(nodes_to_use, split_name)
            else:
                graph_build_result = HyperGraph(nodes_to_use)
            
            # Handle potential tuple return from build_graph_standard
            if isinstance(graph_build_result, tuple):
                 graph = graph_build_result[0] 
                 # Potentially handle other elements in the tuple if needed
            else:
                 graph = graph_build_result
            
            # --- Save Edges to Cache (streaming CSV preferred) --- 
            if graph: # Only save if graph build was successful
                try:
                    print(f"Saving edges for {split_name} graph to streaming cache: {edges_csv_filename}")
                    written = graph.export_edges_csv(edges_csv_filename)
                    print(f"Saved {written} edges for {split_name} to CSV cache.")
                except Exception as e:
                    print(f"Error saving {split_name} edges to CSV cache {edges_csv_filename}: {e}")
                    import traceback
                    traceback.print_exc()
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
        
    graph_construction_time = time.time() - pipeline_start_time

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

    with capture_output(logfile) as logpath:
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
        # NEW: Adaptive Trainer Setup
        # ===============================
        
        # Parse traversal configuration from command line arguments
        traversal_config = parse_traversal_config(args)
        print(f"\nTraversal Configuration:")
        print(f"  Trainer mode: {traversal_config['trainer_mode']}")
        print(f"  Architectures: {traversal_config['architectures']}")
        print(f"  DQN model type: {args.dqn_model}")
        print(f"  Uncertainty head: {args.uncertainty_head}")
        print(f"  MC Dropout samples: {args.mc_dropout_samples}")
        print(f"  Graph uncertainty methods: {graph_uncertainty_methods}")
        
        if traversal_config['enable_switching']:
            print(f"  Traversal switching enabled")
            print(f"  Sequence: {traversal_config['traversal_sequence']}")
            print(f"  Switch epochs: {traversal_config['switch_epochs']}")
        else:
            print(f"  Single traversal: {traversal_config['single_traversal']}")
        
        if traversal_config['test_all_traversals']:
            print(f"  Testing all traversal types for comparison")

        
        # Determine test configurations
        if traversal_config['test_all_traversals']:
            # Test all traversal types individually
            traversal_types = ["comprehensive", "random", "i-value", "i-value-cluster-hop"]
            test_configs = []
            for arch in traversal_config['architectures']:
                for traversal_type in traversal_types:
                    test_configs.append({
                        'arch': arch,
                        'mode': 'single',
                        'traversal': traversal_type,
                        'description': f"{arch}_{traversal_type}"
                    })
        else:
            # Test configured mode only
            test_configs = []
            for arch in traversal_config['architectures']:
                if traversal_config['enable_switching']:
                    test_configs.append({
                        'arch': arch,
                        'mode': 'switching',
                        'traversal_sequence': traversal_config['traversal_sequence'],
                        'switch_epochs': traversal_config['switch_epochs'],
                        'description': f"{arch}_switching"
                    })
                else:
                    test_configs.append({
                        'arch': arch,
                        'mode': 'single',
                        'traversal': traversal_config['single_traversal'],
                        'description': f"{arch}_{traversal_config['single_traversal']}"
                    })
        
        # Calculate graph sizes and training steps
        train_size = len(train_manager.graph.get_nodes())
        val_size = len(val_manager.graph.get_nodes())
        test_size = len(test_manager.graph.get_nodes())
        
        print(f"\nGraph sizes:")
        print(f"Train: {train_size} nodes")
        print(f"Val: {val_size} nodes") 
        print(f"Test: {test_size} nodes")
        
        # Steps config: allow explicit counts or match number of nodes
        if getattr(args, 'train_steps_equal_nodes', False):
            train_steps = len(train_manager.graph.get_nodes())
        else:
            train_steps = getattr(args, 'train_steps', 1000)
        if getattr(args, 'val_steps_equal_nodes', False):
            val_steps = len(val_manager.graph.get_nodes())
        else:
            # Default to 500 instead of 1000 for faster validation, but allow override
            default_val_steps = min(500, len(val_manager.graph.get_nodes()) if val_manager.graph else 500)
            val_steps = getattr(args, 'val_steps', default_val_steps)
        
        # --- Setup run-specific output directory ---
        # Uses secrets, not the seeded `random` module. The previous version drew 8
        # values from the global RNG, but only when --run-id was absent: the web UI
        # always passes one and a manual invocation does not, so a manual run and a
        # UI-launched run at the same --seed consumed different amounts of randomness
        # and therefore produced different traversals.
        import secrets

        def generate_run_id():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"run_{timestamp}_{secrets.token_hex(4)}"

        run_id = args.run_id if hasattr(args, 'run_id') and args.run_id else generate_run_id()
        run_output_dir = Path(f"run_outputs/{run_id}")
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Quanty] All visualizations and outputs for this run will be saved under: {run_output_dir}")

        # Test each configuration
        for config in test_configs:
            print(f"\n{'='*80}")
            print(f"Testing configuration: {config['description']}")
            print(f"{'='*80}\n")
            
            try:
                # Create model
                arch = config['arch']
                model = create_model(
                    arch,
                    f"/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/{config['description']}_{timestamp}.pt",
                    device,
                    dqn_model_type=args.dqn_model,  # Pass DQN model type from args
                    feature_dim=128,  # Default feature dimension for DQN models
                    embedding_dim=512,  # Default embedding dimension
                    uncertainty_head=args.uncertainty_head,
                    mc_dropout_samples=args.mc_dropout_samples,
                    batchensemble_members=args.batchensemble_members,
                    sngp_hidden_dim=args.sngp_hidden_dim,
                    sngp_rff_dim=args.sngp_rff_dim,
                    uncertainty_dropout_rate=args.uncertainty_dropout_rate,
                    graph_uncertainty_methods=graph_uncertainty_methods,
                    graph_degree_penalty_weight=args.graph_degree_penalty_weight,
                    uncertainty_train_frequency=args.uncertainty_train_frequency,
                    sngp_precision_policy=args.sngp_precision_policy,
                    finetune=args.finetune,
                )

                if isinstance(model, CNNModel):
                    counts = model.parameter_counts()
                    print(
                        f"  Trainable parameters: {counts['trainable']:,} / {counts['total']:,}"
                        f" (finetune={counts['finetune']}, backbone_frozen={counts['backbone_frozen']})"
                    )

                # Fit graph-distance statistics once on the training graph and reuse
                # them for val/test. Fitting per split would renormalize a shifted
                # distribution until it matched training, erasing the shift signal.
                if graph_uncertainty_methods and isinstance(model, CNNModel):
                    standardizer = GraphDistanceUncertainty(
                        methods=graph_uncertainty_methods,
                        penalty_weight=args.graph_degree_penalty_weight,
                        robust=args.graph_distance_robust_stats,
                    ).fit(train_manager.get_graph().get_nodes())
                    standardizer.precompute(train_manager.get_graph())
                    model.set_graph_distance_standardizer(standardizer)
                    print(
                        f"  Graph-distance statistics fitted "
                        f"(hash={standardizer.stats_hash}, "
                        f"embedding coverage={standardizer.embedding_coverage:.1%})"
                    )
                    if standardizer.embedding_coverage is not None and standardizer.embedding_coverage < 0.5:
                        print(
                            "  WARNING: fewer than half of nodes have a usable face embedding; "
                            "embedding_distance scores will be dominated by the missing-value sentinel"
                        )

                # Create trainer (always use AdaptiveTrainer)
                trainer = create_adaptive_trainer(train_manager, model, device, attribute_metadata, criterion, args)
                
                # NEW: Set traversal sequence for DQN warm-up if using switching mode
                if config['mode'] == 'switching':
                    trainer.set_traversal_sequence(config['traversal_sequence'])
                
                # Set initial traversal
                if config['mode'] == 'single':
                    initial_traversal = create_traversal(
                        config['traversal'], 
                        train_manager.graph, 
                        num_pointers=1, 
                        num_steps=train_steps,
                        trainer=trainer,
                        bias_hop_period=args.bias_hop_period
                    )
                    trainer.set_traversal(initial_traversal, config['traversal'])
                    print(f"Set single traversal: {config['traversal']}")
                else:  # switching mode
                    initial_traversal_type = config['traversal_sequence'][0]
                    initial_traversal = create_traversal(
                        initial_traversal_type,
                        train_manager.graph,
                        num_pointers=1,
                        num_steps=train_steps,
                        trainer=trainer,
                        bias_hop_period=args.bias_hop_period
                    )
                    trainer.set_traversal(initial_traversal, initial_traversal_type)
                    print(f"Set initial traversal for switching: {initial_traversal_type}")
                
                # Training setup
                best_val_accuracy = 0.0
                best_epoch = 0
                
                # Initialize Graph Reduction Manager if enabled
                reduction_manager = None
                if config.get('reduction_enabled', False):
                    print(f"\n🔧 Initializing Graph Reduction Manager...")
                    reduction_manager = GraphReductionManager(
                        reduction_strategy=config.get('reduction_strategy', 'none'),
                        reduction_percentage=config.get('reduction_percentage', 0.0),
                        reduction_top_percentage=config.get('reduction_top_percentage', 0.0),
                        reduction_bottom_percentage=config.get('reduction_bottom_percentage', 0.0),
                        reduction_interval=config.get('reduction_interval', 'end_of_epoch'),
                        reduction_interval_steps=config.get('reduction_interval_steps', 100),
                        restoration_strategy=config.get('restoration_strategy', 'none'),
                        restoration_percentage=config.get('restoration_percentage', 50.0),
                        restoration_trigger_threshold=config.get('restoration_trigger_threshold', 0.0)
                    )
                    print(f"  Reduction Strategy: {reduction_manager.reduction_strategy}")
                    print(f"  Restoration Strategy: {reduction_manager.restoration_strategy}")
                
                # Initialize I-value visualization tracking if enabled
                viz_tracker = None
                bias_hop_viz = None
                # Check if any of the traversals used will be I-value based
                uses_ivalue_traversal = False
                if config['mode'] == 'single':
                    uses_ivalue_traversal = config['traversal'] in ['i-value', 'i-value-cluster-hop']
                else:  # switching mode
                    uses_ivalue_traversal = any(t in ['i-value', 'i-value-cluster-hop'] for t in config.get('traversal_sequence', []))
                # --- Use run-specific directory for all outputs ---
                config_output_dir = run_output_dir / config['description']
                config_output_dir.mkdir(parents=True, exist_ok=True)
                # Checkpoints: save uniquely per run/config/DQN type to avoid collisions
                ckpt_dir = config_output_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                best_model_checkpoint_path = str(ckpt_dir / f"{args.dqn_model}_best.pth")
                
                # Export node/edge CSVs with subcluster info if enabled
                if getattr(args, 'export_csv_per_run', True):
                    graph = getattr(train_manager, 'graph', None)
                    if graph is not None and hasattr(graph, 'export_csv_with_subclusters'):
                        node_csv_path = config_output_dir / 'nodes.csv'
                        edge_csv_path = config_output_dir / 'edges.csv'
                        graph.export_csv_with_subclusters(str(node_csv_path), str(edge_csv_path))
                        print(f"[Quanty] Exported node/edge CSVs with subcluster info to {config_output_dir}")
                if args.enable_ivalue_viz and uses_ivalue_traversal:
                    print(f"\n📊 Initializing I-value visualization tracking...")
                    viz_save_dir = config_output_dir / "ivalue"
                    viz_save_dir.mkdir(parents=True, exist_ok=True)
                    viz_tracker = IValueVisualizationTracker(save_dir=viz_save_dir)
                    # Set up bias hop visualizer for cluster hop traversal
                    if (config['mode'] == 'single' and config['traversal'] == 'i-value-cluster-hop') or \
                       (config['mode'] == 'switching' and 'i-value-cluster-hop' in config.get('traversal_sequence', [])):
                        bias_hop_viz = BiasHopVisualizer(save_dir=config_output_dir / "bias_hops")
                    # Track specific nodes for detailed analysis
                    sample_nodes = rng_for('viz.node_sample').sample(list(train_manager.graph.get_nodes()), 
                                                min(args.viz_track_nodes, len(train_manager.graph.get_nodes())))
                    viz_tracker.track_specific_nodes(trainer, sample_nodes, max_nodes=args.viz_track_nodes)
                    print(f"   Visualization directory: {viz_save_dir}")
                    print(f"   Tracking {len(sample_nodes)} nodes for detailed analysis")
                    if bias_hop_viz:
                        print(f"   Bias hop visualization enabled")
                # Initialize bias metrics tracking
                print(f"\n🎯 Initializing bias metrics tracking...")
                bias_save_dir = config_output_dir / "bias"
                bias_save_dir.mkdir(parents=True, exist_ok=True)
                bias_tracker = BiasMetricsTracker(save_dir=bias_save_dir)
                
                print(f"\nTraversal settings:")
                print(f"Train: {train_steps} steps with 1 pointer")
                print(f"Val: {val_steps} steps with 1 pointer") 
                print(f"Test: All nodes")
                
                # Training loop
                print(f"\nTraining {config['description']}...")
                for epoch in range(args.num_epochs):
                    print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")

                    # Re-verify strict determinism each epoch, so a run that quietly
                    # fell out of strict mode fails here instead of producing
                    # unreproducible numbers to the end.
                    assert_strict_invariants(f"start of epoch {epoch + 1}")

                    # Let the model apply epoch-boundary policies. SNGP uses this to
                    # reset its Laplace precision so gp_variance stays comparable
                    # between epochs; previously it accumulated across the whole run.
                    for trainer_model in getattr(trainer, 'models', []) or []:
                        if hasattr(trainer_model, 'on_epoch_start'):
                            trainer_model.on_epoch_start(epoch, num_epochs=args.num_epochs)

                    # Handle reversion strategy at start of epoch (if enabled)
                    if reduction_manager and reduction_manager.restoration_strategy == 'reversion' and epoch > 0:
                        print(f"  🔄 Checking for reversion restoration at start of epoch {epoch+1}...")
                        # Reversion happens at start of next epoch, so check previous epoch's validation
                        # For now, we'll check if restoration is needed based on previous validation
                        # This will be triggered after validation in previous epoch, but we check here too
                        if len(reduction_manager.get_removed_nodes()) > 0:
                            # Restore previous epoch's nodes if validation dropped
                            # Note: This is a simplified check - full logic happens after validation
                            pass
                    
                    # Start epoch tracking for visualization
                    if viz_tracker:
                        viz_tracker.start_epoch(epoch)
                    
                    # Handle traversal switching
                    if (config['mode'] == 'switching' and 
                        epoch in config['switch_epochs']):
                        switch_idx = config['switch_epochs'].index(epoch)
                        new_traversal_type = config['traversal_sequence'][switch_idx + 1]
                        print(f"🔄 Switching to {new_traversal_type} traversal at epoch {epoch+1}")
                        trainer.switch_traversal(new_traversal_type, bias_hop_period=args.bias_hop_period)
                        # --- Disconnected switching: reset main detection model and best checkpoint/vars ---
                        if config.get('disconnected_switching', False):
                            print(f"[Disconnected Switching] Resetting main detection model and best checkpoint/vars after traversal switch at epoch {epoch+1}")
                            # Re-instantiate the model with the same parameters
                            arch = config['arch']
                            model = create_model(
                                arch,
                                f"/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/{config['description']}_{timestamp}.pt",
                                device,
                                dqn_model_type=args.dqn_model,  # Pass DQN model type from args
                                feature_dim=128,  # Default feature dimension for DQN models
                                embedding_dim=512,  # Default embedding dimension
                                uncertainty_head=args.uncertainty_head,
                                mc_dropout_samples=args.mc_dropout_samples,
                                batchensemble_members=args.batchensemble_members,
                                sngp_hidden_dim=args.sngp_hidden_dim,
                                sngp_rff_dim=args.sngp_rff_dim,
                                uncertainty_dropout_rate=args.uncertainty_dropout_rate,
                                graph_uncertainty_methods=graph_uncertainty_methods,
                                graph_degree_penalty_weight=args.graph_degree_penalty_weight,
                                uncertainty_train_frequency=args.uncertainty_train_frequency,
                                sngp_precision_policy=args.sngp_precision_policy,
                                finetune=args.finetune,
                            )
                            # Carry the fitted graph-distance statistics onto the
                            # replacement model, so uncertainty stays on the same
                            # scale across a disconnected-switching reset.
                            previous_standardizer = getattr(
                                trainer.models[0], 'graph_distance_standardizer', None
                            )
                            if previous_standardizer is not None:
                                model.set_graph_distance_standardizer(previous_standardizer)
                            trainer.models[0] = model
                            print(f"[Disconnected Switching] Main detection model has been reset.")
                            # Reset best checkpoint/vars
                            best_val_accuracy = 0.0
                            best_epoch = 0
                            if os.path.exists(best_model_checkpoint_path):
                                os.remove(best_model_checkpoint_path)
                                print(f"[Disconnected Switching] Deleted old best model checkpoint: {best_model_checkpoint_path}")
                    
                    # Training step
                    train_start_time = time.time()
                    train_distribution = None
                    
                    # Pass the epoch index through. AdaptiveTrainer.train(epoch=None)
                    # already forwarded it to the capabilities, but it was always
                    # called with no argument, so the capabilities never saw it.
                    train_result = trainer.train(epoch=epoch)
                    if isinstance(train_result, tuple) and len(train_result) == 2:
                        train_metrics, train_distribution = train_result
                    else:
                        train_metrics = train_result
                        train_distribution = None

                    for trainer_model in getattr(trainer, 'models', []) or []:
                        if hasattr(trainer_model, 'on_epoch_end'):
                            trainer_model.on_epoch_end(epoch)
                    # Get current traversal info
                    current_traversal_info = trainer.get_current_traversal_info()
                    print(f"  Current traversal: {current_traversal_info}")
                    if train_metrics and isinstance(train_metrics, dict):
                        print(f"  Training metrics: {json.dumps(train_metrics, indent=2)}")
                    
                    # Check for reduction during training (if interval is every_n_steps)
                    if reduction_manager and reduction_manager.reduction_interval == 'every_n_steps':
                        # Calculate current step (approximate: epoch * steps_per_epoch + current_step_in_epoch)
                        # For simplicity, we'll use epoch * train_steps as approximation
                        current_step = epoch * train_steps
                        if reduction_manager.should_reduce(current_step, epoch):
                            print(f"  🔧 Performing graph reduction at step {current_step}...")
                            removed_nodes, reduction_stats = reduction_manager.reduce_graph(
                                train_manager.graph, trainer, epoch, current_step
                            )
                            if removed_nodes:
                                print(f"  Removed {len(removed_nodes)} nodes from training graph")
                    
                    # Log I-value visualization data at the end of each epoch
                    if viz_tracker:
                        # Update tracked nodes
                        viz_tracker.update_tracked_nodes(trainer)
                        
                        # Log epoch summary statistics
                        viz_tracker.log_epoch_summary(trainer, sample_size=args.viz_sample_size)
                        
                        # Get current traversal for additional data
                        current_traversal = None
                        if hasattr(trainer, 'current_traversal'):
                            current_traversal = trainer.current_traversal
                        
                        # Track bias hop data if available
                        if bias_hop_viz and current_traversal and hasattr(current_traversal, 'get_hop_i_value_history'):
                            hop_history = current_traversal.get_hop_i_value_history()
                            if hop_history:
                                viz_tracker.bias_hop_history.extend(hop_history)
                    
                    # Print training distribution if available
                    if train_distribution:
                        print("  Training Attribute Distribution for this Epoch:")
                        print(json.dumps(train_distribution, indent=4))
                    
                    # Evaluate training bias metrics periodically
                    train_metrics_full = None
                    if (getattr(args, 'enable_train_bias_inference', False)) and (epoch % 5 == 0 or epoch == args.num_epochs - 1):  # gated
                        print(f"  Evaluating training bias metrics...")
                        model_to_eval = trainer.models[0] if trainer.models else None
                        if model_to_eval:
                            model_to_eval.eval()
                            train_sample_nodes = rng_for('eval.train_bias_subsample').sample(
                                list(train_manager.graph.get_nodes()), 
                                min(len(train_manager.graph.get_nodes()), train_steps)
                            )
                            train_metrics_full = evaluate_model(
                                model=model_to_eval,
                                nodes_to_evaluate=train_sample_nodes,
                                loss_fn=criterion,
                                batch_size=args.batch_size,
                                bias_loss_fn=getattr(trainer, 'bias_loss', None),
                                device=device,
                                desc="Training Bias Eval",
                                attribute_metadata=attribute_metadata
                            )
                    
                    # Validation step - Always run validation to track accuracy and save best model
                    if val_nodes_from_graph:
                        model_to_eval = trainer.models[0] if trainer.models else None
                        if not model_to_eval:
                            print(f"ERROR: No model found in trainer. Skipping validation.")
                            continue
                        
                        model_to_eval.eval()
                        # Only calculate bias metrics if enable_val_bias_inference is True
                        bias_loss_fn = getattr(trainer, 'bias_loss', None) if getattr(args, 'enable_val_bias_inference', False) else None
                        # Use parallel image loading for faster validation (4 workers)
                        val_metrics = evaluate_model(
                            model=model_to_eval,
                            nodes_to_evaluate=rng_for('eval.val_subsample').sample(val_nodes_from_graph, min(len(val_nodes_from_graph), val_steps)),
                            loss_fn=criterion,
                            batch_size=args.batch_size,
                            bias_loss_fn=bias_loss_fn,
                            device=device,
                            desc="Validation",
                            attribute_metadata=attribute_metadata if getattr(args, 'enable_val_bias_inference', False) else None,
                            num_workers=getattr(args, 'val_num_workers', 4)  # Allow override, default 4
                        )
                        
                        # Log bias metrics for this epoch (only if bias inference is enabled)
                        if getattr(args, 'enable_val_bias_inference', False):
                            bias_tracker.log_bias_metrics(epoch=epoch, train_metrics=train_metrics_full, val_metrics=val_metrics)
                            
                            # Log validation bias metrics to bias hop visualizer if it exists
                            if bias_hop_viz and val_metrics and 'bias_metrics' in val_metrics:
                                # Calculate subgroup I-values for correlation analysis
                                subgroup_i_values = {}
                                if hasattr(trainer, 'attribute_metadata') and trainer.attribute_metadata:
                                    # Get a sample of validation nodes for I-value calculation
                                    val_sample = rng_for('eval.val_subsample').sample(val_nodes_from_graph, min(100, len(val_nodes_from_graph)))
                                    try:
                                        for node in val_sample:
                                            if hasattr(node, 'attributes') and node.attributes:
                                                # Create race-gender subgroup key
                                                gender = node.attributes.get('Ground Truth Gender')
                                                race = node.attributes.get('Ground Truth Race')
                                                if gender is not None and race is not None:
                                                    subgroup_key = f"Ground Truth Gender_{gender}_Ground Truth Race_{race}"
                                                    if subgroup_key not in subgroup_i_values:
                                                        subgroup_i_values[subgroup_key] = []
                                                    i_value = trainer.get_i_value(node, 0) if hasattr(trainer, 'get_i_value') else 0
                                                    subgroup_i_values[subgroup_key].append(i_value)
                                    except Exception as e:
                                        print(f"Warning: Error calculating subgroup I-values: {e}")
                                
                                # Average the I-values for each subgroup
                                avg_subgroup_i_values = {}
                                for subgroup, i_vals in subgroup_i_values.items():
                                    if i_vals:
                                        avg_subgroup_i_values[subgroup] = np.mean(i_vals)
                                
                                bias_hop_viz.log_validation_bias_metrics(
                                    epoch=epoch, 
                                    bias_metrics=val_metrics['bias_metrics'],
                                    subgroup_i_values=avg_subgroup_i_values
                                )
                        
                        current_val_accuracy = val_metrics.get('accuracy', 0.0)
                        
                        # Check for restoration trigger
                        if reduction_manager and reduction_manager.restoration_enabled():
                            if reduction_manager.check_restoration_trigger(current_val_accuracy, best_val_accuracy):
                                print(f"  🔄 Validation accuracy dropped ({current_val_accuracy:.4f} < {best_val_accuracy:.4f}), triggering restoration...")
                                restored_nodes, restoration_stats = reduction_manager.restore_nodes(
                                    train_manager.graph, trainer, current_val_accuracy, best_val_accuracy
                                )
                                if restored_nodes:
                                    print(f"  Restored {len(restored_nodes)} nodes to training graph")
                        
                        # Check for model rollback
                        model_rollback_enabled = config.get('model_rollback_enabled', False)
                        model_rollback_on_drop = config.get('model_rollback_on_val_drop', False)
                        if model_rollback_enabled and model_rollback_on_drop:
                            if current_val_accuracy < best_val_accuracy and best_val_accuracy > 0:
                                print(f"  ⏪ Model rollback enabled: validation accuracy dropped, rolling back to best model...")
                                if os.path.exists(best_model_checkpoint_path):
                                    model_to_eval.load_checkpoint(best_model_checkpoint_path)
                                    trainer.load_capability_checkpoints(best_model_checkpoint_path)
                                    print(f"  Rolled back to best model from epoch {best_epoch}")
                        
                        # Save best model
                        if current_val_accuracy > best_val_accuracy:
                            best_val_accuracy = current_val_accuracy
                            best_epoch = epoch + 1
                            model_to_eval.save_checkpoint(best_model_checkpoint_path)
                            
                            # Save additional checkpoints for AdaptiveTrainer capabilities
                            trainer.save_capability_checkpoints(best_model_checkpoint_path)
                            
                            print(f"New best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
                        else:
                            print(f"Validation accuracy: {current_val_accuracy:.4f} (best: {best_val_accuracy:.4f} at epoch {best_epoch})")
                    
                    # End of epoch: perform reduction if interval is end_of_epoch
                    if reduction_manager and reduction_manager.reduction_interval == 'end_of_epoch':
                        if reduction_manager.reduction_enabled():
                            print(f"  🔧 Performing graph reduction at end of epoch {epoch+1}...")
                            removed_nodes, reduction_stats = reduction_manager.reduce_graph(
                                train_manager.graph, trainer, epoch, epoch * train_steps
                            )
                            if removed_nodes:
                                print(f"  Removed {len(removed_nodes)} nodes from training graph")
                                # Store epoch state for reversion strategy
                                reduction_manager.store_epoch_state(epoch, removed_nodes)
                    
                    # Log reduction/restoration statistics
                    if reduction_manager:
                        stats = reduction_manager.get_stats()
                        if stats['reduction_stats']['total_reductions'] > 0 or stats['reduction_stats']['total_restorations'] > 0:
                            print(f"  Reduction/Restoration Stats: {stats['reduction_stats']}")
                
                # Final testing
                if args.num_epochs > 0:
                    model_to_eval = trainer.models[0] if trainer.models else None
                    if not model_to_eval:
                        print(f"ERROR: No model found in trainer. Skipping final testing.")
                    else:
                        if os.path.exists(best_model_checkpoint_path):
                            print(f"\n🔍 Loading best model from epoch {best_epoch} for final testing...")
                            model_to_eval.load_checkpoint(best_model_checkpoint_path)
                            # Load additional checkpoints for AdaptiveTrainer
                            trainer.load_capability_checkpoints(best_model_checkpoint_path)
                        else:
                            print(f"\n⚠️  No checkpoint found at {best_model_checkpoint_path}. Using current model state for final testing...")
                        
                        model_to_eval.eval()
                        test_metrics = evaluate_model(
                            model=model_to_eval,
                            nodes_to_evaluate=test_nodes_from_graph,
                            loss_fn=criterion,
                            batch_size=args.batch_size,
                            bias_loss_fn=getattr(trainer, 'bias_loss', None) if getattr(args, 'enable_val_bias_inference', False) else None,
                            device=device,
                            desc="Final Test",
                            attribute_metadata=attribute_metadata if getattr(args, 'enable_val_bias_inference', False) else None
                        )
                        print("\n--- Final Test Results ---")
                        print(json.dumps(test_metrics, indent=2))
                        # Log final test bias metrics (only if bias inference is enabled)
                        if getattr(args, 'enable_val_bias_inference', False):
                            bias_tracker.log_bias_metrics(epoch=best_epoch-1 if best_epoch > 0 else args.num_epochs-1, test_metrics=test_metrics)
                
                # Generate I-value visualization plots and reports if tracking was enabled
                if viz_tracker:
                    print(f"\n📊 Generating I-value visualization plots...")
                    
                    try:
                        # Generate training progression plots
                        viz_tracker.plot_training_progression()
                        
                        # Generate subgroup analysis plots
                        viz_tracker.plot_subgroup_analysis()
                        
                        # Generate tracked nodes plots
                        viz_tracker.plot_tracked_nodes()
                        
                        # Save raw data
                        viz_tracker.save_data()
                        
                        # Generate summary report
                        viz_tracker.generate_summary_report()
                        
                        # Generate bias hop visualizations if available
                        if bias_hop_viz and viz_tracker.bias_hop_history:
                            print(f"\n📊 Generating bias hop visualization plots...")
                            try:
                                # Keep the I-value statistics per hop (as requested)
                                bias_hop_viz.plot_i_value_statistics_per_hop(viz_tracker.bias_hop_history)
                                
                                # Fixed subgroup targeting analysis (shorter x-axis labels)
                                bias_hop_viz.plot_subgroup_targeting_analysis(viz_tracker.bias_hop_history)
                                
                                # New: subgroup bias metrics per validation epoch 
                                bias_hop_viz.plot_subgroup_bias_per_validation_epoch()
                                
                                # New: I-value vs bias correlation (replaces unclear bias reduction)
                                bias_hop_viz.plot_i_value_bias_correlation()
                                
                                # Generate summary report
                                bias_hop_viz.generate_hop_summary_report(viz_tracker.bias_hop_history)
                                
                                print(f"✅ Bias hop visualization complete")
                            except Exception as bias_hop_error:
                                print(f"⚠️  Error generating bias hop visualizations: {bias_hop_error}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"✅ I-value visualization complete for {config['description']}")
                        
                    except Exception as viz_error:
                        print(f"⚠️  Error generating visualizations: {viz_error}")
                        import traceback
                        traceback.print_exc()
                
                # Generate bias metrics visualization plots
                print(f"\n🎯 Generating bias metrics visualization plots...")
                try:
                    bias_tracker.generate_all_plots()
                    print(f"✅ Bias metrics visualization complete for {config['description']}")
                    
                except Exception as bias_viz_error:
                    print(f"⚠️  Error generating bias visualizations: {bias_viz_error}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                log_exception(logfile, *sys.exc_info())
                print(f"\nError in configuration {config['description']}: {str(e)}")
                continue

    print("\nDone!")
    
    if logpath:
        print(f"Output captured in: {logpath}")

if __name__ == "__main__":
    main()
