import os
import sys
import random
import logging
import traceback
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Attempt to import project-specific modules
try:
    from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
    from datasets.AIFaceDataset import AIFaceDataset
    from data.ImageFileData import ImageFileData
    from nodes.atrnode import AttributeNode
except ImportError:
    # This is a fallback for environments where src might not be directly in PYTHONPATH
    # Often happens when running scripts from a subdirectory without package installation
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) # Adjust if your structure is different
    from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
    from datasets.AIFaceDataset import AIFaceDataset
    from data.ImageFileData import ImageFileData
    from nodes.atrnode import AttributeNode

from .logging_utils import NullHandler # Relative import for NullHandler


def balance_nodes_by_subgroup(nodes, target_num_nodes, attributes_to_balance=['race', 'gender']):
    """Balances a list of nodes to achieve a target number, ensuring representation
    across specified subgroups.

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

    # Create deterministic random state based on node IDs for reproducible balancing
    import hashlib
    import random as rand_module
    node_ids = sorted([node.node_id for node in nodes])
    balance_seed = int(hashlib.md5('|'.join(node_ids).encode()).hexdigest()[:8], 16) % (2**32)
    balance_rng = rand_module.Random(balance_seed)
    print(f"Using deterministic seed {balance_seed} for node balancing (based on {len(node_ids)} node IDs)")

    subgroups = defaultdict(list)
    for node in nodes:
        subgroup_key = tuple(node.attributes.get(attr, 'Unknown') for attr in attributes_to_balance)
        subgroups[subgroup_key].append(node)

    num_subgroups = len(subgroups)
    if num_subgroups == 0:
        print("Warning: No subgroups found for balancing. Returning original list (or empty if target > original size).")
        return nodes if len(nodes) == target_num_nodes else []

    nodes_per_subgroup = target_num_nodes // num_subgroups
    remainder = target_num_nodes % num_subgroups

    print(f"Balancing to {target_num_nodes} nodes across {num_subgroups} subgroups.")
    print(f"Base nodes per subgroup: {nodes_per_subgroup}, Remainder: {remainder}")

    balanced_nodes = []
    subgroup_keys = list(subgroups.keys())
    balance_rng.shuffle(subgroup_keys) # Shuffle keys to randomly distribute remainder (deterministic)

    for i, subgroup_key in enumerate(subgroup_keys):
        group_nodes = subgroups[subgroup_key]
        required_size = nodes_per_subgroup + (1 if i < remainder else 0)

        if len(group_nodes) < required_size:
            raise ValueError(
                f"Cannot balance to {target_num_nodes} nodes. Subgroup {subgroup_key} "
                f"has only {len(group_nodes)} nodes, but requires {required_size}."
            )

        if required_size > 0:
            sampled_nodes = balance_rng.sample(group_nodes, required_size)
            balanced_nodes.extend(sampled_nodes)

    balance_rng.shuffle(balanced_nodes) # Shuffle the final list (deterministic)
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
                if isinstance(cache_data, dict):
                    if split_name not in cache_data:
                        print(f"     Reason: Split key '{split_name}' not found in top-level dict.")
                    elif not isinstance(cache_data.get(split_name), dict):
                        print(f"     Reason: Value for key '{split_name}' is not a dict (Type: {type(cache_data.get(split_name))}).")
                    elif 'full' not in cache_data.get(split_name, {}):
                        print(f"     Reason: Key 'full' not found within dict for split '{split_name}'.")
                    elif 'balanced' not in cache_data.get(split_name, {}):
                        print(f"     Reason: Key 'balanced' not found within dict for split '{split_name}'.")

            if isinstance(cache_data, dict) and 'full' in cache_data and 'balanced' in cache_data:
                 print(f"  -> Detected OLD cache format (dict without splits). Loading overall 'full' set as fallback for '{split_name}'.")
                 nodes_to_return = cache_data['balanced'] if balanced else cache_data['full']
                 if balanced:
                      print(f"     Warning: Requested balanced set, returning from overall balanced set ({len(nodes_to_return)} nodes). This might not be split-specific.")
                 else:
                      print(f"     Returning overall full set ({len(nodes_to_return)} nodes). This might not be split-specific.")
                 return nodes_to_return

            elif isinstance(cache_data, list):
                 print(f"  -> Detected VERY OLD cache format (list). Loading as full set for '{split_name}'.")
                 if balanced:
                     print("     Warning: Cannot load balanced set from list format. Returning full list.")
                 print(f"     Returning full list ({len(cache_data)} nodes)." )
                 return cache_data

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
    quality_thresholds = np.linspace(.5, .9, quality_steps)
    symmetry_thresholds = np.linspace(.5, .9, symmetry_steps)
    embedding_thresholds = np.linspace(.9, .999, embedding_steps)
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/threshold_search_{timestamp}.csv"
    
    total_combinations = len(quality_thresholds) * len(symmetry_thresholds) * len(embedding_thresholds)
    print(f"\nRunning grid search with {total_combinations} threshold combinations...")
    
    combinations = list(product(quality_thresholds, symmetry_thresholds, embedding_thresholds))
    progress_bar = tqdm(total=len(combinations), desc="Running grid search")

    for q_thresh, s_thresh, e_thresh in combinations:
        q_thresh_str = round(q_thresh, 2)
        s_thresh_str = round(s_thresh, 2)
        e_thresh_str = round(e_thresh, 2)
        
        progress_bar.set_description(
            f"Combination {progress_bar.n+1}/{len(combinations)} - Testing Q:{q_thresh_str} S:{s_thresh_str} E:{e_thresh_str}"
        )
        progress_bar.update(0)
        
        dataloader = HierarchicalDeepfakeDataloader(
            datasets=[], 
            edge_class=edge_class,
            test_mode=False,
            visualize=False,
            show_viz=False,
            quality_threshold=q_thresh_str,
            symmetry_threshold=s_thresh_str,
            embedding_threshold=e_thresh_str,
            silent_mode=True
        )
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        null_output_path = os.devnull
        
        # Ensure the directory for the null output file exists if it's not os.devnull
        # For os.devnull, this is not strictly necessary but good practice for other paths.
        if null_output_path != os.devnull and os.path.dirname(null_output_path):
             os.makedirs(os.path.dirname(null_output_path), exist_ok=True)
        
        with open(null_output_path, 'w') as null_output_file:
            sys.stdout = null_output_file
            sys.stderr = null_output_file

            original_handlers = {}
            for logger_name in logging.root.manager.loggerDict:
                logger = logging.getLogger(logger_name)
                original_handlers[logger_name] = list(logger.handlers)
                logger.handlers = [NullHandler()]
                
            root_logger = logging.getLogger()
            original_root_handlers = list(root_logger.handlers)
            root_logger.handlers = [NullHandler()]
            
            original_tqdm = tqdm.__init__
            def silent_tqdm__init__(*args, **kwargs):
                kwargs['disable'] = True
                return original_tqdm(*args, **kwargs)
            tqdm.__init__ = silent_tqdm__init__
            
            try:
                graph, num_edges_after_filter = dataloader._build_graph_standard(nodes, split_name)
                fallback_triggered = getattr(graph, 'fallback_triggered', False)
                fallback_nodes_count = getattr(graph, 'fallback_nodes_count', 0)
                fallback_pct = (fallback_nodes_count / len(nodes) * 100) if len(nodes) > 0 else 0
                
                all_nodes_in_graph = graph.get_nodes()
                node_degrees = [len(node.get_adjacent_nodes()) for node in all_nodes_in_graph]
                
                total_edges = sum(node_degrees) // 2
                avg_degree = sum(node_degrees) / len(all_nodes_in_graph) if all_nodes_in_graph else 0
                
                results.append({
                    'quality_threshold': q_thresh_str,
                    'symmetry_threshold': s_thresh_str,
                    'embedding_threshold': e_thresh_str,
                    'average_degree': avg_degree,
                    'total_edges': total_edges,
                    'num_edges_after_filter': num_edges_after_filter,
                    'fallback_triggered': fallback_triggered,
                    'fallback_pct': fallback_pct
                })
                
                # Ensure logs directory exists for the CSV
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                if len(results) == 1:
                    pd.DataFrame([results[0]]).to_csv(log_file, index=False)
                else:
                    pd.DataFrame([results[-1]]).to_csv(log_file, mode='a', header=False, index=False)
                    
                progress_bar.set_postfix(avg_degree=f"{avg_degree:.2f}", total_edges=total_edges)
                
            except Exception as e:
                sys.stdout = original_stdout # Temporarily restore for error printing
                sys.stderr = original_stderr
                print(f"\n--- ERROR ENCOUNTERED during grid search for thresholds: Q={q_thresh_str}, S={s_thresh_str}, E={e_thresh_str} ---")
                traceback.print_exc()
                print("--------------------------------------------------------------------------------")
                print("Stopping grid search due to error.")
                # No need to re-suppress if raising, but good if continuing
                raise e 

            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                for logger_name, handlers in original_handlers.items():
                    logging.getLogger(logger_name).handlers = handlers
                logging.getLogger().handlers = original_root_handlers
                
                tqdm.__init__ = original_tqdm
        
        progress_bar.update(1)
    
    progress_bar.close()
    results_df = pd.DataFrame(results)
    return results_df

def visualize_search_results(results_df, output_prefix):
    """Create visualizations of search results"""
    plot_dir = 'logs/search_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(plot_dir, f'{output_prefix}_3d_plot.png'))
    plt.close()
    
    param_pairs = [
        ('quality_threshold', 'symmetry_threshold', 'embedding_threshold'),
        ('quality_threshold', 'embedding_threshold', 'symmetry_threshold'),
        ('symmetry_threshold', 'embedding_threshold', 'quality_threshold')
    ]
    
    for x_param, y_param, z_param in param_pairs:
        unique_z_values = sorted(results_df[z_param].unique())
        
        fig, axes = plt.subplots(
            nrows=1, 
            ncols=len(unique_z_values), 
            figsize=(5 * len(unique_z_values), 5),
            sharey=True
        )
        
        if len(unique_z_values) == 1:
            axes = [axes]
            
        for i, z_value in enumerate(unique_z_values):
            filtered_data = results_df[results_df[z_param] == z_value]
            pivot_data = filtered_data.pivot_table(
                index=y_param,
                columns=x_param,
                values='average_degree',
                aggfunc='mean'
            )
            
            im = axes[i].imshow(pivot_data, cmap='viridis', aspect='auto', origin='lower')
            axes[i].set_title(f'{z_param}={z_value}')
            axes[i].set_xlabel(x_param)
            if i == 0:
                axes[i].set_ylabel(y_param)
            
            plt.colorbar(im, ax=axes[i], label='Average Degree')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{output_prefix}_{x_param}_{y_param}_heatmap.png'))
        plt.close()
    
    params = ['quality_threshold', 'symmetry_threshold', 'embedding_threshold']
    for param in params:
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
        plt.savefig(os.path.join(plot_dir, f'{output_prefix}_{param}_effect.png'))
        plt.close()
    
    print(f"Visualizations saved to {plot_dir}/{output_prefix}_*.png")

def plot_subgroup_i_values(history, output_filename):
    """Plots the average I-value for each subgroup over hop instances."""
    if not history:
        print("No hop history recorded, skipping I-value plot.")
        return

    try:
        records = []
        for hop_index, hop_data in enumerate(history):
            for subgroup, avg_ivalue in hop_data.items():
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
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(15, 8))
        
        for subgroup in df['Subgroup'].unique():
            subgroup_df = df[df['Subgroup'] == subgroup]
            plt.plot(subgroup_df['HopInstance'], subgroup_df['AvgIValue'], marker='o', linestyle='-', label=subgroup)

        plt.xlabel('Hop Instance Index')
        plt.ylabel('Average I-Value')
        plt.title('Average I-Value per Subgroup Over Bias Hops')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plt.savefig(output_filename)
        print(f"Saved subgroup I-value plot to {output_filename}")
        plt.close()

    except Exception as e:
        print(f"Error generating subgroup I-value plot: {e}")
        import traceback # Redundant if already imported at top, but safe
        traceback.print_exc()

def load_and_prepare_data_splits(args, data_root):
    """Loads, splits, caches, and balances node data based on arguments."""
    train_nodes, val_nodes, test_nodes = None, None, None
    train_nodes_full, val_nodes_full, test_nodes_full = None, None, None

    node_loading_start = time.time()

    # Debug output for cache loading
    print(f"DEBUG: load_and_prepare_data_splits called with:")
    print(f"  args.use_cached: {getattr(args, 'use_cached', 'Not set')}")
    print(f"  args.cache_file: {getattr(args, 'cache_file', 'Not set')}")
    print(f"  args.cached_nodes: {getattr(args, 'cached_nodes', 'Not set')}")
    print(f"  args.dynamic_cache_detection: {getattr(args, 'dynamic_cache_detection', 'Not set')}")

    if args.use_cached:
        print(f"Attempting to load nodes from cache: {args.cache_file}")
        
        # Check if cache file exists
        if not os.path.exists(args.cache_file):
            print(f"ERROR: Cache file does not exist: {args.cache_file}")
            print("Falling back to direct dataset loading")
            args.use_cached = False
        else:
            print(f"Cache file exists: {args.cache_file}")
        
        # Check if dynamic cache detection is enabled
        if hasattr(args, 'dynamic_cache_detection') and args.dynamic_cache_detection:
            print("Dynamic cache detection enabled - will auto-detect cache size")
            # Load cache to determine available node counts
            try:
                with open(args.cache_file, 'rb') as f:
                    cache_data = dill.load(f)
                
                # Determine the appropriate node count based on fairness settings
                train_data = cache_data.get('train', {})
                if isinstance(train_data, dict):
                    # New cache format with separate full/balanced data
                    if args.fair_train and 'balanced' in train_data:
                        detected_count = len(train_data['balanced'])
                        print(f"Detected {detected_count} balanced training nodes from cache")
                    elif not args.fair_train and 'full' in train_data:
                        detected_count = len(train_data['full'])
                        print(f"Detected {detected_count} full training nodes from cache")
                    else:
                        # Fallback to total count
                        detected_count = len(train_data.get('full', train_data.get('balanced', [])))
                        print(f"Detected {detected_count} training nodes from cache (fallback)")
                else:
                    # Old cache format - direct list
                    detected_count = len(train_data) if train_data else 1000
                    print(f"Detected {detected_count} training nodes from cache (legacy format)")
                
                # Update the cached_nodes argument with detected count
                args.cached_nodes = detected_count
                print(f"Auto-detected cache size: {detected_count} nodes")
                
            except Exception as e:
                print(f"Warning: Could not auto-detect cache size: {e}")
                print("Falling back to default cache size of 1000")
                args.cached_nodes = 1000
        
        train_nodes = load_cached_nodes(args.cache_file, 'train', balanced=args.fair_train)
        val_nodes = load_cached_nodes(args.cache_file, 'val', balanced=args.fair_test)
        test_nodes = load_cached_nodes(args.cache_file, 'test', balanced=args.fair_test)

        if train_nodes is None or val_nodes is None or test_nodes is None:
            print("Failed to load one or more splits from cache. Will attempt direct loading.")
            args.use_cached = False  # Force direct load if cache failed
            train_nodes, val_nodes, test_nodes = None, None, None  # Reset
        else:
            print("Successfully loaded nodes from cache.")
            # If loaded from cache, full versions might not be loaded unless cache format changes or they are loaded separately.
            # For now, assume loaded lists are sufficient. Re-caching might not save original full sets.
            train_nodes_full = train_nodes # Placeholder if full not separately cached/loaded
            val_nodes_full = val_nodes   # Placeholder
            test_nodes_full = test_nodes  # Placeholder
            
            # Return early when cache loading succeeds to prevent unnecessary cache regeneration
            node_loading_time = time.time() - node_loading_start
            print(f"Node loading/preparation (from cache) time: {node_loading_time:.2f} seconds")
            return train_nodes, val_nodes, test_nodes, train_nodes_full, val_nodes_full, test_nodes_full, node_loading_time

    if not args.use_cached:
        print("Loading nodes directly from dataset...")
        # Initialize the AIFaceDataset with correct parameters
        # Using imported AIFaceDataset, ImageFileData, AttributeNode directly
        dataset = AIFaceDataset(data_root, ImageFileData, {}, AttributeNode, {"threshold": args.atr_threshold if hasattr(args, 'atr_threshold') else 2})
        
        print("Loading all nodes from dataset object...")
        all_nodes = dataset.load()
            
        print("Separating nodes by split...")
        train_nodes_full = [node for node in all_nodes if node.split == 'train']
        val_nodes_full = [node for node in all_nodes if node.split == 'val']
        test_nodes_full = [node for node in all_nodes if node.split == 'test']
        print(f"  Full Train: {len(train_nodes_full)}, Full Val: {len(val_nodes_full)}, Full Test: {len(test_nodes_full)}")

        if args.cache_nodes:
            print(f"Caching full node lists to {args.cache_file}...")
            save_cached_nodes(train_nodes_full, val_nodes_full, test_nodes_full, args.cache_file, target_num_nodes=args.cached_nodes)

        print("Applying balancing based on flags for graph construction...")
        train_nodes = balance_nodes_by_subgroup(train_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_train else list(train_nodes_full) # Ensure list copy
        val_nodes = balance_nodes_by_subgroup(val_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_test else list(val_nodes_full)   # Ensure list copy
        test_nodes = balance_nodes_by_subgroup(test_nodes_full, target_num_nodes=args.cached_nodes) if args.fair_test else list(test_nodes_full) # Ensure list copy
        
        print(f"  Final Train Nodes used for graph: {len(train_nodes)} ({'Balanced' if args.fair_train else 'Full from source'}) ({'Copy' if not args.fair_train else 'Balanced'}) ")
        print(f"  Final Val Nodes used for graph: {len(val_nodes)} ({'Balanced' if args.fair_test else 'Full from source'}) ({'Copy' if not args.fair_test else 'Balanced'}) ")
        print(f"  Final Test Nodes used for graph: {len(test_nodes)} ({'Balanced' if args.fair_test else 'Full from source'}) ({'Copy' if not args.fair_test else 'Balanced'}) ")

    node_loading_time = time.time() - node_loading_start
    print(f"Node loading/preparation (incl. balancing for graph use) time: {node_loading_time:.2f} seconds")

    return train_nodes, val_nodes, test_nodes, train_nodes_full, val_nodes_full, test_nodes_full, node_loading_time

def check_graph_cache_compatibility(config, data_root, graph_cache_dir="graph_cache"):
    """
    Check for exact graph cache file matches based on configuration.
    
    Returns:
        dict: Information about exact cache file matches for each split
    """
    import hashlib
    import glob
    import os
    
    # Extract configuration parameters
    quality_threshold = config.get('quality_threshold', 0.5)
    symmetry_threshold = config.get('symmetry_threshold', 0.3)
    embedding_threshold = config.get('embedding_threshold', 0.7)
    fair_train = config.get('fair_train', False)
    fair_test = config.get('fair_test', False)
    graph_type = config.get('graph_type', 'clustered')
    
    # Get actual node counts from existing cache if available
    node_cache_file = os.path.join(os.path.dirname(graph_cache_dir), 'node_cache', 'cached_nodes.pkl')
    actual_node_counts = {}
    
    try:
        if os.path.exists(node_cache_file):
            with open(node_cache_file, 'rb') as f:
                cache_data = dill.load(f)
            
            # Use the same dynamic cache detection logic as load_and_prepare_data_splits
            for split in ['train', 'val', 'test']:
                split_data = cache_data.get(split, {})
                if isinstance(split_data, dict):
                    # New cache format with separate full/balanced data
                    if split == 'train' and fair_train and 'balanced' in split_data:
                        actual_node_counts[split] = len(split_data['balanced'])
                    elif split in ['val', 'test'] and fair_test and 'balanced' in split_data:
                        actual_node_counts[split] = len(split_data['balanced'])
                    elif 'full' in split_data:
                        actual_node_counts[split] = len(split_data['full'])
                    else:
                        # Fallback to any available data
                        actual_node_counts[split] = len(split_data.get('full', split_data.get('balanced', [])))
                else:
                    # Old cache format - direct list
                    actual_node_counts[split] = len(split_data) if split_data else 1000
        else:
            # No cache file, use default values
            actual_node_counts = {
                'train': config.get('cached_nodes_count', 1000),
                'val': config.get('cached_nodes_count', 1000),
                'test': config.get('cached_nodes_count', 1000)
            }
    except Exception as e:
        # Fallback to config values if cache reading fails
        actual_node_counts = {
            'train': config.get('cached_nodes_count', 1000),
            'val': config.get('cached_nodes_count', 1000),
            'test': config.get('cached_nodes_count', 1000)
        }
    
    # Format threshold strings
    q_thresh_str = f"{quality_threshold:.3f}"
    s_thresh_str = f"{symmetry_threshold:.3f}"
    e_thresh_str = f"{embedding_threshold:.3f}"
    
    # Determine balancing status (this matches the test script logic)
    train_suffix = "balanced" if fair_train else "full"
    val_suffix = "balanced" if fair_test else "full"
    test_suffix = "balanced" if fair_test else "full"
    
    # Extract dataset name
    dataset_name = os.path.basename(os.path.normpath(data_root)) if data_root else "unknown_dataset"
    
    cache_info = {
        'train': {'exact_match': None, 'expected_filename': None},
        'val': {'exact_match': None, 'expected_filename': None},
        'test': {'exact_match': None, 'expected_filename': None}
    }
    
    # Check each split
    for split_name, suffix in [('train', train_suffix), ('val', val_suffix), ('test', test_suffix)]:
        # Use actual node count for this split
        node_count = actual_node_counts.get(split_name, 1000)
        
        # Build patterns (support legacy pkl and new streaming CSV, with graph_type in filename)
        patterns = [
            os.path.join(
                graph_cache_dir,
                f"{dataset_name}_{split_name}_{graph_type}_{suffix}_nodes_{node_count}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_*_graph.pkl"
            ),
            os.path.join(
                graph_cache_dir,
                f"{dataset_name}_{split_name}_{graph_type}_{suffix}_nodes_{node_count}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_*_edges.csv.gz"
            ),
            # Also accept legacy pattern without graph_type for backward compatibility
            os.path.join(
                graph_cache_dir,
                f"{dataset_name}_{split_name}_{suffix}_nodes_{node_count}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_*_graph.pkl"
            ),
        ]

        exact_match_path = None
        for pat in patterns:
            found = glob.glob(pat)
            if found:
                exact_match_path = found[0]
                break
        if exact_match_path:
            cache_info[split_name]['exact_match'] = exact_match_path

        # Store expected filename pattern (without hash) for UI display
        cache_info[split_name]['expected_filename'] = f"{dataset_name}_{split_name}_{graph_type}_{suffix}_nodes_{node_count}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_[hash]_edges.csv.gz"
    
    return cache_info

def find_existing_graph_caches(graph_cache_dir="graph_cache"):
    """
    Find all existing graph cache files and analyze their configurations.
    
    Returns:
        dict: Information about all existing graph cache files
    """
    import glob
    import os
    import re
    
    cache_files = glob.glob(os.path.join(graph_cache_dir, "*_edges.csv.gz"))

    cache_analysis = {}
    for cache_file in cache_files:
        filename = os.path.basename(cache_file)
        
        # Parse filename to extract configuration; only CSV.gz supported now
        # ai-face_train_clustered_full_nodes_827339_q0.500_s0.300_e0.700_hashHASH_edges.csv.gz
        pattern = r"(.+)_(train|val|test)_(clustered|nonclustered|clustered_subclustered|nonclustered_subclustered)_(balanced|full)_nodes_(\d+)_q([\d.]+)_s([\d.]+)_e([\d.]+)_hash([a-f0-9]+)_edges\.csv\.gz$"
        match = re.match(pattern, filename)
        
        if match:
            dataset, split, graph_type, balancing, node_count, q_thresh, s_thresh, e_thresh, node_hash = match.groups()
            
            config_key = f"{dataset}_{split}_{graph_type}_{balancing}_{node_count}_q{q_thresh}_s{s_thresh}_e{e_thresh}"
            
            if config_key not in cache_analysis:
                cache_analysis[config_key] = {
                    'dataset': dataset,
                    'split': split,
                    'graph_type': graph_type,
                    'balancing': balancing,
                    'node_count': int(node_count),
                    'quality_threshold': float(q_thresh),
                    'symmetry_threshold': float(s_thresh),
                    'embedding_threshold': float(e_thresh),
                    'cache_files': []
                }
            
            cache_analysis[config_key]['cache_files'].append({
                'filename': filename,
                'full_path': cache_file,
                'node_hash': node_hash,
                'file_size': os.path.getsize(cache_file),
                'modified_time': os.path.getmtime(cache_file)
            })

    return cache_analysis
