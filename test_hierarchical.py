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
    parser.add_argument('--cache-file', type=str, default='cached_nodes.pkl', 
                        help='Path to cache file for saving/loading nodes')
    
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
    
    return parser.parse_args()

def load_cached_nodes(cache_file, use_full_cache=False):
    """Load nodes from cache file
    
    Args:
        cache_file: Path to the cache file
        use_full_cache: If True, load the full dataset; if False, load the subset
    
    Returns:
        Tuple of (train_nodes, val_nodes, test_nodes) or (None, None, None) if error
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} does not exist!")
        return None, None, None
    
    print(f"Loading nodes from cache file: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            cached_data = dill.load(f)
            
        # Check cache format
        if isinstance(cached_data, dict) and 'full' in cached_data and 'subset' in cached_data:
            # New dual cache format
            cache_type = 'full' if use_full_cache else 'subset'
            train = cached_data[cache_type]['train']
            val = cached_data[cache_type]['val']
            test = cached_data[cache_type]['test']
            
            # Print cache information
            if 'metadata' in cached_data:
                metadata = cached_data['metadata']
                print(f"Using {'full' if use_full_cache else 'subset'} cache")
                print(f"Full cache contains: {metadata['full_sizes']['train']} train, "
                      f"{metadata['full_sizes']['val']} val, "
                      f"{metadata['full_sizes']['test']} test nodes")
                print(f"Subset cache contains: {metadata['subset_sizes']['train']} train, "
                      f"{metadata['subset_sizes']['val']} val, "
                      f"{metadata['subset_sizes']['test']} test nodes")
        else:
            # Legacy cache format - for backwards compatibility
            print("Using legacy cache format")
            train = cached_data.get('train', [])
            val = cached_data.get('val', [])
            test = cached_data.get('test', [])
            
        print(f"Loaded {len(train)} train, {len(val)} val, {len(test)} test nodes from cache")
        return train, val, test
    except Exception as e:
        print(f"Error loading cached nodes: {e}")
        return None, None, None

def cache_nodes(dataloader=None, cache_file="cached_nodes.pkl", nodes_per_split=1000, train_nodes=None, val_nodes=None, test_nodes=None):
    """Cache nodes to speed up testing
    
    Args:
        dataloader: Dataloader instance (only needed if nodes are not provided)
        cache_file: Path to save the cache file
        nodes_per_split: Number of nodes per split for subset caching
        train_nodes: Pre-loaded train nodes (if None, will load from dataloader)
        val_nodes: Pre-loaded val nodes (if None, will load from dataloader)
        test_nodes: Pre-loaded test nodes (if None, will load from dataloader)
    """
    # Only load nodes if they haven't been provided
    if train_nodes is None or val_nodes is None or test_nodes is None:
        print(f"Loading nodes for caching...")
        start_time = time.time()
        
        # Get nodes from all datasets
        all_nodes = []
        for dataset in dataloader.datasets:
            all_nodes.extend(dataset.load())
            
        # Create node lists for each split
        train_nodes = [node for node in all_nodes if node.split == 'train']
        val_nodes = [node for node in all_nodes if node.split == 'val']
        test_nodes = [node for node in all_nodes if node.split == 'test']
    else:
        print(f"Using pre-loaded nodes for caching...")
        start_time = time.time()
    
    loading_time = time.time() - start_time
    print(f"Loaded {len(train_nodes)} train, {len(val_nodes)} val, {len(test_nodes)} test nodes in {loading_time:.2f} seconds")
    
    # Create subset of nodes for each split
    subset_train = train_nodes[:min(len(train_nodes), nodes_per_split)]
    subset_val = val_nodes[:min(len(val_nodes), nodes_per_split)]
    subset_test = test_nodes[:min(len(test_nodes), nodes_per_split)]
    
    # Create both full and subset caches
    full_data = {
        'train': train_nodes,
        'val': val_nodes,
        'test': test_nodes
    }
    
    subset_data = {
        'train': subset_train,
        'val': subset_val,
        'test': subset_test
    }
    
    # Store both datasets in the cache
    cached_data = {
        'full': full_data,
        'subset': subset_data,
        'metadata': {
            'full_sizes': {
                'train': len(train_nodes),
                'val': len(val_nodes), 
                'test': len(test_nodes)
            },
            'subset_sizes': {
                'train': len(subset_train),
                'val': len(subset_val), 
                'test': len(subset_test)
            },
            'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    
    print(f"\nCaching nodes to: {cache_file}")
    print(f"Full cache: {len(train_nodes)} train, {len(val_nodes)} val, {len(test_nodes)} test nodes")
    print(f"Subset cache: {len(subset_train)} train, {len(subset_val)} val, {len(subset_test)} test nodes")
    
    # Debug: count nodes with embeddings before caching
    embedding_counts = {}
    for split_name, nodes_list in [("train", train_nodes), ("val", val_nodes), ("test", test_nodes)]:
        with_embedding = sum(1 for node in nodes_list if 'face_embedding' in node.attributes)
        valid_embedding = sum(1 for node in nodes_list 
                             if 'face_embedding' in node.attributes 
                             and isinstance(node.attributes['face_embedding'], np.ndarray) 
                             and not np.all(np.isclose(node.attributes['face_embedding'], 0)))
        embedding_counts[split_name] = (with_embedding, valid_embedding, len(nodes_list))
    
    print("\nEmbedding statistics before caching:")
    for split_name, (with_emb, valid_emb, total) in embedding_counts.items():
        print(f"  {split_name}: {with_emb}/{total} nodes with embedding attribute ({with_emb/total*100:.2f}%)")
        if with_emb > 0:
            print(f"    Valid embeddings: {valid_emb}/{with_emb} ({valid_emb/with_emb*100:.2f}%)")
    
    # Use a higher pickle protocol to ensure numpy arrays are properly serialized
    with open(cache_file, 'wb') as f:
        dill.dump(cached_data, f)
    print("Cache saved successfully!")
    
    # Verify cache immediately to ensure embeddings are properly saved
    print("\nVerifying cache...")
    with open(cache_file, 'rb') as f:
        verification_data = dill.load(f)
    
    # Count embeddings in cached data
    print("Embedding statistics after caching:")
    for data_type in ['subset']:  # Just check subset for speed
        for split_name, nodes_list in verification_data[data_type].items():
            with_embedding = sum(1 for node in nodes_list if 'face_embedding' in node.attributes)
            valid_embedding = sum(1 for node in nodes_list 
                               if 'face_embedding' in node.attributes 
                               and isinstance(node.attributes['face_embedding'], np.ndarray) 
                               and not np.all(np.isclose(node.attributes['face_embedding'], 0)))
            total = len(nodes_list)
            print(f"  {data_type}/{split_name}: {with_embedding}/{total} nodes with embedding attribute ({with_embedding/total*100:.2f}%)")
            if with_embedding > 0:
                print(f"    Valid embeddings: {valid_embedding}/{with_embedding} ({valid_embedding/with_embedding*100:.2f}%)")

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

def main():
    # Set higher recursion depth for pickling large graphs
    sys.setrecursionlimit(3000) 
    
    data_root = "/home/brg2890/major/datasets/ai-face"
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/hierarchical_test_{timestamp}.log"
    print(f"Starting test run, logging to: {log_file}")
    
    # Set up a dataloader for loading datasets
    edge_class = Edge
    train_nodes, val_nodes, test_nodes = None, None, None
    
    # Load nodes (either from cache or directly from datasets)
    node_loading_start = time.time()
    
    if args.use_cached:
        # Try to load from cache - either full or subset based on args
        train_nodes, val_nodes, test_nodes = load_cached_nodes(
            cache_file=args.cache_file,
            use_full_cache=args.use_full_cache
        )
        
        if train_nodes is None or val_nodes is None or test_nodes is None:
            print("Error loading from cache, falling back to direct loading")
            train_nodes, val_nodes, test_nodes = None, None, None
    
    if train_nodes is None or val_nodes is None or test_nodes is None:
        # Load nodes directly from datasets
        print("Loading nodes from datasets...")
        # Initialize the AIFaceDataset with correct parameters (using positional arguments)
        dataset = AIFaceDataset(data_root, ImageFileData, {}, AttributeNode, {"threshold": 2})
        
        dataloader = HierarchicalDeepfakeDataloader(
            datasets=[dataset],
            edge_class=Edge
        )
        
        # Load all nodes directly from the dataset (avoid using dataloader.load() which would load again)
        print("Loading nodes from dataset...")
        all_nodes = dataset.load()
            
        # Create node lists for each split
        print("Separating nodes by split...")
        train_nodes = [node for node in all_nodes if node.split == 'train']
        val_nodes = [node for node in all_nodes if node.split == 'val']
        test_nodes = [node for node in all_nodes if node.split == 'test']
        
        # Cache nodes if requested
        if args.cache_nodes:
            print("Caching nodes...")
            cache_nodes(
                cache_file=args.cache_file,
                nodes_per_split=args.cached_nodes,
                train_nodes=train_nodes,
                val_nodes=val_nodes,
                test_nodes=test_nodes
            )
    
    # Limit the number of nodes in test mode
    if args.test and not args.use_full_cache:
        limit = args.cached_nodes
        train_nodes = train_nodes[:min(len(train_nodes), limit)]
        val_nodes = val_nodes[:min(len(val_nodes), limit)]
        test_nodes = test_nodes[:min(len(test_nodes), limit)]
    
    node_loading_time = time.time() - node_loading_start
    print(f"Node loading time: {node_loading_time:.2f} seconds")

    # If in search mode, run grid search
    if args.search:
        # Select nodes for the specified split
        if args.search_split == 'train':
            search_nodes = train_nodes
        elif args.search_split == 'val':
            search_nodes = val_nodes
        else:  # test
            search_nodes = test_nodes
            
        print(f"\nRunning threshold grid search on {args.search_split} split with {len(search_nodes)} nodes")
        results_df = run_threshold_grid_search(
            nodes=search_nodes,
            edge_class=Edge,
            split_name=args.search_split,
            quality_steps=args.quality_steps,
            symmetry_steps=args.symmetry_steps,
            embedding_steps=args.embedding_steps
        )
        
        # Save full results to CSV
        output_file = f"logs/{args.search_results}"
        results_df.to_csv(output_file, index=False)
        print(f"\nSearch results saved to {output_file}")
        
        # Create visualizations
        output_prefix = f"{args.search_split}_{timestamp}"
        visualize_search_results(results_df, output_prefix)
        
        # Print top 5 configurations by average degree
        print("\nTop 5 threshold configurations by average degree:")
        top_configs = results_df.sort_values('average_degree', ascending=False).head(5)
        for _, row in top_configs.iterrows():
            fallback_info = "" if not row.get('fallback_triggered', False) else f"[FALLBACK USED - {row.get('fallback_pct', 0):.1f}% nodes]"
            edges_after_filter = row.get('num_edges_after_filter', 'unknown')
            
            print(f"Quality: {row['quality_threshold']:.2f}, "
                  f"Symmetry: {row['symmetry_threshold']:.2f}, "
                  f"Embedding: {row['embedding_threshold']:.2f}, "
                  f"Avg Degree: {row['average_degree']:.2f}, "
                  f"Total Edges: {row['total_edges']}, "
                  f"Edges After Filter: {edges_after_filter}, "
                  f"{fallback_info}")

        # Print bottom 5 configurations by average degree
        print("\nBottom 5 threshold configurations by average degree:")
        bottom_configs = results_df.sort_values('average_degree').head(5)
        for _, row in bottom_configs.iterrows():
            fallback_info = "" if not row.get('fallback_triggered', False) else f"[FALLBACK USED - {row.get('fallback_pct', 0):.1f}% nodes]"
            edges_after_filter = row.get('num_edges_after_filter', 'unknown')
            
            print(f"Quality: {row['quality_threshold']:.2f}, "
                  f"Symmetry: {row['symmetry_threshold']:.2f}, "
                  f"Embedding: {row['embedding_threshold']:.2f}, "
                  f"Avg Degree: {row['average_degree']:.2f}, "
                  f"Total Edges: {row['total_edges']}, "
                  f"Edges After Filter: {edges_after_filter}, "
                  f"{fallback_info}")
        
        return
    
    # Regular mode: Create dataloader and build graphs
    graph_cache_dir = "graph_cache"
    os.makedirs(graph_cache_dir, exist_ok=True)
    q_thresh_str = f"{args.quality_threshold:.3f}"
    s_thresh_str = f"{args.symmetry_threshold:.3f}"
    e_thresh_str = f"{args.embedding_threshold:.3f}"
    
    graph_construction_start = time.time()
    train_graph, val_graph, test_graph = None, None, None

    for split_name in ['train', 'val', 'test']:
        # Extract dataset name from data_root path (Corrected)
        dataset_name = os.path.basename(os.path.normpath(data_root)) if data_root else "unknown_dataset"
        # Determine subset identifier (Corrected)
        subset_id = f"subset" if args.use_cached and not args.use_full_cache else "full"
        
        dataloader = HierarchicalDeepfakeDataloader(
            datasets=[],  # Empty since we're providing nodes directly
            edge_class=Edge,
            test_mode=args.test,
            visualize=args.visualize,
            show_viz=args.show,
            quality_threshold=args.quality_threshold,
            symmetry_threshold=args.symmetry_threshold,
            embedding_threshold=args.embedding_threshold
        )
        
        # Determine subset identifier
        subset_id = f"subset{args.cached_nodes}" if args.cached_nodes else "full"
        # Create Specific Cache Filename
        cache_filename = os.path.join(
            graph_cache_dir,
            f"{dataset_name}_{split_name}_{subset_id}_q{q_thresh_str}_s{s_thresh_str}_e{e_thresh_str}_edges.pkl"
        )

        # Check/Load Graph Cache
        graph = None
        loaded_from_cache = False

        if os.path.exists(cache_filename):
            try:
                print(f"\nFound edge cache file: {cache_filename}. Attempting to load.")
                # 1. Load Nodes (ensure nodes are loaded for the split)
                split_nodes = train_nodes if split_name == 'train' else val_nodes if split_name == 'val' else test_nodes
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
            split_nodes = train_nodes if split_name == 'train' else val_nodes if split_name == 'val' else test_nodes
            if not split_nodes:
                 print(f"Error: Nodes for split '{split_name}' not available for building graph.")
                 continue # Or handle error appropriately
                 
            print(f"\nBuilding graph for {split_name} split ({len(split_nodes)} nodes)... No suitable cache found or --use-cached=False.")
            # Use the dataloader to build the graph
            # Assuming dataloader.build_graph returns the graph object directly now
            # If it still returns a tuple, adjust accordingly (e.g., graph = dataloader.build_graph(...)[0] )
            graph_build_result = dataloader._build_graph_standard(split_nodes, split_name)
            
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
        
    graph_construction_time = time.time() - graph_construction_start
    total_time = time.time() - node_loading_start

    # Performance Reporting & Validation
    print("\nPerformance:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"  - Node loading: {node_loading_time:.2f} seconds ({node_loading_time/total_time*100:.1f}%)")
    print(f"  - Graph construction: {graph_construction_time:.2f} seconds ({graph_construction_time/total_time*100:.1f}%)")

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
    print(f"Overall processing speed: {total_nodes / total_time:.2f} nodes/second")
    print(f"Graph construction speed: {total_nodes / graph_construction_time:.2f} nodes/second")
    
    # Count total edges
    train_edges = sum(len(node.edges) for node in train_graph.get_nodes()) // 2
    val_edges = sum(len(node.edges) for node in val_graph.get_nodes()) // 2
    test_edges = sum(len(node.edges) for node in test_graph.get_nodes()) // 2
    total_edges = train_edges + val_edges + test_edges
    
    print(f"Created {total_edges} total edges")
    print(f"Edge creation speed: {total_edges / graph_construction_time:.2f} edges/second")
    
    # Print average degree (edges per node)
    print(f"Average degree: {(total_edges * 2) / len(train_graph.get_nodes()):.2f}")
    
    # Create graph managers for each split
    train_manager = NoGraphManager(copy.deepcopy(train_graph))
    val_manager = NoGraphManager(copy.deepcopy(val_graph))
    test_manager = NoGraphManager(copy.deepcopy(test_graph))
    
    # Define architectures to test
    cnn_architectures = [
        "swintransformdf",
        "resnestdf", 
        #"effnetdf",
        #"mesonetdf",
        #"squeezenetdf",
        #"vistransformdf",
        
    ]

    random.seed(13247987501)
    
    # Define traversal types to compare
    #traversal_types = ["comprehensive", "random", "i-value"]
    traversal_types = ["i-value"]
    
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
                # Create model with adjusted learning rate
                model = CNNModel(
                    f"/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/{arch}_{traversal_type}_{timestamp}.pt",
                    arch,
                    0.001, 
                    True
                )
                
                # Create trainer based on traversal type
                if traversal_type == "i-value":
                    trainer = IValueTrainer(
                        train_manager,  # Use train manager
                        None,  # Will be set by get_traversal
                        None,  # Will be set by get_traversal
                        [model],
                        attribute_metadata=attribute_metadata
                    )
                else:  # random or comprehensive traversal
                    trainer = ExperimentTrainer(
                        train_manager,  # Use train manager
                        None,  # Will be set by get_traversal
                        None,  # Will be set by get_traversal
                        None,  # Will be set by get_traversal
                        [model],
                        traversal_type=traversal_type,
                        attribute_metadata=attribute_metadata
                    )
                
                # Create traversals for each split
                # Use more pointers and adjust steps based on graph sizes
                train_size = len(train_manager.graph.get_nodes())
                val_size = len(val_manager.graph.get_nodes())
                test_size = len(test_manager.graph.get_nodes())
                
                print(f"\nGraph sizes:")
                print(f"Train: {train_size} nodes")
                print(f"Val: {val_size} nodes")
                print(f"Test: {test_size} nodes")
                
                # Calculate appropriate number of steps
                train_steps = 2000
                val_steps = 1000 
                test_steps = None  # Use None to visit all test nodes
                
                if traversal_type == "comprehensive":
                    train_traversal = ComprehensiveTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                else:
                    train_traversal = trainer.get_traversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                val_traversal = ComprehensiveTraversal(val_manager.graph, num_pointers=1, num_steps=val_steps)
                test_traversal = ComprehensiveTraversal(test_manager.graph, num_pointers=1, num_steps=test_steps)
                
                print(f"\nTraversal settings:")
                print(f"Train: {train_steps} steps with 1 pointers")
                print(f"Val: {val_steps} steps with 1 pointers")
                print(f"Test: All nodes with 1 pointers")
                
                # Update trainer with correct traversals
                trainer.train_traversal = train_traversal
                trainer.val_traversal = val_traversal
                trainer.test_traversal = test_traversal
                
                try:
                    print(f"Training {arch} with {traversal_type} traversal...")
                    trainer.run(num_epochs=5)
                    print(f"Testing {arch} with {traversal_type} traversal...")
                    test_metrics = trainer.test()
                    
                    # Handle both single dict and list of dicts for backwards compatibility
                    if isinstance(test_metrics, list):
                        for i, metrics in enumerate(test_metrics):
                            print(f"\nModel {i+1} Results:")
                            print(f"Loss: {metrics.get('avg_loss', 0.0):.4f}")
                            print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                            print(f"Bias Loss: {metrics.get('avg_bias_loss', 0.0):.4f}")
                    else:
                        print("\nTest Results:")
                        print(f"Loss: {test_metrics.get('avg_loss', 0.0):.4f}")
                        print(f"Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
                        print(f"Bias Loss: {test_metrics.get('avg_bias_loss', 0.0):.4f}")
                        
                    print(f"\nCompleted evaluation of {arch} with {traversal_type} traversal")
                except Exception as e:
                    print(f"\nError while evaluating {arch} with {traversal_type}: {str(e)}")
                    log_exception(logfile, *sys.exc_info())
                    continue  # Continue with next configuration
                    
            except Exception as e:
                print(f"\nError while setting up {arch} with {traversal_type}: {str(e)}")
                log_exception(logfile, *sys.exc_info())
                continue  # Continue with next configuration


    print("\nDone!")

if __name__ == "__main__":
    main()
