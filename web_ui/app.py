#!/usr/bin/env python3
"""
HyperGraph Test Configuration Web UI

This file is the HTTP layer for the Web UI. It:

This UI allows users to:
- Create and save test configurations
- Load and modify existing configurations  
- Start test runs remotely
- Monitor test progress
- View and compare results
- Export configurations and results
- Generate and manage cache files

Author: Quanty 7
"""

import os
import json
import time
import subprocess
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import yaml
import dill
import glob

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting HyperGraph Test Configuration Web UI...")
logger.info(f"Log file location: {log_file}")

# Add parent directory to path to import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui.config_manager import ConfigManager
from web_ui.test_runner import TestRunner
from test_helpers.data_graph_utils import (
    balance_nodes_by_subgroup, save_cached_nodes, load_cached_nodes,
    load_and_prepare_data_splits, check_graph_cache_compatibility,
    find_existing_graph_caches
)
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from graphs.HyperGraph import HyperGraph
from edges.Edge import Edge

# Import utilities from the new helper module
from test_helpers.logging_utils import NullHandler, capture_output, log_exception, set_seed
from test_helpers.args_utils import parse_args
from test_helpers.data_graph_utils import (
    run_threshold_grid_search, visualize_search_results, plot_subgroup_i_values,
    load_and_prepare_data_splits, check_graph_cache_compatibility
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

# Import the new GPU queue manager
from web_ui.gpu_queue_manager import GPUQueueManager

app = Flask(__name__)
app.secret_key = 'quanty_hypergraph_test_ui_secret_key_2024'

# -----------------------------------------------------------------------------
# Core managers (persistence + execution)
# -----------------------------------------------------------------------------
# `ConfigManager` reads/writes configuration JSON files and template JSON files.
config_manager = ConfigManager()
# `GPUQueueManager` queues runs and starts training/testing subprocesses on GPUs.
gpu_queue_manager = GPUQueueManager()  # Replace test_runner with gpu_queue_manager

# -----------------------------------------------------------------------------
# Debug / operational routes
# -----------------------------------------------------------------------------
@app.route('/debug/ping')
def ping():
    """Simple endpoint to check if server is running."""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/debug/paths')
def debug_paths():
    """Debug endpoint to check important paths."""
    paths = {
        'current_dir': os.getcwd(),
        'app_dir': os.path.dirname(os.path.abspath(__file__)),
        'project_root': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node_cache_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'node_cache'),
        'graph_cache_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graph_cache')
    }
    return jsonify(paths)

# -----------------------------------------------------------------------------
# Page routes (HTML)
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    """Main dashboard showing overview of configurations and results."""
    configs = config_manager.list_configurations()
    runs = gpu_queue_manager.list_runs()  # Use gpu_queue_manager instead of test_runner

    # Compatibility patch for templates:
    # Some run metadata store accuracy nested under `results.final_accuracy`, while templates
    # often expect top-level `final_accuracy` (and sometimes `accuracy`).
    for run in runs:
        if 'final_accuracy' not in run:
            accuracy = None
            if 'results' in run and isinstance(run['results'], dict):
                accuracy = run['results'].get('final_accuracy')
            if accuracy is not None:
                run['final_accuracy'] = accuracy
                run['accuracy'] = accuracy / 100.0 if accuracy > 1.5 else accuracy  # for consistency with /results

    recent_runs = sorted(runs, key=lambda x: x.get('start_time') or '', reverse=True)[:10]
    
    return render_template('index.html', 
                         configs=configs, 
                         recent_runs=recent_runs,
                         active_runs=gpu_queue_manager.get_queue_status())  # Use gpu_queue_manager

@app.route('/cache/status')
def cache_status_page():
    """Page showing cache status information."""
    logger.info("Rendering cache status page")
    return render_template('cache_status.html')

# -----------------------------------------------------------------------------
# Cache APIs
# -----------------------------------------------------------------------------
@app.route('/api/cache/status')
def get_cache_status():
    """API endpoint for getting cache status information."""
    logger.info("=== Cache Status Request Started ===")
    try:
        logger.info("Fetching cache status...")
        
        # Check if cache directories exist
        node_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'node_cache')
        graph_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graph_cache')
        
        logger.info(f"Node cache directory: {node_cache_dir}")
        logger.info(f"Graph cache directory: {graph_cache_dir}")
        
        missing_directories = []
        if not os.path.exists(node_cache_dir):
            logger.info(f"Node cache directory does not exist yet: {node_cache_dir}")
            missing_directories.append(node_cache_dir)

        if not os.path.exists(graph_cache_dir):
            logger.info(f"Graph cache directory does not exist yet: {graph_cache_dir}")
            missing_directories.append(graph_cache_dir)
        
        # Get basic file information first (fast)
        node_cache_status = get_node_cache_status_basic()
        graph_cache_status = get_graph_cache_status_basic()
        
        # Get all existing graph cache configurations
        try:
            existing_graph_caches = find_existing_graph_caches(graph_cache_dir)
        except Exception as e:
            logger.warning(f"Error finding existing graph caches: {e}")
            existing_graph_caches = {}
        
        cache_status = {
            'node_cache': node_cache_status,
            'graph_cache': graph_cache_status,
            'existing_graph_caches': existing_graph_caches,
            'missing_directories': missing_directories,
            'timestamp': time.time()
        }
        
        # Add cache control headers
        response = jsonify(cache_status)
        response.headers['Cache-Control'] = 'public, max-age=300'  # Cache for 5 minutes
        response.headers['ETag'] = f'"{hash(str(cache_status))}"'
        
        logger.info("=== Cache Status Request Completed Successfully ===")
        return response
    except Exception as e:
        logger.error("=== Cache Status Request Failed ===")
        logger.error(f"Error getting cache status: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to get cache status',
            'details': str(e)
        }), 500

def get_node_cache_status_basic():
    """Get status information for node cache files."""
    node_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'node_cache')
    cache_status = {}
    
    try:
        logger.info(f"Checking node cache in: {node_cache_dir}")
        cache_file = os.path.join(node_cache_dir, 'cached_nodes.pkl')
        logger.info(f"Looking for cache file: {cache_file}")
        
        if os.path.exists(cache_file):
            logger.info(f"Found cache file: {cache_file}")
            file_size = os.path.getsize(cache_file)
            last_modified = os.path.getmtime(cache_file)
            
            # Get basic file info
            cache_status = {
                'exists': True,
                'last_modified': last_modified,
                'size': file_size,
                'file_path': cache_file
            }
            
            try:
                logger.info("Loading node cache data...")
                with open(cache_file, 'rb') as f:
                    cache_data = dill.load(f)
                logger.info("Successfully loaded node cache data")
                
                # Get node counts for each split
                for split in ['train', 'val', 'test']:
                    if split in cache_data:
                        split_data = cache_data[split]
                        if isinstance(split_data, dict):
                            cache_status[split] = {
                                'node_count': len(split_data.get('full', [])),
                                'balanced_count': len(split_data.get('balanced', []))
                            }
                        else:
                            cache_status[split] = {
                                'node_count': len(split_data)
                            }
            except Exception as e:
                logger.error(f"Error loading node cache data: {str(e)}", exc_info=True)
                cache_status['load_error'] = str(e)
        else:
            logger.warning(f"Cache file not found at {cache_file}")
            cache_status = {
                'exists': False,
                'error': f'Cache file not found at {cache_file}'
            }
    except Exception as e:
        logger.error(f"Error checking node cache status: {str(e)}", exc_info=True)
        cache_status = {
            'error': f'Error checking cache status: {str(e)}'
        }
    
    return cache_status

def get_graph_cache_status_basic():
    """Get status information for graph cache files."""
    graph_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graph_cache')
    cache_status = {}
    
    try:
        logger.info(f"Checking graph cache in: {graph_cache_dir}")
        pkl_pattern = os.path.join(graph_cache_dir, '*_graph.pkl')
        csv_pattern = os.path.join(graph_cache_dir, '*_edges.csv.gz')
        pkl_files = glob.glob(pkl_pattern)
        csv_files = glob.glob(csv_pattern)
        cache_files = pkl_files + csv_files
        logger.info(f"Found {len(cache_files)} graph cache files (pkl={len(pkl_files)}, csv.gz={len(csv_files)})")
        
        for cache_file in cache_files:
            try:
                filename = os.path.basename(cache_file)
                logger.info(f"Processing cache file: {filename}")
                
                # Extract split name from filename
                split_match = None
                for split in ['train', 'val', 'test']:
                    if f'_{split}_' in filename:
                        split_match = split
                        break
                
                if split_match:
                    file_size = os.path.getsize(cache_file)
                    last_modified = os.path.getmtime(cache_file)
                    
                    split_status = {
                        'exists': True,
                        'last_modified': last_modified,
                        'size': file_size,
                        'file_path': cache_file
                    }
                    
                    # Only attempt to load pickled caches; for CSV.gz report presence only (could be huge)
                    if cache_file.endswith('_graph.pkl'):
                        try:
                            logger.info(f"Loading graph cache data for {split_match}...")
                            with open(cache_file, 'rb') as f:
                                edge_list = dill.load(f)
                            logger.info(f"Successfully loaded graph cache data for {split_match}")
                            split_status['edge_count'] = len(edge_list)
                        except Exception as e:
                            logger.error(f"Error loading graph cache data: {str(e)}", exc_info=True)
                            split_status['load_error'] = str(e)
                    else:
                        split_status['edge_count'] = None
                    
                    cache_status[split_match] = split_status
            except Exception as e:
                logger.error(f"Error processing graph cache file {cache_file}: {str(e)}", exc_info=True)
                if split_match:
                    cache_status[split_match] = {
                        'exists': False,
                        'error': f'Error processing cache file: {str(e)}'
                    }
        
        # Ensure all splits are represented
        for split in ['train', 'val', 'test']:
            if split not in cache_status:
                logger.warning(f"No cache file found for {split} split")
                cache_status[split] = {
                    'exists': False,
                    'error': 'No cache file found'
                }
    except Exception as e:
        logger.error(f"Error checking graph cache status: {str(e)}", exc_info=True)
        cache_status['error'] = f'Error checking cache status: {str(e)}'
    
    return cache_status

@app.route('/api/cache/generate', methods=['POST'])
def generate_cache():
    """API endpoint for generating new cache files."""
    try:
        data = request.get_json()
        
        # Extract parameters
        generate_node_cache = data.get('generateNodeCache', False)
        balance_nodes = data.get('balanceNodes', False)
        generate_graph_cache = data.get('generateGraphCache', False)
        quality_threshold = data.get('qualityThreshold', 0.5)
        symmetry_threshold = data.get('symmetryThreshold', 0.5)
        embedding_threshold = data.get('embeddingThreshold', 0.5)
        graph_type = data.get('graphType', data.get('graph_type', 'clustered'))  # NEW: graph type
        
        # Start cache generation in a background thread
        thread = threading.Thread(
            target=generate_cache_background,
            args=(
                generate_node_cache,
                balance_nodes,
                generate_graph_cache,
                quality_threshold,
                symmetry_threshold,
                embedding_threshold,
                graph_type  # Pass graph type
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Cache generation started in background'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_cache_background(
    generate_node_cache,
    balance_nodes,
    generate_graph_cache,
    quality_threshold,
    symmetry_threshold,
    embedding_threshold,
    graph_type  # NEW: graph type
):
    """Background task to generate cache files."""
    try:
        data_root = "/home/brg2890/major/datasets/ai-face"
        
        # Load and prepare data splits
        train_nodes, val_nodes, test_nodes, \
        train_nodes_full, val_nodes_full, test_nodes_full, \
        node_loading_time = load_and_prepare_data_splits(None, data_root)
        
        # Generate node cache if requested
        if generate_node_cache:
            print("Generating node cache...")
            node_cache_dir = "node_cache"
            os.makedirs(node_cache_dir, exist_ok=True)
            
            # Save nodes for each split
            cache_data = {}
            for split_name, nodes in [
                ('train', train_nodes if balance_nodes else train_nodes_full),
                ('val', val_nodes if balance_nodes else val_nodes_full),
                ('test', test_nodes if balance_nodes else test_nodes_full)
            ]:
                cache_data[split_name] = nodes
            
            # Save to cache file
            cache_file = os.path.join(node_cache_dir, 'cached_nodes.pkl')
            with open(cache_file, 'wb') as f:
                dill.dump(cache_data, f)
            print(f"Node cache saved to {cache_file}")
        
        # Generate graph cache if requested
        if generate_graph_cache:
            print("Generating graph cache...")
            graph_cache_dir = "graph_cache"
            os.makedirs(graph_cache_dir, exist_ok=True)
            
            # Select dataloader based on graph_type
            if graph_type == 'nonclustered':
                dataloader = UnclusteredDeepfakeDataloader(
                    datasets=[],
                    edge_class=Edge,
                    test_mode=False,
                    visualize=False,
                    show_viz=False,
                    quality_threshold=quality_threshold,
                    symmetry_threshold=symmetry_threshold,
                    embedding_threshold=embedding_threshold,
                    silent_mode=True
                )
                graph_type_str = 'nonclustered'
                subclustering = False
            elif graph_type == 'nonclustered_subclustered':
                dataloader = UnclusteredDeepfakeDataloader(
                    datasets=[],
                    edge_class=Edge,
                    test_mode=False,
                    visualize=False,
                    show_viz=False,
                    quality_threshold=quality_threshold,
                    symmetry_threshold=symmetry_threshold,
                    embedding_threshold=embedding_threshold,
                    silent_mode=True
                )
                graph_type_str = 'nonclustered_subclustered'
                subclustering = True
            elif graph_type == 'clustered_subclustered':
                dataloader = HierarchicalDeepfakeDataloader(
                    datasets=[],
                    edge_class=Edge,
                    test_mode=False,
                    visualize=False,
                    show_viz=False,
                    quality_threshold=quality_threshold,
                    symmetry_threshold=symmetry_threshold,
                    embedding_threshold=embedding_threshold,
                    silent_mode=True
                )
                graph_type_str = 'clustered_subclustered'
                subclustering = True
            else:
                dataloader = HierarchicalDeepfakeDataloader(
                    datasets=[],
                    edge_class=Edge,
                    test_mode=False,
                    visualize=False,
                    show_viz=False,
                    quality_threshold=quality_threshold,
                    symmetry_threshold=symmetry_threshold,
                    embedding_threshold=embedding_threshold,
                    silent_mode=True
                )
                graph_type_str = 'clustered'
                subclustering = False
            
            # Generate graphs for each split
            for split_name, nodes_to_use in [
                ('train', train_nodes if balance_nodes else train_nodes_full),
                ('val', val_nodes if balance_nodes else val_nodes_full),
                ('test', test_nodes if balance_nodes else test_nodes_full)
            ]:
                print(f"Building graph for {split_name} split...")
                
                # Build graph
                if split_name == 'train':
                    graph = dataloader._build_graph_standard(nodes_to_use, split_name)[0]
                else:
                    graph = HyperGraph(nodes_to_use)
                
                # Save edge list to cache
                if graph:
                    edge_list = graph.get_edge_list()
                    cache_filename = os.path.join(
                        graph_cache_dir,
                        f"ai-face_{split_name}_{graph_type_str}_{'balanced' if balance_nodes else 'full'}_nodes_{len(nodes_to_use)}_q{quality_threshold:.3f}_s{symmetry_threshold:.3f}_e{embedding_threshold:.3f}_graph.pkl"
                    )
                    with open(cache_filename, 'wb') as f:
                        dill.dump(edge_list, f)
                    print(f"Graph cache saved to {cache_filename}")
        
        print("Cache generation completed successfully")
        
    except Exception as e:
        print(f"Error generating cache: {str(e)}")
        import traceback
        traceback.print_exc()

# -----------------------------------------------------------------------------
# Configuration pages + APIs
# -----------------------------------------------------------------------------
@app.route('/configure', methods=['GET', 'POST'])
def configure():
    """Configuration page for test settings."""
    # Provide default attribute_metadata for new configs
    default_attribute_metadata = [
        {'name': 'Ground Truth Gender', 'type': 'categorical', 'possible_values': [0, 1]},
        {'name': 'Ground Truth Race', 'type': 'categorical', 'possible_values': [0, 1, 2, 3]},
        {'name': 'Ground Truth Age', 'type': 'categorical', 'possible_values': [0, 1, 2, 3]},
        {'name': 'blur', 'type': 'continuous'},
        {'name': 'brightness', 'type': 'continuous'},
        {'name': 'contrast', 'type': 'continuous'},
        {'name': 'compression', 'type': 'continuous'},
        {'name': 'symmetry_eye', 'type': 'continuous'},
        {'name': 'symmetry_mouth', 'type': 'continuous'},
        {'name': 'symmetry_nose', 'type': 'continuous'},
        {'name': 'symmetry_overall', 'type': 'continuous'},
        {'name': 'emotion_angry', 'type': 'continuous'},
        {'name': 'emotion_disgust', 'type': 'continuous'},
        {'name': 'emotion_fear', 'type': 'continuous'},
        {'name': 'emotion_happy', 'type': 'continuous'},
        {'name': 'emotion_sad', 'type': 'continuous'},
        {'name': 'emotion_surprise', 'type': 'continuous'},
        {'name': 'emotion_neutral', 'type': 'continuous'},
        {'name': 'face_embedding', 'type': 'continuous'}
    ]
    return render_template(
        'configure.html',
        config=None,
        config_name=None,
        attribute_metadata=default_attribute_metadata
    )

@app.route('/configure/<config_name>')
def edit_config(config_name):
    """Edit existing configuration."""
    config = config_manager.load_configuration(config_name)
    if not config:
        return redirect(url_for('configure'))
    # Try to get attribute_metadata from config, else use default
    attribute_metadata = config.get('attribute_metadata') if config else None
    if not attribute_metadata:
        attribute_metadata = [
            {'name': 'Ground Truth Gender', 'type': 'categorical', 'possible_values': [0, 1]},
            {'name': 'Ground Truth Race', 'type': 'categorical', 'possible_values': [0, 1, 2, 3]},
            {'name': 'Ground Truth Age', 'type': 'categorical', 'possible_values': [0, 1, 2, 3]},
            {'name': 'blur', 'type': 'continuous'},
            {'name': 'brightness', 'type': 'continuous'},
            {'name': 'contrast', 'type': 'continuous'},
            {'name': 'compression', 'type': 'continuous'},
            {'name': 'symmetry_eye', 'type': 'continuous'},
            {'name': 'symmetry_mouth', 'type': 'continuous'},
            {'name': 'symmetry_nose', 'type': 'continuous'},
            {'name': 'symmetry_overall', 'type': 'continuous'},
            {'name': 'emotion_angry', 'type': 'continuous'},
            {'name': 'emotion_disgust', 'type': 'continuous'},
            {'name': 'emotion_fear', 'type': 'continuous'},
            {'name': 'emotion_happy', 'type': 'continuous'},
            {'name': 'emotion_sad', 'type': 'continuous'},
            {'name': 'emotion_surprise', 'type': 'continuous'},
            {'name': 'emotion_neutral', 'type': 'continuous'},
            {'name': 'face_embedding', 'type': 'continuous'}
        ]
    return render_template('configure.html', config=config, config_name=config_name, attribute_metadata=attribute_metadata)

@app.route('/api/configurations', methods=['GET'])
def api_list_configurations():
    """API endpoint to list all configurations."""
    return jsonify(config_manager.list_configurations())

@app.route('/api/configurations', methods=['POST'])
def api_save_configuration():
    """API endpoint to save a configuration.
    
    Saves the complete configuration dictionary exactly as provided. All fields
    in the config dictionary are preserved and will be reused when running tests.
    The full config dictionary is saved without modification or field filtering.
    """
    data = request.get_json()
    name = data.get('name')
    config = data.get('config')
    
    if not name or not config:
        return jsonify({'error': 'Name and config are required'}), 400
    
    # Log that we're saving the full config dictionary
    logger.info(f"Saving configuration '{name}' with {len(config)} fields in config dictionary")
    
    # Save the complete config dictionary - all fields are preserved
    success = config_manager.save_configuration(name, config)
    if success:
        logger.info(f"Successfully saved configuration '{name}' with full config dictionary")
        return jsonify({'message': 'Configuration saved successfully'})
    else:
        logger.error(f"Failed to save configuration '{name}'")
        return jsonify({'error': 'Failed to save configuration'}), 500

@app.route('/api/configurations/<config_name>', methods=['GET'])
def api_get_configuration(config_name):
    """API endpoint to get a specific configuration."""
    config = config_manager.load_configuration(config_name)
    if config:
        return jsonify(config)
    else:
        return jsonify({'error': 'Configuration not found'}), 404

@app.route('/api/configurations/<config_name>', methods=['DELETE'])
def api_delete_configuration(config_name):
    """API endpoint to delete a configuration."""
    success = config_manager.delete_configuration(config_name)
    if success:
        return jsonify({'message': 'Configuration deleted successfully'})
    else:
        return jsonify({'error': 'Failed to delete configuration'}), 500

# -----------------------------------------------------------------------------
# Run queue / run lifecycle APIs
# -----------------------------------------------------------------------------
@app.route('/api/test-runs', methods=['POST'])
def api_start_test_run():
    """API endpoint to start a test run (supports multiple architectures and DQN models).
    
    This endpoint loads the full saved configuration dictionary and reuses it directly
    for test runs. The saved config contains all fields and settings, which are preserved
    exactly as saved. Only architectures and dqn-model are modified per-run to support
    multiple model combinations.
    """
    data = request.get_json()
    config_name = data.get('config_name')
    
    if not config_name:
        return jsonify({'error': 'Configuration name is required'}), 400
    
    # Load the full saved configuration (includes metadata wrapper)
    config_data = config_manager.load_configuration(config_name)
    if not config_data:
        return jsonify({'error': 'Configuration not found'}), 404
    
    # Extract the inner config object - this is the full saved config dictionary
    # with all fields preserved exactly as they were saved
    config = config_data.get('config', config_data)
    
    # Log that we're using the full saved config (for debugging)
    logger.info(f"Starting test run with config '{config_name}': using full saved config dictionary with {len(config)} fields")
    
    # Parse architectures and dqn-model as lists for multi-run support
    # Note: We read these from the saved config, but will override them per-run
    archs = config.get('architectures', None)
    dqn_models = config.get('dqn-model', config.get('dqn_model', None))

    # Support both comma-separated strings and lists
    if archs is None:
        arch_list = [None]
    elif isinstance(archs, str):
        arch_list = [a.strip() for a in archs.split(',') if a.strip()]
    elif isinstance(archs, list):
        arch_list = archs
    else:
        arch_list = [str(archs)]

    if dqn_models is None:
        dqn_list = [None]
    elif isinstance(dqn_models, str):
        dqn_list = [d.strip() for d in dqn_models.split(',') if d.strip()]
    elif isinstance(dqn_models, list):
        dqn_list = dqn_models
    else:
        dqn_list = [str(dqn_models)]

    # If either is missing, treat as single None (for backward compatibility)
    if not arch_list:
        arch_list = [None]
    if not dqn_list:
        dqn_list = [None]

    run_ids = []
    for arch in arch_list:
        for dqn in dqn_list:
            # Create a deep copy of the FULL saved config for each run
            # This preserves ALL fields from the saved configuration
            run_config = copy.deepcopy(config)
            
            # Only modify architectures and dqn-model for this specific run
            # All other fields remain exactly as saved
            if arch is not None:
                run_config['architectures'] = arch
            if dqn is not None:
                run_config['dqn-model'] = dqn
                run_config['dqn_model'] = dqn  # for compatibility with both keys
            
            # Use a descriptive config name for each run
            run_config_name = f"{config_name}__{arch or 'default'}__{dqn or 'default'}"
            run_id = gpu_queue_manager.queue_run(run_config_name, run_config)
            run_ids.append(run_id)
            logger.info(f"Queued run {run_id} with full config (preserving all {len(run_config)} saved fields)")

    if run_ids:
        return jsonify({'run_ids': run_ids, 'message': f'Queued {len(run_ids)} run(s) for all model/DQN combinations.'})
    else:
        return jsonify({'error': 'Failed to queue test run(s)'}), 500

@app.route('/api/test-runs', methods=['GET'])
def api_list_test_runs():
    """API endpoint to list all test runs."""
    return jsonify(gpu_queue_manager.list_runs())

@app.route('/api/test-runs/<run_id>', methods=['GET'])
def api_get_test_run(run_id):
    """API endpoint to get details of a specific test run."""
    run = gpu_queue_manager.get_run(run_id)
    if run:
        return jsonify(run)
    else:
        return jsonify({'error': 'Test run not found'}), 404

@app.route('/api/test-runs/<run_id>/stop', methods=['POST'])
def api_stop_test_run(run_id):
    """API endpoint to stop a running test."""
    success = gpu_queue_manager.stop_run(run_id)
    if success:
        return jsonify({'message': 'Test run stopped successfully'})
    else:
        return jsonify({'error': 'Failed to stop test run'}), 500

@app.route('/api/test-runs/<run_id>/logs')
def api_get_run_logs(run_id):
    """API endpoint to get logs for a test run."""
    logs = gpu_queue_manager.get_run_logs(run_id)
    return jsonify({'logs': logs})

# -----------------------------------------------------------------------------
# Run pages (HTML)
# -----------------------------------------------------------------------------
@app.route('/runs')
def runs():
    """Test runs management page."""
    runs = gpu_queue_manager.list_runs()
    # Sort runs by created time (most recent first)
    runs.sort(key=lambda x: x.get('created', ''), reverse=True)
    return render_template('runs.html', runs=runs)

@app.route('/runs/<run_id>')
def view_run(run_id):
    """View details of a specific test run."""
    run = gpu_queue_manager.get_run(run_id)
    if not run:
        return redirect(url_for('runs'))
    
    logs = gpu_queue_manager.get_run_logs(run_id)
    return render_template('run_details.html', run=run, logs=logs)

@app.route('/results')
def results():
    """Results comparison page."""
    completed_runs = [run for run in gpu_queue_manager.list_runs() if run.get('status') == 'completed']
    
    # Sort runs by end_time (most recent first)
    completed_runs.sort(key=lambda x: x.get('end_time', ''), reverse=True)
    
    # Debug logging (commented out to avoid broken pipe errors)
    # print(f"DEBUG: Found {len(completed_runs)} completed runs")
    # for i, run in enumerate(completed_runs):
    #     print(f"DEBUG: Run {i}: {run.get('run_id')} - status: {run.get('status')}")
    #     print(f"DEBUG: Run {i}: has 'results': {'results' in run}")
    #     if 'results' in run:
    #         print(f"DEBUG: Run {i}: results keys: {list(run['results'].keys())}")
    #         print(f"DEBUG: Run {i}: final_accuracy: {run['results'].get('final_accuracy')}")
    #     print(f"DEBUG: Run {i}: has 'accuracy': {'accuracy' in run}")
    #     if 'accuracy' in run:
    #         print(f"DEBUG: Run {i}: accuracy value: {run['accuracy']}")
    
    # Flatten results data for template compatibility
    for run in completed_runs:
        if 'results' in run and run['results']:
            # Flatten accuracy
            if 'final_accuracy' in run['results']:
                run['accuracy'] = run['results']['final_accuracy']
            # Flatten loss and duration
            if 'loss' in run['results']:
                run['loss'] = run['results']['loss']
            if 'duration' in run['results']:
                run['duration'] = run['results']['duration']
            # Flatten architecture and traversal
            if 'architecture' in run['results']:
                run['architecture'] = run['results']['architecture']
            if 'traversal_type' in run['results']:
                run['traversal_type'] = run['results']['traversal_type']
        
        # Add architecture and traversal from configuration (fallback)
        if 'config' in run and run['config']:
            config = run['config']
            if 'architecture' not in run and 'architecture' in config:
                run['architecture'] = config['architecture']
            if 'traversal_type' not in run and 'traversal_type' in config:
                run['traversal_type'] = config['traversal_type']
            # Add reduction/restoration strategy info
            if 'reduction_strategy' in config:
                run['reduction_strategy'] = config['reduction_strategy']
            if 'reduction_percentage' in config:
                run['reduction_percentage'] = config['reduction_percentage']
            if 'restoration_strategy' in config:
                run['restoration_strategy'] = config['restoration_strategy']
            if 'restoration_percentage' in config:
                run['restoration_percentage'] = config['restoration_percentage']
    
    return render_template('results.html', runs=completed_runs)

# -----------------------------------------------------------------------------
# Results comparison API
# -----------------------------------------------------------------------------
@app.route('/api/results/compare', methods=['POST'])
def api_compare_results():
    """API endpoint to compare results from multiple runs."""
    data = request.get_json()
    run_ids = data.get('run_ids', [])
    
    if not run_ids:
        return jsonify({'error': 'At least one run ID is required'}), 400
    
    # For now, return basic comparison info since the old test_runner.compare_runs method
    # might not be available in the new GPU queue manager
    comparison = {
        'run_ids': run_ids,
        'runs': []
    }
    
    for run_id in run_ids:
        run = gpu_queue_manager.get_run(run_id)
        if run:
            # Flatten results data for easier access in comparison
            if 'results' in run and run['results']:
                # Copy bias metrics to top level for comparison
                if 'race_gender_bias' in run['results']:
                    run['race_gender_bias'] = run['results']['race_gender_bias']
                if 'gender_bias' in run['results']:
                    run['gender_bias'] = run['results']['gender_bias']
                if 'race_bias' in run['results']:
                    run['race_bias'] = run['results']['race_bias']
                if 'average_attribute_bias' in run['results']:
                    run['average_attribute_bias'] = run['results']['average_attribute_bias']
                # Also flatten accuracy, loss, and duration for consistency
                if 'final_accuracy' in run['results']:
                    run['accuracy'] = run['results']['final_accuracy']
                if 'loss' in run['results']:
                    run['loss'] = run['results']['loss']
                if 'duration' in run['results']:
                    run['duration'] = run['results']['duration']
                # Also flatten architecture and traversal from results
                if 'architecture' in run['results']:
                    run['architecture'] = run['results']['architecture']
                if 'traversal_type' in run['results']:
                    run['traversal_type'] = run['results']['traversal_type']
            
            # Add architecture and traversal from configuration (fallback)
            if 'config' in run and run['config']:
                config = run['config']
                if 'architecture' not in run and 'architecture' in config:
                    run['architecture'] = config['architecture']
                if 'traversal_type' not in run and 'traversal_type' in config:
                    run['traversal_type'] = config['traversal_type']
            
            comparison['runs'].append(run)
    
    return jsonify(comparison)

# -----------------------------------------------------------------------------
# Templates page + API (config templates)
# -----------------------------------------------------------------------------
@app.route('/templates')
def templates():
    """Configuration templates page."""
    templates = config_manager.list_templates()
    return render_template('templates.html', templates=templates)

@app.route('/api/templates/<template_name>')
def api_get_template(template_name):
    """API endpoint to get a configuration template."""
    template = config_manager.get_template(template_name)
    if template:
        return jsonify(template)
    else:
        return jsonify({'error': 'Template not found'}), 404

# -----------------------------------------------------------------------------
# Shutdown + debugging helpers
# -----------------------------------------------------------------------------
@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    """API endpoint to shutdown the server."""
    def shutdown_server():
        import time
        import signal
        import os
        # Give the response time to be sent
        time.sleep(1)
        # Shutdown GPU queue manager
        gpu_queue_manager.shutdown()
        # Send SIGTERM to self
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Start shutdown in a separate thread
    shutdown_thread = threading.Thread(target=shutdown_server)
    shutdown_thread.daemon = True
    shutdown_thread.start()
    
    return jsonify({'message': 'Server is shutting down...'})

@app.route('/debug/logs')
def view_logs():
    """Debug endpoint to view application logs."""
    try:
        with open(log_file, 'r') as f:
            logs = f.readlines()
        return jsonify({
            'log_file': log_file,
            'log_count': len(logs),
            'logs': logs[-100:]  # Return last 100 lines
        })
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to read logs',
            'details': str(e)
        }), 500

# Add a route to check if the API endpoint is accessible
@app.route('/api/cache/test')
def test_cache_api():
    """Test endpoint to verify API accessibility."""
    logger.info("Testing cache API endpoint")
    return jsonify({
        'status': 'ok',
        'message': 'Cache API is accessible',
        'timestamp': datetime.now().isoformat()
    })

# -----------------------------------------------------------------------------
# GPU queue status / admin APIs
# -----------------------------------------------------------------------------
@app.route('/api/gpu/status')
def api_get_gpu_status():
    """API endpoint to get GPU status information."""
    try:
        gpu_info = gpu_queue_manager.get_gpu_info()
        queue_status = gpu_queue_manager.get_queue_status()
        
        return jsonify({
            'gpus': gpu_info,
            'queue_status': queue_status,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to get GPU status',
            'details': str(e)
        }), 500

@app.route('/api/gpu/queue')
def api_get_queue_status():
    """API endpoint to get queue status information."""
    try:
        queue_status = gpu_queue_manager.get_queue_status()
        return jsonify(queue_status)
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to get queue status',
            'details': str(e)
        }), 500

@app.route('/api/gpu/check-orphaned', methods=['POST'])
def api_check_orphaned_runs():
    """API endpoint to manually check for orphaned queued runs."""
    try:
        orphaned_runs = gpu_queue_manager.check_orphaned_queued_runs()
        return jsonify({
            'success': True,
            'orphaned_runs': orphaned_runs,
            'message': f'Found and marked {len(orphaned_runs)} orphaned runs as failed'
        })
    except Exception as e:
        logger.error(f"Error checking for orphaned runs: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to check for orphaned runs',
            'details': str(e)
        }), 500

@app.route('/api/gpu/clear-queue', methods=['POST'])
def api_clear_queue():
    """API endpoint to clear the queue and stop all running runs."""
    try:
        result = gpu_queue_manager.clear_queue()
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Cleared queue: stopped {result.get("total_stopped", 0)} running run(s) and cancelled {result.get("total_cleared", 0)} queued run(s)',
                'stopped_runs': result.get('stopped_runs', []),
                'cleared_runs': result.get('cleared_runs', [])
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to clear queue'),
                'stopped_runs': result.get('stopped_runs', []),
                'cleared_runs': result.get('cleared_runs', [])
            }), 500
    except Exception as e:
        logger.error(f"Error clearing queue: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to clear queue',
            'details': str(e)
        }), 500

# -----------------------------------------------------------------------------
# Run repair / maintenance APIs
# -----------------------------------------------------------------------------
@app.route('/api/results/extract', methods=['POST'])
def api_extract_results():
    """API endpoint to extract results from completed runs that don't have results."""
    try:
        # Get all completed runs
        all_runs = gpu_queue_manager.list_runs()
        completed_runs = [run for run in all_runs if run.get('status') == 'completed']
        
        extracted_count = 0
        for run in completed_runs:
            run_id = run.get('run_id')
            # Always try to extract results (for bias metrics even if accuracy exists)
            gpu_queue_manager._extract_results(run_id)
            extracted_count += 1
        
        return jsonify({
            'success': True,
            'extracted_count': extracted_count,
            'total_completed': len(completed_runs),
            'message': f'Extracted results for {extracted_count} runs'
        })
    except Exception as e:
        logger.error(f"Error extracting results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/runs/fix-status', methods=['POST'])
def api_fix_run_status():
    """API endpoint to fix run status based on log analysis."""
    try:
        # Get all runs
        all_runs = gpu_queue_manager.list_runs()
        
        fixed_count = 0
        status_changes = []
        
        for run in all_runs:
            run_id = run.get('run_id')
            current_status = run.get('status')
            
            # Analyze log to determine correct status
            correct_status = gpu_queue_manager._analyze_run_status_from_log(run_id)
            
            if correct_status and correct_status != current_status:
                # Update the run status
                metadata = gpu_queue_manager._load_run_metadata(run_id)
                if metadata:
                    old_status = metadata.get('status')
                    metadata['status'] = correct_status
                    metadata['last_updated'] = datetime.now().isoformat()
                    
                    # Add error information if changing to failed
                    if correct_status == 'failed':
                        metadata['error'] = f'Status corrected from {old_status} to failed based on log analysis'
                    
                    gpu_queue_manager._save_run_metadata(run_id, metadata)
                    
                    # Update in-memory metadata if present
                    if run_id in gpu_queue_manager.run_metadata:
                        gpu_queue_manager.run_metadata[run_id] = metadata
                    
                    fixed_count += 1
                    status_changes.append({
                        'run_id': run_id,
                        'old_status': old_status,
                        'new_status': correct_status
                    })
                    logger.info(f"Fixed status for {run_id}: {old_status} -> {correct_status}")
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'total_runs': len(all_runs),
            'status_changes': status_changes,
            'message': f'Fixed status for {fixed_count} runs'
        })
    except Exception as e:
        logger.error(f"Error fixing run status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Development entrypoint: run the Flask server directly.
    #
    # In production you would typically run Flask under a process manager
    # (e.g., gunicorn/uwsgi) and avoid debug=True.
    # Create necessary directories
    os.makedirs('web_ui/configs', exist_ok=True)
    os.makedirs('web_ui/runs', exist_ok=True)
    os.makedirs('web_ui/templates', exist_ok=True)
    os.makedirs('web_ui/static/css', exist_ok=True)
    os.makedirs('web_ui/static/js', exist_ok=True)
    
    port = int(os.environ.get('PORT', '5000'))

    print("Starting HyperGraph Test Configuration Web UI...")
    print(f"Access the interface at: http://localhost:{port}")
    print(f"For SSH tunneling, use: ssh -L {port}:localhost:{port} user@server")
    
    app.run(host='0.0.0.0', port=port, debug=True) 
