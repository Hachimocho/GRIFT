#!/usr/bin/env python
"""
Script to debug embedding issues by adding instrumentation at different points
in the data loading and node creation pipeline.
"""

import os
import pickle
import numpy as np
import logging
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EmbeddingDebugger')

def analyze_node_embeddings(nodes, split_name="unknown"):
    """
    Analyze embeddings in a list of nodes
    
    Args:
        nodes: List of nodes to analyze
        split_name: Name of the split for logging
    """
    total_nodes = len(nodes)
    nodes_with_embedding = 0
    valid_embeddings = 0
    zero_embeddings = 0
    
    # Sample some nodes for detailed analysis
    sample_limit = min(5, total_nodes)
    sampled_nodes = []
    
    for i, node in enumerate(nodes):
        # Get node attributes
        attributes = getattr(node, 'attributes', {})
        
        # Check for embedding
        if 'face_embedding' in attributes:
            nodes_with_embedding += 1
            
            # Check if embedding is valid
            embedding = attributes['face_embedding']
            if isinstance(embedding, np.ndarray):
                if not np.all(np.isclose(embedding, 0)):
                    valid_embeddings += 1
                else:
                    zero_embeddings += 1
                    
            # Sample some nodes
            if len(sampled_nodes) < sample_limit:
                sampled_nodes.append((i, node, attributes))
    
    # Print summary
    logger.info(f"\nAnalyzing {total_nodes} nodes from split: {split_name}")
    logger.info(f"Nodes with embedding attribute: {nodes_with_embedding}/{total_nodes} ({nodes_with_embedding/total_nodes*100:.2f}%)")
    
    if nodes_with_embedding > 0:
        logger.info(f"Valid non-zero embeddings: {valid_embeddings}/{nodes_with_embedding} ({valid_embeddings/nodes_with_embedding*100:.2f}%)")
        logger.info(f"Zero embeddings: {zero_embeddings}/{nodes_with_embedding} ({zero_embeddings/nodes_with_embedding*100:.2f}%)")
    
    # Print sample details
    if sampled_nodes:
        logger.info("\nSample nodes:")
        for i, node, attributes in sampled_nodes:
            logger.info(f"Node {i}:")
            
            # Print all attributes (excluding large arrays)
            attr_str = "{"
            for k, v in attributes.items():
                if k == 'face_embedding':
                    if isinstance(v, np.ndarray):
                        attr_str += f"'face_embedding': array(shape={v.shape}, "
                        attr_str += f"min={np.min(v):.4f}, max={np.max(v):.4f}, "
                        attr_str += f"mean={np.mean(v):.4f}, "
                        attr_str += f"zeros={np.sum(np.isclose(v, 0))}/{len(v)}), "
                    else:
                        attr_str += f"'face_embedding': {type(v)}, "
                else:
                    attr_str += f"'{k}': {v}, "
            attr_str = attr_str.rstrip(", ") + "}"
            logger.info(f"  Attributes: {attr_str}")

def analyze_cached_nodes():
    """
    Check the embeddings in the cached nodes file
    """
    cache_file = "cached_nodes.pkl"
    
    if not os.path.exists(cache_file):
        logger.error(f"Cache file not found: {cache_file}")
        return
    
    try:
        logger.info(f"Loading cached nodes from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        if not isinstance(cached_data, dict):
            logger.error(f"Unexpected cache format: {type(cached_data)}")
            return
        
        # Skip metadata
        data_keys = [k for k in cached_data.keys() if k != 'metadata']
        logger.info(f"Found data types: {data_keys}")
        
        for data_type in data_keys:
            logger.info(f"\n=== Analyzing {data_type} cache ===\n")
            
            split_data = cached_data[data_type]
            if not isinstance(split_data, dict):
                logger.error(f"Unexpected format for {data_type}: {type(split_data)}")
                continue
            
            for split_name, nodes in split_data.items():
                analyze_node_embeddings(nodes, f"{data_type}/{split_name}")
    
    except Exception as e:
        logger.error(f"Error analyzing cached nodes: {e}")

def instrument_dataset_functions():
    """
    Instrument key functions in the AIFaceDataset class to track embeddings
    """
    from datasets.AIFaceDataset import AIFaceDataset
    
    # Store original function
    original_parse_func = AIFaceDataset._parse_quality_attributes
    original_create_node = AIFaceDataset.create_node
    
    # Patch the function
    def instrumented_parse_func(self, row_data):
        filename, attrs = original_parse_func(self, row_data)
        
        # Check for face embedding
        if 'face_embedding' in attrs:
            logger.info(f"Embedding parsed for file: {filename}")
            embedding = attrs['face_embedding']
            if isinstance(embedding, np.ndarray):
                logger.info(f"  Shape: {embedding.shape}")
                logger.info(f"  Values: min={np.min(embedding):.4f}, max={np.max(embedding):.4f}, mean={np.mean(embedding):.4f}")
                logger.info(f"  Zeros: {np.sum(np.isclose(embedding, 0))}/{len(embedding)}")
        else:
            logger.info(f"No embedding parsed for file: {filename}")
        
        return filename, attrs
    
    def instrumented_create_node(self, args):
        node = original_create_node(self, args)
        
        # Check node attributes
        if 'face_embedding' in node.attributes:
            logger.info(f"Node created with embedding: {node.attributes.get('subset')}")
            embedding = node.attributes['face_embedding']
            if isinstance(embedding, np.ndarray):
                logger.info(f"  Shape: {embedding.shape}")
                logger.info(f"  Values: min={np.min(embedding):.4f}, max={np.max(embedding):.4f}, mean={np.mean(embedding):.4f}")
                logger.info(f"  Zeros: {np.sum(np.isclose(embedding, 0))}/{len(embedding)}")
        else:
            logger.info(f"Node created without embedding: {node.attributes.get('subset')}")
        
        return node
    
    # Apply patches
    AIFaceDataset._parse_quality_attributes = instrumented_parse_func
    AIFaceDataset.create_node = instrumented_create_node
    
    logger.info("AIFaceDataset functions instrumented for embedding tracking")

def main():
    parser = argparse.ArgumentParser(description="Debug embeddings in nodes")
    parser.add_argument("--analyze-cache", action="store_true", help="Analyze cached nodes")
    parser.add_argument("--instrument", action="store_true", help="Instrument dataset functions")
    
    args = parser.parse_args()
    
    if args.analyze_cache:
        analyze_cached_nodes()
    
    if args.instrument:
        instrument_dataset_functions()
        logger.info("Run your normal script now with instrumented functions")

if __name__ == "__main__":
    main()
