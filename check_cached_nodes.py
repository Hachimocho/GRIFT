#!/usr/bin/env python3
"""
Check the embeddings in the cached nodes file
"""

import pickle
import numpy as np
import argparse
import os

def check_cached_nodes(cache_file):
    """
    Analyze the embeddings in cached nodes
    """
    print(f"Loading cached nodes from: {cache_file}")
    
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading cached nodes: {e}")
        return
    
    print(f"Cache data type: {type(cached_data)}")
    
    # Based on test_hierarchical.py's cache_nodes() function, the structure is:
    # cached_data = {
    #     'full': { 'train': [nodes], 'val': [nodes], 'test': [nodes] },
    #     'subset': { 'train': [nodes], 'val': [nodes], 'test': [nodes] },
    #     'metadata': { ... }
    # }
    
    if not isinstance(cached_data, dict):
        print(f"Unexpected cache format: {type(cached_data)}")
        return
    
    # Skip metadata
    data_keys = [k for k in cached_data.keys() if k != 'metadata']
    print(f"Found data types: {data_keys}")
    
    for data_type in data_keys:
        print(f"\n=== Analyzing {data_type} cache ===\n")
        
        split_data = cached_data[data_type]
        if not isinstance(split_data, dict):
            print(f"Unexpected format for {data_type}: {type(split_data)}")
            continue
        
        for split_name, nodes in split_data.items():
            check_nodes_list(nodes, f"{data_type}/{split_name}")

def check_nodes_list(nodes, split_name):
    """Analyze a list of nodes"""
    print(f"\nAnalyzing {len(nodes)} nodes from split: {split_name}")
    
    # Count nodes with embeddings
    nodes_with_embedding = 0
    valid_embeddings = 0
    zero_embeddings = 0
    embedding_shapes = {}
    embedding_magnitudes = []
    
    # Sample some embeddings
    sample_count = min(5, len(nodes))
    samples = []
    
    for i, node in enumerate(nodes):
        # Get node attributes
        attributes = getattr(node, 'attributes', {})
        if i < 5:
            print(node)
            print(f"Node {i}: {attributes}")
        # Check for embedding
        if 'face_embedding' in attributes:
            nodes_with_embedding += 1
            embedding = attributes['face_embedding']
            
            # Check if embedding is valid
            if isinstance(embedding, np.ndarray):
                shape = embedding.shape
                embedding_shapes[str(shape)] = embedding_shapes.get(str(shape), 0) + 1
                
                # Check if all zeros
                if np.all(np.isclose(embedding, 0)):
                    zero_embeddings += 1
                else:
                    valid_embeddings += 1
                
                # Calculate magnitude
                magnitude = np.linalg.norm(embedding)
                embedding_magnitudes.append(magnitude)
                
                # Save sample
                if i < sample_count:
                    samples.append({
                        'shape': shape,
                        'magnitude': magnitude,
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding)),
                        'mean': float(np.mean(embedding)),
                        'zeros': int(np.sum(np.isclose(embedding, 0))),
                        'is_all_zeros': bool(np.all(np.isclose(embedding, 0)))
                    })
    
    # Print results
    print(f"Nodes with embedding attribute: {nodes_with_embedding}/{len(nodes)} ({nodes_with_embedding/len(nodes)*100:.2f}%)")
    
    if nodes_with_embedding > 0:
        print(f"Valid embeddings: {valid_embeddings}/{nodes_with_embedding} ({valid_embeddings/nodes_with_embedding*100:.2f}%)")
        print(f"Zero embeddings: {zero_embeddings}/{nodes_with_embedding} ({zero_embeddings/nodes_with_embedding*100:.2f}%)")
        print(f"Embedding shapes: {embedding_shapes}")
        
        if embedding_magnitudes:
            print(f"Magnitude stats:")
            print(f"  Min: {np.min(embedding_magnitudes):.4f}")
            print(f"  Max: {np.max(embedding_magnitudes):.4f}")
            print(f"  Mean: {np.mean(embedding_magnitudes):.4f}")
            print(f"  Median: {np.median(embedding_magnitudes):.4f}")
    
    # Print samples
    if samples:
        print("\nSample embeddings:")
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}:")
            print(f"    Shape: {sample['shape']}")
            print(f"    Magnitude: {sample['magnitude']:.4f}")
            print(f"    Range: [{sample['min']:.4f}, {sample['max']:.4f}]")
            print(f"    Mean: {sample['mean']:.4f}")
            print(f"    Zeros: {sample['zeros']}/{np.prod(sample['shape'])} ({sample['zeros']/np.prod(sample['shape'])*100:.2f}%)")
            print(f"    All zeros: {sample['is_all_zeros']}")

def main():
    parser = argparse.ArgumentParser(description="Check embeddings in cached nodes")
    parser.add_argument("--cache-file", type=str, default="cached_nodes.pkl", help="Path to the cached nodes file")
    
    args = parser.parse_args()
    
    # Resolve path
    cache_path = os.path.abspath(args.cache_file)
    
    if not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        return
    
    check_cached_nodes(cache_path)

if __name__ == "__main__":
    main()
