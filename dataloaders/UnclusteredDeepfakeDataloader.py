import random
from itertools import combinations
from tqdm.auto import tqdm
import time
from multiprocessing import Pool, cpu_count, Queue, Manager
import numpy as np
from dataloaders.Dataloader import Dataloader
from graphs.HyperGraph import HyperGraph
from collections import defaultdict
from utils.visualize import visualize_graph
import networkx as nx
import pandas as pd
import os

class UnclusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = {
        "use_lsh": True,  # Whether to use Locality-Sensitive Hashing for faster matching
        "test_mode": True,  # If True, only loads first 100 nodes for testing
        "lsh_bands": 50,  # Number of bands for LSH
        "lsh_band_size": 2,  # Size of each LSH band
        "visualize": False,  # Whether to create and save a visualization
        "show_viz": False,  # Whether to display the visualization
    }

    def _create_attribute_matrix(self, nodes):
        """Convert node attributes to a binary feature matrix and extract embeddings"""
        # Create mappings for each attribute type
        attribute_categories = {
            'race': set(),
            'gender': set(),
            'age': set(),
            'emotion': set(),
            'quality_metrics': set(),
            'symmetry': set()
        }
        
        # Collect all unique attributes by category
        for node in nodes:
            for attr, value in node.attributes.items():
                if attr.startswith('race_'):
                    attribute_categories['race'].add(attr)
                elif attr.startswith('gender_'):
                    attribute_categories['gender'].add(attr)
                elif attr.startswith('age_'):
                    attribute_categories['age'].add(attr)
                elif attr.startswith('emotion_'):
                    attribute_categories['emotion'].add(attr)
                elif attr in ['blur', 'brightness', 'contrast', 'compression']:
                    attribute_categories['quality_metrics'].add(attr)
                elif attr.startswith('symmetry_'):
                    attribute_categories['symmetry'].add(attr)
        
        # Create index mappings for each category
        category_indices = {}
        current_idx = 0
        for category, attrs in attribute_categories.items():
            category_indices[category] = (current_idx, current_idx + len(attrs))
            current_idx += len(attrs)
        
        # Create mapping of all attributes to indices
        attr_to_idx = {}
        idx = 0
        for category, attrs in attribute_categories.items():
            for attr in sorted(attrs):
                attr_to_idx[attr] = idx
                idx += 1
        
        # Create binary feature matrix and collect embeddings
        feature_matrix = np.zeros((len(nodes), len(attr_to_idx)), dtype=np.float32)
        embeddings = np.zeros((len(nodes), 512), dtype=np.float32)  # FaceNet embeddings are 512-dimensional
        
        for i, node in enumerate(nodes):
            # Handle regular attributes
            for attr, value in node.attributes.items():
                if attr in attr_to_idx:
                    # For binary attributes
                    if isinstance(value, bool):
                        feature_matrix[i, attr_to_idx[attr]] = float(value)
                    # For numerical attributes (quality metrics)
                    elif isinstance(value, (int, float)):
                        feature_matrix[i, attr_to_idx[attr]] = value
                    # For categorical attributes (one-hot encoded)
                    else:
                        feature_matrix[i, attr_to_idx[attr]] = 1.0
            
            # Handle face embedding
            if 'face_embedding' in node.attributes:
                embeddings[i] = node.attributes['face_embedding']
                
        return feature_matrix, attr_to_idx, category_indices, embeddings

    def _compute_similarity(self, features1, features2, embeddings1, embeddings2, start_idx, end_idx, category):
        """Compute similarity between feature vectors for a specific attribute range"""
        # For embeddings comparison
        if category == 'embeddings':
            # Compute cosine similarity between embeddings
            norm1 = np.linalg.norm(embeddings1)
            norm2 = np.linalg.norm(embeddings2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(embeddings1, embeddings2) / (norm1 * norm2)
        
        # Extract relevant features for other categories
        f1 = features1[start_idx:end_idx]
        f2 = features2[start_idx:end_idx]
        
        # For quality metrics (numerical features)
        if category == 'quality_metrics':
            # Define acceptable ranges for each metric
            metric_ranges = {
                'blur': 50,          # Allow difference of 50 in blur score
                'brightness': 50,     # Allow difference of 50 in brightness
                'contrast': 50,       # Allow difference of 50 in contrast
                'compression': 20     # Allow difference of 20 in compression score
            }
            
            # Count how many metrics are within acceptable range
            matches = 0
            total = 0
            for i, (v1, v2) in enumerate(zip(f1, f2)):
                if v1 != 0 and v2 != 0:  # Only compare if both values are present
                    total += 1
                    diff = abs(v1 - v2)
                    threshold = list(metric_ranges.values())[i]
                    if diff <= threshold:
                        matches += 1
            
            # Return match ratio for metrics that were present
            return matches / total if total > 0 else 0
            
        # For symmetry metrics
        elif category == 'symmetry':
            # Define acceptable ranges for each symmetry metric
            symmetry_ranges = {
                'symmetry_eye': 0.3,      # Allow 0.3 difference in eye symmetry
                'symmetry_mouth': 0.3,     # Allow 0.3 difference in mouth symmetry
                'symmetry_nose': 0.3,      # Allow 0.3 difference in nose symmetry
                'symmetry_overall': 0.3    # Allow 0.3 difference in overall symmetry
            }
            
            # Count how many symmetry metrics are within acceptable range
            matches = 0
            total = 0
            for i, (v1, v2) in enumerate(zip(f1, f2)):
                if v1 != 0 and v2 != 0:  # Only compare if both values are present
                    total += 1
                    diff = abs(v1 - v2)
                    threshold = list(symmetry_ranges.values())[i]
                    if diff <= threshold:
                        matches += 1
            
            # Return match ratio for metrics that were present
            return matches / total if total > 0 else 0
        
        # For categorical features (exact matching)
        else:
            # Check if features are exactly equal
            return np.array_equal(f1, f2)

    def _hierarchical_match(self, feature_matrix, embeddings, category_indices, threshold=0.8, embedding_threshold=0.7):
        """Perform hierarchical matching of nodes based on attributes"""
        n_nodes = feature_matrix.shape[0]
        edges = []
        
        # Define the attribute matching order - add symmetry as a separate category
        matching_order = ['race', 'gender', 'age', 'emotion', 'embeddings', 'quality_metrics', 'symmetry']
        
        # Initialize with all possible pairs
        valid_pairs = set((i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes))
        
        # Iteratively filter pairs based on each attribute
        for category in matching_order:
            if category == 'embeddings':
                new_valid_pairs = set()
                for i, j in valid_pairs:
                    similarity = self._compute_similarity(
                        None, None,  # No regular features needed for embeddings
                        embeddings[i], 
                        embeddings[j],
                        None, None,  # No indices needed for embeddings
                        category
                    )
                    if similarity >= embedding_threshold:
                        new_valid_pairs.add((i, j))
            else:
                start_idx, end_idx = category_indices[category]
                new_valid_pairs = set()
                
                for i, j in valid_pairs:
                    similarity = self._compute_similarity(
                        feature_matrix[i], 
                        feature_matrix[j],
                        None, None,  # No embeddings needed for regular features
                        start_idx,
                        end_idx,
                        category
                    )
                    
                    # For categorical features, require exact match
                    # For quality metrics and symmetry, use threshold
                    if category in ['quality_metrics', 'symmetry']:
                        if similarity >= threshold:
                            new_valid_pairs.add((i, j))
                    else:
                        if similarity:  # True means exact match
                            new_valid_pairs.add((i, j))
            
            valid_pairs = new_valid_pairs
            if not valid_pairs:
                break
        
        # Convert remaining valid pairs to edges
        edges.extend(valid_pairs)
        return edges

    def process_node_batch(self, args):
        start_idx, end_idx, node_list, feature_matrix, embeddings, category_indices, chunk_id = args
        edges = []
        matches = set()
        
        # Get nodes for this chunk
        chunk_features = feature_matrix[start_idx:end_idx]
        chunk_embeddings = embeddings[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        # Perform hierarchical matching
        chunk_edges = self._hierarchical_match(
            chunk_features,
            chunk_embeddings,
            category_indices,
            threshold=0.8,
            embedding_threshold=0.7
        )
        
        # Adjust indices to global space
        for i, j in chunk_edges:
            global_i = start_idx + i
            global_j = start_idx + j
            edges.append((global_i, global_j))
            matches.add(global_i)
            matches.add(global_j)
        
        return edges, matches

    def load(self):
        start = time.time()
        node_list = []
        
        # Create NetworkX graph alongside main graph
        nx_graph = nx.Graph()
        
        # Load datasets
        print("Loading nodes from datasets...")
        for dataset in self.datasets:
            # Each dataset's load method should handle attribute loading internally
            dataset_nodes = dataset.load()
            print(f"Loaded {len(dataset_nodes)} nodes from {dataset.__class__.__name__}")
            node_list.extend(dataset_nodes)
        
        # Split nodes based on their subset attribute 
        train_nodes = []
        val_nodes = []
        test_nodes = []
        
        # Organize nodes into their respective splits
        for node in node_list:
            split = node.split  # Get the split directly from the node
            
            if split == 'train':
                train_nodes.append(node)
            elif split == 'val':
                val_nodes.append(node)
            elif split == 'test':
                test_nodes.append(node)
            else:
                print(f"Warning: Node has unknown split: {split}, defaulting to train")
                train_nodes.append(node)
        
        # Optionally limit nodes for testing while preserving split distribution
        if self.hyperparameters["test_mode"]:
            print("Running in test mode - sampling 10000 nodes while preserving split distribution")
            total_nodes = len(train_nodes) + len(val_nodes) + len(test_nodes)
            train_ratio = len(train_nodes) / total_nodes
            val_ratio = len(val_nodes) / total_nodes
            test_ratio = len(test_nodes) / total_nodes
            
            target_total = 10000
            train_size = int(target_total * train_ratio)
            val_size = int(target_total * val_ratio)
            test_size = target_total - train_size - val_size  # Ensure we get exactly 10000
            
            # Randomly sample from each split
            if train_nodes:
                train_nodes = random.sample(train_nodes, min(train_size, len(train_nodes)))
            if val_nodes:
                val_nodes = random.sample(val_nodes, min(val_size, len(val_nodes)))
            if test_nodes:
                test_nodes = random.sample(test_nodes, min(test_size, len(test_nodes)))
        
        # Combine all nodes in the correct order
        node_list = train_nodes + val_nodes + test_nodes
        n_nodes = len(node_list)
        print(f"# of nodes: {n_nodes}")
        
        # Add all nodes to NetworkX graph
        nx_graph.add_nodes_from(range(n_nodes))
        
        # Create feature matrix for fast comparison
        print("Creating feature matrix...")
        feature_matrix, attr_to_idx, category_indices, embeddings = self._create_attribute_matrix(node_list)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Embeddings matrix shape: {embeddings.shape}")
        
        # Automatically disable LSH for low-dimensional data
        if feature_matrix.shape[1] < 10 and self.hyperparameters["use_lsh"]:
            print("Automatically disabled LSH due to low feature dimension")
            self.hyperparameters["use_lsh"] = False
        
        # Split work into chunks for parallel processing
        num_processes = min(8, cpu_count())
        chunk_size = max(1000, n_nodes // (num_processes * 4))
        chunks = [(i, min(i + chunk_size, n_nodes), node_list, feature_matrix, embeddings, category_indices, idx) 
                 for idx, i in enumerate(range(0, n_nodes, chunk_size))]
        
        print(f"\nProcessing {len(chunks)} chunks using {num_processes} processes")
        print(f"Chunk size: {chunk_size} nodes")
        print(f"LSH enabled: {self.hyperparameters['use_lsh']}")
        
        # Process chunks in parallel
        all_edges = []
        all_matches = set()
        
        with tqdm(total=len(chunks), desc="Building graph") as pbar:
            with Pool(num_processes) as pool:
                for edges, matches in pool.imap(self.process_node_batch, chunks):
                    all_edges.extend(edges)
                    all_matches.update(matches)
                    pbar.update(1)
        
        print(f"\nFound {len(all_edges)} matching pairs")
        
        # Create edges and add to nodes
        print("Creating edges...")
        nx_edges = set()  # Track edges for NetworkX graph
        for i, j in tqdm(all_edges, desc="Adding edges to nodes"):
            edge = self.edge_class(node_list[i], node_list[j], None, 1)
            node_list[i].add_edge(edge)
            node_list[j].add_edge(edge)
            nx_edges.add(tuple(sorted([i, j])))  # Add to NetworkX edges
        
        # Add all edges to NetworkX graph at once (much faster than one by one)
        nx_graph.add_edges_from(nx_edges)
        
        # Handle disconnected nodes
        unmatched_nodes = set(range(n_nodes)) - all_matches
        if unmatched_nodes:
            print(f"Connecting {len(unmatched_nodes)} disconnected nodes...")
            matched_nodes = list(all_matches)
            new_nx_edges = set()
            for i in tqdm(unmatched_nodes, desc="Fixing disconnected nodes"):
                j = random.choice(matched_nodes)
                edge = self.edge_class(node_list[i], node_list[j], None, 1)
                node_list[i].add_edge(edge)
                node_list[j].add_edge(edge)
                new_nx_edges.add(tuple(sorted([i, j])))
            
            # Add new edges to NetworkX graph
            nx_graph.add_edges_from(new_nx_edges)
        
        # Create separate graphs for train/val/test based on node subsets
        train_nodes = []
        val_nodes = []
        test_nodes = []
        
        # Split nodes based on their subset attribute
        for node in node_list:
            split = node.split  # Get the split directly from the node
            if split == 'train':
                train_nodes.append(node)
            elif split == 'val':
                val_nodes.append(node)
            elif split == 'test':
                test_nodes.append(node)
            else:
                print(f"Warning: Node has unknown split: {split}, defaulting to train")
                train_nodes.append(node)
        
        print(f"\nNode distribution across splits:")
        print(f"Train: {len(train_nodes)} nodes")
        print(f"Val: {len(val_nodes)} nodes")
        print(f"Test: {len(test_nodes)} nodes")
        
        # Create separate graphs
        train_graph = HyperGraph(train_nodes)
        val_graph = HyperGraph(val_nodes)
        test_graph = HyperGraph(test_nodes)
        
        # Print graph metrics for each split
        print("\nGraph Statistics:")
        print("-" * 50)
        
        for split_name, split_graph in [("Train", train_graph), ("Val", val_graph), ("Test", test_graph)]:
            nodes = split_graph.get_nodes()
            print(f"\n{split_name} Graph:")
            print(f"Total Nodes: {len(nodes)}")
            
            # Average edges per node
            total_edges = sum(len(node.edges) for node in nodes)
            avg_edges = total_edges / len(nodes) if nodes else 0
            print(f"Average Edges per Node: {avg_edges:.2f}")
            
            # Label distribution
            label_dist = {}
            for node in nodes:
                label = node.label
                label_dist[label] = label_dist.get(label, 0) + 1
            
            print("\nLabel Distribution:")
            for label, count in sorted(label_dist.items()):
                percentage = (count / len(nodes)) * 100
                print(f"Label {label}: {count} nodes ({percentage:.1f}%)")
        
        # Create visualization if enabled
        if self.hyperparameters["visualize"]:
            # Create attribute labels
            attr_labels = {}
            for i, node in enumerate(node_list):
                attrs = sorted(node.attributes.keys())
                attr_labels[i] = f"({', '.join(attrs)})"
            
            visualize_graph(
                train_graph,
                nx_graph=nx_graph,  # Pass the pre-built NetworkX graph
                attribute_labels=attr_labels,
                title=f"Graph Visualization ({'Test Mode' if self.hyperparameters['test_mode'] else 'Full Dataset'})",
                show=self.hyperparameters["show_viz"]
            )
        
        return train_graph, val_graph, test_graph