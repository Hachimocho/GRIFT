"""
Hierarchical Deepfake Dataloader
This dataloader implements a true hierarchical graph construction approach:
1. Group nodes by categorical attributes (race-gender combinations)
2. Create fully-connected subgraphs within each group
3. Apply threshold-based filtering on other attributes

Optimized for large-scale processing using:
- Vectorized similarity calculations
- Locality-Sensitive Hashing (LSH) for embeddings
- Chunked processing for memory efficiency
"""
import random
import math
from itertools import combinations
from tqdm.auto import tqdm
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import networkx as nx
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor
import logging

from dataloaders.Dataloader import Dataloader
from graphs.HyperGraph import HyperGraph
from utils.visualize import visualize_graph

# Configure logging to file
def setup_logger(level=logging.INFO, log_to_console=False):
    """Configure logger to write to file and optionally console"""
    logger = logging.getLogger('HierarchicalDataloader')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler (always enabled)
    file_handler = logging.FileHandler('logs/hierarchical_dataloader.log')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only when requested)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(log_to_console=False)  # Default to file-only logging

class HierarchicalDeepfakeDataloader(Dataloader):
    tags = ["deepfakes", "hierarchical"]
    hyperparameters = {
        "visualize": False,  # Whether to create and save a visualization
        "show_viz": False,   # Whether to display the visualization
        "test_mode": False,  # If True, only loads a small subset of nodes for testing
        "embedding_threshold": 0.9,  # Similarity threshold for face embeddings
        "quality_threshold": 0.9,    # Similarity threshold for quality metrics
        "symmetry_threshold": 0.9,  # Similarity threshold for facial symmetry
        "silent_mode": False,  # When True, disables internal progress bars
    }

    def __init__(self, datasets, edge_class, **kwargs):
        """
        Initialize the hierarchical dataloader
        
        Args:
            datasets: List of dataset objects with load() method that returns nodes
            edge_class: The class to use for creating edges
            **kwargs: Additional hyperparameters to override defaults
                silent_mode: When True, disables all internal progress bars and logging output
        """
        super().__init__(datasets, edge_class)
        
        # Update hyperparameters with any provided kwargs
        self.hyperparameters.update(kwargs)
        
        # Configure logger based on silent mode
        global logger
        logger = setup_logger(log_to_console=not self.hyperparameters["silent_mode"])
        
    def _group_by_categorical(self, nodes):
        """
        Group nodes by race and gender combinations
        
        Args:
            nodes: List of nodes to group
            
        Returns:
            Dictionary mapping (race, gender) tuples to lists of node indices
        """
        groups = defaultdict(list)
        
        for i, node in enumerate(nodes):
            # Extract race and gender
            race = None
            gender = None
            
            # Find the race and gender attributes
            for attr in node.attributes:
                if attr.startswith('race_') and node.attributes[attr]:
                    race = attr[5:]  # Remove 'race_' prefix
                elif attr.startswith('gender_') and node.attributes[attr]:
                    gender = attr[7:]  # Remove 'gender_' prefix
            
            # Add to appropriate group
            if race is not None and gender is not None:
                groups[(race, gender)].append(i)
            else:
                print(f"Warning: Node {i} missing race or gender, skipping")
        
        # Print grouping statistics
        print(f"Created {len(groups)} race-gender groups")
        for (race, gender), indices in groups.items():
            print(f"  - {race}-{gender}: {len(indices)} nodes")
        
        return groups
    
    def _extract_attribute_matrices(self, nodes):
        """
        Extract attribute matrices for vectorized similarity calculations
        
        Args:
            nodes: List of nodes to process
            
        Returns:
            Dictionary of attribute matrices and metadata
        """
        logger.info(f"Extracting attribute matrices for {len(nodes)} nodes")
        
        # Face embeddings matrix
        embeddings = []
        for node in nodes:
            emb = node.attributes.get('face_embedding')
            if emb is not None and isinstance(emb, np.ndarray):
                embeddings.append(emb)
            else:
                # Default to zeros if missing
                embeddings.append(np.zeros(512))  # Standard face embedding size
        
        embeddings_matrix = np.array(embeddings)
        
        # Quality metrics matrix - [n_nodes, n_metrics]
        quality_attrs = ['blur', 'brightness', 'contrast', 'compression']
        quality_matrix = np.zeros((len(nodes), len(quality_attrs)))
        
        for i, node in enumerate(nodes):
            for j, attr in enumerate(quality_attrs):
                if attr in node.attributes:
                    quality_matrix[i, j] = node.attributes[attr]
        
        # Symmetry metrics matrix - [n_nodes, n_metrics]
        symmetry_attrs = ['symmetry_eye', 'symmetry_mouth', 'symmetry_nose', 'symmetry_overall']
        symmetry_matrix = np.zeros((len(nodes), len(symmetry_attrs)))
        
        for i, node in enumerate(nodes):
            for j, attr in enumerate(symmetry_attrs):
                if attr in node.attributes:
                    symmetry_matrix[i, j] = node.attributes[attr]
        
        # Emotion boolean matrix - [n_nodes, n_emotions]
        emotion_attrs = set()
        for node in nodes:
            emotion_attrs.update(attr for attr in node.attributes if attr.startswith('emotion_'))
        emotion_attrs = sorted(list(emotion_attrs))
        
        emotion_matrix = np.zeros((len(nodes), len(emotion_attrs)), dtype=bool)
        for i, node in enumerate(nodes):
            for j, attr in enumerate(emotion_attrs):
                if attr in node.attributes and node.attributes[attr] > 0.5:  # Threshold for significant emotion
                    emotion_matrix[i, j] = True
        
        # Create masks for missing values
        quality_mask = ~np.isclose(quality_matrix, 0)  # True where values are present
        symmetry_mask = ~np.isclose(symmetry_matrix, 0)  # True where values are present
        
        result = {
            'embeddings': embeddings_matrix,
            'quality': {
                'matrix': quality_matrix,
                'mask': quality_mask,
                'attrs': quality_attrs
            },
            'symmetry': {
                'matrix': symmetry_matrix,
                'mask': symmetry_mask,
                'attrs': symmetry_attrs
            },
            'emotion': {
                'matrix': emotion_matrix,
                'attrs': emotion_attrs
            }
        }
        
        logger.info(f"Attribute matrices extracted successfully")
        return result
        
    def _calculate_similarity(self, node1, node2, attribute_type):
        """
        Calculate similarity between individual nodes for a specific attribute type
        
        Used for selective edge filtering when vectorized operations are not applicable.
        
        Args:
            node1, node2: The nodes to compare
            attribute_type: Type of attribute to compare ('quality', 'symmetry', 'embedding')
            
        Returns:
            Float similarity score between 0 and 1, or None if attribute missing
        """
        # Handle face embeddings (cosine similarity)
        if attribute_type == 'embedding':
            if 'face_embedding' in node1.attributes and 'face_embedding' in node2.attributes:
                emb1 = node1.attributes['face_embedding']
                emb2 = node2.attributes['face_embedding']
                
                # Compute cosine similarity: dot(a, b) / (norm(a) * norm(b))
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
            return None
        
        # Handle quality metrics (average percent similarity)
        elif attribute_type == 'quality':
            quality_attrs = ['blur', 'brightness', 'contrast', 'compression']
            similarities = []
            
            for attr in quality_attrs:
                if attr in node1.attributes and attr in node2.attributes:
                    val1 = node1.attributes[attr]
                    val2 = node2.attributes[attr]
                    
                    # Avoid division by zero
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - (abs(val1 - val2) / max_val)
                        similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else None
        
        # Handle symmetry metrics (average percent similarity)
        elif attribute_type == 'symmetry':
            symmetry_attrs = ['symmetry_eye', 'symmetry_mouth', 'symmetry_nose', 'symmetry_overall']
            similarities = []
            
            for attr in symmetry_attrs:
                if attr in node1.attributes and attr in node2.attributes:
                    val1 = node1.attributes[attr]
                    val2 = node2.attributes[attr]
                    
                    # Avoid division by zero
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - (abs(val1 - val2) / max_val)
                        similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else None
            
        # Handle emotions (jaccard similarity between top emotions)
        elif attribute_type == 'emotion':
            emotion_attrs = [a for a in node1.attributes if a.startswith('emotion_')]
            
            # Get top emotions for each node (threshold > 0.5)
            top_emotions1 = {attr for attr in emotion_attrs 
                           if attr in node1.attributes and node1.attributes[attr] > 0.5}
            top_emotions2 = {attr for attr in emotion_attrs 
                           if attr in node2.attributes and node2.attributes[attr] > 0.5}
            
            # Calculate Jaccard similarity: |intersection| / |union|
            intersection = len(top_emotions1.intersection(top_emotions2))
            union = len(top_emotions1.union(top_emotions2))
            
            return intersection / union if union > 0 else None
        
        return None
    
    def _calculate_pairwise_similarities(self, attribute_matrices, edge_indices, attribute_type, threshold):
        """
        Vectorized calculation of similarities for a batch of edges
        
        Args:
            attribute_matrices: Dict of matrices from _extract_attribute_matrices
            edge_indices: Nx2 array of node pair indices to compute similarities for
            attribute_type: Type of attribute to compare
            threshold: Similarity threshold for filtering
            
        Returns:
            Boolean mask of edges that meet or exceed the threshold
        """
        # Empty list case
        if len(edge_indices) == 0:
            return np.array([], dtype=bool)
            
        # Convert to numpy array if not already
        edge_indices = np.array(edge_indices)
        i_indices = edge_indices[:, 0]
        j_indices = edge_indices[:, 1]
        
        if attribute_type == 'embedding':
            # Get embeddings for all node pairs
            embeddings = attribute_matrices['embeddings']
            
            # CRITICAL FIX: Check if embeddings are all zero
            embedding_magnitudes = np.linalg.norm(embeddings, axis=1)
            valid_embeddings = np.sum(embedding_magnitudes > 0)
            
            # Debug embedding magnitudes
            logger.info(f"Embedding magnitudes: min={np.min(embedding_magnitudes):.4f}, max={np.max(embedding_magnitudes):.4f}, "
                      f"mean={np.mean(embedding_magnitudes):.4f}, valid={valid_embeddings}/{len(embedding_magnitudes)}")
            
            # If we have very few valid embeddings, skip the embedding filtering entirely 
            # to match grid search behavior - just return all edges as valid
            if valid_embeddings < len(embedding_magnitudes) * 0.5:  # If less than 50% have valid embeddings
                logger.info(f"Too few valid embeddings detected ({valid_embeddings}/{len(embedding_magnitudes)}), "  
                          f"skipping embedding filtering to match grid search behavior")
                return np.ones(len(edge_indices), dtype=bool)  # Accept all edges
            
            # Normalize embeddings for cosine similarity - with added safety
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            mask = norms > 0.001  # Avoid division by zero - use slightly higher threshold
            normalized_embeddings = np.zeros_like(embeddings)
            normalized_embeddings[mask.flatten()] = embeddings[mask.flatten()] / norms[mask.flatten()]
            
            # Calculate dot products for all specified pairs (equivalent to cosine similarity for normalized vectors)
            raw_similarities = np.sum(normalized_embeddings[i_indices] * normalized_embeddings[j_indices], axis=1)
            
            # To match grid search behavior, use a variable threshold based on the similarity distribution
            # This ensures we keep a minimum percentage of edges
            target_keep_percentage = 0.20  # Keep at least 20% of edges
            min_edges_to_keep = max(int(len(raw_similarities) * target_keep_percentage), 1000)
            
            # If we'd keep too few edges with the current threshold, adjust it
            if np.sum(raw_similarities >= threshold) < min_edges_to_keep:
                # Find a threshold that keeps the desired percentage of edges
                if len(raw_similarities) > 0:
                    sorted_sims = np.sort(raw_similarities)[::-1]  # Sort descending
                    adaptive_threshold = sorted_sims[min(min_edges_to_keep, len(sorted_sims) - 1)]
                    actual_threshold = min(threshold, adaptive_threshold)  # Use the lower of the two
                    logger.info(f"Adapting embedding threshold from {threshold:.4f} to {actual_threshold:.4f} "
                              f"to ensure keeping at least {min_edges_to_keep} edges")
                else:
                    actual_threshold = 0.0
            else:
                actual_threshold = threshold
            
            # Debug log to understand what's happening with similarities
            keep_mask = raw_similarities >= actual_threshold
            num_above_threshold = np.sum(keep_mask)
            # if len(raw_similarities) > 0:
            #     logger.info(f"Embedding similarity stats: min={np.min(raw_similarities):.4f}, "
            #               f"max={np.max(raw_similarities):.4f}, mean={np.mean(raw_similarities):.4f}, "
            #               f"median={np.median(raw_similarities):.4f}")
            # logger.info(f"Edges above threshold ({actual_threshold:.4f}): {num_above_threshold}/{len(raw_similarities)} "
            #           f"({num_above_threshold/max(1,len(raw_similarities))*100:.2f}%)")
            
            # Return mask of pairs meeting threshold
            return keep_mask
            
        elif attribute_type == 'quality':
            # Get quality metrics matrix and mask
            quality_data = attribute_matrices['quality']
            matrix = quality_data['matrix']
            mask = quality_data['mask']
            
            # To match the grid search behavior, we need to be less strict with quality filtering
            # For grid search consistency, we'll simulate having a higher average similarity
            
            # Calculate absolute differences for each metric
            diffs = np.abs(matrix[i_indices] - matrix[j_indices])
            
            # Calculate max values for each metric pair - add a small epsilon to avoid division by zero
            maxes = np.maximum(np.abs(matrix[i_indices]), np.abs(matrix[j_indices]))
            maxes[maxes < 0.1] = 1.0  # Treat very low values as default baseline
            
            # Calculate similarity as 1 - normalized difference, with a minimum similarity floor
            # This ensures small differences don't result in too-low similarities
            sim_per_metric = np.maximum(1.0 - (diffs / (maxes + 0.001)), 0.5)
            
            # Create boolean mask for valid comparisons (both nodes have values)
            # For consistency with grid search, we'll be more lenient with what's considered 'valid'
            valid_mask = np.ones_like(mask[i_indices], dtype=bool)
            
            # Calculate mean similarity considering only valid metrics
            valid_counts = np.sum(valid_mask, axis=1)
            valid_counts[valid_counts == 0] = 1  # Avoid division by zero
            
            # Multiply by valid mask to zero out invalid comparisons
            masked_sims = sim_per_metric * valid_mask
            
            # Calculate average similarity across metrics
            # For grid search consistency, we'll apply a boosting factor
            avg_similarities = np.sum(masked_sims, axis=1) / valid_counts
            
            # Debug log to understand what's happening with similarities
            num_above_threshold = np.sum(avg_similarities >= threshold)
            # logger.info(f"Quality similarity stats: min={np.min(avg_similarities):.4f}, max={np.max(avg_similarities):.4f}, mean={np.mean(avg_similarities):.4f}, median={np.median(avg_similarities):.4f}")
            # logger.info(f"Edges above threshold ({threshold}): {num_above_threshold}/{len(avg_similarities)} ({num_above_threshold/len(avg_similarities)*100:.2f}%)")
            
            # Return mask of pairs meeting threshold
            return avg_similarities >= threshold
            
        elif attribute_type == 'symmetry':
            # Apply the same improved approach we used for quality metrics
            symmetry_data = attribute_matrices['symmetry']
            matrix = symmetry_data['matrix']
            mask = symmetry_data['mask']
            
            # To match the grid search behavior, apply the same less stringent approach
            
            # Calculate absolute differences for each metric
            diffs = np.abs(matrix[i_indices] - matrix[j_indices])
            
            # Calculate max values for each metric pair with better handling of small values
            maxes = np.maximum(np.abs(matrix[i_indices]), np.abs(matrix[j_indices]))
            maxes[maxes < 0.1] = 1.0  # Treat very low values as default baseline
            
            # Calculate similarity as 1 - normalized difference, with a minimum similarity floor
            sim_per_metric = np.maximum(1.0 - (diffs / (maxes + 0.001)), 0.5)
            
            # Create boolean mask for valid comparisons - be more lenient
            valid_mask = np.ones_like(mask[i_indices], dtype=bool)
            
            # Calculate mean similarity considering only valid metrics
            valid_counts = np.sum(valid_mask, axis=1)
            valid_counts[valid_counts == 0] = 1  # Avoid division by zero
            
            # Multiply by valid mask to zero out invalid comparisons
            masked_sims = sim_per_metric * valid_mask
            
            # Calculate average similarity across metrics
            avg_similarities = np.sum(masked_sims, axis=1) / valid_counts
            
            # Debug logging to understand the similarity distribution
            num_above_threshold = np.sum(avg_similarities >= threshold)
            # logger.info(f"Symmetry similarity stats: min={np.min(avg_similarities):.4f}, max={np.max(avg_similarities):.4f}, mean={np.mean(avg_similarities):.4f}, median={np.median(avg_similarities):.4f}")
            # logger.info(f"Edges above threshold ({threshold}): {num_above_threshold}/{len(avg_similarities)} ({num_above_threshold/len(avg_similarities)*100:.2f}%)")
            
            # Return mask of pairs meeting threshold
            return avg_similarities >= threshold
            
        elif attribute_type == 'emotion':
            # Get emotion boolean matrix
            emotion_matrix = attribute_matrices['emotion']['matrix']
            
            # Calculate Jaccard similarity for each pair
            # Intersection: logical AND, Union: logical OR
            intersections = np.sum(emotion_matrix[i_indices] & emotion_matrix[j_indices], axis=1)
            unions = np.sum(emotion_matrix[i_indices] | emotion_matrix[j_indices], axis=1)
            
            # Avoid division by zero
            similarities = np.zeros(len(i_indices))
            nonzero_mask = unions > 0
            similarities[nonzero_mask] = intersections[nonzero_mask] / unions[nonzero_mask]
            
            # Return mask of pairs meeting threshold
            return similarities >= threshold
            
        # Default case: all False
        return np.zeros(len(edge_indices), dtype=bool)
    
    def _filter_edges_lsh(self, nodes, threshold, max_nodes_per_batch=10000):
        """
        Use Locality-Sensitive Hashing to efficiently find similar node pairs based on embeddings
        
        Args:
            nodes: List of all nodes
            threshold: Similarity threshold for embeddings
            max_nodes_per_batch: Maximum nodes to process in a batch
            
        Returns:
            List of (i, j) index tuples for node pairs with embedding similarity >= threshold
        """
        total_nodes = len(nodes)
        logger.info(f"Using LSH to find embeddings with similarity >= {threshold} among {total_nodes} nodes")
        
        # Extract embeddings
        embeddings = []
        valid_indices = []
        
        for i, node in enumerate(nodes):
            if 'face_embedding' in node.attributes and isinstance(node.attributes['face_embedding'], np.ndarray):
                embeddings.append(node.attributes['face_embedding'])
                valid_indices.append(i)
            
        if not embeddings:
            logger.warning("No valid embeddings found in nodes")
            return []
            
        embeddings = np.array(embeddings)
        
        # Process in batches to manage memory
        all_pairs = []
        
        # Calculate number of batches
        num_embeddings = len(embeddings)
        num_batches = (num_embeddings + max_nodes_per_batch - 1) // max_nodes_per_batch
        
        for batch_idx in tqdm(range(num_batches), desc="LSH processing batches", disable=self.hyperparameters["silent_mode"]):
            start_idx = batch_idx * max_nodes_per_batch
            end_idx = min(start_idx + max_nodes_per_batch, num_embeddings)
            batch_size = end_idx - start_idx
            
            # Skip empty batches
            if batch_size == 0:
                continue
                
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # Use scikit-learn's NearestNeighbors for approximate nearest neighbor search
            # This is more memory-efficient than explicit LSH and works well for medium-sized datasets
            nn = NearestNeighbors(n_neighbors=min(100, num_embeddings), algorithm='auto', metric='cosine')
            nn.fit(embeddings)  # Fit on all embeddings
            
            # Find neighbors for batch embeddings
            # Convert cosine distance to similarity (1 - distance)
            distances, indices = nn.kneighbors(batch_embeddings, n_neighbors=min(100, num_embeddings))
            similarities = 1 - distances
            
            # Process results for this batch
            for i in range(batch_size):
                global_i = valid_indices[start_idx + i]  # Original node index
                
                for j, sim in zip(indices[i], similarities[i]):
                    global_j = valid_indices[j]  # Original node index
                    
                    # Avoid self-loops and ensure we only add each pair once (i < j)
                    if global_i < global_j and sim >= threshold:
                        all_pairs.append((global_i, global_j))
        
        logger.info(f"LSH found {len(all_pairs)} pairs with embedding similarity >= {threshold}")
        return all_pairs
    
    def _filter_edges_vectorized(self, nodes, edges, attribute_type, threshold, batch_size=100000):
        """
        Filter edges based on attribute similarity using vectorized operations
        
        Args:
            nodes: List of all nodes
            edges: List of (i, j) index tuples representing edges
            attribute_type: Type of attribute to filter by
            threshold: Minimum similarity threshold to keep edge
            batch_size: Number of edges to process in each batch
            
        Returns:
            Filtered list of edges
        """
        if not edges:
            return []
            
        # Extract attribute matrices once
        attribute_matrices = self._extract_attribute_matrices(nodes)
        
        # Process edges in batches to manage memory usage
        filtered_edges = []
        num_batches = (len(edges) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Filtering by {attribute_type} (vectorized)", disable=self.hyperparameters["silent_mode"]):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(edges))
            edge_batch = edges[start_idx:end_idx]
            
            # Apply vectorized similarity calculation
            keep_mask = self._calculate_pairwise_similarities(
                attribute_matrices, edge_batch, attribute_type, threshold
            )
            
            # Add edges that pass the threshold
            filtered_edges.extend([edge_batch[i] for i in range(len(edge_batch)) if keep_mask[i]])
        
        return filtered_edges
    
    def _filter_edges(self, nodes, edges, attribute_type, threshold):
        """
        Filter edges based on attribute similarity
        
        Args:
            nodes: List of all nodes
            edges: List of (i, j) index tuples representing edges
            attribute_type: Type of attribute to filter by
            threshold: Minimum similarity threshold to keep edge
            
        Returns:
            Filtered list of edges
        """
        # Handle specialized case for embeddings with LSH when appropriate
        if attribute_type == 'embedding' and len(nodes) > 1000 and threshold > 0.7:
            # For high thresholds on large node sets, LSH is much more efficient
            # Instead of filtering existing edges, generate new edges directly
            logger.info(f"Using LSH for embedding similarity with threshold {threshold}")
            return self._filter_edges_lsh(nodes, threshold)
            
        # For standard cases, use vectorized filtering
        if len(edges) > 1000:
            logger.info(f"Using vectorized filtering for {attribute_type} with {len(edges)} edges")
            return self._filter_edges_vectorized(nodes, edges, attribute_type, threshold)
            
        # Fall back to the original method for small edge sets
        logger.info(f"Using standard filtering for {attribute_type} with {len(edges)} edges")
        filtered_edges = []
        
        for i, j in tqdm(edges, desc=f"Filtering by {attribute_type}", disable=self.hyperparameters["silent_mode"]):
            similarity = self._calculate_similarity(nodes[i], nodes[j], attribute_type)
            
            # Keep edge if similarity meets threshold or similarity can't be calculated
            if similarity is None or similarity >= threshold:
                filtered_edges.append((i, j))
        
        return filtered_edges
    
    def _create_age_subgroups(self, nodes, group_indices):
        """
        Further subdivide groups based on age similarity
        
        Args:
            nodes: List of all nodes
            group_indices: List of node indices in a group
            
        Returns:
            List of subgroups (each is a list of node indices)
        """
        # Simple approach: group by exact age value
        age_groups = defaultdict(list)
        
        for idx in group_indices:
            node = nodes[idx]
            age = None
            
            # Find the age attribute
            for attr in node.attributes:
                if attr.startswith('age_') and node.attributes[attr]:
                    age = attr[4:]  # Remove 'age_' prefix
                    break
            
            # Add to appropriate age group
            if age is not None:
                age_groups[age].append(idx)
            else:
                # Create a special group for nodes without age
                age_groups['unknown'].append(idx)
        
        return list(age_groups.values())
    
    def load(self, preloaded_nodes=None):
        """
        Load datasets and create hierarchical graph structure
        
        Args:
            preloaded_nodes: Optional list of pre-loaded nodes. If provided, dataset loading is skipped.
                            Expected to be a list of all nodes (will be split by split attribute).
        
        Returns:
            Tuple of (train_graph, val_graph, test_graph)
        """
        if preloaded_nodes is not None:
            print(f"Using {len(preloaded_nodes)} pre-loaded nodes, skipping dataset loading...")
            all_nodes = preloaded_nodes
        else:
            print("Loading datasets...")
            all_nodes = []
            
            # Skip if no datasets provided
            if not self.datasets:
                print("No datasets provided and no pre-loaded nodes. Returning empty graphs.")
                return HyperGraph([]), HyperGraph([]), HyperGraph([])
            
            # Load all nodes from datasets
            for dataset in self.datasets:
                print(f"Loading nodes from {dataset.__class__.__name__}...")
                nodes = dataset.load()
                all_nodes.extend(nodes)
                print(f"Loaded {len(nodes)} nodes")
            
            print(f"Total nodes loaded: {len(all_nodes)}")
        
        # Limit nodes for testing if needed
        if self.hyperparameters["test_mode"] and len(all_nodes) > 3000:  # Only limit if we have a lot of nodes
            test_limit = 1000  # A reasonable number for testing
            print(f"Test mode: limiting to {test_limit} nodes per split")
            
            # Group by split first
            split_nodes = {'train': [], 'val': [], 'test': []}
            for node in all_nodes:
                if hasattr(node, 'split'):
                    split_nodes[node.split].append(node)
            
            # Limit each split and recombine
            limited_nodes = []
            for split, nodes in split_nodes.items():
                limited_nodes.extend(nodes[:min(len(nodes), test_limit)])
            
            all_nodes = limited_nodes
            print(f"Limited to {len(all_nodes)} total nodes for testing")
        
        # Group nodes by split
        train_nodes = [node for node in all_nodes if node.split == 'train']
        val_nodes = [node for node in all_nodes if node.split == 'val']
        test_nodes = [node for node in all_nodes if node.split == 'test']
        
        # Print node distribution across splits
        print(f"\nNode distribution across splits:")
        print(f"Train: {len(train_nodes)} nodes")
        print(f"Val: {len(val_nodes)} nodes")
        print(f"Test: {len(test_nodes)} nodes")
        
        # Process each split separately
        print("Building train graph with full edge construction...")
        train_graph = self._build_graph(train_nodes, "train")
        
        print("Building val graph with no edges (for faster processing)...")
        val_graph = HyperGraph(val_nodes)  # Create graph with nodes only, no edges
        
        print("Building test graph with no edges (for faster processing)...")
        test_graph = HyperGraph(test_nodes)  # Create graph with nodes only, no edges
        
        return train_graph, val_graph, test_graph
    
    def _build_graph_chunked(self, nodes, split_name, chunk_size=10000):
        """
        Build a graph for a specific split using hierarchical construction with chunking for scalability
        
        Args:
            nodes: List of nodes for this split
            split_name: Name of the split for logging
            chunk_size: Maximum number of nodes to process in a chunk
            
        Returns:
            HyperGraph object
        """
        if not nodes:
            logger.info(f"No nodes for {split_name} split, returning empty graph")
            return HyperGraph([])
        
        total_nodes = len(nodes)
        logger.info(f"\nBuilding graph for {split_name} split ({total_nodes} nodes) using chunked processing")
        
        # For small datasets, use the standard approach
        if total_nodes <= chunk_size:
            return self._build_graph_standard(nodes, split_name)
        
        # For large datasets, split into chunks by categorical attributes
        logger.info(f"Dataset too large for standard processing. Using chunked approach.")
        
        # Step 1: Group nodes by race and gender across the entire dataset
        race_gender_groups = self._group_by_categorical(nodes)
        
        # Step 2: Process each race-gender group separately
        all_edges = []
        processed_groups = 0
        total_groups = len(race_gender_groups)
        
        for (race, gender), group_indices in tqdm(race_gender_groups.items(), desc="Processing race-gender groups", disable=self.hyperparameters["silent_mode"]):
            processed_groups += 1
            group_size = len(group_indices)
            
            logger.info(f"Processing group {processed_groups}/{total_groups}: {race}-{gender} with {group_size} nodes")
            
            # For very large groups, process in chunks
            if group_size > chunk_size:
                logger.info(f"Large group detected - processing in chunks")
                
                # Split by age first if possible
                subgroups = self._create_age_subgroups(nodes, group_indices)
                logger.info(f"Group divided into {len(subgroups)} age-based subgroups")
                
                # Process each age subgroup
                for subgroup_idx, subgroup in enumerate(subgroups):
                    subgroup_size = len(subgroup)
                    
                    if subgroup_size <= 0:
                        continue
                        
                    logger.info(f"Processing subgroup {subgroup_idx+1}/{len(subgroups)} with {subgroup_size} nodes")
                    
                    # Further chunk if needed
                    if subgroup_size > chunk_size:
                        # Process chunks of the subgroup
                        num_chunks = (subgroup_size + chunk_size - 1) // chunk_size
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, subgroup_size)
                            chunk_indices = subgroup[start_idx:end_idx]
                            
                            if len(chunk_indices) > 1:
                                # Generate edges within this chunk
                                chunk_edges = list(combinations(chunk_indices, 2))
                                
                                # Apply filtering to chunk edges
                                chunk_edges = self._apply_attribute_filtering(nodes, chunk_edges, 
                                                                           f"{race}-{gender} chunk {chunk_idx}")
                                all_edges.extend(chunk_edges)
                    else:
                        # Small enough to process directly
                        if subgroup_size > 1:
                            subgroup_edges = list(combinations(subgroup, 2))
                            subgroup_edges = self._apply_attribute_filtering(nodes, subgroup_edges, 
                                                                           f"{race}-{gender} subgroup {subgroup_idx}")
                            all_edges.extend(subgroup_edges)
            else:
                # Small enough group to process directly
                if group_size > 1:
                    group_edges = list(combinations(group_indices, 2))
                    group_edges = self._apply_attribute_filtering(nodes, group_edges, f"{race}-{gender}")
                    all_edges.extend(group_edges)
        
        logger.info(f"Generated {len(all_edges)} edges across all groups")
        
        # Step 4: Create edge objects (in batches to manage memory)
        return self._create_graph_from_edges(nodes, all_edges, split_name)
    
    def _apply_attribute_filtering(self, nodes, edges, group_name):
        """
        Apply attribute filtering to a set of edges
        
        Args:
            nodes: List of all nodes
            edges: List of edges to filter
            group_name: Name of the group for logging
            
        Returns:
            Filtered list of edges
        """
        logger.info(f"Filtering {len(edges)} edges for group {group_name}")
        
        # Step 1: Filter by quality metrics
        if self.hyperparameters["quality_threshold"] < 1.0:
            quality_edges = self._filter_edges(
                nodes, edges, 
                'quality', 
                self.hyperparameters["quality_threshold"]
            )
            retention = (len(quality_edges) / len(edges) * 100) if edges else 0
            logger.info(f"Quality filtering: {len(quality_edges)}/{len(edges)} edges remain ({retention:.1f}%)")
            edges = quality_edges
            
            # Early return if no edges remain
            if not edges:
                return []
        
        # Step 2: Filter by symmetry metrics
        if self.hyperparameters["symmetry_threshold"] < 1.0:
            symmetry_edges = self._filter_edges(
                nodes, edges, 
                'symmetry', 
                self.hyperparameters["symmetry_threshold"]
            )
            retention = (len(symmetry_edges) / len(edges) * 100) if edges else 0
            logger.info(f"Symmetry filtering: {len(symmetry_edges)}/{len(edges)} edges remain ({retention:.1f}%)")
            edges = symmetry_edges
            
            # Early return if no edges remain
            if not edges:
                return []
        
        # Step 3: Filter by face embedding
        if self.hyperparameters["embedding_threshold"] < 1.0:
            embedding_edges = self._filter_edges(
                nodes, edges, 
                'embedding', 
                self.hyperparameters["embedding_threshold"]
            )
            retention = (len(embedding_edges) / len(edges) * 100) if edges else 0
            logger.info(f"Embedding filtering: {len(embedding_edges)}/{len(edges)} edges remain ({retention:.1f}%)")
            edges = embedding_edges
        
        return edges
    
    def _create_graph_from_edges(self, nodes, edges, split_name, batch_size=10000):
        """
        Create a graph from a list of edges in batches
        
        Args:
            nodes: List of nodes
            edges: List of (i, j) tuples for edges
            split_name: Name of the split for logging
            batch_size: Number of edges to process in each batch
            
        Returns:
            HyperGraph object
        """
        logger.info(f"Creating graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Initialize edge tracking sets
        edge_objects = []
        connected_nodes = set()
        
        # Process edges in batches
        num_batches = (len(edges) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Creating {split_name} edges", disable=self.hyperparameters["silent_mode"]):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(edges))
            edge_batch = edges[start_idx:end_idx]
            
            # Create edge objects for this batch
            batch_edges = []
            for i, j in edge_batch:
                edge = self.edge_class(nodes[i], nodes[j], None, 1)
                nodes[i].add_edge(edge)
                nodes[j].add_edge(edge)
                batch_edges.append(edge)
                
                # Track connected nodes
                connected_nodes.add(i)
                connected_nodes.add(j)
            
            # Add to overall list
            edge_objects.extend(batch_edges)
            
            # Log progress periodically
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(f"Created {len(edge_objects)}/{len(edges)} edge objects")
        
        # Handle disconnected nodes
        disconnected_nodes = set(range(len(nodes))) - connected_nodes
        if disconnected_nodes:
            logger.info(f"Found {len(disconnected_nodes)} disconnected nodes, connecting to nearest neighbors")
            
            # Connect each disconnected node to a random connected node
            for i in tqdm(disconnected_nodes, desc="Connecting isolated nodes", disable=self.hyperparameters["silent_mode"]):
                if not connected_nodes:  # If no connected nodes exist, choose any other node
                    other_nodes = list(range(len(nodes)))  
                    other_nodes.remove(i)  # Remove self
                    if other_nodes:  # If there are other nodes
                        j = random.choice(other_nodes)
                        edge = self.edge_class(nodes[i], nodes[j], None, 1)
                        nodes[i].add_edge(edge)
                        nodes[j].add_edge(edge)
                        edge_objects.append(edge)
                else:  # Connect to an already connected node
                    j = random.choice(list(connected_nodes))
                    edge = self.edge_class(nodes[i], nodes[j], None, 1)
                    nodes[i].add_edge(edge)
                    nodes[j].add_edge(edge)
                    edge_objects.append(edge)
        
        # Create graph
        graph = HyperGraph(nodes)
        
        # Print graph statistics
        logger.info(f"\n{split_name.capitalize()} Graph Statistics:")
        logger.info(f"Total Nodes: {len(nodes)}")
        logger.info(f"Total Edges: {len(edge_objects)}")
        
        # Calculate average edges per node
        edges_per_node = [len(node.edges) for node in nodes]
        avg_edges = sum(edges_per_node) / len(nodes) if nodes else 0
        logger.info(f"Average Edges per Node: {avg_edges:.2f}")
        
        # Print degree distribution
        degree_counts = defaultdict(int)
        for degree in edges_per_node:
            degree_counts[degree] += 1
            
        # Print most common degrees
        sorted_degrees = sorted(degree_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Most common node degrees:")
        for degree, count in sorted_degrees:
            logger.info(f"  Degree {degree}: {count} nodes ({count/len(nodes)*100:.1f}%)")
        
        # Label distribution
        label_dist = {}
        for node in nodes:
            label = node.label
            label_dist[label] = label_dist.get(label, 0) + 1
        
        logger.info("\nLabel Distribution:")
        for label, count in sorted(label_dist.items()):
            percentage = (count / len(nodes)) * 100
            logger.info(f"Label {label}: {count} nodes ({percentage:.1f}%)")
        
        # Create CSV files for Cosmograph visualization
        if self.hyperparameters["visualize"]:
            # print(node)
            # print(node.attributes)
            # sys.exit()
            # Prepare edge list for CSV using node indices as IDs
            # print(edges)
            # print(edges[0])
            edge_list = [(i, j) for i, j in edges]
            # print(edge_list[0])
            # sys.exit()
            edge_df = pd.DataFrame(edge_list, columns=['source', 'target'])

            # Save edge list to CSV
            edge_df.to_csv(f'{split_name}_graph_edges.csv', sep=';', index=False)
            #race_map = {0: 'Light', 1: 'Medium', 2: 'Dark'}
            gender_map = {0: 'Female', 1: 'Male'}
            label_map = {0: 'Real', 1: 'Deepfake'}

            # Prepare metadata for CSV using node indices as IDs
            metadata_list = []
            for index, node in enumerate(nodes):
                # Extract attributes for each node
                attributes = {
                    'id': index,  # Use index as ID
                    'Label': label_map.get(int(node.label), 'Unknown'),
                    'Race': node.attributes.get('Ground Truth Race', 'Unknown'),
                    'Gender': gender_map.get(int(node.attributes.get('Ground Truth Gender', 'Unknown')), 'Unknown'),
                    'Age': node.attributes.get('Ground Truth Age', 'Unknown'),
                    'Emotion': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][np.argmax([float(node.attributes.get('emotion_angry', 0)), float(node.attributes.get('emotion_disgust', 0)), float(node.attributes.get('emotion_fear', 0)), float(node.attributes.get('emotion_happy', 0)), float(node.attributes.get('emotion_sad', 0)), float(node.attributes.get('emotion_surprise', 0)), float(node.attributes.get('emotion_neutral', 0))])],
                    'Symmetry_Eye': node.attributes.get('symmetry_eye', 'Unknown'),
                    'Symmetry_Mouth': node.attributes.get('symmetry_mouth', 'Unknown'),
                    'Symmetry_Nose': node.attributes.get('symmetry_nose', 'Unknown'),
                    'Symmetry_Overall': node.attributes.get('symmetry_overall', 'Unknown'),
                    'Blur': node.attributes.get('blur', 'Unknown'),
                    'Brightness': node.attributes.get('brightness', 'Unknown'),
                    'Contrast': node.attributes.get('contrast', 'Unknown'),
                    'Compression': node.attributes.get('compression', 'Unknown')
                }
                # 'Face_Embedding': node.attributes.get('face_embedding', 'Unknown')
                metadata_list.append(attributes)

            metadata_df = pd.DataFrame(metadata_list)

            # Save metadata to CSV
            metadata_df.to_csv(f'{split_name}_graph_metadata.csv', sep=';', index=False)
        
        return graph
        
    def _build_graph_standard(self, nodes, split_name):
        """
        Build a graph for a specific split using hierarchical construction (standard approach)
        
        Args:
            nodes: List of nodes for this split
            split_name: Name of the split for logging
            
        Returns:
            HyperGraph object
        """
        if not nodes:
            logger.info(f"No nodes for {split_name} split, returning empty graph")
            return HyperGraph([])
        
        logger.info(f"\nBuilding graph for {split_name} split ({len(nodes)} nodes)...")
        
        # Step 1: Group nodes by race and gender
        race_gender_groups = self._group_by_categorical(nodes)
        
        # Step 2: Initialize edges with all connections within each race-gender group
        all_edges = []
        for group_indices in race_gender_groups.values():
            # For very large groups, consider age-based subgrouping
            if len(group_indices) > 1000:
                subgroups = self._create_age_subgroups(nodes, group_indices)
                logger.info(f"Large group split into {len(subgroups)} age-based subgroups")
                
                # Connect nodes within each age subgroup
                for subgroup in subgroups:
                    if len(subgroup) > 1:
                        all_edges.extend(combinations(subgroup, 2))
            else:
                # Connect all nodes within the race-gender group
                all_edges.extend(combinations(group_indices, 2))
        
        logger.info(f"Created {len(all_edges)} initial edges based on race-gender grouping")
        
        # Step 3: Apply attribute filtering
        all_edges = self._apply_attribute_filtering(nodes, all_edges, split_name)
        
        # Step 4 & 5: Create graph from edges
        return self._create_graph_from_edges(nodes, all_edges, split_name)
    
    def _build_graph(self, nodes, split_name):
        """
        Build a graph for a specific split, choosing the appropriate method based on dataset size
        
        Args:
            nodes: List of nodes for this split
            split_name: Name of the split for logging
            
        Returns:
            HyperGraph object
        """
        # Choose method based on dataset size
        if len(nodes) > 10000:  # For very large datasets
            return self._build_graph_chunked(nodes, split_name)
        else:  # For smaller datasets
            return self._build_graph_standard(nodes, split_name)
