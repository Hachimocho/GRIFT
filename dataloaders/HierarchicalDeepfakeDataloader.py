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
from typing import List, Dict, Tuple, Optional, Set, Any

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
        "age_split_threshold": 1000,  # Threshold for age-based subgrouping
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
        edge_indices_np = np.array(edge_indices) # Use a different name to avoid confusion later
        num_original_pairs = len(edge_indices_np)
        
        if num_original_pairs == 0:
            return np.array([], dtype=bool)
        
        i_indices_orig = edge_indices_np[:, 0]
        j_indices_orig = edge_indices_np[:, 1]
        
        # --- START ADDED VALIDATION ---
        num_nodes = len(attribute_matrices['embeddings']) # Get num_nodes from a representative matrix
        max_allowable_index = num_nodes - 1

        # Create masks for valid indices based on the original indices
        valid_i = (i_indices_orig >= 0) & (i_indices_orig <= max_allowable_index)
        valid_j = (j_indices_orig >= 0) & (j_indices_orig <= max_allowable_index)
        valid_pair_mask = valid_i & valid_j

        num_invalid_pairs = num_original_pairs - np.sum(valid_pair_mask)

        # Prepare the final mask, initialized to False
        final_keep_mask = np.zeros(num_original_pairs, dtype=bool)

        if num_invalid_pairs > 0:
            logger.warning(f"Found {num_invalid_pairs} edge pairs with out-of-range indices "
                           f"(max allowed: {max_allowable_index}) in vectorized batch. These pairs will be excluded.")
            # If all pairs are invalid, return the all-False mask
            if np.all(~valid_pair_mask):
                return final_keep_mask
            
        # Filter the indices to only include valid pairs for calculation
        i_indices = i_indices_orig[valid_pair_mask]
        j_indices = j_indices_orig[valid_pair_mask]
        # --- END ADDED VALIDATION ---
        
        # If no valid pairs remain after filtering, return the all-False mask
        if len(i_indices) == 0:
            return final_keep_mask

        # --- Calculate Similarities based on attribute_type --- 
        if attribute_type == 'embedding':
            if not np.any(valid_pair_mask):
                # No valid pairs remain after basic index checks
                return np.zeros(num_original_pairs, dtype=bool)

            # Get the edge indices for valid pairs only
            valid_edges = edge_indices_np[valid_pair_mask]
            i_indices_valid = valid_edges[:, 0]
            j_indices_valid = valid_edges[:, 1]

            embeddings = attribute_matrices['embeddings']

            # 1. Find all unique original indices involved in the VALID pairs
            unique_indices_in_valid_pairs = np.unique(np.concatenate((i_indices_valid, j_indices_valid)))

            # 2. Extract embeddings for these unique indices
            # Ensure indices are within bounds (should be guaranteed by valid_pair_mask, but belt-and-suspenders)
            max_allowable_emb_idx = embeddings.shape[0] - 1
            unique_indices_in_bounds = unique_indices_in_valid_pairs[unique_indices_in_valid_pairs <= max_allowable_emb_idx]
            if len(unique_indices_in_bounds) != len(unique_indices_in_valid_pairs):
                 logger.warning(f"Mismatch finding embeddings for unique indices. This might indicate an issue upstream from _calculate_pairwise_similarities.")
                 # Fallback: proceed with only the indices found within bounds
                 unique_indices_in_valid_pairs = unique_indices_in_bounds
                 if len(unique_indices_in_valid_pairs) == 0:
                     # If no valid indices left, return all False
                     final_keep_mask = np.zeros(num_original_pairs, dtype=bool)
                     final_keep_mask[valid_pair_mask] = False # Explicitly set valid pairs to false
                     return final_keep_mask
            
            unique_embeddings = embeddings[unique_indices_in_valid_pairs]

            # 3. Calculate norms and identify indices with norms > threshold
            norms = np.linalg.norm(unique_embeddings, axis=1)
            valid_norm_mask_for_unique = norms > 1e-8

            # 4. Get the ORIGINAL indices that have valid norms
            original_indices_with_valid_norms = unique_indices_in_valid_pairs[valid_norm_mask_for_unique]
            valid_norm_indices_set = set(original_indices_with_valid_norms)

            # 5. Create a mask for the VALID pairs where BOTH nodes have a valid norm
            cosine_calculable_pair_mask = np.array([
                (i in valid_norm_indices_set) and (j in valid_norm_indices_set)
                for i, j in zip(i_indices_valid, j_indices_valid)
            ], dtype=bool)

            # Initialize similarities for all VALID pairs as below threshold (-1)
            similarities_for_valid_pairs = np.full(len(i_indices_valid), -1.0, dtype=float)

            # Only proceed if there are pairs where cosine sim can actually be calculated
            if np.any(cosine_calculable_pair_mask):
                # Get the pairs where calculation is possible
                calculable_i = i_indices_valid[cosine_calculable_pair_mask]
                calculable_j = j_indices_valid[cosine_calculable_pair_mask]

                # Get unique original indices involved ONLY in calculable pairs
                unique_calculable_indices = np.unique(np.concatenate((calculable_i, calculable_j)))

                # Create a mapping from original index to its position in the normalized vector array
                original_to_normalized_pos = {original_idx: pos for pos, original_idx in enumerate(unique_calculable_indices)}

                # Extract and normalize embeddings ONLY for these calculable indices
                calculable_embeddings = embeddings[unique_calculable_indices]
                # Ensure norms are calculated correctly with keepdims=True for broadcasting
                calculable_norms = np.linalg.norm(calculable_embeddings, axis=1, keepdims=True)
                # We know these norms are > 1e-8 because of cosine_calculable_pair_mask
                normalized_calculable_vectors = calculable_embeddings / calculable_norms # Broadcasting works here

                # Map the 'calculable_i' and 'calculable_j' indices to their positions
                pos_i = np.array([original_to_normalized_pos[idx] for idx in calculable_i])
                pos_j = np.array([original_to_normalized_pos[idx] for idx in calculable_j])

                # Get the corresponding normalized vectors
                norm_vec_i = normalized_calculable_vectors[pos_i]
                norm_vec_j = normalized_calculable_vectors[pos_j]

                # Calculate cosine similarities (dot product of normalized vectors)
                cosine_similarities = np.sum(norm_vec_i * norm_vec_j, axis=1)

                # Store these calculated similarities in the correct positions
                similarities_for_valid_pairs[cosine_calculable_pair_mask] = cosine_similarities

            # Apply the threshold to the calculated similarities (or -1 for failed pairs)
            threshold_met_mask_for_valid_pairs = similarities_for_valid_pairs >= threshold

            # --- Map results back to the original edge_indices_np shape ---
            final_keep_mask = np.zeros(num_original_pairs, dtype=bool)
            final_keep_mask[valid_pair_mask] = threshold_met_mask_for_valid_pairs
            
            # The final_keep_mask now correctly represents which of the ORIGINAL pairs should be kept
            return final_keep_mask
            
        elif attribute_type == 'quality':
            # Get quality metrics matrix and mask
            quality_data = attribute_matrices['quality']
            matrix = quality_data['matrix']
            mask = quality_data['mask'] # Mask indicating valid entries
            
            # Select data only for valid indices
            matrix_i = matrix[i_indices]
            matrix_j = matrix[j_indices]
            mask_i = mask[i_indices]
            mask_j = mask[j_indices]
            
            # Calculate absolute differences for each metric
            diffs = np.abs(matrix_i - matrix_j)
            
            # Calculate max values for each metric pair
            maxes = np.maximum(np.abs(matrix_i), np.abs(matrix_j))
            # Avoid division by zero/very small numbers - use mask for this
            maxes_gt_zero = maxes > 1e-8
            
            # Calculate similarity as 1 - normalized difference, handle division by zero
            sim_per_metric = np.zeros_like(diffs)
            # Only calculate where maxes are significant
            sim_per_metric[maxes_gt_zero] = 1.0 - (diffs[maxes_gt_zero] / (maxes[maxes_gt_zero]))
            sim_per_metric = np.clip(sim_per_metric, 0, 1) # Ensure similarity is [0, 1]
            
            # Create boolean mask for valid comparisons (both nodes must have the metric)
            valid_mask = mask_i & mask_j
            
            # Calculate mean similarity considering only valid metrics for each pair
            valid_counts = np.sum(valid_mask, axis=1)
            mean_sim = np.zeros(len(i_indices), dtype=float)
            has_valid = valid_counts > 0
            
            # Apply valid_mask to sim_per_metric before summing
            # Sum only where the metric comparison itself is valid
            sum_sim = np.sum(sim_per_metric * valid_mask, axis=1)
            mean_sim[has_valid] = sum_sim[has_valid] / valid_counts[has_valid]
            
            valid_results_mask = mean_sim >= threshold
        
        elif attribute_type == 'symmetry':
            # Get symmetry metrics matrix and mask
            symmetry_data = attribute_matrices['symmetry']
            matrix = symmetry_data['matrix']
            mask = symmetry_data['mask'] # Mask indicating valid entries

            # Select data only for valid indices
            matrix_i = matrix[i_indices]
            matrix_j = matrix[j_indices]
            mask_i = mask[i_indices]
            mask_j = mask[j_indices]

            # Calculate absolute differences
            diffs = np.abs(matrix_i - matrix_j)

            # Calculate max values
            maxes = np.maximum(np.abs(matrix_i), np.abs(matrix_j))
            maxes_gt_zero = maxes > 1e-8
            
            # Calculate similarity, handle division by zero
            sim_per_metric = np.zeros_like(diffs)
            sim_per_metric[maxes_gt_zero] = 1.0 - (diffs[maxes_gt_zero] / (maxes[maxes_gt_zero]))
            sim_per_metric = np.clip(sim_per_metric, 0, 1)

            # Mask for valid comparisons
            valid_mask = mask_i & mask_j

            # Calculate mean similarity
            valid_counts = np.sum(valid_mask, axis=1)
            mean_sim = np.zeros(len(i_indices), dtype=float)
            has_valid = valid_counts > 0
            sum_sim = np.sum(sim_per_metric * valid_mask, axis=1)
            mean_sim[has_valid] = sum_sim[has_valid] / valid_counts[has_valid]
            
            valid_results_mask = mean_sim >= threshold
        
        else:
             logger.warning(f"Unsupported attribute type '{attribute_type}' in vectorized calculation.")
             # If attribute type is unknown, assume no pairs pass for safety
             valid_results_mask = np.zeros(len(i_indices), dtype=bool)

        # Place the results for valid pairs into the final mask
        final_keep_mask[valid_pair_mask] = valid_results_mask
        
        return final_keep_mask
    
    def _filter_edges_lsh(self, nodes, edges_to_filter, threshold, max_nodes_per_batch=10000):
        """
        Use Locality-Sensitive Hashing (approximated with NearestNeighbors) to efficiently
        filter an existing list of edges based on embedding similarity.

        Args:
            nodes: List of all nodes.
            edges_to_filter: List of (i, j) index tuples representing edges to be filtered.
            threshold: Similarity threshold for embeddings.
            max_nodes_per_batch: Maximum nodes to process in a batch for NN search.

        Returns:
            List of (i, j) index tuples from edges_to_filter where nodes have
            embedding similarity >= threshold.
        """
        if not edges_to_filter:
            logger.info("LSH filtering: Input edge list is empty.")
            return []

        total_nodes = len(nodes)
        # logger.info(f"Filtering {len(edges_to_filter)} edges using LSH "
        #             f"with similarity >= {threshold} among {total_nodes} nodes")

        # Extract embeddings and map original indices
        embeddings = []
        valid_indices_map = {} # Map original index -> embedding array index
        original_indices = [] # List of original indices corresponding to embeddings array

        for i, node in enumerate(nodes):
            if 'face_embedding' in node.attributes and isinstance(node.attributes['face_embedding'], np.ndarray):
                embedding_idx = len(embeddings)
                embeddings.append(node.attributes['face_embedding'])
                valid_indices_map[i] = embedding_idx
                original_indices.append(i)

        if not embeddings:
            logger.warning("No valid embeddings found in nodes for LSH filtering.")
            # If no embeddings, we can't filter by them, so return the original edges?
            # Or return empty? Returning empty seems safer if embedding filter is required.
            return []

        embeddings = np.array(embeddings)
        num_embeddings = len(embeddings)

        # --- Find all candidate pairs using NearestNeighbors (existing logic) ---
        candidate_pairs = set()
        num_batches = (num_embeddings + max_nodes_per_batch - 1) // max_nodes_per_batch

        # Consider optimizing k based on expected density and threshold
        k_neighbors = min(max(50, int(num_embeddings * 0.05)), 200) # Heuristic k

        nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='cosine', n_jobs=-1)
        nn.fit(embeddings)

        logger.info(f"_filter_edges_lsh: Starting LSH filtering with threshold {threshold:.2f}")
        logger.info(f"_filter_edges_lsh: Input edge count: {len(edges_to_filter)}")

        for batch_idx in tqdm(range(num_batches), desc="LSH candidate search", disable=self.hyperparameters["silent_mode"]):
            start_idx = batch_idx * max_nodes_per_batch
            end_idx = min(start_idx + max_nodes_per_batch, num_embeddings)
            if start_idx >= end_idx: continue

            batch_embeddings = embeddings[start_idx:end_idx]
            distances, indices = nn.kneighbors(batch_embeddings)
            similarities = 1 - distances

            for i_batch in range(len(batch_embeddings)):
                emb_i = start_idx + i_batch # Index within the embeddings array
                global_i = original_indices[emb_i] # Original node index

                for j_neighbor, sim in zip(indices[i_batch], similarities[i_batch]):
                    # j_neighbor is index within embeddings array
                    if emb_i == j_neighbor: # Skip self-comparison
                        continue

                    global_j = original_indices[j_neighbor] # Original node index

                    if sim >= threshold:
                        # Add ordered pair to the set
                        pair = tuple(sorted((global_i, global_j)))
                        candidate_pairs.add(pair)
        # --- End of candidate pair finding ---

        # logger.info(f"LSH identified {len(candidate_pairs)} candidate pairs "
        #             f"with similarity >= {threshold}")
        logger.info(f"_filter_edges_lsh: Found {len(candidate_pairs)} candidate pairs meeting threshold >= {threshold:.2f} via kneighbors")
        if not candidate_pairs:
            return []

        # --- Filter the input edges using the candidate pairs ---
        # Ensure edges_to_filter are also ordered tuples for correct set comparison
        input_edges_set = {tuple(sorted(edge)) for edge in edges_to_filter}

        filtered_edges_set = input_edges_set.intersection(candidate_pairs)

        # logger.info(f"LSH filtering kept {len(filtered_edges_set)} edges "
        #             f"out of {len(edges_to_filter)} original edges.")
        logger.info(f"_filter_edges_lsh: Final filtered edge count after intersection: {len(filtered_edges_set)}")

        # Return as list of tuples
        return [edge for edge in filtered_edges_set] # Convert back to list

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
        logger.info(f"_filter_edges_vectorized: Called")
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
        logger.info(f"_filter_edges: Called")
        # LSH filtering currently deprecated due to filtering issues

        # Handle specialized case for embeddings with LSH when appropriate
        # Check edges list length as well, maybe LSH isn't worth it for few edges
        # if attribute_type == 'embedding' and len(nodes) > 1000 and threshold > 1.0 and len(edges) > 1000:
        #     logger.info(f"Using LSH for filtering {len(edges)} embedding edges with threshold {threshold}")
        #     # Pass the current 'edges' list to _filter_edges_lsh
        #     return self._filter_edges_lsh(nodes, edges, threshold) # Pass 'edges'

        # For standard cases, use vectorized filtering
        if len(edges) > 1000: # Use vectorized for non-embedding or when LSH conditions not met
            logger.info(f"Using vectorized filtering for {attribute_type} with {len(edges)} edges")
            return self._filter_edges_vectorized(nodes, edges, attribute_type, threshold)

        # Fall back to the original method for small edge sets
        logger.info(f"Using standard (iterative) filtering for {attribute_type} with {len(edges)} edges")
        filtered_edges = []

        # Ensure nodes have the necessary attributes pre-fetched if needed by _calculate_similarity
        # Consider pre-calculating similarities if _calculate_similarity is slow and called repeatedly

        for i, j in tqdm(edges, desc=f"Filtering by {attribute_type}", disable=self.hyperparameters["silent_mode"]):
            try:
                # Make sure nodes[i] and nodes[j] are valid indices
                if i >= len(nodes) or j >= len(nodes):
                     logger.warning(f"Invalid node index in edge list: ({i}, {j}). Skipping edge.")
                     continue

                similarity = self._calculate_similarity(nodes[i], nodes[j], attribute_type)

                # Keep edge if similarity meets threshold OR if similarity calculation fails (returns None)
                # Consider if failing similarity should always exclude the edge? Depends on desired behavior.
                if similarity is None or similarity >= threshold:
                    filtered_edges.append((i, j))
            except IndexError:
                 logger.warning(f"IndexError processing edge ({i}, {j}) during {attribute_type} filtering. Max node index: {len(nodes)-1}")
            except Exception as e:
                 logger.exception(f"Error calculating similarity for edge ({i}, {j}), type {attribute_type}: {e}")


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
            # Standard approach handles map generation internally
            return self._build_graph_standard(nodes, split_name)
        
        # For large datasets, split into chunks by categorical attributes
        logger.info(f"Dataset too large for standard processing. Using chunked approach.")

        # --- Generate the final subgroup map upfront for fallback consistency --- 
        logger.info("Generating final subgroup map for fallback connectivity...")
        final_subgroups = self._group_by_categorical(nodes) # Race/Gender first
        # Refine by age
        node_index_to_subgroup_id = {}
        current_subgroup_id_counter = 0
        final_subgroup_details = {}
        for group_key, group_indices in final_subgroups.items():
            age_subgroups = self._create_age_subgroups(nodes, group_indices)
            for age_subgroup_indices in age_subgroups:
                if age_subgroup_indices:
                    subgroup_id = current_subgroup_id_counter
                    final_subgroup_details[subgroup_id] = {'nodes': age_subgroup_indices, 'key': group_key + ('age_group',)} # Store details if needed
                    for node_idx in age_subgroup_indices:
                        node_index_to_subgroup_id[node_idx] = subgroup_id
                    current_subgroup_id_counter += 1
        logger.info(f"Generated {current_subgroup_id_counter} final subgroups for map.")
        # --- End Map Generation --- 
        
        # Step 1: Use initial broad grouping for chunking (Race/Gender)
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
                                
                                # Apply filtering to chunk edges - PASS THE MAP
                                chunk_edges = self._apply_attribute_filtering(nodes, chunk_edges, 
                                                                           f"{race}-{gender} chunk {chunk_idx}",
                                                                           node_index_to_subgroup_id)
                                all_edges.extend(chunk_edges)
                    else:
                        # Small enough to process directly
                        if subgroup_size > 1:
                            subgroup_edges = list(combinations(subgroup, 2))
                            # Apply filtering - PASS THE MAP
                            subgroup_edges = self._apply_attribute_filtering(nodes, subgroup_edges, 
                                                                           f"{race}-{gender} subgroup {subgroup_idx}",
                                                                           node_index_to_subgroup_id)
                            all_edges.extend(subgroup_edges)
            else:
                # Small enough group to process directly
                if group_size > 1:
                    group_edges = list(combinations(group_indices, 2))
                    # Apply filtering - PASS THE MAP
                    group_edges = self._apply_attribute_filtering(nodes, group_edges, f"{race}-{gender}",
                                                               node_index_to_subgroup_id)
                    all_edges.extend(group_edges)
        
        logger.info(f"Generated {len(all_edges)} edges across all groups")
        
        # Step 4: Create edge objects - PASS THE MAP
        return self._create_graph_from_edges(nodes, all_edges, split_name, node_index_to_subgroup_id)
    
    def _apply_attribute_filtering(self, nodes, edges, group_name, node_index_to_subgroup_id):
        """
        Apply attribute filtering to a set of edges
        
        Args:
            nodes: List of all nodes
            edges: List of edges to filter
            group_name: Name of the group for logging
            node_index_to_subgroup_id: Dictionary mapping node indices to subgroup IDs
            
        Returns:
            Filtered list of edges
        """
        logger.info(f"Filtering {len(edges)} edges for group {group_name}")

        # Always apply filtering based on the threshold
        quality_edges = self._filter_edges(nodes, edges, 'quality', self.hyperparameters["quality_threshold"])
        logger.info(f"Edges remaining after quality filtering: {len(quality_edges)}")
        edges = quality_edges
        if not edges:
            logger.info("No edges remaining after quality filtering.")
            return []

        # Always apply filtering based on the threshold
        symmetry_edges = self._filter_edges(nodes, edges, 'symmetry', self.hyperparameters["symmetry_threshold"])
        logger.info(f"Edges remaining after symmetry filtering: {len(symmetry_edges)}")
        edges = symmetry_edges
        if not edges:
            logger.info("No edges remaining after symmetry filtering.")
            return []

        # Always apply filtering based on the threshold
        embedding_edges = self._filter_edges(nodes, edges, 'embedding', self.hyperparameters["embedding_threshold"])
        logger.info(f"Edges remaining after embedding filtering: {len(embedding_edges)}")
        edges = embedding_edges
        if not edges:
            logger.info("No edges remaining after embedding filtering.")
            return []

        # This log should now always reflect the final count after all filters
        logger.info(f"Edges remaining after filtering: {len(edges)}") 
        return edges
    
    def _create_graph_from_edges(self, nodes, edges, split_name, node_index_to_subgroup_id):
        """
        Create a graph from a list of edges in batches
        
        Args:
            nodes: List of nodes
            edges: List of (i, j) tuples for edges
            split_name: Name of the split for logging
            node_index_to_subgroup_id: Dictionary mapping node indices to subgroup IDs
            
        Returns:
            HyperGraph object
        """
        logger.info(f"Creating graph with {len(nodes)} nodes and {len(edges)} potential edges")
        
        # --- IMPORTANT: Reset edges on existing nodes before adding new ones ---
        for node in nodes:
            node.edges = [] # Clear any edges from previous grid search iterations
        # --- End Reset ---

        # Prepare nodes list for quick lookup
        all_nodes = list(nodes)
        edge_objects = [] # List to store the created edge objects
        connected_nodes = set() # Initialize set to track connected node indices
        
        # Use a set to track pairs for which an edge has already been created
        # Store pairs as sorted tuples to handle (i, j) and (j, i) as the same edge
        added_pairs = set()

        fallback_connections_used = 0
        for i, j in tqdm(edges, desc=f"Creating {split_name} edges", unit=" edges", disable=self.hyperparameters["silent_mode"]):
            try:
                node_i = all_nodes[i]
                node_j = all_nodes[j]
                pair = tuple(sorted((i, j))) # Use original indices for pair tracking
                
                # Only create edge if this pair hasn't been added yet
                if pair not in added_pairs:
                    # Create a single edge object for the pair
                    edge_label = f"{node_i.get_label()}-{node_j.get_label()}"
                    edge = self.edge_class(node_i, node_j, edge_label) # node1, node2, x
                    
                    # Add the edge to both nodes
                    node_i.add_edge(edge)
                    node_j.add_edge(edge)
                    
                    # Mark this pair as added
                    added_pairs.add(pair)
                    edge_objects.append(edge)
                    
                    # Correctly track connected nodes
                    connected_nodes.add(i)
                    connected_nodes.add(j)
                # else: # This case is handled by the added_pairs check
                    # This indicates a duplicate in filtered_edges, which shouldn't happen ideally
                    # but this check prevents it from inflating the degree count.
                    # logger.debug(f"Skipping duplicate edge creation for pair ({i}, {j})")
                    # pass 
                    
            except IndexError:
                logger.warning(f"Invalid node index encountered in edge list: ({i}, {j}). Skipping edge.")
            except Exception as e:
                 logger.error(f"Error processing edge ({i}, {j}): {e}")
                 
        logger.info(f"Created {len(edge_objects)} unique edge objects after filtering duplicates.")

        # Handle disconnected nodes
        disconnected_nodes = set(range(len(nodes))) - connected_nodes
        if disconnected_nodes:
            logger.info(f"Found {len(disconnected_nodes)} disconnected nodes, attempting to connect them within subgroups...")
            fallback_violations = 0
            fallback_warnings_logged = 0
            max_fallback_warnings = 5

            # Create a map from subgroup_id to list of node indices in that subgroup for efficient lookup
            subgroup_to_nodes = defaultdict(list)
            if node_index_to_subgroup_id: # Ensure map exists
                for idx, sub_id in node_index_to_subgroup_id.items():
                    subgroup_to_nodes[sub_id].append(idx)
            else:
                logger.warning("node_index_to_subgroup_id map is missing or empty during fallback connection!")

            edge_list_with_fallback = []
            edge_set = set()
            node_degrees = {i: 0 for i in range(len(nodes))}
            for node_idx in tqdm(disconnected_nodes, desc="Connecting isolated nodes", disable=self.hyperparameters["silent_mode"]):
                node = all_nodes[node_idx]
                subgroup_id = node_index_to_subgroup_id.get(node_idx) if node_index_to_subgroup_id else None

                potential_partners = []
                j = -1 # Initialize partner index
                
                # --- Try connecting within the same subgroup --- 
                if subgroup_id is not None and subgroup_to_nodes:
                    # Find potential partners within the same subgroup, excluding self
                    same_subgroup_indices = [idx for idx in subgroup_to_nodes.get(subgroup_id, []) if idx != node_idx]
                    
                    if same_subgroup_indices:
                        # Prefer connecting to already connected nodes within the subgroup
                        connected_in_subgroup = [idx for idx in same_subgroup_indices if idx in connected_nodes]
                        if connected_in_subgroup:
                            potential_partners = connected_in_subgroup
                        else:
                            # If no connected nodes in subgroup, connect to any other node in the subgroup
                            potential_partners = same_subgroup_indices
                
                # If partners found within subgroup, choose one randomly
                if potential_partners:
                    partner_node_idx = random.choice(potential_partners)
                    logger.debug(f"Fallback: Connecting disconnected node {node_idx} to disconnected node {partner_node_idx} in same subgroup {subgroup_id}")
                else:
                    # If no disconnected nodes in the subgroup, connect to ANY node in the same subgroup (excluding self)
                    nodes_in_same_subgroup = [idx for idx, sg_id in node_index_to_subgroup_id.items() if sg_id == subgroup_id and idx != node_idx]
                    if nodes_in_same_subgroup: # Check if there are other nodes in the subgroup
                        partner_node_idx = random.choice(nodes_in_same_subgroup)
                        logger.debug(f"Fallback: Connecting disconnected node {node_idx} to connected node {partner_node_idx} in same subgroup {subgroup_id}")
                    else:
                        # This case should ideally not happen if subgroups have more than one node, but log if it does.
                        logger.warning(f"Fallback: Node {node_idx} is alone in subgroup {subgroup_id} and cannot be connected within the subgroup.")
                        fallback_connections_used += 1
                        continue # Skip creating an edge if no partner is found within the subgroup

                # Add edge and update degrees
                new_edge = tuple(sorted((node_idx, partner_node_idx)))
                if new_edge not in edge_set:
                    edge_list_with_fallback.append(new_edge)
                    edge_set.add(new_edge)
                    node_degrees[node_idx] += 1
                    node_degrees[partner_node_idx] += 1
                    fallback_connections_used += 1
                else:
                    logger.warning(f"Fallback: Tried to add duplicate edge {new_edge} for node {node_idx}")

            if fallback_connections_used > 0:
                logger.info(f"Fallback mechanism used for {fallback_connections_used} nodes.")

            # --- Convert edge tuples to Edge objects ---
            for i, j in edge_list_with_fallback:
                node_i = all_nodes[i]
                node_j = all_nodes[j]
                edge_label = f"{node_i.get_label()}-{node_j.get_label()}"
                edge = self.edge_class(node_i, node_j, edge_label)
                node_i.add_edge(edge)
                node_j.add_edge(edge)
                edge_objects.append(edge)

        # ----- Final Validation Step ----- #
        if node_index_to_subgroup_id: # Only validate if map exists
             logger.info("Starting final edge validation for subgroup constraints...")
             violation_count = 0
             warning_limit = 10
             validation_errors = 0
             # Create node-to-index map for efficient lookup - assuming Node objects are hashable
             try:
                  node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
                  lookup_possible = True
             except TypeError:
                  logger.warning("Node objects are not hashable, cannot create efficient node_to_index map for validation.")
                  lookup_possible = False
                  
             for edge in tqdm(edge_objects, desc="Validating edges", disable=self.hyperparameters["silent_mode"]):
                 try:
                     node1, node2 = edge.get_nodes()
                     idx1, idx2 = -1, -1
                     
                     # Find original indices
                     if lookup_possible:
                          idx1 = node_to_index.get(node1, -1)
                          idx2 = node_to_index.get(node2, -1)
                     else: # Fallback to slower list search if nodes aren't hashable
                          try:
                               idx1 = all_nodes.index(node1)
                               idx2 = all_nodes.index(node2)
                          except ValueError:
                               pass # Indices remain -1
                               
                     if idx1 == -1 or idx2 == -1:
                         node1_id = getattr(node1, 'id', 'UNKNOWN') # Attempt to get an ID
                         node2_id = getattr(node2, 'id', 'UNKNOWN')
                         logger.error(f"Could not find node instance {node1_id} or {node2_id} in all_nodes list during validation.")
                         validation_errors += 1
                         continue

                     subgroup1 = node_index_to_subgroup_id.get(idx1)
                     subgroup2 = node_index_to_subgroup_id.get(idx2)

                     if subgroup1 is None or subgroup2 is None:
                         logger.error(f"Missing subgroup ID for node {idx1} ({subgroup1}) or node {idx2} ({subgroup2}) during validation.")
                         validation_errors += 1
                         continue

                     if subgroup1 != subgroup2:
                         violation_count += 1
                         if violation_count <= warning_limit:
                             logger.warning(f"Subgroup Violation ({violation_count}): Edge connects node {idx1} (subgroup {subgroup1}) and node {idx2} (subgroup {subgroup2})")
                         elif violation_count == warning_limit + 1:
                             logger.warning("Further subgroup violation warnings suppressed.")
                             
                 except Exception as e:
                     logger.error(f"Error during edge validation: {e}", exc_info=True) # Log stack trace
                     validation_errors += 1
                     if validation_errors > 100: # Stop validation if too many errors occur
                         logger.error("Aborting validation due to excessive errors.")
                         break
                         
             if validation_errors > 0:
                 logger.error(f"Encountered {validation_errors} errors during subgroup validation process.")
             logger.info(f"Subgroup validation complete. Found {violation_count} edges potentially violating subgroup constraints.")
        else:
             logger.info("Skipping subgroup validation as node_index_to_subgroup_id map was not provided.")

        # Create the HyperGraph object
        graph = HyperGraph(nodes=all_nodes) 
        return graph
        
    def _build_graph_standard(self, nodes, split_name):
        """
        Build a graph for a specific split using hierarchical construction (standard approach)
        
        Args:
            nodes: List of nodes for this split
            split_name: Name of the split for logging
            
        Returns:
            Tuple[HyperGraph, int]: The constructed graph and the number of edges 
                                   remaining after attribute filtering (before fallback).
        """
        if not nodes:
            logger.info(f"No nodes for {split_name} split, returning empty graph")
            return HyperGraph([]), 0
        
        logger.info(f"\nBuilding graph for {split_name} split ({len(nodes)} nodes)...")
        
        # Step 1: Group nodes by race and gender
        race_gender_groups = self._group_by_categorical(nodes)
        node_index_to_subgroup_id = {} # Initialize the map
        
        # Step 2: Initialize edges with all connections within each race-gender group (and age subgroups)
        all_edges = []
        # Iterate through items to get the key (race, gender) and value (indices)
        for group_key, group_indices in race_gender_groups.items():
            race, gender = group_key # Unpack the key

            # For very large groups, consider age-based subgrouping
            # Threshold can be tuned based on performance/memory
            if len(group_indices) > self.hyperparameters.get("age_split_threshold", 1000):
                logger.debug(f"Creating age subgroups for {group_key} (size {len(group_indices)})")
                age_subgroups = self._create_age_subgroups(nodes, group_indices)
                logger.info(f"Group '{group_key}' split into {len(age_subgroups)} age subgroups")
                
                # Add these more specific subgroups to the final list
                for age_subgroup_indices in age_subgroups:
                    # Assign a unique subgroup ID for each age subgroup
                    # Using a tuple including the age group identifier (e.g., its first node index for simplicity)
                    subgroup_id = (race, gender, f"age_group_{age_subgroup_indices[0]}")
                    for node_idx in age_subgroup_indices:
                        node_index_to_subgroup_id[node_idx] = subgroup_id
            else:
                # Assign subgroup ID (race, gender) for smaller groups
                subgroup_id = (race, gender)
                for node_idx in group_indices:
                    node_index_to_subgroup_id[node_idx] = subgroup_id
                    
            # --- REMOVED OLD EDGE CREATION HERE ---
            # # Connect all nodes within the race-gender group
            # if len(group_indices) > 1: # Ensure there's more than one node to connect
            #     all_edges.extend(combinations(group_indices, 2))
        
        logger.info(f"Generated subgroup mapping for {len(node_index_to_subgroup_id)} nodes across {len(set(node_index_to_subgroup_id.values()))} subgroups.")

        # Step 2.5: Create initial edges STRICTLY within final subgroups
        all_edges = []
        nodes_by_subgroup = defaultdict(list)
        for node_idx, subgroup_id in node_index_to_subgroup_id.items():
            nodes_by_subgroup[subgroup_id].append(node_idx)

        # DEBUG: Log largest subgroup size
        if nodes_by_subgroup:
            max_subgroup_size = max(len(indices) for indices in nodes_by_subgroup.values())
            logger.info(f"Largest subgroup size: {max_subgroup_size}")
        else:
            logger.info("No subgroups found.")

        for subgroup_id, nodes_in_subgroup in nodes_by_subgroup.items():
            if len(nodes_in_subgroup) > 1:
                all_edges.extend(combinations(nodes_in_subgroup, 2))

        # DEBUG: Log initial edge count explicitly
        logger.info(f"Created {len(all_edges)} initial edges strictly within {len(nodes_by_subgroup)} subgroups.")
        initial_edge_count = len(all_edges)
        logger.info(f"Total initial edges before filtering: {initial_edge_count}")
        
        # Step 3: Apply attribute filtering - PASS THE MAP
        # Ensure _apply_attribute_filtering accepts and potentially uses the map if needed
        filtered_edges = self._apply_attribute_filtering(nodes, all_edges, split_name, node_index_to_subgroup_id)
        
        # DEBUG: Log edge count immediately before graph creation
        logger.info(f"Passing {len(filtered_edges)} edges to _create_graph_from_edges")
        # Store the count of edges after filtering
        num_edges_after_filter = len(filtered_edges)
        
        # Step 4 & 5: Create graph from edges - PASS THE MAP
        # _create_graph_from_edges uses the map for fallback and validation
        graph = self._create_graph_from_edges(nodes, filtered_edges, split_name, node_index_to_subgroup_id)
        
        return graph, num_edges_after_filter
    
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
            graph, _ = self._build_graph_standard(nodes, split_name)
            return graph

    def get_graph(self, split='train'):
        """
        Get the graph for a specific split
        
        Args:
            split: Name of the split to retrieve ('train', 'val', 'test')
            
        Returns:
            HyperGraph object
        """
        # Load the graph if not already loaded
        if not hasattr(self, 'graphs'):
            self.load()
        
        # Return the graph for the specified split
        return self.graphs[split]

    def get_graph(self, split='train'):
        """
        Get the graph for a specific split
        
        Args:
            split: Name of the split to retrieve ('train', 'val', 'test')
            
        Returns:
            HyperGraph object
        """
        graph, _ = self._build_graph_standard(nodes, split)
        return graph
