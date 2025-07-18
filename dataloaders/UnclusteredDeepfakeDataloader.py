import random
from itertools import combinations
from tqdm.auto import tqdm
import time
import numpy as np
from dataloaders.Dataloader import Dataloader
from graphs.HyperGraph import HyperGraph
from collections import defaultdict
from utils.visualize import visualize_graph
import networkx as nx
import pandas as pd
import os
import logging

# Configure logging to file
def setup_logger(level=logging.INFO, log_to_console=False):
    """Configure logger to write to file and optionally console"""
    logger = logging.getLogger('UnclusteredDataloader')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler (always enabled)
    file_handler = logging.FileHandler('logs/unclustered_dataloader.log')
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

class UnclusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes", "unclustered"]
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
        Initialize the unclustered dataloader
        
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
        
        # For standard cases, use vectorized filtering
        if len(edges) > 1000: # Use vectorized for non-embedding or when LSH conditions not met
            logger.info(f"Using vectorized filtering for {attribute_type} with {len(edges)} edges")
            return self._filter_edges_vectorized(nodes, edges, attribute_type, threshold)

        # Fall back to the original method for small edge sets
        logger.info(f"Using standard (iterative) filtering for {attribute_type} with {len(edges)} edges")
        filtered_edges = []

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



    def load(self, preloaded_nodes=None):
        """
        Load datasets and create unclustered graph structure
        
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
    
    def _apply_attribute_filtering(self, nodes, edges, group_name, node_index_to_subgroup_id=None):
        """
        Apply attribute filtering to a set of edges
        
        Args:
            nodes: List of all nodes
            edges: List of edges to filter
            group_name: Name of the group for logging
            node_index_to_subgroup_id: Dictionary mapping node indices to subgroup IDs (not used in unclustered)
            
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
    
    def _create_graph_from_edges(self, nodes, edges, split_name, node_index_to_subgroup_id=None):
        """
        Create a graph from a list of edges in batches
        
        Args:
            nodes: List of nodes
            edges: List of (i, j) tuples for edges
            split_name: Name of the split for logging
            node_index_to_subgroup_id: Dictionary mapping node indices to subgroup IDs (not used in unclustered)
            
        Returns:
            HyperGraph object
        """
        logger.info(f"Creating unclustered graph with {len(nodes)} nodes and {len(edges)} potential edges")
        
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
                    
            except IndexError:
                logger.warning(f"Invalid node index encountered in edge list: ({i}, {j}). Skipping edge.")
            except Exception as e:
                 logger.error(f"Error processing edge ({i}, {j}): {e}")
                 
        logger.info(f"Created {len(edge_objects)} unique edge objects after filtering duplicates.")

        # Handle disconnected nodes - connect them randomly to any other node
        disconnected_nodes = set(range(len(nodes))) - connected_nodes
        if disconnected_nodes:
            logger.info(f"Found {len(disconnected_nodes)} disconnected nodes, connecting them randomly...")
            
            edge_list_with_fallback = []
            edge_set = set()
            node_degrees = {i: 0 for i in range(len(nodes))}
            
            for node_idx in tqdm(disconnected_nodes, desc="Connecting isolated nodes", disable=self.hyperparameters["silent_mode"]):
                # Connect to any other node randomly
                other_nodes = [i for i in range(len(nodes)) if i != node_idx]
                if other_nodes:
                    partner_node_idx = random.choice(other_nodes)
                    
                    # Add edge and update degrees
                    new_edge = tuple(sorted((node_idx, partner_node_idx)))
                    if new_edge not in edge_set:
                        edge_list_with_fallback.append(new_edge)
                        edge_set.add(new_edge)
                        node_degrees[node_idx] += 1
                        node_degrees[partner_node_idx] += 1

            # --- Convert edge tuples to Edge objects ---
            for i, j in edge_list_with_fallback:
                node_i = all_nodes[i]
                node_j = all_nodes[j]
                edge_label = f"{node_i.get_label()}-{node_j.get_label()}"
                edge = self.edge_class(node_i, node_j, edge_label)
                node_i.add_edge(edge)
                node_j.add_edge(edge)
                edge_objects.append(edge)

        # Create the HyperGraph object
        graph = HyperGraph(nodes=all_nodes)
        # Assign Louvain subclusters (if available)
        try:
            graph.assign_louvain_subclusters()
        except Exception as e:
            logger.warning(f"Louvain subcluster assignment failed: {e}")
        return graph
        
    def _build_graph_standard(self, nodes, split_name):
        """
        Build a graph for a specific split using unclustered construction (standard approach)
        
        Args:
            nodes: List of nodes for this split
            split_name: Name of the split for logging
            
        Returns:
            Tuple[HyperGraph, int]: The constructed graph and the number of edges 
                                   remaining after attribute filtering.
        """
        if not nodes:
            logger.info(f"No nodes for {split_name} split, returning empty graph")
            return HyperGraph([]), 0
        
        logger.info(f"\nBuilding unclustered graph for {split_name} split ({len(nodes)} nodes)...")
        
        # Generate all possible edge pairs (unclustered approach)
        n_nodes = len(nodes)
        all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        
        logger.info(f"Created {len(all_edges)} initial edges (all pairs)")
        initial_edge_count = len(all_edges)
        logger.info(f"Total initial edges before filtering: {initial_edge_count}")
        
        # Apply attribute filtering
        filtered_edges = self._apply_attribute_filtering(nodes, all_edges, split_name)
        
        # DEBUG: Log edge count immediately before graph creation
        logger.info(f"Passing {len(filtered_edges)} edges to _create_graph_from_edges")
        # Store the count of edges after filtering
        num_edges_after_filter = len(filtered_edges)
        
        # Create graph from edges
        graph = self._create_graph_from_edges(nodes, filtered_edges, split_name)
        
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
        # For unclustered, always use standard approach since we don't have clustering complexity
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