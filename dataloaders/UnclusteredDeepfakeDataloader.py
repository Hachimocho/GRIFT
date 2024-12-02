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

class UnclusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = {
        "use_lsh": True,  # Whether to use Locality-Sensitive Hashing for faster matching
        "test_mode": True,  # If True, only loads first 100 nodes for testing
        "lsh_bands": 50,  # Number of bands for LSH
        "lsh_band_size": 2,  # Size of each LSH band
        "visualize": True,  # Whether to create and save a visualization
        "show_viz": False,  # Whether to display the visualization
    }

    def _create_attribute_matrix(self, nodes):
        """Convert node attributes to a binary feature matrix for fast comparison"""
        # Create a mapping of all unique attributes
        all_attributes = set()
        for node in nodes:
            all_attributes.update(node.attributes.keys())
        
        attr_to_idx = {attr: idx for idx, attr in enumerate(all_attributes)}
        n_features = len(attr_to_idx)
        
        # Create binary feature matrix
        feature_matrix = np.zeros((len(nodes), n_features), dtype=np.int8)
        for i, node in enumerate(nodes):
            for attr in node.attributes:
                feature_matrix[i, attr_to_idx[attr]] = 1
                
        return feature_matrix, attr_to_idx

    def _compute_lsh_buckets(self, feature_matrix, n_bands=50, band_size=2):
        """Use LSH to group similar nodes into buckets"""
        n_nodes, n_features = feature_matrix.shape
        buckets = defaultdict(list)
        
        # Generate random permutation matrices for LSH
        np.random.seed(42)  # For reproducibility
        for band in range(n_bands):
            # Generate band signature
            band_start = band * band_size
            band_end = (band + 1) * band_size
            if band_end > n_features:
                break
                
            band_features = feature_matrix[:, band_start:band_end]
            signatures = np.packbits(band_features, axis=1)
            
            # Add nodes to buckets based on signatures
            for i, sig in enumerate(signatures):
                bucket_key = (band, sig.tobytes())
                buckets[bucket_key].append(i)
        
        return buckets

    def process_node_batch(self, args):
        start_idx, end_idx, node_list, feature_matrix, chunk_id = args
        edges = []
        matches = set()
        
        # Get nodes for this chunk
        chunk_features = feature_matrix[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        if self.hyperparameters["use_lsh"]:
            # Use LSH for faster matching
            buckets = self._compute_lsh_buckets(
                feature_matrix, 
                n_bands=self.hyperparameters["lsh_bands"],
                band_size=self.hyperparameters["lsh_band_size"]
            )
            
            # Process each node in the chunk
            for i in range(chunk_size):
                global_idx = start_idx + i
                if global_idx in matches and len(matches) > chunk_size * 0.8:
                    continue
                
                # Find candidate nodes through LSH buckets
                node_features = chunk_features[i]
                candidates = set()
                
                # Get candidates from same LSH buckets
                for band in range(self.hyperparameters["lsh_bands"]):
                    sig = np.packbits(node_features[:self.hyperparameters["lsh_band_size"]])
                    bucket_key = (band, sig.tobytes())
                    candidates.update(buckets.get(bucket_key, []))
                
                # Filter candidates to those after current node
                candidates = [j for j in candidates if j > global_idx]
                
                if candidates:
                    # Compute similarities vectorized
                    candidate_features = feature_matrix[candidates]
                    similarities = (node_features @ candidate_features.T) / (np.sum(node_features) + np.sum(candidate_features, axis=1))
                    
                    # Find matches above threshold
                    threshold = node_list[global_idx].threshold
                    matches_mask = similarities >= threshold/100
                    
                    # Add edges for matches
                    for idx, is_match in enumerate(matches_mask):
                        if is_match:
                            j = candidates[idx]
                            edges.append((global_idx, j))
                            matches.add(global_idx)
                            matches.add(j)
        else:
            # Fallback to direct comparison with vectorization
            for i in range(chunk_size):
                global_idx = start_idx + i
                if global_idx in matches and len(matches) > chunk_size * 0.8:
                    continue
                
                # Compare with all remaining nodes
                node_features = chunk_features[i]
                remaining_features = feature_matrix[global_idx + 1:]
                if len(remaining_features) > 0:
                    similarities = (node_features @ remaining_features.T) / (np.sum(node_features) + np.sum(remaining_features, axis=1))
                    
                    # Find matches above threshold
                    threshold = node_list[global_idx].threshold
                    matches_mask = similarities >= threshold/100
                    
                    # Add edges for matches
                    for idx, is_match in enumerate(matches_mask):
                        if is_match:
                            j = global_idx + 1 + idx
                            edges.append((global_idx, j))
                            matches.add(global_idx)
                            matches.add(j)
        
        return edges, matches

    def load(self):
        start = time.time()
        node_list = []
        
        # Create NetworkX graph alongside main graph
        nx_graph = nx.Graph()
        
        # Load datasets
        for dataset in self.datasets:
            node_list.extend(dataset.load())
        
        # Optionally limit to first 100 nodes for testing
        if self.hyperparameters["test_mode"]:
            print("Running in test mode - using only first 100 nodes")
            node_list = node_list[:100]
        
        n_nodes = len(node_list)
        print(f"# of nodes: {n_nodes}")
        
        # Add all nodes to NetworkX graph
        nx_graph.add_nodes_from(range(n_nodes))
        
        # Create feature matrix for fast comparison
        print("Creating feature matrix...")
        feature_matrix, attr_to_idx = self._create_attribute_matrix(node_list)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Automatically disable LSH for low-dimensional data
        if feature_matrix.shape[1] < 10 and self.hyperparameters["use_lsh"]:
            print("Automatically disabled LSH due to low feature dimension")
            self.hyperparameters["use_lsh"] = False
        
        # Split work into chunks for parallel processing
        num_processes = min(8, cpu_count())
        chunk_size = max(1000, n_nodes // (num_processes * 4))
        chunks = [(i, min(i + chunk_size, n_nodes), node_list, feature_matrix, idx) 
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
        
        graph = HyperGraph(node_list)
        
        # Create visualization if enabled
        if self.hyperparameters["visualize"]:
            # Create attribute labels
            attr_labels = {}
            for i, node in enumerate(node_list):
                attrs = sorted(node.attributes.keys())
                attr_labels[i] = f"({', '.join(attrs)})"
            
            # Create random splits for demonstration (you can modify this)
            if self.hyperparameters["test_mode"]:
                splits = {
                    'train': list(range(0, 70)),
                    'val': list(range(70, 85)),
                    'test': list(range(85, len(node_list)))
                }
            else:
                # For full dataset, use percentage-based splits
                n = len(node_list)
                splits = {
                    'train': list(range(0, int(0.7 * n))),
                    'val': list(range(int(0.7 * n), int(0.85 * n))),
                    'test': list(range(int(0.85 * n), n))
                }
            
            visualize_graph(
                graph,
                nx_graph=nx_graph,  # Pass the pre-built NetworkX graph
                splits=splits,
                attribute_labels=attr_labels,
                title=f"Graph Visualization ({'Test Mode' if self.hyperparameters['test_mode'] else 'Full Dataset'})",
                show=self.hyperparameters["show_viz"]
            )
        
        print(f"Graph building took {time.time() - start:.2f} seconds")
        return graph