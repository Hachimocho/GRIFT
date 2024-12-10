import random
from itertools import combinations
from tqdm.auto import tqdm
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from dataloaders.Dataloader import Dataloader
from graphs.HyperGraph import HyperGraph
from collections import defaultdict
from utils.visualize import visualize_graph
import networkx as nx

class ClusteredDeepfakeDataloader(Dataloader):
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

    def load(self):
        start = time.time()
        all_nodes = []
        
        # Load datasets
        for dataset in self.datasets:
            print(f"Loading dataset: {dataset.__class__.__name__}")
            dataset_nodes = dataset.load()
            if self.hyperparameters["test_mode"]:
                dataset_nodes = dataset_nodes[:100]
            all_nodes.extend(dataset_nodes)
        
        print(f"Total nodes loaded: {len(all_nodes)}")
        
        # Convert attributes to binary feature matrix for fast comparison
        feature_matrix, _ = self._create_attribute_matrix(all_nodes)
        
        if self.hyperparameters["use_lsh"]:
            print("Using LSH for faster matching...")
            # Compute LSH buckets
            buckets = self._compute_lsh_buckets(
                feature_matrix,
                n_bands=self.hyperparameters["lsh_bands"],
                band_size=self.hyperparameters["lsh_band_size"]
            )
            
            # Process nodes using LSH buckets
            matched_nodes = set()
            for i in tqdm(range(len(all_nodes)), desc="Processing nodes"):
                if i in matched_nodes and len(matched_nodes) > len(all_nodes) * 0.8:
                    continue
                    
                # Find candidate nodes through LSH buckets
                node_features = feature_matrix[i]
                candidates = set()
                
                # Get candidates from same LSH buckets
                for band in range(self.hyperparameters["lsh_bands"]):
                    sig = np.packbits(node_features[:self.hyperparameters["lsh_band_size"]])
                    bucket_key = (band, sig.tobytes())
                    candidates.update(buckets.get(bucket_key, []))
                
                # Filter candidates to those after current node
                candidates = [j for j in candidates if j > i]
                
                if candidates:
                    # Compute similarities vectorized
                    candidate_features = feature_matrix[candidates]
                    similarities = (node_features @ candidate_features.T) / (np.sum(node_features) + np.sum(candidate_features, axis=1))
                    
                    # Find matches above threshold
                    threshold = all_nodes[i].threshold
                    matches_mask = similarities >= threshold/100
                    
                    # Create edges for matches
                    for idx, is_match in enumerate(matches_mask):
                        if is_match:
                            j = candidates[idx]
                            edge = self.edge_class(all_nodes[i], all_nodes[j])
                            all_nodes[i].add_edge(edge)
                            all_nodes[j].add_edge(edge)
                            matched_nodes.add(i)
                            matched_nodes.add(j)
        else:
            # Fallback to original matching method
            print("Using original matching method...")
            matched_nodes = set()
            for i, j in tqdm(combinations(range(len(all_nodes)), 2), 
                           total=sum(1 for _ in combinations(range(len(all_nodes)), 2)), 
                           desc="Building graph..."):
                if all_nodes[i].match(all_nodes[j]):
                    edge = self.edge_class(all_nodes[i], all_nodes[j])
                    all_nodes[i].add_edge(edge)
                    all_nodes[j].add_edge(edge)
                    matched_nodes.add(i)
                    matched_nodes.add(j)
        
        # Add semi-random edges to disconnected nodes
        unmatched_nodes = set(range(len(all_nodes))) - matched_nodes
        if unmatched_nodes:
            print(f"Fixing {len(unmatched_nodes)} disconnected nodes...")
            for i in tqdm(unmatched_nodes, desc="Fixing disconnected nodes"):
                if matched_nodes:
                    j = random.choice(list(matched_nodes))
                    edge = self.edge_class(all_nodes[i], all_nodes[j])
                    all_nodes[i].add_edge(edge)
                    all_nodes[j].add_edge(edge)
                    matched_nodes.add(i)
                else:
                    # If no matched nodes exist, connect to a random unmatched node
                    other_unmatched = list(unmatched_nodes - {i})
                    if other_unmatched:
                        j = random.choice(other_unmatched)
                        edge = self.edge_class(all_nodes[i], all_nodes[j])
                        all_nodes[i].add_edge(edge)
                        all_nodes[j].add_edge(edge)
                        matched_nodes.add(i)
                        matched_nodes.add(j)
                        unmatched_nodes.remove(j)
        
        graph = HyperGraph(all_nodes)
        print(f"Graph generation finished in {time.time() - start:.2f}s")
        print(f"Total nodes: {len(graph.nodes)}")
        print(f"Total edges: {len(graph.edges)}")
        
        if self.hyperparameters["visualize"]:
            visualize_graph(graph, show=self.hyperparameters["show_viz"])
        
        return graph