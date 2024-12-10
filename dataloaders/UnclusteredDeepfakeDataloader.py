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
        
        # Split nodes based on their subset attribute first
        train_nodes = []
        val_nodes = []
        test_nodes = []
        
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