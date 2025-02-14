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
from utils.additional_attributes import compute_embedding_similarity

class ClusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = {
        "use_lsh": True,  # Whether to use Locality-Sensitive Hashing for faster matching
        "test_mode": True,  # If True, only loads first 100 nodes for testing
        "lsh_bands": 50,  # Number of bands for LSH
        "lsh_band_size": 2,  # Size of each LSH band
        "visualize": False,  # Whether to create and save a visualization
        "show_viz": False,  # Whether to display the visualization
        "min_face_similarity": 0.85,  # Minimum cosine similarity between face embeddings to create an edge
        "face_similarity_weight": 0.7,  # Weight for face embedding similarity (0-1)
        "attribute_weight": 0.3,  # Weight for attribute matching (0-1)
        "min_combined_score": 0.8,  # Minimum weighted combination score to create an edge
        "attribute_importance": {  # Importance weights for different attributes
            # Demographics
            "Gender": 1.0,
            "Race": 0.8,
            "Age": 0.6,
            
            # Image Quality
            "blur_score": 0.4,
            "brightness": 0.3,
            "contrast": 0.3,
            "compression_score": 0.4,
            
            # Face Alignment
            "face_yaw": 0.5,
            "face_pitch": 0.5,
            "face_roll": 0.5,
            
            # Facial Symmetry
            "eye_ratio": 0.7,
            "mouth_ratio": 0.6,
            "nose_ratio": 0.6,
            "overall_symmetry": 0.7,
            
            # Emotions
            "emotion_angry": 0.4,
            "emotion_disgust": 0.4,
            "emotion_fear": 0.4,
            "emotion_happy": 0.4,
            "emotion_sad": 0.4,
            "emotion_surprise": 0.4,
            "emotion_neutral": 0.4,
            
            # Facial Attributes
            "hair_color": 0.5,
            "eye_color": 0.5,
            "has_beard": 0.3,
            "has_mustache": 0.3,
            "wearing_glasses": 0.3,
            "wearing_sunglasses": 0.3
        }
    }

    def _create_attribute_matrix(self, nodes):
        """Convert node attributes to a binary feature matrix for fast comparison"""
        # Create a mapping of all unique attributes
        all_attributes = set()
        for node in nodes:
            for attr in node.attributes:
                if attr != 'face_embedding':  # Skip embedding, handle separately
                    all_attributes.add(attr)
        
        attr_to_idx = {attr: idx for idx, attr in enumerate(all_attributes)}
        n_features = len(attr_to_idx)
        
        # Create binary feature matrix
        feature_matrix = np.zeros((len(nodes), n_features), dtype=np.int8)
        embedding_matrix = np.zeros((len(nodes), 512))  # Face embeddings matrix
        
        for i, node in enumerate(nodes):
            # Handle regular attributes
            for attr in node.attributes:
                if attr != 'face_embedding' and attr in attr_to_idx:
                    feature_matrix[i, attr_to_idx[attr]] = 1
            
            # Handle face embedding
            if 'face_embedding' in node.attributes:
                embedding_matrix[i] = np.array(node.attributes['face_embedding'])
                
        return feature_matrix, embedding_matrix, attr_to_idx

    def _compute_lsh_buckets(self, feature_matrix, embedding_matrix, n_bands=50, band_size=2):
        """Use LSH to group similar nodes into buckets, considering both attributes and face embeddings"""
        n_nodes = feature_matrix.shape[0]
        buckets = defaultdict(list)
        
        # Generate random permutation matrices for LSH
        np.random.seed(42)  # For reproducibility
        
        # Process regular attributes
        for band in range(n_bands):
            band_start = band * band_size
            band_end = (band + 1) * band_size
            if band_end > feature_matrix.shape[1]:
                break
                
            band_features = feature_matrix[:, band_start:band_end]
            signatures = np.packbits(band_features, axis=1)
            
            for i, sig in enumerate(signatures):
                bucket_key = (band, sig.tobytes())
                buckets[bucket_key].append(i)
        
        # Process face embeddings using angular LSH
        projection_vectors = np.random.randn(n_bands, 512)  # Random projection vectors
        for band, proj_vec in enumerate(projection_vectors):
            # Project embeddings onto random vector
            projections = np.dot(embedding_matrix, proj_vec)
            # Create buckets based on sign of projection
            for i, proj in enumerate(projections):
                bucket_key = (f'face_{band}', proj > 0)
                buckets[bucket_key].append(i)
        
        return buckets

    def _calculate_attribute_similarity(self, node1, node2):
        """Calculate weighted similarity score based on matching attributes."""
        total_weight = 0
        matched_weight = 0
        
        for attr, importance in self.hyperparameters["attribute_importance"].items():
            if attr in node1.attributes and attr in node2.attributes:
                total_weight += importance
                
                # Handle different types of attributes
                val1 = node1.attributes[attr]
                val2 = node2.attributes[attr]
                
                if isinstance(val1, (bool, str)):
                    # Boolean and string attributes (exact match)
                    if val1 == val2:
                        matched_weight += importance
                else:
                    # Numerical attributes (similarity based on difference)
                    try:
                        # Convert to float in case they're stored as strings
                        num1 = float(val1)
                        num2 = float(val2)
                        
                        # Different similarity calculations for different attributes
                        if attr in ['face_yaw', 'face_pitch', 'face_roll']:
                            # Angle similarity (consider angles close to each other as similar)
                            diff = abs(num1 - num2)
                            if diff <= 15:  # Within 15 degrees
                                matched_weight += importance * (1 - diff/15)
                        elif attr.startswith('emotion_'):
                            # Emotion probability similarity
                            diff = abs(num1 - num2)
                            matched_weight += importance * (1 - diff)
                        elif attr in ['blur_score', 'brightness', 'contrast', 'compression_score']:
                            # Quality metrics similarity (normalized difference)
                            max_val = max(abs(num1), abs(num2))
                            if max_val > 0:
                                diff = abs(num1 - num2) / max_val
                                matched_weight += importance * (1 - min(diff, 1.0))
                        elif attr.endswith('_ratio') or attr.endswith('_symmetry'):
                            # Symmetry score similarity
                            diff = abs(num1 - num2)
                            matched_weight += importance * (1 - diff)
                        else:
                            # Default numerical similarity
                            diff = abs(num1 - num2)
                            if diff <= 0.2:  # Consider values within 20% of each other as similar
                                matched_weight += importance * (1 - diff/0.2)
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to exact matching
                        if val1 == val2:
                            matched_weight += importance
        
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _should_connect_nodes(self, node1, node2, embedding_matrix):
        """Determine if two nodes should be connected based on weighted combination of attributes and face similarity."""
        # Get weights
        face_weight = self.hyperparameters["face_similarity_weight"]
        attr_weight = self.hyperparameters["attribute_weight"]
        
        # Normalize weights to sum to 1
        total_weight = face_weight + attr_weight
        face_weight = face_weight / total_weight
        attr_weight = attr_weight / total_weight
        
        # Calculate face similarity
        idx1, idx2 = node1.id, node2.id
        face_similarity = compute_embedding_similarity(
            embedding_matrix[idx1],
            embedding_matrix[idx2]
        )
        
        # Calculate attribute similarity
        attr_similarity = self._calculate_attribute_similarity(node1, node2)
        
        # Compute weighted combination
        combined_score = (face_similarity * face_weight) + (attr_similarity * attr_weight)
        
        # Store the similarity scores in the edge attributes if we create one
        if combined_score >= self.hyperparameters["min_combined_score"]:
            node1.edge_attributes[node2.id] = {
                'face_similarity': face_similarity,
                'attribute_similarity': attr_similarity,
                'combined_score': combined_score
            }
            node2.edge_attributes[node1.id] = {
                'face_similarity': face_similarity,
                'attribute_similarity': attr_similarity,
                'combined_score': combined_score
            }
            return True
            
        return False

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
        
        # Convert attributes and embeddings to matrices
        feature_matrix, embedding_matrix, _ = self._create_attribute_matrix(all_nodes)
        
        if self.hyperparameters["use_lsh"]:
            print("Using LSH for faster matching...")
            # Compute LSH buckets using both features and embeddings
            buckets = self._compute_lsh_buckets(
                feature_matrix,
                embedding_matrix,
                n_bands=self.hyperparameters["lsh_bands"],
                band_size=self.hyperparameters["lsh_band_size"]
            )
            
            # Process nodes using LSH buckets
            edges = []
            for bucket_nodes in tqdm(buckets.values(), desc="Processing buckets"):
                if len(bucket_nodes) < 2:
                    continue
                    
                # Check all pairs in bucket
                for i, j in combinations(bucket_nodes, 2):
                    if self._should_connect_nodes(all_nodes[i], all_nodes[j], embedding_matrix):
                        edges.append((all_nodes[i], all_nodes[j]))
            
            print(f"Created {len(edges)} edges")
            
            # Create graph
            graph = HyperGraph()
            for node in all_nodes:
                graph.add_node(node)
            for node1, node2 in edges:
                graph.add_edge(node1, node2)
                
            if self.hyperparameters["visualize"]:
                visualize_graph(graph, show=self.hyperparameters["show_viz"])
            
            return graph