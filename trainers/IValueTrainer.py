from trainers.Trainer import Trainer
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
import time
from tqdm.auto import tqdm
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
from models.DQNModel import DQNModel
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from torch.cuda.amp import GradScaler
from nodes.atrnode import AttributeNode
from utils.attribute_utils import AttributeMetadata, AttributeBiasLoss

#1
"""
The idea of I value traversal is as follows:
1. Initialize DQN to predict Q values for nodes based on their attributes
2. Traverse primary model to nodes and generate predictions
3. Use prediction correctness as reward signal for DQN
4. DQN predicts Q values for nearby nodes to guide traversal
5. Calculate I values as 1-Q for exploration
6. Use DQN weights and prediction patterns to measure and correct both
   inter-attribute and intra-attribute bias
"""

class IValueTrainer(Trainer):
    """
    IValueTrainer is a subclass of Trainer that uses DQN to predict I-values
    for efficient graph traversal while maintaining both inter-attribute and
    intra-attribute balance.
    """
    tags = ["i-value"]
    
    def __init__(self, graphmanager, models, device, train_traversal=None, # Allow None initially
                 attribute_metadata=None, use_bias_loss_in_training=False,
                 bias_loss_weight=1.0, loss_fn=None):
        """Initialize the trainer with memory optimizations and DQN setup for embeddings."""
        super().__init__(graphmanager, train_traversal, models, attribute_metadata=attribute_metadata)
        print("Initializing IValueTrainer...")
        self.device = device  # Store the device

        # Loss function setup
        if loss_fn is None:
            raise ValueError("loss_fn must be provided to IValueTrainer")
        self.criterion = loss_fn


        # Extract categorical attributes for tracking
        self.categorical_attrs_for_tracking = []
        if self.attribute_metadata:
            self.categorical_attrs_for_tracking = [
                attr['name'] for attr in self.attribute_metadata if attr.get('type') == 'categorical'
            ]
            if not self.categorical_attrs_for_tracking:
                 print("IValueTrainer: No categorical attributes found in metadata for tracking.")
            else:
                 print(f"IValueTrainer: Will track distribution for attributes: {self.categorical_attrs_for_tracking}")


        # Process attribute_metadata dict list into AttributeMetadata objects FIRST
        if attribute_metadata is not None:
            print("Processing attribute metadata...")
            self.attribute_metadata = [
                AttributeMetadata(
                    name=attr['name'],
                    attr_type=attr['type'],
                    possible_values=attr.get('possible_values', None)
                )
                for attr in attribute_metadata
            ]
            # Create attribute map for efficient lookup
            self.attr_map = {attr.name: attr for attr in self.attribute_metadata}
            # Bias measurement and correction
            self.bias_loss = AttributeBiasLoss(self.attribute_metadata, self.attr_map).cuda() # Use self.attribute_metadata
            self.bias_weight = bias_loss_weight  # Weight for bias loss term
        else:
            print("No attribute metadata provided.")
            self.attribute_metadata = None
            self.attr_map = {}
            self.bias_loss = None
            self.bias_weight = 0.0

        self.embedding_dim = 512 # Define expected embedding dimension
        self.feature_dim = None # Will be calculated

        # Memory optimization settings
        self.batch_size = 32  # Batch size for primary model training steps
        self.mini_batch_size = 8 # Potentially unused? Check usage.
        # Grad accumulation likely applies to primary model training
        self.gradient_accumulation_steps = 8
        self.max_nodes_per_epoch = 10000
        self.prefetch_workers = 4 # Number of threads for prefetching data
        self.scaler = GradScaler() # For mixed precision training of primary model

        # Initialize prediction stats (optional, keep if used)
        self.prediction_stats = defaultdict(lambda: defaultdict(list)) # Restore original prediction stats structure

        

        # Setup DQN Models
        self.dqns = []
        # Check the processed self.attribute_metadata
        if self.attribute_metadata is not None:
            # Get a sample node to determine feature dimension
            try:
                # Get the graph directly from the manager (assumed to be the training graph)
                train_graph = self.graphmanager.get_graph() # Corrected call
                if train_graph is None or len(train_graph.nodes) == 0: # Check graph/nodes directly
                     print("Warning: Train graph not available or empty. Cannot determine DQN feature dimension.")
                     sample_node = None
                else:
                     # Use list conversion for safer iteration if underlying structure is complex
                     sample_node = list(train_graph.get_nodes())[0]

                if sample_node:
                    sample_features, sample_embedding = self._get_dqn_features(sample_node)

                    if sample_features is not None:
                        self.feature_dim = sample_features.shape[0]
                        print(f"Determined DQN feature dimension (excluding embedding): {self.feature_dim}")
                        # Instantiate DQN for each primary model
                        for i in range(len(self.models)):
                            print(f"Initializing DQN for primary model {i}")
                            dqn = DQNModel(
                                feature_dim=self.feature_dim, 
                                embedding_dim=self.embedding_dim,
                                device=self.device  # Pass the device
                            )
                            self.dqns.append(dqn) # DQNModel handles moving itself to device
                    else:
                        print("Warning: Could not determine feature dimension from sample node. DQN not initialized.")
                        self.dqns = None # Use None to indicate no DQNs
                else:
                    print("Warning: Could not get a sample node. DQN not initialized.")
                    self.dqns = None
            except StopIteration: # This might not be needed if we check length/use list
                print("Warning: Train graph has no nodes (StopIteration). DQN not initialized.")
                self.dqns = None
            except Exception as e:
                print(f"Error initializing DQN: {e}. DQN not initialized.")
                self.dqns = None
        else:
            print("Warning: No attribute metadata provided. DQN not initialized.")
            self.dqns = None

        # Setup logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"ivalue_trainer_{timestamp}.json"
        self.metrics_history = []

        # Prefetch queue for data loading
        # Queue size relative to workers and batch size
        self.prefetch_queue = Queue(maxsize=self.prefetch_workers * 2)

        # Cache for computed features (optional, consider LRUCache for size limit)
        self.feature_cache = {}

        # Gradient accumulation steps defined above

        self.use_bias_loss_in_training = use_bias_loss_in_training
    # Add this method to the class
    def set_train_traversal(self, train_traversal):
        """Sets the training traversal strategy after initialization."""
        if self.train_traversal is not None:
            print("Warning: Overwriting existing train_traversal in IValueTrainer.")
        self.train_traversal = train_traversal
        print(f"IValueTrainer: train_traversal set to {type(train_traversal).__name__}")

    def _clear_memory(self):
        """Clear unused memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _optimize_batch(self, batch):
        """Optimize batch data for memory efficiency."""
        # Move batch to CPU if not needed immediately
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu()
        elif isinstance(batch, (list, tuple)):
            batch = [b.cpu() if isinstance(b, torch.Tensor) else b for b in batch]
        return batch

    def _get_dqn_features(self, node):
        """Extract attribute features for DQN input."""
        try:
            # Ensure node is the expected type (e.g., AttributeNode)
            if not isinstance(node, AttributeNode):
                print(f"Warning: Expected AttributeNode, got {type(node)}. Cannot extract DQN features.")
                return None, None

            # Get all attributes as a list
            features_list = []
            embedding_data = None

            # Extract standard attributes based on metadata
            # Check if attribute_metadata was successfully processed in __init__
            if self.attribute_metadata:
                for attr_meta in self.attribute_metadata:
                    attr_name = attr_meta.name # Use AttributeMetadata object properties
                    attr_type = attr_meta.attr_type

                    # Special handling for face embedding
                    if attr_name == 'face_embedding':
                        embedding_data = node.attributes.get(attr_name)
                        continue # Skip adding embedding to features_list here

                    # Handle other attributes
                    if attr_type == 'categorical':
                        # One-hot encode categorical values
                        if attr_meta.possible_values:
                            for possible_value in attr_meta.possible_values:
                                features_list.append(1.0 if node.attributes.get(attr_name) == possible_value else 0.0)
                    else:  # continuous
                        try:
                            features_list.append(float(node.attributes.get(attr_name, 0))) # Default to 0 if missing
                        except (TypeError, ValueError) as e:
                            # This should ideally not happen if embedding is handled above,
                            # but catch potential errors with other numerical types.
                            print(f"Warning: Could not convert attribute '{attr_name}' value '{node.attributes.get(attr_name)}' to float: {e}. Using 0.")
                            features_list.append(0.0)
            elif isinstance(node, AttributeNode): # Fallback if no metadata, but node is AttributeNode
                # Fallback: Use default features if no metadata (e.g., label, degree)
                # This part might need adjustment based on expected fallback behavior
                features_list.append(float(node.label))  # Label (0 or 1)
                features_list.append(len(node.get_adjacent_nodes()) / 100.0)  # Normalized degree

            # Convert features list to tensor
            try:
                features_tensor = torch.tensor(features_list, dtype=torch.float32)
            except Exception as e:
                print(f"Error converting features list to tensor: {e}. List: {features_list}")
                features_tensor = None

            # Handle embedding: convert to tensor or create zero tensor if missing/invalid
            embedding_tensor = None
            if embedding_data is not None:
                try:
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting embedding to tensor: {e}. Using zeros.")
                    embedding_tensor = torch.zeros(self.embedding_dim, dtype=torch.float32)

            # Return both tensors
            return features_tensor, embedding_tensor

        except Exception as e:
            print(f"Error extracting DQN features: {e}")
            return None, None # Return None to indicate failure

    def _get_cnn_features(self, node):
        """Extract image features for CNN input."""
        # Get the image data and transform it for CNN input
        image_data = node.get_data().load_data()
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for torchvision transforms
        image_pil = Image.fromarray(image_rgb)
        
        # Apply transformations using the parent class's transform
        transformed_image = self.transform(image_pil)  # Shape: [C, H, W]
        
        # Add batch dimension and move to GPU
        batched_image = transformed_image.unsqueeze(0).cuda()  # Shape: [1, C, H, W]
        
        return batched_image
    
    def get_i_value(self, node, model_idx):
        """Predicts the I-value for a given node using the specified DQN model.
        I-value = 1 - Q_value(state), where state comes from node features.
        """
        try:
            # Check if DQN models are available
            if self.dqns is None or model_idx >= len(self.dqns):
                print(f"Warning: DQN model {model_idx} not available for I-value calculation.")
                return 0.0 # Default I-value if DQN doesn't exist

            # Get the target DQN model and its device
            dqn_model = self.dqns[model_idx]
            target_device = dqn_model.device # Get device from the DQN model

            # Retrieve features and embedding for the node
            features_tensor, embedding_tensor = self._get_dqn_features(node)

            # Check if feature extraction was successful
            if features_tensor is None:
                print(f"Warning: Could not extract features for node {node.node_id}. Cannot calculate I-value.")
                return 0.0

            # Handle potentially missing embedding tensor
            if embedding_tensor is None:
                # Create a zero tensor with the expected embedding dimension if missing
                embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            elif not isinstance(embedding_tensor, torch.Tensor):
                # Ensure it's a tensor if it exists but isn't one already
                try:
                    embedding_tensor = torch.tensor(embedding_tensor, dtype=torch.float32, device=target_device)
                except Exception as e:
                    print(f"Error converting embedding to tensor for node {node.node_id}: {e}. Using zeros.")
                    embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            else:
                # Ensure existing tensor is on the correct device
                embedding_tensor = embedding_tensor.to(target_device)

            # Features should already be a tensor from _get_dqn_features
            features_tensor = features_tensor.to(target_device)

            # Add batch dimension
            features_tensor = features_tensor.unsqueeze(0)
            # Unsqueeze embedding tensor AFTER ensuring it's on the correct device
            embedding_tensor = embedding_tensor.unsqueeze(0)

            # Ensure DQN is on the correct device
            q_value = dqn_model.predict_i_value(features_tensor.to(dqn_model.device), 
                                                 embedding_tensor.to(dqn_model.device))

            # I-value = 1 - Q-value (assuming Q is normalized or represents probability-like value)
            # Ensure Q-value is detached and moved to CPU for calculation
            q_value = q_value.detach().cpu().item()
            i_value = 1.0 - q_value
            
            # Update prediction stats
            self.update_prediction_stats(node, i_value > 0.5, model_idx)
            
            return i_value
            
        except Exception as e:
            print(f"Error calculating I-value: {str(e)}")
            return 0.0
    
    def get_traversal(self, graph, num_pointers=5, num_steps=100, return_delay=10, warp_chance=0.005):
        """Create a new IValueTraversal instance configured with this trainer."""
        from traversals.IValueTraversal import IValueTraversal
        return IValueTraversal(
            graph=graph,
            num_pointers=num_pointers,
            num_steps=num_steps,
            trainer=self,
            return_delay=return_delay,
            warp_chance=warp_chance
        )
    
    def update_prediction_stats(self, node, correct, model_idx):
        """Update prediction statistics for each attribute value."""
        node_attrs = node.attributes  # Access attributes directly
        
        for attr in self.attribute_metadata:
            if attr.name in node_attrs:
                value = node_attrs[attr.name]
                if attr.attr_type == 'categorical':
                    self.prediction_stats[f'model_{model_idx}_{attr.name}'][value].append(float(correct))
    
    def get_attribute_bias_score(self, model_idx):
        """Measure both inter-attribute and intra-attribute bias."""
        attribute_weights = self.dqns[model_idx].get_attribute_weights()
        # Convert attribute weights to tensor if they aren't already
        if not isinstance(attribute_weights, torch.Tensor):
            attribute_weights = torch.tensor(attribute_weights).cuda()
        bias_score = self.bias_loss(attribute_weights, self.prediction_stats[f'model_{model_idx}'])
        return bias_score  # Return tensor

    def train_dqn(self, model_idx):
        """Train DQN using experience replay with comprehensive bias correction."""
        if len(self.dqns[model_idx].replay_buffer) < self.dqns[model_idx].batch_size:
            return 0.0
        
        # Sample random batch from replay buffer
        batch = random.sample(self.dqns[model_idx].replay_buffer, self.dqns[model_idx].batch_size)
        states = torch.stack([item[0] for item in batch]).cuda()
        rewards = torch.tensor([item[1] for item in batch], dtype=torch.float32).cuda()
        
        # Compute Q values for current states
        q_values = self.dqns[model_idx](states).squeeze()
        
        # Total loss is just Q-learning loss
        q_loss = F.mse_loss(q_values, rewards)
        
        self.dqns[model_idx].optimizer.zero_grad()
        q_loss.backward()
        self.dqns[model_idx].optimizer.step()
        
        return q_loss
    
    def preprocess_batch(self, batch_nodes):
        """Preprocess a batch of nodes to ensure consistent tensor sizes."""
        if not batch_nodes:
            return None, None
        
        try:
            # Get CNN model for transforms
            cnn_model = None
            for model in self.models:
                if hasattr(model, 'transform'):
                    cnn_model = model
                    break
                    
            if cnn_model is None:
                print("No model with transform method found")
                return None, None
                
            # Prepare data and labels
            processed_batch = []
            valid_nodes = []
            
            for node in batch_nodes:
                try:
                    if not isinstance(node, AttributeNode):
                        continue
                        
                    # Get node data
                    data = node.get_data()
                    if data is None:
                        continue
                        
                    # Load image data
                    img_data = data.load_data()
                    if img_data is None:
                        continue
                        
                    # Transform image data using model's transform method
                    if not isinstance(img_data, torch.Tensor):
                        try:
                            # Get the first model's transform method
                            transform = self.models[0].transform
                            img_data = transform(img_data)
                        except Exception as e:
                            print(f"Error transforming image: {str(e)}")
                            continue
                            
                    processed_batch.append(img_data)
                    valid_nodes.append(node)
                        
                except Exception as e:
                    print(f"Error processing node in batch: {str(e)}")
                    continue
                    
            if not processed_batch:
                return None, None
                
            # Stack tensors
            try:
                images = torch.stack(processed_batch).cuda()
                return images, valid_nodes
            except Exception as e:
                print(f"Error stacking tensors: {str(e)}")
                return None, None
                
        except Exception as e:
            print(f"Error in preprocess_batch: {str(e)}")
            return None, None

    def process_node_data(self, node, model_idx):
        """Process node data and update DQN replay buffer with comprehensive bias awareness."""
        try:
            # Get node features for DQN
            dqn_features, dqn_embedding = self._get_dqn_features(node)
            if dqn_features is None:
                return None, None, None, False
                
            # Get image features for CNN
            image_features = self._get_cnn_features(node)
            if image_features is None:
                return None, None, None, False
                
            # Forward pass through model
            output = self.models[model_idx](image_features)
            
            # Get label
            label = torch.tensor([1.0 if node.is_fake() else 0.0], device='cuda').float()
            
            # Check prediction correctness
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct = (predicted == label).item()
            
            # Update prediction stats for bias tracking
            self.update_prediction_stats(node, correct, model_idx)
            
            # Calculate current bias loss
            curr_bias_loss = self.bias_loss(output, [node.attributes])
            
            # Calculate rewards
            # 1. Uncertainty reward: Higher for uncertain predictions
            pred_prob = torch.sigmoid(output).item()
            uncertainty_reward = 1.0 - abs(pred_prob - 0.5) * 2  # Max at 0.5, min at 0 or 1
            
            # 2. Bias reward: Higher for biased predictions (need correction)
            bias_reward = min(curr_bias_loss, 1.0)  # Cap at 1.0
            
            # 3. Error reward: Higher for incorrect predictions (need improvement)
            error_reward = 1.0 - correct
            
            # Combine rewards with weights
            reward = bias_reward

            # Store experience in replay buffer (state, reward)
            self.dqns[model_idx].replay_buffer.append((dqn_features, reward))
            
            # Train DQN
            dqn_loss = self.train_dqn(model_idx)
            
            # Calculate losses
            classification_loss = self.criterion(output, label)
            total_loss = classification_loss + self.bias_weight * curr_bias_loss
            
            # Get I-value for performance tracking
            i_value = self.get_i_value(node, model_idx)
            
            # Update graph manager with performance tracking
            if hasattr(self.graphmanager, 'track_performance'):
                self.graphmanager.track_performance(node, i_value)
            
            # Update graph structure periodically
            if hasattr(self.graphmanager, 'update_graph'):
                self.graphmanager.update_graph()
            
            return total_loss, dqn_loss, curr_bias_loss, correct
            
        except Exception as e:
            print(f"Error in process_node_data: {str(e)}")
            return None, None, None, False
    
    def train_step(self, batch_nodes):
        """Perform a single training step."""
        try:
            # Preprocess batch
            images, nodes = self.preprocess_batch(batch_nodes)
            if images is None or nodes is None:
                return 0.0
                
            # Zero gradients
            for model in self.models:
                model.zero_grad()
                
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = []
                for model in self.models:
                    output = model(images)
                    outputs.append(output)
                    
                # Get labels
                labels = torch.tensor([
                    1.0 if node.is_fake() else 0.0
                    for node in nodes
                ], device='cuda').float()
                
                # Calculate loss
                loss = sum(
                    self.criterion(output.squeeze(), labels)
                    for output in outputs
                ) / len(self.models)
                
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if self.steps % self.gradient_accumulation_steps == 0:
                for model in self.models:
                    self.scaler.step(model.optim)
                self.scaler.update()
                
            return loss.item()
            
        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            return 0.0
    
    def train(self):
        """Train the model for one epoch using memory-efficient approach."""
        try:
            # Set models to training mode
            for model in self.models:
                model.train()
            for dqn in self.dqns:
                dqn.train()
                
            # Initialize metrics
            total_loss = 0
            correct = 0
            total = 0
            batch_count = 0
            total_train_bias_loss = 0
            
            if self.train_traversal is None:
                raise ValueError("train_traversal must be set before training.")

            # Reset traversal for this epoch
            self.train_traversal.reset_pointers()
            
            # Get total nodes for this epoch
            total_nodes = self.train_traversal.num_steps
            print(f"Training on {total_nodes} nodes this epoch")
            
            # --- Clear previous epoch's distribution --- 
            attribute_distribution = defaultdict(lambda: defaultdict(int))
            track_attributes = bool(self.categorical_attrs_for_tracking)
            if track_attributes:
                print(f"Tracking attribute distribution for attributes: {self.categorical_attrs_for_tracking}")
            else:
                print("Warning: attribute tracking is disabled. No attribute distribution will be tracked or returned.")

            # Initialize metrics for the epoch
            total_loss_cnn = 0.0

            # Process nodes in batches
            pbar = tqdm(total=total_nodes, desc=f"Epoch")
            
            nodes_processed = 0
            while nodes_processed < total_nodes:
                try:
                    # Get batch of nodes from traversal
                    batch_nodes = self.train_traversal.traverse(batch_size=self.batch_size)
                    if not batch_nodes:
                        continue  # Skip this iteration but don't break the loop
                        
                    # --- Track Attribute Distribution --- 
                    if track_attributes:
                         for node in batch_nodes:
                             if hasattr(node, 'attributes') and node.attributes: # Check each node
                                 for attr_name in self.categorical_attrs_for_tracking:
                                     if attr_name in node.attributes:
                                         attr_value = node.attributes[attr_name]
                                         # Use str() for potential non-hashable values
                                         attribute_distribution[attr_name][str(attr_value)] += 1 
                    # ------------------------------------

                    # Process nodes
                    batch_data = []
                    batch_labels = []
                    current_batch_nodes = []
                    
                    for node in batch_nodes:
                        try:
                            if not isinstance(node, AttributeNode):
                                continue
                                
                            # Get node data
                            data = node.get_data()
                            if data is None:
                                continue
                                
                            # Load image data
                            img_data = data.load_data()
                            if img_data is None:
                                continue
                            
                            # Transform image data using model's transform method
                            if not isinstance(img_data, torch.Tensor):
                                try:
                                    # Get the first model's transform method
                                    transform = self.models[0].transform
                                    img_data = transform(img_data)
                                except Exception as e:
                                    print(f"Error transforming image: {str(e)}")
                                    continue
                            
                            batch_data.append(img_data)
                            batch_labels.append(float(node.label))
                            current_batch_nodes.append(node)
                            
                        except Exception as e:
                            print(f"Error processing node: {str(e)}")
                            continue
                            
                    if not batch_data:
                        continue
                        
                    # Stack tensors
                    try:
                        features = torch.stack(batch_data).cuda()
                        labels = torch.tensor(batch_labels, dtype=torch.float32).cuda()
                        
                        # Process mini-batches
                        for j in range(0, len(features), self.mini_batch_size):
                            mini_features = features[j:j + self.mini_batch_size]
                            mini_labels = labels[j:j + self.mini_batch_size]
                            mini_batch_nodes = current_batch_nodes[j:j + self.mini_batch_size]
                            
                            # Forward pass
                            outputs = self.models[0](mini_features)
                            primary_loss = self.criterion(outputs, mini_labels.unsqueeze(1))
                            
                            # Calculate bias loss
                            bias_loss_val = torch.tensor(0.0, device=primary_loss.device)
                            if self.bias_loss is not None:
                                try:
                                    valid_indices = [idx for idx, node in enumerate(mini_batch_nodes)
                                                     if hasattr(node, 'attributes') and isinstance(node.attributes, dict)]
                                    if valid_indices:
                                        valid_nodes = [mini_batch_nodes[i] for i in valid_indices]
                                        valid_outputs = outputs[valid_indices]
                                        valid_labels = mini_labels[valid_indices]
                                        
                                        current_bias_loss = self.bias_loss(valid_outputs, valid_labels.unsqueeze(1), valid_nodes)
                                        if isinstance(current_bias_loss, torch.Tensor):
                                            bias_loss_val = current_bias_loss
                                            total_train_bias_loss += bias_loss_val.item()
                                        else:
                                            print(f"Warning: Training bias loss calculation returned non-tensor: {current_bias_loss}")
                                except Exception as e:
                                    print(f"Error calculating training bias loss: {e}")
                            
                            # Combine losses
                            combined_loss = primary_loss
                            if self.use_bias_loss_in_training and self.bias_loss is not None:
                                combined_loss = combined_loss + self.bias_weight * bias_loss_val
                            
                            # Backward pass
                            combined_loss.backward()
                            
                            # Step optimizer for the primary model
                            self.models[0].optim.step()
                            
                            # Update metrics
                            total_loss += primary_loss.item()
                            predicted = (torch.sigmoid(outputs) > 0.5).float()
                            correct += (predicted == mini_labels.unsqueeze(1)).sum().item()
                            total += len(mini_labels)
                            
                            # Step optimizer if needed
                            if (batch_count + 1) % self.gradient_accumulation_steps == 0:
                                for model in self.models:
                                    model.optim.step()
                                    model.optim.zero_grad()
                                    
                            batch_count += 1
                            
                        nodes_processed += len(batch_nodes)
                        pbar.update(len(batch_nodes))
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("WARNING: out of memory")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                            
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
                    
            pbar.close()
            
            # Compute epoch metrics
            if batch_count == 0:
                return self._get_empty_metrics()
                
            metrics = {
                'avg_loss': total_loss / batch_count,
                'accuracy': correct / max(1, total),
                'avg_bias_loss': total_train_bias_loss / batch_count
            }
            
            # Log metrics
            self.log_metrics(metrics)
            
            return metrics, attribute_distribution
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return self._get_empty_metrics()

    def get_model_by_id(self, node_id):
        # Simple hash function to distribute nodes among models
        return node_id % len(self.models)

    def _get_empty_metrics(self):
        """Return empty metrics structure for when no valid data is processed."""
        return {
            'avg_loss': 0.0,
            'accuracy': 0.0,
            'avg_bias_loss': 0.0
        }
        
    def log_metrics(self, metrics):
        """Log training metrics to file."""
        metrics_dict = {}
        for key, value in metrics.items():
            # Convert tensors to float/int
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics_dict[key] = value
        
        # Add timestamp
        metrics_dict['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics_dict)
        
        # Write to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Also print to console
        print(f"Metrics: {metrics_dict}")

    def get_all_final_i_values(self, graph_split: str) -> dict:
        """Calculates the final I-value for all nodes in the specified graph split.

        Args:
            graph_split: The name of the graph split ('train', 'val', or 'test').

        Returns:
            A dictionary mapping node IDs to their final I-values.
        """
        if graph_split == 'train':
            graph = self.train_manager.graph
        elif graph_split == 'val':
            graph = self.val_manager.graph
        elif graph_split == 'test':
            graph = self.test_manager.graph
        else:
            raise ValueError(f"Invalid graph_split: {graph_split}. Must be 'train', 'val', or 'test'.")

        if not graph:
            print(f"Warning: Graph for split '{graph_split}' not found in IValueTrainer. Cannot calculate I-values.")
            return {}

        nodes = graph.get_nodes()
        if not nodes:
            print(f"Warning: No nodes found in graph for split '{graph_split}'.")
            return {}

        final_i_values = {}
        self.dqns[0].eval()  # Set model to evaluation mode

        with torch.no_grad():
            for node in nodes:
                if hasattr(node, 'attributes') and isinstance(node.attributes, dict):
                    try:
                        # Prepare input tensor (handle potential missing attributes or non-numeric values)
                        # Assuming attributes are pre-processed or can be converted
                        # This part might need adjustment based on how attributes are structured and fed to the model
                        attr_tensor = self._get_dqn_features(node)[0].to(self.device)
                        # Predict Q-values
                        q_values = self.dqns[0](attr_tensor)
                        # I-value = 1 - max(Q-values) - Assuming DQN predicts Q-values for actions/neighbors
                        # Adjust based on your DQN output definition. If it outputs a single value, use that.
                        i_value = 1.0 - torch.max(q_values).item() 
                        final_i_values[node.id] = i_value
                    except Exception as e:
                        print(f"Warning: Could not calculate I-value for node {node.id} in split {graph_split}: {e}")
                        final_i_values[node.id] = -1.0 # Or some indicator value
                else:
                    # Handle nodes without attributes if necessary
                    final_i_values[node.id] = -1.0 # Assign a default/indicator value

        return final_i_values
