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

class AttributeMetadata:
    def __init__(self, name, attr_type, possible_values=None):
        self.name = name
        self.attr_type = attr_type  # 'categorical' or 'continuous'
        self.possible_values = possible_values  # For categorical attributes
        self.value_counts = defaultdict(int)  # Track distribution of values
        self.predictions = defaultdict(list)  # Track predictions per value

class AttributeBiasLoss(nn.Module):
    def __init__(self, attribute_metadata, attr_map):
        super(AttributeBiasLoss, self).__init__()
        self.attribute_metadata = attribute_metadata
        self.attr_map = attr_map
        
    def forward(self, predictions, node_attrs_list):
        """Calculate bias loss based on attribute predictions."""
        if isinstance(node_attrs_list, torch.Tensor) or not node_attrs_list:
            return torch.zeros(1, device=predictions.device)
            
        batch_size = predictions.size(0)
        
        # Get predictions per sample
        pred_probs = predictions.sigmoid()
        
        # Group predictions by attribute values
        attr_predictions = defaultdict(lambda: defaultdict(list))
        
        # First pass: collect predictions for each attribute value
        for i, node_attrs in enumerate(node_attrs_list):
            if not isinstance(node_attrs, dict):
                continue
                
            for attr_name, attr_val in node_attrs.items():
                if attr_name in self.attr_map:
                    attr_predictions[attr_name][attr_val].append(pred_probs[i])
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        num_comparisons = 0
        
        # Second pass: calculate bias loss between different attribute values
        for attr_name, value_preds in attr_predictions.items():
            if len(value_preds) < 2:  # Need at least 2 different values to compare
                continue
                
            # Calculate mean prediction for each attribute value
            value_means = {}
            for value, preds in value_preds.items():
                if preds:
                    value_means[value] = torch.stack(preds).mean()
            
            # Compare means between different attribute values
            values = list(value_means.keys())
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    val1, val2 = values[i], values[j]
                    mean1, mean2 = value_means[val1], value_means[val2]
                    # Bias loss is the squared difference between means
                    bias_loss = F.mse_loss(mean1, mean2)
                    total_loss += bias_loss
                    num_comparisons += 1
        
        # Return average bias loss
        if num_comparisons > 0:
            return total_loss / num_comparisons
        return torch.zeros(1, device=predictions.device)

class IValueTrainer(Trainer):
    """
    IValueTrainer is a subclass of Trainer that uses DQN to predict I-values
    for efficient graph traversal while maintaining both inter-attribute and
    intra-attribute balance.
    """
    tags = ["i-value"]
    
    def __init__(self, graphmanager, train_traversal, val_traversal, models, attribute_metadata=None):
        """Initialize the trainer with memory optimizations."""
        super().__init__(graphmanager, train_traversal, val_traversal, models)
        self.attribute_metadata = attribute_metadata
        
        # Memory optimization settings
        self.batch_size = 32  # Reduced from 128
        self.mini_batch_size = 8  # Reduced from 16
        self.gradient_accumulation_steps = 8  # Increased from 4 for more gradual updates
        self.max_nodes_per_epoch = 10000  # Increased to match validation size
        self.steps = 0  # Track total steps for gradient accumulation
        
        # Initialize scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Enable memory-efficient attention if using transformer models
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Set PyTorch memory allocator settings
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.8)  # Reserve some GPU memory
            
        # Initialize DQN models for each split
        input_dim = 2  # Base features: label and degree
        if attribute_metadata:
            for attr in attribute_metadata:
                if attr['type'] == 'categorical':
                    input_dim += len(attr.get('possible_values', []))
                else:
                    input_dim += 1
                    
        self.dqns = [DQNModel(input_dim).cuda() for _ in range(len(models))]
        
        # Connect DQN models to graph manager for performance tracking
        if hasattr(self.graphmanager, 'set_i_value_predictor'):
            self.graphmanager.set_i_value_predictor(self.dqns[0])  # Use first DQN model
            
        # Store attribute metadata
        if attribute_metadata is not None:
            self.attribute_metadata = [
                AttributeMetadata(
                    name=attr['name'],
                    attr_type=attr['type'],
                    possible_values=attr['possible_values']
                )
                for attr in attribute_metadata
            ]
            
            # Create attribute map for efficient lookup
            self.attr_map = {attr.name: attr for attr in self.attribute_metadata}
            
            # Bias measurement and correction
            self.bias_loss = AttributeBiasLoss(self.attribute_metadata, self.attr_map).cuda()
            self.bias_weight = 0.1  # Weight for bias loss term
            
            # Track prediction statistics per attribute value
            self.prediction_stats = defaultdict(lambda: defaultdict(list))
            
            # Setup logging
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"ivalue_trainer_{timestamp}.json"
            self.metrics_history = []
            
            # Prefetch queue for data loading
            self.prefetch_queue = Queue(maxsize=3)
            
            # Cache for computed features
            self.feature_cache = {}
            
            # Accumulate gradients for larger effective batch
            self.gradient_accumulation_steps = 4
        
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
            # Get all attributes as a list
            features = []
            
            # Add basic node features
            features.append(float(node.label))  # Label (0 or 1)
            features.append(len(node.get_adjacent_nodes()) / 100.0)  # Normalized degree
            
            # Add attribute features if available
            if hasattr(node, 'attributes'):
                for attr_meta in self.attribute_metadata:
                    attr_name = attr_meta.name
                    if attr_name in node.attributes:
                        value = node.attributes[attr_name]
                        if attr_meta.attr_type == 'categorical':
                            # One-hot encode categorical values
                            if attr_meta.possible_values:
                                for possible_value in attr_meta.possible_values:
                                    features.append(1.0 if value == possible_value else 0.0)
                        else:  # continuous
                            features.append(float(value))
            
            # Convert to tensor and keep on CPU initially
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error extracting DQN features: {str(e)}")
            return None

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
        """Calculate I-value as 1-Q for a given node."""
        try:
            # Get node features
            node_features = self._get_dqn_features(node)
            if node_features is None:
                return 0.0
                
            # Forward pass through DQN
            with torch.no_grad():
                # DQN will handle moving tensor to correct device
                q_value = self.dqns[model_idx](node_features)
                
            # Convert to scalar and calculate I-value
            q_value = q_value.item()
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
                    data = node.get_data()
                    if data is None:
                        continue
                        
                    img_data = data.load_data()
                    if img_data is None:
                        continue
                        
                    # Transform image using model's transform method
                    transformed_img = cnn_model.transform(img_data)
                    if transformed_img is not None:
                        processed_batch.append(transformed_img)
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
            dqn_features = self._get_dqn_features(node)
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
            classification_loss = self.models[model_idx].loss(output, label)
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
                    model.loss(output.squeeze(), labels)
                    for model, output in zip(self.models, outputs)
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
    
    def train(self, epoch):
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
            
            # Reset traversal for this epoch
            self.train_traversal.reset_pointers()
            
            # Get total nodes for this epoch
            total_nodes = self.train_traversal.num_steps
            print(f"Training on {total_nodes} nodes this epoch")
            
            # Process nodes in batches
            pbar = tqdm(total=total_nodes, desc=f"Epoch {epoch}")
            
            nodes_processed = 0
            while nodes_processed < total_nodes:
                try:
                    # Get batch of nodes from traversal
                    batch_nodes = self.train_traversal.traverse(batch_size=self.batch_size)
                    if not batch_nodes:
                        continue  # Skip this iteration but don't break the loop
                        
                    # Process nodes
                    batch_data = []
                    batch_labels = []
                    
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
                            
                        except Exception as e:
                            print(f"Error processing node: {str(e)}")
                            continue
                            
                    if not batch_data:
                        continue
                        
                    # Stack batch data
                    try:
                        features = torch.stack(batch_data).cuda()
                        labels = torch.tensor(batch_labels, dtype=torch.float32).cuda()
                        
                        # Process mini-batches
                        for j in range(0, len(features), self.mini_batch_size):
                            mini_features = features[j:j + self.mini_batch_size]
                            mini_labels = labels[j:j + self.mini_batch_size]
                            
                            # Forward pass
                            outputs = self.models[0](mini_features)
                            loss = self.models[0].loss(outputs, mini_labels.unsqueeze(1))
                            
                            # Backward pass
                            loss.backward()
                            
                            # Update metrics
                            total_loss += loss.item()
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
                return self._get_empty_metrics(epoch)
                
            metrics = {
                'epoch': epoch,
                'avg_loss': total_loss / batch_count,
                'accuracy': correct / max(1, total),
                'avg_bias_loss': 0.0 # TBD
            }
            
            # Log metrics
            self.log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return self._get_empty_metrics(epoch)
    
    def val(self, epoch):
        """Validation phase using I-value based traversal."""
        total_loss = 0
        total_bias_loss = 0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        mini_batch_size = 32
        
        # Reset validation traversal at start of validation
        self.val_traversal.reset_pointers()
        
        try:
            # Get total number of nodes for progress bar
            total_nodes = len(list(self.val_traversal.graph.get_nodes()))
            progress_bar = tqdm(total=total_nodes, desc="Validation Progress")
            nodes_processed = 0
            
            while True:  # Keep going until we get an empty batch
                batch = self.val_traversal.traverse()
                if not batch:  # If batch is empty, we're done
                    break
                
                # Process in mini-batches
                for i in range(0, len(batch), mini_batch_size):
                    mini_batch = batch[i:i + mini_batch_size]
                    nodes_processed += len(mini_batch)
                    
                    features_list = []
                    labels_list = []
                    
                    # Collect features and labels
                    for node in mini_batch:
                        if not hasattr(node, 'get_label') or not hasattr(node, 'attributes'):
                            continue
                            
                        try:
                            features = self._get_cnn_features(node)
                            features_list.append(features)
                            labels_list.append(node.get_label())
                        except Exception as e:
                            continue
                    
                    if not features_list:
                        continue
                    
                    # Stack features and labels
                    if len(features_list) == 1:
                        features_batch = features_list[0]  # Single tensor, no need to stack
                    else:
                        features_batch = torch.cat(features_list, dim=0)
                    labels_batch = torch.tensor(labels_list).cuda()
                    
                    # Validate each model
                    for model_idx, model in enumerate(self.models):
                        model.model.eval()
                        
                        with torch.no_grad():
                            # Forward pass
                            outputs = model(features_batch)
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                            
                            # Calculate metrics
                            labels_batch = labels_batch.float().unsqueeze(1)
                            correct = (preds == labels_batch).sum().item()
                            total_predictions += len(labels_batch)
                            correct_predictions += correct
                            
                            # Calculate losses
                            loss = model.loss(outputs, labels_batch)
                            total_loss += loss.item()
                            
                            # Calculate bias loss using all nodes in mini-batch
                            try:
                                batch_attrs = [node.attributes for node in mini_batch if hasattr(node, 'attributes')]
                                if batch_attrs:
                                    bias_loss = self.bias_loss(outputs, batch_attrs)
                                    total_bias_loss += bias_loss.item() if isinstance(bias_loss, torch.Tensor) else 0
                            except Exception as e:
                                print(f"Error calculating validation bias loss: {e}")
                            
                            num_batches += 1
                    
                    # Update progress
                    progress_bar.update(len(mini_batch))
                    progress_bar.set_description(
                        f"Validation Progress | Loss: {total_loss/max(num_batches,1):.4f} | Acc: {correct_predictions/max(total_predictions,1):.4f}"
                    )
            
            progress_bar.close()
            
        except Exception as e:
            progress_bar.close()
            print(f"Error in validation: {e}")
            return self._get_empty_metrics(epoch)
            
        return {
            'avg_loss': total_loss / max(num_batches, 1),
            'avg_bias_loss': total_bias_loss / max(num_batches, 1),
            'accuracy': correct_predictions / total_predictions
        }

    def test(self):
        """
        Testing phase using I-value based traversal.
        Similar to validation but using the test traversal.
        """
        metrics_per_model = [{
            'total_loss': 0,
            'total_bias_loss': 0,
            'correct_predictions': 0,
            'total_predictions': 0,
            'num_batches': 0
        } for _ in self.models]
        
        mini_batch_size = 32
        
        try:
            # Get total number of nodes for progress bar
            total_nodes = len(list(self.test_traversal.graph.get_nodes()))
            progress_bar = tqdm(total=total_nodes, desc="Testing Progress")
            nodes_processed = 0
            
            # Testing loop
            while True:  # Keep going until we get an empty batch
                batch = self.test_traversal.traverse()  # Get next batch
                if not batch:  # If batch is empty, we're done
                    break
                
                # Process in mini-batches
                for i in range(0, len(batch), mini_batch_size):
                    mini_batch = batch[i:i + mini_batch_size]
                    nodes_processed += len(mini_batch)
                    
                    features_list = []
                    labels_list = []
                    
                    # Collect features and labels
                    for node in mini_batch:
                        if not hasattr(node, 'get_label') or not hasattr(node, 'attributes'):
                            continue
                            
                        try:
                            features = self._get_cnn_features(node)
                            features_list.append(features)
                            labels_list.append(node.get_label())
                        except Exception as e:
                            continue
                    
                    if not features_list:
                        continue
                    
                    # Stack features and labels
                    if len(features_list) == 1:
                        features_batch = features_list[0]  # Single tensor, no need to stack
                    else:
                        features_batch = torch.cat(features_list, dim=0)
                    labels_batch = torch.tensor(labels_list).cuda()
                    
                    # Test each model separately
                    for model_idx, model in enumerate(self.models):
                        model.model.eval()
                        metrics = metrics_per_model[model_idx]
                        
                        with torch.no_grad():
                            # Forward pass
                            outputs = model(features_batch)
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                            
                            # Calculate metrics
                            labels_batch = labels_batch.float().unsqueeze(1)
                            correct = (preds == labels_batch).sum().item()
                            metrics['total_predictions'] += len(labels_batch)
                            metrics['correct_predictions'] += correct
                            
                            # Calculate losses
                            loss = model.loss(outputs, labels_batch)
                            metrics['total_loss'] += loss.item()
                            
                            # Calculate bias loss using all nodes in mini-batch
                            try:
                                batch_attrs = [node.attributes for node in mini_batch if hasattr(node, 'attributes')]
                                if batch_attrs:
                                    bias_loss = self.bias_loss(outputs, batch_attrs)
                                    metrics['total_bias_loss'] += bias_loss.item() if isinstance(bias_loss, torch.Tensor) else 0
                            except Exception as e:
                                print(f"Error calculating test bias loss for model {model_idx}: {e}")
                            
                            metrics['num_batches'] += 1
                    
                    # Update progress with average metrics across models
                    avg_loss = sum(m['total_loss']/max(m['num_batches'],1) for m in metrics_per_model) / len(self.models)
                    avg_acc = sum(m['correct_predictions']/max(m['total_predictions'],1) for m in metrics_per_model) / len(self.models)
                    progress_bar.update(len(mini_batch))
                    progress_bar.set_description(
                        f"Testing Progress | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}"
                    )
            
            progress_bar.close()
            
            # Return separate metrics for each model
            return [{
                'avg_loss': metrics['total_loss'] / max(metrics['num_batches'], 1),
                'avg_bias_loss': metrics['total_bias_loss'] / max(metrics['num_batches'], 1),
                'accuracy': metrics['correct_predictions'] / max(metrics['total_predictions'], 1)
            } for metrics in metrics_per_model]
            
        except Exception as e:
            progress_bar.close()
            print(f"Error in testing: {e}")
            return [{
                'avg_loss': 0.0,
                'avg_bias_loss': 0.0,
                'accuracy': 0.0
            } for _ in self.models]

    def _get_empty_metrics(self, epoch):
        """Return empty metrics structure for when no valid data is processed."""
        return {
            'epoch': epoch,
            'avg_loss': 0.0,
            'accuracy': 0.0,
            'avg_bias_loss': 0.0
        }
        
    def run(self, num_epochs=15):
        """Run training with early stopping and reduced validation."""
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            try:
                # Training phase
                train_metrics = self.train(epoch)
                if train_metrics is None:
                    train_metrics = self._get_empty_metrics(epoch)
                
                # Validation phase
                val_metrics = self.val(epoch)
                if val_metrics is None:
                    val_metrics = self._get_empty_metrics(epoch)
                
                # Log metrics
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_metrics.get('avg_loss', 0.0),
                    'train_accuracy': train_metrics.get('accuracy', 0.0),
                    'train_bias_loss': train_metrics.get('avg_bias_loss', 0.0),
                    'val_loss': val_metrics.get('avg_loss', 0.0),
                    'val_accuracy': val_metrics.get('accuracy', 0.0),
                    'val_bias_loss': val_metrics.get('avg_bias_loss', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Log metrics
                self.log_metrics(metrics)
                
                # Early stopping check
                current_val_loss = val_metrics.get('avg_loss', float('inf'))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    # Save best model
                    for i, model in enumerate(self.models):
                        model.save()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                        
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
                
    def test_run(self):
        """Run evaluation on the test set."""
        try:
            # Load best models
            for model in self.models:
                model.load()
                
            # Run test
            test_metrics = self.test()
            if test_metrics is None:
                test_metrics = self._get_empty_metrics(-1)
                
            print("\nTest Results:")
            for i, metrics in enumerate(test_metrics):
                print(f"Model {i+1}:")
                print(f"Loss: {metrics.get('avg_loss', 0.0):.4f}")
                print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                print(f"Bias Loss: {metrics.get('avg_bias_loss', 0.0):.4f}")
            
            return test_metrics
            
        except Exception as e:
            print(f"Error in test run: {str(e)}")
            return self._get_empty_metrics(-1)
    
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
