import torch
import torch.nn.functional as F
import random
import time
from collections import defaultdict, deque
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
import os

from models.DQNModel import DQNModel
from utils.attribute_utils import AttributeMetadata, AttributeBiasLoss
from nodes.atrnode import AttributeNode


class DQNCapability:
    """Encapsulates all DQN-related functionality."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        self.attribute_metadata = trainer.attribute_metadata
        # Respect selected DQN model type on the trainer (default to 'basic')
        self.dqn_model_type = getattr(trainer, 'dqn_model_type', 'basic')
        print(f"DQNCapability: Using DQN model type '{self.dqn_model_type}'")
        
        # DQN settings
        self.embedding_dim = 512
        self.feature_dim = None
        self.dqns = []
        
        # Training settings
        self.batch_size = 32
        self.gradient_accumulation_steps = 8
        self.max_nodes_per_epoch = 10000
        self.scaler = GradScaler()
        
        # Prediction stats for bias tracking
        self.prediction_stats = defaultdict(lambda: defaultdict(list))
        
        # Initialize DQN models if attribute metadata exists
        if self.attribute_metadata:
            self._initialize_dqns()
        else:
            print("DQNCapability: No attribute metadata provided. DQN not initialized.")
            
    def _initialize_dqns(self):
        """Initialize DQN models based on attribute metadata and selected model type."""
        try:
            # Get a sample node to determine feature dimensions
            sample_nodes = list(self.trainer.graphmanager.get_graph().get_nodes())
            if not sample_nodes:
                raise ValueError("No nodes available for DQN initialization")
                
            sample_node = sample_nodes[0]
            
            # Calculate feature dimensions
            features_tensor, embedding_tensor = self._get_dqn_features(sample_node)
            if features_tensor is None:
                raise ValueError("Could not extract features from sample node")
                
            self.feature_dim = features_tensor.shape[0]
            print(f"DQNCapability: Calculated feature dimension: {self.feature_dim}")
            
            # Initialize DQN for each model
            for i, model in enumerate(self.trainer.models):
                # Select model class based on configuration
                dqn = None
                model_type = (self.dqn_model_type or 'basic').lower()
                if model_type == 'basic':
                    dqn = DQNModel(
                        feature_dim=self.feature_dim,
                        embedding_dim=self.embedding_dim,
                        device=self.device
                    )
                elif model_type == 'residual':
                    from models.EnhancedDQNModels import ResidualDQNModel
                    dqn = ResidualDQNModel(
                        feature_dim=self.feature_dim,
                        device=self.device,
                        embedding_dim=self.embedding_dim
                    )
                elif model_type == 'attention':
                    from models.EnhancedDQNModels import AttentionDQNModel
                    dqn = AttentionDQNModel(
                        feature_dim=self.feature_dim,
                        device=self.device,
                        embedding_dim=self.embedding_dim
                    )
                elif model_type == 'conv_embedding':
                    from models.EnhancedDQNModels import ConvEmbeddingDQN
                    dqn = ConvEmbeddingDQN(
                        feature_dim=self.feature_dim,
                        device=self.device,
                        embedding_dim=self.embedding_dim
                    )
                elif model_type == 'ensemble':
                    from models.EnhancedDQNModels import EnsembleDQNModel
                    dqn = EnsembleDQNModel(
                        feature_dim=self.feature_dim,
                        device=self.device,
                        embedding_dim=self.embedding_dim
                    )
                else:
                    print(f"DQNCapability: Unknown dqn_model_type '{self.dqn_model_type}', falling back to 'basic'.")
                    dqn = DQNModel(
                        feature_dim=self.feature_dim,
                        embedding_dim=self.embedding_dim,
                        device=self.device
                    )
                self.dqns.append(dqn)
                print(f"DQNCapability: Initialized DQN {i} (type={model_type}) with feature_dim={self.feature_dim}")
                
        except Exception as e:
            print(f"DQNCapability: Error initializing DQN: {e}. DQN not initialized.")
            self.dqns = []
            
    def _get_dqn_features(self, node):
        """Extract attribute features for DQN input."""
        try:
            if not isinstance(node, AttributeNode):
                print(f"Warning: Expected AttributeNode, got {type(node)}. Cannot extract DQN features.")
                return None, None

            features_list = []
            embedding_data = None

            # Extract standard attributes based on metadata
            if self.attribute_metadata:
                for attr_meta in self.attribute_metadata:
                    attr_name = attr_meta['name'] if isinstance(attr_meta, dict) else attr_meta.name
                    attr_type = attr_meta['type'] if isinstance(attr_meta, dict) else attr_meta.attr_type

                    # Special handling for face embedding
                    if attr_name == 'face_embedding':
                        embedding_data = node.attributes.get(attr_name)
                        continue

                    # Handle other attributes
                    if attr_type == 'categorical':
                        # One-hot encode categorical values
                        possible_values = (attr_meta.get('possible_values') if isinstance(attr_meta, dict) 
                                         else attr_meta.possible_values)
                        if possible_values:
                            for possible_value in possible_values:
                                features_list.append(1.0 if node.attributes.get(attr_name) == possible_value else 0.0)
                    else:  # continuous
                        try:
                            features_list.append(float(node.attributes.get(attr_name, 0)))
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not convert attribute '{attr_name}' value '{node.attributes.get(attr_name)}' to float: {e}. Using 0.")
                            features_list.append(0.0)
            elif isinstance(node, AttributeNode):
                # Fallback: Use default features if no metadata
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

            return features_tensor, embedding_tensor

        except Exception as e:
            print(f"Error extracting DQN features: {e}")
            return None, None
            
    def get_i_value(self, node, model_idx=0):
        """Calculate I-value using DQN."""
        try:
            # Check if DQN models are available
            if not self.dqns or model_idx >= len(self.dqns):
                print(f"Warning: DQN model {model_idx} not available for I-value calculation.")
                return 0.0

            # Get the target DQN model and its device
            dqn_model = self.dqns[model_idx]
            target_device = dqn_model.device

            # Retrieve features and embedding for the node
            features_tensor, embedding_tensor = self._get_dqn_features(node)

            # Check if feature extraction was successful
            if features_tensor is None:
                print(f"Warning: Could not extract features for node {node.node_id}. Cannot calculate I-value.")
                return 0.0

            # Handle potentially missing embedding tensor
            if embedding_tensor is None:
                embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            elif not isinstance(embedding_tensor, torch.Tensor):
                try:
                    embedding_tensor = torch.tensor(embedding_tensor, dtype=torch.float32, device=target_device)
                except Exception as e:
                    print(f"Error converting embedding to tensor for node {node.node_id}: {e}. Using zeros.")
                    embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            else:
                embedding_tensor = embedding_tensor.to(target_device)

            # Features should already be a tensor
            features_tensor = features_tensor.to(target_device)

            # Add batch dimension
            features_tensor = features_tensor.unsqueeze(0)
            embedding_tensor = embedding_tensor.unsqueeze(0)

            # Get I-value from DQN
            i_value = dqn_model.predict_i_value(features_tensor.to(dqn_model.device), 
                                                embedding_tensor.to(dqn_model.device))

            i_value = i_value.detach().cpu().item()
            
            # Update prediction stats
            self.update_prediction_stats(node, i_value > 0.5, model_idx)
            
            return i_value
            
        except Exception as e:
            print(f"Error calculating I-value: {str(e)}")
            return 0.0
            
    def update_prediction_stats(self, node, correct, model_idx):
        """Update prediction statistics for each attribute value."""
        if not self.attribute_metadata:
            return
            
        node_attrs = node.attributes
        
        for attr in self.attribute_metadata:
            attr_name = attr['name'] if isinstance(attr, dict) else attr.name
            if attr_name in node_attrs:
                value = node_attrs[attr_name]
                attr_type = attr['type'] if isinstance(attr, dict) else attr.attr_type
                if attr_type == 'categorical':
                    self.prediction_stats[f'model_{model_idx}_{attr_name}'][value].append(float(correct))
                    
    def train_with_dqn(self, traversal, epoch=None):
        """Training loop with DQN integration."""
        try:
            # Set models to training mode
            for model in self.trainer.models:
                model.train()
            for dqn in self.dqns:
                dqn.train()
                
            # Initialize metrics
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            total_train_bias_loss = 0.0

            # Reset traversal for this epoch
            traversal.reset_pointers()
            
            # Get total nodes for this epoch
            total_nodes = traversal.num_steps
            print(f"Training on {total_nodes} nodes this epoch with DQN")
            
            # Track attribute distribution
            attribute_distribution = defaultdict(lambda: defaultdict(int))
            track_attributes = bool(self.trainer.categorical_attrs_for_tracking)
            
            nodes_processed = 0
            pbar = tqdm(total=min(total_nodes, self.max_nodes_per_epoch), desc="DQN Training")
            
            while nodes_processed < min(total_nodes, self.max_nodes_per_epoch):
                try:
                    # Get batch from traversal
                    batch_nodes = traversal.traverse(self.batch_size)
                    if not batch_nodes:
                        break
                        
                    # Preprocess batch
                    images, batch_nodes_loaded = self._preprocess_batch(batch_nodes)
                    if images is None or not batch_nodes_loaded:
                        continue
                        
                    # Extract labels
                    batch_labels_loaded = [float(node.get_label()) for node in batch_nodes_loaded]
                    batch_labels_tensor = torch.tensor(batch_labels_loaded, dtype=torch.float).unsqueeze(1).to(self.device)
                    
                    # Forward pass
                    outputs = self.trainer.models[0](images)
                    loss = self.trainer.criterion(outputs, batch_labels_tensor)
                    # Add bias loss if available
                    bias_loss_val = 0.0
                    bias_weight = 0.0
                    bias_loss_fn = self.trainer.capabilities.get_bias_loss()
                    if bias_loss_fn:
                        try:
                            bias_loss_val = bias_loss_fn(outputs, batch_labels_tensor, batch_nodes_loaded)
                            bias_weight = getattr(self.trainer.capabilities.bias_capability, 'bias_weight', 0.0)
                            total_train_bias_loss += bias_loss_val.item()
                        except Exception as e:
                            print(f"Warning: Error calculating bias loss: {e}")
                    # Combine losses
                    total_loss_for_backward = loss + bias_weight * bias_loss_val
                    # Calculate metrics
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == batch_labels_tensor).sum().item()
                    total_loss += loss.item()
                    total += len(batch_labels_loaded)
                    # Track attribute distribution
                    if track_attributes:
                        for node in batch_nodes_loaded:
                            for attr_name in self.trainer.categorical_attrs_for_tracking:
                                if attr_name in node.attributes:
                                    attr_value = node.attributes[attr_name]
                                    attribute_distribution[attr_name][attr_value] += 1
                    # Backward pass
                    self.scaler.scale(total_loss_for_backward).backward()
                    self.scaler.step(self.trainer.models[0].optim)
                    self.scaler.update()
                    self.trainer.models[0].optim.zero_grad()
                    
                    # DQN Training Integration
                    if self.dqns:
                        self._train_dqn_on_batch(batch_nodes_loaded, outputs, batch_labels_loaded)
                    
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
                'avg_bias_loss': total_train_bias_loss / batch_count if total_train_bias_loss > 0 else 0.0
            }
            
            return metrics, attribute_distribution
            
        except Exception as e:
            print(f"Error in DQN training: {str(e)}")
            return self._get_empty_metrics()
            
    def _preprocess_batch(self, batch_nodes):
        """Preprocess a batch of nodes to ensure consistent tensor sizes."""
        if not batch_nodes:
            return None, None
        
        try:
            # Get CNN model for transforms
            cnn_model = None
            for model in self.trainer.models:
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
                            transform = self.trainer.models[0].transform
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
            
    def _train_dqn_on_batch(self, batch_nodes_loaded, outputs, batch_labels_loaded):
        """Train DQN models on a batch of nodes."""
        if not self.dqns:
            return
            
        dqn_model = self.dqns[0]  # Use first DQN for now
        
        for i, node in enumerate(batch_nodes_loaded):
            # Calculate reward for DQN
            prediction_probability = torch.sigmoid(outputs[i]).item()
            is_correct = (prediction_probability > 0.5) == (batch_labels_loaded[i] > 0.5)
            
            # Calculate confidence-based reward
            confidence = abs(prediction_probability - 0.5) * 2
            reward_sign = 1.0 if is_correct else -1.0
            dqn_reward = reward_sign * confidence
        
            # Get DQN state
            dqn_features, dqn_embedding = self._get_dqn_features(node)
        
            if dqn_features is None:
                continue
        
            # Move features to DQN device
            dqn_features = dqn_features.to(dqn_model.device) 
            if dqn_embedding is not None:
                dqn_embedding = dqn_embedding.to(dqn_model.device)
        
            # Push experience to replay buffer
            dqn_model.replay_buffer.append((
                dqn_features.detach(), 
                dqn_embedding.detach() if dqn_embedding is not None else None, 
                dqn_reward
            ))

        # Perform DQN learning step if buffer is large enough
        if len(dqn_model.replay_buffer) >= dqn_model.batch_size:
            dqn_transitions = random.sample(dqn_model.replay_buffer, dqn_model.batch_size)
            dqn_model.train_step(dqn_transitions)
            
    def _get_empty_metrics(self):
        """Return empty metrics structure for when no valid data is processed."""
        return {
            'avg_loss': 0.0,
            'accuracy': 0.0,
            'avg_bias_loss': 0.0
        }
    
    def save_checkpoint(self, checkpoint_path):
        """Save DQN models to checkpoint."""
        try:
            if self.dqns:
                # For now, save the first DQN model
                # In the future, we could save all DQN models
                self.dqns[0].save_checkpoint(checkpoint_path)
                print(f"DQN checkpoint saved to {checkpoint_path}")
                return True
            else:
                print("No DQN models to save")
                return False
        except Exception as e:
            print(f"Error saving DQN checkpoint: {e}")
            return False
            
    def load_checkpoint(self, checkpoint_path):
        """Load DQN models from checkpoint."""
        try:
            if self.dqns:
                # Check if file exists before attempting load
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: DQN Checkpoint file not found at {checkpoint_path}. Skipping load.")
                    return False
                    
                # Load into the first DQN model
                self.dqns[0].load_checkpoint(checkpoint_path)
                print(f"DQN checkpoint loaded from {checkpoint_path}")
                return True
            else:
                print("No DQN models to load checkpoint into")
                return False
        except Exception as e:
            print(f"Error loading DQN checkpoint: {e}")
            return False 