from trainers.Trainer import Trainer
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
import json
import cv2
from torch.cuda.amp import GradScaler
from collections import defaultdict
from traversals.IValueTraversal import IValueTraversal
from traversals.RandomTraversal import RandomTraversal
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from models.DQNModel import DQNModel

class ExperimentTrainer(Trainer):
    """
    Trainer class that supports both IValueTraversal and RandomTraversal for experimental comparison.
    """
    tags = ["experiment"]
    
    def __init__(self, graphmanager, train_traversal, test_traversal, val_traversal, models, traversal_type="random", attribute_metadata=None):
        """Initialize the trainer with specified traversal type."""
        super().__init__(graphmanager, train_traversal, test_traversal, models)
        
        self.traversal_type = traversal_type
        self.attribute_metadata = attribute_metadata
        self.val_traversal = val_traversal
        
        # Training settings
        self.batch_size = 32  # Reduced from 128 to handle memory better
        self.max_nodes_per_epoch = 5000  # Reduced from 10000 to handle memory better
        self.max_batch_memory = 1024 * 1024 * 1024  # 1GB max memory per batch
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        torch.cuda.empty_cache()  # Clear GPU cache before starting
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Setup optims and learning rate
        self.learning_rate = 0.001
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(model.optim, mode='min', factor=0.5, patience=2) 
                          for model in models]
        
        # Setup DQN only for I-value traversal
        if self.traversal_type == "i-value" and attribute_metadata is not None:
            sample_node = next(iter(self.graphmanager.get_graph().get_nodes()))
            input_dim = len(self._get_node_features(sample_node))
            self.dqns = [DQNModel(input_dim).cuda() for _ in self.models]
        else:
            self.dqns = None
            
        # Setup logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"experiment_{self.traversal_type}_{timestamp}.json"
        self.metrics_history = []

    def _get_node_features(self, node):
        """Extract features from node attributes."""
        if not hasattr(node, 'attributes'):
            return torch.zeros(1)
            
        features = []
        for attr_meta in self.attribute_metadata:
            attr_value = node.attributes.get(attr_meta['name'], 0)
            if attr_meta['type'] == 'categorical':
                # One-hot encoding for categorical variables
                one_hot = [1 if i == attr_value else 0 for i in attr_meta['possible_values']]
                features.extend(one_hot)
            else:
                features.append(float(attr_value))
                
        return torch.tensor(features, dtype=torch.float32)

    def get_traversal(self, graph, num_pointers=5, num_steps=100):
        """Create appropriate traversal based on type."""
        if self.traversal_type == "i-value":
            return IValueTraversal(
                graph,
                num_pointers,
                num_steps,
                self
            )
        elif self.traversal_type == "comprehensive":
            return ComprehensiveTraversal(
                graph,
                num_pointers,
                num_steps
            )
        else:  # random traversal
            return RandomTraversal(
                graph,
                num_pointers,
                num_steps
            )

    def get_i_value(self, node, model_idx):
        """Calculate I-value if using I-value traversal, else return random value."""
        if self.traversal_type == "i-value" and self.dqns is not None:
            features = self._get_node_features(node).cuda()
            with torch.no_grad():
                q_value = self.dqns[model_idx](features.unsqueeze(0))
                return 1.0 - q_value.item()  # Convert Q-value to I-value
        else:
            return np.random.random()  # Random value for random traversal

    def train(self, epoch):
        """Train for one epoch."""
        for model_idx, model in enumerate(self.models):
            model.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            label_counts = {0: 0, 1: 0}  # Track distribution of labels
            
            # Reset traversal state before each epoch
            if hasattr(self.train_traversal, 't'):
                self.train_traversal.t = 0
            if hasattr(self.train_traversal, 'steps_taken'):
                self.train_traversal.steps_taken = 0
            self.train_traversal.reset_pointers()
            
            # Get nodes from traversal
            try:
                nodes = []
                while True:
                    batch = self.train_traversal.traverse()
                    if not batch:
                        break
                    nodes.extend(batch)
                    if len(nodes) >= self.max_nodes_per_epoch:
                        break
                        
                if not nodes:
                    print("Warning: No nodes returned from traversal")
                    continue
                    
                # Limit number of nodes per epoch
                nodes = nodes[:self.max_nodes_per_epoch]
                print(f"Processing {len(nodes)} nodes for epoch {epoch}")
                
                # Print label distribution
                for node in nodes:
                    label = node.label
                    if label in label_counts:
                        label_counts[label] += 1
                print(f"Label distribution - Real (0): {label_counts[0]}, Fake (1): {label_counts[1]}")
                
            except Exception as e:
                print(f"Error during traversal: {str(e)}")
                continue
            
            # Process in batches with memory management
            chunk_size = 16  # Process at most 16 images at once
            for i in range(0, len(nodes), self.batch_size):
                try:
                    # Clear GPU cache before each batch
                    torch.cuda.empty_cache()
                    
                    batch_nodes = nodes[i:i + self.batch_size]
                    
                    # Prepare batch data with memory check
                    batch_data = []
                    batch_labels = []
                    batch_outputs = []
                    
                    # Process nodes in chunks to manage memory
                    for j in range(0, len(batch_nodes), chunk_size):
                        try:
                            chunk_nodes = batch_nodes[j:j + chunk_size]
                            chunk_data = []
                            chunk_labels = []
                            
                            # Load images for chunk
                            for node in chunk_nodes:
                                try:
                                    data = node.get_data()
                                    if data is not None:
                                        img_data = data.load_data()
                                        if img_data is not None:
                                            chunk_data.append(img_data)
                                            chunk_labels.append(node.label)
                                        else:
                                            print(f"Warning: Could not load image data for node")
                                    else:
                                        print(f"Warning: No data for node")
                                except Exception as e:
                                    print(f"Error processing node: {str(e)}")
                                    continue
                            
                            if not chunk_data:  # Skip if no valid images in chunk
                                continue
                                
                            # Convert chunk to tensors
                            chunk_tensor = torch.stack([model.transform(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)) for d in chunk_data]).cuda()
                            chunk_labels_tensor = torch.tensor(chunk_labels, dtype=torch.float32).cuda()
                            chunk_labels_tensor = chunk_labels_tensor.view(-1, 1)  # Reshape to [batch_size, 1]
                            
                            # Forward pass with mixed precision
                            with torch.cuda.amp.autocast():
                                chunk_outputs = model(chunk_tensor)
                                loss = model.loss(chunk_outputs, chunk_labels_tensor)
                            
                            # Backward pass with gradient scaling
                            model.optim.zero_grad()
                            self.scaler.scale(loss).backward()
                            self.scaler.step(model.optim)
                            self.scaler.update()
                            
                            # Update metrics
                            running_loss += loss.item()
                            predicted = (torch.sigmoid(chunk_outputs) > 0.5).float()
                            total += chunk_labels_tensor.size(0)
                            correct += (predicted == chunk_labels_tensor).sum().item()
                            
                            # Clear chunk tensors
                            del chunk_tensor, chunk_outputs, chunk_labels_tensor, predicted
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing chunk: {str(e)}")
                            continue
                    
                    # Update DQN for I-value traversal if needed
                    if self.traversal_type == "i-value" and self.dqns is not None:
                        self._update_dqn(batch_nodes, None, model_idx)  # No predictions available here
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: GPU out of memory, clearing cache and skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    raise e
            
            # Calculate epoch metrics
            epoch_loss = running_loss / (total / self.batch_size) if total > 0 else float('inf')
            epoch_acc = 100 * correct / total if total > 0 else 0
            
            # Step the scheduler with epoch loss
            model.scheduler.step(epoch_loss)
            
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
            
            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'traversal_type': self.traversal_type
            }
            self.log_metrics(metrics)

    def _update_dqn(self, nodes, correct_predictions, model_idx):
        """Update DQN based on prediction correctness."""
        if self.dqns is None:
            return
            
        for node, correct in zip(nodes, correct_predictions):
            features = self._get_node_features(node).cuda()
            target = float(correct)
            
            # Update DQN
            self.dqns[model_idx].train()
            q_value = self.dqns[model_idx](features.unsqueeze(0))
            loss = nn.MSELoss()(q_value, torch.tensor([[target]], device='cuda'))
            loss.backward()
            
            # Update stored prediction accuracy
            self.stored_prediction_accuracy[model_idx][node].append(float(correct))

    def log_metrics(self, metrics):
        """Log metrics to file."""
        self.metrics_history.append(metrics)
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def val(self):
        """Validate the models."""
        print(f"\nValidating models trained with {self.traversal_type} traversal")
        
        # Reset validation traversal
        self.val_traversal.reset_pointers()
        
        for model_idx, model in enumerate(self.models):
            model.eval()  # Use model's eval() method which handles both model and transforms
            correct = 0
            total = 0
            running_loss = 0.0
            num_batches = 0
            label_counts = {0: 0, 1: 0}  # Track distribution of labels
            
            with torch.no_grad():
                # Get validation nodes in batches
                pbar = tqdm(desc="Validating")
                while True:
                    batch_nodes = self.val_traversal.traverse(batch_size=self.batch_size)
                    if not batch_nodes:
                        break
                        
                    batch_data = []
                    batch_labels = []
                    
                    # Prepare batch data
                    for node in batch_nodes:
                        try:
                            data = node.get_data()
                            if data is None:
                                continue
                                
                            # Load the actual image data
                            img_data = data.load_data()
                            if img_data is None:
                                continue
                                
                            batch_data.append(img_data)
                            batch_labels.append(node.label)
                            
                            # Update label counts
                            if node.label in label_counts:
                                label_counts[node.label] += 1
                            
                        except Exception as e:
                            print(f"Error processing validation node: {str(e)}")
                            continue
                    
                    if not batch_data:
                        continue
                        
                    try:
                        # Convert to tensors
                        batch_tensor = torch.stack([model.transform(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)) for d in batch_data]).cuda()
                        labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).cuda()
                        labels_tensor = labels_tensor.view(-1, 1)  # Reshape to [batch_size, 1]
                        
                        # Get predictions
                        outputs = model(batch_tensor)  # Use model's __call__ method
                        outputs = outputs.view(labels_tensor.shape)
                        
                        # Calculate loss without scaling
                        with torch.cuda.amp.autocast():
                            loss = model.loss(outputs, labels_tensor)
                        
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        
                        # Update metrics
                        running_loss += loss.item()  # Don't multiply by batch size
                        num_batches += 1
                        total += len(batch_data)
                        correct += (predicted == labels_tensor).sum().item()
                        
                        # Update progress bar
                        pbar.update(len(batch_data))
                        pbar.set_description(f"Validating ({total} samples)")
                        
                    except Exception as e:
                        print(f"Error processing validation batch: {str(e)}")
                        continue
                
                pbar.close()
            
            # Print label distribution
            print(f"Validation label distribution - Real (0): {label_counts[0]}, Fake (1): {label_counts[1]}")
            
            # Calculate and log validation metrics
            val_loss = running_loss / num_batches if num_batches > 0 else 0  # Average over batches
            val_acc = 100 * correct / total if total > 0 else 0
            
            # Update learning rate scheduler based on validation loss
            model.scheduler.step(val_loss)
            
            metrics = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'traversal_type': self.traversal_type,
                'total_samples': total
            }
            self.log_metrics(metrics)
            
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}% on {total} samples')
            
            # Set model back to training mode
            model.train()  # Use model's train() method

    def test(self, num_samples=None):
        """Test the model."""
        print(f"\nTesting models trained with {self.traversal_type} traversal")
        for model_idx, model in enumerate(self.models):
            model.eval()
            correct = 0
            total = 0
            label_counts = {0: 0, 1: 0}  # Track distribution of labels
            
            try:
                # Get test nodes
                test_nodes = list(self.test_traversal.traverse())
                if num_samples and len(test_nodes) > num_samples:
                    test_nodes = random.sample(test_nodes, num_samples)
                
                # Process in batches
                for i in range(0, len(test_nodes), self.batch_size):
                    batch_nodes = test_nodes[i:i + self.batch_size]
                    batch_data = []
                    batch_labels = []
                    
                    # Load images for batch
                    for node in batch_nodes:
                        try:
                            data = node.get_data()
                            if data is not None:
                                img_data = data.load_data()
                                if img_data is not None:
                                    batch_data.append(img_data)
                                    batch_labels.append(node.label)
                                else:
                                    print(f"Warning: Could not load test image data")
                            else:
                                print(f"Warning: No test data for node")
                        except Exception as e:
                            print(f"Error processing test node: {str(e)}")
                            continue
                    
                    if not batch_data:
                        continue
                        
                    try:
                        # Convert to tensors
                        batch_tensor = torch.stack([model.transform(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)) for d in batch_data]).cuda()
                        labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).cuda()
                        labels_tensor = labels_tensor.view(-1, 1)  # Reshape to [batch_size, 1]
                        
                        # Get predictions
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_tensor)
                            predicted = (torch.sigmoid(outputs) > 0.5).float()
                        
                        # Update metrics
                        total += labels_tensor.size(0)
                        correct += (predicted == labels_tensor).sum().item()
                        
                        # Update label counts
                        for label in batch_labels:
                            if label in label_counts:
                                label_counts[label] += 1
                        
                        # Clear memory
                        del batch_tensor, outputs, labels_tensor, predicted
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error during testing: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error during testing: {str(e)}")
                continue
            
            # Calculate and log test metrics
            test_acc = 100 * correct / total if total > 0 else 0
            metrics = {
                'test_acc': test_acc,
                'traversal_type': self.traversal_type
            }
            self.log_metrics(metrics)
            
            print(f'Test Accuracy: {test_acc:.2f}%')

    def run(self, num_epochs=15):
        """Run training with specified number of epochs."""
        print(f"Starting training with {self.traversal_type} traversal")
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.train(epoch)
            self.val()  # Add validation step after each epoch
            
        print(f"Completed training with {self.traversal_type} traversal")
