import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from collections import defaultdict
import random

from trainers.capabilities.uncertainty_logging import (
    uncertainty_summary_for_logging as _uncertainty_summary_for_logging,
)


class BasicTrainingCapability:
    """Basic training functionality for simple traversals."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        
        # Training settings
        self.batch_size = 32
        self.max_nodes_per_epoch = 5000
        self.scaler = GradScaler()
        
        # Setup CUDA optimizations
        torch.cuda.empty_cache()
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
    def train_basic(self, traversal, epoch=None):
        """Basic training loop without DQN or bias correction."""
        try:
            # Set models to training mode
            for model in self.trainer.models:
                if hasattr(model, 'model'):
                    model.model.train()
                else:
                    model.train()
            
            # Initialize metrics
            running_loss = 0.0
            correct = 0
            total = 0
            label_counts = {0: 0, 1: 0}
            uncertainty_sums = defaultdict(float)
            uncertainty_counts = defaultdict(int)
            
            # Reset traversal state before each epoch
            if hasattr(traversal, 't'):
                traversal.t = 0
            if hasattr(traversal, 'steps_taken'):
                traversal.steps_taken = 0
            traversal.reset_pointers()
            
            # Get nodes from traversal
            try:
                nodes = []
                while True:
                    batch = traversal.traverse()
                    if not batch:
                        break
                    nodes.extend(batch)
                    if len(nodes) >= self.max_nodes_per_epoch:
                        break
                        
                if not nodes:
                    print("Warning: No nodes returned from traversal")
                    return self._get_empty_metrics()
                    
                # Limit number of nodes per epoch
                nodes = nodes[:self.max_nodes_per_epoch]
                print(f"Processing {len(nodes)} nodes for basic training")
                
                # Print label distribution
                for node in nodes:
                    label = node.label if hasattr(node, 'label') else node.get_label()
                    if label in label_counts:
                        label_counts[label] += 1
                print(f"Label distribution - Real (0): {label_counts[0]}, Fake (1): {label_counts[1]}")
                
            except Exception as e:
                print(f"Error during traversal: {str(e)}")
                return self._get_empty_metrics()
            
            # Track attribute distribution
            attribute_distribution = defaultdict(lambda: defaultdict(int))
            
            # Process in batches with memory management
            chunk_size = 16
            for i in tqdm(range(0, len(nodes), self.batch_size), desc=f"Basic Training Epoch {epoch if epoch else 'N/A'}", unit="batch"):
                try:
                    # Clear GPU cache before each batch
                    torch.cuda.empty_cache()
                    
                    batch_nodes = nodes[i:i + self.batch_size]
                    
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
                                            # Set model to training mode for transforms
                                            if hasattr(self.trainer.models[0], 'current_mode'):
                                                self.trainer.models[0].current_mode = "train"
                                            # Use the model's internal transform
                                            img_tensor = self.trainer.models[0].transform(img_data)
                                            chunk_data.append(img_tensor) 
                                            chunk_labels.append(node.label if hasattr(node, 'label') else node.get_label())
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
                            chunk_tensor = torch.stack(chunk_data).to(self.device)
                            chunk_labels_tensor = torch.tensor(chunk_labels, dtype=torch.float32).to(self.device)
                            chunk_labels_tensor = chunk_labels_tensor.view(-1, 1)
                            
                            # Forward pass with mixed precision
                            with torch.cuda.amp.autocast():
                                model = self.trainer.models[0]
                                if hasattr(model, 'forward_with_uncertainty'):
                                    step_index = (i // self.batch_size) + (j // chunk_size)
                                    summarize_now = (
                                        step_index % max(1, model.uncertainty_train_frequency) == 0
                                    )
                                    # The loss always uses a single deterministic pass.
                                    # MC dropout used to run inside the loss path every
                                    # Nth batch, which made the optimization objective
                                    # change periodically -- a non-stationary loss. MC
                                    # statistics are now gathered separately below, under
                                    # no_grad, purely for logging.
                                    prediction_bundle = model.forward_with_uncertainty(
                                        chunk_tensor,
                                        nodes=chunk_nodes,
                                        update_precision=True,
                                        use_mc_dropout=False,
                                        compute_variance=summarize_now,
                                    )
                                    chunk_outputs = prediction_bundle.logits
                                    loss = model.compute_loss(
                                        prediction_bundle,
                                        chunk_labels_tensor,
                                        base_criterion=self.trainer.criterion,
                                    )
                                    if summarize_now:
                                        for name, value in _uncertainty_summary_for_logging(
                                            model, prediction_bundle, chunk_tensor, chunk_nodes
                                        ).items():
                                            uncertainty_sums[name] += float(value) * len(chunk_nodes)
                                            uncertainty_counts[name] += len(chunk_nodes)
                                else:
                                    chunk_outputs = model(chunk_tensor)
                                    loss = self.trainer.criterion(chunk_outputs, chunk_labels_tensor)
                                # Calculate bias loss if available
                                bias_loss_val = 0.0
                                bias_loss_fn = getattr(self.trainer.capabilities, 'get_bias_loss', None)
                                bias_weight = 0.0
                                if bias_loss_fn is not None:
                                    bias_loss = bias_loss_fn()
                                    if bias_loss is not None:
                                        try:
                                            bias_loss_val = bias_loss(chunk_outputs, chunk_labels_tensor, chunk_nodes)
                                            # Get bias weight from capability if present
                                            bias_weight = getattr(self.trainer.capabilities.bias_capability, 'bias_weight', 0.0)
                                        except Exception as e:
                                            print(f"Warning: Error calculating bias loss: {e}")
                                # Combine losses
                                total_loss = loss + bias_weight * bias_loss_val
                            
                            # Backward pass with gradient scaling
                            self.trainer.models[0].optim.zero_grad()
                            self.scaler.scale(total_loss).backward()
                            self.scaler.step(self.trainer.models[0].optim)
                            self.scaler.update()
                            
                            # Update metrics
                            running_loss += loss.item()
                            predicted = (torch.sigmoid(chunk_outputs) > 0.5).float()
                            total += chunk_labels_tensor.size(0)
                            correct += (predicted == chunk_labels_tensor).sum().item()
                            
                            # Track attribute distribution
                            for node in chunk_nodes:
                                if hasattr(node, 'attributes') and self.trainer.categorical_attrs_for_tracking:
                                    for attr_name in self.trainer.categorical_attrs_for_tracking:
                                        if attr_name in node.attributes:
                                            attr_value = node.attributes[attr_name]
                                            attribute_distribution[attr_name][attr_value] += 1
                            
                            # Clear chunk tensors
                            del chunk_tensor, chunk_outputs, chunk_labels_tensor, predicted
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing chunk: {str(e)}")
                            continue
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: GPU out of memory, clearing cache and skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    raise e
            
            # Calculate epoch metrics
            epoch_loss = running_loss / (total / self.batch_size) if total > 0 else float('inf')
            epoch_acc = 100 * correct / total if total > 0 else 0
            
            # Step the scheduler if available
            if hasattr(self.trainer.models[0], 'scheduler'):
                self.trainer.models[0].scheduler.step(epoch_loss)
            
            print(f"Basic Training - Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
            
            # Prepare metrics
            metrics = {
                'avg_loss': epoch_loss,
                'accuracy': epoch_acc / 100.0,  # Convert to decimal for consistency
                'train_loss': epoch_loss,
                'train_acc': epoch_acc
            }
            if uncertainty_sums:
                metrics['uncertainty_summary'] = {
                    name: uncertainty_sums[name] / max(1, uncertainty_counts[name])
                    for name in uncertainty_sums
                }
            
            return metrics, attribute_distribution
            
        except Exception as e:
            print(f"Error in basic training: {str(e)}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self):
        """Return empty metrics structure for when no valid data is processed."""
        empty_metrics = {
            'avg_loss': 0.0,
            'accuracy': 0.0,
            'train_loss': 0.0,
            'train_acc': 0.0
        }
        empty_distribution = defaultdict(lambda: defaultdict(int))
        return empty_metrics, empty_distribution 
