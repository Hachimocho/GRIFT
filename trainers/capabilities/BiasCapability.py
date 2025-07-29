import torch
from utils.attribute_utils import AttributeMetadata, AttributeBiasLoss


class BiasCapability:
    """Encapsulates bias measurement and correction functionality."""
    
    def __init__(self, trainer, attribute_weights=None):
        self.trainer = trainer
        self.device = trainer.device
        self.attribute_metadata = trainer.attribute_metadata
        
        # Bias loss components
        self.bias_loss = None
        self.attr_map = {}
        self.bias_weight = 1.0
        self.attribute_weights = attribute_weights
        
        if self.attribute_metadata:
            self._initialize_bias_loss()
        else:
            print("BiasCapability: No attribute metadata provided. Bias loss not initialized.")
            
    def _initialize_bias_loss(self):
        """Initialize bias loss computation."""
        try:
            # Process attribute_metadata dict list into AttributeMetadata objects
            processed_metadata = []
            for attr in self.attribute_metadata:
                if isinstance(attr, dict):
                    # Convert dict to AttributeMetadata object
                    attr_meta = AttributeMetadata(
                        name=attr['name'],
                        attr_type=attr['type'],
                        possible_values=attr.get('possible_values', None)
                    )
                    processed_metadata.append(attr_meta)
                else:
                    # Already an AttributeMetadata object
                    processed_metadata.append(attr)
            
            # Create attribute map for efficient lookup
            self.attr_map = {attr.name: attr for attr in processed_metadata}
            
            # Set up per-attribute weights (default 1.0 for all attributes)
            if self.attribute_weights is None:
                self.attribute_weights = {attr.name: 1.0 for attr in processed_metadata}
            
            # Initialize bias loss function
            self.bias_loss = AttributeBiasLoss(processed_metadata, self.attr_map, self.attribute_weights).to(self.device)
            
            print(f"BiasCapability: Initialized bias loss for {len(processed_metadata)} attributes with weights: {self.attribute_weights}")
            
        except Exception as e:
            print(f"BiasCapability: Error initializing bias loss: {e}")
            self.bias_loss = None
            
    def get_bias_loss(self):
        """Get the bias loss function if available."""
        return self.bias_loss
        
    def calculate_bias_loss(self, outputs, labels, nodes):
        """Calculate bias loss for a batch of nodes."""
        if not self.bias_loss:
            return torch.tensor(0.0, device=self.device)
            
        try:
            return self.bias_loss(outputs, labels, nodes)
        except Exception as e:
            print(f"BiasCapability: Error calculating bias loss: {e}")
            return torch.tensor(0.0, device=self.device)
            
    def get_bias_metrics(self, predictions, labels, nodes):
        """Calculate bias metrics for evaluation."""
        if not self.attribute_metadata:
            return {}
            
        try:
            bias_metrics = {}
            
            # Group predictions by attribute values
            attribute_groups = {}
            for i, node in enumerate(nodes):
                if hasattr(node, 'attributes'):
                    for attr_meta in self.attribute_metadata:
                        attr_name = attr_meta['name'] if isinstance(attr_meta, dict) else attr_meta.name
                        attr_type = attr_meta['type'] if isinstance(attr_meta, dict) else attr_meta.attr_type
                        
                        if attr_type == 'categorical' and attr_name in node.attributes:
                            attr_value = node.attributes[attr_name]
                            group_key = f"{attr_name}_{attr_value}"
                            
                            if group_key not in attribute_groups:
                                attribute_groups[group_key] = {'predictions': [], 'labels': []}
                                
                            attribute_groups[group_key]['predictions'].append(predictions[i])
                            attribute_groups[group_key]['labels'].append(labels[i])
            
            # Calculate accuracy for each group
            group_accuracies = {}
            for group_key, data in attribute_groups.items():
                if len(data['predictions']) > 0:
                    correct = sum(1 for p, l in zip(data['predictions'], data['labels']) if p == l)
                    accuracy = correct / len(data['predictions'])
                    group_accuracies[group_key] = accuracy
            
            # Calculate overall bias metrics
            if group_accuracies:
                accuracies = list(group_accuracies.values())
                bias_metrics['group_accuracies'] = group_accuracies
                bias_metrics['min_accuracy'] = min(accuracies)
                bias_metrics['max_accuracy'] = max(accuracies)
                bias_metrics['accuracy_range'] = max(accuracies) - min(accuracies)
                bias_metrics['mean_accuracy'] = sum(accuracies) / len(accuracies)
            
            return bias_metrics
            
        except Exception as e:
            print(f"BiasCapability: Error calculating bias metrics: {e}")
            return {}
    
    def save_checkpoint(self, checkpoint_path):
        """Save bias capability state (mainly configuration)."""
        try:
            # For BiasCapability, there's not much state to save beyond configuration
            # The bias loss function is recreated from metadata
            checkpoint_data = {
                'bias_weight': self.bias_weight,
                'has_bias_loss': self.bias_loss is not None,
                'num_attributes': len(self.attribute_metadata) if self.attribute_metadata else 0
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"BiasCapability checkpoint saved to {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error saving BiasCapability checkpoint: {e}")
            return False
            
    def load_checkpoint(self, checkpoint_path):
        """Load bias capability state."""
        try:
            import os
            if not os.path.exists(checkpoint_path):
                print(f"BiasCapability checkpoint not found: {checkpoint_path}")
                return False
                
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore configuration
            self.bias_weight = checkpoint_data.get('bias_weight', 1.0)
            
            # Re-initialize bias loss if needed (it should already be initialized)
            if checkpoint_data.get('has_bias_loss', False) and not self.bias_loss:
                self._initialize_bias_loss()
            
            print(f"BiasCapability checkpoint loaded from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading BiasCapability checkpoint: {e}")
            return False 