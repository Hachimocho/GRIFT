import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
import json
import random
from collections import defaultdict
from torch.cuda.amp import GradScaler

from trainers.Trainer import Trainer
from trainers.capabilities.CapabilityManager import CapabilityManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.RandomTraversal import RandomTraversal
from test_helpers.args_utils import canonical_traversal_type
from traversals.IValueTraversal import IValueTraversal


class AdaptiveTrainer(Trainer):
    """
    Unified trainer that can adapt to different traversal requirements.
    Uses composition and strategy patterns to support dynamic capability switching.
    """
    tags = ["adaptive"]
    
    def __init__(self, graphmanager, models, device, attribute_metadata=None, 
                 loss_fn=None, attribute_weights=None, bias_group_weights=None, **kwargs):
        """Initialize the adaptive trainer with capability management."""
        super().__init__(graphmanager, None, models, attribute_metadata=attribute_metadata)
        
        self.device = device
        if loss_fn is None:
            raise ValueError("loss_fn must be provided to AdaptiveTrainer")
        self.criterion = loss_fn
        self.attribute_metadata = attribute_metadata
        # Selected DQN model type for I-value prediction (e.g., 'basic', 'residual', ...)
        self.dqn_model_type = kwargs.get('dqn_model_type', 'basic')
        print(f"AdaptiveTrainer: DQN model type set to '{self.dqn_model_type}'")
        
        # Initialize capability components
        self.capabilities = CapabilityManager(self)
        
        # Training state
        self.current_traversal = None
        self.current_traversal_type = None
        
        # Training settings
        self.batch_size = 32
        self.max_nodes_per_epoch = 10000
        self.scaler = GradScaler()
        
        # Setup logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"adaptive_trainer_{timestamp}.json"
        self.metrics_history = []
        
        # Extract categorical attributes for tracking if metadata is provided
        self.categorical_attrs_for_tracking = []
        if self.attribute_metadata:
            self.categorical_attrs_for_tracking = [
                attr['name'] for attr in self.attribute_metadata if attr.get('type') == 'categorical'
            ]
            if not self.categorical_attrs_for_tracking:
                print("AdaptiveTrainer: No categorical attributes found in metadata for tracking.")
            else:
                print(f"AdaptiveTrainer: Will track distribution for attributes: {self.categorical_attrs_for_tracking}")
        
        # Set bias loss weight on BiasCapability if provided (store for later if not available yet)
        self.bias_loss_weight = kwargs.get('bias_loss_weight', None)
        self.attribute_weights = attribute_weights
        self.bias_group_weights = bias_group_weights
        # Map group weights to per-attribute weights if group weights are provided
        if self.bias_group_weights is not None and self.attribute_metadata is not None:
            group_map = {
                'Gender': ['Ground Truth Gender'],
                'Race': ['Ground Truth Race'],
                'Age': ['Ground Truth Age'],
                'Emotion': [
                    'emotion_angry', 'emotion_disgust', 'emotion_fear', 'emotion_happy',
                    'emotion_sad', 'emotion_surprise', 'emotion_neutral'
                ],
                'Quality': ['blur', 'brightness', 'contrast', 'compression'],
                'FaceEmbedding': ['face_embedding']
            }
            attr_weights = {}
            for group, attrs in group_map.items():
                weight = self.bias_group_weights.get(group, 1.0)
                for attr in attrs:
                    attr_weights[attr] = weight
            # For any attribute not in a group, default to 1.0
            for attr in self.attribute_metadata:
                name = attr['name'] if isinstance(attr, dict) else attr.name
                if name not in attr_weights:
                    attr_weights[name] = 1.0
            self.attribute_weights = attr_weights
        if hasattr(self.capabilities, 'bias_capability') and self.capabilities.bias_capability is not None:
            if self.bias_loss_weight is not None:
                self.capabilities.bias_capability.bias_weight = self.bias_loss_weight
            if self.attribute_weights is not None:
                self.capabilities.bias_capability.attribute_weights = self.attribute_weights
                self.capabilities.bias_capability._initialize_bias_loss()
            print(f"[DEBUG] Set bias loss weight to {self.bias_loss_weight} and attribute weights to {self.attribute_weights} on BiasCapability.")
        elif self.bias_loss_weight is not None or self.attribute_weights is not None:
            print(f"[DEBUG] BiasCapability not initialized yet; will set bias_loss_weight={self.bias_loss_weight} and attribute_weights={self.attribute_weights} when available.")
            
    def set_traversal(self, traversal_instance, traversal_type):
        """Dynamically set traversal and enable required capabilities."""
        self.current_traversal = traversal_instance
        self.current_traversal_type = traversal_type
        
        # Enable required capabilities based on traversal type
        self.capabilities.configure_for_traversal(traversal_type)

        # Ensure bias loss weight and attribute weights are set if capability is now available
        if hasattr(self.capabilities, 'bias_capability') and self.capabilities.bias_capability is not None:
            if self.bias_loss_weight is not None:
                self.capabilities.bias_capability.bias_weight = self.bias_loss_weight
            if self.attribute_weights is not None:
                self.capabilities.bias_capability.attribute_weights = self.attribute_weights
                self.capabilities.bias_capability._initialize_bias_loss()
            print(f"[DEBUG] Set bias loss weight to {self.bias_loss_weight} and attribute weights to {self.attribute_weights} on BiasCapability (post-initialization).")
        
        # Set trainer reference in traversal if needed
        if hasattr(traversal_instance, 'trainer'):
            traversal_instance.trainer = self
        elif hasattr(traversal_instance, 'set_trainer'):
            traversal_instance.set_trainer(self)
            
        print(f"AdaptiveTrainer: Set traversal to {traversal_type} ({type(traversal_instance).__name__})")
        
    def set_traversal_sequence(self, sequence):
        """Set the full traversal sequence for DQN warm-up planning."""
        self.capabilities.set_traversal_sequence(sequence)
        print(f"AdaptiveTrainer: Set traversal sequence: {sequence}")
            
    def switch_traversal(self, new_traversal_type, **traversal_kwargs):
        """Switch to a different traversal method during training."""
        old_type = self.current_traversal_type
        print(f"Switching traversal from {old_type} to {new_traversal_type}")
        
        # Create new traversal instance
        new_traversal = self._create_traversal(new_traversal_type, **traversal_kwargs)
        
        # Transfer state if possible
        # `is not None`, not truthiness: a truthiness test calls __len__, which the
        # Random* traversals do not implement (Traversal.__len__ raises), so every
        # run using one died here. For the traversals that *do* implement it, a
        # zero-length traversal would have been silently treated as absent.
        if self.current_traversal is not None and hasattr(self.current_traversal, 'get_state'):
            try:
                state = self.current_traversal.get_state()
                if hasattr(new_traversal, 'set_state'):
                    new_traversal.set_state(state)
                    print(f"Successfully transferred state from {old_type} to {new_traversal_type}")
            except Exception as e:
                print(f"Warning: Could not transfer state from {old_type} to {new_traversal_type}: {e}")
        
        # Set new traversal
        self.set_traversal(new_traversal, new_traversal_type)
        
    def _create_traversal(self, traversal_type, **kwargs):
        """Factory for traversal instances.

        The I-value traversal is one class now, and it picks its own walk from the graph --
        this method used to fan `"i-value"` out across four subclasses on `graph_type` and a
        `subclusters` probe, duplicating a decision the traversal is better placed to make.
        The two subcluster subclasses are gone entirely.
        """
        graph = kwargs.get('graph', self.graphmanager.get_graph())
        num_pointers = kwargs.get('num_pointers', 1)
        num_steps = kwargs.get('num_steps', 1000)

        traversal_type = canonical_traversal_type(traversal_type, quiet=True)
        if traversal_type == "i-value":
            return IValueTraversal(
                graph, num_pointers, num_steps, trainer=self,
                bias_hop_period=kwargs.get('bias_hop_period', 2),
                # An explicit --graph-type from the caller wins over the graph's own
                # attribute, which a cached graph shell may not carry.
                cluster_hop=(
                    str(kwargs['graph_type']).startswith('clustered')
                    if kwargs.get('graph_type') else None
                ),
            )
        elif traversal_type == "comprehensive":
            return ComprehensiveTraversal(graph, num_pointers, num_steps)
        elif traversal_type == "random":
            return RandomTraversal(graph, num_pointers, num_steps)
        else:
            raise ValueError(f"Unknown traversal type: {traversal_type}")

    def get_i_value(self, node, model_idx=0):
        """Get I-value using appropriate capability, recording it for the graph manager.

        This is the single funnel every predicted I-value passes through -- the traversals,
        the reduction manager, and the visualizers all call it -- so it is the one place a
        graph updater can observe the values training already computes without paying for a
        second DQN forward pass per node.

        That matters because the alternative was a separate sampling pass, which at a
        million nodes is a million forward passes *per update*. `PerformanceGraphManager`
        documented this as its input, and until this hook existed nothing called
        `track_performance` at all: it logged "0 node(s) measured", never reached the
        minimum for a quantile, and pruned nothing -- so three sweep cells came back with
        byte-identical record tables.

        A manager with no `track_performance` (the default `NoGraphManager`) is untouched.
        """
        value = self.capabilities.get_i_value(node, model_idx)
        tracker = getattr(self.graphmanager, 'track_performance', None)
        if tracker is not None:
            try:
                tracker(node, value)
            except Exception as error:
                # Bookkeeping must never break training.
                print(f"Warning: could not record I-value for "
                      f"{getattr(node, 'node_id', '?')}: {error}")
        return value
        
    def train(self, epoch=None):
        """Train using current traversal method."""
        # See _switch_traversal: truthiness on a traversal invokes __len__.
        if self.current_traversal is None:
            raise ValueError("No traversal method set")
            
        return self.capabilities.train_with_traversal(self.current_traversal, epoch)
    
    def log_metrics(self, metrics):
        """Log training metrics to file."""
        metrics_dict = {}
        for key, value in metrics.items():
            # Convert tensors to float/int
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics_dict[key] = value
        
        # Add timestamp and traversal type
        metrics_dict['timestamp'] = datetime.now().isoformat()
        metrics_dict['traversal_type'] = self.current_traversal_type
        self.metrics_history.append(metrics_dict)
        
        # Write to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Also print to console
        print(f"Metrics: {metrics_dict}")
    
    def save_capability_checkpoints(self, base_path):
        """Save checkpoints for all enabled capabilities."""
        try:
            self.capabilities.save_checkpoints(base_path)
            print(f"Capability checkpoints saved with base path: {base_path}")
        except Exception as e:
            print(f"Warning: Could not save capability checkpoints: {e}")
    
    def load_capability_checkpoints(self, base_path):
        """Load checkpoints for all enabled capabilities."""
        try:
            self.capabilities.load_checkpoints(base_path)
            print(f"Capability checkpoints loaded from base path: {base_path}")
        except Exception as e:
            print(f"Warning: Could not load capability checkpoints: {e}")
    
    def get_current_traversal_info(self):
        """Get information about the current traversal configuration."""
        if self.current_traversal is None:
            return "No traversal set"
        
        info = {
            'type': self.current_traversal_type,
            'class': type(self.current_traversal).__name__,
            'enabled_capabilities': list(self.capabilities.enabled_capabilities)
        }
        
        # Add traversal-specific info
        if hasattr(self.current_traversal, 'num_pointers'):
            info['num_pointers'] = self.current_traversal.num_pointers
        if hasattr(self.current_traversal, 'num_steps'):
            info['num_steps'] = self.current_traversal.num_steps
        if hasattr(self.current_traversal, 'bias_hop_period'):
            info['bias_hop_period'] = self.current_traversal.bias_hop_period
            
        return info 