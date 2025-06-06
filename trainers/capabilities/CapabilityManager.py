import random
from trainers.capabilities.DQNCapability import DQNCapability
from trainers.capabilities.BiasCapability import BiasCapability
from trainers.capabilities.BasicTrainingCapability import BasicTrainingCapability


class CapabilityManager:
    """
    Manages different capabilities (DQN, bias loss, etc.) needed by different traversals.
    Uses composition to enable/disable features as needed.
    """
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        
        # Capability components
        self.dqn_capability = None
        self.bias_capability = None
        self.basic_training_capability = BasicTrainingCapability(trainer)
        
        # Current configuration
        self.enabled_capabilities = set()
        
    def configure_for_traversal(self, traversal_type):
        """Enable capabilities needed for specific traversal type."""
        print(f"CapabilityManager: Configuring for traversal type '{traversal_type}'")
        
        if traversal_type in ["i-value", "i-value-cluster-hop"]:
            self._enable_dqn_capability()
            self._enable_bias_capability()
        else:
            # For basic traversals, we don't disable existing capabilities
            # This allows for seamless switching between traversal types
            print(f"CapabilityManager: Using basic capabilities for '{traversal_type}'")
            
    def _enable_dqn_capability(self):
        """Enable DQN functionality."""
        if "dqn" not in self.enabled_capabilities:
            try:
                self.dqn_capability = DQNCapability(self.trainer)
                self.enabled_capabilities.add("dqn")
                print("CapabilityManager: DQN capability enabled")
            except Exception as e:
                print(f"CapabilityManager: Failed to enable DQN capability: {e}")
                self.dqn_capability = None
                
    def _enable_bias_capability(self):
        """Enable bias loss functionality."""
        if "bias" not in self.enabled_capabilities:
            try:
                self.bias_capability = BiasCapability(self.trainer)
                self.enabled_capabilities.add("bias")
                print("CapabilityManager: Bias capability enabled")
            except Exception as e:
                print(f"CapabilityManager: Failed to enable bias capability: {e}")
                self.bias_capability = None
                
    def _disable_dqn_capability(self):
        """Disable DQN functionality (optional, for memory management)."""
        if "dqn" in self.enabled_capabilities:
            self.dqn_capability = None
            self.enabled_capabilities.discard("dqn")
            print("CapabilityManager: DQN capability disabled")
            
    def _disable_bias_capability(self):
        """Disable bias loss functionality (optional, for memory management)."""
        if "bias" in self.enabled_capabilities:
            self.bias_capability = None
            self.enabled_capabilities.discard("bias")
            print("CapabilityManager: Bias capability disabled")
            
    def get_i_value(self, node, model_idx=0):
        """Get I-value using appropriate method."""
        if self.dqn_capability:
            return self.dqn_capability.get_i_value(node, model_idx)
        else:
            # Fallback for non-I-value traversals
            return random.random()
            
    def train_with_traversal(self, traversal, epoch=None):
        """Execute training with current capabilities."""
        if self.dqn_capability and hasattr(self.dqn_capability, 'train_with_dqn'):
            return self.dqn_capability.train_with_dqn(traversal, epoch)
        else:
            return self.basic_training_capability.train_basic(traversal, epoch)
            
    def get_bias_loss(self):
        """Get bias loss function if available."""
        if self.bias_capability:
            return self.bias_capability.get_bias_loss()
        return None
    
    def save_checkpoints(self, base_path):
        """Save checkpoints for all enabled capabilities."""
        saved_capabilities = []
        
        # Save DQN checkpoint if enabled
        if self.dqn_capability and "dqn" in self.enabled_capabilities:
            try:
                dqn_path = base_path.replace('.pth', '_dqn.pth').replace('.pt', '_dqn.pt')
                if self.dqn_capability.save_checkpoint(dqn_path):
                    saved_capabilities.append("dqn")
            except Exception as e:
                print(f"Warning: Could not save DQN checkpoint: {e}")
        
        # Save bias capability state if needed (usually just configuration)
        if self.bias_capability and "bias" in self.enabled_capabilities:
            try:
                bias_path = base_path.replace('.pth', '_bias.pth').replace('.pt', '_bias.pt')
                if self.bias_capability.save_checkpoint(bias_path):
                    saved_capabilities.append("bias")
            except Exception as e:
                print(f"Warning: Could not save bias checkpoint: {e}")
        
        if saved_capabilities:
            print(f"Saved capability checkpoints: {saved_capabilities}")
        else:
            print("No capability checkpoints to save")
    
    def load_checkpoints(self, base_path):
        """Load checkpoints for all enabled capabilities."""
        loaded_capabilities = []
        
        # Load DQN checkpoint if enabled
        if self.dqn_capability and "dqn" in self.enabled_capabilities:
            try:
                dqn_path = base_path.replace('.pth', '_dqn.pth').replace('.pt', '_dqn.pt')
                if self.dqn_capability.load_checkpoint(dqn_path):
                    loaded_capabilities.append("dqn")
            except Exception as e:
                print(f"Warning: Could not load DQN checkpoint: {e}")
        
        # Load bias capability state if needed
        if self.bias_capability and "bias" in self.enabled_capabilities:
            try:
                bias_path = base_path.replace('.pth', '_bias.pth').replace('.pt', '_bias.pt')
                if self.bias_capability.load_checkpoint(bias_path):
                    loaded_capabilities.append("bias")
            except Exception as e:
                print(f"Warning: Could not load bias checkpoint: {e}")
        
        if loaded_capabilities:
            print(f"Loaded capability checkpoints: {loaded_capabilities}")
        else:
            print("No capability checkpoints to load") 