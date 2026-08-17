import random

from test_helpers.args_utils import IVALUE_TRAVERSAL_ALIASES
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
        
        # NEW: Sequence awareness for DQN warm-up
        self.traversal_sequence = None
        self.requires_dqn_warmup = False
        
    def set_traversal_sequence(self, sequence):
        """Set the full traversal sequence to enable DQN warm-up if needed."""
        self.traversal_sequence = sequence
        self.requires_dqn_warmup = any(t in IVALUE_TRAVERSAL_ALIASES for t in sequence)
        
        if self.requires_dqn_warmup:
            print(f"CapabilityManager: I-value traversal detected in sequence {sequence}")
            print(f"CapabilityManager: DQN will be trained during ALL traversals for warm-up")
            # Enable DQN immediately if I-value traversal is used anywhere
            self._enable_dqn_capability()
            self._enable_bias_capability()
        
    def configure_for_traversal(self, traversal_type):
        """Enable capabilities needed for specific traversal type."""
        print(f"CapabilityManager: Configuring for traversal type '{traversal_type}'")
        
        # Membership test against the alias set rather than a hand-kept list. The list
        # named only two of the four I-value traversals that existed, so the two
        # `*-subcluster` variants silently got basic capabilities: no DQN, and
        # `get_i_value` falling through to a random draw. They were named "i-value" and
        # ran on random numbers.
        if traversal_type in IVALUE_TRAVERSAL_ALIASES:
            self._enable_dqn_capability()
            self._enable_bias_capability()
        else:
            # For basic traversals, check if DQN warm-up is needed
            if self.requires_dqn_warmup:
                print(f"CapabilityManager: Using DQN-enabled capabilities for '{traversal_type}' (warm-up mode)")
                # Keep DQN enabled for warm-up
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

        # Fallback for non-I-value traversals. This draws from a dedicated stream
        # rather than the global `random` module, which matters more here than
        # anywhere else: IValueTraversal.reset_pointers calls this once per node per
        # pointer per epoch, so the number of global draws consumed scaled with
        # *graph size* -- meaning changing the node count shifted every subsequent
        # random decision in the run.
        return self._ivalue_fallback_rng().random()

    _IVALUE_FALLBACK_SEED = 1013904223

    def _ivalue_fallback_rng(self):
        """Lazily bind this manager's private I-value RNG."""
        if getattr(self, '_ivalue_rng', None) is not None:
            return self._ivalue_rng

        seeded = None
        try:
            from test_helpers.determinism import is_configured, rng_for
            if is_configured():
                seeded = rng_for("ivalue.fallback")
        except ImportError:
            pass

        self._ivalue_rng = seeded or random.Random(self._IVALUE_FALLBACK_SEED)
        return self._ivalue_rng
            
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