# HyperGraph Traversal Refactoring Proposal

## Problem Statement

The current codebase has tight coupling between trainer classes and traversal methods:
- `IValueTrainer` is required for I-value based traversals
- `ExperimentTrainer` is used for simple traversals
- Switching traversal methods mid-training is impossible due to this coupling
- Different trainers have incompatible interfaces and capabilities

## Proposed Solution: Unified Trainer with Capability Components

### Core Architecture Changes

#### 1. Create a Unified `AdaptiveTrainer` Class

```python
class AdaptiveTrainer(Trainer):
    """
    Unified trainer that can adapt to different traversal requirements.
    Uses composition and strategy patterns to support dynamic capability switching.
    """
    
    def __init__(self, graphmanager, models, device, attribute_metadata=None, 
                 loss_fn=None, **kwargs):
        super().__init__(graphmanager, None, models, attribute_metadata=attribute_metadata)
        
        self.device = device
        self.criterion = loss_fn
        self.attribute_metadata = attribute_metadata
        
        # Initialize capability components
        self.capabilities = CapabilityManager(self)
        
        # Training state
        self.current_traversal = None
        self.current_traversal_type = None
        
    def set_traversal(self, traversal_instance, traversal_type):
        """Dynamically set traversal and enable required capabilities."""
        self.current_traversal = traversal_instance
        self.current_traversal_type = traversal_type
        
        # Enable required capabilities based on traversal type
        self.capabilities.configure_for_traversal(traversal_type)
        
        # Set trainer reference in traversal if needed
        if hasattr(traversal_instance, 'trainer'):
            traversal_instance.trainer = self
            
    def switch_traversal(self, new_traversal_type, **traversal_kwargs):
        """Switch to a different traversal method during training."""
        old_type = self.current_traversal_type
        print(f"Switching traversal from {old_type} to {new_traversal_type}")
        
        # Create new traversal instance
        new_traversal = self._create_traversal(new_traversal_type, **traversal_kwargs)
        
        # Transfer state if possible
        if self.current_traversal and hasattr(self.current_traversal, 'get_state'):
            state = self.current_traversal.get_state()
            if hasattr(new_traversal, 'set_state'):
                new_traversal.set_state(state)
        
        # Set new traversal
        self.set_traversal(new_traversal, new_traversal_type)
        
    def get_i_value(self, node, model_idx=0):
        """Get I-value using appropriate capability."""
        return self.capabilities.get_i_value(node, model_idx)
        
    def train(self):
        """Train using current traversal method."""
        if not self.current_traversal:
            raise ValueError("No traversal method set")
            
        return self.capabilities.train_with_traversal(self.current_traversal)
```

#### 2. Create a `CapabilityManager` Component

```python
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
        if traversal_type in ["i-value", "i-value-cluster-hop"]:
            self._enable_dqn_capability()
            self._enable_bias_capability()
        else:
            self._disable_dqn_capability()
            self._disable_bias_capability()
            
    def _enable_dqn_capability(self):
        """Enable DQN functionality."""
        if "dqn" not in self.enabled_capabilities:
            self.dqn_capability = DQNCapability(self.trainer)
            self.enabled_capabilities.add("dqn")
            print("DQN capability enabled")
            
    def _enable_bias_capability(self):
        """Enable bias loss functionality."""
        if "bias" not in self.enabled_capabilities:
            self.bias_capability = BiasCapability(self.trainer)
            self.enabled_capabilities.add("bias")
            print("Bias capability enabled")
            
    def get_i_value(self, node, model_idx=0):
        """Get I-value using appropriate method."""
        if self.dqn_capability:
            return self.dqn_capability.get_i_value(node, model_idx)
        else:
            return random.random()  # Fallback for non-I-value traversals
            
    def train_with_traversal(self, traversal):
        """Execute training with current capabilities."""
        if self.dqn_capability:
            return self.dqn_capability.train_with_dqn(traversal)
        else:
            return self.basic_training_capability.train_basic(traversal)
```

#### 3. Create Capability Components

```python
class DQNCapability:
    """Encapsulates all DQN-related functionality."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        self.dqns = []
        
        # Initialize DQN models if attribute metadata exists
        if trainer.attribute_metadata:
            self._initialize_dqns()
            
    def _initialize_dqns(self):
        """Initialize DQN models based on attribute metadata."""
        # DQN initialization logic from IValueTrainer
        pass
        
    def get_i_value(self, node, model_idx):
        """Calculate I-value using DQN."""
        # I-value calculation logic from IValueTrainer
        pass
        
    def train_with_dqn(self, traversal):
        """Training loop with DQN integration."""
        # DQN training logic from IValueTrainer
        pass

class BiasCapability:
    """Encapsulates bias measurement and correction functionality."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        if trainer.attribute_metadata:
            self._initialize_bias_loss()
            
    def _initialize_bias_loss(self):
        """Initialize bias loss computation."""
        # Bias loss initialization from IValueTrainer
        pass

class BasicTrainingCapability:
    """Basic training functionality for simple traversals."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        
    def train_basic(self, traversal):
        """Basic training loop without DQN or bias correction."""
        # Basic training logic from ExperimentTrainer
        pass
```

#### 4. Refactor Traversal Classes

```python
class BaseTraversal:
    """Base traversal class with optional trainer dependency."""
    
    def __init__(self, graph, num_pointers, num_steps, trainer=None):
        self.graph = graph
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.trainer = trainer  # Optional trainer reference
        
    def set_trainer(self, trainer):
        """Set trainer reference after initialization."""
        self.trainer = trainer
        
    def get_state(self):
        """Get current traversal state for transfer."""
        return {
            'pointers': getattr(self, 'pointers', []),
            'step_count': getattr(self, 't', 0)
        }
        
    def set_state(self, state):
        """Set traversal state from another traversal."""
        if 'pointers' in state:
            self.pointers = state['pointers']
        if 'step_count' in state:
            self.t = state['step_count']

class IValueTraversal(BaseTraversal):
    """I-value traversal that works with any trainer having I-value capability."""
    
    def get_i_value(self, node, model_idx=0):
        """Get I-value from trainer if available."""
        if self.trainer and hasattr(self.trainer, 'get_i_value'):
            return self.trainer.get_i_value(node, model_idx)
        else:
            return random.random()  # Fallback
```

### Implementation Plan

#### Phase 1: Create Core Infrastructure
1. Implement `AdaptiveTrainer` class
2. Implement `CapabilityManager` class
3. Create capability component classes

#### Phase 2: Refactor Existing Components
1. Extract common functionality from `IValueTrainer` and `ExperimentTrainer`
2. Move DQN logic to `DQNCapability`
3. Move bias logic to `BiasCapability`
4. Update traversal classes to use optional trainer reference

#### Phase 3: Update Main Script
1. Replace trainer selection logic with `AdaptiveTrainer`
2. Add traversal switching functionality
3. Update training loop to support dynamic switching

#### Phase 4: Add Dynamic Switching Features
1. Implement traversal switching based on performance metrics
2. Add configuration for switching schedules
3. Add state transfer between traversal methods

### Benefits

1. **Dynamic Switching**: Can change traversal methods during training
2. **Reduced Coupling**: Traversals no longer tightly coupled to specific trainers
3. **Code Reuse**: Common functionality shared across all traversals
4. **Extensibility**: Easy to add new traversal methods or capabilities
5. **Maintainability**: Clear separation of concerns

### Migration Strategy

1. **Backward Compatibility**: Keep existing trainer classes during transition
2. **Gradual Migration**: Migrate one traversal type at a time
3. **Testing**: Comprehensive testing to ensure equivalent functionality
4. **Documentation**: Update documentation for new architecture

### Usage Example

```python
# Create unified trainer
trainer = AdaptiveTrainer(
    graphmanager=train_manager,
    models=[model],
    device=device,
    attribute_metadata=attribute_metadata,
    loss_fn=criterion
)

# Start with comprehensive traversal
comprehensive_traversal = ComprehensiveTraversal(
    graph=train_manager.graph,
    num_pointers=1,
    num_steps=1000
)
trainer.set_traversal(comprehensive_traversal, "comprehensive")

# Train for some epochs
for epoch in range(5):
    trainer.train()

# Switch to I-value traversal mid-training
trainer.switch_traversal("i-value-cluster-hop", 
                        bias_hop_period=2,
                        num_pointers=1,
                        num_steps=1000)

# Continue training with new traversal
for epoch in range(5, 10):
    trainer.train()
```

This architecture provides the flexibility to switch traversal methods during training while maintaining clean separation of concerns and code reusability. 