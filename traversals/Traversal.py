class Traversal:
    """
    Abstract class, defines a method through which pointers located on nodes can move to other nodes in the graph.
    Also implements the iterator protocol to make traversals directly iterable.
    Enhanced to support state transfer and optional trainer dependencies.
    """
    tags = ["none"]
    
    def __init__(self):
        """Initialize base traversal."""
        self.trainer = None  # Optional trainer reference

    @property
    def rng(self):
        """This traversal's private RNG stream, derived from its class name.

        Every traversal previously drew from the process-global ``random`` module,
        which made their decisions depend on how much randomness anything else
        happened to consume first -- so adding a log line that called
        ``random.random()``, or changing the graph size (which changes how many
        I-value fallback draws occur), silently changed which nodes were visited.

        A lazy property rather than a constructor assignment, because the concrete
        traversals do not consistently call ``super().__init__()``.
        """
        stream = getattr(self, '_rng', None)
        if stream is None:
            from test_helpers.determinism import component_rng
            stream = component_rng(f"traversal.{type(self).__name__}")
            self._rng = stream
        return stream

    def __iter__(self):
        """Make traversal iterable."""
        return self
    
    def __next__(self):
        """Get next batch of nodes from traversal."""
        try:
            self.traverse()
            return [pointer['current_node'] for pointer in self.get_pointers()]
        except RuntimeError:
            raise StopIteration
    
    def __len__(self):
        """Return the number of steps in the traversal."""
        raise NotImplementedError("Subclass must implement __len__()")
    
    def traverse(self, batch_size=32):
        """Move pointers to next nodes."""
        raise NotImplementedError("Subclass must implement traverse()")
    
    def get_pointers(self):
        """Get current pointer states."""
        raise NotImplementedError("Subclass must implement get_pointers()")
    
    def reset_pointers(self):
        """Reset pointers to initial state."""
        raise NotImplementedError("Subclass must implement reset_pointers()")
    
    def set_trainer(self, trainer):
        """Set trainer reference after initialization."""
        self.trainer = trainer
        
    def get_state(self):
        """Get current traversal state for transfer to another traversal."""
        return {
            'pointers': getattr(self, 'pointers', []),
            'step_count': getattr(self, 't', 0),
            'steps_taken': getattr(self, 'steps_taken', 0),
            'visited_nodes': getattr(self, 'visited_nodes', set()),
            'traversal_type': self.__class__.__name__
        }
        
    def set_state(self, state):
        """Set traversal state from another traversal for seamless switching."""
        try:
            if 'pointers' in state and state['pointers']:
                # Only set pointers if they exist and are compatible
                self.pointers = state['pointers']
                print(f"Transferred {len(self.pointers)} pointers")
                
            if 'step_count' in state:
                self.t = state['step_count']
                print(f"Transferred step count: {self.t}")
                
            if 'steps_taken' in state:
                self.steps_taken = state['steps_taken']
                print(f"Transferred steps taken: {self.steps_taken}")
                
            if 'visited_nodes' in state and hasattr(self, 'visited_nodes'):
                self.visited_nodes = state['visited_nodes']
                print(f"Transferred {len(self.visited_nodes)} visited nodes")
                
            print(f"Successfully transferred state from {state.get('traversal_type', 'unknown')} to {self.__class__.__name__}")
            
        except Exception as e:
            print(f"Warning: Error setting traversal state: {e}")
            # Continue with default initialization if state transfer fails