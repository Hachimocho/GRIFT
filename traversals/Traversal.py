class Traversal():
    """
    Abstract class, defines a method through which pointers located on nodes can move to other nodes in the graph.
    """
    tags = ["none"]
    # No hyperparameters, since this class should never be used without subclassing.
    
    def traverse(self, graph):
        raise NotImplementedError("Subclass must implement traverse()")
    
    def get_pointers(self):
        raise NotImplementedError("Subclass must implement get_pointers()")
    
    def reset_pointers(self):
        raise NotImplementedError("Subclass must implement reset_pointers()")