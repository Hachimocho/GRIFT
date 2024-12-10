class Traversal:
    """
    Abstract class, defines a method through which pointers located on nodes can move to other nodes in the graph.
    Also implements the iterator protocol to make traversals directly iterable.
    """
    tags = ["none"]
    
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
    
    def traverse(self):
        """Move pointers to next nodes."""
        raise NotImplementedError("Subclass must implement traverse()")
    
    def get_pointers(self):
        """Get current pointer states."""
        raise NotImplementedError("Subclass must implement get_pointers()")
    
    def reset_pointers(self):
        """Reset pointers to initial state."""
        raise NotImplementedError("Subclass must implement reset_pointers()")