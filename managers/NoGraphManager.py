from managers.GraphManager import GraphManager

class NoGraphManager(GraphManager):
    """
    A GraphManager that does nothing. Use for any static environment.
    """
    tags = ["any"]
    hyperparameters = None
    
    def update_graph(self, steps_taken=1):
        """
        Dummy update function.

        Accepts ``steps_taken`` so callers can advance any manager uniformly without
        branching on its type; a static graph has nothing to advance.
        """
        return None
    