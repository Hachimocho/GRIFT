from managers.GraphManager import GraphManager

class NoGraphManager(GraphManager):
    """
    A GraphManager that does nothing. Use for any static environment.
    """
    tags = ["any"]
    hyperparameters = None
    
    def update_graph(self):
        """
        Dummy update function.
        """
        pass
    