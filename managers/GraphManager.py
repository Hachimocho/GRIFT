from graphs.HyperGraph import HyperGraph

class GraphManager(HyperGraph):
    """
    Defines a HyperGraph which changes over time. May use traversals, model performance, or just time to adjust the graph.
    Most commonly used to represent a specific environment or to perform automatic data augmentation.
    Abstract class, overwrite for actual usage.
    """
    tags = ["none"]
    # No hyperparameters, since this class should never be used without subclassing.
    hyperparameters: dict | None = None
    def __init__(self, graph):
        """
        Initialize a GraphManager object.

        Args:
            graph (HyperGraph): The graph to manage.
        """
        self.graph = graph

    def set_graph(self, graph):
        """
        Set the graph managed by the GraphManager object.

        Args:
            graph (HyperGraph): The graph to manage.
        """
        self.graph = graph

    def get_graph(self):
        """
        Get the graph managed by the GraphManager object.

        Returns:
            HyperGraph: The graph managed by the GraphManager object.
        """
        return self.graph
    
    def update_graph(self):
        """
        Update the graph managed by the GraphManager object.
        """
        raise NotImplementedError("Subclass must implement update_graph()")
    