class Edge():
    """
    Base edge class for connecting nodes.
    All edges must have a set of tags to denote what data types/sets they can be used with.
    Bad tags could break things, so please don't do that.
    """
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, node1, node2, x, traversal_weight=1):
        self.node1 = node1
        self.node2 = node2
        self.x = x
        self.traversal_weight = traversal_weight
        
    def _invalidate_endpoints(self, *nodes):
        """Drop cached neighbour lists on every node this edge's change can affect.

        Re-pointing an edge changes what its *other* endpoint sees as a neighbour without
        touching either node's `edges` list, so the identity-and-length validation in
        `Node.get_adjacent_nodes` cannot notice it. This is the one path that has to say
        so explicitly.
        """
        for node in nodes:
            invalidate = getattr(node, 'invalidate_adjacency_cache', None)
            if invalidate is not None:
                invalidate()

    def set_node1(self, node):
        self._invalidate_endpoints(self.node1, self.node2, node)
        self.node1 = node

    def set_node2(self, node):
        self._invalidate_endpoints(self.node1, self.node2, node)
        self.node2 = node

    def set_nodes(self, node1, node2):
        self._invalidate_endpoints(self.node1, self.node2, node1, node2)
        self.node1 = node1
        self.node2 = node2
        
    def get_node1(self):
        return self.node1
    
    def get_node2(self):
        return self.node2
    
    def get_nodes(self):
        return self.node1, self.node2
    
    def set_data(self, x):
        self.x = x
        
    def get_data(self):
        return self.x
    
    def set_traversal_weight(self, w):
        self.traversal_weight = w
        
    def get_traversal_weight(self):
        return self.traversal_weight
    
    
        