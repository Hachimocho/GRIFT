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
        
    def set_node1(self, node):
        self.node1 = node
        
    def set_node2(self, node):
        self.node2 = node
        
    def set_nodes(self, node1, node2):
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
    
    
        