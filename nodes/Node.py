class Node():
    """
    Each node class must have a set of tags which matches what data types and/or datasets it can be used with.
    Invalid tags might cause bad things, so don't do that.
    """ 
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, node_id, split, data, edges, label):
        self.node_id = node_id
        self.split = split
        self.data = data
        self.edges = edges
        self.label = label
        
    def match(self, other):
        if isinstance(other, Node):
            return True
        else:
            return False
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        return self.data
    
    def set_data(self, data):
        self.data = data
        
    def get_adjacent_nodes(self):
        adjacent_nodes = []
        for edge in self.edges:
            for node in edge.get_nodes():
                if node != self:
                    adjacent_nodes.append(node)
        return adjacent_nodes
        
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        else:
            return False
        
    def __hash__(self):
        return hash(self.node_id)
    
    def get_split(self):
        return self.split
    
    def set_split(self, split):
        self.split = split
        
    def get_label(self):
        return self.label

    def add_edge(self, edge):
        self.edges.append(edge)