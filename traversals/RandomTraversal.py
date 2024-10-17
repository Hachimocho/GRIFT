import random
from traversals.Traversal import Traversal

class RandomTraversal(Traversal):
    """
    Traverses the graph using randomly moving pointers.
    """
    tags = ["any"]
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500}
        }
    }

    def __init__(self, graph, num_pointers, num_steps):
        """
        Initialize a RandomTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int): The number of steps to take each pointer. If negative, will move pointers indefinitely.
        """
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.graph = graph
        self.t = 0
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        self.pointers = [{'current_node': self.graph.get_random_node()} for _ in range(self.num_pointers)]
    
    def traverse(self):
        if self.t > self.num_steps:
            raise RuntimeError("Maximum number of steps exceeded.")
        
        self.t += 1
        for i, pointer in enumerate(self.pointers):
            # Get the indices of the adjacent nodes
            adj_nodes = pointer['current_node'].get_adjacent_nodes()

            # If there are no adjacent nodes,
            # move the pointer to a random node
            if not adj_nodes:
                pointer['current_node'] = self.graph.get_random_node()
            else:
                # Randomly select an adjacent node
                pointer['current_node'] = random.choice(adj_nodes)

