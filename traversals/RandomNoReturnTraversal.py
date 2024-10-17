import random
from traversals.Traversal import Traversal

class RandomNoReturnTraversal(Traversal):
    """
    Traverses the graph using randomly moving pointers, without returning to recently visited nodes.
    """
    tags = ["any"]
    
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "return_delay": {"distribution": "int_uniform", "min": 10, "max": 100}
        }
    }

    def __init__(self, graph, num_pointers, num_steps, return_delay):
        """
        Initialize a RandomTraversal object.

        Args:
            graph (HyperGraph): The graph to traverse.
            num_pointers (int): The number of pointers to move around the graph.
            num_steps (int): The number of steps to take each pointer. If negative, will move pointers indefinitely.
        """
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.return_delay = return_delay
        self.graph = graph
        self.t = 0
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        self.pointers = [{'current_node': self.graph.get_random_node(), 'last_visited': {}} for _ in range(self.num_pointers)]
    
    def traverse(self):
        if self.t > self.num_steps:
            raise RuntimeError("Maximum number of steps exceeded.")
        self.t += 1
        
        for i, pointer in enumerate(self.pointers):
            current_node = pointer['current_node']
            # Get the adjacent nodes
            adj_nodes = pointer['current_node'].get_adjacent_nodes()

            # Filter out the nodes that were visited in the last X timesteps
            adj_nodes = [node for node in adj_nodes if node in self.pointers[i]['last_visited'].keys() and self.t - self.pointers[i]['last_visited'][node] > self.X]

            if adj_nodes:
                # Randomly select an adjacent node
                pointer['current_node'] = random.choice(adj_nodes)
            else:
                # If there are no adjacent nodes,
                # move the pointer to a random node (can visit recently visited nodes this way, prevents hardlocks on small graphs)
                pointer['current_node'] = self.graph.get_random_node()

            for key in pointer['last_visited'].keys():
                pointer['last_visited'][key] -= 1
                if pointer['last_visited'][key] <= 0:
                    pointer['last_visited'].pop(key)
            pointer['last_visited'][current_node] = self.return_delay

