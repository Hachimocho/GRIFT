import random
from traversals.Traversal import Traversal

class RandomTraversal(Traversal):
    """
    Traverses the graph randomly using multiple pointers.
    """
    tags = ["any"]
    hyperparameters = None

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
        self.visited_nodes = set()  # Track visited nodes across traversals
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        """Reset pointers to random nodes and clear visited nodes."""
        self.pointers = []
        self.visited_nodes.clear()  # Clear visited nodes on reset
        for _ in range(self.num_pointers):
            self.pointers.append({'current_node': self.graph.get_random_node()})
        
    def traverse(self, batch_size=32):
        """
        Move pointers randomly around the graph, accumulating nodes over multiple steps.
        
        Args:
            batch_size (int): Number of nodes to return per batch
            
        Returns:
            list: List of nodes visited by the pointers
        """
        if self.t >= self.num_steps:
            return []
        
        all_nodes = []
        steps_without_new = 0
        max_steps_without_new = 100  # Prevent infinite loops
        
        # Keep collecting nodes until we reach num_steps or can't find new nodes
        while self.t < self.num_steps and steps_without_new < max_steps_without_new:
            self.t += 1
            found_new = False
            
            # Move each pointer and collect nodes
            for pointer in self.pointers:
                # Get adjacent nodes
                adj_nodes = pointer['current_node'].get_adjacent_nodes()
                
                # If there are no adjacent nodes or with small probability,
                # move the pointer to a random node
                if not adj_nodes or random.random() < 0.001:  # .1% chance to jump randomly
                    pointer['current_node'] = self.graph.get_random_node()
                else:
                    # Randomly select an adjacent node
                    pointer['current_node'] = random.choice(adj_nodes)
                
                # Add current node if not already visited
                if pointer['current_node'] not in self.visited_nodes:
                    all_nodes.append(pointer['current_node'])
                    self.visited_nodes.add(pointer['current_node'])
                    found_new = True
            
            # Update counter for steps without finding new nodes
            if found_new:
                steps_without_new = 0
            else:
                steps_without_new += 1
        
        print(f"Traversal collected {len(all_nodes)} nodes in {self.t} steps")
        return all_nodes
#aaaarrrgghhh