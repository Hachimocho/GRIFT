import random
from traversals.Traversal import Traversal

class RandomNoReturnWarpTraversal(Traversal):
    """
    Traverses the graph using randomly moving pointers, without returning to recently visited nodes.
    """
    tags = ["any"]
    
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "return_delay": {"distribution": "int_uniform", "min": 10, "max": 100},
            "warp_chance": {"distribution": "uniform", "min": 0.0, "max": 0.999}
        }
    }

    def __init__(self, graph, num_pointers, num_steps, return_delay, warp_chance):
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
        self.warp_chance = warp_chance
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
            # Get the adjacent nodes
            adj_nodes = pointer['current_node'].get_adjacent_nodes()

            # Filter out the nodes that were visited in the last X timesteps
            adj_nodes = [node for node in adj_nodes if t - last_visited[node] > self.X]

            if adj_nodes and (random.random() > self.warp_chance):
                # Randomly select an adjacent node
                pointer['current_node'] = random.choice(adj_nodes)
            else:
                # If there are no adjacent nodes or the warp triggers,
                # move the pointer to a random node (can visit recently visited nodes this way, prevents hardlocks on small graphs)
                pointer['current_node'] = self.graph.get_random_node()

            for key in pointer['last_visited'].keys():
                pointer['last_visited'][key] -= 1
                if pointer['last_visited'][key] <= 0:
                    pointer['last_visited'].pop(key)
            pointer['last_visited'][current_node] = self.return_delay
            
            
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
          
            
    # Startup pointers
            num_attributes = len(data.x[0][1])
            attribute_set = [[key_attribute_index, key_attribute_value] for key_attribute_index, key_attribute_value in zip([i for i in self.key_attributes for _ in range(2)], [[0, 2][j % 2] for j in range(len(self.key_attributes) * 2)])]
            pointers = [{'key_attributes': attribute_set[i], 'current_node': random.choice([num for num in range(num_nodes) if int(data.x[num][1][attribute_set[i][0]]) != int(attribute_set[i][1])]), 'last_visited': [-self.X for _ in range(num_nodes)], 'ever_visited': [0 for _ in range(num_nodes)]} for i in range(self.num_pointers)]
            # Move the pointers and process the node data
            while(t < num_steps if num_steps > 0 else True):
                t += 1
                for i, pointer in enumerate(pointers):
                    current_node = pointer['current_node']
                    last_visited = pointer['last_visited']

                    # Get the indices of the adjacent nodes
                    adj_nodes = data.edge_index[1][data.edge_index[0] == current_node].tolist()

                    # Filter out the nodes that were visited in the last X timesteps
                    adj_nodes = [node for node in adj_nodes if t - last_visited[node] > self.X]
                    
                    # Filter out the nodes that don't have the correct key attribute
                    adj_nodes = [node for node in adj_nodes if int(data.x[node][1][pointer["key_attributes"][0]]) != int(pointer["key_attributes"][1])]

                    # If there are no adjacent nodes or the RNG call is below the threshold,
                    # move the pointer to a random not recently visited node
                    if not adj_nodes or random.random() < self.rng_threshold:
                        not_recently_visited_nodes = [node for node in range(num_nodes) if t - last_visited[node] > self.X]
                        if not_recently_visited_nodes:
                            current_node = random.choice(not_recently_visited_nodes)
                        else:
                            continue
                    else:
                        # Randomly select an adjacent node
                        current_node = random.choice(adj_nodes)

                    # Get the nodes X hops away from the current node
                    nodes_X_hops_away, _, _, _ = k_hop_subgraph(current_node, self.K, data.edge_index)
                    node_list = []
                    labels = []
                    # Process the data of the nodes X hops away
                    for node in nodes_X_hops_away.tolist():
                        node_list.append(data.x[node])
                        labels.append(data.y[node])
                    self.process_node_data(node_list, i, labels, mode)

                    # Update the current node and the last visited time of the pointer
                    pointer['current_node'] = current_node
                    pointer['last_visited'][current_node] = t

