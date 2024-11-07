import random
from traversals.Traversal import Traversal
import importlib

class IValueTraversal(Traversal):
    """
    Traverses the graph by moving pointers to information-rich nodes.
    """
    tags = ["attributes"]
    
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "return_delay": {"distribution": "int_uniform", "min": 10, "max": 100},
            "warp_chance": {"distribution": "uniform", "min": 0.0, "max": 0.999},
            "predictor_update_period": {"distribution": "int_uniform", "min": 10, "max": 100},
            "predictor_model": {"values": ["RandomIValuePredictor", "DQNIValuePredictor"]},
        }
    }

    def __init__(self, graph, num_pointers, num_steps, return_delay, warp_chance, predictor_update_period, predictor_model):
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
        self.predictor_update_period = predictor_update_period
        self.predictor_model = importlib.import_module(predictor_model)
        self.reset_pointers()
        
    def get_pointers(self):
        return self.pointers
    
    def reset_pointers(self):
        self.pointers = [{'current_node': self.graph.get_random_node(), 'last_visited': {}, 'i_values': {node: random.random() for node in self.graph.get_nodes()}, 'i_value_predictor': self.predictor_model()} for _ in range(self.num_pointers)]
    
    def traverse(self):
        if self.t > self.num_steps:
            raise RuntimeError("Maximum number of steps exceeded.")
        self.t += 1
        
        for i, pointer in enumerate(self.pointers):
            # Update predictor if necessary
            if self.t % self.predictor_update_period == 0:
                pointer['i_value_predictor'].update()
            
            # Get the adjacent nodes
            adj_nodes = pointer['current_node'].get_adjacent_nodes()

            # Filter out the nodes that were visited in the last X timesteps
            adj_nodes = [node for node in adj_nodes if self.t - pointer['last_visited'][node] > self.return_delay]
            
            # Update I values for adjacent nodes
            for node in adj_nodes:
                pointer['i_values'][node] = pointer['i_value_predictor'].predict(node)

            if adj_nodes and (random.random() > self.warp_chance):
                # Select the node with the highest I value
                max_i_value = max(pointer['i_values'].values())
                max_i_nodes = [node for node, i_value in pointer['i_values'].items() if i_value == max_i_value]
                current_node = random.choice(max_i_nodes)
                pointer['current_node'] = current_node
            else:
                # If there are no adjacent nodes or the warp triggers,
                # move the pointer to a random node (can visit recently visited nodes this way, prevents hardlocks on small graphs)
                pointer['current_node'] = self.graph.get_random_node()

            for key in pointer['last_visited'].keys():
                pointer['last_visited'][key] -= 1
                if pointer['last_visited'][key] <= 0:
                    pointer['last_visited'].pop(key)
            pointer['last_visited'][current_node] = self.return_delay