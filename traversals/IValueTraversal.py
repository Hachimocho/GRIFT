import random
from traversals.Traversal import Traversal
import importlib
from models.DQNModel import DQNModel
from nodes.atrnode import AttributeNode

#1
class IValueTraversal(Traversal):
    """Traverses the graph by moving pointers to information-rich nodes.
    Uses DQN models from IValueTrainer for I-value prediction.
    """
    tags = ["attributes"]
    
    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "return_delay": {"distribution": "int_uniform", "min": 10, "max": 100},
            "warp_chance": {"distribution": "uniform", "min": 0.0, "max": 0.999},
            "predictor_update_period": {"distribution": "int_uniform", "min": 10, "max": 100},
        }
    }

    def __init__(self, graph, num_pointers, num_steps, trainer=None, return_delay=10, warp_chance=0.005, predictor_update_period=50):
        """Initialize an IValueTraversal object."""
        super().__init__()
        self.graph = graph
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.return_delay = return_delay
        self.t = 0
        self.warp_chance = warp_chance
        self.predictor_update_period = predictor_update_period
        self.trainer = trainer
        self.current_batch_nodes = []  # Store nodes from current batch
        self.reset_pointers()
    
    def __len__(self):
        """Return the number of steps in the traversal."""
        return self.num_steps
    
    def get_pointers(self):
        return self.pointers
    
    def get_current_batch_nodes(self):
        """Get the current batch of nodes being processed."""
        return [self.pointers[i]['current_node'] for i in range(self.num_pointers) if self.pointers[i]['current_node'] is not None]
    
    def reset_pointers(self):
        """Reset pointers and initialize I-values."""
        self.t = 0  # Reset time step counter
        self.pointers = []
        
        # Initialize pointers with random nodes
        for _ in range(self.num_pointers):
            current_node = self.graph.get_random_node(rng=self.rng)
            pointer = {
                'current_node': current_node,
                'last_visited': {},
                'i_values': {}
            }
            
            # Initialize I-values for all nodes
            for node in self.graph.get_nodes():
                if self.trainer:
                    # Use trainer's DQN to predict I-values
                    pointer['i_values'][node] = self.trainer.get_i_value(node, 0)  # Using first model's DQN
                else:
                    # Fallback to random I-values if no trainer
                    pointer['i_values'][node] = self.rng.random()
            
            self.pointers.append(pointer)
    
    def update_i_values(self, pointer_idx):
        """Update I-values using trainer's DQN predictions."""
        if not self.trainer:
            return
            
        pointer = self.pointers[pointer_idx]
        for node in self.graph.get_nodes():
            pointer['i_values'][node] = self.trainer.get_i_value(node, 0)  # Using first model's DQN
    
    def traverse(self, batch_size=32):
        """Move pointers based on I-values and constraints."""
        if self.t >= self.num_steps:
            return []
            
        self.t += 1
        batch_nodes = []
        visited_this_batch = set()
        
        # Update I-values periodically using trainer's predictions
        if self.trainer and self.t % self.predictor_update_period == 0:
            for pointer_idx in range(len(self.pointers)):
                self.update_i_values(pointer_idx)
                
        # Keep collecting nodes until we have enough or can't find more
        while len(batch_nodes) < batch_size:
            new_nodes = []
            for pointer in self.pointers:
                try:
                    # Random warp with probability warp_chance
                    if self.rng.random() < self.warp_chance:
                        new_node = self.graph.get_random_node(rng=self.rng)
                        pointer['current_node'] = new_node
                        if new_node not in visited_this_batch:
                            new_nodes.append(new_node)
                            visited_this_batch.add(new_node)
                        continue
                        
                    # Get neighboring nodes
                    neighbors = pointer['current_node'].get_adjacent_nodes()
                    if not neighbors:
                        new_node = self.graph.get_random_node(rng=self.rng)
                        pointer['current_node'] = new_node
                        if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                            new_nodes.append(new_node)
                            visited_this_batch.add(new_node)
                        continue
                        
                    # Filter out recently visited nodes
                    current_time = self.t
                    valid_neighbors = [
                        n for n in neighbors
                        if current_time - pointer['last_visited'].get(n, -self.return_delay) >= self.return_delay
                        and n not in visited_this_batch
                        and isinstance(n, AttributeNode)  # Only consider AttributeNodes
                    ]
                    
                    if not valid_neighbors:
                        new_node = self.graph.get_random_node(rng=self.rng)
                        pointer['current_node'] = new_node
                        if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                            new_nodes.append(new_node)
                            visited_this_batch.add(new_node)
                        continue
                        
                    # Choose next node based on I-values
                    i_values = [pointer['i_values'].get(n, 0.5) for n in valid_neighbors]  # Default to 0.5 if not found
                    next_node = valid_neighbors[i_values.index(max(i_values))]
                    
                    # Update visited time and move pointer
                    pointer['last_visited'][next_node] = current_time
                    pointer['current_node'] = next_node
                    
                    if next_node not in visited_this_batch:
                        new_nodes.append(next_node)
                        visited_this_batch.add(next_node)
                        
                except Exception as e:
                    print(f"Error in traverse: {str(e)}")
                    continue
                    
            # If we couldn't find any new nodes, break
            if not new_nodes:
                # If we haven't found enough nodes for a minimal batch, try random sampling
                if len(batch_nodes) < 8:  # Minimum batch size threshold
                    # Sorted before sampling: this list comes from a set of Node objects, and
                    # Node.__hash__ hashes a string node_id, so its order is
                    # PYTHONHASHSEED-dependent and varies between processes.
                    remaining_nodes = sorted(
                        set(self.graph.get_nodes()) - visited_this_batch,
                        key=lambda node: str(node.node_id),
                    )
                    if remaining_nodes:
                        random_nodes = self.rng.sample(remaining_nodes, min(batch_size - len(batch_nodes), len(remaining_nodes)))
                        batch_nodes.extend([n for n in random_nodes if isinstance(n, AttributeNode)])
                        visited_this_batch.update(random_nodes)
                break
                
            # Add new nodes to batch
            batch_nodes.extend(new_nodes)
            
            # If we've collected more than batch_size nodes, trim the excess
            if len(batch_nodes) > batch_size:
                batch_nodes = batch_nodes[:batch_size]
                break
                
        # If we still don't have enough nodes for a minimal batch, skip this traversal
        if len(batch_nodes) < 8:  # Minimum batch size threshold
            return []
                
        self.current_batch_nodes = batch_nodes
        return batch_nodes

class IValueTraversalSubcluster(IValueTraversal):
    """
    Traverses the graph by moving pointers to information-rich nodes, but uses Louvain subclusters for area selection.
    At each step, selects a subcluster with high average I-value (softmax), then a node within that subcluster (excluding outliers, softmax).
    Maintains epsilon-based random jumps. Falls back to standard traversal if no subcluster info.
    """
    def __init__(self, graph, num_pointers, num_steps, trainer=None, return_delay=10, warp_chance=0.005, predictor_update_period=50, outlier_std=2.0, softmax_temp=0.5):
        super().__init__(graph, num_pointers, num_steps, trainer, return_delay, warp_chance, predictor_update_period)
        self.outlier_std = outlier_std
        self.softmax_temp = softmax_temp

    def _softmax(self, values, temp=1.0):
        import numpy as np
        v = np.array(values)
        v = v - np.max(v)  # for numerical stability
        exp_v = np.exp(v / temp)
        probs = exp_v / np.sum(exp_v)
        return probs

    def traverse(self, batch_size=32):
        if self.t >= self.num_steps:
            return []
        self.t += 1
        batch_nodes = []
        visited_this_batch = set()
        # Update I-values periodically using trainer's predictions
        if self.trainer and self.t % self.predictor_update_period == 0:
            for pointer_idx in range(len(self.pointers)):
                self.update_i_values(pointer_idx)
        # Use subclusters if available
        subclusters = getattr(self.graph, 'subclusters', None)
        if not subclusters:
            # Fallback to standard traversal
            return super().traverse(batch_size)
        # Build subcluster -> [nodes] map
        from collections import defaultdict
        subcluster_to_nodes = defaultdict(list)
        for node in self.graph.get_nodes():
            sc_id = subclusters.get(node.node_id, None)
            if sc_id is not None:
                subcluster_to_nodes[sc_id].append(node)
        # Precompute I-value stats for each subcluster
        pointer = self.pointers[0]  # Use first pointer for stats (could generalize)
        subcluster_stats = {}
        for sc_id, nodes in subcluster_to_nodes.items():
            i_vals = [pointer['i_values'].get(n, 0.5) for n in nodes]
            if not i_vals:
                continue
            mean = sum(i_vals) / len(i_vals)
            std = (sum((x - mean) ** 2 for x in i_vals) / len(i_vals)) ** 0.5 if len(i_vals) > 1 else 0.0
            subcluster_stats[sc_id] = {'mean': mean, 'std': std, 'nodes': nodes, 'i_vals': i_vals}
        # Exclude subclusters where all nodes are outliers
        eligible_subclusters = []
        eligible_means = []
        for sc_id, stats in subcluster_stats.items():
            mean, std, nodes, i_vals = stats['mean'], stats['std'], stats['nodes'], stats['i_vals']
            non_outliers = [n for n, v in zip(nodes, i_vals) if v < mean + self.outlier_std * std]
            if non_outliers:
                eligible_subclusters.append(sc_id)
                eligible_means.append(mean)
        if not eligible_subclusters:
            # Fallback to standard traversal
            return super().traverse(batch_size)
        # Softmax sampling over subcluster means
        probs = self._softmax(eligible_means, temp=self.softmax_temp)
        chosen_sc = self.rng.choices(eligible_subclusters, weights=probs, k=1)[0]
        # Within chosen subcluster, filter out outlier nodes
        stats = subcluster_stats[chosen_sc]
        mean, std, nodes, i_vals = stats['mean'], stats['std'], stats['nodes'], stats['i_vals']
        candidates = [(n, v) for n, v in zip(nodes, i_vals) if v < mean + self.outlier_std * std and n not in visited_this_batch]
        if not candidates:
            # Fallback to standard traversal
            return super().traverse(batch_size)
        # Softmax sampling among candidates
        cand_nodes, cand_vals = zip(*candidates)
        cand_probs = self._softmax(cand_vals, temp=self.softmax_temp)
        # Epsilon/random warp logic
        for pointer in self.pointers:
            if self.rng.random() < self.warp_chance:
                new_node = self.graph.get_random_node(rng=self.rng)
                pointer['current_node'] = new_node
                if new_node not in visited_this_batch:
                    batch_nodes.append(new_node)
                    visited_this_batch.add(new_node)
                continue
            # Otherwise, pick from candidates
            next_node = self.rng.choices(cand_nodes, weights=cand_probs, k=1)[0]
            pointer['current_node'] = next_node
            if next_node not in visited_this_batch:
                batch_nodes.append(next_node)
                visited_this_batch.add(next_node)
        self.current_batch_nodes = batch_nodes
        return batch_nodes