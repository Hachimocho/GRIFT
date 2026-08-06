import random
from collections import defaultdict, deque
from traversals.Traversal import Traversal
import importlib
from models.DQNModel import DQNModel
from nodes.atrnode import AttributeNode

class IValueTraversalClusterHop(Traversal):
    """Traverses the graph by moving pointers to information-rich nodes,
    periodically hopping to clusters with the highest average I-value.
    Uses DQN models from IValueTrainer for I-value prediction.
    """
    tags = ["i-value", "cluster-hop"]
    hyperparameters: dict | None = None

    def __init__(self, graph, num_pointers, num_steps, trainer=None,
                 return_delay=10, warp_chance=0.005, predictor_update_period=50,
                 bias_hop_period=100, pessimistic_i_value=1.0):
        """Initialize an IValueTraversal object with cluster hopping."""
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
        self.current_bias_hop_pointer_index = 0  # New attribute
        self.hop_i_value_history = []  # Stores dicts of {subgroup: avg_ival} per hop
        
        # --- Cluster Hop Parameters --- 
        self.bias_hop_period = bias_hop_period
        self.pessimistic_i_value = pessimistic_i_value
        self.bias_attributes = ['Ground Truth Gender', 'Ground Truth Race', 'Ground Truth Age']
        # --- End Cluster Hop Parameters ---
        
        self.reset_pointers()

    def _get_i_value(self, pointer_data, node, use_din=False):
        """Safely get the I-value for a node, using cache, trainer (optionally DQN), or pessimistic default."""

        # If using DQN, prioritize trainer call
        if use_din:
            if self.trainer:
                try:
                    # Use the DQN-based method
                    i_val = self.trainer.get_i_value(node, 0)
                    if isinstance(i_val, (int, float)):
                        pointer_data['i_values'][node] = i_val # Update cache
                        #print(f"Using trainer I-value {i_val} for node {node.node_id}")
                        return i_val
                    else:
                        print(f"Trainer returned non-numeric I-value for {node.node_id}: {i_val}")
                except Exception as e:
                    print(f"Trainer error getting I-value for {node.node_id}: {e}")
                    # Fall through to cache/pessimistic if trainer fails
            else:
                print(f"No trainer available for DQN I-value prediction for {node.node_id}")
            
            # If trainer failed or wasn't available, try cache as fallback before pessimistic
            i_val_cached = pointer_data['i_values'].get(node, None)
            if i_val_cached is not None:
                #print(f"Using cached I-value {i_val_cached} for node {node.node_id} after trainer failure/absence")
                return i_val_cached
            else:
                # If trainer failed AND not in cache, use pessimistic
                pointer_data['i_values'][node] = self.pessimistic_i_value 
                #print(f"Using pessimistic I-value {self.pessimistic_i_value} for node {node.node_id} (trainer failed, not cached)")
                return self.pessimistic_i_value

        # If not using DQN (use_din=False), check cache first
        else: 
            i_val_cached = pointer_data['i_values'].get(node, None)
            if i_val_cached is not None:
                #print(f"Using cached I-value {i_val_cached} for node {node.node_id} (use_din=False)")
                return i_val_cached
            else:
                # Fallback to pessimistic value if not in cache and not using DQN
                pointer_data['i_values'][node] = self.pessimistic_i_value
                #print(f"Using pessimistic I-value {self.pessimistic_i_value} for node {node.node_id} (not cached, use_din=False)")
                return self.pessimistic_i_value

    def reset_pointers(self):
        """Reset pointers and initialize I-values pessimistically."""
        self.t = 0  # Reset time step counter
        self.pointers = []
        all_nodes = list(self.graph.get_nodes()) # Get nodes once
        if not all_nodes:
             print("Warning: No nodes found in graph during reset_pointers.")
             return
             
        # Initialize pointers with random nodes
        for _ in range(self.num_pointers):
            current_node = self.rng.choice(all_nodes) # Use pre-fetched list
            pointer = {
                'current_node': current_node,
                'last_visited': {},
                'i_values': {}, # Initialize empty, values will be fetched/defaulted in _get_i_value
                'path': [],  # New attribute
                'last_node_id': None,  # New attribute
                'visited_nodes': set(),  # New attribute
                'steps': 0  # New attribute
            }
            self.pointers.append(pointer)
            # Pre-warm I-values (optional, could be slow) - Removed pre-warming for lazy loading
            # for node in all_nodes:
            #    self._get_i_value(pointer, node) # Initialize using the safe getter

    def update_i_values(self, pointer_idx):
        """Update I-values for all nodes using the trainer's prediction."""
        if not self.trainer:
            return
            
        pointer = self.pointers[pointer_idx]
        for node in self.graph.get_nodes():
            # Use the safe getter which handles trainer errors and stores the value
            self._get_i_value(pointer, node, use_din=True)
            # Old direct call: pointer['i_values'][node] = self.trainer.get_i_value(node, 0)  # Using first model's DQN
    
    def get_pointers(self):
        return self.pointers
    
    def get_current_batch_nodes(self):
        """Get the current batch of nodes being processed."""
        return [self.pointers[i]['current_node'] for i in range(self.num_pointers) if self.pointers[i]['current_node'] is not None]
    
    def get_hop_i_value_history(self):
        """Returns the history of calculated average I-values per subgroup during hops."""
        return self.hop_i_value_history

    def traverse(self, batch_size=32):
        """Move pointers based on I-values, constraints, and periodic bias hops."""
        if self.t >= self.num_steps:
            return []
            
        self.t += 1
        batch_nodes = []
        visited_this_batch = set()
        #print(f"\n--- Traversal Step {self.t} ---")
        # print(f"Bias hop period: {self.bias_hop_period}")
        # print(f"self.t % self.bias_hop_period: {self.t % self.bias_hop_period}")
        # --- Bias Hop Logic --- 
        if self.bias_hop_period > 0 and self.t > 0 and self.t % self.bias_hop_period == 0:
            pointer_to_hop_idx = self.current_bias_hop_pointer_index % self.num_pointers
            pointer_to_hop_data = self.pointers[pointer_to_hop_idx]
            #print(f"\n--- Bias Hop Check at t={self.t} for Pointer {pointer_to_hop_idx} ---")
            
            # Calculate average I-value for each subgroup defined by bias_attributes combination
            subgroup_i_values = defaultdict(lambda: {'sum': 0.0, 'count': 0})
            all_nodes_for_hop = list(self.graph.get_nodes()) # Consider all nodes for hop target pool
            
            for node in all_nodes_for_hop:
                if not isinstance(node, AttributeNode) or not hasattr(node, 'attributes') or not node.attributes:
                    continue # Skip nodes without attributes
                    
                # Create subgroup tuple (handle missing attributes)
                subgroup_key_list = []
                skip_node = False
                for attr_name in self.bias_attributes:
                    attr_value = node.attributes.get(attr_name, 'MISSING')
                    # Optional: Skip nodes missing any bias attribute for hop calculation
                    if attr_value == 'MISSING':
                        print(f"Skipping node {node.node_id} due to missing bias attribute {attr_name}")
                        skip_node = True
                        break
                    subgroup_key_list.append(f"{attr_name}_{attr_value}") # Create descriptive string keys
                
                if skip_node:
                    continue
                    
                subgroup_key = tuple(sorted(subgroup_key_list)) # Use sorted tuple as dict key
                
                # Get I-value safely (using the pointer's perspective for consistency)
                # USE CACHED/DEFAULT FOR HOP CALCULATION
                i_val = self._get_i_value(pointer_to_hop_data, node, use_din=False) 
                
                subgroup_i_values[subgroup_key]['sum'] += i_val
                subgroup_i_values[subgroup_key]['count'] += 1
                
            # Calculate averages and find the best subgroup
            avg_i_values = {}
            max_avg_i_value = -float('inf')
            best_subgroup_key = None
            
            # Remove noisy print statements
            for subgroup_key, data in subgroup_i_values.items():
                if data['count'] > 0:
                    avg = data['sum'] / data['count']
                    avg_i_values[subgroup_key] = avg
                    if avg > max_avg_i_value:
                        max_avg_i_value = avg
                        best_subgroup_key = subgroup_key
                # else:
                #    print(f"  {subgroup_key}: N/A (Count: 0)")
            
            # Store the calculated averages for this hop instance
            if avg_i_values:
                self.hop_i_value_history.append(avg_i_values)

            # Hop to a random node within the best subgroup
            if best_subgroup_key:
                # print(f"Target Subgroup for Hop: {best_subgroup_key} (Max Avg I-Value: {max_avg_i_value:.4f})")
                target_nodes_in_subgroup = []
                for node in all_nodes_for_hop:
                     if not isinstance(node, AttributeNode) or not hasattr(node, 'attributes') or not node.attributes:
                         continue
                     # Recreate the key for comparison
                     current_node_key_list = []
                     valid_node = True
                     for attr_name in self.bias_attributes:
                          attr_value = node.attributes.get(attr_name, 'MISSING')
                          # if attr_value == 'MISSING': # Apply same skip logic as above if used
                          #    valid_node = False
                          #    break
                          current_node_key_list.append(f"{attr_name}_{attr_value}")
                          
                     if not valid_node:
                         continue
                         
                     current_node_key = tuple(sorted(current_node_key_list))
                     if current_node_key == best_subgroup_key:
                          target_nodes_in_subgroup.append(node)
                          
                if target_nodes_in_subgroup:
                    # --- Prevent hopping to the same node if it's the only one in the best subgroup ---
                    if len(target_nodes_in_subgroup) == 1 and target_nodes_in_subgroup[0] == pointer_to_hop_data['current_node']:
                        # print(f"Skipping hop for Pointer {pointer_to_hop_idx}: Target subgroup {best_subgroup_key} only contains the current node {pointer_to_hop_data['current_node'].node_id}.")
                        pass
                    else:
                        hop_node = self.rng.choice(target_nodes_in_subgroup)
                        # print(f"Hopping Pointer {pointer_to_hop_idx} to node {hop_node.node_id} in subgroup {best_subgroup_key}")
                        pointer_to_hop_data['current_node'] = hop_node
                        pointer_to_hop_data['last_visited'] = {}
                else:
                    # print(f"Warning: No nodes found for the best subgroup {best_subgroup_key}. No hop performed.")
                    pass
            else:
                # print("Warning: Could not determine best subgroup. No hop performed.")
                pass

            self.current_bias_hop_pointer_index += 1 # Move to the next pointer for the next hop cycle
        # --- End Bias Hop Logic --- 

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
                    # Use the safe getter for I-values
                    #print("Updating I-values for valid neighbors...")
                    i_values = [self._get_i_value(pointer, n, use_din=True) for n in valid_neighbors]
                    # Old: i_values = [pointer['i_values'].get(n, self.pessimistic_i_value) for n in valid_neighbors] # Use pessimistic if not found
                    
                    if not i_values: # Should not happen if valid_neighbors is not empty, but check
                        print(f"Warning: No I-values for valid neighbors of node {pointer['current_node'].id}")
                        continue
                        
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
                
        self.current_batch_nodes = batch_nodes # Store the collected nodes
        return batch_nodes

    def __len__(self):
        """Return the number of steps in the traversal."""
        return self.num_steps

class IValueTraversalClusterHopSubcluster(IValueTraversalClusterHop):
    """
    Traverses the graph by moving pointers to information-rich nodes, using Louvain subclusters for area selection.
    During bias-hop, hops to a subcluster with high average I-value (softmax), then to a node within that subcluster (excluding outliers).
    Maintains epsilon-based random jumps and all bias-hop logic. Falls back to standard cluster-hop traversal if no subcluster info.
    """
    def __init__(self, graph, num_pointers, num_steps, trainer=None, return_delay=10, warp_chance=0.005, predictor_update_period=50, bias_hop_period=100, pessimistic_i_value=1.0, outlier_std=2.0, softmax_temp=0.5):
        super().__init__(graph, num_pointers, num_steps, trainer, return_delay, warp_chance, predictor_update_period, bias_hop_period, pessimistic_i_value)
        self.outlier_std = outlier_std
        self.softmax_temp = softmax_temp

    def _softmax(self, values, temp=1.0):
        import numpy as np
        v = np.array(values)
        v = v - np.max(v)  # for numerical stability
        exp_v = np.exp(v / temp)
        probs = exp_v / np.sum(exp_v)
        return probs

    def _choose_subcluster(self, pointer):
        subclusters = getattr(self.graph, 'subclusters', None)
        if not subclusters:
            return None, None
        from collections import defaultdict
        subcluster_to_nodes = defaultdict(list)
        for node in self.graph.get_nodes():
            sc_id = subclusters.get(node.node_id, None)
            if sc_id is not None:
                subcluster_to_nodes[sc_id].append(node)
        subcluster_stats = {}
        for sc_id, nodes in subcluster_to_nodes.items():
            i_vals = [pointer['i_values'].get(n, 0.5) for n in nodes]
            if not i_vals:
                continue
            mean = sum(i_vals) / len(i_vals)
            std = (sum((x - mean) ** 2 for x in i_vals) / len(i_vals)) ** 0.5 if len(i_vals) > 1 else 0.0
            subcluster_stats[sc_id] = {'mean': mean, 'std': std, 'nodes': nodes, 'i_vals': i_vals}
        eligible_subclusters = []
        eligible_means = []
        for sc_id, stats in subcluster_stats.items():
            mean, std, nodes, i_vals = stats['mean'], stats['std'], stats['nodes'], stats['i_vals']
            non_outliers = [n for n, v in zip(nodes, i_vals) if v < mean + self.outlier_std * std]
            if non_outliers:
                eligible_subclusters.append(sc_id)
                eligible_means.append(mean)
        if not eligible_subclusters:
            return None, None
        probs = self._softmax(eligible_means, temp=self.softmax_temp)
        chosen_sc = self.rng.choices(eligible_subclusters, weights=probs, k=1)[0]
        stats = subcluster_stats[chosen_sc]
        mean, std, nodes, i_vals = stats['mean'], stats['std'], stats['nodes'], stats['i_vals']
        candidates = [(n, v) for n, v in zip(nodes, i_vals) if v < mean + self.outlier_std * std]
        if not candidates:
            return None, None
        cand_nodes, cand_vals = zip(*candidates)
        cand_probs = self._softmax(cand_vals, temp=self.softmax_temp)
        return cand_nodes, cand_probs

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
        subclusters = getattr(self.graph, 'subclusters', None)
        if not subclusters:
            return super().traverse(batch_size)
        for pointer in self.pointers:
            if self.rng.random() < self.warp_chance:
                new_node = self.graph.get_random_node(rng=self.rng)
                pointer['current_node'] = new_node
                if new_node not in visited_this_batch:
                    batch_nodes.append(new_node)
                    visited_this_batch.add(new_node)
                continue
            cand_nodes, cand_probs = self._choose_subcluster(pointer)
            if not cand_nodes:
                # fallback
                continue
            next_node = self.rng.choices(cand_nodes, weights=cand_probs, k=1)[0]
            pointer['current_node'] = next_node
            if next_node not in visited_this_batch:
                batch_nodes.append(next_node)
                visited_this_batch.add(next_node)
        self.current_batch_nodes = batch_nodes
        return batch_nodes