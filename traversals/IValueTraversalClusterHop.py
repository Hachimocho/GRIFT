import random
from collections import defaultdict
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
        
        # --- Cluster Hop Parameters --- 
        self.bias_hop_period = bias_hop_period
        self.pessimistic_i_value = pessimistic_i_value
        self.bias_attributes = ['Ground Truth Gender', 'Ground Truth Race', 'Ground Truth Age']
        # --- End Cluster Hop Parameters ---
        
        self.reset_pointers()

    def _get_i_value(self, pointer_data, node):
        """Safely get I-value for a node, using pessimistic default."""
        # Try getting from the pointer's dictionary first
        i_val = pointer_data['i_values'].get(node, None)
        if i_val is not None:
            return i_val
            
        # If not found, try getting from the trainer (might be expensive)
        if self.trainer:
            try:
                # Assume get_i_value might fail or return an indicator, handle appropriately
                i_val = self.trainer.get_i_value(node, 0) # Using first model's DQN
                # Ensure trainer's value is stored and returned, handle potential errors if needed
                if isinstance(i_val, (int, float)): # Basic check
                    pointer_data['i_values'][node] = i_val
                    return i_val
            except Exception as e:
                # Log error if needed, e.g., print(f"Trainer error getting I-value for {node.id}: {e}")
                pass # Fall through to pessimistic value

        # Fallback to pessimistic value
        pointer_data['i_values'][node] = self.pessimistic_i_value
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
            current_node = random.choice(all_nodes) # Use pre-fetched list
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
            self._get_i_value(pointer, node)
            # Old direct call: pointer['i_values'][node] = self.trainer.get_i_value(node, 0)  # Using first model's DQN
    
    def get_pointers(self):
        return self.pointers
    
    def get_current_batch_nodes(self):
        """Get the current batch of nodes being processed."""
        return [self.pointers[i]['current_node'] for i in range(self.num_pointers) if self.pointers[i]['current_node'] is not None]
    
    def traverse(self, batch_size=32):
        """Move pointers based on I-values, constraints, and periodic bias hops."""
        if self.t >= self.num_steps:
            return []
            
        self.t += 1
        batch_nodes = []
        visited_this_batch = set()
        
        # --- Periodic Bias Hop --- 
        if self.bias_hop_period > 0 and self.t > 0 and self.t % self.bias_hop_period == 0:
            print(f"\n--- Bias Hop Check at t={self.t} ---")
            all_nodes = self.graph.get_nodes()
            avg_i_values_by_group = defaultdict(lambda: {'total_i': 0, 'count': 0})
            group_nodes = defaultdict(list) # Store nodes per group

            # Calculate average I-value for each specified attribute group
            for node in all_nodes:
                # Ensure we use a consistent pointer reference for getting I-values during the check
                # For simplicity, let's use the I-values from the first pointer's perspective
                # Note: This could be refined later if needed.
                i_value = self._get_i_value(self.pointers[0], node) 
                for attr_name in self.bias_attributes:
                    if attr_name in node.attributes:
                        attr_value = node.attributes[attr_name]
                        group_key = f"{attr_name}_{attr_value}"
                        avg_i_values_by_group[group_key]['total_i'] += i_value
                        avg_i_values_by_group[group_key]['count'] += 1
                        group_nodes[group_key].append(node) # Store node

            print("Average I-Values per Group:") # New logging
            calculated_averages = {}
            for group, data in avg_i_values_by_group.items():
                if data['count'] > 0:
                    avg = data['total_i'] / data['count']
                    calculated_averages[group] = avg
                    print(f"  {group}: {avg:.4f} (Count: {data['count']})") # New logging
                else:
                    print(f"  {group}: N/A (Count: 0)") # New logging
                    calculated_averages[group] = self.pessimistic_i_value # Use pessimistic if no nodes seen

            # Find the group with the highest average I-value
            if calculated_averages:
                max_avg_i_value = -1
                target_group = None
                # Sort groups alphabetically for consistent selection in case of ties (though unlikely with floats)
                sorted_groups = sorted(calculated_averages.keys())
                for group in sorted_groups:
                    avg = calculated_averages[group]
                    if avg > max_avg_i_value:
                        max_avg_i_value = avg
                        target_group = group
                
                print(f"Target Group for Hop: {target_group} (Max Avg I-Value: {max_avg_i_value:.4f})") # New logging

                if target_group and group_nodes[target_group]:
                    # Select a random node from the target group
                    selected_node = random.choice(group_nodes[target_group])
                    print(f"Selected Node for Hop: {selected_node.id} (from group {target_group})") # New logging

                    # Determine which pointer to move (cycle through them)
                    pointer_to_move_idx = self.current_bias_hop_pointer_index
                    pointer_to_move = self.pointers[pointer_to_move_idx]
                    print(f"Moving Pointer Index: {pointer_to_move_idx} to Node {selected_node.id}") # New logging

                    # Move the pointer
                    pointer_to_move['current_node'] = selected_node
                    pointer_to_move['path'].append(selected_node.id) # Add node to path
                    pointer_to_move['last_node_id'] = selected_node.id # Update last node id
                    # Reset step counter for this pointer after hop?
                    # pointer_to_move['steps'] = 0 
                    # Add target node to visited nodes for this pointer?
                    # pointer_to_move['visited_nodes'].add(selected_node.id)

                    # Increment and wrap the pointer index for the next hop
                    self.current_bias_hop_pointer_index = (self.current_bias_hop_pointer_index + 1) % self.num_pointers
                else:
                    print(f"Warning: Target group '{target_group}' has no nodes or no groups calculated. Skipping hop.")
            else:
                 print("Warning: No groups to calculate average I-values from. Skipping hop.")
            print("--- End Bias Hop Check ---")
        # --- End Bias Hop ---
        
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
                    if random.random() < self.warp_chance:
                        new_node = self.graph.get_random_node()
                        pointer['current_node'] = new_node
                        if new_node not in visited_this_batch:
                            new_nodes.append(new_node)
                            visited_this_batch.add(new_node)
                        continue
                        
                    # Get neighboring nodes
                    neighbors = pointer['current_node'].get_adjacent_nodes()
                    if not neighbors:
                        new_node = self.graph.get_random_node()
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
                        new_node = self.graph.get_random_node()
                        pointer['current_node'] = new_node
                        if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                            new_nodes.append(new_node)
                            visited_this_batch.add(new_node)
                        continue
                        
                    # Choose next node based on I-values
                    # Use the safe getter for I-values
                    i_values = [self._get_i_value(pointer, n) for n in valid_neighbors]
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
                    remaining_nodes = list(set(self.graph.get_nodes()) - visited_this_batch)
                    if remaining_nodes:
                        random_nodes = random.sample(remaining_nodes, min(batch_size - len(batch_nodes), len(remaining_nodes)))
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