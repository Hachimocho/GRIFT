"""
Graph Reduction and Restoration Manager

Manages dynamic graph reduction and restoration strategies during training.
Supports multiple reduction strategies (Max/Min/Mix-Max Ival, Random) and
restoration strategies (Random Pool, Targeted, Reversion).
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def _stream(component, fallback_seed=0):
    """Private RNG stream for `component`.

    Replaces draws from the process-global `random` module. Sharing that global
    stream meant RNG consumption anywhere upstream shifted these decisions.
    """
    from test_helpers.determinism import component_rng
    return component_rng(component, fallback_seed=fallback_seed)


class GraphReductionManager:
    """
    Manages graph reduction and restoration strategies during training.
    
    This class handles:
    - Executing reduction strategies at configurable intervals
    - Tracking removed nodes with their I-values
    - Executing restoration strategies when validation performance drops
    - Maintaining state between epochs for reversion strategy
    """

    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "reduction_strategy": {"values": ["none", "lowest_ivalue", "threshold"]},
            "reduction_interval": {"values": ["end_of_epoch", "every_n_steps"]},
        }
    }

    def __init__(self,
                 reduction_strategy: str = "none",
                 reduction_percentage: float = 0.0,
                 reduction_top_percentage: float = 0.0,
                 reduction_bottom_percentage: float = 0.0,
                 reduction_interval: str = "end_of_epoch",
                 reduction_interval_steps: int = 100,
                 restoration_strategy: str = "none",
                 restoration_percentage: float = 50.0,
                 restoration_trigger_threshold: float = 0.0):
        """
        Initialize the Graph Reduction Manager.
        
        Args:
            reduction_strategy: Strategy for reduction ("max_ival", "min_ival", "mix_max_ival", "random", "none")
            reduction_percentage: Percentage of nodes to remove (0-100)
            reduction_top_percentage: Top percentage for mix_max strategy (0-100)
            reduction_bottom_percentage: Bottom percentage for mix_max strategy (0-100)
            reduction_interval: When to reduce ("end_of_epoch" or "every_n_steps")
            reduction_interval_steps: Number of steps between reductions (if interval is "every_n_steps")
            restoration_strategy: Strategy for restoration ("random_pool", "targeted", "reversion", "none")
            restoration_percentage: Percentage of removed nodes to restore (0-100)
            restoration_trigger_threshold: Minimum drop threshold for restoration (default: 0.0)
        """
        self.reduction_strategy = reduction_strategy
        self.reduction_percentage = reduction_percentage
        self.reduction_top_percentage = reduction_top_percentage
        self.reduction_bottom_percentage = reduction_bottom_percentage
        self.reduction_interval = reduction_interval
        self.reduction_interval_steps = reduction_interval_steps
        self.restoration_strategy = restoration_strategy
        self.restoration_percentage = restoration_percentage
        self.restoration_trigger_threshold = restoration_trigger_threshold
        
        # State tracking
        self.removed_nodes_pool: List = []  # List of removed node objects
        self.removed_nodes_ivalues: Dict[str, float] = {}  # node_id -> i_value
        self.epoch_removal_history: Dict[int, List] = {}  # epoch -> list of removed nodes
        self.step_counter = 0
        self.last_reduction_step = 0
        
        # Statistics
        self.reduction_stats = {
            'total_reductions': 0,
            'total_nodes_removed': 0,
            'total_restorations': 0,
            'total_nodes_restored': 0
        }
    
    def should_reduce(self, current_step: int, epoch: int) -> bool:
        """
        Check if reduction should be performed at this point.
        
        Args:
            current_step: Current training step
            epoch: Current epoch
            
        Returns:
            bool: True if reduction should be performed
        """
        if self.reduction_strategy == "none" or not self.reduction_enabled():
            return False
        
        if self.reduction_interval == "end_of_epoch":
            # Reduction happens at end of epoch, caller should check epoch boundaries
            return False  # Caller handles epoch boundaries
        elif self.reduction_interval == "every_n_steps":
            steps_since_last = current_step - self.last_reduction_step
            return steps_since_last >= self.reduction_interval_steps
        
        return False
    
    def reduction_enabled(self) -> bool:
        """Check if reduction is enabled."""
        return self.reduction_strategy != "none" and self.reduction_percentage > 0
    
    def restoration_enabled(self) -> bool:
        """Check if restoration is enabled."""
        return self.restoration_strategy != "none"
    
    def reduce_graph(self, graph, trainer, epoch: int = 0, step: int = 0) -> Tuple[List, Dict]:
        """
        Execute graph reduction based on configured strategy.
        
        Args:
            graph: HyperGraph instance to reduce
            trainer: Trainer instance (for I-value access)
            epoch: Current epoch number
            step: Current step number
            
        Returns:
            Tuple of (removed_nodes_list, removal_stats_dict)
        """
        if not self.reduction_enabled():
            return [], {}
        
        # Check if reduction should happen now
        if self.reduction_interval == "every_n_steps":
            if not self.should_reduce(step, epoch):
                return [], {}
            self.last_reduction_step = step
        
        removed_nodes = []
        stats = {
            'strategy': self.reduction_strategy,
            'nodes_removed': 0,
            'epoch': epoch,
            'step': step
        }
        
        try:
            if self.reduction_strategy == "max_ival":
                removed_nodes = self._reduce_max_ival(graph, trainer)
            elif self.reduction_strategy == "min_ival":
                removed_nodes = self._reduce_min_ival(graph, trainer)
            elif self.reduction_strategy == "mix_max_ival":
                removed_nodes = self._reduce_mix_max_ival(graph, trainer)
            elif self.reduction_strategy == "random":
                removed_nodes = self._reduce_random(graph)
            else:
                print(f"Warning: Unknown reduction strategy: {self.reduction_strategy}")
                return [], {}
            
            # Store removed nodes in pool
            for node in removed_nodes:
                self.removed_nodes_pool.append(node)
                # Store I-value if available
                if hasattr(trainer, 'get_i_value'):
                    try:
                        i_val = trainer.get_i_value(node, 0)
                        self.removed_nodes_ivalues[node.node_id] = i_val
                    except Exception as e:
                        print(f"Warning: Could not get I-value for node {node.node_id}: {e}")
            
            # Store epoch state for reversion
            if epoch not in self.epoch_removal_history:
                self.epoch_removal_history[epoch] = []
            self.epoch_removal_history[epoch].extend(removed_nodes)
            
            stats['nodes_removed'] = len(removed_nodes)
            self.reduction_stats['total_reductions'] += 1
            self.reduction_stats['total_nodes_removed'] += len(removed_nodes)
            
            print(f"Reduced graph: removed {len(removed_nodes)} nodes using {self.reduction_strategy} strategy")
            
        except Exception as e:
            print(f"Error during graph reduction: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
        
        return removed_nodes, stats
    
    def _reduce_max_ival(self, graph, trainer) -> List:
        """Remove top X% nodes by I-value."""
        if not hasattr(trainer, 'get_i_value'):
            raise ValueError("Trainer does not have get_i_value method. Cannot use I-value reduction without I-value capability.")
        
        nodes = list(graph.get_nodes())
        if len(nodes) == 0:
            return []
        
        # Get I-values for all nodes
        node_ivalues = []
        for node in nodes:
            try:
                i_val = trainer.get_i_value(node, 0)
                node_ivalues.append((node, i_val))
            except Exception as e:
                print(f"Warning: Could not get I-value for node {node.node_id}: {e}")
                # Use a default high I-value to prioritize removal
                node_ivalues.append((node, 1.0))
        
        # Sort by I-value descending (highest first)
        node_ivalues.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate number to remove
        num_to_remove = max(1, int(len(nodes) * self.reduction_percentage / 100.0))
        num_to_remove = min(num_to_remove, len(nodes) - 1)  # Keep at least one node
        
        # Remove top nodes
        removed_nodes = []
        for node, _ in node_ivalues[:num_to_remove]:
            if self._remove_node_from_graph(graph, node):
                removed_nodes.append(node)
        
        return removed_nodes
    
    def _reduce_min_ival(self, graph, trainer) -> List:
        """Remove bottom Y% nodes by I-value."""
        if not hasattr(trainer, 'get_i_value'):
            raise ValueError("Trainer does not have get_i_value method. Cannot use I-value reduction without I-value capability.")
        
        nodes = list(graph.get_nodes())
        if len(nodes) == 0:
            return []
        
        # Get I-values for all nodes
        node_ivalues = []
        for node in nodes:
            try:
                i_val = trainer.get_i_value(node, 0)
                node_ivalues.append((node, i_val))
            except Exception as e:
                print(f"Warning: Could not get I-value for node {node.node_id}: {e}")
                # Use a default low I-value to avoid removal
                node_ivalues.append((node, 0.0))
        
        # Sort by I-value ascending (lowest first)
        node_ivalues.sort(key=lambda x: x[1])
        
        # Calculate number to remove
        num_to_remove = max(1, int(len(nodes) * self.reduction_percentage / 100.0))
        num_to_remove = min(num_to_remove, len(nodes) - 1)  # Keep at least one node
        
        # Remove bottom nodes
        removed_nodes = []
        for node, _ in node_ivalues[:num_to_remove]:
            if self._remove_node_from_graph(graph, node):
                removed_nodes.append(node)
        
        return removed_nodes
    
    def _reduce_mix_max_ival(self, graph, trainer) -> List:
        """Remove top X% and bottom Y% nodes by I-value (mutually exclusive)."""
        if not hasattr(trainer, 'get_i_value'):
            raise ValueError("Trainer does not have get_i_value method. Cannot use I-value reduction without I-value capability.")
        
        nodes = list(graph.get_nodes())
        if len(nodes) == 0:
            return []
        
        # Get I-values for all nodes
        node_ivalues = []
        for node in nodes:
            try:
                i_val = trainer.get_i_value(node, 0)
                node_ivalues.append((node, i_val))
            except Exception as e:
                print(f"Warning: Could not get I-value for node {node.node_id}: {e}")
                # Use a default medium I-value
                node_ivalues.append((node, 0.5))
        
        # Sort by I-value
        node_ivalues.sort(key=lambda x: x[1])
        
        # Calculate numbers to remove
        num_top = max(1, int(len(nodes) * self.reduction_top_percentage / 100.0))
        num_bottom = max(1, int(len(nodes) * self.reduction_bottom_percentage / 100.0))
        
        # Ensure we don't remove more than available
        total_to_remove = num_top + num_bottom
        if total_to_remove >= len(nodes):
            # If trying to remove too many, scale down proportionally
            scale = (len(nodes) - 1) / total_to_remove
            num_top = max(1, int(num_top * scale))
            num_bottom = max(1, int(num_bottom * scale))
        
        # Remove top (highest I-value) and bottom (lowest I-value)
        # Top are at the end of sorted list, bottom are at the beginning
        removed_nodes = []
        
        # Remove bottom nodes first
        for node, _ in node_ivalues[:num_bottom]:
            if self._remove_node_from_graph(graph, node):
                removed_nodes.append(node)
        
        # Remove top nodes (from end, but skip already removed ones)
        remaining_nodes = [n for n in node_ivalues if n[0] not in removed_nodes]
        for node, _ in remaining_nodes[-num_top:]:
            if self._remove_node_from_graph(graph, node):
                removed_nodes.append(node)
        
        return removed_nodes
    
    def _reduce_random(self, graph) -> List:
        """Remove Z% of nodes randomly."""
        nodes = list(graph.get_nodes())
        if len(nodes) == 0:
            return []
        
        # Calculate number to remove
        num_to_remove = max(1, int(len(nodes) * self.reduction_percentage / 100.0))
        num_to_remove = min(num_to_remove, len(nodes) - 1)  # Keep at least one node
        
        # Randomly select nodes to remove
        nodes_to_remove = _stream('reduction.remove').sample(nodes, num_to_remove)
        
        removed_nodes = []
        for node in nodes_to_remove:
            if self._remove_node_from_graph(graph, node):
                removed_nodes.append(node)
        
        return removed_nodes
    
    def _remove_node_from_graph(self, graph, node) -> bool:
        """
        Remove a node from the graph and clean up its edges.
        
        Args:
            graph: HyperGraph instance
            node: Node to remove
            
        Returns:
            bool: True if node was successfully removed
        """
        try:
            # Find node index in graph
            nodes = graph.get_nodes()
            node_index = None
            for i, n in enumerate(nodes):
                if n.node_id == node.node_id:
                    node_index = i
                    break
            
            if node_index is None:
                print(f"Warning: Node {node.node_id} not found in graph")
                return False
            
            # Remove edges connected to this node
            edges_to_remove = list(node.edges)
            for edge in edges_to_remove:
                node1, node2 = edge.get_nodes()
                # Remove edge from both nodes
                if edge in node1.edges:
                    node1.edges.remove(edge)
                if edge in node2.edges:
                    node2.edges.remove(edge)
            
            # Remove node from graph
            graph.nodes.pop(node_index)
            # Update node data map
            if hasattr(graph, '_node_data_map') and node.node_id in graph._node_data_map:
                del graph._node_data_map[node.node_id]
            
            return True
            
        except Exception as e:
            print(f"Error removing node {node.node_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_restoration_trigger(self, current_val_acc: float, best_val_acc: float) -> bool:
        """
        Check if restoration should be triggered.
        
        Args:
            current_val_acc: Current validation accuracy
            best_val_acc: Best validation accuracy seen so far
            
        Returns:
            bool: True if restoration should be triggered
        """
        if not self.restoration_enabled():
            return False
        
        if len(self.removed_nodes_pool) == 0:
            return False
        
        # Check if performance dropped below threshold
        drop = best_val_acc - current_val_acc
        return drop > self.restoration_trigger_threshold
    
    def restore_nodes(self, graph, trainer, current_val_acc: float, best_val_acc: float) -> Tuple[List, Dict]:
        """
        Execute node restoration based on configured strategy.
        
        Args:
            graph: HyperGraph instance to restore nodes to
            trainer: Trainer instance (for I-value access if needed)
            current_val_acc: Current validation accuracy
            best_val_acc: Best validation accuracy
            
        Returns:
            Tuple of (restored_nodes_list, restoration_stats_dict)
        """
        if not self.restoration_enabled():
            return [], {}
        
        if not self.check_restoration_trigger(current_val_acc, best_val_acc):
            return [], {}
        
        if len(self.removed_nodes_pool) == 0:
            print("No nodes available for restoration")
            return [], {}
        
        restored_nodes = []
        stats = {
            'strategy': self.restoration_strategy,
            'nodes_restored': 0,
            'val_acc_drop': best_val_acc - current_val_acc
        }
        
        try:
            if self.restoration_strategy == "random_pool":
                restored_nodes = self._restore_random_pool(graph)
            elif self.restoration_strategy == "targeted":
                restored_nodes = self._restore_targeted(graph, trainer)
            elif self.restoration_strategy == "reversion":
                restored_nodes = self._restore_reversion(graph, trainer)
            else:
                print(f"Warning: Unknown restoration strategy: {self.restoration_strategy}")
                return [], {}
            
            stats['nodes_restored'] = len(restored_nodes)
            self.reduction_stats['total_restorations'] += 1
            self.reduction_stats['total_nodes_restored'] += len(restored_nodes)
            
            print(f"Restored {len(restored_nodes)} nodes using {self.restoration_strategy} strategy")
            
        except Exception as e:
            print(f"Error during node restoration: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
        
        return restored_nodes, stats
    
    def _restore_random_pool(self, graph) -> List:
        """Restore random selection from removed nodes pool."""
        if len(self.removed_nodes_pool) == 0:
            return []
        
        # Calculate number to restore
        num_to_restore = max(1, int(len(self.removed_nodes_pool) * self.restoration_percentage / 100.0))
        num_to_restore = min(num_to_restore, len(self.removed_nodes_pool))
        
        # Randomly select nodes to restore
        nodes_to_restore = _stream('reduction.restore').sample(self.removed_nodes_pool, num_to_restore)
        
        restored_nodes = []
        for node in nodes_to_restore:
            if self._add_node_to_graph(graph, node):
                restored_nodes.append(node)
                self.removed_nodes_pool.remove(node)
                # Remove I-value entry if exists
                if node.node_id in self.removed_nodes_ivalues:
                    del self.removed_nodes_ivalues[node.node_id]
        
        return restored_nodes
    
    def _restore_targeted(self, graph, trainer) -> List:
        """Restore nodes with I-values closest to average."""
        if len(self.removed_nodes_pool) == 0:
            return []
        
        # Calculate average I-value of removed nodes
        ivalues = []
        for node in self.removed_nodes_pool:
            if node.node_id in self.removed_nodes_ivalues:
                ivalues.append(self.removed_nodes_ivalues[node.node_id])
        
        if len(ivalues) == 0:
            # Fallback to random if no I-values available
            print("Warning: No I-values available for targeted restoration, falling back to random")
            return self._restore_random_pool(graph)
        
        avg_ivalue = np.mean(ivalues)
        
        # Find nodes closest to average
        node_distances = []
        for node in self.removed_nodes_pool:
            if node.node_id in self.removed_nodes_ivalues:
                distance = abs(self.removed_nodes_ivalues[node.node_id] - avg_ivalue)
                node_distances.append((node, distance))
        
        # Sort by distance to average
        node_distances.sort(key=lambda x: x[1])
        
        # Calculate number to restore
        num_to_restore = max(1, int(len(self.removed_nodes_pool) * self.restoration_percentage / 100.0))
        num_to_restore = min(num_to_restore, len(node_distances))
        
        # Restore closest nodes
        restored_nodes = []
        for node, _ in node_distances[:num_to_restore]:
            if self._add_node_to_graph(graph, node):
                restored_nodes.append(node)
                self.removed_nodes_pool.remove(node)
                if node.node_id in self.removed_nodes_ivalues:
                    del self.removed_nodes_ivalues[node.node_id]
        
        return restored_nodes
    
    def _restore_reversion(self, graph, trainer) -> List:
        """
        Restore nodes from previous epoch (reversion strategy).
        Note: This should be called at the start of an epoch after reduction.
        """
        # Get previous epoch's removed nodes
        current_epoch = max(self.epoch_removal_history.keys()) if self.epoch_removal_history else -1
        previous_epoch = current_epoch - 1
        
        if previous_epoch not in self.epoch_removal_history:
            print(f"No removal history for epoch {previous_epoch}, cannot revert")
            return []
        
        nodes_to_restore = self.epoch_removal_history[previous_epoch].copy()
        
        restored_nodes = []
        for node in nodes_to_restore:
            # Check if node is still in pool (might have been restored already)
            if node in self.removed_nodes_pool:
                if self._add_node_to_graph(graph, node):
                    restored_nodes.append(node)
                    self.removed_nodes_pool.remove(node)
                    if node.node_id in self.removed_nodes_ivalues:
                        del self.removed_nodes_ivalues[node.node_id]
        
        return restored_nodes
    
    def _add_node_to_graph(self, graph, node) -> bool:
        """
        Add a node back to the graph.
        
        Args:
            graph: HyperGraph instance
            node: Node to add
            
        Returns:
            bool: True if node was successfully added
        """
        try:
            # Check if node already exists
            if hasattr(graph, '_node_data_map') and node.node_id in graph._node_data_map:
                print(f"Warning: Node {node.node_id} already exists in graph")
                return False
            
            # Add node to graph
            graph.add_node(node)
            
            return True
            
        except Exception as e:
            print(f"Error adding node {node.node_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def store_epoch_state(self, epoch: int, removed_nodes: List):
        """
        Store epoch state for reversion strategy.
        
        Args:
            epoch: Epoch number
            removed_nodes: List of nodes removed in this epoch
        """
        if epoch not in self.epoch_removal_history:
            self.epoch_removal_history[epoch] = []
        self.epoch_removal_history[epoch].extend(removed_nodes)
    
    def get_removed_nodes(self) -> List:
        """Get current pool of removed nodes."""
        return self.removed_nodes_pool.copy()
    
    def get_stats(self) -> Dict:
        """Get reduction/restoration statistics."""
        return {
            'reduction_stats': self.reduction_stats.copy(),
            'removed_nodes_count': len(self.removed_nodes_pool),
            'epochs_with_reductions': len(self.epoch_removal_history)
        }
