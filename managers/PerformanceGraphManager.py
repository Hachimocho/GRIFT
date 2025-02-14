import torch
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from managers.GraphManager import GraphManager
from edges.Edge import Edge
from models.DQNModel import DQNModel

class PerformanceGraphManager(GraphManager):
    """
    A graph manager that dynamically rewires the graph based on model performance.
    Uses I-value predictions to identify weak and strong groups, and adjusts the graph
    structure accordingly.
    """
    tags = ["performance", "i-value"]
    hyperparameters = {
        "parameters": {
            "rewire_threshold": {"distribution": "uniform", "min": 0.6, "max": 0.9},
            "edge_removal_threshold": {"distribution": "uniform", "min": 0.1, "max": 0.3},
            "max_edges_per_node": {"distribution": "uniform", "min": 5, "max": 15},
            "update_interval": {"distribution": "uniform", "min": 100, "max": 500}
        }
    }

    def __init__(self, graph, rewire_threshold=0.8, edge_removal_threshold=0.2, 
                 max_edges_per_node=10, update_interval=200):
        """
        Initialize the performance-based graph manager.

        Args:
            graph (HyperGraph): The graph to manage
            rewire_threshold (float): Threshold for I-value above which nodes are considered weak
            edge_removal_threshold (float): Threshold for I-value below which nodes are considered strong
            max_edges_per_node (int): Maximum number of edges per node
            update_interval (int): Number of steps between graph updates
        """
        super().__init__(graph)
        self.rewire_threshold = rewire_threshold
        self.edge_removal_threshold = edge_removal_threshold
        self.max_edges_per_node = max_edges_per_node
        self.update_interval = update_interval
        
        # Initialize tracking
        self.steps = 0
        self.node_performance = defaultdict(list)
        self.i_value_predictor = None
        
    def set_i_value_predictor(self, predictor: DQNModel):
        """Set the I-value predictor model"""
        self.i_value_predictor = predictor

    def track_performance(self, node, i_value: float):
        """
        Track the performance (I-value) for a given node.
        
        Args:
            node: The node to track
            i_value (float): The predicted I-value for the node
        """
        self.node_performance[node].append(i_value)
        # Keep only recent history
        if len(self.node_performance[node]) > 100:
            self.node_performance[node] = self.node_performance[node][-100:]

    def get_node_avg_performance(self, node) -> float:
        """Get the average I-value (performance) for a node"""
        if not self.node_performance[node]:
            return 0.5  # Default middle value if no history
        return np.mean(self.node_performance[node])

    def identify_weak_nodes(self) -> List:
        """Identify nodes with consistently high I-values (poor performance)"""
        weak_nodes = []
        for node in self.graph.get_nodes():
            avg_perf = self.get_node_avg_performance(node)
            if avg_perf > self.rewire_threshold:
                weak_nodes.append(node)
        return weak_nodes

    def identify_strong_nodes(self) -> List:
        """Identify nodes with consistently low I-values (good performance)"""
        strong_nodes = []
        for node in self.graph.get_nodes():
            avg_perf = self.get_node_avg_performance(node)
            if avg_perf < self.edge_removal_threshold:
                strong_nodes.append(node)
        return strong_nodes

    def add_edges_to_weak_node(self, node, num_edges: int = 2):
        """
        Add edges to help a weak node by connecting it to strong nodes.
        
        Args:
            node: The weak node to add edges to
            num_edges (int): Number of new edges to add
        """
        strong_nodes = self.identify_strong_nodes()
        if not strong_nodes:
            return

        # Don't exceed max edges per node
        current_edges = len(node.edges)
        if current_edges >= self.max_edges_per_node:
            return

        # Add edges to random strong nodes
        num_edges = min(num_edges, self.max_edges_per_node - current_edges)
        selected_nodes = np.random.choice(strong_nodes, size=min(num_edges, len(strong_nodes)), replace=False)
        
        for strong_node in selected_nodes:
            # Create bidirectional edges
            edge = Edge(node, strong_node, None)
            node.edges.append(edge)
            strong_node.edges.append(edge)

    def remove_edges_from_strong_node(self, node, num_edges: int = 1):
        """
        Remove edges from a strong node to reduce computational overhead.
        
        Args:
            node: The strong node to remove edges from
            num_edges (int): Number of edges to remove
        """
        if len(node.edges) <= 1:  # Keep at least one edge
            return
            
        # Remove random edges
        edges_to_remove = np.random.choice(node.edges, size=min(num_edges, len(node.edges)-1), replace=False)
        for edge in edges_to_remove:
            # Remove edge from both nodes
            node1, node2 = edge.get_nodes()
            if edge in node1.edges:
                node1.edges.remove(edge)
            if edge in node2.edges:
                node2.edges.remove(edge)

    def update_graph(self):
        """
        Update the graph structure based on node performance.
        Called periodically during training.
        """
        self.steps += 1
        
        # Only update periodically
        if self.steps % self.update_interval != 0:
            return
            
        if self.i_value_predictor is None:
            return

        # Identify weak and strong nodes
        weak_nodes = self.identify_weak_nodes()
        strong_nodes = self.identify_strong_nodes()

        # Add edges to weak nodes
        for node in weak_nodes:
            self.add_edges_to_weak_node(node)

        # Remove edges from strong nodes
        for node in strong_nodes:
            self.remove_edges_from_strong_node(node)
