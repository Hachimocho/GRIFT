import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List
from tqdm.auto import tqdm
import random

def random_layout_with_progress(G, center=None, dim=2):
    """Generate a random layout with progress bar."""
    print("Generating random layout...")
    pos = {}
    # Use a larger space for node placement (0 to 10 instead of 0 to 1)
    with tqdm(total=len(G), desc="Positioning nodes") as pbar:
        for node in G.nodes():
            pos[node] = 10 * np.random.random(dim)
            pbar.update(1)
    return pos

def sample_large_graph(G, max_nodes=1000):
    """Sample a subgraph for visualization if the graph is too large."""
    if len(G) <= max_nodes:
        return G, None
    
    print(f"\nGraph is too large ({len(G)} nodes). Sampling {max_nodes} nodes for visualization...")
    
    # Select random seed nodes
    seed_nodes = random.sample(list(G.nodes()), max_nodes // 2)
    
    # Add neighbors of seed nodes until we reach max_nodes
    nodes = set(seed_nodes)
    with tqdm(total=max_nodes - len(nodes), desc="Sampling nodes") as pbar:
        for node in seed_nodes:
            if len(nodes) >= max_nodes:
                break
            neighbors = set(G.neighbors(node)) - nodes
            nodes.update(list(neighbors)[:min(5, max_nodes - len(nodes))])
            pbar.update(min(5, max_nodes - len(nodes)))
    
    # Create node mapping for the sampled graph
    node_mapping = {old: new for new, old in enumerate(nodes)}
    
    # Create new graph with sampled nodes
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    
    # Add edges between sampled nodes
    with tqdm(total=len(nodes), desc="Building sampled graph") as pbar:
        for old_node in nodes:
            new_node = node_mapping[old_node]
            for neighbor in G.neighbors(old_node):
                if neighbor in node_mapping:
                    H.add_edge(new_node, node_mapping[neighbor])
            pbar.update(1)
    
    return H, node_mapping

def visualize_graph(graph, 
                   save_path: Optional[str] = None,
                   show: bool = True,
                   splits: Optional[Dict[str, List[int]]] = None,
                   attribute_labels: Optional[Dict[int, str]] = None,
                   title: str = "Graph Visualization",
                   figsize: tuple = (20, 16),  # Increased figure size
                   nx_graph: Optional[nx.Graph] = None,
                   max_nodes: int = 1000):
    """
    Visualize a hypergraph using NetworkX and Matplotlib.
    
    Args:
        graph: HyperGraph object to visualize
        save_path: Optional path to save the visualization. If None, saves to logs directory
        show: Whether to display the plot
        splits: Dictionary mapping split names to node indices (e.g., {'train': [0,1,2], 'val': [3,4]})
        attribute_labels: Dictionary mapping node indices to attribute labels
        title: Title for the visualization
        figsize: Figure size as (width, height)
        nx_graph: Pre-built NetworkX graph (optional, for better performance)
        max_nodes: Maximum number of nodes to visualize (will sample if exceeded)
    """
    print("\nPreparing graph visualization...")
    
    # Use provided NetworkX graph or create new one
    if nx_graph is None:
        print("Building NetworkX graph (this may take a while)...")
        G = nx.Graph()
        G.add_nodes_from(range(len(graph.nodes)))
        
        # Add edges with progress bar
        total_edges = sum(len(node.edges) for node in graph.nodes)
        edge_pbar = tqdm(total=total_edges, desc="Adding edges")
        
        # Track added edges to avoid duplicates
        added_edges = set()
        for node in graph.nodes:
            for edge in node.edges:
                n1, n2 = edge.get_nodes()
                n1_idx = graph.nodes.index(n1)
                n2_idx = graph.nodes.index(n2)
                
                # Add edge only if we haven't seen it before
                edge_key = tuple(sorted([n1_idx, n2_idx]))
                if edge_key not in added_edges:
                    G.add_edge(n1_idx, n2_idx)
                    added_edges.add(edge_key)
                edge_pbar.update(1)
        
        edge_pbar.close()
    else:
        print("Using pre-built NetworkX graph")
        G = nx_graph
    
    # Sample graph if it's too large
    G, node_mapping = sample_large_graph(G, max_nodes)
    
    # Set up colors for different splits
    split_colors = {
        'train': '#2ecc71',  # Green
        'val': '#e74c3c',    # Red
        'test': '#3498db',   # Blue
        'default': '#95a5a6'  # Gray
    }
    
    # Assign colors based on splits
    print("Assigning node colors and sizes...")
    node_colors = []
    if splits:
        split_map = {}
        for split_name, indices in splits.items():
            for idx in indices:
                if node_mapping:
                    if idx in node_mapping:
                        split_map[node_mapping[idx]] = split_name
                else:
                    split_map[idx] = split_name
        
        for i in range(len(G)):
            split = split_map.get(i, 'default')
            node_colors.append(split_colors[split])
    else:
        node_colors = [split_colors['default']] * len(G)
    
    # Calculate node sizes based on degree, with smaller base size and scale
    max_degree = max(G.degree(n) for n in G.nodes())
    if max_degree > 0:
        # Normalize node sizes based on degree
        node_sizes = [20 + 30 * (G.degree(n) / max_degree) for n in G.nodes()]
    else:
        node_sizes = [20] * len(G.nodes())
    
    # Create the plot
    plt.figure(figsize=figsize)
    if node_mapping:
        title += f" (Sampled {len(G)} nodes)"
    plt.title(title)
    
    # Use simple random layout for very large graphs
    print("Computing layout...")
    pos = random_layout_with_progress(G)
    
    print("Drawing graph elements...")
    # Draw the network with smaller nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)  # Thinner edges
    
    # Add node labels if attribute information is provided
    if attribute_labels:
        if node_mapping:
            # Update attribute labels for sampled graph
            labels = {node_mapping[i]: f"{i}\n{attr}" 
                     for i, attr in attribute_labels.items() 
                     if i in node_mapping}
        else:
            labels = {i: f"{i}\n{attr}" for i, attr in attribute_labels.items()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6)  # Smaller font
    else:
        if node_mapping:
            # Show original node indices for sampled graph
            reverse_mapping = {new: old for old, new in node_mapping.items()}
            labels = {new: str(old) for new, old in reverse_mapping.items()}
        else:
            labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6)  # Smaller font
    
    # Add legend for splits with smaller markers
    if splits:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=split,
                                    markersize=6)  # Smaller legend markers
                         for split, color in split_colors.items()
                         if split in splits or split == 'default']
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Save the visualization
    if save_path is None:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(logs_dir, f'graph_viz_{timestamp}.png')
    
    if save_path:
        print(f"Saving visualization to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("Visualization saved successfully!")
    
    if show:
        plt.show()
    else:
        plt.close()
