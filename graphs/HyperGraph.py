import random
import matplotlib.pyplot as plt
from nodes.Node import Node
from edges.Edge import Edge

"""
Planning/TODO/brainstorming:

Multigraphs need to be level-agnostic: a multigraph functions the same at level 4 as level 400 (except for level 0)
Level 0 multigraphs are NetworkX Graphs that store data directly.

Level 1+ multigraphs could be NetworkX Graphs that store other multigraphs (which are NetworkX Graphs themselves)
or they could be containers which store other multigraphs and NetworkX Graphs (miss out on extra functions, but would those work anyway?)

Any multigraph can be initialized with pointers that move according to some specified method

Pointers can traverse between multigraphs if user-approved



----- How to make L0 networks functional with existing NetworkX infrastructure ---

???

----------------------------------------------------------------------------------


Do leaves need to be different from non-leaves?
One MultiGraph class for all levels, just leave in and genericize the traversal and training functions?
Try and report back
Leaf is just MultiGraph with no subgraphs?
That means each MultiGraph can hold data and/or graphs
2 node types - Graph and Data
That's just two graph types again lol

Solution - Use NetworkX graphs all the way, store edges normally but use string-based edge attributes.
Handle everything else pointer-side: traversal and learning will access edge attributes and treat them differently, even though NetworkX
    sees them the same way
    
Generic dataloader formatted as follows:

0. Get base directory
1. Find all files in base directory except edges.json
2. Load all files as nodes into top-level graph
3. Connect nodes with edges in edges.json if present
4. Find all directories in base directory with same name (-extensions?) as a file in base directory
5. Repeat 1-4 with each new directory, loading result as graph attatched to matching node
6. Once done, you have a single graph where each node is a file + (optional) a graph, and that graph has the same property

Generic trainer/traverser formatted as follows:

0. Get graph created by dataloader
1. Initialize n_0 pointers with random position
2. For each pointer, if it is in a node with a graph, initialize n_1 pointers with random position within that graph (or n_default if not specified)
3. Repeat 2 until graphs all the way down are filled
4. Do training on that node (undefined in base/abstract implementation, needs to be overwritten)
5. Move to adjacent node
6. Repeat 2-5 for num_steps
7. Repeat 1-6 (or 2-6?) with validation
8. Repeat 1-7 (or 2-7?) for num_epochs
9. Repeat 1-6 with testing
(NOTE: Only works for low-n hypergraphs. Will need level-moving pointers or some other solution for high-n graphs due to overhead.)

HyperGraph

init: Get 
basic utility functions: add and remove data, get all or specific data
"""

class HyperGraph():
    """
    This is an agent-based multigraph dataset class.
    It provides several basic functions for management and traversal of data graphs.
    Normally we would use an abstract class here, but Hypergraphs are so fundamental that we can really only use one.
    """
    tags = ["any"]
    hyperparameters: dict | None = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, nodes: list):
        """
        Initialize a HyperGraph object.

        Args:
            nodes (list): The nodes that make up the graph.
        """
        # Store nodes and create a lookup map for quick access by node ID
        self.nodes = nodes
        self._node_data_map = {node.node_id: node for node in self.nodes} # Use node_id as key
        
    def __len__(self):
        """
        Get the number of nodes in the hypergraph.

        Returns:
            int: The number of nodes in the hypergraph.
        """
        return len(self.nodes)
    
    def get_node(self, index):
        """
        Get a node from the hypergraph.

        Args:
            index (int): The index of the node to retrieve.

        Returns:
            Node: The node at the given index.

        Raises:
            Exception: If the index is out of range.
        """
        if index > (len(self.nodes) + 1):
            raise Exception("Invalid index for get_node.")
        return self.nodes[index]
    
    def get_nodes(self):
        """
        Get all nodes in the hypergraph.

        Returns:
            list: A list of all nodes in the hypergraph.
        """
        return self.nodes
    
    def set_node(self, index, node):
        """
        Set a node in the hypergraph.

        Args:
            index (int): The index of the node to set.
            node (Node): The node to set at the given index.

        Raises:
            Exception: If the index is out of range.
        """
        if index > (len(self.nodes) + 1):
            raise Exception("Invalid index for set_node.")
        self.nodes[index] = node
        
    def remove_node(self, index):
        """
        Remove a node from the hypergraph.

        Args:
            index (int): The index of the node to remove.

        Raises:
            Exception: If the index is out of range.
        """
        if index > (len(self.nodes) + 1):
            raise Exception("Invalid index for remove_node.")
        self.nodes.pop(index)
        
    def add_node(self, node):
        """
        Add a node to the hypergraph.

        Args:
            node (Node): The node to add to the hypergraph.
        """
        if node.node_id not in self._node_data_map: # Check using node_id
            self.nodes.append(node)
            self._node_data_map[node.node_id] = node # Add using node_id
        else:
            # Handle duplicate node add attempt if necessary, e.g., log warning
            print(f"Warning: Node with ID {node.node_id} already exists.")
            
    def get_random_node(self):
        """
        Get a random node from the hypergraph.

        Returns:
            Node: A random node from the hypergraph.
        """
        return random.choice(self.nodes)
    
    def k_hop_subgraph(self, node, k, duplicates=False):
        """
        Get the k-hop subgraph of a node in the hypergraph.

        Args:
            node (Node): The node to get the k-hop subgraph of.
            k (int): The number of hops to go.
            duplicates (bool, optional): Whether to include duplicate nodes. Defaults to False.

        Returns:
            HyperGraph: The k-hop subgraph of the node.
        """
        k_hop_nodes = set()
        current_hop = [node]
        for i in range(k):
            next_hop = set()
            for n in current_hop:
                for neighbor in n.get_neighbors():
                    if neighbor not in k_hop_nodes:
                        next_hop.add(neighbor)
            k_hop_nodes.update(next_hop)
            current_hop = next_hop
        if not duplicates:
            k_hop_nodes.remove(node)
        return HyperGraph(list(k_hop_nodes))
        
    def k_hop_list(self, node, k, duplicates=False):
        """
        Get the k-hop ordered list of a node in the hypergraph, where the first entry is the node itself, 
        the second entry is the node's neighbors, and so on.

        Args:
            node (Node): The node to get the k-hop list of.
            k (int): The number of hops to go.
            duplicates (bool, optional): Whether to include duplicate nodes. Defaults to False.

        Returns:
            list: The k-hop list of the node.
        """
        k_hop_list = [node]
        current_hop = [node]
        for i in range(k):
            next_hop = set()
            for n in current_hop:
                for neighbor in n.get_neighbors():
                    if neighbor not in next_hop and (duplicates or neighbor not in k_hop_list):
                        next_hop.add(neighbor)
            k_hop_list.extend(list(next_hop))
            current_hop = list(next_hop)
        return k_hop_list

    def get_edge_list(self):
        """
        Extracts a list of unique edges represented as tuples of node identifiers.
        Ensures edges are stored consistently, e.g., (min_id, max_id).

        Returns:
            list: A list of tuples, where each tuple is (node1_id, node2_id).
        """
        edge_set = set()
        for node in self.nodes:
            # Access edges directly if Node stores them, or use get_adjacent_nodes/get_edges
            # Assuming node.edges exists and contains Edge objects
            if hasattr(node, 'edges'):
                for edge in node.edges:
                    node1, node2 = edge.get_nodes()
                    id1 = node1.node_id # Use node_id
                    id2 = node2.node_id # Use node_id
                    # Ensure consistent ordering and add to set to handle duplicates
                    edge_tuple = tuple(sorted((id1, id2)))
                    edge_set.add(edge_tuple)
            else:
                # Fallback or alternative if edges aren't directly accessible
                # This part might need adjustment based on actual Node/Edge implementation
                pass 
        return list(edge_set)

    def add_edges_from_list(self, edge_list):
        """
        Adds edges to the graph based on a list of node identifier pairs.

        Args:
            edge_list (list): A list of tuples, where each tuple is (node1_id, node2_id).
        """
        if not self._node_data_map:
             # Rebuild map if it wasn't created during init or is empty
             self._node_data_map = {node.node_id: node for node in self.nodes} # Use node_id
             
        edges_added_count = 0
        for id1, id2 in edge_list: # Assume these are node_ids
            node1 = self._node_data_map.get(id1)
            node2 = self._node_data_map.get(id2)

            if node1 and node2:
                # Create a new Edge object. Assuming Edge takes node1, node2, and optionally data/weight.
                # Using None for edge data 'x' as it's not stored in the simple list.
                new_edge = Edge(node1, node2, x=None) 
                
                # Add the edge to both nodes. Assumes Node.add_edge exists.
                if hasattr(node1, 'add_edge') and hasattr(node2, 'add_edge'):
                    node1.add_edge(new_edge)
                    node2.add_edge(new_edge)
                    edges_added_count += 1
                else:
                    print(f"Warning: Nodes {id1} or {id2} missing 'add_edge' method.")
            else:
                print(f"Warning: Could not find nodes for edge ({id1}, {id2}). Skipping.")
        print(f"Added {edges_added_count} edges from the list.")

    def save_display(self, path):
        """
        Save and display the hypergraph.

        Args:
            path (str): The path to save the hypergraph to.
        """
        pos = {}
        colors = {}
        node_type_set = set()
        for node in self.nodes:
            node_type_set.add(node.__class__)
        color_index = 0
        for node_type in node_type_set:
            colors[node_type] = plt.cm.tab20(color_index)
            color_index += 1
        for node in self.nodes:
            if node not in pos:
                pos[node] = (random.random() * 2 - 1, random.random() * 2 - 1)
            for neighbor in node.get_neighbors():
                if neighbor not in pos:
                    pos[neighbor] = (pos[node][0] + (random.random() - 0.5) / 10, pos[node][1] + (random.random() - 0.5) / 10)
        fig, ax = plt.subplots()
        for node in self.nodes:
            ax.scatter(*pos[node], c=[colors[node.__class__]], s=100)
        for node in self.nodes:
            for neighbor in node.get_neighbors():
                ax.plot([pos[node][0], pos[neighbor][0]], [pos[node][1], pos[neighbor][1]], c='black', alpha=0.1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)