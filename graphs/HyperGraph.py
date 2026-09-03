import random
import matplotlib.pyplot as plt
from nodes.Node import Node
from edges.Edge import Edge
import collections
import csv
import gzip
import os
import shutil

# Optional: Louvain/Label Propagation imports (fail gracefully if not installed)
try:
    import networkx as nx
    import community as community_louvain
    _louvain_available = True
except ImportError:
    _louvain_available = False

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
        self.subclusters = None  # node_id -> subcluster_id
        
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
        
    def remove_nodes(self, nodes):
        """Remove many nodes and their incident edges in one pass. Returns the count.

        Two reasons this exists rather than looping `remove_node`:

        * **`remove_node` takes an index and leaves `_node_data_map` stale.** A node removed
          through it stays in the id map, so `add_node` later refuses to re-add it as a
          duplicate -- which silently breaks any restore-after-pruning scheme.
        * **Cost.** Callers only have node objects, so removing k of them one at a time
          means k linear scans for their indices: O(N*k). At a million nodes, withdrawing 2%
          would be 2e10 comparisons. This is a single O(N + E_touched) pass.
        """
        doomed_ids = {node.node_id for node in nodes}
        if not doomed_ids:
            return 0

        # Detach incident edges from the endpoints that survive, so the remaining graph holds
        # no edge pointing at a removed node.
        for node in nodes:
            for edge in list(getattr(node, 'edges', []) or []):
                first, second = edge.get_nodes()
                for endpoint in (first, second):
                    edges = getattr(endpoint, 'edges', None)
                    if edges is not None and edge in edges:
                        edges.remove(edge)

        before = len(self.nodes)
        self.nodes = [node for node in self.nodes if node.node_id not in doomed_ids]
        for node_id in doomed_ids:
            self._node_data_map.pop(node_id, None)
        return before - len(self.nodes)

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
            
    def get_random_node(self, rng=None):
        """
        Get a random node from the hypergraph.

        Args:
            rng: Optional ``random.Random`` to draw from. Callers with their own
                stream (e.g. traversals) should pass it, so their node choices do
                not depend on unrelated consumption of the global RNG.

        Returns:
            Node: A random node from the hypergraph.
        """
        if rng is None:
            from test_helpers.determinism import component_rng
            rng = component_rng("graph.random_node")
        return rng.choice(self.nodes)
    
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
            # discard, not remove: for k=1 the accumulated set holds only the seed's
            # neighbors, so the seed is absent and remove() raised KeyError -- which
            # made k_hop_subgraph(node, 1) unconditionally broken.
            k_hop_nodes.discard(node)
        # Sorted, because Node.__hash__ hashes a string node_id, so set iteration
        # order is PYTHONHASHSEED-dependent and varies between processes.
        return HyperGraph(sorted(k_hop_nodes, key=lambda item: str(item.node_id)))
        
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
            ordered_hop = sorted(next_hop, key=lambda item: str(item.node_id))
            k_hop_list.extend(ordered_hop)
            current_hop = ordered_hop
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
        # Sorted, not list(set): these are tuples of *string* node ids, so set
        # iteration order depends on PYTHONHASHSEED. This list is written to the
        # pickle graph cache, so an unsorted order meant a cache written under one
        # hash seed replayed its edges in a different order than one written under
        # another -- and edge order determines traversal tie-breaks.
        return sorted(edge_set)

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
        edges_skipped_count = 0
        
        # Pre-calculate missing nodes for better performance
        if len(edge_list) > 100:  # Only do pre-check for large edge lists
            print("Pre-validating edge compatibility...")
            all_edge_node_ids = set()
            for id1, id2 in edge_list:
                all_edge_node_ids.add(id1)
                all_edge_node_ids.add(id2)
            
            available_node_ids = set(self._node_data_map.keys())
            missing_node_ids = all_edge_node_ids - available_node_ids
            
            if missing_node_ids:
                print(f"Pre-validation: {len(missing_node_ids)} unique node IDs from edges are missing from graph")
                print(f"This will result in skipping edges that reference these missing nodes")
        
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
                    edges_skipped_count += 1
            else:
                edges_skipped_count += 1
                # Only print individual warnings for small edge lists
                if len(edge_list) <= 100:
                    print(f"Warning: Could not find nodes for edge ({id1}, {id2}). Skipping.")
        
        self.canonicalize_edge_order()

        print(f"Edge loading complete: {edges_added_count} edges added, {edges_skipped_count} edges skipped")
        if edges_skipped_count > 0:
            success_rate = (edges_added_count / (edges_added_count + edges_skipped_count)) * 100
            print(f"Edge loading success rate: {success_rate:.1f}%")

    def export_edges_csv(self, path, delimiter=",", include_header=True):
        """
        Stream edges to a CSV (optionally gzipped) without materializing all edges in memory.

        Args:
            path (str): Output file path. If it ends with '.gz', the file is gzip-compressed.
            delimiter (str): CSV delimiter, default ','.
            include_header (bool): Whether to write a header row.
        """
        if not self.nodes:
            # Nothing to write
            return 0

        # Choose opener based on extension
        open_fn = gzip.open if path.endswith('.gz') else open
        mode = 'wt' if path.endswith('.gz') else 'w'

        # Pre-flight: log destination directory and free space
        try:
            out_dir = os.path.dirname(path) or '.'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            usage = shutil.disk_usage(out_dir)
            print(f"[Cache] Exporting edges to {path} (free disk: {usage.free/1e9:.2f} GB)")
        except Exception as e:
            print(f"[Cache][Warning] Unable to check disk usage for {path}: {e}")

        # Write streaming CSV of unique edges (source,target) by ordering node IDs
        num_written = 0
        progress_interval = 1_000_000
        try:
            with open_fn(path, mode, newline='') as f:
                writer = csv.writer(f, delimiter=delimiter)
                if include_header:
                    writer.writerow(['source', 'target'])

                # Iterate nodes and their edges in canonical order, so the exported
                # file is byte-stable: export -> load -> export is a fixpoint, and two
                # runs that built the same graph produce identical cache files.
                #
                # Each undirected edge is emitted exactly once, from its
                # lexicographically smaller endpoint. The previous condition compared
                # the edge's *stored* (node1, node2) order instead -- but every
                # undirected edge is visited twice (once per endpoint) while
                # `edge.get_nodes()` returns the same orientation both times. So an
                # edge stored ascending was written TWICE and one stored descending was
                # written ZERO times, silently dropping roughly half the edges
                # depending only on the order each Edge object happened to be
                # constructed in. A cache-loaded graph was therefore materially
                # sparser than the graph that produced it, and the reported edge count
                # looked plausible because it counted the duplicates.
                for node in sorted(self.nodes, key=lambda item: str(item.node_id)):
                    node_id = getattr(node, 'node_id', None)
                    if node_id is None or not hasattr(node, 'edges'):
                        continue
                    for edge in sorted(
                        getattr(node, 'edges', []),
                        key=lambda item, _node=node: self._edge_sort_key(_node, item),
                    ):
                        try:
                            n1, n2 = edge.get_nodes()
                            id1 = getattr(n1, 'node_id', None)
                            id2 = getattr(n2, 'node_id', None)
                            if id1 is None or id2 is None:
                                continue
                            peer_id = id2 if id1 == node_id else id1
                            # Emit only from the smaller endpoint, in canonical
                            # (min, max) order. Self-loops are emitted once.
                            if str(node_id) <= str(peer_id):
                                writer.writerow([node_id, peer_id])
                                num_written += 1
                                if num_written % progress_interval == 0:
                                    print(f"[Cache] Export progress: {num_written} edges written...")
                        except Exception as row_e:
                            print(f"[Cache][Warning] Failed writing edge row: {row_e}")
                            continue
        except Exception as io_e:
            print(f"[Cache][Error] Failed exporting edges to {path}: {io_e}")
            return num_written

        try:
            size_bytes = os.path.getsize(path)
            print(f"[Cache] Export complete: {num_written} edges -> {size_bytes/1e6:.1f} MB at {path}")
        except Exception:
            print(f"[Cache] Export complete: {num_written} edges -> size unknown (path: {path})")

        return num_written

    def load_edges_from_csv(self, path, delimiter=",", has_header=True):
        """
        Stream edges from a CSV (optionally gzipped) and add them to this graph without
        building a large in-memory list.

        Args:
            path (str): Input CSV path. If it ends with '.gz', file is treated as gzip-compressed.
            delimiter (str): CSV delimiter, default ','.
            has_header (bool): Whether the first row is a header.

        Returns:
            int: Number of edges successfully added.
        """
        if not self._node_data_map:
            self._node_data_map = {node.node_id: node for node in self.nodes}

        open_fn = gzip.open if path.endswith('.gz') else open
        mode = 'rt' if path.endswith('.gz') else 'r'

        print(f"[Cache] Loading edges from {path} ...")
        edges_added = 0
        progress_interval = 1_000_000
        line_no = 0
        try:
            with open_fn(path, mode, newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                # Skip header when present
                if has_header:
                    try:
                        next(reader)
                        line_no += 1
                    except StopIteration:
                        print("[Cache][Warning] Edge CSV appears empty.")
                        return 0

                for row in reader:
                    line_no += 1
                    try:
                        if not row or len(row) < 2:
                            continue
                        id1, id2 = row[0], row[1]
                        node1 = self._node_data_map.get(id1)
                        node2 = self._node_data_map.get(id2)
                        if not node1 or not node2:
                            # Skip edges whose nodes aren't present in this graph
                            continue
                        # Create and attach a simple Edge with no extra data
                        new_edge = Edge(node1, node2, x=None)
                        if hasattr(node1, 'add_edge') and hasattr(node2, 'add_edge'):
                            node1.add_edge(new_edge)
                            node2.add_edge(new_edge)
                            edges_added += 1
                            if edges_added % progress_interval == 0:
                                print(f"[Cache] Load progress: {edges_added} edges added...")
                    except Exception as row_e:
                        print(f"[Cache][Warning] Error parsing row {line_no}: {row_e}")
                        continue
        except Exception as io_e:
            print(f"[Cache][Error] Failed loading edges from {path} at line {line_no}: {io_e}")
            return edges_added

        # Canonicalize so a cache-loaded graph induces the same traversals as a
        # freshly built one at the same seed.
        self.canonicalize_edge_order()

        print(f"[Cache] Load complete: {edges_added} edges from {path}")
        return edges_added

    def canonicalize_edge_order(self, sort_nodes=False):
        """Put every node's adjacency into a canonical order.

        Without this, a graph loaded from the edge cache and a graph built fresh
        disagree on adjacency *order* even though they agree on the edge *set*:
        ``export_edges_csv`` writes by iterating nodes then their edges, while
        ``load_edges_from_csv`` appends in file order. Traversals do
        ``random.choice(adjacent)`` and argmax-over-neighbors with ties broken by
        list position, so the same seed produced different traversals depending on
        whether the graph came from cache -- one of the least obvious
        irreproducibility sources in the pipeline.

        Call after any build, load, or bulk edge insertion.
        """
        for node in self.nodes:
            edges = getattr(node, 'edges', None)
            if not edges:
                continue
            node.edges = sorted(edges, key=lambda edge: self._edge_sort_key(node, edge))
        if sort_nodes:
            self.nodes = sorted(self.nodes, key=lambda node: str(node.node_id))
            self._node_data_map = {node.node_id: node for node in self.nodes}
        return self

    @staticmethod
    def _edge_sort_key(node, edge):
        """Sort key for one of ``node``'s edges: the peer's id, then the pair."""
        try:
            first, second = edge.get_nodes()
        except Exception:
            return ("", "")
        node_id = str(getattr(node, 'node_id', ''))
        first_id = str(getattr(first, 'node_id', ''))
        second_id = str(getattr(second, 'node_id', ''))
        peer_id = second_id if first_id == node_id else first_id
        # Include the ordered pair so parallel edges between the same two nodes
        # still have a stable relative order.
        return (peer_id, first_id, second_id)

    def num_edges(self):
        return len(self.get_edge_list())

    def assign_louvain_subclusters(self):
        """
        Assign Louvain subclusters to nodes using networkx + python-louvain.
        Stores mapping in self.subclusters (node_id -> subcluster_id).
        """
        if not _louvain_available:
            # NOTE: python-louvain ("community") is not installed in the current
            # training environment, so this is currently a silent no-op and every
            # *_subclustered graph type runs on the `subclusters is None` fallbacks.
            print(
                "Louvain/community-louvain not available. Skipping subcluster assignment "
                "-- subcluster-based traversals will fall back to their no-subcluster "
                "paths. Install python-louvain to enable them."
            )
            self.subclusters = None
            return None

        # Build the networkx graph deterministically: Louvain iterates G.nodes() in
        # insertion order, so unsorted insertion makes the partition depend on node
        # ordering rather than only on the graph.
        G = nx.Graph()
        G.add_nodes_from(sorted(str(node.node_id) for node in self.nodes))
        edge_pairs = set()
        for node in self.nodes:
            for edge in getattr(node, 'edges', []) or []:
                first, second = edge.get_nodes()
                edge_pairs.add(tuple(sorted((str(first.node_id), str(second.node_id)))))
        G.add_edges_from(sorted(edge_pairs))

        # Seeded: best_partition() otherwise falls back to the global numpy RNG, so
        # the partition depended on how much randomness had been consumed earlier.
        try:
            from test_helpers.determinism import is_configured, seed_for
            random_state = seed_for("graph.louvain") if is_configured() else 0
        except ImportError:
            random_state = 0
        partition = community_louvain.best_partition(G, random_state=random_state)
        self.subclusters = partition  # {node_id: subcluster_id}
        # Optionally, store subcluster on node
        for node in self.nodes:
            node.subcluster_id = partition.get(node.node_id, None)
        return partition

    def get_nodes_by_subcluster(self, subcluster_id):
        """Return list of nodes in a given subcluster."""
        if self.subclusters is None:
            return []
        return [node for node in self.nodes if self.subclusters.get(node.node_id) == subcluster_id]

    def save_with_subclusters(self, path):
        """Save edge list and subclusters to a file (dill)."""
        import dill
        cache_data = {
            'edges': self.get_edge_list(),
            'subclusters': self.subclusters
        }
        with open(path, 'wb') as f:
            dill.dump(cache_data, f)

    @staticmethod
    def load_with_subclusters(path, nodes):
        """Load edge list and subclusters from a file, assign to a new HyperGraph."""
        import dill
        with open(path, 'rb') as f:
            cache_data = dill.load(f)
        hg = HyperGraph(nodes)
        if 'edges' in cache_data:
            hg.add_edges_from_list(cache_data['edges'])
        if 'subclusters' in cache_data:
            hg.subclusters = cache_data['subclusters']
            for node in hg.nodes:
                node.subcluster_id = hg.subclusters.get(node.node_id, None)
        return hg

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

    def export_csv_with_subclusters(self, node_path, edge_path):
        """
        Export nodes and edges to CSV, including subcluster info for Cosmograph or similar tools.
        Args:
            node_path (str): Path to save the node CSV.
            edge_path (str): Path to save the edge CSV.
        """
        import csv
        # Export nodes
        with open(node_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'subcluster', 'label'])  # Add more columns as needed
            for node in self.nodes:
                subcluster = self.subclusters.get(node.node_id) if self.subclusters else None
                label = getattr(node, 'label', getattr(node, 'name', ''))
                writer.writerow([node.node_id, subcluster, label])
        # Export edges
        with open(edge_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])
            for node in self.nodes:
                if hasattr(node, 'edges'):
                    for edge in node.edges:
                        n1, n2 = edge.get_nodes()
                        # Only write each edge once
                        if n1.node_id < n2.node_id:
                            writer.writerow([n1.node_id, n2.node_id])