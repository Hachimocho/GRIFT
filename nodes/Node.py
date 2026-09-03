class Node():
    """
    Each node class must have a set of tags which matches what data types and/or datasets it can be used with.
    Invalid tags might cause bad things, so don't do that.
    """ 
    tags = ["any"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, node_id, split, data, edges, label):
        self.node_id = node_id
        self.split = split
        self.data = data
        self.edges = edges
        self.label = label
        
    def match(self, other):
        if isinstance(other, Node):
            return True
        else:
            return False
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        return self.data
    
    def set_data(self, data):
        self.data = data
        
    #: Memoised `get_adjacent_nodes` result, as `(edges_list, length, neighbours)`, or None.
    #:
    #: Class-level default so an instance restored by pickle -- whose `__dict__` never sees
    #: `__init__` -- still reads a valid "no cache" rather than raising AttributeError.
    _adjacency_cache = None

    def get_adjacent_nodes(self):
        """Neighbours of this node, one entry per incident edge.

        Memoised, because this was the single most expensive thing in training: a
        full-scale i-value profile recorded **1,344,952 calls for 1,504 trained nodes**
        (~894 per node, 30.5 s of a 40.9 s epoch). The walk asks for a node's neighbours
        once per step and `get_degree` asks again per candidate, so an O(degree) rebuild
        ran ~9 times per traversal step and dominated the run while the GPU idled at 6%.

        The cache is validated, not merely invalidated, which is what makes it safe
        against code that mutates `edges` in place without telling this object:

        * a reassignment (`node.edges = []`, `canonicalize_edge_order`) changes the list's
          *identity*;
        * an append (`add_edge`) or a removal (`GraphReductionManager`,
          `HyperGraph.remove_nodes`) changes its *length*.

        Both are O(1) to check, so every mutation path in the codebase is covered without
        the cache having to be told. The one case identity and length cannot see is an
        edge being re-pointed at a different node while both lists keep their size, so
        `Edge.set_node1`/`set_node2`/`set_nodes` invalidate the affected nodes explicitly.

        The returned list is the cached object itself and must be treated as read-only --
        every caller today either takes `len()` of it or iterates it.
        """
        edges = self.edges
        cached = self._adjacency_cache
        if cached is not None and cached[0] is edges and cached[1] == len(edges):
            return cached[2]

        adjacent_nodes = []
        for edge in edges:
            for node in edge.get_nodes():
                if node != self:
                    adjacent_nodes.append(node)
        self._adjacency_cache = (edges, len(edges), adjacent_nodes)
        return adjacent_nodes

    def invalidate_adjacency_cache(self):
        """Drop the memoised neighbour list. Cheap and always safe to call."""
        self._adjacency_cache = None

    def __getstate__(self):
        """Pickle without the cache: it is derived state, and it holds references to
        every neighbour, which would bloat the node and graph caches for no gain."""
        state = self.__dict__.copy()
        state.pop('_adjacency_cache', None)
        return state

    def get_neighbors(self):
        return self.get_adjacent_nodes()

    def get_degree(self):
        return len(self.get_adjacent_nodes())
        
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        else:
            return False
        
    def __hash__(self):
        return hash(self.node_id)
    
    def get_split(self):
        return self.split
    
    def set_split(self, split):
        self.split = split
        
    def get_label(self):
        return self.label

    def add_edge(self, edge):
        self.edges.append(edge)
