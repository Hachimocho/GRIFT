from collections import deque
import random

def ensure_graph_connected(nodes, edge_class):
    """
    Main function:
    - Finds connected components
    - Identifies the largest component
    - Connects all smaller components to the main one
    - Returns updated nodes
    """
    connected_components = find_connected_components(nodes)

    if(len(connected_components) == 1):
        return nodes
    
    largest_component = max(connected_components, key=len)
    for component in connected_components:
        if component is not largest_component:

            #if we want an edge from each node in the smaller component to a random node in the largest component
            # for node in component:
            #     node.add_edge(edge_class(node, random.choice(largest_component)))

            #optimised version: (connected only one edge per node in the smaller component to a random node in the largest component)
            nodeDisconnected = random.choice(list(component))
            nodeConnected = random.choice(list(largest_component))
            edge = edge_class(nodeDisconnected, nodeConnected)
            nodeDisconnected.add_edge(edge)
            nodeConnected.add_edge(edge)
    return nodes


def find_connected_components(nodes):
    """
    Runs full BFS/DFS to find ALL connected components.
    Returns: list of components, each component is a list/set of nodes.
    """
    connected_components = []
    visited = set()
    for node in nodes:
        if node not in visited:
            component = bfs(node)
            connected_components.append(component)
            visited.update(component)  #add all nodes in the component to the visited set
    return connected_components

def bfs(start_node):
    """
    Runs BFS from a given node.
    Returns: list of nodes visited by BFS.
    """
    visited = set([start_node])
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in current_node.get_adjacent_nodes():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
   
