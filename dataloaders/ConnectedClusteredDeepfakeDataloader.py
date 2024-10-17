import random
import tqdm
from itertools import combinations
from tqdm import tqdm
import time
from dataloaders.Dataloader import Dataloader
from nodes.Node import Node
from graphs.HyperGraph import HyperGraph

class ConnectedClusteredDeepfakeDataloader(Dataloader):
    tags = ["deepfakes"]
    hyperparameters = {
        "parameters": {
            "buffer_connect_chance": {"distribution": "uniform", "min": .001, "max": 1}
        }
    }
    
    def __init__(self, datasets, edge_class, buffer_connect_chance):
        super().__init__(datasets, edge_class)
        self.buffer_connect_chance = buffer_connect_chance

    def load(self):
        # TODO: Add check for disconnected clusters, set local file roots, add support for other datasets
        start = time.time()
        node_list = []
        
        # Load datasets
        for dataset in self.datasets:
            dataset_node_list = dataset.load()
        
            print("# of nodes:")
            print(len(dataset_node_list))
            
            matched = 0
            matched_set = []
            unmatched_set = list(range(len(dataset_node_list)))
            
            # Iterate over all pairs of nodes
            for i, j in tqdm(combinations(range(len(dataset_node_list)), 2), total= sum(1 for _ in combinations(range(len(dataset_node_list)), 2)), desc="Building graph..."):
                if dataset_node_list[i].match(dataset_node_list[j]):
                    matched += 1
                    edge = self.edge_class(dataset_node_list[i], dataset_node_list[j])
                    dataset_node_list[i].add_edge(edge)
                    dataset_node_list[j].add_edge(edge)
                    # Update arrays of matched and unmatched nodes
                    if i in unmatched_set:
                        unmatched_set.remove(i)
                        matched_set.append(i)
                    if j in unmatched_set:
                        unmatched_set.remove(j)
                        matched_set.append(j)
                        
            # Add semi-random edges to disconnected nodes so every node is reachable
            for i in tqdm(range(len(dataset_node_list)), desc="Fixing disconnected nodes..."):
                if i in unmatched_set:
                    j = random.choice(matched_set)
                    edge = self.edge_class(dataset_node_list[i], dataset_node_list[j])
                    dataset_node_list[i].add_edge(edge)
                    dataset_node_list[j].add_edge(edge)
                    unmatched_set.remove(i)
                    matched_set.append(i)
                    if j in unmatched_set:
                        unmatched_set.remove(j)
                        matched_set.append(j)
                    matched += 1
                    
            node_list.extend(dataset_node_list)
            
        buffer_node = Node("train", None, [], None)
        for node in node_list:
            if random.random() < self.buffer_connect_chance:
                edge = self.edge_class(buffer_node, node)
                buffer_node.add_edge(edge)
                node.add_edge(edge)
        node_list.append(buffer_node)
        graph = HyperGraph(node_list)
        print("Graph generation finished in " + str(time.time() - start) + "s.")
        
        return graph