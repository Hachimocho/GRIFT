from graphs.HyperGraph import HyperGraph

class Dataloader():
    """
    Takes a bunch of datasets and loads them into a HyperGraph.
    """
    tags = ["none"]
    hyperparameters: dict | None = None
    
    def __init__(self, datasets, edge_class):
        self.datasets = datasets
        self.edge_class = edge_class
    
    def load(self) -> HyperGraph:
        raise NotImplementedError("Overwrite this!")
    
