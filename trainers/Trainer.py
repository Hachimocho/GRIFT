import tqdm
from tqdm import tqdm
import time
import wandb
from datasets import *
from dataloaders import *
from managers import *

class Trainer():
    """
    Base class for pointer/agent based traversal and training on Hypergraphs.
    """
    tags = ["none"]
    hyperparameters: dict | None = None
    def __init__(self, graphmanager, train_traversal, test_traversal, models):
        self.graphmanager = graphmanager
        self.models = models
        self.train_traversal = train_traversal
        self.test_traversal = test_traversal
        self.epochs = 15
        
    def run(self):
        print("Running trainer.")
        t = time.time()
        best_acc = 0
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            avg_train_acc, train_loss = self.train()
            avg_val_acc, val_loss = self.val()
            self.graphmanager.update_graph()
            wandb.log({"epoch": epoch, "train_acc": avg_train_acc, "val_acc": avg_val_acc})
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                self.model.save_checkpoint()
            else:
                self.model.load_checkpoint()
        wandb.log({"best_acc": best_acc})
        wandb.log({"time": time.time() - t})
        
    def test_run(self):
        print("Test run!")
        t = time.time()
        best_acc = 0
        for epoch in tqdm(range(self.num_epochs), desc="Number of epochs run"):
            avg_train_acc, train_loss = self.train()
            avg_val_acc, val_loss = self.val()
            self.graphmanager.update_graph()
            wandb.log({"epoch": epoch, "train_acc": avg_train_acc, "val_acc": avg_val_acc})
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                self.model.save_checkpoint()
            else:
                self.model.load_checkpoint()
        avg_test_acc = self.test()
        wandb.log({"test_acc": avg_test_acc})
        wandb.log({"time": time.time() - t})

    def process_node_data(self):
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        raise NotImplementedError("Overwrite this!")
        
    def val(self):
        raise NotImplementedError("Overwrite this!")
    
    def test(self):
        raise NotImplementedError("Overwrite this!")