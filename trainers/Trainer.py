import tqdm
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import wandb
from torchvision import transforms
from torchmetrics import Accuracy, F1Score, AUROC
from datasets import *
from dataloaders import *
from managers import *

class Trainer():
    """
    Base class for pointer/agent based traversal and training on Hypergraphs.
    """
    tags = ["none"]
    hyperparameters: dict | None = None
    def __init__(self, graphmanager, train_traversal, test_traversal, models, num_steps=1000):
        self.graphmanager = graphmanager
        self.models = models
        self.optims = [torch.optim.Adam(model.model.parameters(), lr=0.001) for model in self.models]
        self.losses = [nn.BCEWithLogitsLoss() for model in self.models]
        self.train_traversal = train_traversal
        self.test_traversal = test_traversal
        self.epochs = 15
        self.batches = [[] for model in self.models]
        self.batch_size = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Resize((255, 255)) # Resize to 255x255

        ])
        self.train_acc = None
        self.train_acc_history = [[] for model in self.models]
        # self.train_f1 = F1Score()
        self.val_acc = None
        self.val_acc_history = [[] for model in self.models]
        # self.train_auroc = AUROC()
        self.test_acc = None
        self.test_acc_history = [[] for model in self.models]
        self.num_steps = num_steps
        # self.test_f1 = F1Score()
        # self.test_auroc = AUROC()
        # Pessimistic accuracy initialization allows for high starting I values for unseen nodes
        # Based on "optimism in the face of uncertainty" from standard RL
        self.stored_prediction_accuracy = [{node: [0] for node in self.graphmanager.get_graph().get_nodes()} for model in self.models]
        
    def run(self):
        print("Running trainer.")
        t = time.time()
        best_accs = [0 for model in self.models]
        for epoch in tqdm(range(self.epochs), desc="Number of epochs run"):
            for _ in range(self.num_steps):
                self.train()
            for _ in range(self.num_steps):
                self.val()
            self.graphmanager.update_graph()
            for i, model in enumerate(self.models):
                avg_train_acc = sum(self.train_acc_history[i]) / len(self.train_acc_history[i]) if len(self.train_acc_history[i]) > 0 else 0
                if avg_val_acc > best_accs[i]:
                    best_accs[i] = avg_val_acc
                    self.models[i].save_checkpoint()
                else:
                    self.models[i].load_checkpoint()
                avg_val_acc = sum(self.val_acc_history[i]) / len(self.val_acc_history[i]) if len(self.val_acc_history[i]) > 0 else 0
                wandb.log({"epoch": epoch, f"train_acc_model_{i}": avg_train_acc, f"val_acc_model_{i}": avg_val_acc})
        for i, acc in enumerate(best_accs):
            wandb.log({f"best_acc_model_{i}": acc})
        wandb.log({"time": time.time() - t})
        
    def test_run(self):
        print("Test run!")
        t = time.time()
        for _ in range(self.num_steps):
            self.test()
        for i, model in enumerate(self.models):
            avg_test_acc = sum(self.test_acc_history[i]) / len(self.test_acc_history[i]) if len(self.test_acc_history[i]) > 0 else 0
            wandb.log({f"test_acc_model_{i}": avg_test_acc})
        wandb.log({"time": time.time() - t})

    def process_node_data(self):
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        raise NotImplementedError("Overwrite this!")
        
    def val(self):
        raise NotImplementedError("Overwrite this!")
    
    def test(self):
        raise NotImplementedError("Overwrite this!")