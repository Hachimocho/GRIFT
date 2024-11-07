import torch
import torch.nn as nn
import cv2
from models.Model import Model
from torchvision import transforms
import importlib

class CNNModel(Model):
    tags = ["cnn", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "model_name": {"values": ["effnetdf", "resnestdf", "mesonetdf", "squeezenetdf", "vistransformdf", "swintransformdf"]},
            "lr": {"distribution": "uniform", "min": 0.0001, "max": 0.001},
            "amsgrad": {"values": [True, False]}
        }
    }
    def __init__(self, save_path, model_name, lr, amsgrad):
        super().__init__(save_path)
        ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
        self.model = ActiveModel(output_classes=2, classification_strategy='binary')
        self.model.model.cuda()
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.model.model.parameters(), lr, amsgrad=amsgrad)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Resize((255, 255)) # Resize to 255x255

        ])
    def process_node_data(self, data, labels, mode):
        batch = [self.transform(cv2.cvtColor(subdata.load_data(), cv2.COLOR_BGR2RGB)) for subdata in data]
        y = labels

        # Train on input data
        if mode == "train":
            for model in self.models:
                model.train()
            y_hat = self.model(torch.stack(batch).cuda())
            y = torch.tensor(y).unsqueeze(1).cuda()
            loss = self.loss(y_hat, y.float())
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            
            # update and log
            # self.train_acc.update(y_hat, y)
            # self.train_f1.update(y_hat, y)
            # self.train_auroc.update(y_hat, y)
            # self.log_dict({
            #     "train_loss": loss, "train_acc": self.train_acc,
            #     "train_f1": self.train_f1, "train_auroc": self.train_auroc
            # }, on_epoch=True, on_step=False)  # sync_dist=True on multigpu
            #print(loss)
            
        # Perform validation
        elif mode == "val":
            self.model.eval()
            y_hat = self.model(torch.stack(batch).cuda())
            y = torch.tensor(y).unsqueeze(1).cuda()
            loss = self.loss(y_hat, y.float())
        
        # Run testing 
        elif mode == "test":
            self.model.eval()
            y_hat = self.model(torch.stack(batch).cuda())
            y = torch.tensor(y).unsqueeze(1).cuda()
            loss = self.loss(y_hat, y.float())
            
            
        # Should never occur due to checks in traverse_graph code.
        else:
            raise ValueError("Invalid mode, this should not occur!")
        
        self.stored_loss.append(loss.detach())
        acc = self.accuracy(y_hat, y)
        self.stored_accuracy.append(acc.detach())
            
        self.stored_mode = mode
        
    def save_checkpoint(self):
        torch.save(self.model.model.state_dict(), self.save_path)
        
    def load_checkpoint(self):
        self.model.model.load_state_dict(torch.load(self.save_path))
        
    def train(self):
        self.model.model.train()
        
    def val(self):
        self.model.model.eval()
        
    def test(self):
        self.model.model.eval()