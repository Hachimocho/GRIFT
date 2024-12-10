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
        self.model = ActiveModel(output_classes=1, classification_strategy='binary')  
        self.model.model.cuda()
        self.loss = nn.BCEWithLogitsLoss()
        
        # Add weight decay for regularization
        self.optim = torch.optim.AdamW(
            self.model.model.parameters(), 
            lr=lr,
            weight_decay=0.001,  
            amsgrad=amsgrad
        )
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Common transforms for both training and validation
        self.common_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Additional augmentation transforms for training only
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2)  # Help prevent overfitting
        ])
        
        self.current_mode = "train"

    def transform(self, img):
        """Apply appropriate transforms based on current mode"""
        try:
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms based on mode
            if self.current_mode == "train":
                return self.train_transforms(img)
            else:
                return self.common_transforms(img)
        except Exception as e:
            print(f"Transform error in CNNModel: {str(e)}")
            # If transform fails, try basic resize and normalize
            img = cv2.resize(img, (255, 255))
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            return img

    def __call__(self, x):
        """Forward pass through the model"""
        return self.model.model(x)

    def train(self):
        """Set model to training mode"""
        self.current_mode = "train"
        self.model.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.current_mode = "eval" 
        self.model.model.eval()

    def process_node_data(self, data, labels, mode):
        batch = [self.transform(subdata.load_data()) for subdata in data]
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
            self.scheduler.step(loss)
            
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

    def save(self):
        self.save_checkpoint()
        
    def load(self):
        self.load_checkpoint()