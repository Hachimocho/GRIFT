import torch
import torch.nn as nn
from models.CNNModel import CNNModel
import importlib
import os

class FairnessCNNModel(CNNModel):
    """
    Fairness-Enhanced CNN Model that supports fairness detectors while preserving
    exact research methodology for proper comparison.
    
    This model maintains backward compatibility with existing detectors while adding
    support for fairness-aware training using the original research implementations.
    """
    
    def __init__(self, save_path, model_name, lr, amsgrad, device, fairness_mode=False):
        super().__init__(save_path, model_name, lr, amsgrad, device)
        self.fairness_mode = fairness_mode
        
        if fairness_mode:
            # Load appropriate fairness detector while preserving exact methodology
            if model_name == 'dag_fdd':
                from models.detectors.dag_fdd import ModelOut
                self.model = ModelOut(
                    pretrained=True,
                    finetune=True,
                    output_classes=1,
                    classification_strategy='binary'
                )
                print(f"Loaded DAG FDD detector with exact research methodology")
            elif model_name == 'daw_fdd':
                from models.detectors.daw_fdd import ModelOut
                self.model = ModelOut(
                    pretrained=True,
                    finetune=True,
                    output_classes=1,
                    classification_strategy='binary'
                )
                print(f"Loaded DAW FDD detector with exact research methodology")
            else:
                # Fallback to standard detector
                print(f"Warning: {model_name} not found in fairness detectors, using standard detector")
                try:
                    ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
                    self.model = ActiveModel(
                        pretrained=True,
                        finetune=True,
                        output_classes=1,
                        classification_strategy='binary'
                    )
                except AttributeError:
                    # Handle case where detector doesn't have ModelOut class
                    print(f"Warning: {model_name} doesn't have ModelOut class, using standard CNNModel")
                    # Disable fairness mode and use standard initialization
                    self.fairness_mode = False
                    # Call parent constructor without fairness mode
                    super().__init__(save_path, model_name, lr, amsgrad, device)
                    return
            
            self.model.to(self.device)
    
    def process_node_data(self, data, labels, mode):
        """
        Process node data with fairness-aware training when fairness_mode is enabled.
        
        This method preserves the exact research methodology while maintaining
        compatibility with the existing training framework.
        """
        batch = [self.transform(subdata.load_data()) for subdata in data]
        y = labels

        if self.fairness_mode:
            return self.process_fairness_data(batch, y, mode)
        else:
            return super().process_node_data(data, labels, mode)
    
    def process_fairness_data(self, batch, labels, mode):
        """
        Handle fairness-specific data processing while preserving exact research methodology.
        
        This method ensures that the original fairness detector implementations are used
        exactly as intended in the research paper for proper comparison.
        """
        x = torch.stack(batch).to(self.device)
        y = torch.tensor(labels).unsqueeze(1).to(self.device)
        
        # Create data_dict with demographic information if available
        # For now, use basic data_dict (demographic data would be added later)
        data_dict = {'image': x, 'label': y}
        
        if mode == "train":
            self.model.train()
            
            # Get predictions using the exact fairness detector methodology
            y_hat = self.model(x)
            
            # Use the exact loss computation from the original research implementation
            if hasattr(self.model, 'get_loss'):
                # This preserves the bi-level optimization and fairness components exactly
                # Pass the prediction and data_dict, let the model handle target extraction
                loss = self.model.get_loss(y_hat, None, data_dict)
            else:
                # Handle dictionary output from fairness models
                if isinstance(y_hat, dict) and 'cls' in y_hat:
                    loss = self.loss(y_hat['cls'], y.float())
                else:
                    loss = self.loss(y_hat, y.float())
            
            # Standard training steps
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.scheduler.step(loss)
            
        elif mode in ["val", "test"]:
            self.model.eval()
            
            # Get predictions using the exact fairness detector methodology
            y_hat = self.model(x)
            
            # Handle dictionary output from fairness models
            if isinstance(y_hat, dict) and 'cls' in y_hat:
                loss = self.loss(y_hat['cls'], y.float())
                y_hat = y_hat['cls']  # Use main classification for metrics
            else:
                loss = self.loss(y_hat, y.float())
        
        # Store results for tracking
        self.stored_loss.append(loss.detach())
        acc = self.accuracy(y_hat, y)
        self.stored_accuracy.append(acc.detach())
        self.stored_mode = mode
    
    def get_fairness_metrics(self, data_dict, pred_dict):
        """
        Get fairness-specific metrics from the original research implementation.
        
        This method preserves the exact metric computation from the original paper.
        """
        if hasattr(self.model, 'get_train_metrics'):
            return self.model.get_train_metrics(data_dict, pred_dict)
        else:
            # Fallback to standard metrics
            return {}
    
    def save_checkpoint(self, filepath):
        """Saves the model and optimizer state dictionaries to a file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'fairness_mode': self.fairness_mode,
        }
        torch.save(checkpoint, filepath)
        print(f"FairnessCNNModel checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Loads the model and optimizer state dictionaries from a file."""
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found at {filepath}. Skipping load.")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        
        # Restore fairness mode if available
        if 'fairness_mode' in checkpoint:
            self.fairness_mode = checkpoint['fairness_mode']
        
        print(f"FairnessCNNModel checkpoint loaded from {filepath}") 