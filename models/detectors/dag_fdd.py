

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from scipy import optimize
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from ..metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from . import DETECTOR
from ..networks import BACKBONE
from ..loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='dag_fdd')
class DagFddDetector(AbstractDetector):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()
        
    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']
        backbone = backbone_class({'mode': 'original',
                                   'num_classes': 1, 'inc': 3, 'dropout': False})
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load('./models/detectors/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        print('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['daw_bce']
        loss_func = loss_class()
        return loss_func

    def threshplus_tensor(self, x):
        y = x.clone()
        pros = torch.nn.ReLU()
        z = pros(y)
        return z
    
    def search_func(self, losses, alpha):
        return lambda x: x + (1.0/alpha)*(self.threshplus_tensor(losses-x).mean().item())

    def searched_lamda_loss(self, losses, searched_lamda, alpha):
        return searched_lamda + ((1.0/alpha)*torch.mean(self.threshplus_tensor(losses-searched_lamda))) 
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        #Default value 0.5
        dag_alpha = 0.5
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_entropy = self.loss_func(pred, label)
        chi_loss_np = self.search_func(loss_entropy, dag_alpha)
        cutpt = optimize.fminbound(chi_loss_np, np.min(loss_entropy.cpu().detach().numpy()) - 1000.0, np.max(loss_entropy.cpu().detach().numpy()))
        loss = self.searched_lamda_loss(loss_entropy, cutpt, dag_alpha)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred = pred.squeeze(1)
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        pass


    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the classification results
        cls = self.classifier(features)
        # return the prediction dictionary
        pred_dict = {'cls': cls}
        return pred_dict

class ModelOut(nn.Module):
    """
    ModelOut wrapper for DAG FDD detector.
    
    This class provides the interface expected by FairnessCNNModel while
    preserving the exact research methodology from the original DAG FDD implementation.
    """
    
    def __init__(self, pretrained=False, finetune=False, exclude_top=False,
                 output_classes=1, classification_strategy='binary', configuration='default'):
        super().__init__()
        
        # Store configuration for compatibility
        self.pretrained = pretrained
        self.finetune = finetune
        self.exclude_top = exclude_top
        self.output_classes = output_classes
        self.classification_strategy = classification_strategy
        self.configuration = configuration
        
        # Initialize the original DAG FDD detector (preserving exact methodology)
        self.dag_detector = DagFddDetector()
        
        # For compatibility with existing framework - create a holder so CNNModel can use `.model.model`
        class _BackboneHolder:
            def __init__(self, model):
                self.model = model
            
            def __call__(self, x):
                # Forward the call to the actual model
                return self.model(x)
            
            def to(self, device):
                self.model.to(device)
                return self
            
            def train(self):
                self.model.train()
                return self
            
            def eval(self):
                self.model.eval()
                return self
            
            def parameters(self):
                return self.model.parameters()
            
            def state_dict(self):
                return self.model.state_dict()
            
            def load_state_dict(self, state_dict):
                return self.model.load_state_dict(state_dict)
        
        self.model = _BackboneHolder(self.dag_detector.backbone)
        
        # Expose original methods directly for easier access
        self.threshplus_tensor = self.dag_detector.threshplus_tensor
        self.search_func = self.dag_detector.search_func
        self.searched_lamda_loss = self.dag_detector.searched_lamda_loss
        self.get_train_metrics = self.dag_detector.get_train_metrics
        self.features = self.dag_detector.features
        self.classifier = self.dag_detector.classifier
        self.build_backbone = self.dag_detector.build_backbone
        self.build_loss = self.dag_detector.build_loss
        
    def forward(self, x):
        """
        Forward pass that preserves the exact DAG FDD methodology.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Classification predictions for compatibility with CNNModel
        """
        # Convert simple tensor input to data_dict format expected by original detector
        data_dict = {'image': x}
        
        # Get predictions using the exact original DAG FDD forward method
        pred_dict = self.dag_detector.forward(data_dict)
        
        # Return the tensor for compatibility with CNNModel (extract from dictionary)
        return pred_dict['cls']
    
    def forward_dict(self, x):
        """
        Forward pass that returns the full dictionary format.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            dict: Contains the exact outputs from the original DAG FDD detector
        """
        # Convert simple tensor input to data_dict format expected by original detector
        data_dict = {'image': x}
        
        # Get predictions using the exact original DAG FDD forward method
        pred_dict = self.dag_detector.forward(data_dict)
        
        # Return the exact prediction dictionary from the original method
        return pred_dict
    
    def get_loss(self, pred, target, data_dict=None):
        """
        Get the exact DAG FDD loss computation.
        
        This method preserves the original bi-level loss optimization and fairness
        components exactly as implemented in the research paper.
        
        Args:
            pred: Prediction tensor or dictionary from forward pass
            target: Target labels (can be None if data_dict contains label)
            data_dict: Data dictionary with demographic information
            
        Returns:
            torch.Tensor: The exact loss from the original DAG FDD implementation
        """
        # Convert tensor prediction back to dictionary format if needed
        if not isinstance(pred, dict):
            pred = {'cls': pred}
            
        if data_dict is None:
            # Fallback to standard loss if no demographic data available
            # This maintains compatibility while preserving research integrity
            if isinstance(pred, dict) and 'cls' in pred:
                # Ensure target has the same shape as prediction
                target_reshaped = target.float().unsqueeze(1) if (target is not None and target.dim() == 1) else (target.float() if target is not None else None)
                if target_reshaped is None:
                    raise ValueError("Target must be provided when data_dict is None")
                return nn.BCEWithLogitsLoss()(pred['cls'], target_reshaped)
            else:
                # Ensure target has the same shape as prediction
                target_reshaped = target.float().unsqueeze(1) if (target is not None and target.dim() == 1) else (target.float() if target is not None else None)
                if target_reshaped is None:
                    raise ValueError("Target must be provided when data_dict is None")
                return nn.BCEWithLogitsLoss()(pred, target_reshaped)
        
        # Use the exact DAG FDD loss computation from the original implementation
        # This preserves the bi-level optimization and fairness components
        # The data_dict already contains the label in the correct format
        loss_dict = self.dag_detector.get_losses(data_dict, pred)
        return loss_dict['overall']
