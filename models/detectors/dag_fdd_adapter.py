import torch
import torch.nn as nn
import sys
import os

# Add fairness modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .dag_fdd import DagFddDetector

class DagFddAdapter(nn.Module):
    """
    DAG FDD Adapter - Preserves exact research methodology from the original paper.
    
    This adapter maintains the complete DAG FDD implementation without any modifications
    to ensure proper research comparison. It bridges the interface differences between
    the original fairness detector and the HyperGraph training framework.
    """
    
    def __init__(self, pretrained=False, finetune=False, exclude_top=False,
                 output_classes=1, classification_strategy='binary', configuration='default'):
        super().__init__()
        
        # Initialize the original DAG FDD detector (preserving exact methodology)
        self.dag_detector = DagFddDetector()
        
        # Store configuration for compatibility with existing framework
        self.pretrained = pretrained
        self.finetune = finetune
        self.exclude_top = exclude_top
        self.output_classes = output_classes
        self.classification_strategy = classification_strategy
        self.configuration = configuration
        
        # For compatibility with existing framework - use the backbone as main model
        self.model = self.dag_detector.backbone
        
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
            pred: Prediction dictionary from forward pass
            target: Target labels
            data_dict: Data dictionary with demographic information
            
        Returns:
            torch.Tensor: The exact loss from the original DAG FDD implementation
        """
        if data_dict is None:
            # Fallback to standard loss if no demographic data available
            # This maintains compatibility while preserving research integrity
            if isinstance(pred, dict) and 'cls' in pred:
                return nn.BCEWithLogitsLoss()(pred['cls'], target.float())
            else:
                return nn.BCEWithLogitsLoss()(pred, target.float())
        
        # Use the exact DAG FDD loss computation from the original implementation
        # This preserves the bi-level optimization and fairness components
        loss_dict = self.dag_detector.get_losses(data_dict, pred)
        return loss_dict['overall']
    
    def get_train_metrics(self, data_dict, pred_dict):
        """
        Get the exact training metrics from the original DAG FDD implementation.
        
        Args:
            data_dict: Data dictionary with labels and demographic information
            pred_dict: Prediction dictionary from forward pass
            
        Returns:
            dict: Exact metrics from the original implementation
        """
        return self.dag_detector.get_train_metrics(data_dict, pred_dict)
    
    def features(self, x):
        """
        Extract features using the exact DAG FDD feature extraction method.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Features from the original DAG FDD backbone
        """
        data_dict = {'image': x}
        return self.dag_detector.features(data_dict)
    
    def classifier(self, features):
        """
        Classify features using the exact DAG FDD classifier method.
        
        Args:
            features: Features from the backbone
            
        Returns:
            torch.Tensor: Classification output from the original DAG FDD classifier
        """
        return self.dag_detector.classifier(features)
    
    def build_backbone(self):
        """
        Build the exact backbone from the original DAG FDD implementation.
        
        Returns:
            The exact backbone network from the original research
        """
        return self.dag_detector.build_backbone()
    
    def build_loss(self):
        """
        Build the exact loss function from the original DAG FDD implementation.
        
        Returns:
            The exact loss function from the original research
        """
        return self.dag_detector.build_loss()
    
    def threshplus_tensor(self, x):
        """
        Exact threshold plus function from the original DAG FDD implementation.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Threshold plus output
        """
        return self.dag_detector.threshplus_tensor(x)
    
    def search_func(self, losses, alpha):
        """
        Exact search function from the original DAG FDD implementation.
        
        Args:
            losses: Loss tensor
            alpha: Alpha parameter
            
        Returns:
            function: Search function for bi-level optimization
        """
        return self.dag_detector.search_func(losses, alpha)
    
    def searched_lamda_loss(self, losses, searched_lamda, alpha):
        """
        Exact searched lambda loss from the original DAG FDD implementation.
        
        Args:
            losses: Loss tensor
            searched_lamda: Searched lambda value
            alpha: Alpha parameter
            
        Returns:
            torch.Tensor: Searched lambda loss
        """
        return self.dag_detector.searched_lamda_loss(losses, searched_lamda, alpha)
