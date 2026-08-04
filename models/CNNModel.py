import torch
import torch.nn as nn
import cv2
from models.Model import Model
from torchvision import transforms
import importlib
import os

from models.uncertainty import (
    BatchEnsembleBinaryHead,
    BinaryEvidentialHead,
    EvidentialBinaryClassificationLoss,
    PredictionBundle,
    SNGPBinaryHead,
    compute_batch_graph_uncertainty,
    mc_dropout_predict,
)

class CNNModel(Model):
    tags = ["cnn", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "model_name": {"values": ["effnetdf", "resnestdf", "mesonetdf", "squeezenetdf", "vistransformdf", "swintransformdf"]},
            "lr": {"distribution": "uniform", "min": 0.0001, "max": 0.001},
            "amsgrad": {"values": [True, False]}
        }
    }
    def __init__(
        self,
        save_path,
        model_name,
        lr,
        amsgrad,
        device,
        uncertainty_head='none',
        mc_dropout_samples=0,
        batchensemble_members=4,
        sngp_hidden_dim=256,
        sngp_rff_dim=256,
        uncertainty_dropout_rate=0.2,
        graph_uncertainty_methods=None,
        graph_degree_penalty_weight=1.0,
        uncertainty_train_frequency=10,
    ):
        super().__init__(save_path)
        self.device = device  # Store the device
        ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
        self.model = ActiveModel(
            pretrained=True,  # Use pretrained weights
            finetune=True,    # Enable proper fine-tuning
            output_classes=1,
            classification_strategy='binary'
        )  
        self.model.model.to(self.device) # Move model to the specified device
        self.loss = nn.BCEWithLogitsLoss()
        self.uncertainty_head_type = uncertainty_head
        self.mc_dropout_samples = max(0, int(mc_dropout_samples))
        self.batchensemble_members = max(1, int(batchensemble_members))
        self.sngp_hidden_dim = max(16, int(sngp_hidden_dim))
        self.sngp_rff_dim = max(16, int(sngp_rff_dim))
        self.uncertainty_dropout_rate = float(uncertainty_dropout_rate)
        self.graph_uncertainty_methods = [
            method.strip() for method in (graph_uncertainty_methods or []) if method.strip()
        ]
        self.graph_degree_penalty_weight = float(graph_degree_penalty_weight)
        self.uncertainty_train_frequency = max(1, int(uncertainty_train_frequency))
        self.output_head = None
        self._last_penultimate_features = None
        self._supports_external_uncertainty_head = False
        self.dropout_controller = nn.Module()
        self.dropout_controller.add_module("backbone", self.model.model)
        self.evidential_loss = None

        self.final_linear_path, final_linear = self._find_last_linear(self.model.model)
        if final_linear is not None and self.uncertainty_head_type == 'none':
            final_linear.register_forward_pre_hook(self._capture_penultimate_features_hook)

        if self.uncertainty_head_type != 'none':
            if final_linear is None:
                raise ValueError(
                    f"Could not locate a final linear layer for uncertainty head '{self.uncertainty_head_type}'."
                )
            self._replace_module(self.model.model, self.final_linear_path, nn.Identity())
            self.output_head = self._build_uncertainty_head(final_linear.in_features)
            self.output_head.to(self.device)
            self.dropout_controller.add_module("uncertainty_head", self.output_head)
            self._supports_external_uncertainty_head = True
            if self.uncertainty_head_type == 'evidential':
                self.evidential_loss = EvidentialBinaryClassificationLoss(annealing_steps=1000)

        # Add weight decay for regularization
        self.optim = torch.optim.AdamW(
            self._parameters_for_optimization(),
            lr=lr,
            weight_decay=1e-5,  
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

    def _parameters_for_optimization(self):
        params = list(self.model.model.parameters())
        if self.output_head is not None:
            params.extend(list(self.output_head.parameters()))
        return params

    def _find_last_linear(self, module):
        last_name = None
        last_module = None
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear):
                last_name = name
                last_module = child
        return last_name, last_module

    def _capture_penultimate_features_hook(self, module, inputs):
        features = inputs[0]
        if isinstance(features, tuple):
            features = features[0]
        if features is not None and hasattr(features, "dim") and features.dim() > 2:
            features = torch.flatten(features, 1)
        self._last_penultimate_features = features

    def _replace_module(self, root_module, module_path, new_module):
        path_parts = module_path.split('.')
        parent = root_module
        for part in path_parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

        leaf = path_parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_module
        else:
            setattr(parent, leaf, new_module)

    def _build_uncertainty_head(self, in_features):
        if self.uncertainty_head_type == 'evidential':
            return BinaryEvidentialHead(
                in_features=in_features,
                hidden_features=self.sngp_hidden_dim,
                dropout=self.uncertainty_dropout_rate,
            )
        if self.uncertainty_head_type == 'batchensemble':
            return BatchEnsembleBinaryHead(
                in_features=in_features,
                ensemble_size=self.batchensemble_members,
                hidden_features=self.sngp_hidden_dim,
                dropout=self.uncertainty_dropout_rate,
            )
        if self.uncertainty_head_type == 'sngp':
            return SNGPBinaryHead(
                in_features=in_features,
                hidden_features=self.sngp_hidden_dim,
                rff_features=self.sngp_rff_dim,
                dropout=self.uncertainty_dropout_rate,
            )
        raise ValueError(f"Unsupported uncertainty head: {self.uncertainty_head_type}")

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
        bundle = self.forward_with_uncertainty(x)
        return bundle.logits

    def _build_prediction_bundle(self, raw_output, features=None):
        if isinstance(raw_output, PredictionBundle):
            return raw_output.with_predictions()

        if isinstance(raw_output, dict):
            logits = raw_output["logits"]
            probabilities = raw_output.get("probabilities")
            if probabilities is None:
                probabilities = torch.sigmoid(logits)
            bundle = PredictionBundle(
                logits=logits,
                probabilities=probabilities,
                features=features if features is not None else raw_output.get("features"),
                uncertainty=raw_output.get("uncertainty", {}),
                evidence=raw_output.get("evidence"),
                alpha=raw_output.get("alpha"),
                member_logits=raw_output.get("member_logits"),
                gp_variance=raw_output.get("gp_variance"),
            )
            return bundle.with_predictions()

        logits = raw_output
        probabilities = torch.sigmoid(logits)
        return PredictionBundle(
            logits=logits,
            probabilities=probabilities,
            features=features,
            uncertainty={},
        ).with_predictions()

    def _forward_prediction_bundle(self, x, nodes=None, update_precision=False):
        self._last_penultimate_features = None
        backbone_output = self.model(x)
        if isinstance(backbone_output, tuple):
            backbone_output = backbone_output[0]

        if self.output_head is not None:
            features = backbone_output
            if features.dim() > 2:
                features = torch.flatten(features, 1)
            raw_output = self.output_head(features) if self.uncertainty_head_type != 'sngp' else self.output_head(features, update_precision=update_precision)
            bundle = self._build_prediction_bundle(raw_output, features=features)
        else:
            features = self._last_penultimate_features
            if features is not None and hasattr(features, "dim") and features.dim() > 2:
                features = torch.flatten(features, 1)
            bundle = self._build_prediction_bundle(backbone_output, features=features)

        if nodes and self.graph_uncertainty_methods:
            graph_uncertainty = compute_batch_graph_uncertainty(
                nodes,
                self.graph_uncertainty_methods,
                penalty_weight=self.graph_degree_penalty_weight,
            )
            bundle.uncertainty.update(graph_uncertainty)

        return bundle.with_predictions()

    def forward_with_uncertainty(self, x, nodes=None, update_precision=False, use_mc_dropout=False):
        if use_mc_dropout and self.mc_dropout_samples > 1:
            return mc_dropout_predict(
                self.dropout_controller,
                lambda: self._forward_prediction_bundle(x, nodes=nodes, update_precision=update_precision),
                self.mc_dropout_samples,
            )

        return self._forward_prediction_bundle(x, nodes=nodes, update_precision=update_precision)

    def compute_loss(self, bundle_or_logits, labels, base_criterion=None):
        if isinstance(bundle_or_logits, PredictionBundle):
            bundle = bundle_or_logits
        else:
            bundle = self._build_prediction_bundle(bundle_or_logits)

        if self.uncertainty_head_type == 'evidential':
            return self.evidential_loss(bundle, labels)

        criterion = base_criterion or self.loss
        if self.uncertainty_head_type == 'batchensemble' and bundle.member_logits is not None:
            member_logits = bundle.member_logits.view(-1, 1)
            expanded_labels = labels.repeat(1, bundle.member_logits.size(1)).view(-1, 1)
            return criterion(member_logits, expanded_labels)

        return criterion(bundle.logits, labels)

    def summarize_uncertainty(self, bundle):
        summary = {}
        for name, value in bundle.uncertainty.items():
            tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
            summary[name] = float(tensor_value.detach().float().mean().item())
        return summary

    def train(self):
        """Set model to training mode"""
        self.current_mode = "train"
        self.model.model.train()
        if self.output_head is not None:
            self.output_head.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.current_mode = "eval" 
        self.model.model.eval()
        if self.output_head is not None:
            self.output_head.eval()

    def process_node_data(self, data, labels, mode):
        batch = [self.transform(subdata.load_data()) for subdata in data]
        y = labels

        # Train on input data
        if mode == "train":
            for model in self.models:
                model.train()
            y_hat = self.model(torch.stack(batch).to(self.device))
            y = torch.tensor(y).unsqueeze(1).to(self.device)
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
            y_hat = self.model(torch.stack(batch).to(self.device))
            y = torch.tensor(y).unsqueeze(1).to(self.device)
            loss = self.loss(y_hat, y.float())
        
        # Run testing 
        elif mode == "test":
            self.model.eval()
            y_hat = self.model(torch.stack(batch).to(self.device))
            y = torch.tensor(y).unsqueeze(1).to(self.device)
            loss = self.loss(y_hat, y.float())
            
            
        # Should never occur due to checks in traverse_graph code.
        else:
            raise ValueError("Invalid mode, this should not occur!")
        
        self.stored_loss.append(loss.detach())
        acc = self.accuracy(y_hat, y)
        self.stored_accuracy.append(acc.detach())
            
        self.stored_mode = mode
        
    def save_checkpoint(self, filepath):
        """Saves the model and optimizer state dictionaries to a file."""
        checkpoint = {
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'uncertainty_head_type': self.uncertainty_head_type,
        }
        if self.output_head is not None:
            checkpoint['uncertainty_head_state_dict'] = self.output_head.state_dict()
        torch.save(checkpoint, filepath)
        # print(f"CNNModel checkpoint saved to {filepath}") # Optional: for debugging

    def load_checkpoint(self, filepath):
        """Loads the model and optimizer state dictionaries from a file."""
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found at {filepath}. Skipping load.")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device) # Load to the correct device
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        if self.output_head is not None and 'uncertainty_head_state_dict' in checkpoint:
            self.output_head.load_state_dict(checkpoint['uncertainty_head_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.model.to(self.device) # Ensure model is on the correct device after loading
        if self.output_head is not None:
            self.output_head.to(self.device)
        # print(f"CNNModel checkpoint loaded from {filepath}") # Optional: for debugging

    def save(self):
        self.save_checkpoint(self.save_path)
        
    def load(self):
        self.load_checkpoint(self.save_path)
