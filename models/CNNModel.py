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
    count_stochastic_dropout_sites,
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
        sngp_precision_policy='per-epoch',
        finetune=False,
    ):
        super().__init__(save_path)
        self.device = device  # Store the device
        self.model_name = model_name
        ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
        # `finetune=True` does NOT mean "fine-tune" in the detectors -- effnetdf and
        # swintransformdf read it as "freeze every parameter whose name lacks
        # 'classifier'/'head'", i.e. train the classifier only. This was hardcoded
        # True, so every run on those two architectures was a linear probe rather
        # than a fine-tuned detector, and a deep ensemble over them would have
        # measured head-initialization variance alone. Now it defaults to full
        # fine-tuning and is exposed via --finetune/--no-finetune.
        self.finetune = bool(finetune)
        self.model = ActiveModel(
            pretrained=True,  # Use pretrained weights
            finetune=self.finetune,
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
        self.sngp_precision_policy = sngp_precision_policy
        self.uncertainty_dropout_rate = float(uncertainty_dropout_rate)
        self.graph_uncertainty_methods = [
            method.strip() for method in (graph_uncertainty_methods or []) if method.strip()
        ]
        self.graph_degree_penalty_weight = float(graph_degree_penalty_weight)
        self.uncertainty_train_frequency = max(1, int(uncertainty_train_frequency))
        self.output_head = None
        self.graph_distance_standardizer = None
        self._warned_missing_standardizer = False
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
        # `verbose` was deprecated and then removed from ReduceLROnPlateau -- absent
        # on newer torch builds (2.6+), present but warning on others. It only ever
        # controlled a console print on LR drop, so dropping it changes no numerics.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.5,
            patience=2,
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

    def parameter_counts(self):
        """Total vs trainable parameter counts.

        Recorded in run manifests so a frozen backbone is visible in the results
        rather than something a reader has to infer. See the `finetune` note in
        ``__init__``.
        """
        params = self._parameters_for_optimization()
        return {
            "total": sum(param.numel() for param in params),
            "trainable": sum(param.numel() for param in params if param.requires_grad),
            "finetune": self.finetune,
            "backbone_frozen": not any(
                param.requires_grad for param in self.model.model.parameters()
            ),
        }

    def set_graph_distance_standardizer(self, standardizer):
        """Attach fitted graph-distance statistics.

        Fitted once on the *training* graph and reused for val/test/OOD -- fitting
        per split would renormalize a shifted distribution until it matched the
        training one, erasing the very signal being measured. Persisted in the
        checkpoint so scoring after a resume uses the same statistics.
        """
        self.graph_distance_standardizer = standardizer
        self._warned_missing_standardizer = False
        return standardizer

    def graph_uncertainty_ready(self):
        """Whether graph-distance methods can actually be scored right now."""
        return bool(self.graph_uncertainty_methods) and (
            self.graph_distance_standardizer is not None
        )

    def on_epoch_start(self, epoch, num_epochs=None):
        """Forward the epoch boundary to the uncertainty head, if it wants it.

        Discovered by ``hasattr`` in the training capabilities, so the seam stays
        duck-typed and any future model can opt in without changing call sites.
        SNGP uses this to apply its precision-reset policy.
        """
        if self.output_head is not None and hasattr(self.output_head, "on_epoch_start"):
            self.output_head.on_epoch_start(epoch, num_epochs=num_epochs)

    def on_epoch_end(self, epoch):
        if self.output_head is not None and hasattr(self.output_head, "on_epoch_end"):
            self.output_head.on_epoch_end(epoch)

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

    @staticmethod
    def _component_seed(component, fallback):
        """Sub-seed for a head's initialization.

        Falls back to a fixed constant when determinism has not been configured
        (e.g. a bare unit test constructing a model directly), so head init is
        always reproducible rather than depending on how much of the ambient torch
        RNG stream happened to be consumed first.
        """
        try:
            from test_helpers.determinism import is_configured, seed_for
            if is_configured():
                return seed_for(component)
        except ImportError:
            pass
        return fallback

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
                init_seed=self._component_seed("model.batchensemble_init", 0xBE0001),
            )
        if self.uncertainty_head_type == 'sngp':
            return SNGPBinaryHead(
                in_features=in_features,
                hidden_features=self.sngp_hidden_dim,
                rff_features=self.sngp_rff_dim,
                dropout=self.uncertainty_dropout_rate,
                rff_seed=self._component_seed("model.sngp_rff", 0x5A6D01),
                precision_policy=self.sngp_precision_policy,
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

    def _forward_prediction_bundle(self, x, nodes=None, update_precision=False,
                                   compute_variance=True):
        self._last_penultimate_features = None
        backbone_output = self.model(x)
        if isinstance(backbone_output, tuple):
            backbone_output = backbone_output[0]

        if self.output_head is not None:
            features = backbone_output
            if features.dim() > 2:
                features = torch.flatten(features, 1)
            if self.uncertainty_head_type == 'sngp':
                raw_output = self.output_head(
                    features,
                    update_precision=update_precision,
                    compute_variance=compute_variance,
                )
            else:
                raw_output = self.output_head(features)
            bundle = self._build_prediction_bundle(raw_output, features=features)
        else:
            features = self._last_penultimate_features
            if features is not None and hasattr(features, "dim") and features.dim() > 2:
                features = torch.flatten(features, 1)
            bundle = self._build_prediction_bundle(backbone_output, features=features)

        if nodes and self.graph_uncertainty_methods:
            if self.graph_distance_standardizer is not None:
                bundle.uncertainty.update(
                    compute_batch_graph_uncertainty(
                        nodes,
                        self.graph_uncertainty_methods,
                        penalty_weight=self.graph_degree_penalty_weight,
                        standardizer=self.graph_distance_standardizer,
                    )
                )
            elif not self._warned_missing_standardizer:
                # Warn once and skip rather than raising mid-run. Silently emitting
                # nothing would be worse: the benchmark would simply have no
                # graph-uncertainty column and nobody would notice.
                self._warned_missing_standardizer = True
                print(
                    "WARNING: graph uncertainty methods "
                    f"{self.graph_uncertainty_methods} were requested but no fitted "
                    "standardizer is attached, so no graph uncertainty will be recorded. "
                    "Call model.set_graph_distance_standardizer(...) with statistics "
                    "fitted on the training graph."
                )

        return bundle.with_predictions()

    def mc_dropout_available(self):
        """Whether MC dropout can produce a real signal on this model.

        False when no dropout module in the graph has ``p > 0``. That is not a
        hypothetical: ``vistransformdf`` (the CLI default) has 37 ``nn.Dropout``
        modules and every one is ``p=0.0``, because torchvision's
        ``VisionTransformer`` defaults ``dropout=0.0`` and the detector passes
        nothing. Sampling it yields identical passes and therefore *identically
        zero* variance -- a silently wrong measurement rather than an error.
        """
        return count_stochastic_dropout_sites(self.dropout_controller) > 0

    def forward_with_uncertainty(self, x, nodes=None, update_precision=False,
                                 use_mc_dropout=False, compute_variance=True):
        if use_mc_dropout and self.mc_dropout_samples > 1:
            return mc_dropout_predict(
                self.dropout_controller,
                lambda: self._forward_prediction_bundle(
                    x, nodes=nodes, update_precision=update_precision,
                    compute_variance=compute_variance,
                ),
                self.mc_dropout_samples,
            )

        return self._forward_prediction_bundle(
            x, nodes=nodes, update_precision=update_precision,
            compute_variance=compute_variance,
        )

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

    CHECKPOINT_FORMAT_VERSION = 2

    def save_checkpoint(self, filepath):
        """Saves model, optimizer, and all auxiliary training state.

        Beyond the model and optimizer, three pieces of state were previously lost
        and each silently changed a resumed run's behavior:

        * the evidential loss's KL-annealing counter, so a resumed run restarted
          its annealing schedule from zero;
        * the LR scheduler, so ``ReduceLROnPlateau``'s patience/num_bad_epochs
          counters reset;
        * the graph-distance standardizer statistics, without which uncertainty
          values before and after a resume are not on the same scale.
        """
        checkpoint = {
            'format_version': self.CHECKPOINT_FORMAT_VERSION,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'uncertainty_head_type': self.uncertainty_head_type,
            'model_name': getattr(self, 'model_name', None),
            'finetune': getattr(self, 'finetune', None),
            'sngp_precision_policy': getattr(self, 'sngp_precision_policy', None),
        }
        if self.output_head is not None:
            checkpoint['uncertainty_head_state_dict'] = self.output_head.state_dict()
        if self.evidential_loss is not None:
            checkpoint['evidential_loss_state_dict'] = self.evidential_loss.state_dict()
        if getattr(self, 'scheduler', None) is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        standardizer = getattr(self, 'graph_distance_standardizer', None)
        if standardizer is not None:
            checkpoint['graph_distance_standardizer'] = standardizer.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Loads a checkpoint, reporting any state it could not restore.

        Missing keys are tolerated so pre-v2 checkpoints still load, but they are
        printed rather than silently ignored -- a quietly reset annealing counter
        or scheduler is exactly the kind of thing that makes two "identical" runs
        diverge for no visible reason.
        """
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found at {filepath}. Skipping load.")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        # Validate the head type *before* touching the backbone. Grafting a head
        # replaces the backbone's final Linear with nn.Identity, so a checkpoint
        # written under one head does not fit a model built with another -- and
        # letting load_state_dict discover that surfaces it as an opaque
        # "Missing key(s) in state_dict" rather than the actual problem.
        saved_head = checkpoint.get('uncertainty_head_type')
        if saved_head is not None and saved_head != self.uncertainty_head_type:
            raise ValueError(
                f"Checkpoint {filepath} was written with uncertainty_head={saved_head!r} "
                f"but this model was built with {self.uncertainty_head_type!r}. Attaching an "
                f"uncertainty head replaces the backbone's final Linear with nn.Identity, so "
                f"their state dicts are not interchangeable. Rebuild the model with "
                f"uncertainty_head={saved_head!r} to load this checkpoint."
            )

        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        missing = []

        def restore(key, target, loader='load_state_dict'):
            if target is None:
                return
            if key in checkpoint:
                getattr(target, loader)(checkpoint[key])
            else:
                missing.append(key)

        restore('uncertainty_head_state_dict', self.output_head)
        restore('evidential_loss_state_dict', self.evidential_loss)
        restore('scheduler_state_dict', getattr(self, 'scheduler', None))
        restore('graph_distance_standardizer', getattr(self, 'graph_distance_standardizer', None))

        self.model.model.to(self.device)
        if self.output_head is not None:
            self.output_head.to(self.device)

        if missing:
            print(
                f"Warning: checkpoint {filepath} (format v{checkpoint.get('format_version', 1)}) "
                f"had no entry for {missing}; that state was left at its initial value."
            )

    def save(self):
        self.save_checkpoint(self.save_path)
        
    def load(self):
        self.load_checkpoint(self.save_path)
