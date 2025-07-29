import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class AttributeMetadata:
    """Stores metadata about a specific attribute, including its type and potential values."""
    def __init__(self, name, attr_type, possible_values=None):
        self.name = name
        self.attr_type = attr_type  # 'categorical' or 'continuous'
        self.possible_values = possible_values  # For categorical attributes
        # These might be used for tracking stats, keeping them for now
        self.value_counts = defaultdict(int)
        self.predictions = defaultdict(list)

class AttributeBiasLoss(nn.Module):
    """Calculates a bias loss based on the difference in average predictions
    across different values of sensitive attributes, with per-attribute weights."""
    def __init__(self, attribute_metadata_list, attr_map, attribute_weights=None):
        super().__init__()
        self.attribute_metadata = attribute_metadata_list
        self.attr_map = attr_map
        # Set up per-attribute weights (default 1.0)
        if attribute_weights is None:
            self.attribute_weights = {attr.name: 1.0 for attr in attribute_metadata_list}
        else:
            self.attribute_weights = attribute_weights
        print(f"Initialized bias loss module with attribute weights: {self.attribute_weights}")

    def forward(self, predictions, labels, nodes):
        """Calculate bias loss based on attribute predictions.

        Args:
            predictions (torch.Tensor): Model output probabilities/logits (batch_size, num_classes or batch_size).
            labels (torch.Tensor): Ground truth labels (batch_size).
            nodes (list[AttributeNode]): List of nodes corresponding to the batch samples.
                                        Used to access node attributes.

        Returns:
            torch.Tensor: A scalar tensor representing the average bias loss.
        """
        # print("Calculating bias loss...")
        # Initial checks for valid inputs
        if predictions is None or labels is None or nodes is None or len(nodes) == 0:
            print("Warning: Missing input data for bias calculation.")
            return torch.zeros(1, device=predictions.device, requires_grad=True)

        # Ensure predictions and labels are on the same device
        if predictions.device != labels.device:
            labels = labels.to(predictions.device)

        batch_size = predictions.size(0)

        # Assuming binary classification or using the probability of the positive class
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Use softmax/sigmoid depending on model output; assuming sigmoid for binary-like case
            pred_probs = predictions.sigmoid()[:, 1] # Or appropriate index/logic
        elif predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            # Handle 1D output or [batch_size, 1] output by squeezing
            pred_probs = predictions.squeeze(dim=-1).sigmoid()
        else:
            print(f"Warning: Unexpected prediction shape: {predictions.shape}")
            return torch.zeros(1, device=predictions.device, requires_grad=True)

        # Group predictions by attribute values
        attr_predictions = defaultdict(lambda: defaultdict(list))

        # First pass: collect predictions for each attribute value for relevant attributes
        for i, node in enumerate(nodes):
            if not hasattr(node, 'attributes') or not isinstance(node.attributes, dict):
                # print(f"Warning: Node {i} has no attributes dict.")
                continue

            node_attrs = node.attributes
            for attr_name, attr_val in node_attrs.items():
                # Check if this attribute is one we are tracking for bias (via attr_map)
                if attr_name in self.attr_map:
                    # Ensure the value is hashable (e.g., convert lists/arrays if necessary)
                    hashable_val = tuple(attr_val) if isinstance(attr_val, list) else attr_val
                    try:
                        attr_predictions[attr_name][hashable_val].append(pred_probs[i])
                    except TypeError:
                        # Handle unhashable types if necessary, e.g., by skipping or converting
                        # print(f"Warning: Unhashable attribute value {attr_val} for {attr_name}. Skipping.")
                        pass

        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        num_comparisons = 0
        total_weight = 0.0

        # Second pass: calculate bias loss between different attribute values
        for attr_name, value_preds in attr_predictions.items():
            if len(value_preds) < 2:
                continue
            value_means = {}
            for value, preds_list in value_preds.items():
                if preds_list:
                    valid_preds = [p for p in preds_list if isinstance(p, torch.Tensor)]
                    if valid_preds:
                        value_means[value] = torch.stack(valid_preds).mean()
            values = list(value_means.keys())
            if len(values) >= 2:
                attr_weight = self.attribute_weights.get(attr_name, 1.0)
                for i, val1 in enumerate(values):
                    for val2 in values[i+1:]:
                        mean1, mean2 = value_means[val1], value_means[val2]
                        current_loss = F.mse_loss(mean1, mean2)
                        total_loss = total_loss + attr_weight * current_loss
                        num_comparisons += 1
                        total_weight += attr_weight
        if num_comparisons > 0 and total_weight > 0:
            avg_loss = total_loss / total_weight
            return avg_loss
        else:
            return torch.zeros(1, device=predictions.device, requires_grad=True)
