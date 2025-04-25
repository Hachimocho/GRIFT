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
    across different values of sensitive attributes.
    """
    def __init__(self, attribute_metadata_list, attr_map):
        """
        Args:
            attribute_metadata_list (list[AttributeMetadata]): List of metadata objects for attributes.
            attr_map (dict): A mapping from attribute name to AttributeMetadata object (or index).
                           Used to quickly check if an attribute should be considered for bias.
        """
        super().__init__()
        # Store the metadata and map for reference within forward pass
        self.attribute_metadata = attribute_metadata_list
        self.attr_map = attr_map
        print("Initialized bias loss module.")

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

        # Second pass: calculate bias loss between different attribute values
        for attr_name, value_preds in attr_predictions.items():
            if len(value_preds) < 2:  # Need at least 2 different values to compare
                continue

            # Calculate mean prediction for each attribute value
            value_means = {}
            for value, preds_list in value_preds.items():
                if preds_list:
                    # Ensure preds_list contains tensors before stacking
                    valid_preds = [p for p in preds_list if isinstance(p, torch.Tensor)]
                    if valid_preds:
                         # Detach to treat means as constants in this comparison phase if needed,
                         # but keep grad for the loss calculation itself.
                         value_means[value] = torch.stack(valid_preds).mean()

            values = list(value_means.keys())
            if len(values) >= 2: # Check if there are at least two groups to compare
                # Calculate pairwise differences
                for i, val1 in enumerate(values):
                    for val2 in values[i+1:]:
                        mean1, mean2 = value_means[val1], value_means[val2]

                        # Bias loss is the squared difference between means (MSE)
                        # Ensure requires_grad=True propagates if needed
                        current_loss = F.mse_loss(mean1, mean2)
                        # Accumulate the loss properly to maintain computation graph
                        total_loss = total_loss + current_loss
                        num_comparisons += 1

        # Return average bias loss, handle division by zero
        if num_comparisons > 0:
            avg_loss = total_loss / num_comparisons
            return avg_loss
        else:
            # Return zero tensor that requires grad if no comparisons were made
            return torch.zeros(1, device=predictions.device, requires_grad=True)
