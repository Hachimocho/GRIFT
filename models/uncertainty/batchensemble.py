import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleBinaryHead(nn.Module):
    def __init__(
        self,
        in_features,
        ensemble_size=4,
        hidden_features=256,
        dropout=0.2,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.shared_hidden = nn.Linear(in_features, hidden_features)
        self.shared_output = nn.Linear(hidden_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.input_fast_weights = nn.Parameter(torch.ones(ensemble_size, in_features))
        self.hidden_fast_weights = nn.Parameter(torch.ones(ensemble_size, hidden_features))
        self.member_bias = nn.Parameter(torch.zeros(ensemble_size, 1))

    def forward(self, features):
        scaled_inputs = features.unsqueeze(1) * self.input_fast_weights.unsqueeze(0)
        hidden = torch.einsum("bmd,hd->bmh", scaled_inputs, self.shared_hidden.weight)
        hidden = hidden + self.shared_hidden.bias.view(1, 1, -1)
        hidden = F.relu(hidden, inplace=False)
        hidden = self.dropout(hidden)
        hidden = hidden * self.hidden_fast_weights.unsqueeze(0)

        member_logits = torch.einsum("bmh,oh->bmo", hidden, self.shared_output.weight)
        member_logits = member_logits + self.shared_output.bias.view(1, 1, -1)
        member_logits = member_logits + self.member_bias.unsqueeze(0)
        member_logits = member_logits.squeeze(-1)

        member_probabilities = torch.sigmoid(member_logits)
        probabilities = member_probabilities.mean(dim=1, keepdim=True)
        logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
        ensemble_variance = member_probabilities.var(dim=1, unbiased=False, keepdim=True)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "member_logits": member_logits.unsqueeze(-1),
            "uncertainty": {
                "batchensemble_variance": ensemble_variance,
            },
        }
