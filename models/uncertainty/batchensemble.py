import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleBinaryHead(nn.Module):
    """Rank-1 BatchEnsemble head over shared weights.

    Each member modulates the shared layers with its own rank-1 "fast weight"
    vector. Those vectors **must** be initialized to random +/-1 signs: with an
    all-ones initialization every member computes an identical function, receives
    an identical gradient, and stays identical forever. At eval, with dropout
    disabled, the reported ensemble variance is then exactly 0.0 for every input
    -- the method silently measures nothing. Random sign vectors are what the
    original BatchEnsemble formulation prescribes, and they are what make member
    disagreement a real signal.
    """

    def __init__(
        self,
        in_features,
        ensemble_size=4,
        hidden_features=256,
        dropout=0.2,
        init_seed=None,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.shared_hidden = nn.Linear(in_features, hidden_features)
        self.shared_output = nn.Linear(hidden_features, 1)
        self.dropout = nn.Dropout(dropout)

        generator = None
        if init_seed is not None:
            generator = torch.Generator().manual_seed(int(init_seed))

        def sign_vectors(*shape):
            draws = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
            return draws.to(torch.float32) * 2.0 - 1.0

        self.input_fast_weights = nn.Parameter(sign_vectors(ensemble_size, in_features))
        self.hidden_fast_weights = nn.Parameter(sign_vectors(ensemble_size, hidden_features))
        # Zero is canonical here -- the sign vectors already break the symmetry.
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
