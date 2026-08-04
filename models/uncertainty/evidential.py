import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryEvidentialHead(nn.Module):
    def __init__(self, in_features, hidden_features=256, dropout=0.2):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_features, 2)

    def forward(self, features):
        hidden = F.relu(self.hidden(features), inplace=False)
        hidden = self.dropout(hidden)
        evidence = F.softplus(self.output(hidden))
        alpha = evidence + 1.0
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        probabilities = (alpha[:, 1:2] / alpha_sum).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(probabilities) - torch.log1p(-probabilities)
        vacuity = 2.0 / alpha_sum

        return {
            "logits": logits,
            "probabilities": probabilities,
            "evidence": evidence,
            "alpha": alpha,
            "uncertainty": {
                "evidential_vacuity": vacuity,
                "evidential_total_evidence": evidence.sum(dim=1, keepdim=True),
            },
        }


def _dirichlet_kl_divergence(alpha, num_classes=2):
    ones = torch.ones_like(alpha)
    alpha0 = alpha.sum(dim=1, keepdim=True)
    beta0 = ones.sum(dim=1, keepdim=True)

    first = (
        torch.lgamma(alpha0)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(beta0)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
    )
    second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(alpha0))).sum(dim=1, keepdim=True)
    return first + second


class EvidentialBinaryClassificationLoss(nn.Module):
    def __init__(self, annealing_steps=1000):
        super().__init__()
        self.annealing_steps = max(1, annealing_steps)
        self.register_buffer("global_step", torch.zeros(1))

    def forward(self, prediction_bundle, targets):
        if prediction_bundle.alpha is None:
            raise ValueError("Evidential loss requires alpha values in the prediction bundle.")

        alpha = prediction_bundle.alpha
        targets = targets.view(-1, 1).float()
        one_hot = torch.cat([1.0 - targets, targets], dim=1)
        alpha_sum = alpha.sum(dim=1, keepdim=True)

        log_likelihood = torch.sum(one_hot * (torch.digamma(alpha_sum) - torch.digamma(alpha)), dim=1, keepdim=True)
        anneal = min(1.0, float(self.global_step.item()) / float(self.annealing_steps))
        adjusted_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
        kl_term = _dirichlet_kl_divergence(adjusted_alpha)

        loss = log_likelihood + anneal * kl_term
        self.global_step += 1
        return loss.mean()
