import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.scale = scale
        self.register_buffer("weight", torch.randn(in_features, out_features))
        self.register_buffer("bias", 2 * math.pi * torch.rand(out_features))

    def forward(self, inputs):
        projections = (inputs @ self.weight) / self.scale + self.bias
        return math.sqrt(2.0 / self.weight.size(1)) * torch.cos(projections)


class SNGPBinaryHead(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=256,
        rff_features=256,
        ridge_penalty=1.0,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden = spectral_norm(nn.Linear(in_features, hidden_features))
        self.dropout = nn.Dropout(dropout)
        self.random_features = RandomFourierFeatures(hidden_features, rff_features)
        self.beta = nn.Linear(rff_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.ridge_penalty = ridge_penalty
        self.register_buffer("precision_matrix", ridge_penalty * torch.eye(rff_features))
        self._cached_covariance = None

    def reset_precision_matrix(self):
        eye = torch.eye(self.precision_matrix.size(0), device=self.precision_matrix.device)
        self.precision_matrix.copy_(self.ridge_penalty * eye)
        self._cached_covariance = None

    def _update_precision(self, random_features):
        self.precision_matrix += random_features.transpose(0, 1) @ random_features
        self._cached_covariance = None

    def _covariance(self):
        if self._cached_covariance is None:
            eye = torch.eye(self.precision_matrix.size(0), device=self.precision_matrix.device)
            stabilized_precision = self.precision_matrix + 1e-6 * eye
            self._cached_covariance = torch.linalg.pinv(stabilized_precision)
        return self._cached_covariance

    def forward(self, features, update_precision=False):
        hidden = F.relu(self.hidden(features), inplace=False)
        hidden = self.dropout(hidden)
        random_features = self.random_features(hidden)

        if self.training and update_precision:
            self._update_precision(random_features.detach())

        gp_mean = self.beta(random_features) + self.bias
        covariance = self._covariance()
        gp_variance = (random_features @ covariance * random_features).sum(dim=1, keepdim=True)
        mean_field_logits = gp_mean / torch.sqrt(1.0 + (math.pi / 8.0) * gp_variance)
        probabilities = torch.sigmoid(mean_field_logits)

        return {
            "logits": gp_mean,
            "probabilities": probabilities,
            "gp_variance": gp_variance,
            "uncertainty": {
                "sngp_variance": gp_variance,
            },
        }
