import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


PRECISION_POLICIES = ("per-epoch", "final-epoch", "never-reset")


class RandomFourierFeatures(nn.Module):
    """Fixed random Fourier projection approximating an RBF kernel.

    ``weight`` and ``bias`` are buffers, so they survive ``state_dict`` -- but on a
    fresh build they consume the ambient torch RNG in a size-dependent way. Pass
    ``seed`` so the projection is reproducible regardless of how much randomness
    was drawn before the head happened to be constructed.
    """

    def __init__(self, in_features, out_features, scale=1.0, seed=None):
        super().__init__()
        self.scale = scale
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer(
            "weight", torch.randn(in_features, out_features, generator=generator)
        )
        self.register_buffer(
            "bias", 2 * math.pi * torch.rand(out_features, generator=generator)
        )

    def forward(self, inputs):
        projections = (inputs @ self.weight) / self.scale + self.bias
        return math.sqrt(2.0 / self.weight.size(1)) * torch.cos(projections)


class SNGPBinaryHead(nn.Module):
    """Spectral-normalized neural Gaussian process head for a binary output.

    Predictive variance comes from a Laplace approximation: a running precision
    matrix ``ridge*I + sum(phi phi^T)`` over random-feature activations, inverted
    to a covariance.

    Two things about that accumulation matter, and both were previously wrong.

    **Reset policy.** The precision must be reset before the epoch(s) whose data it
    is meant to summarize. ``reset_precision_matrix`` existed but had no callers
    anywhere, so precision accumulated across every epoch of a run and
    ``gp_variance`` shrank monotonically for reasons unrelated to the data --
    leaving it incomparable between epochs, which is exactly the comparison a
    benchmark needs to make. Policies:

    ``per-epoch``    reset at the start of every epoch and accumulate throughout
                     (default; gives epoch-comparable variance).
    ``final-epoch``  accumulate only during the last epoch. Closest to the original
                     single-pass formulation, but yields no per-epoch curve.
    ``never-reset``  the original, broken behavior. Retained solely to reproduce
                     pre-fix runs.

    **Numerics.** Accumulation and inversion are forced to float32 outside
    autocast. Training runs under ``torch.cuda.amp.autocast``, and an fp16
    Gram-matrix accumulation into a registered buffer followed by an fp16
    pseudo-inverse is not numerically sound. The inverse is also computed lazily
    behind a dirty flag, and via Cholesky (the matrix is symmetric positive
    definite by construction) rather than a ``pinv`` on every forward -- the cache
    used to be invalidated by each accumulation step, so a 256x256 pseudo-inverse
    ran once per training batch.
    """

    def __init__(
        self,
        in_features,
        hidden_features=256,
        rff_features=256,
        ridge_penalty=1.0,
        dropout=0.2,
        rff_seed=None,
        precision_policy="per-epoch",
    ):
        super().__init__()
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"precision_policy must be one of {PRECISION_POLICIES}, "
                f"got {precision_policy!r}"
            )
        self.hidden = spectral_norm(nn.Linear(in_features, hidden_features))
        self.dropout = nn.Dropout(dropout)
        self.random_features = RandomFourierFeatures(
            hidden_features, rff_features, seed=rff_seed
        )
        self.beta = nn.Linear(rff_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.ridge_penalty = ridge_penalty
        self.precision_policy = precision_policy
        self.register_buffer(
            "precision_matrix",
            ridge_penalty * torch.eye(rff_features, dtype=torch.float32),
        )
        self._cached_covariance = None
        self._precision_dirty = True
        #: Whether the current epoch should accumulate precision. Set by
        #: ``on_epoch_start``; defaults True so a head driven without epoch hooks
        #: behaves like ``per-epoch``.
        self.should_accumulate_precision = True

    # -- epoch hooks -------------------------------------------------------- #

    def on_epoch_start(self, epoch, num_epochs=None):
        """Apply the reset policy for the epoch about to begin."""
        if self.precision_policy == "never-reset":
            self.should_accumulate_precision = True
            return

        if self.precision_policy == "per-epoch":
            self.reset_precision_matrix()
            self.should_accumulate_precision = True
            return

        is_final = num_epochs is None or epoch >= num_epochs - 1
        if is_final:
            self.reset_precision_matrix()
        self.should_accumulate_precision = bool(is_final)

    def on_epoch_end(self, epoch):
        """Paired with ``on_epoch_start`` for callers; currently a no-op."""
        return None

    # -- precision ---------------------------------------------------------- #

    def reset_precision_matrix(self):
        eye = torch.eye(
            self.precision_matrix.size(0),
            device=self.precision_matrix.device,
            dtype=torch.float32,
        )
        self.precision_matrix.copy_(self.ridge_penalty * eye)
        self._cached_covariance = None
        self._precision_dirty = True

    def _update_precision(self, random_features):
        with torch.autocast(device_type=random_features.device.type, enabled=False):
            features = random_features.detach().to(torch.float32)
            self.precision_matrix += features.transpose(0, 1) @ features
        self._precision_dirty = True

    def _covariance(self):
        if self._cached_covariance is None or self._precision_dirty:
            with torch.autocast(device_type=self.precision_matrix.device.type, enabled=False):
                precision = self.precision_matrix.to(torch.float32)
                stabilized = precision + 1e-6 * torch.eye(
                    precision.size(0), device=precision.device, dtype=torch.float32
                )
                try:
                    factor = torch.linalg.cholesky(stabilized)
                    self._cached_covariance = torch.cholesky_inverse(factor)
                except RuntimeError:
                    # Cholesky needs strict positive-definiteness; fall back if
                    # accumulated round-off pushes the matrix out of that regime.
                    self._cached_covariance = torch.linalg.pinv(stabilized)
            self._precision_dirty = False
        return self._cached_covariance

    # -- forward ------------------------------------------------------------ #

    def forward(self, features, update_precision=False, compute_variance=True):
        hidden = F.relu(self.hidden(features), inplace=False)
        hidden = self.dropout(hidden)
        random_features = self.random_features(hidden)

        if self.training and update_precision and self.should_accumulate_precision:
            self._update_precision(random_features)

        gp_mean = self.beta(random_features) + self.bias

        if not compute_variance:
            # Skip the factorization entirely for training batches whose
            # uncertainty is not being summarized on this step.
            return {
                "logits": gp_mean,
                "probabilities": torch.sigmoid(gp_mean),
                "gp_variance": None,
                "uncertainty": {},
            }

        covariance = self._covariance().to(random_features.dtype)
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
