"""Scale a sample's loss by its I-value, in one place used by both training paths.

Why weighting at all: every scheme that used the I-value to *select* has lost to plain i.i.d.
sampling, and Phase 0 measured why -- choosing among a node's k-NN neighbours yields batches
2.3x less diverse than i.i.d. ones (0.0385 vs 0.0872 mean pairwise embedding distance).
Weighting leaves the sampler alone, so the batches stay diverse, and uses the signal only to
decide how much each sample contributes to the gradient.

Why one module: `CapabilityManager.train_with_traversal` routes *every* traversal through
`DQNCapability.train_with_dqn` as soon as a DQN exists, so an implementation living only in
`BasicTrainingCapability` is unreachable exactly when the I-value it needs is available. That
is not a hypothetical -- the first version of this weighting was silently dead for that reason,
and the arm would have run as its own control. Both paths now call the same function.
"""

import torch

#: How an I-value may scale a sample's loss.
#:
#: ``linear`` scales by the I-value normalised within the batch; ``rank`` scales by its rank,
#: which is invariant to the estimator's output range -- and that matters, because the legacy
#: estimators live in a 0.02-wide band around 0.31 while the fixed ones are unbounded.
LOSS_WEIGHT_MODES = ("none", "linear", "rank")

#: Weights are clipped to `[1/clip, clip]`, geometric about 1 so the mean weight stays ~1 and
#: a weighted arm remains comparable to its control in total gradient magnitude. Unbounded
#: weights on a target this heavy-tailed (kurtosis +45) would let a few samples own the step.
DEFAULT_WEIGHT_CLIP = 2.0


def ivalue_weights(values, mode, clip=DEFAULT_WEIGHT_CLIP, device=None):
    """Bounded per-sample weights from raw I-values. Returns None when there is no signal."""
    if mode == "none" or values is None or len(values) == 0:
        return None

    raw = torch.as_tensor(list(values), dtype=torch.float32, device=device)
    finite = torch.isfinite(raw)
    if not bool(finite.any()):
        return None
    # A single unreadable I-value must not discard the batch, nor poison the scaling.
    raw = torch.nan_to_num(raw, nan=float(raw[finite].mean()),
                           posinf=float(raw[finite].max()), neginf=float(raw[finite].min()))

    if raw.numel() == 1:
        return torch.ones_like(raw)

    if mode == "rank":
        order = raw.argsort().argsort().float()
        scaled = order / float(raw.numel() - 1)
    else:
        spread = float(raw.max() - raw.min())
        scaled = ((raw - raw.min()) / spread if spread > 1e-12
                  else torch.full_like(raw, 0.5))

    clip = max(1.0, float(clip))
    return clip ** (2.0 * scaled - 1.0)


class LossWeighter:
    """Holds the mode and clip, applies the weights, and tracks what it applied.

    The running totals exist because a weighted arm no longer takes the same-sized steps as
    its control, so "equal nodes per epoch" stops meaning equal gradient magnitude. The mean
    weight has to be reported alongside the accuracy or the comparison is not readable.
    """

    def __init__(self, mode="none", clip=DEFAULT_WEIGHT_CLIP):
        if mode not in LOSS_WEIGHT_MODES:
            raise ValueError(
                f"unknown loss weight mode {mode!r}; choose from "
                f"{', '.join(LOSS_WEIGHT_MODES)}"
            )
        self.mode = mode
        self.clip = float(clip)
        self.weight_applied = 0.0
        self.weighted_samples = 0

    @property
    def enabled(self):
        return self.mode != "none"

    def apply(self, per_sample_loss, values):
        """Weighted mean of `per_sample_loss`; the plain mean when there is no signal."""
        if not self.enabled:
            return per_sample_loss.mean()
        weights = ivalue_weights(values, self.mode, self.clip,
                                 device=per_sample_loss.device)
        if weights is None or weights.numel() != per_sample_loss.numel():
            return per_sample_loss.mean()
        self.weight_applied += float(weights.sum())
        self.weighted_samples += int(weights.numel())
        return (weights * per_sample_loss).mean()

    def summary_and_reset(self):
        """`(mean_weight, n)` since the last call, or None if nothing was weighted."""
        if not self.weighted_samples:
            return None
        mean = self.weight_applied / self.weighted_samples
        count = self.weighted_samples
        self.weight_applied = 0.0
        self.weighted_samples = 0
        return mean, count


__all__ = ["DEFAULT_WEIGHT_CLIP", "LOSS_WEIGHT_MODES", "LossWeighter", "ivalue_weights"]
