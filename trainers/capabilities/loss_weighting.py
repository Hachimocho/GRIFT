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

`midband` is also reused, unmodified, by `IValueTraversal`'s `selection_mode="midband"` --
see its docstring for why sharing this implementation (rather than writing the shape twice)
matters here specifically.
"""

import torch

#: How an I-value may scale a sample's loss.
#:
#: ``linear`` and ``rank`` are monotonic: the highest I-value always gets the most weight,
#: which is the wrong shape for "train on what the model half-knows" -- the very top of the
#: I-value range is exactly where mislabelled and corrupted samples land (nothing about a high
#: I-value distinguishes "hard" from "wrong"), and the very bottom is samples already mastered.
#: ``midband`` scales by ``trapezoid_desirability`` instead, which peaks on a plateau inside
#: `band` and tapers to zero at both ends -- a smooth, distribution-shaped analogue of
#: `IValueTraversal`'s hard-cutoff `selection_mode="band"`, sharing its `band` parameter.
#:
#: ``rank`` scales by the I-value's rank within the batch, which is invariant to the
#: estimator's output range -- and that matters, because the legacy estimators live in a
#: 0.02-wide band around 0.31 while the fixed ones are unbounded. `midband` inherits this by
#: also ranking first, for the same reason.
LOSS_WEIGHT_MODES = ("none", "linear", "rank", "midband")

#: Weights are clipped to `[1/clip, clip]`, geometric about 1 so the mean weight stays ~1 and
#: a weighted arm remains comparable to its control in total gradient magnitude. Unbounded
#: weights on a target this heavy-tailed (kurtosis +45) would let a few samples own the step.
DEFAULT_WEIGHT_CLIP = 2.0

#: `--ivalue-band`'s default, reused here so `midband` has a sane shape with no flags of its
#: own -- see `trapezoid_desirability` for what the two numbers mean.
DEFAULT_BAND = (0.4, 0.7)

#: Fraction of the band's width spent ramping up (and, symmetrically, ramping down) at each
#: edge, rather than jumping straight from 0 to full weight. 0.3 leaves a plateau covering the
#: middle 40% of the band -- narrow enough that "mid-band" still means something, wide enough
#: that the plateau is not a knife-edge a single rank can fall off of. There is no principled
#: value here; it is a shape choice, which is exactly why it is a constant and not another CLI
#: flag -- one polished trapezoid beats a family of untested ones.
RAMP_FRACTION = 0.3


def trapezoid_desirability(scaled, low, high, ramp_fraction=RAMP_FRACTION):
    """`d in [0, 1]`, 0 at and beyond `low`/`high`, 1 on a plateau strictly inside them.

    `scaled` is a rank-quantile in `[0, 1]` (0 = lowest I-value in the batch, 1 = highest),
    exactly the quantity `ivalue_weights(mode="rank")` already computes -- this is a drop-in
    replacement for that mode's monotonic ramp, not a new quantity. Works identically on a
    python float or a `torch.Tensor` of any shape: every operation below is elementwise and
    defined for both.

    The ramp is a *fraction* of the band's width, not an absolute quantile amount, so the
    plateau always occupies the band's middle `1 - 2 * ramp_fraction` (40% at the default
    0.3) -- a narrow band gets a narrow plateau rather than losing it outright. Only a
    `ramp_fraction >= 0.5` removes the plateau entirely, degrading to a pure triangle (tent)
    peaking at the band's midpoint; the default is well under that.
    """
    low, high = float(low), float(high)
    if high <= low:
        raise ValueError(f"band must have low < high, got ({low}, {high})")

    ramp = max(1e-6, ramp_fraction * (high - low))
    # Two triangular ramps: one rising out of `low`, one falling into `high`. Their pointwise
    # minimum is 0 outside the band, rises to 1 by `low + ramp`, stays at 1 (whichever ramp
    # is not yet saturated) until `high - ramp`, then falls back to 0 by `high` -- a trapezoid,
    # with the narrow-band/triangle case falling out for free when the two ramps cross before
    # either reaches 1.
    rising = (scaled - low) / ramp
    falling = (high - scaled) / ramp
    return _clamp01(_min(rising, falling))


def _min(a, b):
    return torch.minimum(a, b) if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) \
        else min(a, b)


def _clamp01(value):
    return value.clamp(0.0, 1.0) if isinstance(value, torch.Tensor) \
        else max(0.0, min(1.0, value))


def ivalue_weights(values, mode, clip=DEFAULT_WEIGHT_CLIP, band=DEFAULT_BAND, device=None):
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

    if mode in ("rank", "midband"):
        order = raw.argsort().argsort().float()
        scaled = order / float(raw.numel() - 1)
    else:
        spread = float(raw.max() - raw.min())
        scaled = ((raw - raw.min()) / spread if spread > 1e-12
                  else torch.full_like(raw, 0.5))

    if mode == "midband":
        low, high = band
        scaled = trapezoid_desirability(scaled, low, high)

    clip = max(1.0, float(clip))
    return clip ** (2.0 * scaled - 1.0)


class LossWeighter:
    """Holds the mode and clip, applies the weights, and tracks what it applied.

    The running totals exist because a weighted arm no longer takes the same-sized steps as
    its control, so "equal nodes per epoch" stops meaning equal gradient magnitude. The mean
    weight has to be reported alongside the accuracy or the comparison is not readable.
    """

    def __init__(self, mode="none", clip=DEFAULT_WEIGHT_CLIP, band=DEFAULT_BAND):
        if mode not in LOSS_WEIGHT_MODES:
            raise ValueError(
                f"unknown loss weight mode {mode!r}; choose from "
                f"{', '.join(LOSS_WEIGHT_MODES)}"
            )
        self.mode = mode
        self.clip = float(clip)
        # Unused by every mode except `midband`; stored regardless so `apply` needs no
        # special-casing and a caller can always pass the same `--ivalue-band` value it
        # already threads to `IValueTraversal`.
        self.band = tuple(band)
        self.weight_applied = 0.0
        self.weighted_samples = 0

    @property
    def enabled(self):
        return self.mode != "none"

    def apply(self, per_sample_loss, values):
        """Weighted mean of `per_sample_loss`; the plain mean when there is no signal."""
        if not self.enabled:
            return per_sample_loss.mean()
        weights = ivalue_weights(values, self.mode, self.clip, band=self.band,
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
