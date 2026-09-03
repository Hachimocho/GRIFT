from contextlib import contextmanager
import dataclasses

import torch
import torch.nn as nn

from .types import PredictionBundle  # noqa: F401  (re-exported for callers)


DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@contextmanager
def _dropout_train_mode(module):
    original_modes = {}
    for child in module.modules():
        if isinstance(child, DROPOUT_TYPES):
            original_modes[child] = child.training
            child.train(True)
    try:
        yield
    finally:
        for child, training in original_modes.items():
            child.train(training)


def count_stochastic_dropout_sites(module):
    """Number of dropout modules that actually randomize, i.e. have ``p > 0``.

    Presence of ``nn.Dropout`` is not sufficient for MC dropout to mean anything.
    ``torchvision``'s ``VisionTransformer`` defaults ``dropout=0.0`` and
    ``attention_dropout=0.0``, and ``vistransformdf`` passes neither -- so it has
    37 ``nn.Dropout`` modules, every one a no-op. Sampling such a model yields
    identical passes and therefore *identically zero* variance: a silently wrong
    measurement rather than an error. ``swin_t`` has the same property (24 sites,
    all ``p=0``), while ``squeezenetdf`` -- which cannot host an external head at
    all -- is the one that works out of the box (``classifier.0`` at ``p=0.5``).

    Because the count depends on the user's ``--uncertainty-dropout-rate`` and on
    modules that only exist after construction, this must be probed at runtime
    rather than declared in a static per-architecture table.
    """
    return sum(
        1 for child in module.modules()
        if isinstance(child, DROPOUT_TYPES) and float(getattr(child, "p", 0.0)) > 0.0
    )


def _mean_or_none(values):
    """Mean across MC samples, or None if no sample carried the field."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return torch.stack(present, dim=0).mean(dim=0)


def _binary_entropy(probabilities):
    probabilities = probabilities.clamp(1e-6, 1 - 1e-6)
    return -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities)
    )


def mc_dropout_predict(model_module, predictor_fn, num_samples):
    """Average ``num_samples`` stochastic forward passes into a single bundle.

    The result is built with :func:`dataclasses.replace` off the last sample
    rather than constructed field by field. That matters: the previous hand-rolled
    construction silently omitted ``alpha``, ``evidence``, ``member_logits`` and
    ``gp_variance``, so ``CNNModel.compute_loss`` -- which dispatches on exactly
    those fields -- raised ``ValueError`` for the evidential head, was swallowed by
    ``evaluate_model``'s per-batch ``except Exception: continue``, and reported
    ``accuracy 0.0`` with no traceback. With ``replace``, any field added to
    :class:`PredictionBundle` later is carried through by default instead of being
    lost by omission.

    Head-specific tensors are averaged across samples. For ``alpha`` that keeps the
    Dirichlet concentration >= 1 (a mean of values >= 1), so the evidential loss
    stays well defined.
    """
    if num_samples <= 1:
        return predictor_fn()

    bundles = []
    with _dropout_train_mode(model_module):
        for _ in range(num_samples):
            bundles.append(predictor_fn())

    stacked_probabilities = torch.stack([bundle.probabilities for bundle in bundles], dim=0)
    mean_probabilities = stacked_probabilities.mean(dim=0)
    logits = torch.logit(mean_probabilities.clamp(1e-6, 1 - 1e-6))
    predictive_variance = stacked_probabilities.var(dim=0, unbiased=False)
    predictive_entropy = _binary_entropy(mean_probabilities)
    expected_entropy = _binary_entropy(stacked_probabilities).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy
    variation_ratio = 1.0 - torch.max(mean_probabilities, 1.0 - mean_probabilities)

    uncertainty = dict(bundles[-1].uncertainty)
    uncertainty.update({
        "mc_dropout_variance": predictive_variance,
        "mc_dropout_entropy": predictive_entropy,
        "mc_dropout_mutual_information": mutual_information,
        "mc_dropout_variation_ratio": variation_ratio,
    })

    return dataclasses.replace(
        bundles[-1],
        logits=logits,
        probabilities=mean_probabilities,
        predictions=None,  # recomputed by with_predictions() from the mean
        features=_mean_or_none([bundle.features for bundle in bundles]),
        uncertainty=uncertainty,
        evidence=_mean_or_none([bundle.evidence for bundle in bundles]),
        alpha=_mean_or_none([bundle.alpha for bundle in bundles]),
        member_logits=_mean_or_none([bundle.member_logits for bundle in bundles]),
        gp_variance=_mean_or_none([bundle.gp_variance for bundle in bundles]),
    ).with_predictions()
