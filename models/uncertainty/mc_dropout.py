from contextlib import contextmanager

import torch
import torch.nn as nn

from .types import PredictionBundle


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


def _binary_entropy(probabilities):
    probabilities = probabilities.clamp(1e-6, 1 - 1e-6)
    return -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities)
    )


def mc_dropout_predict(model_module, predictor_fn, num_samples):
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

    features = bundles[-1].features
    if features is None:
        stacked_features = [bundle.features for bundle in bundles if bundle.features is not None]
        if stacked_features:
            features = torch.stack(stacked_features, dim=0).mean(dim=0)

    return PredictionBundle(
        logits=logits,
        probabilities=mean_probabilities,
        features=features,
        uncertainty=uncertainty,
    ).with_predictions()
