"""Rank-based metrics must be invariant to monotone rescaling of the score.

This is the property that makes cross-method comparison legitimate at all. The
in-tree uncertainty signals live on incompatible scales:

    evidential_vacuity        (0, 1]
    mc_dropout_variance       [0, 0.25]
    mc_dropout_entropy        [0, log 2]
    sngp_variance             unbounded above
    hybrid_distance           roughly [0, 2]
    degree_penalty            (0, 1]

Any metric used to rank methods against each other must therefore depend only on
the *ordering* of the scores, never their magnitudes. If one of these assertions
fails, the corresponding metric silently favors whichever method happens to
produce larger numbers.
"""

import numpy as np
import pytest

from evaluation.uq.metrics import (
    accuracy_at_coverage, aupr_error, ood_detection, risk_coverage_curve,
    uncertainty_error_auroc,
)

#: Strictly increasing transforms. A rank-based metric cannot distinguish these.
MONOTONE_TRANSFORMS = {
    "identity": lambda x: x,
    "scale_x7": lambda x: x * 7.0,
    "shift_plus3": lambda x: x + 3.0,
    "scale_and_shift": lambda x: 0.001 * x + 100.0,
    "exp": lambda x: np.exp(x),
    "log1p": lambda x: np.log1p(x),
    "sqrt": lambda x: np.sqrt(x),
    "rank": lambda x: np.argsort(np.argsort(x)).astype(float),
}


@pytest.fixture
def scenario():
    """A moderately hard case: 200 samples, ~25% errors, informative uncertainty."""
    rng = np.random.Generator(np.random.PCG64(11))
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Probabilities correlated with the label, so some predictions are wrong.
    probabilities = np.clip(
        y_true * 0.65 + 0.175 + rng.normal(0, 0.22, size=n), 0.01, 0.99
    )
    errors = (probabilities > 0.5).astype(int) != y_true
    # Uncertainty that is genuinely informative but imperfect, and strictly positive
    # so that log/sqrt transforms stay defined.
    uncertainty = 0.1 + errors * 0.4 + rng.uniform(0, 0.3, size=n)
    return y_true, probabilities, uncertainty


@pytest.mark.parametrize("name", sorted(MONOTONE_TRANSFORMS))
def test_uncertainty_error_auroc_is_scale_free(scenario, name):
    y_true, probabilities, uncertainty = scenario
    baseline = uncertainty_error_auroc(y_true, probabilities, uncertainty).value
    transformed = uncertainty_error_auroc(
        y_true, probabilities, MONOTONE_TRANSFORMS[name](uncertainty)
    ).value
    assert transformed == pytest.approx(baseline, abs=1e-12), (
        f"AUROC-error changed under a monotone transform ({name})"
    )


@pytest.mark.parametrize("name", sorted(MONOTONE_TRANSFORMS))
def test_aupr_error_is_scale_free(scenario, name):
    y_true, probabilities, uncertainty = scenario
    baseline = aupr_error(y_true, probabilities, uncertainty).value
    transformed = aupr_error(
        y_true, probabilities, MONOTONE_TRANSFORMS[name](uncertainty)
    ).value
    assert transformed == pytest.approx(baseline, abs=1e-12)


@pytest.mark.parametrize("name", sorted(MONOTONE_TRANSFORMS))
def test_aurc_and_eaurc_are_scale_free(scenario, name):
    y_true, probabilities, uncertainty = scenario
    baseline = risk_coverage_curve(y_true, probabilities, uncertainty)
    transformed = risk_coverage_curve(
        y_true, probabilities, MONOTONE_TRANSFORMS[name](uncertainty)
    )
    assert transformed.aurc == pytest.approx(baseline.aurc, abs=1e-12)
    assert transformed.eaurc == pytest.approx(baseline.eaurc, abs=1e-12)
    assert np.allclose(transformed.risk, baseline.risk, atol=1e-12)


@pytest.mark.parametrize("name", sorted(MONOTONE_TRANSFORMS))
def test_accuracy_at_coverage_is_scale_free(scenario, name):
    y_true, probabilities, uncertainty = scenario
    baseline = accuracy_at_coverage(y_true, probabilities, uncertainty)
    transformed = accuracy_at_coverage(
        y_true, probabilities, MONOTONE_TRANSFORMS[name](uncertainty)
    )
    for level, value in baseline.items():
        assert transformed[level] == pytest.approx(value, abs=1e-12), f"{level} under {name}"


@pytest.mark.parametrize("name", sorted(MONOTONE_TRANSFORMS))
def test_ood_auroc_is_scale_free(name):
    rng = np.random.Generator(np.random.PCG64(13))
    uncertainty_id = 0.1 + rng.uniform(0, 0.4, size=150)
    uncertainty_ood = 0.3 + rng.uniform(0, 0.5, size=80)

    # The transform must be applied to the *combined* scores and then re-split.
    # Transforming each partition separately would not be a single monotone map --
    # `rank`, for instance, would renumber each side independently and destroy the
    # cross-partition ordering the metric is measuring.
    transform = MONOTONE_TRANSFORMS[name]
    combined = transform(np.concatenate([uncertainty_id, uncertainty_ood]))

    baseline = ood_detection(uncertainty_id, uncertainty_ood)
    transformed = ood_detection(combined[:150], combined[150:])
    assert transformed.auroc == pytest.approx(baseline.auroc, abs=1e-12)
    assert transformed.fpr_at_95_tpr == pytest.approx(baseline.fpr_at_95_tpr, abs=1e-12)


def test_a_decreasing_transform_inverts_the_metric():
    """Sanity check that the invariance tests are not vacuous.

    If negating the score left AUROC unchanged, the metric would be ignoring the
    score entirely and the invariance assertions above would prove nothing.
    """
    rng = np.random.Generator(np.random.PCG64(17))
    y_true = rng.integers(0, 2, size=120)
    probabilities = np.clip(y_true * 0.6 + 0.2 + rng.normal(0, 0.2, size=120), 0.01, 0.99)
    errors = (probabilities > 0.5).astype(int) != y_true
    uncertainty = 0.1 + errors * 0.5

    forward = uncertainty_error_auroc(y_true, probabilities, uncertainty).value
    inverted = uncertainty_error_auroc(y_true, probabilities, -uncertainty).value
    assert forward > 0.6, "fixture should give an informative score"
    assert inverted == pytest.approx(1.0 - forward, abs=1e-12)


def test_real_uncertainty_scales_are_comparable_after_ranking():
    """Two methods on wildly different scales but identical orderings must tie.

    Emulates comparing e.g. sngp_variance against hybrid_distance.
    """
    rng = np.random.Generator(np.random.PCG64(19))
    y_true = rng.integers(0, 2, size=150)
    probabilities = np.clip(y_true * 0.6 + 0.2 + rng.normal(0, 0.25, size=150), 0.01, 0.99)

    ordering = rng.uniform(0.0, 1.0, size=150)
    bounded = 0.25 * ordering                 # like mc_dropout_variance
    unbounded = 500.0 * ordering + 1e4        # like sngp_variance

    bounded_auroc = uncertainty_error_auroc(y_true, probabilities, bounded).value
    unbounded_auroc = uncertainty_error_auroc(y_true, probabilities, unbounded).value
    assert bounded_auroc == pytest.approx(unbounded_auroc, abs=1e-12)
