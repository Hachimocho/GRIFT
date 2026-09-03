"""Temperature scaling.

The two properties worth pinning: it improves calibration, and it provably cannot
change any ranking metric. The second is why it must not be reported as a distinct
ranking result.
"""

import numpy as np
import pytest

from evaluation.uq.metrics import (
    binary_calibration, negative_log_likelihood, risk_coverage_curve,
    score_max_probability, uncertainty_error_auroc,
)
from evaluation.uq.temperature import (
    LOG_T_BOUNDS, apply_temperature, apply_to_records, fit_from_records, fit_temperature,
    load_fit, probabilities_to_logits, save_fit,
)


def overconfident_data(n=500, seed=0, sharpness=3.0):
    """Logits deliberately too large for their accuracy: T > 1 should help."""
    rng = np.random.Generator(np.random.PCG64(seed))
    labels = rng.integers(0, 2, size=n).astype(float)
    signal = (labels * 2.0 - 1.0) + rng.normal(0, 1.0, size=n)
    return labels, signal * sharpness


def underconfident_data(n=500, seed=1):
    """Logits too small for their accuracy: T < 1 should help."""
    rng = np.random.Generator(np.random.PCG64(seed))
    labels = rng.integers(0, 2, size=n).astype(float)
    signal = (labels * 2.0 - 1.0) * 3.0 + rng.normal(0, 0.5, size=n)
    return labels, signal * 0.15


def make_frame(labels, logits):
    import pandas as pd

    probabilities = apply_temperature(logits, 1.0)
    return pd.DataFrame({
        "record_id": np.arange(len(labels)),
        "label": labels.astype(int),
        "logit": logits,
        "prob": probabilities,
        "pred": (probabilities > 0.5).astype(int),
        "correct": ((probabilities > 0.5).astype(int) == labels.astype(int)).astype(int),
    })


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def test_overconfident_logits_are_smoothed():
    labels, logits = overconfident_data()
    fit = fit_temperature(labels, logits=logits)
    assert fit.temperature > 1.0, f"expected smoothing, got T={fit.temperature}"
    assert fit.converged
    assert fit.nll_after < fit.nll_before
    assert fit.ece_after < fit.ece_before


def test_underconfident_logits_are_sharpened():
    labels, logits = underconfident_data()
    fit = fit_temperature(labels, logits=logits)
    assert fit.temperature < 1.0, f"expected sharpening, got T={fit.temperature}"
    assert fit.nll_after <= fit.nll_before + 1e-9


def test_already_calibrated_logits_yield_temperature_near_one():
    rng = np.random.Generator(np.random.PCG64(5))
    probabilities = rng.uniform(0.02, 0.98, size=4000)
    labels = (rng.uniform(size=4000) < probabilities).astype(float)
    fit = fit_temperature(labels, probabilities=probabilities)
    assert fit.temperature == pytest.approx(1.0, abs=0.12), (
        f"calibrated data should need little correction, got T={fit.temperature}"
    )


def test_fit_matches_a_fine_grid_search():
    """Cross-check the optimizer against brute force."""
    labels, logits = overconfident_data(n=300, seed=7)
    fit = fit_temperature(labels, logits=logits)

    grid = np.exp(np.linspace(*LOG_T_BOUNDS, 4001))
    losses = [
        negative_log_likelihood(labels, apply_temperature(logits, t)).value for t in grid
    ]
    best = grid[int(np.argmin(losses))]
    assert fit.temperature == pytest.approx(best, rel=1e-3), (
        f"optimizer found T={fit.temperature}, grid search found {best}"
    )


def test_fit_is_reproducible():
    labels, logits = overconfident_data(n=200, seed=3)
    first = fit_temperature(labels, logits=logits)
    second = fit_temperature(labels, logits=logits)
    assert first.temperature == second.temperature
    assert first.as_dict() == second.as_dict()


def test_fit_accepts_probabilities_instead_of_logits():
    labels, logits = overconfident_data(n=200, seed=11)
    probabilities = apply_temperature(logits, 1.0)
    from_logits = fit_temperature(labels, logits=logits)
    from_probs = fit_temperature(labels, probabilities=probabilities)
    assert from_probs.temperature == pytest.approx(from_logits.temperature, rel=1e-4)


def test_fit_requires_some_input():
    with pytest.raises(ValueError, match="logits or probabilities"):
        fit_temperature(np.array([1.0, 0.0]))


def test_single_class_fit_returns_the_identity():
    """With one class, NLL is minimized by a degenerate T, so refuse to fit."""
    labels = np.ones(50)
    _, logits = overconfident_data(n=50, seed=2)
    fit = fit_temperature(labels, logits=logits)
    assert fit.temperature == 1.0
    assert fit.converged is False


def test_too_few_samples_returns_the_identity():
    fit = fit_temperature(np.array([1.0]), logits=np.array([0.5]))
    assert fit.temperature == 1.0
    assert fit.n_val == 1


# --------------------------------------------------------------------------- #
# The ranking-invariance property
# --------------------------------------------------------------------------- #

#: Temperatures that do not push probabilities into float64 saturation. Below ~0.5
#: the logits are scaled up enough that `1 - max(p, 1-p)` underflows to exactly 0.0
#: for the most confident samples, so distinct scores collapse into ties -- the
#: transform stays monotone but stops being injective. See the saturation test below.
NON_SATURATING_TEMPERATURES = [0.5, 1.0, 2.0, 5.0, 17.0]


@pytest.mark.parametrize("temperature", NON_SATURATING_TEMPERATURES)
def test_temperature_cannot_change_any_ranking_metric(temperature):
    """Dividing a logit by T > 0 is monotone, so the ordering is untouched.

    This is exactly why temperature scaling must not be reported as a separate
    ranking result: it would print numbers identical to the baseline and imply a
    difference that does not exist.
    """
    labels, logits = overconfident_data(n=400, seed=13)
    baseline_probs = apply_temperature(logits, 1.0)
    scaled_probs = apply_temperature(logits, temperature)

    baseline_auroc = uncertainty_error_auroc(
        labels, baseline_probs, score_max_probability(baseline_probs)
    ).value
    scaled_auroc = uncertainty_error_auroc(
        labels, scaled_probs, score_max_probability(scaled_probs)
    ).value
    assert scaled_auroc == pytest.approx(baseline_auroc, abs=1e-12)

    baseline_rc = risk_coverage_curve(
        labels, baseline_probs, score_max_probability(baseline_probs)
    )
    scaled_rc = risk_coverage_curve(
        labels, scaled_probs, score_max_probability(scaled_probs)
    )
    assert scaled_rc.aurc == pytest.approx(baseline_rc.aurc, abs=1e-12)
    assert scaled_rc.eaurc == pytest.approx(baseline_rc.eaurc, abs=1e-12)


def test_aggressive_sharpening_saturates_and_creates_ties():
    """Documents the one place the invariance is only approximate.

    At small T the logits are scaled up enough that `1 - max(p, 1-p)` underflows to
    exactly 0.0 for the most confident samples. The transform is still monotone, but
    no longer injective, so previously distinct scores become ties. AUROC is
    tie-aware and unaffected; AURC's cumulative ordering shifts slightly because ties
    resolve by record index.

    Worth pinning because it bounds the invariance claim: a fitted temperature that
    sharpens this hard would be an unusual result, but the report should not assert
    exact equality if one occurs.
    """
    labels, logits = overconfident_data(n=400, seed=13)
    saturating = score_max_probability(apply_temperature(logits, 0.25))
    baseline = score_max_probability(apply_temperature(logits, 1.0))

    assert (saturating == 0.0).sum() > 0, "expected float saturation at T=0.25"
    assert len(np.unique(saturating)) < len(np.unique(baseline)), (
        "saturation should collapse distinct scores into ties"
    )

    # AUROC survives, because it averages over tied ranks.
    assert uncertainty_error_auroc(
        labels, apply_temperature(logits, 0.25), saturating
    ).value == pytest.approx(
        uncertainty_error_auroc(labels, apply_temperature(logits, 1.0), baseline).value,
        abs=1e-12,
    )
    # AURC drifts, but only slightly.
    saturated_aurc = risk_coverage_curve(
        labels, apply_temperature(logits, 0.25), saturating
    ).aurc
    baseline_aurc = risk_coverage_curve(
        labels, apply_temperature(logits, 1.0), baseline
    ).aurc
    assert saturated_aurc == pytest.approx(baseline_aurc, abs=1e-3)


@pytest.mark.parametrize("temperature", [0.5, 2.0, 5.0])
def test_temperature_preserves_predictions_and_accuracy(temperature):
    """The 0.5 threshold maps to logit 0, which a positive rescaling fixes."""
    labels, logits = overconfident_data(n=300, seed=19)
    baseline = (apply_temperature(logits, 1.0) > 0.5).astype(int)
    scaled = (apply_temperature(logits, temperature) > 0.5).astype(int)
    assert np.array_equal(baseline, scaled)


def test_temperature_does_change_calibration():
    """The point of the method: same ranking, better probabilities."""
    labels, logits = overconfident_data(n=600, seed=23)
    fit = fit_temperature(labels, logits=logits)

    before = binary_calibration(labels, apply_temperature(logits, 1.0))
    after = binary_calibration(labels, apply_temperature(logits, fit.temperature))
    assert after.ece < before.ece * 0.9, (
        f"ECE barely moved ({before.ece:.4f} -> {after.ece:.4f}) despite T="
        f"{fit.temperature:.3f}"
    )


def test_apply_temperature_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.array([0.5]), 0.0)


def test_probabilities_to_logits_roundtrips():
    probabilities = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    recovered = apply_temperature(probabilities_to_logits(probabilities), 1.0)
    assert np.allclose(recovered, probabilities, atol=1e-9)


# --------------------------------------------------------------------------- #
# Record integration
# --------------------------------------------------------------------------- #

def test_fit_from_records_uses_the_logit_column():
    labels, logits = overconfident_data(n=300, seed=29)
    frame = make_frame(labels, logits)
    fit = fit_from_records(frame, records_sha256="abc123")
    assert fit.temperature > 1.0
    assert fit.records_sha256 == "abc123"
    assert fit.fit_split == "val"


def test_fit_from_records_falls_back_to_probabilities():
    labels, logits = overconfident_data(n=300, seed=31)
    frame = make_frame(labels, logits)
    frame["logit"] = np.nan  # force the fallback path
    fit = fit_from_records(frame)
    assert fit.temperature > 1.0


def test_apply_to_records_produces_a_scoreable_table():
    labels, logits = overconfident_data(n=200, seed=37)
    frame = make_frame(labels, logits)
    fit = fit_from_records(frame)
    scaled = apply_to_records(frame, fit)

    assert "u_temp_maxprob" in scaled.columns
    assert scaled["temperature"].eq(fit.temperature).all()
    # Predictions and accuracy are unchanged, since the rescaling is monotone.
    assert np.array_equal(scaled["pred"].to_numpy(), frame["pred"].to_numpy())
    assert scaled["correct"].sum() == frame["correct"].sum()
    # But the probabilities differ.
    assert not np.allclose(scaled["prob"].to_numpy(), frame["prob"].to_numpy())


def test_scored_temperature_cell_improves_calibration_only():
    """End to end through the scoring path: ECE improves, ranking does not move."""
    from evaluation.uq.scoring import Cell, score_cells

    labels, logits = overconfident_data(n=500, seed=41)
    frame = make_frame(labels, logits)
    fit = fit_from_records(frame)
    scaled = apply_to_records(frame, fit)

    results = score_cells([
        Cell(detector="effnetdf", method_id="baseline_maxprob",
             score_column="u_maxprob", frame=frame, coverage=1.0,
             determinism_mode="strict"),
        Cell(detector="effnetdf", method_id="temperature_scaling",
             score_column="u_temp_maxprob", frame=scaled, coverage=1.0,
             determinism_mode="strict"),
    ], require_comparable=False).set_index("method_id")

    assert results.loc["temperature_scaling", "ece_confidence"] < results.loc[
        "baseline_maxprob", "ece_confidence"
    ], "temperature scaling should improve ECE"
    assert results.loc["temperature_scaling", "auroc_error"] == pytest.approx(
        results.loc["baseline_maxprob", "auroc_error"], abs=1e-12
    ), "and must leave the ranking metric untouched"


def test_fit_persistence_roundtrip(tmp_path):
    labels, logits = overconfident_data(n=200, seed=43)
    fit = fit_temperature(labels, logits=logits, records_sha256="deadbeef")
    path = save_fit(fit, tmp_path / "temperature.json")
    restored = load_fit(path)
    assert restored.temperature == fit.temperature
    assert restored.records_sha256 == "deadbeef"
    assert restored.as_dict() == fit.as_dict()


def test_fitting_on_test_is_not_what_the_api_encourages():
    """The recorded split is part of the artifact, so misuse is auditable.

    Fitting on the split you report is fitting the metric, not calibrating. The API
    cannot prevent it, but it records which split was used.
    """
    labels, logits = overconfident_data(n=200, seed=47)
    fit = fit_temperature(labels, logits=logits, fit_split="test")
    assert fit.fit_split == "test", "the split must be recorded so this is visible"
