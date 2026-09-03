"""`trapezoid_desirability`: the shared "avoid both ends" weighting shape.

Shared between `IValueTraversal`'s `selection_mode="midband"` and
`ivalue_weights(mode="midband")` specifically so the "peak in the middle-high, taper at
both extremes" shape cannot drift between the selection and loss-weighting versions of
the same idea -- see `trainers/capabilities/loss_weighting.py`'s module docstring for
why that has already happened once to this project's weighting code.
"""

import pytest
import torch

from trainers.capabilities.loss_weighting import trapezoid_desirability


def test_zero_at_and_beyond_the_low_edge():
    assert trapezoid_desirability(0.0, 0.4, 0.7) == 0.0
    assert trapezoid_desirability(0.2, 0.4, 0.7) == 0.0
    assert trapezoid_desirability(0.4, 0.4, 0.7) == 0.0


def test_zero_at_and_beyond_the_high_edge():
    assert trapezoid_desirability(0.7, 0.4, 0.7) == 0.0
    assert trapezoid_desirability(0.85, 0.4, 0.7) == 0.0
    assert trapezoid_desirability(1.0, 0.4, 0.7) == 0.0


def test_one_on_the_plateau():
    """0.3-width ramp on a (0.4, 0.7) band -> ramp = 0.09, plateau ~[0.49, 0.61]."""
    assert trapezoid_desirability(0.55, 0.4, 0.7) == pytest.approx(1.0)
    assert trapezoid_desirability(0.50, 0.4, 0.7) == pytest.approx(1.0)
    assert trapezoid_desirability(0.60, 0.4, 0.7) == pytest.approx(1.0)


def test_ramps_are_monotonic():
    """Approaching the plateau from either side must never decrease desirability."""
    rising = [trapezoid_desirability(q, 0.4, 0.7) for q in
             (0.40, 0.42, 0.44, 0.46, 0.48, 0.50)]
    assert rising == sorted(rising)
    falling = [trapezoid_desirability(q, 0.4, 0.7) for q in
              (0.60, 0.62, 0.64, 0.66, 0.68, 0.70)]
    assert falling == sorted(falling, reverse=True)


def test_symmetric_about_the_bands_midpoint_for_a_symmetric_band():
    midpoint = 0.55
    for offset in (0.01, 0.05, 0.1, 0.14):
        left = trapezoid_desirability(midpoint - offset, 0.4, 0.7)
        right = trapezoid_desirability(midpoint + offset, 0.4, 0.7)
        assert left == pytest.approx(right, abs=1e-9)


def test_a_narrow_band_gets_a_narrow_plateau_not_no_plateau():
    """The ramp is a *fraction* of the band's width, so a narrow band still has a
    (proportionally narrow) flat top rather than losing it outright."""
    low, high = 0.5, 0.52
    width = high - low
    plateau_center = low + width / 2.0
    assert trapezoid_desirability(plateau_center, low, high) == pytest.approx(1.0)
    assert trapezoid_desirability(low, low, high) == 0.0
    assert trapezoid_desirability(high, low, high) == 0.0


def test_ramp_fraction_of_one_half_degrades_to_a_pure_triangle():
    """At the boundary where the two ramps exactly meet, there is no flat top left --
    only `trapezoid_desirability`'s documented exception to "always has a plateau"."""
    low, high, midpoint = 0.4, 0.7, 0.55
    peak = trapezoid_desirability(midpoint, low, high, ramp_fraction=0.5)
    just_off_peak = trapezoid_desirability(midpoint + 0.01, low, high, ramp_fraction=0.5)
    assert peak == pytest.approx(1.0)
    assert just_off_peak < peak, "a pure triangle has no flat top around its apex"


def test_zero_width_band_is_refused():
    with pytest.raises(ValueError, match="low < high"):
        trapezoid_desirability(0.5, 0.5, 0.5)


def test_inverted_band_is_refused():
    with pytest.raises(ValueError, match="low < high"):
        trapezoid_desirability(0.5, 0.7, 0.4)


def test_works_on_a_bare_python_float():
    result = trapezoid_desirability(0.55, 0.4, 0.7)
    assert isinstance(result, float)


def test_works_elementwise_on_a_tensor():
    scaled = torch.tensor([0.0, 0.3, 0.55, 0.8, 1.0])
    result = trapezoid_desirability(scaled, 0.4, 0.7)
    assert isinstance(result, torch.Tensor)
    assert result.shape == scaled.shape
    expected = torch.tensor([
        trapezoid_desirability(float(value), 0.4, 0.7) for value in scaled
    ])
    assert torch.allclose(result, expected)


def test_full_range_band_is_one_everywhere_inside_the_ramps():
    """(0, 1) is the widest legal band -- most of [0, 1] should reach the plateau."""
    values = [trapezoid_desirability(q, 0.0, 1.0) for q in
             (0.35, 0.5, 0.65)]
    assert all(value == pytest.approx(1.0) for value in values)


def test_output_never_exceeds_the_unit_interval():
    for low, high in ((0.0, 1.0), (0.3, 0.35), (0.45, 0.55)):
        for q in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            value = trapezoid_desirability(q, low, high)
            assert 0.0 <= value <= 1.0
