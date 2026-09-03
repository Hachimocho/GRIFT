"""The corruption ladder: `evaluation/uq/corruptions.py`.

Three properties carry the whole protocol, and each one fails silently if broken:

* **severity 0 is byte-identical to clean** -- otherwise the severity-0 row of a shift
  table is not the in-distribution row, and every "degradation at severity 1" is
  measured from the wrong baseline;
* **stochastic corruptions are order-independent** -- they run across `num_workers`
  threads, so a global RNG would make the result depend on batch size and thread
  scheduling, and two runs of the "same" benchmark would disagree;
* **the output really is uint8 (H, W, 3)** -- `CNNModel.transform` has a bare-except
  fallback that returns an *unnormalized BGR* tensor, so a malformed array produces
  quietly wrong numbers with no error and no coverage hit.
"""

import numpy as np
import pytest

from evaluation.uq import corruptions
from evaluation.uq.corruptions import (
    CORRUPTIONS,
    SEVERITIES,
    CorruptionError,
    ImageCorruption,
    apply,
    full_matrix,
    severity_ladder,
)

STOCHASTIC = ("gaussian_noise",)
DETERMINISTIC = ("gaussian_blur", "jpeg")
ACTIVE = DETERMINISTIC + STOCHASTIC


def make_image(width=64, height=48, seed=0):
    """A uint8 BGR image, non-square so a transposed shape is detectable."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# The output contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corruption", ACTIVE)
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_output_is_uint8_bgr_of_the_same_shape(corruption, severity):
    image = make_image()
    result = apply(image, corruption, severity, key="a/b.png")
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert result.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("corruption", ACTIVE)
def test_non_square_images_are_not_transposed(corruption):
    """A (48, 64, 3) image must come back (48, 64, 3), not (64, 48, 3)."""
    image = make_image(width=64, height=48)
    assert apply(image, corruption, 3, key="x").shape == (48, 64, 3)


@pytest.mark.parametrize("corruption", ACTIVE)
def test_values_stay_in_range(corruption):
    """Clipping matters: noise pushes past both ends on a saturated image."""
    image = np.full((16, 16, 3), 250, dtype=np.uint8)
    image[:8] = 5
    result = apply(image, corruption, 5, key="x")
    assert result.min() >= 0 and result.max() <= 255


def test_a_wrong_dtype_input_is_refused():
    with pytest.raises(CorruptionError, match="expected uint8"):
        apply(make_image().astype(np.float32), "gaussian_blur", 2)


def test_a_grayscale_input_is_refused():
    with pytest.raises(CorruptionError, match=r"expected \(H, W, 3\)"):
        apply(np.zeros((8, 8), dtype=np.uint8), "gaussian_blur", 2)


def test_a_non_array_input_is_refused():
    with pytest.raises(CorruptionError, match="expected an ndarray"):
        apply([[1, 2, 3]], "gaussian_blur", 2)


def test_an_unknown_corruption_is_refused():
    with pytest.raises(CorruptionError, match="unknown corruption"):
        apply(make_image(), "swirl", 2)


@pytest.mark.parametrize("severity", [-1, 6, 99, 1.5])
def test_an_out_of_range_severity_is_refused(severity):
    with pytest.raises(CorruptionError, match="severity must be"):
        apply(make_image(), "gaussian_blur", severity)


# --------------------------------------------------------------------------- #
# Severity 0
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_severity_zero_is_byte_identical(corruption):
    """Not "approximately clean" -- exactly the input bytes.

    A JPEG round-trip at quality 100 is *not* lossless, so this has to short-circuit
    before the encoder rather than rely on the encoder being an identity.
    """
    image = make_image()
    result = apply(image, corruption, 0, key="a/b.png")
    assert result.tobytes() == image.tobytes()


def test_the_none_corruption_is_identity_at_every_severity():
    image = make_image()
    for severity in SEVERITIES:
        assert apply(image, "none", severity).tobytes() == image.tobytes()


@pytest.mark.parametrize("corruption", ACTIVE)
def test_severity_one_actually_changes_the_image(corruption):
    """A ladder whose first rung is a no-op measures nothing."""
    image = make_image()
    assert apply(image, corruption, 1, key="x").tobytes() != image.tobytes()


# --------------------------------------------------------------------------- #
# Determinism and order independence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corruption", ACTIVE)
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_the_same_inputs_give_the_same_bytes(corruption, severity):
    image = make_image()
    first = apply(image, corruption, severity, key="a/b.png")
    second = apply(image, corruption, severity, key="a/b.png")
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("corruption", STOCHASTIC)
def test_noise_survives_an_intervening_global_reseed(corruption):
    """The point of per-image seeds: no global RNG is consulted.

    A global-RNG implementation would pass the repeat test above and fail this one,
    which is exactly the situation under a ThreadPoolExecutor where other code is
    drawing from numpy's global state concurrently.
    """
    image = make_image()
    first = apply(image, corruption, 3, key="a/b.png")

    np.random.seed(999)
    np.random.random(1000)
    import random as global_random
    global_random.seed(12345)
    global_random.random()

    second = apply(image, corruption, 3, key="a/b.png")
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("corruption", STOCHASTIC)
def test_noise_does_not_depend_on_processing_order(corruption):
    """Corrupting images in reverse order must give the same per-image result.

    Under a global RNG the second image's noise would depend on how many draws the
    first consumed, so the whole batch would change with batch size and worker count.
    """
    images = {f"img{index}.png": make_image(seed=index) for index in range(5)}
    forward = {
        key: apply(image, corruption, 2, key=key).tobytes()
        for key, image in images.items()
    }
    reverse = {
        key: apply(images[key], corruption, 2, key=key).tobytes()
        for key in reversed(list(images))
    }
    assert forward == reverse


@pytest.mark.parametrize("corruption", STOCHASTIC)
def test_different_images_get_different_noise(corruption):
    """Same pixels, different identity -> different noise.

    Otherwise every image in the dataset receives the *same* noise field, which is a
    systematic perturbation rather than sensor noise and would be far easier to detect.
    """
    image = make_image()
    first = apply(image, corruption, 3, key="a/one.png")
    second = apply(image, corruption, 3, key="a/two.png")
    assert first.tobytes() != second.tobytes()


@pytest.mark.parametrize("corruption", STOCHASTIC)
def test_the_salt_rerandomizes_without_changing_severity(corruption):
    """The honest way to check a shift result is not one lucky noise draw."""
    image = make_image()
    first = apply(image, corruption, 3, key="x", salt="salt-a")
    second = apply(image, corruption, 3, key="x", salt="salt-b")
    assert first.tobytes() != second.tobytes()
    # Same magnitude, different realization.
    difference = np.abs(first.astype(int) - image.astype(int)).mean()
    other = np.abs(second.astype(int) - image.astype(int)).mean()
    assert abs(difference - other) < 0.5 * max(difference, other)


@pytest.mark.parametrize("corruption", DETERMINISTIC)
def test_deterministic_corruptions_ignore_the_key(corruption):
    """Blur and JPEG have no randomness, so identity must not affect them."""
    image = make_image()
    assert (apply(image, corruption, 3, key="a").tobytes()
            == apply(image, corruption, 3, key="b").tobytes())


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #

def test_blur_is_never_a_silent_noop_on_a_small_image():
    """The MIN_BLUR_SIGMA regression.

    Blur sigma is a fraction of source width, applied before the 255px resize. On a
    small source the low severities work out sub-pixel, OpenCV derives a 1x1 kernel,
    and severities 1-2 become byte-identical to clean -- so the ladder would report
    "no degradation until severity 3" as a finding rather than as quantization.
    """
    for width in (32, 64, 160, 255, 512):
        image = make_image(width=width, height=width, seed=width)
        for severity in SEVERITIES[1:]:
            result = apply(image, "gaussian_blur", severity)
            assert result.tobytes() != image.tobytes(), (
                f"blur severity {severity} was a no-op at width {width}"
            )


def test_the_blur_floor_does_not_engage_at_the_model_resolution():
    """At 255px and above the fraction dominates, so the floor is inert.

    Worth pinning: if the floor started binding at real resolutions it would flatten
    the low end of the ladder into a constant.
    """
    assert corruptions.BLUR_SIGMA_FRACTION[1] * 255 > corruptions.MIN_BLUR_SIGMA


def test_blur_severity_monotonically_reduces_high_frequency_content():
    """Higher severity must be *more* blurred, or the ladder is not ordered.

    Measured as the variance of the Laplacian, the standard sharpness proxy and the
    same quantity the dataset's own `blur_score` uses.
    """
    import cv2

    image = make_image(width=128, height=128, seed=3)
    sharpness = [
        cv2.Laplacian(apply(image, "gaussian_blur", severity), cv2.CV_64F).var()
        for severity in SEVERITIES
    ]
    assert sharpness == sorted(sharpness, reverse=True), sharpness


def test_jpeg_severity_monotonically_increases_distortion():
    image = make_image(width=128, height=128, seed=4)
    error = [
        np.abs(apply(image, "jpeg", severity).astype(int) - image.astype(int)).mean()
        for severity in SEVERITIES
    ]
    assert error[0] == 0.0, "severity 0 must be exact"
    assert error[1:] == sorted(error[1:]), error


def test_noise_severity_monotonically_increases_distortion():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    error = [
        np.abs(apply(image, "gaussian_noise", severity, key="x").astype(int)
               - image.astype(int)).std()
        for severity in SEVERITIES
    ]
    assert error[0] == 0.0
    assert error[1:] == sorted(error[1:]), error


def test_blur_severity_is_resolution_invariant():
    """Sigma scales with width, so severity means the same at any source resolution.

    Absolute-pixel severities would blur a 160px video crop far harder than a 1024px
    FFHQ image, so a "shift" result would partly be a source-resolution result -- and
    resolution correlates with source, which is what the holdout axis already varies.
    """
    import cv2

    small = make_image(width=64, height=64, seed=5)
    large = cv2.resize(small, (256, 256), interpolation=cv2.INTER_CUBIC)

    def relative_sharpness_drop(image):
        clean = cv2.Laplacian(image, cv2.CV_64F).var()
        blurred = cv2.Laplacian(apply(image, "gaussian_blur", 3), cv2.CV_64F).var()
        return blurred / clean

    small_drop = relative_sharpness_drop(small)
    large_drop = relative_sharpness_drop(large)
    # Loose bound: upscaling is not information-preserving, so these cannot match
    # exactly. The claim is only that they are the same order, which absolute-pixel
    # sigmas would not satisfy.
    assert 0.2 < small_drop / large_drop < 5.0, (small_drop, large_drop)


# --------------------------------------------------------------------------- #
# ImageCorruption
# --------------------------------------------------------------------------- #

def test_the_callable_labels_itself():
    corruption = ImageCorruption("jpeg", 3)
    assert corruption.corruption == "jpeg"
    assert corruption.severity == 3
    assert not corruption.is_identity


@pytest.mark.parametrize("spec", [("none", 0), ("none", 3), ("jpeg", 0)])
def test_identity_cells_report_themselves_as_identity(spec):
    assert ImageCorruption(*spec).is_identity


def test_the_callable_applies_and_counts():
    corruption = ImageCorruption("gaussian_blur", 2)
    image = make_image()
    result = corruption(image)
    assert result.tobytes() != image.tobytes()
    assert corruption.n_applied == 1
    assert corruption.summary()["severity"] == 2


class FakeNode:
    def __init__(self, node_id):
        self.node_id = node_id


def test_the_key_is_the_relative_path_not_the_absolute_one():
    """Moving the dataset must not redraw every noise sample.

    An absolute-path key embeds the data root, so a re-run from a different mount
    would produce different noise and silently fail to reproduce the recorded table.
    """
    corruption = ImageCorruption("gaussian_noise", 3, data_root="/mnt/a/ai-face")
    other = ImageCorruption("gaussian_noise", 3, data_root="/mnt/b/ai-face")
    image = make_image()
    first = corruption(image, FakeNode("/mnt/a/ai-face/FFHQ/1.png"))
    second = other(image, FakeNode("/mnt/b/ai-face/FFHQ/1.png"))
    assert first.tobytes() == second.tobytes()


def test_an_invalid_spec_is_refused_at_construction():
    """Fail when the cell is defined, not when the first image arrives mid-sweep."""
    with pytest.raises(CorruptionError):
        ImageCorruption("swirl", 1)
    with pytest.raises(CorruptionError):
        ImageCorruption("jpeg", 9)


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #

def test_the_ladder_covers_every_severity():
    ladder = severity_ladder("jpeg")
    assert [severity for _name, severity in ladder] == list(SEVERITIES)


def test_the_ladder_can_omit_clean():
    ladder = severity_ladder("jpeg", include_clean=False)
    assert 0 not in [severity for _name, severity in ladder]


def test_the_none_ladder_is_a_single_cell():
    assert severity_ladder("none") == [("none", 0)]


def test_the_full_matrix_includes_clean_exactly_once():
    """Clean is one cell, not one per family: the three would be identical rows."""
    matrix = full_matrix()
    assert matrix.count(("none", 0)) == 1
    assert sum(1 for _name, severity in matrix if severity == 0) == 1


def test_the_full_matrix_has_no_duplicates():
    matrix = full_matrix()
    assert len(matrix) == len(set(matrix))
    # 1 clean + 3 families x 5 severities.
    assert len(matrix) == 1 + 3 * 5


def test_describe_names_every_family():
    text = corruptions.describe()
    for corruption in CORRUPTIONS:
        if corruption != "none":
            assert corruption in text
