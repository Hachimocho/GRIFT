"""Deterministic image corruptions: the secondary distribution-shift axis.

Applied to the decoded image *before* `CNNModel.transform`, so no detector is touched
and the ladder is model-agnostic by construction.

Four facts about that insertion point shape everything here, and each one silently
corrupts the benchmark if ignored:

**The image is uint8, (H, W, 3), BGR, pre-resize.** BGR because the RGB conversion
happens inside `transform`. Pre-resize because `transform` does the 255x255 resize
afterwards -- so a corruption measured in absolute pixels means something different for
a 1024px FFHQ image than for a 160px video crop. Severities are therefore parameterized
in **fractions of image width**, not pixels.

**The output must be uint8, (H, W, 3), C-contiguous.** `transform` has a bare-except
fallback that silently returns an *unnormalized BGR* tensor, so a malformed corruption
produces quietly wrong numbers with no exception and no coverage hit. `apply` asserts
the contract itself rather than relying on anything downstream to notice.

**Stochastic corruptions run across `num_workers` threads.** So randomness is derived
per image from `blake2s(corruption | severity | rel_path | salt)`, never from a global
RNG. That makes output bit-exact independently of batch size, worker count, thread
scheduling, and evaluation order -- none of which a global RNG under the existing
`ThreadPoolExecutor` would give.

**Severity 0 is byte-identical to clean.** Not "approximately clean": the identity
branch returns the input array unchanged, so the severity-0 row of a shift table is
literally the in-distribution row and any difference is a bug rather than a finding.

The ladder is deliberately three families, not the full ImageNet-C set: blur (optics
and downscaling), JPEG (the compression every real pipeline applies), and sensor noise.
More families would multiply fixture surface without adding a distinct question.
"""

import hashlib

import numpy as np

#: Severity levels. 0 is the identity.
SEVERITIES = (0, 1, 2, 3, 4, 5)

#: Gaussian blur sigma as a fraction of image width, so severity means the same thing
#: at every source resolution. Calibrated against the 255px the model actually sees:
#: at that width these are sigma 0.51 -> 4.08, the range over which a face goes from
#: crisp to unrecognizable. Because `transform` resizes source -> 255 *after* this
#: runs, a fixed fraction of source width arrives as the same fraction of 255.
BLUR_SIGMA_FRACTION = (0.0, 0.002, 0.004, 0.008, 0.012, 0.016)

#: Absolute floor on blur sigma, in source pixels.
#:
#: Necessary because the fraction is applied *pre*-resize. On a small source (video
#: crops run ~160px) the low severities work out sub-pixel, and OpenCV derives its
#: kernel from sigma -- so it picks a 1x1 kernel and the corruption becomes a silent
#: no-op. The severity ladder would then have identical rows at severities 1-2 and
#: report "no degradation until severity 3", which is a measurement artifact reported
#: as a finding.
#:
#: The floor costs exact resolution-invariance below 255px wide: those images are
#: upscaled afterwards, so a floored 0.5px blur lands slightly stronger than the
#: fraction intended. That is the right trade -- non-zero and monotone beats
#: proportional and quantized to nothing.
MIN_BLUR_SIGMA = 0.5

#: JPEG quality. Resolution-independent already, so these are absolute.
JPEG_QUALITY = (100, 75, 55, 40, 25, 15)

#: Additive Gaussian noise sigma in [0, 1] intensity units, applied before the uint8
#: round-trip.
NOISE_SIGMA = (0.0, 0.02, 0.04, 0.07, 0.10, 0.13)

CORRUPTIONS = ("none", "gaussian_blur", "jpeg", "gaussian_noise")

#: Mixed into every per-image seed. Bump it to re-randomize a whole sweep without
#: changing any severity, which is the only honest way to check that a shift result is
#: not an artifact of one particular noise draw.
DEFAULT_SALT = "grift-uq-v1"


class CorruptionError(ValueError):
    """Raised for an unknown corruption or an out-of-range severity."""


def _seed_for_image(corruption, severity, key, salt):
    """A per-image seed, independent of evaluation order.

    blake2s over the identifying tuple: stdlib, fast enough to run per image, and
    PYTHONHASHSEED-independent (unlike `hash()` of a str).
    """
    payload = f"{corruption}|{int(severity)}|{key}|{salt}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def _validate(corruption, severity):
    if corruption not in CORRUPTIONS:
        raise CorruptionError(
            f"unknown corruption {corruption!r}; expected one of {CORRUPTIONS}"
        )
    if severity not in SEVERITIES:
        raise CorruptionError(
            f"severity must be one of {SEVERITIES}, got {severity!r}"
        )


def _check_image(image, where):
    if not isinstance(image, np.ndarray):
        raise CorruptionError(f"{where}: expected an ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise CorruptionError(f"{where}: expected uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise CorruptionError(f"{where}: expected (H, W, 3), got {image.shape}")


def apply(image, corruption="none", severity=0, key="", salt=DEFAULT_SALT):
    """Corrupt one decoded BGR uint8 image. Returns a uint8 (H, W, 3) array.

    ``key`` identifies the image (its ``rel_path`` or ``node_id``) and is what makes
    the result reproducible per image rather than per call.
    """
    _validate(corruption, severity)
    _check_image(image, "input")

    # Severity 0 short-circuits before any work, so the identity is exact rather than
    # a round-trip through an encoder that happens to be lossless.
    if corruption == "none" or severity == 0:
        return image

    if corruption == "gaussian_blur":
        result = _gaussian_blur(image, severity)
    elif corruption == "jpeg":
        result = _jpeg(image, severity)
    elif corruption == "gaussian_noise":
        result = _gaussian_noise(image, severity, corruption, key, salt)
    else:  # pragma: no cover - _validate already rejected everything else
        raise CorruptionError(corruption)

    # Assert our own contract. transform()'s bare-except fallback would otherwise turn
    # a malformed array into an unnormalized tensor: no error, no coverage hit, wrong
    # numbers.
    _check_image(result, f"output of {corruption} severity {severity}")
    if result.shape != image.shape:
        raise CorruptionError(
            f"{corruption} severity {severity} changed the shape "
            f"{image.shape} -> {result.shape}"
        )
    return np.ascontiguousarray(result)


def _gaussian_blur(image, severity):
    """Blur with sigma scaled to image width, so severity is resolution-invariant."""
    import cv2

    width = image.shape[1]
    fraction = BLUR_SIGMA_FRACTION[severity]
    if fraction <= 0:
        return image
    # See MIN_BLUR_SIGMA: without the floor, a sub-pixel sigma makes OpenCV pick a 1x1
    # kernel and the corruption silently does nothing.
    sigma = max(MIN_BLUR_SIGMA, fraction * width)
    # ksize=(0, 0) lets OpenCV derive the kernel from sigma, which keeps the kernel
    # proportional too -- a fixed kernel would clip the blur at high sigma.
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def _jpeg(image, severity):
    """Encode and decode as JPEG. Deterministic, but encoder-version dependent.

    The manifest records `cv2.__version__` for exactly this reason: the same quality
    setting on a different OpenCV build produces different bytes, so a severity ladder
    is only comparable within one recorded version.
    """
    import cv2

    quality = JPEG_QUALITY[severity]
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise CorruptionError(f"cv2.imencode failed at quality {quality}")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise CorruptionError(f"cv2.imdecode failed at quality {quality}")
    return decoded


def _gaussian_noise(image, severity, corruption, key, salt):
    """Additive noise from a per-image stream, so worker count cannot change it."""
    seed = _seed_for_image(corruption, severity, key, salt)
    generator = np.random.Generator(np.random.PCG64(seed))
    sigma = NOISE_SIGMA[severity]
    scaled = image.astype(np.float32) / 255.0
    noisy = scaled + generator.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)


class ImageCorruption:
    """A callable corruption, ready to hand to ``evaluate_model``.

    Carries `corruption` and `severity` so the record collector can label rows without
    the caller threading them separately -- a mislabelled severity is a silent error
    that survives into the report.
    """

    __slots__ = ("corruption", "severity", "salt", "data_root", "n_applied", "n_failed")

    def __init__(self, corruption="none", severity=0, salt=DEFAULT_SALT,
                 data_root=None):
        _validate(corruption, severity)
        self.corruption = corruption
        self.severity = severity
        self.salt = salt
        self.data_root = data_root
        self.n_applied = 0
        self.n_failed = 0

    @property
    def is_identity(self):
        return self.corruption == "none" or self.severity == 0

    def key_for(self, node):
        """The identity a node's noise stream derives from.

        The dataset-relative path, not the absolute one: an absolute path embeds the
        data root, so moving the dataset would silently redraw every noise sample and
        make a re-run non-comparable with the recorded one.
        """
        from evaluation.uq.records import relative_path

        return relative_path(getattr(node, "node_id", node), self.data_root)

    def __call__(self, image, node=None):
        key = self.key_for(node) if node is not None else ""
        result = apply(image, self.corruption, self.severity, key=key, salt=self.salt)
        self.n_applied += 1
        return result

    def summary(self):
        return {
            "corruption": self.corruption,
            "severity": self.severity,
            "salt": self.salt,
            "n_applied": self.n_applied,
            "n_failed": self.n_failed,
        }

    def __repr__(self):
        return f"ImageCorruption({self.corruption!r}, severity={self.severity})"


def severity_ladder(corruption, include_clean=True):
    """Every ``(corruption, severity)`` cell for one family."""
    if corruption == "none":
        return [("none", 0)]
    severities = SEVERITIES if include_clean else SEVERITIES[1:]
    return [(corruption, severity) for severity in severities]


def full_matrix(include_clean=True):
    """Every corruption cell. Clean appears once, not once per family."""
    cells = [("none", 0)] if include_clean else []
    for corruption in CORRUPTIONS:
        if corruption == "none":
            continue
        cells.extend(severity_ladder(corruption, include_clean=False))
    return cells


def describe():
    """The ladder as a human-readable table, for the report and the docs."""
    lines = ["Corruption ladder (severity 0 = byte-identical to clean):"]
    lines.append("  gaussian_blur  sigma as a fraction of image width: " + ", ".join(
        f"{value:g}" for value in BLUR_SIGMA_FRACTION))
    lines.append("  jpeg           quality: " + ", ".join(
        str(value) for value in JPEG_QUALITY))
    lines.append("  gaussian_noise sigma in [0,1] intensity: " + ", ".join(
        f"{value:g}" for value in NOISE_SIGMA))
    return "\n".join(lines)
