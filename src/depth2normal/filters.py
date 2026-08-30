"""Separable gradient filters.

Every gradient method here is separable: a derivative along one axis followed
by a smoothing pass along the other.  Keeping the kernels as 1-D taps means the
NumPy path and the exported ONNX graph are built from a single definition, so
the two cannot drift apart.

Padding is ``reflect`` in the NumPy sense (``a b c | b a``), which is the same
border rule as the ONNX ``Pad`` operator's ``reflect`` mode.  That is why the
exported model matches this implementation pixel for pixel.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

METHODS = ("gaussian", "sobel", "scharr")
Method = Literal["gaussian", "sobel", "scharr"]

#: How many standard deviations the Gaussian kernel covers before it is cut.
GAUSSIAN_TRUNCATE = 4.0

# Normalised so every method estimates the actual derivative: the smoothing
# taps sum to 1 and the derivative taps are a central difference.  A ramp of
# one unit per pixel then reads as 1 whichever method is used, which is what
# lets `strength` mean the same thing across methods and lets the perspective
# form work in metric units.
_SOBEL_DERIVATIVE = np.array([-1.0, 0.0, 1.0]) / 2.0
_SOBEL_SMOOTHING = np.array([1.0, 2.0, 1.0]) / 4.0
_SCHARR_SMOOTHING = np.array([3.0, 10.0, 3.0]) / 16.0


def _gaussian_taps(sigma: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First-derivative-of-Gaussian and Gaussian taps for *sigma*."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = int(GAUSSIAN_TRUNCATE * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    smoothing = np.exp(-0.5 / (sigma * sigma) * x * x)
    smoothing /= smoothing.sum()
    derivative = x / (sigma * sigma) * smoothing
    return derivative, smoothing


def gradient_taps(
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``(derivative, smoothing)`` taps for a gradient method.

    Args:
        method: One of :data:`METHODS`.
        sigma: Gaussian kernel width.  Ignored unless *method* is ``"gaussian"``.

    Returns:
        Two odd-length 1-D arrays of the same length.  A horizontal gradient is
        the derivative taps applied along x and the smoothing taps along y; a
        vertical gradient is the same pair with the axes swapped.

    Raises:
        ValueError: If *method* is unknown or *sigma* is not positive.
    """
    if method == "gaussian":
        return _gaussian_taps(sigma)
    if method == "sobel":
        return _SOBEL_DERIVATIVE, _SOBEL_SMOOTHING
    if method == "scharr":
        return _SOBEL_DERIVATIVE, _SCHARR_SMOOTHING
    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def correlate1d(
    image: NDArray[np.float64],
    taps: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    """Correlate *image* with 1-D *taps* along *axis*, reflecting at the border."""
    radius = len(taps) // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="reflect")

    out = np.zeros_like(image, dtype=np.float64)
    for offset, weight in enumerate(taps):
        if weight == 0.0:
            continue
        window = [slice(None)] * image.ndim
        window[axis] = slice(offset, offset + image.shape[axis])
        out += weight * padded[tuple(window)]
    return out


def gradients(
    depth: NDArray[np.float64],
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate ``(dx, dy)`` surface gradients of a 2-D depth array."""
    derivative, smoothing = gradient_taps(method, sigma)
    dx = correlate1d(correlate1d(depth, derivative, axis=1), smoothing, axis=0)
    dy = correlate1d(correlate1d(depth, derivative, axis=0), smoothing, axis=1)
    return dx, dy


def kernel_2d(
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the equivalent 2-D correlation kernels ``(kx, ky)``.

    The separable form is what the NumPy path runs; this outer-product form is
    what the ONNX export bakes into two ``Conv`` nodes.
    """
    derivative, smoothing = gradient_taps(method, sigma)
    return np.outer(smoothing, derivative), np.outer(derivative, smoothing)
