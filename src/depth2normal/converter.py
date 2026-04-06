"""Core depth-to-normal-map conversion logic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import correlate, gaussian_filter, sobel

METHODS = ("gaussian", "sobel", "scharr")
Method = Literal["gaussian", "sobel", "scharr"]

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
_SCHARR_Y = _SCHARR_X.T


def _gradients_sobel(depth: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    return sobel(depth, axis=1), sobel(depth, axis=0)


def _gradients_scharr(depth: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    return correlate(depth, _SCHARR_X), correlate(depth, _SCHARR_Y)


def _gradients_gaussian(
    depth: NDArray[np.float64],
    sigma: float,
) -> tuple[NDArray, NDArray]:
    dx = gaussian_filter(depth, sigma=sigma, order=[0, 1])
    dy = gaussian_filter(depth, sigma=sigma, order=[1, 0])
    return dx, dy


def depth_to_normal(
    depth: NDArray[np.floating],
    strength: float = 1.0,
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> NDArray[np.uint8]:
    """Convert a 2-D depth array to an RGB normal map.

    Args:
        depth: Grayscale depth image as a 2-D float array.  Gradients are
            computed on the raw values so that the original pixel range
            (e.g. 0-255) drives the normal intensity.
        strength: Multiplier applied to the surface gradients.  Higher
            values produce more pronounced normals.
        method: Gradient algorithm -- ``"gaussian"`` (smooth, best quality),
            ``"sobel"`` (fast, sharp), or ``"scharr"`` (better rotational
            accuracy than Sobel).
        sigma: Standard deviation for the Gaussian derivative kernel.
            Higher values produce smoother normals at the cost of fine
            detail.  Only used when *method* is ``"gaussian"``.

    Returns:
        An (H, W, 3) uint8 RGB array where each pixel encodes the
        surface normal mapped to the [0, 255] range.

    Raises:
        ValueError: If *depth* is not 2-D or *method* is unknown.
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth array, got shape {depth.shape}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")

    depth = depth.astype(np.float64)

    if method == "gaussian":
        dx, dy = _gradients_gaussian(depth, sigma)
    elif method == "scharr":
        dx, dy = _gradients_scharr(depth)
    else:
        dx, dy = _gradients_sobel(depth)

    dx *= strength
    dy *= strength

    normal = np.dstack((-dx, -dy, np.ones_like(dx)))

    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    normal = normal / norm

    normal_uint8 = ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return normal_uint8


def load_depth(path: str | Path) -> NDArray[np.floating]:
    """Load an image file as a 2-D float64 depth array.

    Multichannel images are converted to grayscale.
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float64)


def convert(
    input_path: str | Path,
    output_path: str | Path = "normal_map.png",
    strength: float = 1.0,
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> None:
    """Convert a depth map image file to a normal map image file.

    Args:
        input_path: Path to the source depth map image.
        output_path: Destination path for the generated normal map.
            Defaults to ``"normal_map.png"``.
        strength: Gradient multiplier (see :func:`depth_to_normal`).
        method: Gradient algorithm (see :func:`depth_to_normal`).
        sigma: Gaussian sigma (see :func:`depth_to_normal`).
    """
    depth = load_depth(input_path)
    normal = depth_to_normal(depth, strength=strength, method=method, sigma=sigma)
    Image.fromarray(normal, mode="RGB").save(output_path)
