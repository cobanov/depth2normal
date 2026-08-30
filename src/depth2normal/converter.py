"""Core depth-to-normal-map conversion logic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from depth2normal.filters import METHODS, Method, gradients

#: Depth range the gradients are computed on.  Fixing it here is what makes
#: ``strength`` mean the same thing for an 8-bit and a 16-bit copy of the same
#: depth map.
DEPTH_SCALE = 255.0

RANGES = ("auto", "minmax", "raw")
Range = Literal["auto", "minmax", "raw"]

__all__ = [
    "DEPTH_SCALE",
    "METHODS",
    "RANGES",
    "convert",
    "depth_to_normal",
    "load_depth",
    "rescale_depth",
]


def rescale_depth(
    depth: NDArray[np.floating],
    depth_range: Range = "auto",
    invert: bool = False,
) -> NDArray[np.float64]:
    """Bring a depth array onto the fixed 0-255 working range.

    Args:
        depth: 2-D depth array in any numeric range.
        depth_range: How to get there.  ``"auto"`` picks a divisor from the
            value range (0-1 float, 8-bit, or 16-bit), which keeps the relative
            contrast of the source and stays consistent across the frames of a
            sequence.  ``"minmax"`` stretches each image to the full range,
            which maximises relief but makes every image its own reference.
            ``"raw"`` leaves the values untouched.
        invert: Flip near and far.  Use it when bright means far in your depth
            map rather than near.

    Returns:
        A float64 array, normally spanning 0-255.
    """
    if depth_range not in RANGES:
        raise ValueError(f"Unknown range {depth_range!r}, choose from {RANGES}")

    values = depth.astype(np.float64)
    if invert:
        values = values.max() - values
    if depth_range == "raw":
        return values

    low, high = float(values.min()), float(values.max())
    if depth_range == "minmax":
        span = high - low
        return (values - low) / span * DEPTH_SCALE if span else np.zeros_like(values)

    if high <= 1.0:
        return values * DEPTH_SCALE
    if high <= 255.0:
        return values
    if high <= 65535.0:
        return values * (DEPTH_SCALE / 65535.0)
    return rescale_depth(values, "minmax")


def _perspective_normals(
    depth: NDArray[np.float64],
    dx: NDArray[np.float64],
    dy: NDArray[np.float64],
    focal: float,
) -> NDArray[np.float64]:
    """Surface normals of the 3-D points a pinhole camera would see.

    Each pixel is unprojected to ``P = ((u - cx) Z / f, (v - cy) Z / f, Z)``
    and the normal is the cross product of the two surface tangents.  As the
    focal length grows this converges on the height field form, with the
    relief scaled by ``f / Z``, which is what makes a metric depth map produce
    the same surface whatever its distance from the camera.
    """
    height, width = depth.shape
    v, u = np.mgrid[0:height, 0:width]
    x = u - (width - 1) / 2.0
    y = v - (height - 1) / 2.0

    du = np.dstack(((depth + x * dx) / focal, y * dx / focal, dx))
    dv = np.dstack((x * dy / focal, (depth + y * dy) / focal, dy))
    return np.cross(du, dv)


def depth_to_normal(
    depth: NDArray[np.floating],
    strength: float = 1.0,
    method: Method = "gaussian",
    sigma: float = 1.0,
    depth_range: Range = "auto",
    invert: bool = False,
    focal: float | None = None,
) -> NDArray[np.uint8]:
    """Convert a 2-D depth array to an RGB normal map.

    Args:
        depth: Grayscale depth image as a 2-D array.
        strength: Multiplier applied to the surface gradients.  Higher values
            produce more pronounced normals.
        method: Gradient algorithm -- ``"gaussian"`` (smooth, best quality),
            ``"sobel"`` (fast, sharp), or ``"scharr"`` (better rotational
            accuracy than Sobel).
        sigma: Standard deviation for the Gaussian derivative kernel.  Higher
            values produce smoother normals at the cost of fine detail.  Only
            used when *method* is ``"gaussian"``.
        depth_range: Input scaling, see :func:`rescale_depth`.
        invert: Flip near and far before computing gradients.
        focal: Focal length in pixels.  Given one, the depth is treated as
            metric distance from a pinhole camera at the image centre and the
            normals come from the unprojected 3-D surface, which is what a
            reconstruction wants.  Left out, the depth is treated as a height
            field, which is what a shading normal map wants.  Use it with
            ``depth_range="raw"`` so the metric values survive.

    Returns:
        An (H, W, 3) uint8 RGB array where each pixel encodes the surface
        normal mapped to the [0, 255] range.

    Raises:
        ValueError: If *depth* is not 2-D or *method* is unknown.
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth array, got shape {depth.shape}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")

    values = rescale_depth(depth, depth_range, invert)
    dx, dy = gradients(values, method, sigma)
    dx *= strength
    dy *= strength

    if focal is None:
        # A height field: the z component is a constant 1, so the length can
        # never reach zero.
        normal = np.dstack((-dx, -dy, np.ones_like(dx)))
    else:
        if focal <= 0:
            raise ValueError(f"focal must be positive, got {focal}")
        normal = _perspective_normals(values, dx, dy, focal)

    length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= np.where(length == 0, 1, length)

    return ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)


def load_depth(path: str | Path) -> NDArray[np.floating]:
    """Load an image file as a 2-D float64 depth array.

    Preserve native single-channel grayscale depth formats such as 8-bit,
    16-bit, and floating-point images. Multichannel images are converted
    to grayscale.
    """
    with Image.open(path) as img:
        if len(img.getbands()) != 1 or img.mode == "P":
            img = img.convert("L")
        return np.asarray(img, dtype=np.float64)


def convert(
    input_path: str | Path,
    output_path: str | Path = "normal_map.png",
    strength: float = 1.0,
    method: Method = "gaussian",
    sigma: float = 1.0,
    depth_range: Range = "auto",
    invert: bool = False,
    focal: float | None = None,
) -> None:
    """Convert a depth map image file to a normal map image file.

    Args:
        input_path: Path to the source depth map image.
        output_path: Destination path for the generated normal map.
            Defaults to ``"normal_map.png"``.
        strength: Gradient multiplier (see :func:`depth_to_normal`).
        method: Gradient algorithm (see :func:`depth_to_normal`).
        sigma: Gaussian sigma (see :func:`depth_to_normal`).
        depth_range: Input scaling (see :func:`rescale_depth`).
        invert: Flip near and far (see :func:`rescale_depth`).
        focal: Focal length in pixels (see :func:`depth_to_normal`).
    """
    depth = load_depth(input_path)
    normal = depth_to_normal(
        depth,
        strength=strength,
        method=method,
        sigma=sigma,
        depth_range=depth_range,
        invert=invert,
        focal=focal,
    )
    Image.fromarray(normal, mode="RGB").save(output_path)
