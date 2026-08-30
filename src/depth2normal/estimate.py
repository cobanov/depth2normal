"""Estimate a depth map from an ordinary photograph.

The hard half of a normal map is the depth map, and most people do not have
one.  This module fills that gap with Depth Anything V2 Small, an Apache-2.0
model run through ONNX Runtime, so a photo can go in and a normal map can come
out without any other tool.

The model is roughly 99 MB.  It is downloaded once, checksummed, and cached.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MODEL_NAME = "depth-anything-v2-small"
MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/"
    "resolve/main/onnx/model.onnx"
)
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
MODEL_BYTES = 99_207_628

#: Preprocessing constants, from the model's own ``preprocessor_config.json``.
INPUT_SIZE = 518
SIZE_MULTIPLE = 14
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MISSING_RUNTIME = (
    "Depth estimation needs ONNX Runtime: pip install 'depth2normal[estimate]'"
)

_session: Any = None


def cache_dir() -> Path:
    """Directory the model is cached in.

    ``DEPTH2NORMAL_CACHE`` overrides it, otherwise ``XDG_CACHE_HOME`` or
    ``~/.cache`` is used.
    """
    override = os.environ.get("DEPTH2NORMAL_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "depth2normal"


def model_path() -> Path:
    """Where the depth model lives once downloaded."""
    return cache_dir() / f"{MODEL_NAME}.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(progress: Callable[[int, int], None] | None = None) -> Path:
    """Return the cached model path, downloading it the first time.

    Args:
        progress: Optional ``(downloaded_bytes, total_bytes)`` callback.

    Raises:
        OSError: If the download does not match the expected checksum.
    """
    path = model_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MODEL_URL) as response:  # noqa: S310 - fixed URL
        total = int(response.headers.get("Content-Length") or MODEL_BYTES)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                downloaded = 0
                while block := response.read(1 << 20):
                    handle.write(block)
                    downloaded += len(block)
                    if progress:
                        progress(downloaded, total)
            digest = _sha256(tmp)
            if digest != MODEL_SHA256:
                raise OSError(
                    f"Downloaded model checksum {digest} does not match "
                    f"the expected {MODEL_SHA256}"
                )
            shutil.move(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def _load_session() -> Any:
    global _session
    if _session is None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise ModuleNotFoundError(_MISSING_RUNTIME) from exc
        _session = ort.InferenceSession(
            str(ensure_model()), providers=["CPUExecutionProvider"]
        )
    return _session


def _input_size(height: int, width: int) -> tuple[int, int]:
    """Target size the model expects: aspect kept, both sides multiples of 14."""
    scale_h = INPUT_SIZE / height
    scale_w = INPUT_SIZE / width
    scale = scale_w if abs(1 - scale_w) < abs(1 - scale_h) else scale_h

    def snap(value: float) -> int:
        return max(SIZE_MULTIPLE, int(round(value / SIZE_MULTIPLE) * SIZE_MULTIPLE))

    return snap(scale * height), snap(scale * width)


def estimate_depth(
    image_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Estimate a depth map from a photograph.

    Args:
        image_path: Path to any image ONNX Runtime's dependencies can open.
        progress: Optional download progress callback, see :func:`ensure_model`.

    Returns:
        A 2-D float64 array on the 0-255 range, at the input image's own
        resolution, where bright means near.
    """
    ensure_model(progress)
    session = _load_session()

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        target_h, target_w = _input_size(height, width)
        resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    batch = pixels.transpose(2, 0, 1)[None]

    predicted = session.run(None, {"pixel_values": batch})[0][0]

    depth = Image.fromarray(predicted.astype(np.float32), mode="F")
    depth = depth.resize((width, height), Image.Resampling.BICUBIC)
    values = np.asarray(depth, dtype=np.float64)

    low, high = float(values.min()), float(values.max())
    span = high - low
    if not span:
        return np.zeros_like(values)
    return (values - low) / span * 255.0
