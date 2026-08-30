"""depth2normal -- convert depth maps to normal maps."""

from depth2normal.converter import (
    METHODS,
    RANGES,
    convert,
    depth_to_normal,
    load_depth,
    rescale_depth,
)
from depth2normal.estimate import estimate_depth
from depth2normal.onnx_export import build_model, export_onnx

__version__ = "2.0.1"
__all__ = [
    "METHODS",
    "RANGES",
    "__version__",
    "build_model",
    "convert",
    "depth_to_normal",
    "estimate_depth",
    "export_onnx",
    "load_depth",
    "rescale_depth",
]
