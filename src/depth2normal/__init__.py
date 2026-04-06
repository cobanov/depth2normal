"""depth2normal -- convert depth maps to normal maps."""

from depth2normal.converter import METHODS, convert, depth_to_normal, load_depth

__version__ = "1.0.0"
__all__ = ["METHODS", "convert", "depth_to_normal", "load_depth", "__version__"]
