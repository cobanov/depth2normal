#!/usr/bin/env python3
"""Export the ONNX models the browser demo loads.

Strength is a graph input, so one model covers every strength.  The gradient
kernel is not, so there is one model per method and per Gaussian sigma.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from depth2normal.onnx_export import export_onnx  # noqa: E402

VARIANTS: tuple[tuple[str, float], ...] = (
    ("gaussian", 0.5),
    ("gaussian", 1.0),
    ("gaussian", 2.0),
    ("gaussian", 3.0),
    ("sobel", 1.0),
    ("scharr", 1.0),
)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "web" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, sigma in VARIANTS:
        name = f"{method}-{sigma:g}.onnx" if method == "gaussian" else f"{method}.onnx"
        path = export_onnx(out_dir / name, method=method, sigma=sigma)
        size_kb = path.stat().st_size / 1024
        print(f"{path.relative_to(out_dir.parent.parent)}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
