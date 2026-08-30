#!/usr/bin/env python3
"""Time the conversion on NumPy and on ONNX Runtime, at several image sizes.

The numbers in the README come from this script:

    python tools/benchmark.py
    python tools/benchmark.py --photo photo.jpg   # also times depth estimation
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from depth2normal import depth_to_normal, export_onnx, rescale_depth  # noqa: E402

DEFAULT_SIZES = ("1920x1080", "1080x1920", "4096x2304", "2304x4096", "8192x4608")


def synthetic_depth(width: int, height: int) -> np.ndarray:
    """A smooth, deterministic depth map, so timings are comparable."""
    y, x = np.mgrid[0:height, 0:width]
    surface = (np.sin(x / 97.0) + np.cos(y / 61.0) + np.sin((x + y) / 143.0) + 3) / 6
    return surface * 255.0


def median_ms(work: Callable[[], object], repeat: int) -> float:
    work()
    timings = []
    for _ in range(repeat):
        started = time.perf_counter()
        work()
        timings.append(time.perf_counter() - started)
    return sorted(timings)[len(timings) // 2] * 1000


def providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [name for name in wanted if name in available]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(DEFAULT_SIZES))
    parser.add_argument("--method", default="gaussian")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--photo", help="Also time depth estimation on this image")
    args = parser.parse_args()

    import onnxruntime as ort

    print(f"{platform.platform()}  python {platform.python_version()}")
    print(f"numpy {np.__version__}  onnxruntime {ort.__version__}")
    print(f"method {args.method}, sigma {args.sigma}, median of {args.repeat}\n")

    model = export_onnx(
        Path(__file__).with_name("_benchmark.onnx"), args.method, args.sigma
    )
    sessions = {
        name: ort.InferenceSession(str(model), providers=[name]) for name in providers()
    }

    columns = ["NumPy", *(name.replace("ExecutionProvider", "") for name in sessions)]
    print("| Size | " + " | ".join(columns) + " |")
    print("| --- |" + " --- |" * len(columns))

    for size in args.sizes.split(","):
        width, height = (int(part) for part in size.lower().split("x"))
        depth = synthetic_depth(width, height)
        convert = partial(depth_to_normal, depth, method=args.method, sigma=args.sigma)
        row = [f"{median_ms(convert, args.repeat):.0f} ms"]

        feeds = {
            "depth": rescale_depth(depth).astype(np.float32)[None, None],
            "strength": np.array(1.0, dtype=np.float32),
        }
        for session in sessions.values():
            run = partial(session.run, None, feeds)
            row.append(f"{median_ms(run, args.repeat):.0f} ms")
        print(f"| {width} x {height} | " + " | ".join(row) + " |")

    if args.photo:
        from depth2normal.estimate import ensure_model, estimate_depth

        ensure_model()
        estimate = partial(estimate_depth, args.photo)
        print()
        elapsed = median_ms(estimate, max(3, args.repeat // 2))
        print(f"depth estimation ({args.photo}): {elapsed:.0f} ms")

    model.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
