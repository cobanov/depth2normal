"""Export the conversion as a standalone ONNX graph.

The maths in :mod:`depth2normal.converter` is a pair of convolutions and a
normalisation, which is exactly what an ONNX graph is good at.  Exporting it
means the conversion runs wherever ONNX Runtime runs (a browser, C#, Unity, a
ComfyUI node) with no Python involved.

The graph takes depth already on the 0-255 working range, the same range
:func:`depth2normal.rescale_depth` produces, and a ``strength`` scalar that can
be changed per call without re-exporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from depth2normal.filters import METHODS, Method, kernel_2d

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import onnx

DEFAULT_OPSET = 17

_MISSING_ONNX = (
    "The ONNX export needs the 'onnx' package: pip install 'depth2normal[onnx]'"
)


def _require_onnx() -> Any:
    try:
        import onnx
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(_MISSING_ONNX) from exc
    return onnx


def build_model(
    method: Method = "gaussian",
    sigma: float = 1.0,
    opset: int = DEFAULT_OPSET,
) -> onnx.ModelProto:
    """Build the depth-to-normal graph for one gradient method.

    Args:
        method: Gradient algorithm baked into the convolution weights.
        sigma: Gaussian sigma, baked in as well (only used for ``"gaussian"``).
        opset: ONNX opset to target.

    Returns:
        A checked ``ModelProto`` with inputs ``depth`` (float32, NCHW, one
        channel, dynamic height and width) and ``strength`` (float32 scalar),
        and output ``normal`` (uint8, NHWC, three channels).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")

    onnx = _require_onnx()
    from onnx import TensorProto, helper, numpy_helper

    kx, ky = kernel_2d(method, sigma)
    radius = kx.shape[0] // 2

    def const(name: str, array: np.ndarray) -> Any:
        return numpy_helper.from_array(np.ascontiguousarray(array), name)

    initializers = [
        const("kernel_x", kx.astype(np.float32)[None, None]),
        const("kernel_y", ky.astype(np.float32)[None, None]),
        const("pads", np.array([0, 0, radius, radius] * 2, dtype=np.int64)),
        const("one", np.array(1.0, dtype=np.float32)),
        const("half_range", np.array(127.5, dtype=np.float32)),
        const("floor", np.array(0.0, dtype=np.float32)),
        const("ceiling", np.array(255.0, dtype=np.float32)),
    ]

    nodes = [
        helper.make_node("Pad", ["depth", "pads"], ["padded"], mode="reflect"),
        helper.make_node("Conv", ["padded", "kernel_x"], ["dx"]),
        helper.make_node("Conv", ["padded", "kernel_y"], ["dy"]),
        # Negating the strength folds the minus sign of (-dx, -dy, 1) into the
        # same multiply.
        helper.make_node("Neg", ["strength"], ["negated"]),
        helper.make_node("Mul", ["dx", "negated"], ["nx"]),
        helper.make_node("Mul", ["dy", "negated"], ["ny"]),
        helper.make_node("Shape", ["dx"], ["shape"]),
        helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["nz"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0]),
        ),
        helper.make_node("Concat", ["nx", "ny", "nz"], ["normal"], axis=1),
        # The z component is a constant 1, so the length is never zero.
        helper.make_node("ReduceL2", ["normal"], ["length"], axes=[1], keepdims=1),
        helper.make_node("Div", ["normal", "length"], ["unit"]),
        helper.make_node("Add", ["unit", "one"], ["shifted"]),
        helper.make_node("Mul", ["shifted", "half_range"], ["scaled"]),
        helper.make_node("Clip", ["scaled", "floor", "ceiling"], ["clipped"]),
        helper.make_node("Transpose", ["clipped"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Cast", ["nhwc"], ["normal_map"], to=TensorProto.UINT8),
    ]

    graph = helper.make_graph(
        nodes,
        f"depth2normal_{method}",
        inputs=[
            helper.make_tensor_value_info(
                "depth", TensorProto.FLOAT, ["batch", 1, "height", "width"]
            ),
            helper.make_tensor_value_info("strength", TensorProto.FLOAT, []),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "normal_map", TensorProto.UINT8, ["batch", "height", "width", 3]
            )
        ],
        initializer=initializers,
        doc_string=("depth in 0-255 float32 NCHW, normal map out as uint8 NHWC RGB"),
    )

    model = helper.make_model(
        graph,
        producer_name="depth2normal",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.metadata_props.add(key="method", value=method)
    model.metadata_props.add(key="sigma", value=str(sigma))
    model.metadata_props.add(key="depth_range", value="0-255")
    onnx.checker.check_model(model)
    return model


def export_onnx(
    output_path: str | Path = "depth2normal.onnx",
    method: Method = "gaussian",
    sigma: float = 1.0,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Write the graph from :func:`build_model` to *output_path*."""
    onnx = _require_onnx()
    path = Path(output_path)
    onnx.save(build_model(method, sigma, opset), path)
    return path
