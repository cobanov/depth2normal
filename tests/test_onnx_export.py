"""The exported ONNX graph must agree with the NumPy path."""

from __future__ import annotations

import numpy as np
import pytest

from depth2normal import depth_to_normal, export_onnx, rescale_depth
from depth2normal.filters import METHODS

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _session(tmp_path, method="gaussian", sigma=1.0):
    path = export_onnx(tmp_path / f"{method}.onnx", method=method, sigma=sigma)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(session, depth, strength):
    inputs = {
        "depth": rescale_depth(depth).astype(np.float32)[None, None],
        "strength": np.array(strength, dtype=np.float32),
    }
    return session.run(None, inputs)[0][0]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("strength", [1.0, 4.0])
def test_matches_the_numpy_path(tmp_path, method, strength):
    rng = np.random.default_rng(7)
    depth = rng.random((61, 97)) * 255

    expected = depth_to_normal(depth, strength=strength, method=method)
    actual = _run(_session(tmp_path, method), depth, strength)

    difference = np.abs(expected.astype(int) - actual.astype(int))
    # float32 in the graph against float64 in NumPy, so a pixel can land on
    # the other side of a rounding boundary.  Never further than that.
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.05


def test_sigma_is_baked_in(tmp_path):
    rng = np.random.default_rng(8)
    depth = rng.random((48, 48)) * 255
    session = _session(tmp_path, "gaussian", sigma=2.5)

    expected = depth_to_normal(depth, method="gaussian", sigma=2.5)
    actual = _run(session, depth, 1.0)
    assert np.abs(expected.astype(int) - actual.astype(int)).max() <= 1


def test_one_model_handles_any_size(tmp_path):
    session = _session(tmp_path)
    for shape in ((32, 32), (17, 129)):
        output = _run(session, np.zeros(shape), 1.0)
        assert output.shape == (*shape, 3)


def test_output_is_nhwc_uint8(tmp_path):
    output = _run(_session(tmp_path), np.zeros((12, 20)), 1.0)
    assert output.dtype == np.uint8
    assert output.shape == (12, 20, 3)
    # A flat depth map points straight at the viewer.
    assert output[6, 10].tolist() == [127, 127, 255]


def test_records_its_own_settings(tmp_path):
    path = export_onnx(tmp_path / "meta.onnx", method="scharr", sigma=1.5)
    model = onnx.load(path)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata["method"] == "scharr"
    assert metadata["sigma"] == "1.5"
    assert metadata["depth_range"] == "0-255"


def test_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown method"):
        export_onnx(tmp_path / "bad.onnx", method="nope")
