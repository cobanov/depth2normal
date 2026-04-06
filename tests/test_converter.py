"""Tests for depth2normal conversion logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth2normal import METHODS, convert, depth_to_normal, load_depth


class TestDepthToNormal:
    def test_output_shape_matches_input(self):
        depth = np.random.rand(64, 128).astype(np.float64) * 255
        result = depth_to_normal(depth)
        assert result.shape == (64, 128, 3)

    def test_output_dtype_is_uint8(self):
        depth = np.random.rand(32, 32).astype(np.float64) * 255
        result = depth_to_normal(depth)
        assert result.dtype == np.uint8

    def test_flat_depth_produces_upward_normals(self):
        depth = np.ones((50, 50), dtype=np.float64) * 128
        result = depth_to_normal(depth)
        center = result[25, 25]
        assert center[0] == pytest.approx(127, abs=2)
        assert center[1] == pytest.approx(127, abs=2)
        assert center[2] == pytest.approx(255, abs=2)

    def test_strength_scales_gradients(self):
        depth = np.random.rand(64, 64).astype(np.float64) * 255
        weak = depth_to_normal(depth, strength=0.5)
        strong = depth_to_normal(depth, strength=5.0)
        weak_var = np.var(weak[:, :, :2].astype(float))
        strong_var = np.var(strong[:, :, :2].astype(float))
        assert strong_var > weak_var

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            depth_to_normal(np.zeros((10, 10, 3)))

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            depth_to_normal(np.zeros((10, 10)), method="invalid")

    def test_pixel_values_in_valid_range(self):
        depth = np.random.rand(100, 100).astype(np.float64) * 255
        result = depth_to_normal(depth)
        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("method", METHODS)
    def test_all_methods_produce_valid_output(self, method):
        depth = np.random.rand(64, 64).astype(np.float64) * 255
        result = depth_to_normal(depth, method=method)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("method", METHODS)
    def test_flat_depth_all_methods(self, method):
        depth = np.ones((50, 50), dtype=np.float64) * 128
        result = depth_to_normal(depth, method=method)
        center = result[25, 25]
        assert center[0] == pytest.approx(127, abs=2)
        assert center[1] == pytest.approx(127, abs=2)
        assert center[2] == pytest.approx(255, abs=2)

    def test_gaussian_sigma_controls_smoothness(self):
        depth = np.random.rand(64, 64).astype(np.float64) * 255
        sharp = depth_to_normal(depth, method="gaussian", sigma=0.5)
        smooth = depth_to_normal(depth, method="gaussian", sigma=3.0)
        sharp_var = np.var(sharp[:, :, :2].astype(float))
        smooth_var = np.var(smooth[:, :, :2].astype(float))
        assert sharp_var > smooth_var


class TestLoadDepth:
    def test_loads_grayscale_png(self, tmp_path: Path):
        data = np.random.randint(0, 255, (48, 64), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        path = tmp_path / "depth.png"
        img.save(path)

        arr = load_depth(path)
        assert arr.shape == (48, 64)
        assert arr.dtype == np.float64

    def test_converts_rgb_to_grayscale(self, tmp_path: Path):
        data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(data, mode="RGB")
        path = tmp_path / "color.png"
        img.save(path)

        arr = load_depth(path)
        assert arr.ndim == 2


class TestConvert:
    def test_file_to_file_roundtrip(self, tmp_path: Path):
        depth_arr = np.random.randint(0, 255, (48, 64), dtype=np.uint8)
        in_path = tmp_path / "depth.png"
        out_path = tmp_path / "normal.png"
        Image.fromarray(depth_arr, mode="L").save(in_path)

        convert(in_path, out_path)

        assert out_path.exists()
        result = np.asarray(Image.open(out_path))
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    def test_default_output_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        depth_arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        in_path = tmp_path / "depth.png"
        Image.fromarray(depth_arr, mode="L").save(in_path)

        convert(str(in_path))

        assert (tmp_path / "normal_map.png").exists()

    @pytest.mark.parametrize("method", METHODS)
    def test_convert_with_all_methods(self, tmp_path: Path, method):
        depth_arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        in_path = tmp_path / "depth.png"
        out_path = tmp_path / f"normal_{method}.png"
        Image.fromarray(depth_arr, mode="L").save(in_path)

        convert(in_path, out_path, method=method)

        assert out_path.exists()
