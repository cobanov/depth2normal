"""Tests for depth2normal conversion logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth2normal import (
    METHODS,
    convert,
    depth_to_normal,
    load_depth,
    rescale_depth,
)


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

    def test_preserves_16bit_grayscale_depth(self, tmp_path: Path):
        data = np.array([[0, 256, 1024, 4096, 65535]], dtype=np.uint16)
        path = tmp_path / "depth16.png"
        Image.fromarray(data).save(path)

        arr = load_depth(path)

        assert arr.shape == data.shape
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, data.astype(np.float64))


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


class TestDepthRange:
    @staticmethod
    def _surface() -> np.ndarray:
        y, x = np.mgrid[0:64, 0:64]
        return (np.sin(x / 8.0) + np.cos(y / 11.0) + 2) / 4

    def test_scaling_the_input_does_not_change_the_result(self):
        """The same surface at 8-bit and 16-bit scale must agree exactly."""
        surface = self._surface()
        eight_bit = depth_to_normal(surface * 255, strength=3.0)
        sixteen_bit = depth_to_normal(surface * 65535, strength=3.0)
        np.testing.assert_array_equal(eight_bit, sixteen_bit)

    def test_float_zero_to_one_matches_eight_bit(self):
        surface = self._surface()
        np.testing.assert_array_equal(
            depth_to_normal(surface, strength=2.0),
            depth_to_normal(surface * 255, strength=2.0),
        )

    def test_quantized_sixteen_bit_tracks_eight_bit(self):
        """What is left between real 8-bit and 16-bit files is quantization."""
        surface = self._surface()
        difference = np.abs(
            depth_to_normal(np.round(surface * 255), strength=3.0).astype(int)
            - depth_to_normal(np.round(surface * 65535), strength=3.0).astype(int)
        )
        assert difference.mean() < 2

    def test_raw_range_keeps_the_old_behaviour(self):
        depth = np.linspace(0, 65535, 64 * 64).reshape(64, 64)
        scaled = depth_to_normal(depth, depth_range="raw")
        normalized = depth_to_normal(depth, depth_range="auto")
        assert not np.array_equal(scaled, normalized)

    def test_minmax_stretches_a_low_contrast_map(self):
        depth = np.linspace(100, 120, 64 * 64).reshape(64, 64)
        stretched = rescale_depth(depth, "minmax")
        assert stretched.min() == pytest.approx(0.0)
        assert stretched.max() == pytest.approx(255.0)

    def test_invert_flips_the_gradient_sign(self):
        y, x = np.mgrid[0:32, 0:32]
        depth = (x * 4).astype(np.float64)
        straight = depth_to_normal(depth, strength=2.0)
        flipped = depth_to_normal(depth, strength=2.0, invert=True)
        assert straight[16, 16, 0] != flipped[16, 16, 0]
        assert straight[16, 16, 0] + flipped[16, 16, 0] == pytest.approx(254, abs=2)

    def test_rejects_unknown_range(self):
        with pytest.raises(ValueError, match="Unknown range"):
            rescale_depth(np.zeros((8, 8)), "nope")


class TestConvertFiles:
    def test_sixteen_bit_file_matches_eight_bit_file(self, tmp_path: Path):
        """The bug 2.0 fixes, end to end through real files."""
        y, x = np.mgrid[0:64, 0:64]
        surface = (np.sin(x / 9.0) + np.cos(y / 13.0) + 2) / 4

        eight = tmp_path / "depth8.png"
        sixteen = tmp_path / "depth16.png"
        Image.fromarray(np.round(surface * 255).astype(np.uint8), mode="L").save(eight)
        Image.fromarray(np.round(surface * 65535).astype(np.uint16)).save(sixteen)

        convert(eight, tmp_path / "from8.png", strength=3.0)
        convert(sixteen, tmp_path / "from16.png", strength=3.0)

        from_eight = np.asarray(Image.open(tmp_path / "from8.png"), dtype=int)
        from_sixteen = np.asarray(Image.open(tmp_path / "from16.png"), dtype=int)
        assert np.abs(from_eight - from_sixteen).mean() < 2

    def test_convert_passes_range_and_invert_through(self, tmp_path: Path):
        depth_arr = np.random.default_rng(6).integers(0, 255, (32, 32), dtype=np.uint8)
        in_path = tmp_path / "depth.png"
        Image.fromarray(depth_arr, mode="L").save(in_path)

        convert(in_path, tmp_path / "plain.png")
        convert(in_path, tmp_path / "flipped.png", invert=True, depth_range="minmax")

        plain = np.asarray(Image.open(tmp_path / "plain.png"))
        flipped = np.asarray(Image.open(tmp_path / "flipped.png"))
        assert not np.array_equal(plain, flipped)
