"""Tests for depth estimation, none of which download the model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from depth2normal import estimate as estimate_module
from depth2normal.estimate import _input_size, cache_dir, estimate_depth, model_path


class TestInputSize:
    def test_sides_are_multiples_of_fourteen(self):
        for height, width in ((480, 640), (1080, 1920), (37, 41)):
            new_height, new_width = _input_size(height, width)
            assert new_height % 14 == 0
            assert new_width % 14 == 0

    def test_aspect_ratio_is_kept(self):
        new_height, new_width = _input_size(480, 640)
        assert new_height == 518
        assert new_width == pytest.approx(518 * 640 / 480, abs=14)

    def test_square_input_lands_on_the_model_size(self):
        assert _input_size(1000, 1000) == (518, 518)


class TestCache:
    def test_environment_overrides_the_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEPTH2NORMAL_CACHE", str(tmp_path))
        assert cache_dir() == tmp_path
        assert model_path().parent == tmp_path

    def test_falls_back_to_xdg(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEPTH2NORMAL_CACHE", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_dir() == tmp_path / "depth2normal"


def test_uses_the_cached_model_and_returns_the_input_size(tmp_path, monkeypatch):
    """The session is stubbed, so this exercises pre- and postprocessing only."""
    photo = tmp_path / "photo.png"
    Image.fromarray(
        np.random.default_rng(5).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    ).save(photo)

    captured = {}

    class FakeSession:
        def run(self, _outputs, inputs):
            captured["shape"] = inputs["pixel_values"].shape
            height, width = inputs["pixel_values"].shape[2:]
            ramp = np.linspace(0, 5, height * width, dtype=np.float32)
            return [ramp.reshape(1, height, width)]

    monkeypatch.setattr(estimate_module, "ensure_model", lambda *a, **k: tmp_path)
    monkeypatch.setattr(estimate_module, "_load_session", lambda: FakeSession())

    depth = estimate_depth(photo)

    assert captured["shape"] == (1, 3, *_input_size(60, 80))
    assert depth.shape == (60, 80)
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(255.0)
