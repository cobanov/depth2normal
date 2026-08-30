"""Tests for the command-line interface."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from depth2normal.cli import cli


@pytest.fixture
def depth_file(tmp_path):
    data = np.random.default_rng(4).integers(0, 255, (32, 48), dtype=np.uint8)
    path = tmp_path / "depth.png"
    Image.fromarray(data, mode="L").save(path)
    return path


def test_bare_path_still_converts(tmp_path, depth_file):
    out = tmp_path / "normal.png"
    result = CliRunner().invoke(cli, [str(depth_file), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_convert_subcommand(tmp_path, depth_file):
    out = tmp_path / "normal.png"
    result = CliRunner().invoke(cli, ["convert", str(depth_file), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert Image.open(out).size == (48, 32)


def test_invert_changes_the_output(tmp_path, depth_file):
    runner = CliRunner()
    straight = tmp_path / "straight.png"
    flipped = tmp_path / "flipped.png"
    runner.invoke(cli, [str(depth_file), "-o", str(straight)])
    runner.invoke(cli, [str(depth_file), "-o", str(flipped), "--invert"])
    assert straight.read_bytes() != flipped.read_bytes()


def test_save_depth_needs_estimate(tmp_path, depth_file):
    result = CliRunner().invoke(
        cli, [str(depth_file), "--save-depth", str(tmp_path / "d.png")]
    )
    assert result.exit_code != 0
    assert "--save-depth only applies" in result.output


def test_export_writes_a_model(tmp_path):
    pytest.importorskip("onnx")
    out = tmp_path / "model.onnx"
    result = CliRunner().invoke(cli, ["export", "-o", str(out), "-m", "sobel"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "normal_map uint8" in result.output


def test_version_flag():
    """The CLI reports the installed version, whatever it is."""
    from importlib.metadata import version

    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert version("depth2normal") in result.output
