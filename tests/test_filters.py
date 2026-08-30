"""Tests for the separable gradient filters."""

from __future__ import annotations

import numpy as np
import pytest

from depth2normal.filters import (
    METHODS,
    correlate1d,
    gradient_taps,
    gradients,
    kernel_2d,
)


@pytest.mark.parametrize("method", METHODS)
def test_derivative_taps_sum_to_zero(method):
    derivative, _ = gradient_taps(method)
    assert derivative.sum() == pytest.approx(0.0, abs=1e-12)


def test_gaussian_smoothing_taps_sum_to_one():
    _, smoothing = gradient_taps("gaussian", sigma=1.5)
    assert smoothing.sum() == pytest.approx(1.0)


def test_gaussian_radius_follows_sigma():
    small, _ = gradient_taps("gaussian", sigma=1.0)
    large, _ = gradient_taps("gaussian", sigma=3.0)
    assert len(small) == 9
    assert len(large) == 25


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        gradient_taps("nope")


def test_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="sigma must be positive"):
        gradient_taps("gaussian", sigma=0.0)


@pytest.mark.parametrize("method", METHODS)
def test_flat_image_has_no_gradient(method):
    dx, dy = gradients(np.full((32, 32), 42.0), method)
    assert np.abs(dx).max() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(dy).max() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_gradient_signs_follow_the_slope(method):
    _, x = np.mgrid[0:32, 0:32]
    dx, dy = gradients(x.astype(np.float64), method)
    assert dx[16, 16] > 0
    assert dy[16, 16] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_two_dimensional_kernel_matches_the_separable_pass(method):
    rng = np.random.default_rng(3)
    image = rng.random((24, 29)) * 255
    kx, ky = kernel_2d(method)
    radius = kx.shape[0] // 2
    padded = np.pad(image, radius, mode="reflect")

    windows = np.lib.stride_tricks.sliding_window_view(padded, kx.shape)
    direct_x = np.einsum("ijkl,kl->ij", windows, kx)
    direct_y = np.einsum("ijkl,kl->ij", windows, ky)

    dx, dy = gradients(image, method)
    np.testing.assert_allclose(dx, direct_x, atol=1e-9)
    np.testing.assert_allclose(dy, direct_y, atol=1e-9)


def test_correlate1d_reflects_at_the_border():
    row = np.array([[1.0, 2.0, 3.0, 4.0]])
    smoothed = correlate1d(row, np.array([1.0, 1.0, 1.0]), axis=1)
    # Reflect padding mirrors without repeating the edge: 2 | 1 2 3 4 | 3
    assert smoothed[0, 0] == pytest.approx(2.0 + 1.0 + 2.0)
    assert smoothed[0, -1] == pytest.approx(3.0 + 4.0 + 3.0)
