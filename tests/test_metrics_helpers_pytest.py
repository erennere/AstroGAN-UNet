"""Tests for pure helper functions in src/evaluation/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    aggregate_df,
    calculate_psnr,
    generate_distance_weights,
    generate_gaussian_weights,
    pad_image,
    scale_image,
    sliding_window_generator,
    strip_pad,
)


# ---------------------------------------------------------------------------
# generate_gaussian_weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_generate_gaussian_weights_shape_and_peak():
    ps = (32, 32, 1)
    weights = generate_gaussian_weights(ps, sigma=16)
    assert weights is not None
    assert weights.shape == (32, 32, 1)
    assert np.all(weights > 0)
    # Peak should be at the centre
    centre_value = weights[16, 16, 0]
    corner_value = weights[0, 0, 0]
    assert centre_value > corner_value


@pytest.mark.unit
def test_generate_gaussian_weights_returns_none_for_bad_patch_size():
    assert generate_gaussian_weights(None) is None
    assert generate_gaussian_weights((32, 32)) is None  # only 2 elements
    assert generate_gaussian_weights((0, 0, 0)) is not None  # valid shape tuple


# ---------------------------------------------------------------------------
# generate_distance_weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_generate_distance_weights_shape_and_non_negative():
    ps = (20, 20, 1)
    weights = generate_distance_weights(ps)
    assert weights is not None
    assert weights.shape == (20, 20, 1)
    assert np.all(weights >= 0)
    # Centre should be 1.0 (or close to it)
    assert weights[10, 10, 0] >= weights[0, 0, 0]


@pytest.mark.unit
def test_generate_distance_weights_returns_none_for_bad_input():
    assert generate_distance_weights(None) is None
    assert generate_distance_weights((32,)) is None  # only 1 element


# ---------------------------------------------------------------------------
# pad_image / strip_pad round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pad_and_strip_are_inverse_operations():
    rng = np.random.default_rng(42)
    h, w = 70, 90
    image = rng.uniform(0, 1, (h, w, 1)).astype(np.float32)
    patch_size = (32, 32, 1)
    stride = (16, 16, 1)

    result = pad_image(image, patch_size, stride)
    assert result is not None
    padded, pad_amounts = result
    assert padded.ndim == 3

    restored = strip_pad(padded, *pad_amounts)
    assert restored is not None
    assert restored.shape == image.shape
    assert np.allclose(restored, image)


@pytest.mark.unit
def test_pad_image_returns_none_for_bad_inputs():
    rng = np.random.default_rng(7)
    img = rng.uniform(0, 1, (64, 64, 1)).astype(np.float32)
    assert pad_image(None, (32, 32, 1), (16, 16, 1)) is None
    assert pad_image(img, None, (16, 16, 1)) is None
    assert pad_image(img, (32, 32), (16, 16, 1)) is None  # wrong tuple length


@pytest.mark.unit
def test_strip_pad_returns_none_for_none_image():
    assert strip_pad(None, 0, 0, 0, 0, 0, 0) is None


@pytest.mark.unit
def test_strip_pad_returns_none_when_padding_exceeds_dimensions():
    rng = np.random.default_rng(5)
    small = rng.uniform(0, 1, (4, 4, 1)).astype(np.float32)
    # pad_eq_h * 2 = 10 > 4
    assert strip_pad(small, 0, 0, 0, 0, 5, 0) is None


# ---------------------------------------------------------------------------
# sliding_window_generator
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sliding_window_generator_yields_correct_patches_and_positions():
    rng = np.random.default_rng(99)
    image = rng.uniform(0, 1, (64, 64, 1)).astype(np.float32)
    patch_size = (32, 32, 1)
    stride = (32, 32, 1)

    patches = list(sliding_window_generator(image, patch_size, stride))
    assert len(patches) == 4  # 2x2 grid with no overlap
    for patch, (row, col) in patches:
        assert patch.shape == (32, 32, 1)
        assert 0 <= row < 64
        assert 0 <= col < 64


@pytest.mark.unit
def test_sliding_window_generator_overlap_produces_more_patches():
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 1, (64, 64, 1)).astype(np.float32)
    non_overlap = list(sliding_window_generator(image, (32, 32, 1), (32, 32, 1)))
    overlap = list(sliding_window_generator(image, (32, 32, 1), (16, 16, 1)))
    assert len(overlap) > len(non_overlap)


# ---------------------------------------------------------------------------
# scale_image
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_scale_image_returns_pil_image_and_bounds():
    from PIL import Image
    rng = np.random.default_rng(66)
    image = rng.uniform(0, 100, (32, 32)).astype(np.float32)
    result = scale_image(image)
    assert result is not None
    img, vmin, vmax = result
    assert isinstance(img, Image.Image)
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)


@pytest.mark.unit
def test_scale_image_accepts_single_channel_3d_array():
    from PIL import Image
    rng = np.random.default_rng(77)
    image = rng.uniform(0, 1, (16, 16, 1)).astype(np.float32)
    result = scale_image(image)
    assert result is not None


@pytest.mark.unit
def test_scale_image_returns_none_for_non_array():
    result = scale_image("not an array")
    assert result is None


# ---------------------------------------------------------------------------
# calculate_psnr
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_calculate_psnr_identical_images_returns_inf():
    rng = np.random.default_rng(11)
    img = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    psnr, mse = calculate_psnr(img, img)
    assert psnr == float('inf')
    assert mse == 0.0


@pytest.mark.unit
def test_calculate_psnr_different_images_returns_finite():
    rng = np.random.default_rng(22)
    original = rng.uniform(0, 1, (32, 32)).astype(np.float32)
    noisy = original + rng.uniform(0, 0.1, (32, 32)).astype(np.float32)
    psnr, mse = calculate_psnr(original, noisy)
    assert np.isfinite(psnr)
    assert mse > 0


@pytest.mark.unit
def test_calculate_psnr_returns_none_for_shape_mismatch():
    rng = np.random.default_rng(33)
    a = rng.uniform(0, 1, (8, 8)).astype(np.float32)
    b = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    result = calculate_psnr(a, b)
    assert result == (None, None)


@pytest.mark.unit
def test_calculate_psnr_uniform_image_returns_nan():
    img = np.ones((16, 16), dtype=np.float32)
    psnr, mse = calculate_psnr(img, img * 2)
    assert np.isnan(psnr)


# ---------------------------------------------------------------------------
# aggregate_df
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_aggregate_df_returns_expected_keys_and_types():
    rng = np.random.default_rng(55)
    n = 20
    df = pd.DataFrame({
        'TP': rng.integers(5, 20, n).astype(float),
        'FP': rng.integers(0, 5, n).astype(float),
        'FN': rng.integers(0, 5, n).astype(float),
        'MSE_rec': rng.uniform(0.001, 0.1, n),
        'MSE_noisy': rng.uniform(0.01, 0.2, n),
        'PSNR_L': rng.uniform(1.0, 5.0, n),
        'PSNR_rec': rng.uniform(20, 40, n),
        'PSNR_noisy': rng.uniform(15, 30, n),
        'SSIM_rec': rng.uniform(0.7, 1.0, n),
        'SSIM_noisy': rng.uniform(0.5, 0.9, n),
        'IoU': rng.uniform(0.4, 0.9, n),
    })
    flux_org = rng.uniform(1, 100, n)
    flux_rec = rng.uniform(1, 100, n)
    flux_err_org = rng.uniform(0.1, 5, n)
    flux_err_rec = rng.uniform(0.1, 5, n)

    result = aggregate_df(df, flux_rec, flux_org, flux_err_rec, flux_err_org)

    assert isinstance(result, dict)
    for key in ('TP', 'FP', 'FN', 'Precision', 'Recall', 'F-measure', 'RFE', 'SNR_org', 'SNR_rec',
                 'PSNR_rec', 'PSNR_noisy', 'SSIM_rec', 'SSIM_noisy', 'IoU'):
        assert key in result

    assert 0 <= result['Precision'] <= 1
    assert 0 <= result['Recall'] <= 1


@pytest.mark.unit
def test_aggregate_df_returns_none_for_non_dataframe():
    result = aggregate_df("not a df", None, None, None, None)
    assert result is None
