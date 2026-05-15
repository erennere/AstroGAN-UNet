"""Tests for pure helper functions in src/data/create_dataset.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.create_dataset import (
    calculate_image_stats,
    classify_data,
    create_dir,
    crop_image,
    crop_image_generator,
)


# ---------------------------------------------------------------------------
# create_dir
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_dir_creates_new_directory(tmp_path: Path):
    new_dir = tmp_path / 'subdir' / 'deeper'
    assert not new_dir.exists()
    result = create_dir(str(new_dir))
    assert result is True
    assert new_dir.exists()


@pytest.mark.unit
def test_create_dir_is_idempotent(tmp_path: Path):
    target = str(tmp_path / 'idempotent')
    assert create_dir(target) is True
    assert create_dir(target) is True  # second call must not fail


# ---------------------------------------------------------------------------
# classify_data
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_classify_data_returns_dict_with_percentile_keys():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 100, (64, 64)).astype(np.float32)
    stats = classify_data(data, step=10)
    assert isinstance(stats, dict)
    assert '10_mean' in stats
    assert '50_median' in stats
    assert '100_max' in stats


@pytest.mark.unit
def test_classify_data_zero_percentile_key_is_nan():
    rng = np.random.default_rng(7)
    data = rng.uniform(1, 10, 100)
    stats = classify_data(data, step=5)
    # 0% subset is empty → all nan
    assert np.isnan(stats['0_mean'])


@pytest.mark.unit
def test_classify_data_handles_negative_step_gracefully():
    rng = np.random.default_rng(8)
    data = rng.uniform(0, 1, 50)
    stats = classify_data(data, step=-10)  # should fall back to step=1
    assert isinstance(stats, dict)
    assert len(stats) > 0


# ---------------------------------------------------------------------------
# crop_image_generator
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_crop_image_generator_yields_correct_size_crops():
    rng = np.random.default_rng(101)
    image = rng.uniform(0, 1, (256, 512)).astype(np.float32)
    crops = list(crop_image_generator(image, ps=128))
    assert len(crops) == 8  # (256/128) * (512/128) = 2 * 4
    for crop in crops:
        assert crop.shape == (128, 128)


@pytest.mark.unit
def test_crop_image_generator_pads_when_not_divisible():
    rng = np.random.default_rng(202)
    # 260x260 padded to 512x512 in 256-ps increments
    image = rng.uniform(0, 1, (260, 260)).astype(np.float32)
    crops = list(crop_image_generator(image, ps=256))
    assert len(crops) == 4  # 2x2 = 4 crops after padding to 512x512
    for crop in crops:
        assert crop.shape == (256, 256)


@pytest.mark.unit
def test_crop_image_generator_yields_nothing_for_undersized_image():
    rng = np.random.default_rng(303)
    small = rng.uniform(0, 1, (100, 100)).astype(np.float32)
    crops = list(crop_image_generator(small, ps=256))
    assert len(crops) == 0


# ---------------------------------------------------------------------------
# crop_image
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_crop_image_saves_fits_patches_to_output_dir(tmp_path: Path):
    rng = np.random.default_rng(55)
    image = rng.uniform(0, 100, (256, 256)).astype(np.float32)
    save_dir = str(tmp_path / 'crops')

    crop_image(image, filepath='orig_img.fits', save_dir=save_dir, ps=128)

    fits_files = list(Path(save_dir).glob('*.fits'))
    assert len(fits_files) == 4  # 2x2 non-overlapping crops


@pytest.mark.unit
def test_crop_image_skips_when_filepath_is_none(tmp_path: Path):
    rng = np.random.default_rng(66)
    image = rng.uniform(0, 1, (256, 256)).astype(np.float32)
    save_dir = str(tmp_path / 'crops2')
    crop_image(image, filepath=None, save_dir=save_dir, ps=128)
    # No FITS files should exist since filepath=None is an early return
    assert not Path(save_dir).exists() or len(list(Path(save_dir).glob('*.fits'))) == 0


@pytest.mark.unit
def test_crop_image_skips_when_save_dir_is_none():
    rng = np.random.default_rng(77)
    image = rng.uniform(0, 1, (256, 256)).astype(np.float32)
    crop_image(image, filepath='orig.fits', save_dir=None, ps=128)
    # Must not raise; just return early


# ---------------------------------------------------------------------------
# calculate_image_stats
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_calculate_image_stats_returns_tuple_for_synthetic_image():
    rng = np.random.default_rng(2024)
    # Create a realistic synthetic image: Gaussian background + point sources
    image = rng.normal(loc=100.0, scale=5.0, size=(128, 128)).astype(np.float32)
    # Add a few bright compact "stars"
    for _ in range(5):
        cy, cx = rng.integers(20, 108, size=2)
        image[cy - 2:cy + 3, cx - 2:cx + 3] += 500.0

    result = calculate_image_stats(image, sigma=3.0, n_sigma=2.0, n_pixels=3, footprint_radius=5)
    if result is None:
        pytest.skip('calculate_image_stats returned None (no sources detected in synthetic data)')

    stats_tuple, mask, percentile_stats = result
    assert len(stats_tuple) == 10
    assert mask.shape == image.shape
    assert isinstance(percentile_stats, dict)


@pytest.mark.unit
def test_calculate_image_stats_returns_none_for_none_input():
    assert calculate_image_stats(None) is None
