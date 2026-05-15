"""Tests for remaining compute-heavy helpers in src/evaluation/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    calculate_iou,
    compute_ssim,
    decide_scale,
    get_model_by_modulo,
    get_top_best_models,
)


# ---------------------------------------------------------------------------
# compute_ssim
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_ssim_identical_images_returns_one():
    rng = np.random.default_rng(10)
    img = rng.uniform(0, 100, (64, 64)).astype(np.float32)
    mean_val, ssim_map = compute_ssim(img, img)
    assert mean_val is not None
    assert pytest.approx(mean_val, abs=0.01) == 1.0
    assert ssim_map.shape == img.shape


@pytest.mark.unit
def test_compute_ssim_noisy_returns_less_than_one():
    rng = np.random.default_rng(11)
    original = rng.uniform(0, 100, (64, 64)).astype(np.float32)
    noisy = original + rng.normal(0, 10, (64, 64)).astype(np.float32)
    mean_val, ssim_map = compute_ssim(original, noisy)
    assert mean_val is not None
    assert 0.0 <= mean_val < 1.0


@pytest.mark.unit
def test_compute_ssim_single_channel_3d_input():
    rng = np.random.default_rng(12)
    img = rng.uniform(0, 100, (32, 32, 1)).astype(np.float32)
    mean_val, ssim_map = compute_ssim(img, img)
    assert mean_val is not None
    assert ssim_map.shape == img.shape


@pytest.mark.unit
def test_compute_ssim_returns_none_on_shape_mismatch():
    rng = np.random.default_rng(13)
    a = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    b = rng.uniform(0, 1, (32, 32)).astype(np.float32)
    mean_val, ssim_map = compute_ssim(a, b)
    assert mean_val is None


@pytest.mark.unit
def test_compute_ssim_uniform_image_returns_nan():
    img = np.ones((32, 32), dtype=np.float32)
    mean_val, ssim_map = compute_ssim(img, img * 2)
    assert np.isnan(mean_val)


# ---------------------------------------------------------------------------
# calculate_iou
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_calculate_iou_identical_masks_returns_one():
    rng = np.random.default_rng(20)
    mask = rng.integers(0, 2, (32, 32), dtype=bool)
    iou, union = calculate_iou(mask, mask)
    assert iou == pytest.approx(1.0)
    assert union == np.sum(mask)


@pytest.mark.unit
def test_calculate_iou_disjoint_masks_returns_zero():
    mask_a = np.zeros((16, 16), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    mask_a[:8, :8] = True
    mask_b[8:, 8:] = True
    iou, union = calculate_iou(mask_a, mask_b)
    assert iou == pytest.approx(0.0)
    assert union > 0


@pytest.mark.unit
def test_calculate_iou_partial_overlap():
    mask_a = np.zeros((16, 16), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    mask_a[:8, :] = True
    mask_b[4:12, :] = True
    iou, union = calculate_iou(mask_a, mask_b)
    # intersection = 4 rows (4-7), union = 12 rows
    expected_iou = 4 * 16 / (12 * 16)
    assert iou == pytest.approx(expected_iou, abs=0.001)


@pytest.mark.unit
def test_calculate_iou_shape_mismatch_returns_nan():
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    iou, union = calculate_iou(mask_a, mask_b)
    assert np.isnan(iou)
    assert np.isnan(union)


@pytest.mark.unit
def test_calculate_iou_empty_masks_returns_nan():
    mask = np.zeros((16, 16), dtype=bool)
    iou, union = calculate_iou(mask, mask)
    assert np.isnan(iou)
    assert union == 0


# ---------------------------------------------------------------------------
# decide_scale
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_decide_scale_from_path_log_min_max():
    result = decide_scale('/models/log_min_max_model/checkpoint.json')
    assert result is not None
    fn, inv_fn = result
    assert callable(fn)
    assert callable(inv_fn)


@pytest.mark.unit
def test_decide_scale_from_path_min_max():
    result = decide_scale('/models/min_max_model/')
    assert result is not None


@pytest.mark.unit
def test_decide_scale_from_path_z_scale():
    result = decide_scale('/some/path/z_scale_model/')
    assert result is not None


@pytest.mark.unit
def test_decide_scale_returns_none_for_unknown_path():
    result = decide_scale('/models/unknown_scaling/')
    assert result is None


# ---------------------------------------------------------------------------
# get_top_best_models
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_top_best_models_returns_x_most_recent():
    files = [
        '/models/checkpoint_epoch_100.h5',
        '/models/checkpoint_epoch_300.h5',
        '/models/checkpoint_epoch_200.h5',
        '/models/checkpoint_epoch_050.h5',
    ]
    result = get_top_best_models(files, 2)
    assert len(result) == 2
    assert '/models/checkpoint_epoch_300.h5' in result
    assert '/models/checkpoint_epoch_200.h5' in result


@pytest.mark.unit
def test_get_top_best_models_handles_fewer_files_than_x():
    files = ['/models/checkpoint_epoch_100.h5']
    result = get_top_best_models(files, 5)
    assert len(result) == 1


@pytest.mark.unit
def test_get_top_best_models_returns_empty_for_invalid_x():
    files = ['/models/checkpoint_epoch_100.h5']
    assert get_top_best_models(files, 0) == []
    assert get_top_best_models(files, -1) == []


@pytest.mark.unit
def test_get_top_best_models_skips_malformed_filenames():
    files = [
        '/models/checkpoint_epoch_100.h5',
        '/models/invalid_no_number.h5',
        '/models/checkpoint_epoch_200.h5',
    ]
    result = get_top_best_models(files, 3)
    assert '/models/invalid_no_number.h5' not in result


# ---------------------------------------------------------------------------
# get_model_by_modulo
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_model_by_modulo_includes_multiples_of_modulo():
    files = [
        '/m/checkpoint_75.h5',
        '/m/checkpoint_100.h5',
        '/m/checkpoint_150.h5',
        '/m/checkpoint_200.h5',
    ]
    result = get_model_by_modulo(files, prototype='', modulo=75)
    # 75 and 150 are multiples of 75 and <= 550
    assert '/m/checkpoint_75.h5' in result
    assert '/m/checkpoint_150.h5' in result


@pytest.mark.unit
def test_get_model_by_modulo_always_includes_final():
    files = [
        '/m/checkpoint_100.h5',
        '/m/checkpoint_final.h5',
    ]
    result = get_model_by_modulo(files, prototype='')
    assert '/m/checkpoint_final.h5' in result


@pytest.mark.unit
def test_get_model_by_modulo_includes_last_epoch():
    files = [
        '/m/checkpoint_1.h5',
        '/m/checkpoint_2.h5',
        '/m/checkpoint_3.h5',
    ]
    result = get_model_by_modulo(files, prototype='', modulo=75)
    # None are multiples of 75, but the last one should be included
    assert '/m/checkpoint_3.h5' in result
