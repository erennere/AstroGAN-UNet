"""Additional branch-heavy tests for src/evaluation/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as metrics_mod


class _Model:
    def predict(self, patches):
        return np.asarray(patches)


@pytest.mark.unit
def test_sliding_window_inference_pad_none_and_unpack_error(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((8, 8, 1), dtype=np.float32)
    monkeypatch.setattr(metrics_mod, 'pad_image', lambda *_args, **_kwargs: None)
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1)) is None

    monkeypatch.setattr(metrics_mod, 'pad_image', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('boom')))
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1)) is None


@pytest.mark.unit
def test_sliding_window_inference_weight_map_none_paths(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((8, 8, 1), dtype=np.float32)
    pad_ok = (image, (0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(metrics_mod, 'pad_image', lambda *_args, **_kwargs: pad_ok)

    monkeypatch.setattr(metrics_mod, 'generate_gaussian_weights', lambda *_args, **_kwargs: None)
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1), weighting='gaussian') is None

    monkeypatch.setattr(metrics_mod, 'generate_distance_weights', lambda *_args, **_kwargs: None)
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1), weighting='distance') is None

    monkeypatch.setattr(metrics_mod.np, 'ones', lambda *args, **kwargs: None)
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1), weighting='average') is None


@pytest.mark.unit
def test_sliding_window_inference_strip_pad_none_and_exception(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((8, 8, 1), dtype=np.float32)
    pad_ok = (image, (0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(metrics_mod, 'pad_image', lambda *_args, **_kwargs: pad_ok)

    ds = tf.data.Dataset.from_tensors((np.ones((4, 4, 1), dtype=np.float32), np.array([0, 0], dtype=np.int64))).batch(1)
    monkeypatch.setattr(metrics_mod, 'create_prediction_dataset', lambda *_args, **_kwargs: ds)

    calls = {'n': 0}

    def _strip(*args, **kwargs):
        calls['n'] += 1
        return None if calls['n'] == 1 else np.ones((8, 8, 1), dtype=np.float32)

    monkeypatch.setattr(metrics_mod, 'strip_pad', _strip)
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1)) is None

    monkeypatch.setattr(metrics_mod, 'strip_pad', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('strip fail')))
    assert metrics_mod.sliding_window_inference(image, _Model(), patch_size=(4, 4, 1), stride=(2, 2, 1)) is None


@pytest.mark.unit
def test_plot_source_comparison_extra_validation_and_outer_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    x = np.array([1.0])
    y = np.array([1.0])
    idx = np.array([0])

    metrics_mod.plot_source_comparison(
        np.array([1.0], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        str(tmp_path / 'a.png'),
    )

    metrics_mod.plot_source_comparison(
        np.array([['x']], dtype=object),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        str(tmp_path / 'b.png'),
    )

    monkeypatch.setattr(metrics_mod.plt, 'figure', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('plt fail')))
    metrics_mod.plot_source_comparison(
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        str(tmp_path / 'c.png'),
    )


@pytest.mark.unit
def test_plot_source_comparison_sep_extra_validation_and_outer_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    x = np.array([1.0])
    y = np.array([1.0])
    idx = np.array([0])
    ab = np.array([1.0])
    th = np.array([0.0])

    metrics_mod.plot_source_comparison_sep(
        np.array([1.0], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        ab, ab, ab, ab, th, th,
        str(tmp_path / 'd.png'),
    )

    metrics_mod.plot_source_comparison_sep(
        np.ones((2, 2), dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        ab, ab, ab, ab, th, th,
        str(tmp_path / 'd2.png'),
    )

    metrics_mod.plot_source_comparison_sep(
        np.ones((2, 2), dtype=np.float32),
        np.array([['x']], dtype=object),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        ab, ab, ab, ab, th, th,
        str(tmp_path / 'd3.png'),
    )

    monkeypatch.setattr(metrics_mod.plt, 'figure', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('plt fail')))
    metrics_mod.plot_source_comparison_sep(
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        x, y, x, y, idx, idx, idx, idx,
        ab, ab, ab, ab, th, th,
        str(tmp_path / 'e.png'),
    )


@pytest.mark.unit
def test_aggregate_df_else_branches_and_detect_source_input_guards(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({'TP': [1], 'FP': [0], 'FN': [1], 'SSIM_rec': [0.5], 'SSIM_noisy': [0.4], 'IoU': [0.1]})
    stats = metrics_mod.aggregate_df(df, flux_rec=[1, 2], flux_org=[1, 2], flux_error_rec=[1], flux_error_org=[1])
    assert stats['RFE'] is np.nan or np.isnan(stats['RFE'])
    assert stats['SNR_rec'] == 0
    assert stats['SNR_org'] == 0

    assert metrics_mod.detect_sources_in_image(np.ones((8, 8), dtype=np.float32), kwargs={}) is None
    assert metrics_mod.detect_sources_in_image(np.ones((8, 8, 1), dtype=np.float32), kwargs={
        'sigma': 3, 'maxiters': 5, 'nsigma': 1, 'npixels': 3,
        'nlevels': 16, 'contrast': 0.001, 'footprint_radius': 2,
        'deblend': False, 'deblend_timeout': 0.1, 'bkg_box_size': 4,
    }) is None

    monkeypatch.setattr(metrics_mod, 'Background2D', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bkg fail')))
    monkeypatch.setattr(metrics_mod, 'detect_threshold', lambda *args, **kwargs: np.ones((8, 8), dtype=np.float32))

    class Seg:
        def make_source_mask(self, footprint=None):
            return np.zeros((8, 8), dtype=bool)

    monkeypatch.setattr(metrics_mod, 'detect_sources', lambda *args, **kwargs: Seg())
    monkeypatch.setattr(metrics_mod, 'SourceCatalog', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('catalog fail')))

    out = metrics_mod.detect_sources_in_image(
        np.ones((8, 8), dtype=np.float32),
        {
            'sigma': 3, 'maxiters': 5, 'nsigma': 1, 'npixels': 3,
            'nlevels': 16, 'contrast': 0.001, 'footprint_radius': 2,
            'deblend': False, 'deblend_timeout': 0.1, 'bkg_box_size': 4,
        },
    )
    assert out is None


@pytest.mark.unit
def test_detect_sources_none_segment_and_wrap_extract_sources(monkeypatch: pytest.MonkeyPatch):
    class Bkg:
        background = np.zeros((8, 8), dtype=np.float32)
        background_rms = np.ones((8, 8), dtype=np.float32)

    monkeypatch.setattr(metrics_mod, 'Background2D', lambda *args, **kwargs: Bkg())
    monkeypatch.setattr(metrics_mod, 'detect_sources', lambda *args, **kwargs: None)

    out = metrics_mod.detect_sources_in_image(
        np.ones((8, 8), dtype=np.float32),
        {
            'sigma': 3, 'maxiters': 5, 'nsigma': 1, 'npixels': 3,
            'nlevels': 16, 'contrast': 0.001, 'footprint_radius': 2,
            'deblend': False, 'deblend_timeout': 0.1, 'bkg_box_size': 4,
        },
    )
    assert out is None

    df = pd.DataFrame({'x': [1.0], 'y': [1.0], 'flux': [2.0], 'flux_err': [0.1], 'a': [1.0], 'b': [1.0], 'theta': [0.0]})
    monkeypatch.setattr(metrics_mod, 'extract_sources', lambda *args, **kwargs: (df, np.zeros((2, 2), dtype=bool)))
    wrapped = metrics_mod.wrap_extract_sources(np.ones((2, 2), dtype=np.float32), 'original', {})
    assert len(wrapped) == 9


@pytest.mark.unit
def test_compare_images_input_and_detection_exception_guards(tmp_path: Path):
    kwargs = {'func': lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('boom'))}

    assert metrics_mod.compare_images('bad', np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 'id', 1.0, 0.5, str(tmp_path), kwargs, False) is None
    assert metrics_mod.compare_images(np.ones((2, 2), dtype=np.float32), np.ones((3, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 'id', 1.0, 0.5, str(tmp_path), kwargs, False) is None
    assert metrics_mod.compare_images(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 'id', 1.0, 0.5, str(tmp_path), kwargs, False) is None
