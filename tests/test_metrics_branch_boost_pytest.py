"""Branch-focused coverage boosts for src/evaluation/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as metrics_mod


class _Future:
    def __init__(self, fn=None, value=None, exc: Exception | None = None):
        self._value = value if fn is None else fn()
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class _Executor:
    def __init__(self, max_workers=1, fail=False):
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        if self.fail:
            return _Future(value=None, exc=RuntimeError('future fail'))
        return _Future(fn=lambda: fn(*args, **kwargs))


@pytest.mark.unit
def test_detect_sources_invalid_segmentation_returns_none(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((16, 16), dtype=np.float32)
    kwargs = {
        'sigma': 3.0,
        'maxiters': 5,
        'nsigma': 2.0,
        'npixels': 5,
        'nlevels': 8,
        'contrast': 0.001,
        'footprint_radius': 3,
        'deblend': False,
        'deblend_timeout': 0.1,
    }

    class NoMaskSeg:
        pass

    monkeypatch.setattr(metrics_mod, 'detect_sources', lambda *args, **kwargs: NoMaskSeg())
    assert metrics_mod.detect_sources_in_image(image, kwargs) is None


@pytest.mark.unit
def test_detect_sources_catalog_empty_returns_none(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((16, 16), dtype=np.float32)
    kwargs = {
        'sigma': 3.0,
        'maxiters': 5,
        'nsigma': 2.0,
        'npixels': 5,
        'nlevels': 8,
        'contrast': 0.001,
        'footprint_radius': 3,
        'deblend': False,
        'deblend_timeout': 0.1,
    }

    class Seg:
        def make_source_mask(self, footprint=None):
            return np.zeros((16, 16), dtype=bool)

    class EmptyCatalog:
        def __len__(self):
            return 0

    monkeypatch.setattr(metrics_mod, 'detect_sources', lambda *args, **kwargs: Seg())
    monkeypatch.setattr(metrics_mod, 'SourceCatalog', lambda *args, **kwargs: EmptyCatalog())

    assert metrics_mod.detect_sources_in_image(image, kwargs) is None


@pytest.mark.unit
def test_detect_sources_deblend_exception_fallback(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((16, 16), dtype=np.float32)
    image[5:8, 5:8] = 10.0
    kwargs = {
        'sigma': 3.0,
        'maxiters': 5,
        'nsigma': 1.0,
        'npixels': 3,
        'nlevels': 8,
        'contrast': 0.001,
        'footprint_radius': 2,
        'deblend': True,
        'deblend_timeout': 0.1,
    }

    monkeypatch.setattr(metrics_mod, 'deblend_sources', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('deblend fail')))

    result = metrics_mod.detect_sources_in_image(image, kwargs)
    assert result is not None


@pytest.mark.unit
def test_process_models_parallel_future_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(metrics_mod, 'ProcessPoolExecutor', lambda max_workers=1: _Executor(max_workers=max_workers, fail=True))
    monkeypatch.setattr(metrics_mod, 'as_completed', lambda futures: futures)

    all_metrics, aggregated, (org, noisy, rec) = metrics_mod.process_models(
        job=('m', ['f1.keras'], None, pd.DataFrame(), {'model_alias_hex': 'x'}),
        kwargs_source={},
        workers=1,
        parallel=True,
        all_metrics_csv=str(tmp_path / 'all_*.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg_*.csv'),
        org_catalog_csv=str(tmp_path / 'org_*.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy_*.csv'),
        rec_catalog_csv=str(tmp_path / 'rec_*.csv'),
    )

    assert all_metrics.empty and aggregated.empty
    assert org.empty and noisy.empty and rec.empty


@pytest.mark.unit
def test_metrics_main_parallel_job_future_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(metrics_mod, 'get_test_images', lambda *args, **kwargs: pd.DataFrame({'location': ['a']}))
    monkeypatch.setattr(metrics_mod, 'find_best_performing_models', lambda *args, **kwargs: {'m': ['c.keras']})
    monkeypatch.setattr(metrics_mod, '_decode_models_dir', lambda *_: {'model_alias_hex': 'x'})
    monkeypatch.setattr(metrics_mod, 'decide_scale', lambda *_: None)
    monkeypatch.setattr(metrics_mod, 'ProcessPoolExecutor', lambda max_workers=1: _Executor(max_workers=max_workers, fail=True))
    monkeypatch.setattr(metrics_mod, 'as_completed', lambda futures: futures)

    metrics_mod.main(
        models_dir=str(tmp_path),
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={},
        total_workers=1,
        max_workers=1,
        frac=0.1,
        condition=lambda *_: True,
        filter_model=lambda files, *args: files,
        n=1,
        model_prototype='*.keras',
        all_metrics_csv=str(tmp_path / 'all.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg.csv'),
        org_catalog_csv=str(tmp_path / 'org.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy.csv'),
        rec_catalog_csv=str(tmp_path / 'rec.csv'),
        parallel=True,
        parallel_epoch=False,
        scaling='min_max',
        index=0,
        concurrent_workers=1,
    )


@pytest.mark.unit
def test_compare_images_kdtree_failure_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_detect(image, kwargs):
        x = np.array([1.0, 2.0], dtype=float)
        y = np.array([1.0, 2.0], dtype=float)
        flux = np.array([1.0, 2.0], dtype=float)
        ferr = np.array([0.1, 0.2], dtype=float)
        mask = np.zeros((8, 8), dtype=bool)
        return x, y, flux, ferr, mask

    class BadTree:
        def __init__(self, arr):
            raise RuntimeError('kdtree fail')

    monkeypatch.setattr(metrics_mod, 'cKDTree', BadTree)

    kwargs = {
        'func': fake_detect,
        'distance_threshold': 2.0,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k1': 0.01,
        'k2': 0.03,
        'win_size': 7,
        'win_sigma': 1.0,
    }

    image = np.ones((8, 8), dtype=np.float32)
    assert metrics_mod.compare_images(image, image, image, 'id', 1.0, 0.5, str(tmp_path), kwargs, False) is None
