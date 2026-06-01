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


@pytest.mark.unit
def test_process_single_model_required_columns_and_load_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_file = tmp_path / 'model_001.keras'
    model_file.write_text('x', encoding='utf-8')

    bad_df = pd.DataFrame({'location': ['x.fits']})
    assert metrics_mod.process_single_model(str(model_file), 'm', None, bad_df, {}) is None

    good_df = pd.DataFrame({
        'location': ['x.fits'],
        'sci_actual_duration': [10.0],
        'new_exp_time': [5.0],
        'combined_sigma': [1.0],
    })
    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('load failed')))
    assert metrics_mod.process_single_model(str(model_file), 'm', None, good_df, {}) is None


@pytest.mark.unit
def test_process_single_model_open_fits_reconstruct_and_dir_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_file = tmp_path / 'model_002.keras'
    model_file.write_text('x', encoding='utf-8')
    img = tmp_path / 'img.fits'
    img.write_text('x', encoding='utf-8')

    df = pd.DataFrame({
        'location': [str(img), str(img), str(img), str(img)],
        'sci_actual_duration': [10.0, 10.0, 10.0, 10.0],
        'new_exp_time': [5.0, 5.0, 5.0, 5.0],
        'combined_sigma': [1.0, 1.0, 1.0, 1.0],
    })

    class DummyModel:
        pass

    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *args, **kwargs: DummyModel())

    calls = {'n': 0}

    def fake_open(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('read fail')
        if calls['n'] == 2:
            return None
        if calls['n'] == 3:
            return 'bad-type'
        return np.ones((4, 4), dtype=np.float32)

    monkeypatch.setattr(metrics_mod, 'open_fits', fake_open)
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda *args, **kwargs: None)

    out = metrics_mod.process_single_model(
        str(model_file),
        'm',
        None,
        df,
        {'distance_threshold': 2.0, 'func': lambda *_args, **_kwargs: None},
        frac=1.0,
        noise_fn=lambda base, row, sigma: base,
        combined_images_dir=str(tmp_path / 'c'),
        png_dir=str(tmp_path / 'p'),
        org_dir=str(tmp_path / 'o'),
        noisy_dir=str(tmp_path / 'n'),
        rec_dir=str(tmp_path / 'r'),
    )
    assert out is not None
    metrics_df, aggregated_df, dfs = out
    assert metrics_df.empty and aggregated_df.empty
    assert all(d.empty for d in dfs)


@pytest.mark.unit
def test_process_single_model_image_save_and_aggregate_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    model_file = tmp_path / 'model_003.keras'
    model_file.write_text('x', encoding='utf-8')
    img = tmp_path / 'img.fits'
    img.write_text('x', encoding='utf-8')

    df = pd.DataFrame({
        'location': [str(img)],
        'sci_actual_duration': [10.0],
        'new_exp_time': [5.0],
        'combined_sigma': [1.0],
    })

    class DummyModel:
        pass

    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(metrics_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch + 1.0)
    monkeypatch.setattr(metrics_mod, 'ensure_directory_exists', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('mkdir fail')))
    monkeypatch.setattr(metrics_mod, 'compare_images', lambda *args, **kwargs: (
        {'TP': 1.0, 'FP': 0.0, 'FN': 0.0},
        ([1.0], [1.0]),
        ([0.1], [0.1]),
        (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    ))
    monkeypatch.setattr(metrics_mod, 'aggregate_df', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('agg fail')))
    monkeypatch.setattr(metrics_mod, 'save_fits', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('save fail')))
    monkeypatch.setattr(metrics_mod, 'scale_image', lambda img: (Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255)), 0.0, 1.0))

    out = metrics_mod.process_single_model(
        str(model_file),
        'm',
        None,
        df,
        {'distance_threshold': 2.0, 'func': lambda *_args, **_kwargs: None},
        frac=1.0,
        noise_fn=lambda base, row, sigma: base,
        combined_images_dir=str(tmp_path / 'c'),
        png_dir=str(tmp_path / 'p'),
        org_dir=str(tmp_path / 'o'),
        noisy_dir=str(tmp_path / 'n'),
        rec_dir=str(tmp_path / 'r'),
    )
    assert out is not None


@pytest.mark.unit
def test_process_models_parallel_future_exception_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeFuture:
        def result(self):
            raise RuntimeError('future failed')

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = FakeFuture()
            self._futures.append(fut)
            return fut

    monkeypatch.setattr(metrics_mod, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(metrics_mod, 'as_completed', lambda futures: futures)

    out = metrics_mod.process_models(
        job=('m', ['a.keras'], None, pd.DataFrame(), {'model_alias_hex': 'h'}),
        kwargs_source={},
        parallel=True,
        workers=1,
        all_metrics_csv=str(tmp_path / 'all_*_metrics.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg_*_metrics.csv'),
        org_catalog_csv=str(tmp_path / 'org_*_catalog.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy_*_catalog.csv'),
        rec_catalog_csv=str(tmp_path / 'rec_*_catalog.csv'),
    )
    all_metrics, aggregated_metrics, (org_df, noisy_df, rec_df) = out
    assert all_metrics.empty and aggregated_metrics.empty
    assert org_df.empty and noisy_df.empty and rec_df.empty


@pytest.mark.unit
def test_compare_images_wrap_extract_plot_sep_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def wrap_extract_sources(image, flag, kwargs):
        x = np.array([1.0, 2.0], dtype=float)
        y = np.array([1.0, 2.0], dtype=float)
        flux = np.array([10.0, 20.0], dtype=float)
        flux_err = np.array([1.0, 2.0], dtype=float)
        mask = np.zeros((4, 4), dtype=bool)
        a = np.array([1.0, 1.1], dtype=float)
        b = np.array([0.9, 1.0], dtype=float)
        theta = np.array([0.0, 0.1], dtype=float)
        df = pd.DataFrame({'x': x, 'y': y})
        return x, y, flux, flux_err, mask, a, b, theta, df

    monkeypatch.setattr(metrics_mod, 'compute_ssim', lambda *args, **kwargs: (0.9, np.ones((4, 4), dtype=np.float32)))
    monkeypatch.setattr(metrics_mod, 'calculate_psnr', lambda *args, **kwargs: (30.0, 0.01))
    monkeypatch.setattr(metrics_mod, 'calculate_iou', lambda *args, **kwargs: (0.5, 10))
    called = {'n': 0}
    monkeypatch.setattr(metrics_mod, 'plot_source_comparison_sep', lambda *args, **kwargs: called.__setitem__('n', called['n'] + 1))

    monkeypatch.setattr(metrics_mod, 'wrap_extract_sources', wrap_extract_sources)
    kwargs = {
        'func': wrap_extract_sources,
        'distance_threshold': 5.0,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k1': 0.01,
        'k2': 0.03,
        'win_size': 3,
        'win_sigma': 1.5,
    }

    image = np.ones((4, 4), dtype=np.float32)
    out = metrics_mod.compare_images(image, image, image, 'id', 10.0, 5.0, str(tmp_path), kwargs, True)
    assert out is not None
    assert called['n'] == 1
