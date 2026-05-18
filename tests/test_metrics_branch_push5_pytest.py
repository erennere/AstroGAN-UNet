from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import starter
from src.evaluation import metrics as metrics_mod


class _FluxLike:
    def __init__(self, data):
        self.data = np.asarray(data)

    def __getitem__(self, key):
        return _FluxLike(self.data[key])

    def __ne__(self, other):
        return self.data != other

    def flatten(self):
        return self.data.flatten()


@pytest.mark.unit
def test_plot_source_comparison_sep_non_array_guard(tmp_path: Path):
    out = metrics_mod.plot_source_comparison_sep(
        'not-array',
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0]), np.array([0]), np.array([0]), np.array([0]),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]),
        str(tmp_path / 'x.png'),
    )
    assert out is None


@pytest.mark.unit
def test_detect_sources_in_image_no_sources_detected_branch(monkeypatch: pytest.MonkeyPatch):
    class Bkg:
        background = np.zeros((8, 8), dtype=np.float32)
        background_rms = np.ones((8, 8), dtype=np.float32)

    monkeypatch.setattr(metrics_mod, 'Background2D', lambda *args, **kwargs: Bkg())
    monkeypatch.setattr(metrics_mod, 'detect_threshold', lambda *args, **kwargs: np.ones((8, 8), dtype=np.float32))
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


@pytest.mark.unit
def test_detect_sources_in_image_deblend_none_branch(monkeypatch: pytest.MonkeyPatch):
    class Bkg:
        background = np.zeros((8, 8), dtype=np.float32)
        background_rms = np.ones((8, 8), dtype=np.float32)

    class Seg:
        pass

    monkeypatch.setattr(metrics_mod, 'Background2D', lambda *args, **kwargs: Bkg())
    monkeypatch.setattr(metrics_mod, 'detect_threshold', lambda *args, **kwargs: np.ones((8, 8), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, 'detect_sources', lambda *args, **kwargs: Seg())
    monkeypatch.setattr(metrics_mod, 'deblend_sources', lambda *args, **kwargs: None)

    out = metrics_mod.detect_sources_in_image(
        np.ones((8, 8), dtype=np.float32),
        {
            'sigma': 3, 'maxiters': 5, 'nsigma': 1, 'npixels': 3,
            'nlevels': 16, 'contrast': 0.001, 'footprint_radius': 2,
            'deblend': True, 'deblend_timeout': 1.0, 'bkg_box_size': 4,
        },
    )
    assert out is None


@pytest.mark.unit
def test_compare_images_rfe_snr_else_branches_and_plot_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_extract(_img, _kwargs):
        x = np.array([1.0], dtype=float)
        y = np.array([1.0], dtype=float)
        flux = np.array([10.0], dtype=float)
        flux_err = np.array([1.0], dtype=float)
        mask = np.zeros((4, 4), dtype=bool)
        return x, y, flux, flux_err, mask

    monkeypatch.setattr(metrics_mod, 'compute_ssim', lambda *args, **kwargs: (0.9, np.ones((4, 4), dtype=np.float32)))
    monkeypatch.setattr(metrics_mod, 'calculate_psnr', lambda *args, **kwargs: (30.0, 0.01))
    monkeypatch.setattr(metrics_mod, 'calculate_iou', lambda *args, **kwargs: (0.5, 10))
    monkeypatch.setattr(metrics_mod, 'plot_source_comparison', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('plot fail')))

    kwargs = {
        'func': fake_extract,
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


@pytest.mark.unit
def test_compare_images_fallback_stats_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_extract(_img, _kwargs):
        x = np.array([1.0, 2.0], dtype=float)
        y = np.array([1.0, 2.0], dtype=float)
        flux = _FluxLike([10.0, 20.0])
        flux_err = _FluxLike([1.0, 2.0])
        mask = np.zeros((4, 4), dtype=bool)
        return x, y, flux, flux_err, mask

    monkeypatch.setattr(metrics_mod, 'compute_ssim', lambda *args, **kwargs: (0.9, np.ones((4, 4), dtype=np.float32)))
    monkeypatch.setattr(metrics_mod, 'calculate_psnr', lambda *args, **kwargs: (30.0, 0.01))
    monkeypatch.setattr(metrics_mod, 'calculate_iou', lambda *args, **kwargs: (0.5, 10))

    kwargs = {
        'func': fake_extract,
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
    out = metrics_mod.compare_images(image, image, image, 'id', 10.0, 5.0, str(tmp_path), kwargs, False)
    assert out is not None
    stats = out[0]
    assert np.isnan(stats['RFE'])
    assert stats['SNR_rec'] == 0
    assert stats['SNR_org'] == 0


@pytest.mark.unit
def test_find_best_performing_models_continue_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models_root = tmp_path / 'models'
    models_root.mkdir()

    wrong_depth = models_root / 'too_short' / 'checkpoints'
    depth10_empty = models_root / 'a' / 'b' / 'c' / 'd' / 'e' / 'f' / 'g' / 'h' / 'i' / 'j' / 'checkpoints'
    depth10_files = models_root / 'u' / 'v' / 'w' / 'x' / 'y' / 'z' / 'aa' / 'bb' / 'cc' / 'dd' / 'checkpoints'

    for p in [wrong_depth, depth10_empty, depth10_files]:
        p.mkdir(parents=True, exist_ok=True)

    bad_ckpt = depth10_files / 'model_badname.keras'
    bad_ckpt.write_text('x', encoding='utf-8')

    def fake_glob(pattern: str):
        root = os.path.dirname(pattern)
        if root == str(depth10_empty):
            return []
        if root == str(depth10_files):
            return [str(bad_ckpt)]
        return []

    def filter_model(files, *_args):
        return files

    def condition(model_info):
        if isinstance(model_info, pd.DataFrame):
            raise RuntimeError('force fallback condition path')
        return False

    monkeypatch.setattr(metrics_mod.glob, 'glob', fake_glob)

    out = metrics_mod.find_best_performing_models(
        str(models_root),
        condition,
        filter_model,
        '*.keras',
        n=1,
        index=0,
        concurrent_workers=1,
    )
    assert isinstance(out, dict)


@pytest.mark.unit
def test_find_best_performing_models_selected_models_empty_continue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models_root = tmp_path / 'models'
    models_root.mkdir()

    depth10_files = models_root / 'u' / 'v' / 'w' / 'x' / 'y' / 'z' / 'aa' / 'bb' / 'cc' / 'dd' / 'checkpoints'
    depth10_files.mkdir(parents=True, exist_ok=True)
    ckpt = depth10_files / 'model_001.keras'
    ckpt.write_text('x', encoding='utf-8')

    monkeypatch.setattr(metrics_mod.glob, 'glob', lambda pattern: [str(ckpt)])

    out = metrics_mod.find_best_performing_models(
        str(models_root),
        lambda _info: True,
        lambda _files, *_args: [],
        '*.keras',
        n=1,
        index=0,
        concurrent_workers=1,
    )
    assert out == {}


@pytest.mark.unit
def test_reconstruct_patch_remaining_paths(monkeypatch: pytest.MonkeyPatch):
    class ModelOK:
        def predict(self, arr):
            return np.asarray(arr)

    class ModelFail:
        def predict(self, _arr):
            raise RuntimeError('predict-fail')

    img = np.ones((4, 4), dtype=np.float32)

    # scales != None, use_mosaic=False happy path
    scales = (
        lambda x: (x, 2.0),
        lambda y, s: y / s,
    )
    out = metrics_mod._reconstruct_patch(
        img,
        scales,
        ModelOK(),
        use_mosaic=False,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='gaussian',
        batch_size=1,
    )
    assert isinstance(out, np.ndarray)

    # scales != None, reconstruction exception path
    out_fail = metrics_mod._reconstruct_patch(
        img,
        scales,
        ModelFail(),
        use_mosaic=False,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='gaussian',
        batch_size=1,
    )
    assert out_fail is None

    # scales != None, outer scaling exception path
    bad_scales = (lambda _x: (_ for _ in ()).throw(RuntimeError('scale-fail')), lambda y: y)
    out_scale_fail = metrics_mod._reconstruct_patch(
        img,
        bad_scales,
        ModelOK(),
        use_mosaic=False,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='gaussian',
        batch_size=1,
    )
    assert out_scale_fail is None

    # scales=None, use_mosaic=True path
    monkeypatch.setattr(metrics_mod, 'sliding_window_inference', lambda *args, **kwargs: np.ones((4, 4, 1), dtype=np.float32))
    out_mosaic = metrics_mod._reconstruct_patch(
        img,
        None,
        ModelOK(),
        use_mosaic=True,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='gaussian',
        batch_size=1,
    )
    assert isinstance(out_mosaic, np.ndarray)

    # scales=None exception path
    out_none_fail = metrics_mod._reconstruct_patch(
        img,
        None,
        ModelFail(),
        use_mosaic=False,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='gaussian',
        batch_size=1,
    )
    assert out_none_fail is None


@pytest.mark.unit
def test_finalise_dfs_model_index_nonzero_branch():
    rec = [pd.DataFrame({'x': [1.0]})]
    org_df, noisy_df, rec_df = metrics_mod._finalise_dfs(rec, [], [], epoch='1', model_dir='m', model_index=1)
    assert org_df.empty
    assert noisy_df.empty
    assert not rec_df.empty


@pytest.mark.unit
def test_process_single_model_save_image_exception_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_file = tmp_path / 'model_010.keras'
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
        def predict(self, arr):
            return np.asarray(arr)

    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(metrics_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch)
    monkeypatch.setattr(metrics_mod, 'ensure_directory_exists', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics_mod, 'save_fits', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('save-fits-fail')))
    monkeypatch.setattr(metrics_mod, 'compare_images', lambda *args, **kwargs: None)

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
def test_process_models_collect_none_result_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(metrics_mod, 'process_single_model', lambda *args, **kwargs: None)

    out = metrics_mod.process_models(
        job=('m', ['a.keras'], None, pd.DataFrame(), {'model_alias_hex': 'h'}),
        kwargs_source={},
        parallel=False,
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
def test_main_get_test_images_exception_and_none_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metrics_mod, 'get_test_images', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))
    out = metrics_mod.main(
        models_dir='models',
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={},
        total_workers=1,
        max_workers=1,
        frac=0.1,
        condition=metrics_mod.condition,
        filter_model=metrics_mod.get_model_by_modulo,
        n=1,
        model_prototype='*.keras',
        all_metrics_csv=str(tmp_path / 'all.csv'),
    )
    assert out is None

    monkeypatch.setattr(metrics_mod, 'get_test_images', lambda *args, **kwargs: None)
    out2 = metrics_mod.main(
        models_dir='models',
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={},
        total_workers=1,
        max_workers=1,
        frac=0.1,
        condition=metrics_mod.condition,
        filter_model=metrics_mod.get_model_by_modulo,
        n=1,
        model_prototype='*.keras',
        all_metrics_csv=str(tmp_path / 'all2.csv'),
    )
    assert out2 is None


@pytest.mark.unit
def test_main_parallel_future_none_and_nonparallel_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(metrics_mod, 'get_test_images', lambda *args, **kwargs: pd.DataFrame({'x': [1]}))
    monkeypatch.setattr(metrics_mod, 'ensure_parent_dir_exists', lambda *args, **kwargs: None)
    monkeypatch.setattr(metrics_mod, 'find_best_performing_models', lambda *args, **kwargs: {'m': ['a.keras']})
    monkeypatch.setattr(metrics_mod, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics_mod, '_decode_models_dir', lambda *_args, **_kwargs: {'model_alias_hex': 'h'})

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return object()

    monkeypatch.setattr(metrics_mod, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(metrics_mod, 'as_completed', lambda _futures: [None])

    metrics_mod.main(
        models_dir='models',
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={},
        total_workers=1,
        max_workers=1,
        frac=0.1,
        condition=metrics_mod.condition,
        filter_model=metrics_mod.get_model_by_modulo,
        n=1,
        model_prototype='*.keras',
        parallel=True,
        all_metrics_csv=str(tmp_path / 'all.csv'),
    )

    monkeypatch.setattr(metrics_mod, 'process_models', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('job-fail')))
    metrics_mod.main(
        models_dir='models',
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={},
        total_workers=1,
        max_workers=1,
        frac=0.1,
        condition=metrics_mod.condition,
        filter_model=metrics_mod.get_model_by_modulo,
        n=1,
        model_prototype='*.keras',
        parallel=False,
        all_metrics_csv=str(tmp_path / 'all2.csv'),
    )


@pytest.mark.unit
def test_condition_and_module_main_guard(monkeypatch: pytest.MonkeyPatch):
    assert metrics_mod.condition([1, 2, 3]) == [True, True, True]

    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        starter,
        'load_config',
        lambda **kwargs: {
            'metrics': {
                'models_dir': 'models',
                'data_kwargs': {},
                'model_kwargs': {},
                'kwargs_source': {},
                'total_workers': 1,
                'max_workers': 1,
                'frac': 0.1,
                'n': 1,
                'model_prototype': '*.keras',
                'all_metrics_csv': 'all.csv',
                'aggregated_metrics_csv': 'agg.csv',
                'org_catalog_csv': 'org.csv',
                'noisy_catalog_csv': 'noisy.csv',
                'rec_catalog_csv': 'rec.csv',
                'parallel': False,
                'parallel_epoch': False,
                'scaling': None,
            }
        },
    )
    monkeypatch.setattr(sys, 'argv', ['metrics.py', '3', '4'])
    runpy.run_module('src.evaluation.metrics', run_name='__main__')
