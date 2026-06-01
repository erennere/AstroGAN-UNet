"""Additional branch-focused tests for src/evaluation/uncropped_metrics.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starter
import src.training.utils as utils_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import uncropped_metrics as um


@pytest.mark.unit
def test_process_subdf_returns_none_when_model_load_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sub_df = pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'combined_sigma': [1.0],
        }
    )
    monkeypatch.setattr(um, 'load_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('load fail')))
    out = um.process_subdf(
        sub_df,
        'model.keras',
        str(tmp_path),
        {
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )
    assert out is None


@pytest.mark.unit
def test_process_subdf_open_fits_none_and_rec_none_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sub_df = pd.DataFrame(
        {
            'exp_ratio': [2.0, 2.0],
            'location': ['a.fits', 'b.fits'],
            'name': ['a', 'b'],
            'exp_time': [100.0, 100.0],
            'new_exp_time': [50.0, 50.0],
            'combined_sigma': [1.0, 1.0],
        }
    )

    monkeypatch.setattr(um, 'load_checkpoint_model', lambda *args, **kwargs: object())
    monkeypatch.setattr(um, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(um, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(um, 'open_fits', lambda filepath, **kwargs: None if filepath.endswith('a.fits') else np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(um, 'crop_image_generator', lambda *_args, **_kwargs: iter([np.ones((2, 2), dtype=np.float32)]))
    monkeypatch.setattr(um, '_reconstruct_patch', lambda *_args, **_kwargs: None)

    out = um.process_subdf(
        sub_df,
        'model.keras',
        str(tmp_path),
        {
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )

    assert out is not None
    results, _hists, (org_df, noisy_df, rec_df) = out
    assert results == []
    assert org_df.empty and noisy_df.empty and rec_df.empty


@pytest.mark.unit
def test_process_subdf_accepts_numpy_float_patch_stride(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sub_df = pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'combined_sigma': [1.0],
        }
    )

    seen = {'ps': None}

    monkeypatch.setattr(um, 'load_checkpoint_model', lambda *args, **kwargs: object())
    monkeypatch.setattr(um, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(um, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(um, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        um,
        'crop_image_generator',
        lambda image, ps=0: seen.__setitem__('ps', ps) or iter([np.ones((2, 2), dtype=np.float32)]),
    )
    monkeypatch.setattr(um, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch)
    monkeypatch.setattr(um, 'compare_images', lambda *args, **kwargs: None)

    out = um.process_subdf(
        sub_df,
        'model.keras',
        str(tmp_path),
        {
            'uncropped_patch_size': (np.float64(2.0), np.float64(2.0), np.float64(1.0)),
            'uncropped_stride': (np.float64(2.0), np.float64(2.0), np.float64(1.0)),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': np.float64(1.0),
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )

    assert out is not None
    assert seen['ps'] == 2


@pytest.mark.unit
def test_process_subdf_accepts_singleton_tuple_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sub_df = pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'combined_sigma': [1.0],
        }
    )

    monkeypatch.setattr(um, 'load_checkpoint_model', lambda *args, **kwargs: object())
    monkeypatch.setattr(um, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(um, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(um, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(um, 'crop_image_generator', lambda *_args, **_kwargs: iter([np.ones((2, 2), dtype=np.float32)]))
    monkeypatch.setattr(um, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch)
    monkeypatch.setattr(um, 'compare_images', lambda *args, **kwargs: None)

    kwargs_source = ({
        'uncropped_patch_size': (2, 2, 1),
        'uncropped_stride': (2, 2, 1),
        'uncropped_weighting': 'average',
        'uncropped_batch_size': 1,
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'sigma_key': 'combined_sigma',
        'type_of_image': 'SCI',
        'noise_fn': lambda img, row, sigma: img,
        'uncropped_use_mosaic': True,
    },)

    out = um.process_subdf(
        sub_df,
        'model.keras',
        str(tmp_path),
        kwargs_source,
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )

    assert out is not None


@pytest.mark.unit
def test_process_subdf_non_mosaic_uses_batched_predict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sub_df = pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'combined_sigma': [1.0],
        }
    )

    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, x, batch_size=None, verbose=0):
            self.calls.append((x.shape[0], batch_size, verbose))
            return x

    fake_model = FakeModel()

    monkeypatch.setattr(um, 'load_checkpoint_model', lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(um, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(um, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(um, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        um,
        'crop_image_generator',
        lambda *_args, **_kwargs: iter([
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32) * 2,
            np.ones((2, 2), dtype=np.float32) * 3,
        ]),
    )
    monkeypatch.setattr(um, 'compare_images', lambda *args, **kwargs: None)

    out = um.process_subdf(
        sub_df,
        'model.keras',
        str(tmp_path),
        {
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 2,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': False,
        },
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )

    assert out is not None
    assert fake_model.calls == [(2, 2, 0), (1, 2, 0)]


@pytest.mark.unit
def test_uncropped_main_future_exception_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_csv = tmp_path / 'meta.csv'
    pd.DataFrame({'exp_ratio': [2.0], 'location': ['a.fits'], 'name': ['a'], 'exp_time': [100.0], 'new_exp_time': [50.0]}).to_csv(metadata_csv, index=False)

    class _Future:
        def result(self):
            raise RuntimeError('future boom')

    class _Exec:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future()

    monkeypatch.setattr(um, 'ProcessPoolExecutor', _Exec)
    monkeypatch.setattr(um, 'as_completed', lambda futures: futures)
    monkeypatch.setattr(
        um,
        'candidates_based_on_range',
        lambda row, kwargs: [
            {
                'exp_ratio': row['exp_ratio'],
                'new_exp_time': row['new_exp_time'],
                'location': row['location'],
                'name': row['name'],
            }
        ],
    )

    output_paths = {
        'results_csv': str(tmp_path / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    eval_cfg = {
        'uncropped_output_dir': str(tmp_path / 'out'),
        'uncropped_combined_images_dir': str(tmp_path / 'combined'),
        'uncropped_sampled_data_csv': str(tmp_path / 'sampled.csv'),
        'uncropped_results_csv': 'results.csv',
        'uncropped_org_catalog_csv': 'org.csv',
        'uncropped_noisy_catalog_csv': 'noisy.csv',
        'uncropped_rec_catalog_csv': 'rec.csv',
        'uncropped_hist_data_csv': 'hist_data.csv',
        'uncropped_hist_png_template': 'hist_{exp_ratio}.png',
        'uncropped_n': 1,
        'uncropped_workers': 2,
        'hist_min_exp': -1,
        'hist_max_exp': 1,
        'uncropped_save_images': False,
        'uncropped_save_combined_images': False,
        'uncropped_overwrite': True,
        'uncropped_single_parallel': False,
        'uncropped_write_histograms': True,
        'filter_by_last_name': False,
        'last_name_col': 'sci_pi_last_name',
        'last_name_filter_value': ['FABER'],
        'kwargs_source': {
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
    }
    data_cfg = {'low': 1.0, 'log_domain_clip_max': 1.0, 'data_alias_enriched_hex': 'data_alias'}

    um.orchestrate_uncropped_evaluation(
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        metadata_filepath=str(metadata_csv),
        model_filepath='model.keras',
        data_alias_enriched_hex='data_alias',
        model_alias_hex='model_alias',
        epoch='1',
    )

    assert Path(tmp_path / 'out' / 'results.csv').exists()
    assert Path(tmp_path / 'out' / 'hist_data.csv').exists()


@pytest.mark.unit
def test_orchestrate_uncropped_evaluation_accepts_int_epoch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_csv = tmp_path / 'meta_int_epoch.csv'
    pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'sci_pi_last_name': ['FABER'],
        }
    ).to_csv(metadata_csv, index=False)

    calls = []
    monkeypatch.setattr(um, 'process_data', lambda **kwargs: calls.append(kwargs) or pd.DataFrame())
    monkeypatch.setattr(um, 'orchestrate_single_run', lambda *args, **kwargs: calls.append(kwargs))

    eval_cfg = {
        'uncropped_output_dir': str(tmp_path / 'out' / '#' / '&' / '*'),
        'uncropped_combined_images_dir': str(tmp_path / 'combined' / '#' / '&' / '*'),
        'uncropped_sampled_data_csv': str(tmp_path / 'sampled' / '#' / '&' / '*' / 'sampled.csv'),
        'uncropped_results_csv': 'results_&.csv',
        'uncropped_org_catalog_csv': 'org_&.csv',
        'uncropped_noisy_catalog_csv': 'noisy_&.csv',
        'uncropped_rec_catalog_csv': 'rec_&.csv',
        'uncropped_hist_data_csv': 'hist_&.csv',
        'uncropped_hist_png_template': 'hist_&_{exp_ratio}.png',
        'uncropped_n': 1,
        'uncropped_workers': 1,
        'hist_min_exp': -1,
        'hist_max_exp': 1,
        'uncropped_save_images': False,
        'uncropped_save_combined_images': False,
        'uncropped_overwrite': True,
        'uncropped_single_parallel': False,
        'uncropped_write_histograms': True,
        'filter_by_last_name': False,
        'last_name_col': 'sci_pi_last_name',
        'last_name_filter_value': ['FABER'],
        'kwargs_source': {},
    }

    um.orchestrate_uncropped_evaluation(
        eval_cfg=eval_cfg,
        data_cfg={'low': 1.0},
        metadata_filepath=str(metadata_csv),
        model_filepath='model.keras',
        data_alias_enriched_hex='data_alias',
        model_alias_hex='model_alias',
        epoch=1,
    )

    assert '1' in calls[0]['output_filepath']


@pytest.mark.unit
def test_orchestrate_single_run_accepts_numpy_float_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_csv = tmp_path / 'meta_float_controls.csv'
    pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'combined_sigma': [1.0],
        }
    ).to_csv(metadata_csv, index=False)

    monkeypatch.setattr(
        um,
        'process_subdf',
        lambda *args, **kwargs: (
            [],
            {2.0: [np.zeros(1), np.zeros(1), np.zeros(1)]},
            (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        ),
    )

    output_paths = {
        'results_csv': str(tmp_path / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    um.orchestrate_single_run(
        N=np.float64(1.0),
        model_filepath='model.keras',
        metadata_filepath=str(metadata_csv),
        output_dir=str(tmp_path / 'out'),
        kwargs={
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        workers=np.float64(1.0),
        min_exp=np.float64(-1.0),
        max_exp=np.float64(1.0),
        output_paths=output_paths,
        save_eval_images=False,
        single_parallel=False,
        write_histograms=False,
    )

    assert Path(output_paths['results_csv']).exists()


@pytest.mark.unit
def test_uncropped_main_decodes_full_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    decoded = []
    calls = []
    find_calls = []
    model_dir = 'data_alias/UNET/ATTN/MAE/z_scale/DO0p2/ACTleakyrelu/OUTnone/DACTleakyrelu/DOUTnone'
    model_path = tmp_path / 'models_root' / model_dir / 'checkpoints' / 'model.keras'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text('x', encoding='utf-8')

    class _Future:
        def result(self):
            return None

    class _Exec:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            calls.append(kwargs)
            return _Future()

    monkeypatch.setattr(um, 'sys', type('Sys', (), {'argv': ['uncropped_metrics.py']}))
    monkeypatch.setattr(um, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        um,
        'load_config',
        lambda **kwargs: {
            'uncropped_metrics': {
                'data_kwargs': {
                    'kwargs_data': {
                        'low': 1.0,
                        'log_domain_clip_max': 1.0,
                        'data_alias_enriched_hex': 'data_alias',
                        'model_alias_hex': 'model_alias_cfg',
                    }
                },
                'metadata_filepath': str(tmp_path / 'meta.csv'),
                'models_dir': 'models',
                'model_prototype': '*.keras',
                'modulo': 1,
                'max_workers': 1,
                'uncropped_workers': 1,
                'checkpoint_info_filename': 'checkpoint_info.txt',
            }
        },
    )
    monkeypatch.setattr(
        um,
        'find_best_performing_models',
        lambda *args, **kwargs: find_calls.append(kwargs) or {
            model_dir: pd.DataFrame({'filepath': [str(model_path)], 'epoch': ['001']})
        },
    )
    monkeypatch.setattr(um, '_decode_models_dir', lambda value: decoded.append(value) or {'model_alias_hex': 'decoded_alias'})
    monkeypatch.setattr(um, 'set_checkpoint_info_filename', lambda *_: None)
    monkeypatch.setattr(um, 'set_log_domain_clip_max', lambda *_: None)
    monkeypatch.setattr(um, 'ProcessPoolExecutor', _Exec)
    monkeypatch.setattr(um, 'as_completed', lambda futures: futures)
    monkeypatch.setattr(um, 'orchestrate_uncropped_evaluation', lambda **kwargs: None)

    um.main()

    assert decoded == [str(model_path.parent.parent)]
    assert find_calls[0]['condition_kwargs']['config_model_alias_hex'] == 'model_alias_cfg'
    assert len(calls) == 1
    assert calls[0]['model_alias_hex'] == 'decoded_alias'


@pytest.mark.unit
def test_uncropped_module_main_guard_executes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_csv = tmp_path / 'meta_main.csv'
    pd.DataFrame(
        {
            'exp_ratio': [2.0],
            'location': ['a.fits'],
            'name': ['a'],
            'exp_time': [100.0],
            'new_exp_time': [50.0],
            'sci_pi_last_name': ['FABER'],
        }
    ).to_csv(metadata_csv, index=False)

    monkeypatch.setattr(utils_mod, 'candidates_based_on_range', lambda row, kwargs: [{'exp_ratio': row['exp_ratio'], 'new_exp_time': row['new_exp_time'], 'location': row['location'], 'name': row['name']}])
    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        starter,
        'load_config',
        lambda **kwargs: {
            'uncropped_metrics': {
                'data_kwargs': {
                    'kwargs_data': {
                        'low': 1.0,
                        'log_domain_clip_max': 1.0,
                        'data_alias_enriched_hex': 'data_alias',
                        'model_alias_hex': 'model_alias_cfg',
                    }
                },
                'metadata_filepath': str(metadata_csv),
                'uncropped_output_dir': str(tmp_path / 'out'),
                'uncropped_combined_images_dir': str(tmp_path / 'combined'),
                'uncropped_results_csv': 'results.csv',
                'uncropped_org_catalog_csv': 'org.csv',
                'uncropped_noisy_catalog_csv': 'noisy.csv',
                'uncropped_rec_catalog_csv': 'rec.csv',
                'uncropped_hist_data_csv': 'hist.csv',
                'uncropped_hist_png_template': 'hist_{exp_ratio}.png',
                'uncropped_n': 1,
                'uncropped_sampled_data_csv': str(tmp_path / 'sampled.csv'),
                'models_dir': 'models',
                'model_prototype': '*.keras',
                'modulo': 1,
                'max_workers': 1,
                'kwargs_source': {
                    'uncropped_patch_size': (2, 2, 1),
                    'uncropped_stride': (2, 2, 1),
                    'uncropped_weighting': 'average',
                    'uncropped_batch_size': 1,
                    'nan_value': 0.0,
                    'posinf_value': 0.0,
                    'neginf_value': 0.0,
                    'sigma_key': 'combined_sigma',
                    'type_of_image': 'SCI',
                    'noise_fn': lambda img, row, sigma: img,
                    'uncropped_use_mosaic': True,
                },
                'uncropped_workers': 1,
                'hist_min_exp': -1,
                'hist_max_exp': 1,
                'uncropped_save_images': False,
                'uncropped_save_combined_images': False,
                'filter_by_last_name': False,
                'last_name_col': 'sci_pi_last_name',
                'last_name_filter_value': ['FABER'],
                'checkpoint_info_filename': 'checkpoint_info.txt',
            }
        },
    )

    runpy.run_module('src.evaluation.uncropped_metrics', run_name='__main__')
