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

    output_paths = {
        'results_csv': str(tmp_path / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    um.main(
        N=1,
        model_filepath='model.keras',
        metadata_filepath=str(metadata_csv),
        output_dir=str(tmp_path / 'plots'),
        kwargs={'sigma_key': 'combined_sigma'},
        workers=1,
        min_exp=-1,
        max_exp=1,
        output_paths=output_paths,
        save_eval_images=False,
    )

    assert Path(output_paths['results_csv']).exists()
    assert Path(output_paths['hist_data_csv']).exists()


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
            'evaluation': {
                'data_kwargs': {
                    'kwargs_data': {'low': 1.0}
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
                'filter_by_last_name': False,
                'last_name_col': 'sci_pi_last_name',
                'last_name_filter_value': ['FABER'],
            }
        },
    )

    runpy.run_module('src.evaluation.uncropped_metrics', run_name='__main__')
