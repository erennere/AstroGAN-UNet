"""Coverage for larger evaluation orchestration paths in metrics and uncropped_metrics."""
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

from src.evaluation import metrics as metrics_mod
from src.evaluation.metrics import process_models, process_single_model
from src.evaluation import uncropped_metrics as uncropped_mod


@pytest.mark.unit
def test_process_single_model_with_mocked_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    model_file = tmp_path / 'model_005.keras'
    model_file.write_text('placeholder', encoding='utf-8')
    image_path = tmp_path / 'image.fits'
    image_path.write_text('placeholder', encoding='utf-8')
    test_images_df = pd.DataFrame({
        'location': [str(image_path)],
        'sci_actual_duration': [100.0],
        'new_exp_time': [50.0],
        'combined_sigma': [1.5],
    })

    class DummyModel:
        pass

    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch + 2.0)
    monkeypatch.setattr(metrics_mod, 'compare_images', lambda *args, **kwargs: (
        {'TP': 1.0, 'FP': 0.0, 'FN': 0.0},
        ([1.0], [2.0]),
        ([0.1], [0.2]),
        (pd.DataFrame({'x': [1]}), pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [3]})),
    ))
    monkeypatch.setattr(metrics_mod, 'aggregate_df', lambda *args, **kwargs: {'TP': 1.0, 'FP': 0.0, 'FN': 0.0})
    monkeypatch.setattr(metrics_mod, 'save_fits', lambda *args, **kwargs: None)
    monkeypatch.setattr(metrics_mod, 'ensure_directory_exists', lambda path: path)
    monkeypatch.setattr(metrics_mod, 'scale_image', lambda img: (Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255)), 0.0, 1.0))
    monkeypatch.setattr(metrics_mod, 'decide_scale', lambda *_args, **_kwargs: None)

    metrics_df, aggregated_df, dfs = process_single_model(
        str(model_file),
        'model_a',
        None,
        test_images_df,
        kwargs_source={'thresh': 1.0},
        patch_size=(2, 2, 1),
        stride=(2, 2, 1),
        weighting='average',
        batch_size=1,
        frac=1.0,
        model_index=0,
        use_mosaic=True,
        gaussian_sigma=2,
        type_of_image='SCI',
        noise_fn=lambda base_image, row, sigma: base_image + sigma,
        combined_images_dir=str(tmp_path / 'combined'),
        png_dir=str(tmp_path / 'pngs'),
        org_dir=str(tmp_path / 'org'),
        noisy_dir=str(tmp_path / 'noisy'),
        rec_dir=str(tmp_path / 'rec'),
        model_alias_hex='alias',
    )

    assert not metrics_df.empty
    assert not aggregated_df.empty
    org_df, noisy_df, rec_df = dfs
    assert not org_df.empty
    assert not noisy_df.empty
    assert not rec_df.empty


@pytest.mark.unit
def test_process_models_sequential_writeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_file = str(tmp_path / 'model_010.keras')
    test_images_df = pd.DataFrame({'location': ['img.fits'], 'sci_actual_duration': [100.0], 'new_exp_time': [50.0], 'combined_sigma': [1.0]})

    monkeypatch.setattr(metrics_mod, 'process_single_model', lambda *args, **kwargs: (
        pd.DataFrame({'epoch': ['010'], 'metric': [1.0]}),
        pd.DataFrame({'epoch': ['010'], 'agg': [1.0]}),
        (pd.DataFrame({'x': [1]}), pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [3]})),
    ))

    all_metrics_csv = str(tmp_path / 'all_*_metrics.csv')
    aggregated_metrics_csv = str(tmp_path / 'agg_*_metrics.csv')
    org_catalog_csv = str(tmp_path / 'org_*_catalog.csv')
    noisy_catalog_csv = str(tmp_path / 'noisy_*_catalog.csv')
    rec_catalog_csv = str(tmp_path / 'rec_*_catalog.csv')

    all_metrics, aggregated_metrics, dfs = process_models(
        job=('model_dir', [model_file], None, test_images_df, {'model_alias_hex': 'alias'}),
        kwargs_source={'thresh': 1.0},
        workers=1,
        frac=1.0,
        parallel=False,
        patch_size=(2, 2, 1),
        stride=(2, 2, 1),
        weighting='average',
        batch_size=1,
        use_mosaic=True,
        gaussian_sigma=2,
        type_of_image='SCI',
        all_metrics_csv=all_metrics_csv,
        aggregated_metrics_csv=aggregated_metrics_csv,
        org_catalog_csv=org_catalog_csv,
        noisy_catalog_csv=noisy_catalog_csv,
        rec_catalog_csv=rec_catalog_csv,
    )

    assert not all_metrics.empty
    assert not aggregated_metrics.empty
    assert all(path.exists() for path in [
        tmp_path / 'all_alias_metrics.csv',
        tmp_path / 'agg_alias_metrics.csv',
        tmp_path / 'org_alias_catalog.csv',
        tmp_path / 'noisy_alias_catalog.csv',
        tmp_path / 'rec_alias_catalog.csv',
    ])
    org_df, noisy_df, rec_df = dfs
    assert not org_df.empty and not noisy_df.empty and not rec_df.empty


@pytest.mark.unit
def test_process_models_invalid_job_returns_empty_frames():
    all_metrics, aggregated_metrics, dfs = process_models(
        job=None,
        kwargs_source={},
        all_metrics_csv='all.csv',
        aggregated_metrics_csv='agg.csv',
        org_catalog_csv='org.csv',
        noisy_catalog_csv='noisy.csv',
        rec_catalog_csv='rec.csv',
    )
    assert all_metrics.empty
    assert aggregated_metrics.empty
    org_df, noisy_df, rec_df = dfs
    assert org_df.empty and noisy_df.empty and rec_df.empty


@pytest.mark.unit
def test_process_models_raises_for_non_string_output_templates(tmp_path: Path):
    with pytest.raises(ValueError):
        process_models(
            job=('model_dir', [], None, pd.DataFrame(), {'model_alias_hex': 'alias'}),
            kwargs_source={},
            workers=1,
            parallel=False,
            all_metrics_csv=None,
            aggregated_metrics_csv='agg.csv',
            org_catalog_csv='org.csv',
            noisy_catalog_csv='noisy.csv',
            rec_catalog_csv='rec.csv',
        )


@pytest.mark.unit
def test_uncropped_process_data_and_subdf_with_mocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata = pd.DataFrame({
        'name': ['img1'],
        'last_name': ['smith'],
        'exp_time': [100.0],
        'location': [str(tmp_path / 'image.fits')],
    })
    metadata_path = tmp_path / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    (tmp_path / 'image.fits').write_text('placeholder', encoding='utf-8')

    monkeypatch.setattr(uncropped_mod, 'candidates_based_on_range', lambda row, kwargs_data: [{
        'new_exp_time': 50.0,
        'exp_ratio': 2.0,
        'location': row['location'],
        'exp_time': row['exp_time'],
        'name': row['name'],
        'combined_sigma': 1.5,
    }])
    monkeypatch.setattr(uncropped_mod, 'ensure_parent_dir_exists', lambda path: path)

    filtered = uncropped_mod.process_data(
        str(metadata_path),
        kwargs_data={'low': 10, 'name_col': 'name', 'location_col': 'location', 'dataset': 'dataset', 'exposure_col': 'exp_time', 'stats_column_map': {}, 'original_stats_prefix': 'orig_'},
        kwargs_eval={'filter_by_last_name': False, 'last_name_filter_value': [], 'last_name_col': 'last_name'},
        output_filepath=str(tmp_path / 'filtered.csv'),
    )
    assert not filtered.empty
    assert (tmp_path / 'filtered.csv').exists()

    monkeypatch.setattr(uncropped_mod, 'load_checkpoint_model', lambda *args, **kwargs: object())
    monkeypatch.setattr(uncropped_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(uncropped_mod, 'decide_scale', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(uncropped_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(uncropped_mod, 'crop_image_generator', lambda *_args, **_kwargs: iter([np.ones((2, 2), dtype=np.float32)]))
    monkeypatch.setattr(uncropped_mod, '_reconstruct_patch', lambda noisy_patch, *args, **kwargs: noisy_patch + 1.0)
    monkeypatch.setattr(uncropped_mod, 'compare_images', lambda *args, **kwargs: (
        {'TP': 1.0},
        ([1.0], [2.0]),
        ([0.1], [0.2]),
        (pd.DataFrame({'x': [1]}), pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [3]})),
    ))

    results_df, all_hists, dfs = uncropped_mod.process_subdf(
        filtered,
        str(tmp_path / 'model_005.keras'),
        str(tmp_path / 'plots'),
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
            'noise_fn': lambda cropped_image, row, noisy_sigma: cropped_image + noisy_sigma,
            'uncropped_use_mosaic': True,
        },
        bins=np.array([0.0, 1.0, 2.0]),
        save_eval_images=False,
    )
    assert results_df
    assert 2.0 in all_hists
    org_df, noisy_df, rec_df = dfs
    assert not org_df.empty and not noisy_df.empty and not rec_df.empty
