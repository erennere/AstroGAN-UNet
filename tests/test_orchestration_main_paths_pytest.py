"""Coverage tests for main orchestration paths and large plotting blocks."""
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

from src.data import create_dataset as create_dataset_mod
from src.data import mast as mast_mod
from src.evaluation import metrics as metrics_mod
from src.training import new_train as new_train_mod
from src.visualization import prepare_images as prep_images_mod


@pytest.mark.unit
def test_metrics_plot_source_comparison_functions_save_outputs(tmp_path: Path):
    image = np.ones((32, 32), dtype=np.float32)
    x_org = np.array([10.0, 20.0])
    y_org = np.array([10.0, 20.0])
    x_rec = np.array([11.0, 19.0])
    y_rec = np.array([11.0, 19.0])
    matched = np.array([0])
    unmatched = np.array([1])

    out_simple = tmp_path / 'comparison.png'
    metrics_mod.plot_source_comparison(
        image,
        image,
        image,
        x_org,
        y_org,
        x_rec,
        y_rec,
        matched,
        unmatched,
        matched,
        unmatched,
        str(out_simple),
    )

    out_sep = tmp_path / 'comparison_sep.png'
    a = np.array([1.0, 1.2])
    b = np.array([0.8, 1.1])
    theta = np.array([0.0, 0.2])
    metrics_mod.plot_source_comparison_sep(
        image,
        image,
        image,
        x_org,
        y_org,
        x_rec,
        y_rec,
        matched,
        unmatched,
        matched,
        unmatched,
        a,
        b,
        a,
        b,
        theta,
        theta,
        str(out_sep),
    )

    assert out_simple.exists()
    assert out_sep.exists()


@pytest.mark.unit
def test_metrics_plot_source_comparison_invalid_and_scaling_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Invalid image type should return early.
    assert metrics_mod.plot_source_comparison(
        'bad',
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0]),
        np.array([], dtype=int),
        np.array([0]),
        np.array([], dtype=int),
        str(tmp_path / 'bad.png'),
    ) is None

    monkeypatch.setattr(metrics_mod, 'scale_image', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('scale fail')))
    assert metrics_mod.plot_source_comparison_sep(
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0]),
        np.array([], dtype=int),
        np.array([0]),
        np.array([], dtype=int),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        str(tmp_path / 'scale_fail.png'),
    ) is None


@pytest.mark.unit
def test_find_best_performing_models_selects_checkpoint_files(tmp_path: Path):
    models_root = tmp_path / 'models'
    parts = [
        'gan',
        'attn',
        'mse',
        'alias',
        'min_max',
        'drop0',
        'relu',
        'sigmoid',
        'lrelu',
        'sigmoid',
    ]
    checkpoints_dir = models_root
    for part in parts:
        checkpoints_dir = checkpoints_dir / part
    checkpoints_dir = checkpoints_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for filename in ('best_model_010.keras', 'model_005.keras', 'final_model.keras'):
        (checkpoints_dir / filename).write_text('x', encoding='utf-8')

    selected = metrics_mod.find_best_performing_models(
        str(models_root),
        condition=lambda info: info['epoch'] >= 0,
        filter_model=lambda files, *args: files,
        model_prototype='*.keras',
        n=3,
        index=0,
        concurrent_workers=1,
    )

    assert len(selected) == 1
    selected_files = list(selected.values())[0]
    assert any('best_model_010.keras' in path for path in selected_files)


@pytest.mark.unit
def test_find_best_performing_models_rejects_empty_prototype(tmp_path: Path):
    with pytest.raises(ValueError):
        metrics_mod.find_best_performing_models(
            str(tmp_path),
            condition=lambda _: True,
            filter_model=lambda files, *args: files,
            model_prototype='',
            n=1,
            index=0,
            concurrent_workers=1,
        )


@pytest.mark.unit
def test_reconstruct_patch_with_and_without_scaling(monkeypatch: pytest.MonkeyPatch):
    class DummyModel:
        def predict(self, arr):
            return np.asarray(arr, dtype=np.float32) * 0.5

    noisy_patch = np.ones((16, 16), dtype=np.float32)

    out_direct = metrics_mod._reconstruct_patch(
        noisy_patch,
        scales=None,
        model=DummyModel(),
        use_mosaic=False,
        patch_size=(8, 8, 1),
        stride=(4, 4, 1),
        weighting='average',
        batch_size=2,
    )
    assert out_direct is not None
    assert out_direct.shape == noisy_patch.shape

    monkeypatch.setattr(
        metrics_mod,
        'sliding_window_inference',
        lambda image, model, *args, **kwargs: image,
    )

    def scale_fn(image):
        return image * 2.0, 2.0

    def descale_fn(image, factor):
        return image / factor

    out_scaled = metrics_mod._reconstruct_patch(
        noisy_patch,
        scales=(scale_fn, descale_fn),
        model=DummyModel(),
        use_mosaic=True,
        patch_size=(8, 8, 1),
        stride=(4, 4, 1),
        weighting='average',
        batch_size=2,
    )
    assert out_scaled is not None
    assert out_scaled.shape == noisy_patch.shape


@pytest.mark.unit
def test_metrics_main_non_parallel_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []

    monkeypatch.setattr(metrics_mod, 'get_test_images', lambda *args, **kwargs: pd.DataFrame({'location': ['a'], 'exp_time': [1.0]}))
    monkeypatch.setattr(metrics_mod, 'find_best_performing_models', lambda *args, **kwargs: {'gan/min_max': ['m1.keras']})
    monkeypatch.setattr(metrics_mod, '_decode_models_dir', lambda *_: {})
    monkeypatch.setattr(metrics_mod, 'decide_scale', lambda *_: None)

    def fake_process_models(job, kwargs_source, workers, frac, parallel, **kwargs):
        calls.append((job, kwargs_source, workers, frac, parallel, kwargs))

    monkeypatch.setattr(metrics_mod, 'process_models', fake_process_models)

    metrics_mod.main(
        models_dir=str(tmp_path / 'models'),
        data_kwargs={'kwargs_data': {}},
        model_kwargs={},
        kwargs_source={'distance_threshold': 2.0},
        total_workers=2,
        max_workers=1,
        frac=0.5,
        condition=lambda _: True,
        filter_model=lambda files, *args: files,
        n=1,
        model_prototype='*.keras',
        all_metrics_csv=str(tmp_path / 'all.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg.csv'),
        org_catalog_csv=str(tmp_path / 'org.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy.csv'),
        rec_catalog_csv=str(tmp_path / 'rec.csv'),
        parallel=False,
        parallel_epoch=False,
        scaling='min_max',
        index=0,
        concurrent_workers=1,
    )

    assert len(calls) == 1


@pytest.mark.unit
def test_prepare_images_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_path = tmp_path / 'metadata.csv'
    pd.DataFrame(
        {
            'location': ['test/file.fits'],
            'exp_time': [120.0],
            'target': ['M51'],
            'dataset': ['id001'],
        }
    ).to_csv(metadata_path, index=False)

    cfg = {
        'prepare_images': {
            'data_kwargs': {'ratio_initial': 2, 'ratio_count': 1, 'ratio_growth': 1, 'type_of_image': 'SCI'},
            'output_dir': str(tmp_path / 'out'),
            'low': 1,
            'metadata_filepath': str(metadata_path),
            'dataset': 'dataset',
            'sample_n': 1,
            'ps': 16,
            'kwargs_source': {'distance_threshold': 2.0, 'func': object()},
            'exp_column': 'exp_time',
            'targ_col': 'target',
            'type_of_image': 'SCI',
            'ratio_initial': 2,
            'ratio_count': 1,
            'ratio_growth': 1,
            'model_dir': 'models',
            'model_prototype': '*.keras',
            'scaling': 'min_max',
        }
    }

    monkeypatch.setattr(prep_images_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(prep_images_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(prep_images_mod, 'find_best_performing_models', lambda *args, **kwargs: 'dummy_model.keras')
    monkeypatch.setattr(prep_images_mod, 'read_checkpoint_info', lambda *_: {'scaling': 'min_max'})
    monkeypatch.setattr(prep_images_mod, 'load_checkpoint_model', lambda *args, **kwargs: object())

    calls = {'plot': 0, 'detect': 0}

    monkeypatch.setattr(
        prep_images_mod,
        'create_image',
        lambda row, model, kwargs_data, ps=256: [
            (
                np.ones((16, 16), dtype=np.float32),
                [np.ones((16, 16), dtype=np.float32)],
                [np.ones((16, 16), dtype=np.float32)],
                [2.0],
            )
        ],
    )
    monkeypatch.setattr(prep_images_mod, 'create_composite_plot', lambda *args, **kwargs: calls.__setitem__('plot', calls['plot'] + 1))
    monkeypatch.setattr(prep_images_mod, 'coordinate_detect_source', lambda *args, **kwargs: calls.__setitem__('detect', calls['detect'] + 1))

    prep_images_mod.main()

    assert calls['plot'] == 1
    assert calls['detect'] == 1


@pytest.mark.unit
def test_mast_main_fetch_resolve_and_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    metadata_out = tmp_path / 'mast' / 'metadata.csv'
    save_dir = tmp_path / 'downloads'

    cfg = {
        'mast': {
            'mission': 'HST',
            'filters': {'foo': 'bar'},
            'main_column': 'exp',
            'max_requests': 2,
            'reset_after': 3,
            'max_workers': 1,
            'download': True,
            'chunk_size': 10,
            'resolve_max_retries': 1,
            'resolve_retry_delay': 0.01,
            'metadata_output': str(metadata_out),
            'id_column': 'dataset_id',
            'url_column': 'resolved_url',
            'fetch_metadata': True,
            'resolve_urls': True,
            'prefer_token': 'drz',
            'save_dir': str(save_dir),
        }
    }

    base_table = pd.DataFrame({'dataset_id': ['A'], 'exp': [100.0]})
    merged_table = pd.DataFrame({'dataset_id': ['A'], 'exp': [100.0], 'resolved_url': ['https://resolved/A.fits']})

    async def fake_download_images(ids, urls, save_dir, max_requests=5, reset_after=10):
        return {'count': len(ids), 'save_dir': save_dir}

    monkeypatch.setattr(mast_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(mast_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(mast_mod, 'filter_out_mast', lambda mission, filters: base_table.copy())
    monkeypatch.setattr(mast_mod, 'merge_products_with_metadata', lambda *args, **kwargs: merged_table.copy())
    monkeypatch.setattr(mast_mod, 'download_images', fake_download_images)

    mast_mod.main()

    assert metadata_out.exists()
    loaded = pd.read_csv(metadata_out)
    assert 'resolved_url' in loaded.columns


@pytest.mark.unit
def test_create_dataset_main_delegates_to_control_flow(monkeypatch: pytest.MonkeyPatch):
    calls = []

    cfg = {
        'create_dataset': {
            'dataset_dir': 'd',
            'metadata_filepath': 'm.csv',
            'survey_column': 'survey',
            'exp_column': 'exp',
            'id_column': 'id',
            'url_column': 'url',
            'allowed_survey': ['IR'],
            'split_dirs': ['train', 'test', 'eval'],
            'originals_subdir': 'orig',
            'masked_images_dirname': 'masked',
            'file_extension': '.fits',
            'filtered_metadata_output_file': 'f.csv',
            'noisy_filtered_metadata_output_file': 'n.csv',
            'cropped_stats_output_file': 'c.csv',
            'url_filename_split_token': '/',
            'crop_name_separator': '_',
            'crop_prefix_parts': 1,
            'original_filename_suffix': '_drz.fits',
            'stats_column_tokens': ['mean', 'median', 'std', 'max', 'abs'],
            'temp_index_column': 'temp_index',
            'filename_column': 'filename',
            'original_filename_column': 'original_filename',
            'location_col': 'location',
            'masked_filename_prefix': 'masked_',
            'stats_column_map': {
                'mean_bkg': 'mean_bkg',
                'median_bkg': 'median_bkg',
                'std_bkg': 'std_bkg',
                'max_bkg': 'max_bkg',
                'abs_mean': 'abs_mean',
                'abs_median': 'abs_median',
                'mean_src': 'mean_src',
                'median_src': 'median_src',
                'std_src': 'std_src',
                'max_src': 'max_src',
            },
            'original_stats_prefix': 'orig_',
            'max_iterations': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'download': False,
            'cropping': True,
            'stats_on_crops': False,
            'save': False,
            'size': 10,
            'low': 1,
            'high': 1000,
            'seed': 42,
            'split': [60, 20, 20],
            'max_requests': 2,
            'reset_after': 5,
            'type_of_image': 'SCI',
            'sigma': 3,
            'nsigma': 2,
            'npixels': 5,
            'footprint_radius': 3,
            'maxiters': 5,
            'bkg_box_size': 32,
            'exclude_percentile': 10.0,
            'ps': 64,
            'max_workers': 1,
            'step': 5,
            'filter_surveys': True,
            'filter_by_last_name': False,
            'last_name_filter_value': [],
            'last_name_col': 'pi',
        }
    }

    monkeypatch.setattr(create_dataset_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(create_dataset_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(create_dataset_mod, 'control_flow', lambda **kwargs: calls.append(kwargs))

    create_dataset_mod.main()

    assert len(calls) == 1


@pytest.mark.unit
def test_new_train_main_delegates_to_train_network(monkeypatch: pytest.MonkeyPatch):
    calls = []
    cfg = {
        'new_train': {
            'training': {
                'patch_size': [16, 16, 1],
                'n_epochs': 1,
                'data_kwargs': {'training_path': 'train', 'eval_path': 'eval', 'results_path': 'results'},
                'network_kwargs': {},
                'discriminator_kwargs': {},
                'gan_kwargs': {},
                'data_generator': 'prepare_data',
                'batch_size': 2,
                'optimizer': 'adam',
                'change_learning_rate': False,
                'g_loss_fn': 'mse',
                'learning_rate': 1e-3,
                'beta_1': 0.9,
                'start_from_best': False,
                'start_from_last': False,
                'save_freq': 1,
                'eval_save_percentage': 0.5,
                'ds_save_percentage': 0.5,
                'scaling': 'min_max',
                'use_gan': False,
                'training_results_dir': 'results',
                'training_metrics_csv_path': 'metrics.csv',
                'training_history_json_path': 'history.json',
                'validation_loss_filename': 'val.png',
                'training_metrics_filename': 'train.png',
            }
        }
    }

    monkeypatch.setattr(new_train_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(new_train_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(new_train_mod, 'train_network', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(new_train_mod.tf.config.experimental, 'list_physical_devices', lambda *_: [])

    new_train_mod.main()

    assert len(calls) == 1
