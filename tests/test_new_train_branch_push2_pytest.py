"""Additional branch-heavy tests for src/training/new_train.py."""
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

import src.training.new_train as new_train_mod


@pytest.mark.unit
def test_prepare_data_with_info_cached_missing_location_logs_and_fallback(monkeypatch: pytest.MonkeyPatch):
    sampled_data = pd.DataFrame({'location': ['a.fits'], 'tok': ['x']})
    info_cached = pd.DataFrame({'other': [1]})

    monkeypatch.setattr(new_train_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        new_train_mod,
        'apply_scaling_and_stats',
        lambda simulated, clean, scaling, stats_name: (simulated, clean, stats_name),
    )
    monkeypatch.setattr(new_train_mod.random, 'choice', lambda seq: seq[0])

    out = list(
        new_train_mod.prepare_data(
            sampled_data=sampled_data,
            fit_data=None,
            training=False,
            scaling='min_max',
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            info_cached_df=info_cached,
            max_workers=1,
        )
    )
    assert len(out) == 1


@pytest.mark.unit
def test_prepare_data_warning_when_scaling_always_none(monkeypatch: pytest.MonkeyPatch):
    sampled_data = pd.DataFrame({'location': ['a.fits'], 'tok': ['x']})
    monkeypatch.setattr(new_train_mod, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(new_train_mod, 'apply_scaling_and_stats', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(new_train_mod.random, 'choice', lambda seq: seq[0])

    out = list(
        new_train_mod.prepare_data(
            sampled_data=sampled_data,
            fit_data=None,
            training=False,
            scaling='min_max',
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            info_cached_df=None,
            max_workers=1,
        )
    )
    assert out == []


@pytest.mark.unit
def test_data_augment_pluggable_fit_data_and_subsample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    new_train_mod.MEM_CACHED = None
    new_train_mod.MEM_CACHED_EVAL = None
    new_train_mod.MEM_CACHED_TEST = None
    new_train_mod.INFO_CACHED = None
    new_train_mod.INFO_CACHED_EVAL = None
    new_train_mod.INFO_CACHED_TEST = None

    metadata = pd.DataFrame(
        {
            'location': ['training/a.fits', 'training/b.fits'],
            'exp': [100.0, 120.0],
            'name': ['a', 'b'],
        }
    )
    metadata_fp = tmp_path / 'meta.csv'
    metadata.to_csv(metadata_fp, index=False)

    fit_df = pd.DataFrame({'x': [1.0]})
    fit_fp = tmp_path / 'fit.csv'
    fit_df.to_csv(fit_fp, index=False)

    monkeypatch.setattr(new_train_mod, 'ensure_parent_dir_exists', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(new_train_mod.random, 'shuffle', lambda seq: None)

    def fake_prepare_data(**kwargs):
        sampled = kwargs['sampled_data']
        for _idx, _row in sampled.iterrows():
            yield np.zeros((2, 2, 1), dtype=np.float32), np.zeros((2, 2, 1), dtype=np.float32), 's'

    monkeypatch.setattr(new_train_mod, 'prepare_data', fake_prepare_data)

    kwargs = {
        'metadata_filepath': str(metadata_fp),
        'fit_data_filepath': str(fit_fp),
        'sigma_kernel_requires_fit_data': True,
        'name_col': 'name',
        'location_col': 'location',
        'dataset': 'dataset',
        'exposure_col': 'exp',
        'times': 1,
        'low': 1,
        'high': 1000,
        'samples': 2,
        'val_samples': 2,
        'test_samples': 1,
        'training': True,
        'cache_raw_metadata': True,
        'training_cache_filepath': str(tmp_path / 'train_cache.csv'),
        'eval_cache_filepath': str(tmp_path / 'eval_cache.csv'),
        'test_cache_filepath': str(tmp_path / 'test_cache.csv'),
        'info_filepath': str(tmp_path / 'info.csv'),
        'sub_sample_train': 0.5,
        'sub_sample_eval': None,
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'sigma_key': 'sigma',
        'max_workers': 1,
        'type_of_image': 'SCI',
        'candidates_fn': (lambda row, kwargs: [{'location': row['location'], 'sigma': 1.0}]),
        'post_filter_fn': (lambda info, kwargs: info),
        'sample_fn': (lambda info, x, kwargs: info.head(x).reset_index(drop=True)),
        'sigma_kernel_fn': (lambda row, fit_data, sigma_key: 1.0),
        'noise_fn': (lambda clean, row, sigma: clean),
        'stats_name_fn': (lambda filepath, row, sigma_alias: 's'),
    }

    out = list(new_train_mod.data_augment_pluggable(images=[], kwargs_data=kwargs, scaling='min_max'))
    assert len(out) == 1


@pytest.mark.unit
def test_data_augment_pluggable_raises_for_missing_or_empty_fit_data(tmp_path: Path):
    metadata = pd.DataFrame({'location': ['training/a.fits'], 'exp': [100.0], 'name': ['a']})
    metadata_fp = tmp_path / 'meta.csv'
    metadata.to_csv(metadata_fp, index=False)

    kwargs = {
        'metadata_filepath': str(metadata_fp),
        'fit_data_filepath': str(tmp_path / 'missing_fit.csv'),
        'sigma_kernel_requires_fit_data': True,
        'exposure_col': 'exp',
        'times': 1,
        'low': 1,
        'high': 1000,
        'training': True,
        'samples': 1,
        'val_samples': 1,
        'test_samples': 1,
        'cache_raw_metadata': False,
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'sigma_key': 'sigma',
        'max_workers': 1,
        'type_of_image': 'SCI',
        'sub_sample_train': None,
        'sub_sample_eval': None,
        'candidates_fn': (lambda row, kwargs: [{'location': row['location'], 'sigma': 1.0}]),
        'post_filter_fn': (lambda info, kwargs: info),
        'sample_fn': (lambda info, x, kwargs: info.head(x).reset_index(drop=True)),
        'sigma_kernel_fn': (lambda row, fit_data, sigma_key: 1.0),
        'noise_fn': (lambda clean, row, sigma: clean),
        'stats_name_fn': (lambda filepath, row, sigma_alias: 's'),
    }

    with pytest.raises(FileNotFoundError):
        next(new_train_mod.data_augment_pluggable(images=[], kwargs_data=kwargs, scaling='min_max'))

    empty_fit = tmp_path / 'fit.csv'
    pd.DataFrame(columns=['x']).to_csv(empty_fit, index=False)
    kwargs['fit_data_filepath'] = str(empty_fit)
    with pytest.raises(ValueError):
        next(new_train_mod.data_augment_pluggable(images=[], kwargs_data=kwargs, scaling='min_max'))


@pytest.mark.unit
def test_data_augment_pluggable_uses_cached_training_eval_and_test(monkeypatch: pytest.MonkeyPatch):
    def fake_prepare_data(**kwargs):
        sampled = kwargs['sampled_data']
        for _idx, _row in sampled.iterrows():
            yield np.zeros((1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1), dtype=np.float32), 's'

    monkeypatch.setattr(new_train_mod, 'prepare_data', fake_prepare_data)

    base_kwargs = {
        'metadata_filepath': 'unused.csv',
        'exposure_col': 'exp',
        'times': 1,
        'low': 0,
        'high': 1000,
        'samples': 1,
        'val_samples': 1,
        'test_samples': 1,
        'cache_raw_metadata': True,
        'sub_sample_train': None,
        'sub_sample_eval': None,
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'sigma_key': 'sigma',
        'max_workers': 1,
        'type_of_image': 'SCI',
        'candidates_fn': (lambda row, kwargs: []),
        'post_filter_fn': (lambda info, kwargs: info),
        'sample_fn': (lambda info, x, kwargs: info),
        'sigma_kernel_fn': (lambda row, fit_data, sigma_key: 1.0),
        'noise_fn': (lambda clean, row, sigma: clean),
        'stats_name_fn': (lambda filepath, row, sigma_alias: 's'),
    }

    new_train_mod.MEM_CACHED = pd.DataFrame({'location': ['training/a.fits']})
    new_train_mod.MEM_CACHED_EVAL = pd.DataFrame({'location': ['eval/a.fits']})
    new_train_mod.MEM_CACHED_TEST = pd.DataFrame({'location': ['test/a.fits']})

    out_train = list(new_train_mod.data_augment_pluggable(images=[], kwargs_data={**base_kwargs, 'training': True}, scaling='min_max'))
    out_eval = list(new_train_mod.data_augment_pluggable(images=[], kwargs_data={**base_kwargs, 'training': False}, scaling='min_max'))
    out_test = list(new_train_mod.data_augment_pluggable(images=[], kwargs_data={**base_kwargs, 'training': True, 'test': True}, scaling='min_max'))

    assert len(out_train) == 1
    assert len(out_eval) == 1
    assert len(out_test) == 1


@pytest.mark.unit
def test_train_network_restore_none_model_and_empty_dataset_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    train_dir = tmp_path / 'train'
    eval_dir = tmp_path / 'eval'
    results_dir = tmp_path / 'results'
    train_dir.mkdir()
    eval_dir.mkdir()
    results_dir.mkdir()
    (train_dir / 'a.fits').write_text('x', encoding='utf-8')
    (eval_dir / 'b.fits').write_text('x', encoding='utf-8')

    cfg_data = {
        'training_path': str(train_dir),
        'eval_path': str(eval_dir),
        'results_path': str(results_dir),
        'checkpoint_custom_epoch': None,
        'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
    }

    tiny = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(1, 1),
    ])

    monkeypatch.setattr(new_train_mod, 'network', lambda *args, **kwargs: tiny)
    monkeypatch.setattr(new_train_mod, 'load_model', lambda *args, **kwargs: (None, 4))

    empty_ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.zeros((0, 8, 8, 1), dtype=tf.float32),
            tf.zeros((0, 8, 8, 1), dtype=tf.float32),
            tf.zeros((0, 1), dtype=tf.string),
        )
    ).batch(1)
    monkeypatch.setattr(new_train_mod, 'create_tf_dataset', lambda *args, **kwargs: empty_ds)

    with pytest.raises(ValueError, match='Training dataset produced zero batches'):
        new_train_mod.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data=cfg_data,
            kwargs_network={},
            data_generator=lambda *args, **kwargs: None,
            batch_size=1,
            optimizer=tf.keras.optimizers.Adam,
            change_learning_rate=[(0, 1e-3)],
            G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            learning_rate=1e-3,
            beta_1=0.9,
            start_from_best=False,
            start_from_last=False,
            save_freq=1,
            eval_save_percentage=100,
            ds_save_percentage=100,
            scaling='min_max',
            discriminator_kwargs=None,
            gan_kwargs=None,
            use_gan=False,
            training_results_dir=str(tmp_path / 'tr'),
            training_metrics_csv_path=str(tmp_path / 'tr' / 'metrics.csv'),
            training_history_json_path=str(tmp_path / 'tr' / 'history.json'),
            validation_loss_filename='val.txt',
            training_metrics_filename='train.txt',
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )


@pytest.mark.unit
def test_train_network_path_and_callable_guards(tmp_path: Path):
    with pytest.raises(ValueError, match='must be callable'):
        new_train_mod.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data={'training_path': None, 'eval_path': None, 'results_path': str(tmp_path), 'checkpoint_custom_epoch': None, 'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'}},
            kwargs_network={},
            data_generator=lambda *args, **kwargs: None,
            G_loss_fn='bad',
            use_gan=False,
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )

    with pytest.raises(FileNotFoundError):
        new_train_mod.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data={'training_path': str(tmp_path / 'missing'), 'eval_path': str(tmp_path / 'also_missing'), 'results_path': str(tmp_path), 'checkpoint_custom_epoch': None, 'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'}},
            kwargs_network={},
            data_generator=lambda *args, **kwargs: None,
            G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            use_gan=False,
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )


@pytest.mark.unit
def test_train_network_eval_empty_and_restored_gan_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    train_dir = tmp_path / 'train2'
    eval_dir = tmp_path / 'eval2'
    results_dir = tmp_path / 'results2'
    train_dir.mkdir()
    eval_dir.mkdir()
    results_dir.mkdir()
    (train_dir / 'a.fits').write_text('x', encoding='utf-8')
    (eval_dir / 'b.fits').write_text('x', encoding='utf-8')

    cfg_data = {
        'training_path': str(train_dir),
        'eval_path': str(eval_dir),
        'results_path': str(results_dir),
        'checkpoint_custom_epoch': None,
        'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
        'training': True,
    }

    base_gen = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(1, 1),
    ])

    class Restored:
        def __init__(self, generator):
            self.generator = generator

    class DummyGAN:
        def __init__(self, generator, discriminator, g_optimizer, d_optimizer, **kwargs):
            self.generator = generator
            self.g_optimizer = g_optimizer

        def compile(self):
            return None

        def fit(self, *args, **kwargs):
            class H:
                history = {'loss': [1.0]}
            return H()

    monkeypatch.setattr(new_train_mod, 'network', lambda *args, **kwargs: base_gen)
    monkeypatch.setattr(new_train_mod, 'load_model', lambda *args, **kwargs: (Restored(base_gen), 1))
    monkeypatch.setattr(new_train_mod, 'get_discriminator', lambda *args, **kwargs: object())
    monkeypatch.setattr(new_train_mod, 'GAN', DummyGAN)

    nonempty_train = tf.data.Dataset.from_tensors(
        (
            tf.zeros((1, 8, 8, 1), dtype=tf.float32),
            tf.zeros((1, 8, 8, 1), dtype=tf.float32),
            tf.constant([[b'a']], dtype=tf.string),
        )
    )
    empty_eval = tf.data.Dataset.from_tensor_slices(
        (
            tf.zeros((0, 8, 8, 1), dtype=tf.float32),
            tf.zeros((0, 8, 8, 1), dtype=tf.float32),
            tf.zeros((0, 1), dtype=tf.string),
        )
    ).batch(1)

    monkeypatch.setattr(
        new_train_mod,
        'create_tf_dataset',
        lambda *args, **kwargs: nonempty_train if kwargs.get('augment', False) else empty_eval,
    )

    with pytest.raises(ValueError, match='Validation dataset produced zero batches'):
        new_train_mod.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data=cfg_data,
            kwargs_network={},
            data_generator=lambda *args, **kwargs: None,
            batch_size=1,
            optimizer=tf.keras.optimizers.Adam,
            change_learning_rate=[(0, 1e-3)],
            G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            learning_rate=1e-3,
            beta_1=0.9,
            start_from_best=False,
            start_from_last=False,
            save_freq=1,
            eval_save_percentage=100,
            ds_save_percentage=100,
            scaling='min_max',
            discriminator_kwargs=None,
            gan_kwargs={
                'loss_fn': tf.keras.losses.BinaryCrossentropy(),
                'adversarial_loss_weight': 1.0,
                'reconstruction_loss_weight': 1.0,
                'label_smoothing': 0.0,
                'd_learning_rate': 1e-4,
            },
            use_gan=True,
            training_results_dir=str(tmp_path / 'tr2'),
            training_metrics_csv_path=str(tmp_path / 'tr2' / 'metrics.csv'),
            training_history_json_path=str(tmp_path / 'tr2' / 'history.json'),
            validation_loss_filename='val.txt',
            training_metrics_filename='train.txt',
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )
