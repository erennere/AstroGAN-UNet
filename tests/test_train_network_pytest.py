"""Coverage for src/training/new_train.py orchestration."""
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
def test_train_network_runs_generator_only_path(tmp_path: Path, mock_cfg, tiny_unet, monkeypatch: pytest.MonkeyPatch):
    train_dir = tmp_path / 'train'
    eval_dir = tmp_path / 'eval'
    results_dir = tmp_path / 'results'
    train_dir.mkdir()
    eval_dir.mkdir()
    results_dir.mkdir()
    (train_dir / 'sample_train.fits').write_text('train', encoding='utf-8')
    (eval_dir / 'sample_eval.fits').write_text('eval', encoding='utf-8')

    data_cfg = dict(mock_cfg['new_train']['data'])
    data_cfg.update(
        {
            'training_path': str(train_dir),
            'eval_path': str(eval_dir),
            'results_path': str(results_dir),
            'checkpoint_custom_epoch': None,
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras', 'compile': False},
        }
    )

    train_batch = (
        tf.zeros((1, 64, 64, 1), dtype=tf.float32),
        tf.ones((1, 64, 64, 1), dtype=tf.float32),
        tf.constant([[b'sample_train.fits', b'1.0', b'2.0', b'3.0']], dtype=tf.string),
    )
    eval_batch = (
        tf.zeros((1, 64, 64, 1), dtype=tf.float32),
        tf.ones((1, 64, 64, 1), dtype=tf.float32),
        tf.constant([[b'sample_eval.fits', b'1.0', b'2.0', b'3.0']], dtype=tf.string),
    )

    def fake_create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32, scaling=None, augment=True):
        if augment:
            return tf.data.Dataset.from_tensors(train_batch)
        return tf.data.Dataset.from_tensors(eval_batch)

    monkeypatch.setattr(new_train_mod, 'network', lambda *args, **kwargs: tiny_unet)
    monkeypatch.setattr(new_train_mod, 'create_tf_dataset', fake_create_tf_dataset)
    monkeypatch.setattr(new_train_mod, 'load_model', lambda *args, **kwargs: None)

    new_train_mod.train_network(
        input_shape=(64, 64, 1),
        n_epochs=1,
        kwargs_data=data_cfg,
        kwargs_network=dict(mock_cfg['new_train']['training']['network_kwargs']),
        data_generator=lambda *args, **kwargs: None,
        batch_size=1,
        optimizer=tf.keras.optimizers.Adam,
        change_learning_rate=[(0, 1e-3)],
        G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        learning_rate=1e-3,
        beta_1=0.5,
        start_from_best=False,
        start_from_last=False,
        save_freq=1,
        eval_save_percentage=100,
        ds_save_percentage=100,
        scaling='min_max',
        discriminator_kwargs={},
        gan_kwargs={'loss_fn': tf.keras.losses.BinaryCrossentropy(), 'adversarial_loss_weight': 1.0, 'reconstruction_loss_weight': 1.0, 'label_smoothing': 0.0},
        use_gan=False,
        training_results_dir=str(tmp_path / 'train_results'),
        training_metrics_csv_path=str(tmp_path / 'train_results' / 'history.csv'),
        training_history_json_path=str(tmp_path / 'train_results' / 'history.json'),
        validation_loss_filename='validation_loss.txt',
        training_metrics_filename='training_metrics.txt',
        config=mock_cfg['new_train'],
    )

    assert (tmp_path / 'train_results' / 'history.json').exists()
    assert (tmp_path / 'train_results' / 'validation_loss.txt').exists()
    assert (tmp_path / 'train_results' / 'training_metrics.txt').exists()
    assert (results_dir / 'model_001.keras').exists() or any(results_dir.glob('*.keras'))


@pytest.mark.unit
def test_train_network_runs_gan_path_with_dummy_gan(tmp_path: Path, mock_cfg, tiny_unet, monkeypatch: pytest.MonkeyPatch):
    train_dir = tmp_path / 'train_gan'
    eval_dir = tmp_path / 'eval_gan'
    results_dir = tmp_path / 'results_gan'
    train_dir.mkdir()
    eval_dir.mkdir()
    results_dir.mkdir()
    (train_dir / 'sample_train.fits').write_text('train', encoding='utf-8')
    (eval_dir / 'sample_eval.fits').write_text('eval', encoding='utf-8')

    data_cfg = dict(mock_cfg['new_train']['data'])
    data_cfg.update(
        {
            'training_path': str(train_dir),
            'eval_path': str(eval_dir),
            'results_path': str(results_dir),
            'checkpoint_custom_epoch': None,
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras', 'compile': False},
            'training': True,
        }
    )

    train_batch = (
        tf.zeros((1, 64, 64, 1), dtype=tf.float32),
        tf.ones((1, 64, 64, 1), dtype=tf.float32),
        tf.constant([[b'sample_train.fits', b'1.0', b'2.0', b'3.0']], dtype=tf.string),
    )
    eval_batch = (
        tf.zeros((1, 64, 64, 1), dtype=tf.float32),
        tf.ones((1, 64, 64, 1), dtype=tf.float32),
        tf.constant([[b'sample_eval.fits', b'1.0', b'2.0', b'3.0']], dtype=tf.string),
    )

    def fake_create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32, scaling=None, augment=True):
        if augment:
            return tf.data.Dataset.from_tensors(train_batch)
        return tf.data.Dataset.from_tensors(eval_batch)

    class DummyGAN:
        def __init__(self, generator, discriminator, g_optimizer, d_optimizer, **kwargs):
            self.generator = generator
            self.discriminator = discriminator
            self.g_optimizer = g_optimizer
            self.d_optimizer = d_optimizer

        def compile(self):
            return None

        def fit(self, dataset, epochs, steps_per_epoch, callbacks):
            class _History:
                history = {'loss': [1.0]}
            return _History()

    class DummyCallback:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(new_train_mod, 'network', lambda *args, **kwargs: tiny_unet)
    monkeypatch.setattr(new_train_mod, 'get_discriminator', lambda *args, **kwargs: object())
    monkeypatch.setattr(new_train_mod, 'GAN', DummyGAN)
    monkeypatch.setattr(new_train_mod, 'Callback', DummyCallback)
    monkeypatch.setattr(new_train_mod, 'create_tf_dataset', fake_create_tf_dataset)
    monkeypatch.setattr(new_train_mod, 'load_model', lambda *args, **kwargs: None)

    new_train_mod.train_network(
        input_shape=(64, 64, 1),
        n_epochs=1,
        kwargs_data=data_cfg,
        kwargs_network=dict(mock_cfg['new_train']['training']['network_kwargs']),
        data_generator=lambda *args, **kwargs: None,
        batch_size=1,
        optimizer=tf.keras.optimizers.Adam,
        change_learning_rate=[(0, 1e-3)],
        G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        learning_rate=1e-3,
        beta_1=0.5,
        start_from_best=False,
        start_from_last=False,
        save_freq=1,
        eval_save_percentage=100,
        ds_save_percentage=100,
        scaling='min_max',
        discriminator_kwargs={},
        gan_kwargs={
            'loss_fn': tf.keras.losses.BinaryCrossentropy(),
            'adversarial_loss_weight': 1.0,
            'reconstruction_loss_weight': 1.0,
            'label_smoothing': 0.0,
        },
        use_gan=True,
        training_results_dir=str(tmp_path / 'train_results_gan'),
        training_metrics_csv_path=str(tmp_path / 'train_results_gan' / 'history.csv'),
        training_history_json_path=str(tmp_path / 'train_results_gan' / 'history.json'),
        validation_loss_filename='validation_loss.txt',
        training_metrics_filename='training_metrics.txt',
        config=mock_cfg['new_train'],
    )

    assert (tmp_path / 'train_results_gan' / 'history.json').exists()
