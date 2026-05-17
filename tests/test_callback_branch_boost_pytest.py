"""Branch-focused tests for src/training/callback.py."""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.callback import Callback


class DummyModel:
    def __call__(self, x, training=False):
        return x

    def compiled_loss(self, y, pred):
        return tf.reduce_mean(tf.abs(y - pred))


def _empty_dataset(batch_size=1):
    x = tf.zeros((0, 2, 2, 1), dtype=tf.float32)
    y = tf.zeros((0, 2, 2, 1), dtype=tf.float32)
    m = tf.zeros((0, 1), dtype=tf.string)
    return tf.data.Dataset.from_tensor_slices((x, y, m)).batch(batch_size)


@pytest.mark.unit
def test_callback_create_directory_handles_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cb = Callback(
        dataset=_empty_dataset(),
        dataset_size=0,
        epoch=1,
        start_epoch=0,
        eval_save_percentage=10,
        ds_save_percentage=10,
        save_freq=1,
        sample_generator=None,
        kwargs_validation={},
        validation_dataset=_empty_dataset(),
        validation_dataset_size=0,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        change_learning_rate=[],
        batch_size=1,
        scaling='min_max',
        use_gan=False,
        training_results_dir=str(tmp_path),
        checkpoints_dir=str(tmp_path / 'ckpt'),
        training_metrics_csv_path=str(tmp_path / 'metrics.csv'),
        config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
    )
    monkeypatch.setattr('src.training.callback.ensure_directory_exists', lambda *_: (_ for _ in ()).throw(RuntimeError('x')))
    cb.create_directory(str(tmp_path / 'x'))


@pytest.mark.unit
def test_callback_on_epoch_end_handles_empty_validation_and_save_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cb = Callback(
        dataset=_empty_dataset(),
        dataset_size=0,
        epoch=1,
        start_epoch=0,
        eval_save_percentage=10,
        ds_save_percentage=10,
        save_freq=1,
        sample_generator=None,
        kwargs_validation={},
        validation_dataset=_empty_dataset(),
        validation_dataset_size=0,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        change_learning_rate=[(0, 1e-4)],
        batch_size=1,
        scaling='min_max',
        use_gan=False,
        training_results_dir=str(tmp_path),
        checkpoints_dir=str(tmp_path / 'ckpt'),
        training_metrics_csv_path=str(tmp_path / 'metrics.csv'),
        config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
    )
    cb.set_model(DummyModel())
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})

    monkeypatch.setattr('src.training.callback.save_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('save fail')))
    monkeypatch.setattr('src.training.callback.ensure_parent_dir_exists', lambda *_: (_ for _ in ()).throw(RuntimeError('mkdir fail')))

    cb.on_epoch_end(0, logs={'loss': 0.5})
    assert len(cb.training_metrics) == 1


@pytest.mark.unit
def test_callback_on_train_end_handles_write_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cb = Callback(
        dataset=_empty_dataset(),
        dataset_size=0,
        epoch=1,
        start_epoch=0,
        eval_save_percentage=10,
        ds_save_percentage=10,
        save_freq=1,
        sample_generator=None,
        kwargs_validation={},
        validation_dataset=_empty_dataset(),
        validation_dataset_size=0,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        change_learning_rate=[],
        batch_size=1,
        scaling='min_max',
        use_gan=False,
        training_results_dir=str(tmp_path),
        checkpoints_dir=str(tmp_path / 'ckpt'),
        training_metrics_csv_path=str(tmp_path / 'metrics.csv'),
        config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
    )
    cb.set_model(DummyModel())
    cb.training_metrics.append({'train_loss': 1.0, 'validation_loss': 1.0, 'epoch_time': 1.0})

    monkeypatch.setattr('src.training.callback.save_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('save fail')))
    monkeypatch.setattr('src.training.callback.ensure_parent_dir_exists', lambda *_: (_ for _ in ()).throw(RuntimeError('mkdir fail')))

    cb.on_train_end()
