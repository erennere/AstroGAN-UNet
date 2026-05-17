from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.training.new_train as nt
from src.training.callback import Callback
from src.training import math_helpers as mh


class _EchoModel:
    def __call__(self, x, training=False):
        return x


class _GanModel(_EchoModel):
    def __init__(self, with_reconstruction=True):
        if with_reconstruction:
            self.reconstruction_loss_fn = lambda y, p: tf.reduce_mean(tf.abs(y - p))


class _LossFnModel(_EchoModel):
    compiled_loss = None

    def loss_fn(self, y, p):
        return tf.reduce_mean(tf.abs(y - p))


class _BareModel:
    def __call__(self, x, training=False):
        return x


class _FakePreviewDataset:
    def __iter__(self):
        x = tf.ones((1, 2, 2, 1), dtype=tf.float32)
        y = tf.ones((1, 2, 2, 1), dtype=tf.float32)
        m = tf.constant([[b'a.fits', b'0', b'1', b'0', b'1']])
        yield x, y, m

    def skip(self, _index):
        return self

    def take(self, _n):
        return tf.data.Dataset.from_tensor_slices(
            (
                tf.zeros((0, 2, 2, 1), dtype=tf.float32),
                tf.zeros((0, 2, 2, 1), dtype=tf.float32),
                tf.zeros((0, 5), dtype=tf.string),
            )
        ).batch(1)


def _dataset_with_metadata(meta_row: list[bytes], batch=1):
    x = tf.ones((batch, 2, 2, 1), dtype=tf.float32)
    y = tf.ones((batch, 2, 2, 1), dtype=tf.float32)
    m = tf.constant([meta_row] * batch)
    return tf.data.Dataset.from_tensor_slices((x, y, m)).batch(batch)


def _mk_callback(tmp_path: Path, *, dataset, validation_dataset, validation_size=0, use_gan=False, scaling='min_max', batch_size=1, ds_save_percentage=0, config=None):
    if config is None:
        config = {
            'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'},
            'data': {'type_of_image': 'SCI'},
        }
    return Callback(
        dataset=dataset,
        dataset_size=1,
        epoch=1,
        start_epoch=0,
        eval_save_percentage=100,
        ds_save_percentage=ds_save_percentage,
        save_freq=1,
        sample_generator=None,
        kwargs_validation={},
        validation_dataset=validation_dataset,
        validation_dataset_size=validation_size,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        change_learning_rate=[],
        batch_size=batch_size,
        scaling=scaling,
        use_gan=use_gan,
        training_results_dir=str(tmp_path / 'results'),
        checkpoints_dir=str(tmp_path / 'checkpoints'),
        training_metrics_csv_path=str(tmp_path / 'results' / 'history.csv'),
        validation_loss_filename='validation_loss.txt',
        training_metrics_filename='training_metrics.txt',
        config=config,
    )


@pytest.mark.unit
def test_callback_config_validation_branches(tmp_path: Path):
    ds = _dataset_with_metadata([b's.fits'])
    with pytest.raises(ValueError, match='requires config'):
        _mk_callback(tmp_path, dataset=ds, validation_dataset=ds, config={'data': {'type_of_image': 'SCI'}})

    with pytest.raises(ValueError, match='must be a non-empty string'):
        _mk_callback(
            tmp_path,
            dataset=ds,
            validation_dataset=ds,
            config={'training': {'checkpoint_filename_pattern': ''}, 'data': {'type_of_image': 'SCI'}},
        )


@pytest.mark.unit
def test_callback_preview_break_and_processed_any_false_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ds = _dataset_with_metadata([b'a.fits', b'0', b'1', b'0', b'1'])
    cb = _mk_callback(
        tmp_path,
        dataset=ds,
        validation_dataset=_dataset_with_metadata([b'a.fits']),
        validation_size=0,
        use_gan=True,
        ds_save_percentage=0,
    )
    cb.set_model(_GanModel(with_reconstruction=False))
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})

    monkeypatch.setattr('src.training.callback.save_fits', lambda *args, **kwargs: None)
    cb.on_epoch_end(0, logs=None)


@pytest.mark.unit
def test_callback_preview_inner_break_and_inverse_assignment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ds = _dataset_with_metadata([b'a.fits', b'0.0', b'1.0', b'0.0', b'1.0'], batch=2)
    cb = _mk_callback(
        tmp_path,
        dataset=ds,
        validation_dataset=_dataset_with_metadata([b'v.fits']),
        validation_size=0,
        use_gan=False,
        ds_save_percentage=60,
    )
    cb.dataset_size = 2
    cb.set_model(_EchoModel())
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})

    monkeypatch.setattr('src.training.callback.save_fits', lambda *args, **kwargs: None)
    cb.on_epoch_end(0, logs={'loss': 0.1})


@pytest.mark.unit
def test_callback_preview_processed_any_false_debug_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cb = _mk_callback(
        tmp_path,
        dataset=_FakePreviewDataset(),
        validation_dataset=_dataset_with_metadata([b'v.fits']),
        validation_size=0,
        use_gan=False,
        ds_save_percentage=100,
    )
    cb.dataset_size = 1
    cb.set_model(_EchoModel())
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})
    monkeypatch.setattr('src.training.callback.save_fits', lambda *args, **kwargs: None)
    cb.on_epoch_end(0, logs={'loss': 0.1})


@pytest.mark.unit
def test_callback_inverse_and_validation_loss_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ds = _dataset_with_metadata([b'a.fits', b'0.0', b'1.0', b'0.0', b'1.0'])
    val = _dataset_with_metadata([b'v.fits'])

    # Use GAN with explicit reconstruction fn (line 301 path).
    cb = _mk_callback(tmp_path, dataset=ds, validation_dataset=val, validation_size=1, use_gan=True)
    cb.set_model(_GanModel(with_reconstruction=True))
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})

    monkeypatch.setattr('src.training.callback.save_fits', lambda *args, **kwargs: None)
    monkeypatch.setattr('src.training.callback.save_checkpoint_model', lambda *args, **kwargs: None)
    cb.on_epoch_end(0, logs={'d_loss': 1.0, 'g_loss': 0.5})


@pytest.mark.unit
def test_callback_loss_fn_and_fallback_loss_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ds = _dataset_with_metadata([b'a.fits'])
    val = _dataset_with_metadata([b'v.fits'])

    cb_loss_fn = _mk_callback(tmp_path, dataset=ds, validation_dataset=val, validation_size=1, use_gan=False)
    cb_loss_fn.set_model(_LossFnModel())
    cb_loss_fn.on_train_begin()
    cb_loss_fn.on_epoch_begin(0, logs={})
    cb_loss_fn.on_epoch_end(0, logs={'loss': 0.2})

    cb_fallback = _mk_callback(tmp_path, dataset=ds, validation_dataset=val, validation_size=1, use_gan=False)
    cb_fallback.set_model(_BareModel())
    cb_fallback.on_train_begin()
    cb_fallback.on_epoch_begin(0, logs={})

    # Force append mode branch by pre-creating CSV.
    csv_path = Path(cb_fallback.training_metrics_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text('epoch,train_loss,g_loss,validation_loss,epoch_time\n', encoding='utf-8')
    cb_fallback.on_epoch_end(0, logs={'loss': 0.2})


@pytest.mark.unit
def test_callback_gan_fallback_validation_loss_branch(tmp_path: Path):
    ds = _dataset_with_metadata([b'a.fits'])
    val = _dataset_with_metadata([b'v.fits'])
    cb = _mk_callback(tmp_path, dataset=ds, validation_dataset=val, validation_size=1, use_gan=True)
    cb.set_model(_GanModel(with_reconstruction=False))
    cb.on_train_begin()
    cb.on_epoch_begin(0, logs={})
    cb.on_epoch_end(0, logs={'d_loss': 0.1, 'g_loss': 0.1})


@pytest.mark.unit
def test_math_helpers_remaining_branches(monkeypatch: pytest.MonkeyPatch):
    assert mh.evenly_spaced_numbers(2, 2.1, 1) == []
    assert mh.evenly_spaced_numbers(1, 2, 0.1) == []
    assert mh._exp_bounds(pd.Series([0.0, np.nan, 0.0])) == (0, 0)

    image = np.ones((2, 2), dtype=np.float32)
    assert np.array_equal(mh.create_simulated_image_poisson(image, original_exposure=10.0, exposure_ratio=0.0), image)

    row = pd.Series({'exp_time': 10.0, 'exp_ratio': 2.0, 'full_median_bkg': 1.0})
    out = mh._simulated_image_from_poisson(image, row, new_sigma=1.0)
    assert out.shape == image.shape

    assert mh._stats_name_from_sigma('x.fits', row, 1.23456).startswith('x_1_23456')
    assert mh._stats_name_from_exp_ratio('x.fits', row, 0.0).startswith('x_2-0')

    bad = np.array([[1.0, np.nan]], dtype=np.float32)
    assert mh.adaptive_log_transform_and_normalize(bad) == (None, None, None, None)
    flat = np.ones((4, 4), dtype=np.float32)
    assert mh.adaptive_log_transform_and_normalize(flat) == (None, None, None, None)

    original_isfinite = mh.np.isfinite

    def fake_isfinite(value):
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return np.ones(arr.shape, dtype=bool)
        return False

    monkeypatch.setattr(mh.np, 'isfinite', fake_isfinite)
    assert mh.adaptive_log_transform_and_normalize(np.array([[1.0, 2.0]], dtype=np.float32)) == (None, None, None, None)
    monkeypatch.setattr(mh.np, 'isfinite', original_isfinite)

    assert mh.apply_scaling_and_stats(flat, flat, 'min_max', 'n') is None

    original_minmax = mh.min_max_normalization

    def fake_minmax(arr):
        if np.max(arr) == 1.0:
            return arr, None, 1.0
        return arr, 0.0, 1.0

    noisy = np.ones((2, 2), dtype=np.float32)
    clean = np.ones((2, 2), dtype=np.float32) * 2.0
    monkeypatch.setattr(mh, 'min_max_normalization', fake_minmax)
    assert mh.apply_scaling_and_stats(noisy, clean, 'min_max', 'x') is None

    def fake_minmax_clean_none(arr):
        if np.max(arr) == 2.0:
            return arr, None, 1.0
        return arr, 0.0, 1.0

    monkeypatch.setattr(mh, 'min_max_normalization', fake_minmax_clean_none)
    assert mh.apply_scaling_and_stats(noisy, clean, 'min_max', 'x') is None

    def fake_minmax_nonfinite(arr):
        return np.array([[np.nan]], dtype=np.float32), 0.0, 1.0

    monkeypatch.setattr(mh, 'min_max_normalization', fake_minmax_nonfinite)
    assert mh.apply_scaling_and_stats(noisy, clean, 'min_max', 'x') is None

    def fake_minmax_bad_clean_stats(arr):
        if np.max(arr) == 2.0:
            return np.array([[0.1]], dtype=np.float32), np.inf, 1.0
        return np.array([[0.2]], dtype=np.float32), 0.0, 1.0

    monkeypatch.setattr(mh, 'min_max_normalization', fake_minmax_bad_clean_stats)
    assert mh.apply_scaling_and_stats(noisy, clean, 'min_max', 'x') is None

    def fake_minmax_bad_noisy_stats(arr):
        if np.max(arr) == 1.0:
            return np.array([[0.2]], dtype=np.float32), np.inf, 1.0
        return np.array([[0.1]], dtype=np.float32), 0.0, 1.0

    monkeypatch.setattr(mh, 'min_max_normalization', fake_minmax_bad_noisy_stats)
    assert mh.apply_scaling_and_stats(noisy, clean, 'min_max', 'x') is None

    monkeypatch.setattr(mh, 'min_max_normalization', original_minmax)


@pytest.mark.unit
def test_new_train_remaining_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    img = np.ones((4, 4), dtype=np.float32)
    saved = {'n': 0}

    monkeypatch.setattr(nt, 'save_fits', lambda *args, **kwargs: saved.__setitem__('n', saved['n'] + 1))
    out = nt.poisson_noise_with_extra_components(img, ratio=0.5, exp_time=10.0, save=True, sv_name='a.fits', path=str(tmp_path))
    assert out is not None
    assert saved['n'] == 1

    class WeirdZero:
        def __init__(self):
            self.eq_calls = 0

        def __eq__(self, other):
            if other != 0:
                return False
            self.eq_calls += 1
            return self.eq_calls >= 2

        def __gt__(self, other):
            return False

    assert nt.black_level(10, 10, np.ones((10, 10), dtype=np.float32), ps=WeirdZero(), steps=1) is None


@pytest.mark.unit
def test_prepare_data_retry_and_debug_branches(monkeypatch: pytest.MonkeyPatch):
    # Initial pass reaches count==100 to hit debug branch at line 405.
    rows = pd.DataFrame({'location': ['ok.fits'] * 100, 'token': [f't{i}' for i in range(100)]})

    monkeypatch.setattr(nt, 'open_fits', lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(nt, 'apply_scaling_and_stats', lambda sim, clean, *_args, **_kwargs: (sim, clean, ['s']))

    out = list(
        nt.prepare_data(
            sampled_data=rows,
            fit_data=None,
            training=False,
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            scaling='min_max',
            info_cached_df=None,
            max_workers=1,
        )
    )
    assert len(out) == 100

    # info_cached retry loop: first continue, then count reaches 100 for line 429.
    sampled_bad = pd.DataFrame({'location': ['bad.fits'] * 100, 'token': [f'b{i}' for i in range(100)]})
    retry = pd.DataFrame(
        {
            'location': ['bad2.fits'] + [f'ok2_{i}.fits' for i in range(100)],
            'token': ['x'] + [f'g{i}' for i in range(100)],
        }
    )

    def open_fits_retry(filepath, **_kwargs):
        return None if 'bad' in str(filepath) else np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(nt, 'open_fits', open_fits_retry)

    def choice_first(seq):
        return sorted(seq)[0]

    monkeypatch.setattr(nt.random, 'choice', choice_first)

    out_retry = list(
        nt.prepare_data(
            sampled_data=sampled_bad,
            fit_data=None,
            training=False,
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            scaling='min_max',
            info_cached_df=retry,
            max_workers=1,
        )
    )
    assert len(out_retry) == 100

    # Fallback loop with info_cached_df=None: initial pass yields 1, then fills to 100.
    sampled_mix = pd.DataFrame({'location': ['ok3.fits'] + ['bad3.fits'] * 99, 'token': [f'm{i}' for i in range(100)]})

    def open_fits_mix(filepath, **_kwargs):
        return np.ones((2, 2), dtype=np.float32) if filepath == 'ok3.fits' else None

    monkeypatch.setattr(nt, 'open_fits', open_fits_mix)
    monkeypatch.setattr(nt.random, 'choice', lambda seq: seq[0])

    out_fallback = list(
        nt.prepare_data(
            sampled_data=sampled_mix,
            fit_data=None,
            training=False,
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            scaling='min_max',
            info_cached_df=None,
            max_workers=1,
        )
    )
    assert len(out_fallback) == 100


@pytest.mark.unit
def test_prepare_data_grouped_rows_empty_and_info_cached_missing_location(monkeypatch: pytest.MonkeyPatch):
    sampled_nan = pd.DataFrame({'location': [np.nan], 'token': ['x']})
    out_empty = list(
        nt.prepare_data(
            sampled_data=sampled_nan,
            fit_data=None,
            training=False,
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            scaling='min_max',
            info_cached_df=None,
            max_workers=1,
        )
    )
    assert out_empty == []

    sampled = pd.DataFrame({'location': ['ok.fits', 'bad.fits'], 'token': ['a', 'b']})

    def _open_mixed(filepath, **_kwargs):
        return None if filepath == 'bad.fits' else np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(nt, 'open_fits', _open_mixed)
    monkeypatch.setattr(nt, 'apply_scaling_and_stats', lambda sim, clean, *_args, **_kwargs: (sim, clean, ['s']))
    monkeypatch.setattr(nt.random, 'choice', lambda seq: seq[0])
    out_missing_loc = list(
        nt.prepare_data(
            sampled_data=sampled,
            fit_data=None,
            training=False,
            sigma_kernel_fn=lambda *_args, **_kwargs: 1.0,
            noise_fn=lambda clean, *_args, **_kwargs: clean,
            stats_name_fn=lambda *_args, **_kwargs: 's',
            type_of_image='SCI',
            preprocess_nan_value=0.0,
            preprocess_posinf_value=0.0,
            preprocess_neginf_value=0.0,
            sigma_key='combined_sigma',
            scaling='min_max',
            info_cached_df=pd.DataFrame({'x': [1]}),
            max_workers=1,
        )
    )
    assert len(out_missing_loc) == 2


@pytest.mark.unit
def test_train_network_and_main_remaining_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    train_dir = tmp_path / 'train'
    eval_dir = tmp_path / 'eval'
    results_dir = tmp_path / 'results'
    train_dir.mkdir()
    eval_dir.mkdir()
    results_dir.mkdir()
    (train_dir / 'a.fits').write_text('x', encoding='utf-8')
    (eval_dir / 'b.fits').write_text('x', encoding='utf-8')

    # Missing eval path branch.
    with pytest.raises(FileNotFoundError):
        nt.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data={
                'training_path': str(train_dir),
                'eval_path': str(tmp_path / 'missing_eval'),
                'results_path': str(results_dir),
                'checkpoint_custom_epoch': None,
                'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
            },
            kwargs_network={},
            data_generator=lambda *args, **kwargs: None,
            batch_size=1,
            optimizer=tf.keras.optimizers.Adam,
            G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            use_gan=False,
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )

    tiny = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(1, 1),
    ])

    monkeypatch.setattr(nt, 'network', lambda *args, **kwargs: tiny)
    monkeypatch.setattr(nt, 'load_model', lambda *args, **kwargs: (tiny, 0))
    monkeypatch.setattr(nt, 'get_discriminator', lambda *args, **kwargs: tiny)

    class DummyGAN:
        def __init__(self, *args, **kwargs):
            self.g_optimizer = tf.keras.optimizers.Adam(1e-3)

        def compile(self):
            return None

        def fit(self, *args, **kwargs):
            class H:
                history = {'loss': [1.0]}

            return H()

    monkeypatch.setattr(nt, 'GAN', DummyGAN)

    one = tf.data.Dataset.from_tensor_slices(
        (
            tf.ones((1, 8, 8, 1), dtype=tf.float32),
            tf.ones((1, 8, 8, 1), dtype=tf.float32),
            tf.constant([[b'a.fits']], dtype=tf.string),
        )
    ).batch(1)
    monkeypatch.setattr(nt, 'create_tf_dataset', lambda *args, **kwargs: one)

    monkeypatch.setattr(nt, 'Callback', lambda **kwargs: [])
    monkeypatch.setattr(nt, 'ensure_parent_dir_exists', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('mkdir fail')))

    with pytest.raises(KeyError, match='loss_fn'):
        nt.train_network(
            input_shape=(8, 8, 1),
            n_epochs=1,
            kwargs_data={
                'training_path': str(train_dir),
                'eval_path': str(eval_dir),
                'results_path': str(results_dir),
                'checkpoint_custom_epoch': None,
                'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
                'training': True,
            },
            kwargs_network={},
            data_generator=lambda *args, **kwargs: iter(()),
            batch_size=1,
            optimizer=tf.keras.optimizers.Adam,
            G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            learning_rate=1e-3,
            beta_1=0.9,
            discriminator_kwargs=None,
            gan_kwargs=None,
            use_gan=True,
            config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
        )

    monkeypatch.setattr(nt, 'ensure_parent_dir_exists', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('mkdir fail')))
    monkeypatch.setattr(nt, 'load_model', lambda *args, **kwargs: None)
    monkeypatch.setattr(nt, 'create_tf_dataset', lambda *args, **kwargs: one)
    monkeypatch.setattr(nt, 'Callback', lambda **kwargs: [])
    nt.train_network(
        input_shape=(8, 8, 1),
        n_epochs=1,
        kwargs_data={
            'training_path': str(train_dir),
            'eval_path': str(eval_dir),
            'results_path': str(results_dir),
            'checkpoint_custom_epoch': None,
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
            'training': True,
        },
        kwargs_network={},
        data_generator=lambda *args, **kwargs: iter(()),
        batch_size=1,
        optimizer=tf.keras.optimizers.Adam,
        G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        learning_rate=1e-3,
        beta_1=0.9,
        use_gan=False,
        config={'training': {'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras'}, 'data': {'type_of_image': 'SCI'}},
    )

    monkeypatch.setattr(nt, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(
        nt,
        'load_config',
        lambda **_kwargs: {
            'training': {
                'patch_size': [8, 8, 1],
                'n_epochs': 1,
                'data_kwargs': {'training_path': str(train_dir), 'eval_path': str(eval_dir), 'results_path': str(results_dir), 'checkpoint_custom_epoch': None, 'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'}},
                'network_kwargs': {},
                'discriminator_kwargs': {},
                'gan_kwargs': {'loss_fn': tf.keras.losses.BinaryCrossentropy(), 'adversarial_loss_weight': 1.0, 'reconstruction_loss_weight': 1.0, 'label_smoothing': 0.0},
                'data_generator': lambda *args, **kwargs: iter(()),
                'batch_size': 1,
                'optimizer': tf.keras.optimizers.Adam,
                'change_learning_rate': [],
                'g_loss_fn': tf.keras.losses.MeanAbsoluteError(),
                'learning_rate': 1e-3,
                'beta_1': 0.9,
                'start_from_best': False,
                'start_from_last': False,
                'save_freq': 1,
                'eval_save_percentage': 100,
                'ds_save_percentage': 100,
                'scaling': 'min_max',
                'use_gan': False,
                'training_results_dir': str(tmp_path / 'res2'),
                'training_metrics_csv_path': str(tmp_path / 'res2' / 'm.csv'),
                'training_history_json_path': str(tmp_path / 'res2' / 'h.json'),
                'validation_loss_filename': 'v.txt',
                'training_metrics_filename': 't.txt',
            },
            'data': {'type_of_image': 'SCI'},
        },
    )
    monkeypatch.setattr(nt, 'train_network', lambda *args, **kwargs: None)
    monkeypatch.setattr(nt.os, 'chdir', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nt.tf.config.experimental, 'list_physical_devices', lambda *_args, **_kwargs: [])

    with pytest.raises(FileNotFoundError):
        runpy.run_module('src.training.new_train', run_name='__main__')
