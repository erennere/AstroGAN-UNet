from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.training.callback import Callback
from src.training.new_train import data_augment_pluggable
from src.training.utils import (
    build_checkpoint_filename,
    create_tf_dataset,
    open_fits,
    read_checkpoint_info,
    resolve_registry_function,
    save_fits,
)
from src.training.utils import candidates_based_on_range, candidates_based_on_ratio, post_filter, sample_range
from src.training.utils import log_cosh_loss, scale_invariant_mae, ssim_loss
from src.models.network import GAN, get_discriminator, network


@pytest.mark.unit
def test_resolve_registry_function_returns_expected_callables():
    assert resolve_registry_function('candidates_fn', 'exposure_ratio') is candidates_based_on_ratio
    assert resolve_registry_function('candidates_fn', 'sigma_range') is candidates_based_on_range


@pytest.mark.unit
def test_resolve_registry_function_rejects_unknown_name():
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_registry_function('candidates_fn', 'unknown_name')


@pytest.mark.unit
def test_candidate_functions_expand_rows_and_preserve_location(tiny_metadata_df, mock_cfg):
    row = tiny_metadata_df.iloc[0]
    ratio_rows = candidates_based_on_ratio(row, mock_cfg['data'])

    assert len(ratio_rows) > 1
    assert {entry['location'] for entry in ratio_rows} == {row['location']}


@pytest.mark.unit
def test_sample_range_returns_at_most_requested_rows():
    info = pd.DataFrame(
        {
            'dataset': ['a', 'a', 'b', 'b', 'c', 'c'],
            'abs_mean': [5, 4, 6, 3, 7, 2],
            'base': [1, 2, 1, 2, 1, 2],
            'exponent_diff': [0, 1, 0, 1, 0, 1],
            'full_abs_mean': [10] * 6,
            'crop_abs_mean': [9, 8, 9, 8, 9, 8],
        }
    )
    sampled = sample_range(info, 4, {'delta': 0.3})

    assert len(sampled) <= 4


@pytest.mark.unit
def test_create_tf_dataset_yields_expected_batches():
    def generator(images, kwargs_data, scaling=None):
        metadata = np.array(['sample'], dtype='S32')
        for index in range(4):
            noisy = np.full((64, 64, 1), index, dtype=np.float32)
            clean = noisy + 1.0
            yield noisy, clean, metadata

    dataset = create_tf_dataset(
        images=['a', 'b', 'c', 'd'],
        sample_generator=generator,
        generator_kwargs={'ps': 64},
        batch_size=2,
        scaling=None,
        augment=False,
    )

    batches_epoch_1 = list(dataset)
    batches_epoch_2 = list(create_tf_dataset(['a', 'b', 'c', 'd'], generator, {'ps': 64}, batch_size=2, scaling=None, augment=False))

    assert len(batches_epoch_1) == len(batches_epoch_2) == 2
    noisy, clean, metadata = batches_epoch_1[0]
    assert noisy.shape == (2, 64, 64, 1)
    assert clean.shape == (2, 64, 64, 1)
    assert noisy.dtype == tf.float32
    assert clean.dtype == tf.float32
    assert metadata.dtype == tf.string


@pytest.mark.unit
@pytest.mark.parametrize('shape', [(64, 64), (128, 128)])
def test_save_fits_and_open_fits_roundtrip(tmp_path: Path, shape):
    array = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    save_dir = tmp_path / 'nested' / 'fits'
    save_fits(array, 'roundtrip.fits', str(save_dir), type_of_image='SCI')
    loaded = open_fits(str(save_dir / 'roundtrip.fits'), type_of_image='SCI')

    assert loaded.shape == shape
    assert np.allclose(loaded, array)


@pytest.mark.unit
def test_open_fits_missing_file_returns_none(tmp_path: Path):
    assert open_fits(str(tmp_path / 'missing.fits'), type_of_image='SCI') is None


@pytest.mark.unit
def test_tiny_checkpoint_fixture_contains_expected_metadata(tiny_checkpoint_path: Path):
    info = read_checkpoint_info(str(tiny_checkpoint_path))

    assert info['model_type'] == 'UNET'
    assert info['scaling'] == 'min_max'


@pytest.mark.integration
@pytest.mark.parametrize('loss_fn', [tf.keras.losses.MeanAbsoluteError(), scale_invariant_mae, log_cosh_loss, ssim_loss])
def test_unet_single_train_step_changes_weights(mock_cfg, loss_fn):
    model = network(mock_cfg['input_shape'], **mock_cfg['training']['network_kwargs'])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    x = tf.random.uniform((2, 64, 64, 1))
    y = tf.random.uniform((2, 64, 64, 1))
    before = [weight.numpy().copy() for weight in model.trainable_weights]
    loss = model.train_on_batch(x, y)
    after = [weight.numpy() for weight in model.trainable_weights]

    assert np.isfinite(float(loss))
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))


@pytest.mark.integration
def test_gan_single_train_step_and_adversarial_weight_effect(mock_cfg):
    x = tf.random.uniform((2, 64, 64, 1), seed=21)
    y = tf.random.uniform((2, 64, 64, 1), seed=22)

    def build_gan(adv_weight):
        generator = network(mock_cfg['input_shape'], **mock_cfg['training']['network_kwargs'])
        discriminator = get_discriminator(mock_cfg['input_shape'], **mock_cfg['training']['discriminator_kwargs'])
        model = GAN(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=tf.keras.optimizers.SGD(0.0),
            d_optimizer=tf.keras.optimizers.SGD(0.0),
            adversarial_loss_fn=tf.keras.losses.BinaryCrossentropy(),
            reconstruction_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            adversarial_loss_weight=adv_weight,
            reconstruction_loss_weight=2.0,
            label_smoothing=0.0,
        )
        model.compile()
        return model

    gan_full = build_gan(1.0)
    gan_zero = build_gan(0.0)
    metrics_full = gan_full.train_step((x, y))
    metrics_zero = gan_zero.train_step((x, y))

    assert np.isfinite(float(metrics_full['g_loss']))
    assert np.isfinite(float(metrics_full['d_loss']))
    assert float(metrics_zero['g_loss']) == pytest.approx(2.0 * float(metrics_zero['g_rec_loss']), rel=1e-5)
    assert float(metrics_full['g_loss']) > float(metrics_zero['g_loss'])


@pytest.mark.integration
def test_callback_writes_expected_files(tmp_path: Path, tiny_unet, mock_cfg):
    noisy = tf.random.uniform((2, 64, 64, 1))
    clean = tf.random.uniform((2, 64, 64, 1))
    metadata = tf.constant([[b'sample_a'], [b'sample_b']])
    dataset = tf.data.Dataset.from_tensor_slices((noisy, clean, metadata)).batch(2)
    callback = Callback(
        dataset=dataset,
        dataset_size=2,
        epoch=1,
        start_epoch=0,
        eval_save_percentage=100,
        ds_save_percentage=0,
        save_freq=1,
        sample_generator=None,
        kwargs_validation={},
        validation_dataset=dataset,
        validation_dataset_size=2,
        optimizer=tiny_unet.optimizer,
        change_learning_rate=[(0, 1e-3)],
        batch_size=2,
        scaling='min_max',
        use_gan=False,
        training_results_dir=str(tmp_path / 'results'),
        checkpoints_dir=str(tmp_path / 'checkpoints'),
        training_metrics_csv_path=str(tmp_path / 'results' / 'history.csv'),
        validation_loss_filename='validation_loss.txt',
        training_metrics_filename='training_metrics.txt',
        config=mock_cfg,
    )
    callback.set_model(tiny_unet)
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {'loss': 0.25})

    validation_loss_file = tmp_path / 'results' / 'validation_loss.txt'
    history_csv = tmp_path / 'results' / 'history.csv'
    checkpoint_file = tmp_path / 'checkpoints' / build_checkpoint_filename('model', 1, mock_cfg['training']['checkpoint_filename_pattern'])

    assert validation_loss_file.exists()
    assert 'Epoch 1' in validation_loss_file.read_text(encoding='utf-8')
    assert checkpoint_file.exists()
    history = pd.read_csv(history_csv)
    assert list(history.columns) == ['epoch', 'train_loss', 'g_loss', 'validation_loss', 'epoch_time']


@pytest.mark.integration
@pytest.mark.slow
def test_data_augment_pluggable_creates_split_caches_and_info_file(mock_cfg):
    train_kwargs = dict(mock_cfg['data'])
    train_kwargs['training'] = True
    train_samples = list(islice(data_augment_pluggable([], train_kwargs, scaling=mock_cfg['training']['scaling']), 2))

    eval_kwargs = dict(mock_cfg['data'])
    eval_kwargs['training'] = False
    eval_samples = list(islice(data_augment_pluggable([], eval_kwargs, scaling=mock_cfg['training']['scaling']), 1))

    test_kwargs = dict(mock_cfg['data'])
    test_kwargs['training'] = False
    test_kwargs['test'] = True
    test_samples = list(data_augment_pluggable([], test_kwargs, scaling=mock_cfg['training']['scaling']))

    training_cache = pd.read_csv(mock_cfg['data']['training_cache_filepath'])
    eval_cache = pd.read_csv(mock_cfg['data']['eval_cache_filepath'])
    test_cache = pd.read_csv(mock_cfg['data']['test_cache_filepath'])
    info_df = pd.read_csv(mock_cfg['data']['info_filepath'])

    assert train_samples
    assert eval_samples
    assert test_samples
    assert Path(mock_cfg['data']['training_cache_filepath']).exists()
    assert Path(mock_cfg['data']['eval_cache_filepath']).exists()
    assert Path(mock_cfg['data']['test_cache_filepath']).exists()
    assert Path(mock_cfg['data']['info_filepath']).exists()
    assert not info_df.empty
    for frame in (training_cache, eval_cache, test_cache):
        assert (frame['combined_sigma'] > frame['org_sigma']).all()
