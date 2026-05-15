from __future__ import annotations

from pathlib import Path
import copy
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter import load_config
from src.models.network import GAN, get_discriminator, network
from src.training.new_train import _instantiate_optimizer
from src.training.utils import save_checkpoint_model


@pytest.fixture(autouse=True)
def deterministic_seeds():
    np.random.seed(1234)
    tf.random.set_seed(1234)


@pytest.fixture(autouse=True)
def reset_data_augment_globals():
    from src.training import new_train

    old_values = (
        new_train.MEM_CACHED,
        new_train.MEM_CACHED_EVAL,
        new_train.MEM_CACHED_TEST,
        new_train.INFO_CACHED,
        new_train.INFO_CACHED_EVAL,
        new_train.INFO_CACHED_TEST,
    )
    new_train.MEM_CACHED = None
    new_train.MEM_CACHED_EVAL = None
    new_train.MEM_CACHED_TEST = None
    new_train.INFO_CACHED = None
    new_train.INFO_CACHED_EVAL = None
    new_train.INFO_CACHED_TEST = None
    yield
    (
        new_train.MEM_CACHED,
        new_train.MEM_CACHED_EVAL,
        new_train.MEM_CACHED_TEST,
        new_train.INFO_CACHED,
        new_train.INFO_CACHED_EVAL,
        new_train.INFO_CACHED_TEST,
    ) = old_values


@pytest.fixture
def tiny_fits_array() -> np.ndarray:
    y, x = np.mgrid[0:64, 0:64]
    image = np.full((64, 64), 12.0, dtype=np.float32)
    for xc, yc, amp, sigma in ((18, 20, 24.0, 1.5), (42, 16, 18.0, 2.0), (34, 45, 30.0, 1.8)):
        image += (amp * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma**2))).astype(np.float32)
    image += (0.2 * np.sin(x / 7.0) + 0.15 * np.cos(y / 5.0)).astype(np.float32)
    return image.astype(np.float32)


@pytest.fixture
def tiny_fits_file(tmp_path: Path, tiny_fits_array: np.ndarray) -> Path:
    path = tmp_path / 'tiny_image.fits'
    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(data=tiny_fits_array.astype(np.float32), name='SCI')
    sci.header['EXPTIME'] = 120.0
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)
    return path


@pytest.fixture
def tiny_metadata_df(tmp_path_factory: pytest.TempPathFactory, tiny_fits_array: np.ndarray) -> pd.DataFrame:
    root = Path('/tmp/astrogan_unet_synthetic') / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    surveys = ['IR', 'GRISM1024', 'IR-UVIS']
    last_names = ['FABER', 'DOE', 'SMITH']
    split_cycle = ['training', 'eval', 'test']
    for idx in range(12):
        split = split_cycle[idx % len(split_cycle)]
        exp_time = float(120 + idx * 15)
        org_sigma = 0.8 + idx * 0.05
        exp_ratio = 1.5 + idx * 0.2
        combined_sigma = org_sigma * 1.6
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        file_path = split_dir / f'synthetic_{split}_{idx}.fits'
        array = (tiny_fits_array + idx * 0.05).astype(np.float32)
        primary = fits.PrimaryHDU()
        sci = fits.ImageHDU(data=array, name='SCI')
        sci.header['EXPTIME'] = exp_time
        fits.HDUList([primary, sci]).writeto(file_path, overwrite=True)

        row = {
            'filename': file_path.name,
            'location': str(file_path),
            'sci_actual_duration': exp_time,
            'exp_time': exp_time,
            'bkg_sigma': org_sigma,
            'bkg_median': 12.0 + idx * 0.01,
            'bkg_rms': org_sigma,
            'combined_sigma': combined_sigma,
            'org_sigma': org_sigma,
            'exp_ratio': exp_ratio,
            'new_exp_time': exp_time / exp_ratio,
            'sci_data_set_name': f'dataset_{idx % 4}',
            'sci_aper_1234': surveys[idx % len(surveys)],
            'sci_pi_last_name': last_names[idx % len(last_names)],
            'mean_bkg': 12.0,
            'median_bkg': 12.0,
            'std_bkg': org_sigma,
            'max_bkg': 13.0 + idx * 0.1,
            'mean_src': 22.0 + idx,
            'median_src': 19.0 + idx,
            'std_src': 3.0 + idx * 0.05,
            'max_src': 40.0 + idx,
            'abs_mean': 15.0 + idx * 0.1,
            'abs_median': 14.5 + idx * 0.1,
            'org_mean_bkg': 11.5,
            'org_median_bkg': 11.5,
            'org_std_bkg': org_sigma,
            'org_max_bkg': 12.5,
            'org_mean_src': 24.0 + idx,
            'org_median_src': 20.0 + idx,
            'org_std_src': 2.5 + idx * 0.05,
            'org_max_src': 42.0 + idx,
            'org_abs_mean': 16.0 + idx * 0.1,
            'org_abs_median': 15.5 + idx * 0.1,
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def mock_cfg(tmp_path_factory: pytest.TempPathFactory, tiny_metadata_df: pd.DataFrame) -> dict:
    cfg = load_config(
        model_type='unet',
        attention=False,
        scaling='min_max',
        loss_name='MeanAbsoluteError',
        output_activation='sigmoid',
        activation_name='ReLU',
        discriminator_activation='LeakyReLU',
        discriminator_output_activation='sigmoid',
    )
    cfg = copy.deepcopy(cfg)
    root = tmp_path_factory.mktemp('mock_cfg')
    metadata_csv = root / 'synthetic_metadata.csv'
    tiny_metadata_df.to_csv(metadata_csv, index=False)

    cfg['paths'].update(
        {
            'data_dir': str(root / 'data'),
            'training_path': str(root / 'data' / 'training_images'),
            'eval_path': str(root / 'data' / 'eval_images'),
            'test_path': str(root / 'data' / 'test_images'),
            'metadata_csv': str(metadata_csv),
            'fit_info_csv': str(root / 'data' / 'fit_info.csv'),
            'models_dir': str(root / 'models'),
            'plots_dir': str(root / 'plots'),
            'multimodal_metrics_dir': str(root / 'metrics' / 'multi_modal'),
            'singlemodal_metrics_dir': str(root / 'metrics' / 'single_modal'),
        }
    )

    cfg['data'].update(
        {
            'ps': 64,
            'steps': 4,
            'samples': 6,
            'val_samples': 3,
            'test_samples': 3,
            'times': 1.0,
            'low': 1.0,
            'high': 1e6,
            'training': True,
            'cache_raw_metadata': True,
            'sub_sample_train': None,
            'sub_sample_eval': None,
            'metadata_filepath': str(metadata_csv),
            'training_path': cfg['paths']['training_path'],
            'eval_path': cfg['paths']['eval_path'],
            'fit_data_filepath': cfg['paths']['fit_info_csv'],
            'training_cache_filepath': str(root / 'models' / 'sampled_training.csv'),
            'eval_cache_filepath': str(root / 'models' / 'sampled_eval.csv'),
            'test_cache_filepath': str(root / 'models' / 'sampled_test.csv'),
            'info_filepath': str(root / 'models' / 'info.csv'),
            'results_path': str(root / 'models' / 'checkpoints'),
            'max_workers': 2,
            'min_exp_ratio': 1.2,
            'max_exp_ratio': 10.0,
            'min_exp_time': 20.0,
        }
    )
    cfg['network'].update(
        {
            'depth': 2,
            'kernel_size': 3,
            'filter_size': 2,
            'pooling_size': 2,
            'n_of_initial_channels': 4,
            'func': tf.keras.layers.ReLU,
            'func_kwargs': {},
            'batch_normalization': False,
            'use_bias': True,
            'dropout_rate': 0.0,
            'block_depth': 2,
            'dropout_from_layer': 99,
            'attention': False,
            'output_activation': 'sigmoid',
            'kernel_initializer': 'he_normal',
        }
    )
    cfg['discriminator'].update(
        {
            'depth': 2,
            'n_initial_filters': 4,
            'filter_size': 2,
            'kernel_size': (3, 3),
            'dropout_rate': 0.0,
            'func': tf.keras.layers.LeakyReLU,
            'func_kwargs': {},
            'output_activation': 'sigmoid',
            'block_depth': 1,
            'batch_normalization': False,
            'use_bias': True,
            'dropout_from_layer': 99,
        }
    )
    cfg['gan'].update(
        {
            'd_learning_rate': 1e-3,
            'loss_fn': tf.keras.losses.BinaryCrossentropy(),
            'adversarial_loss_weight': 1.0,
            'reconstruction_loss_weight': 10.0,
            'label_smoothing': 0.0,
        }
    )
    cfg['training'].update(
        {
            'batch_size': 2,
            'use_gan': False,
            'optimizer': tf.keras.optimizers.Adam,
            'learning_rate': 1e-3,
            'beta_1': 0.5,
            'g_loss_fn': tf.keras.losses.MeanAbsoluteError(),
            'scaling': 'min_max',
            'data_generator': None,
            'checkpoint_filename_pattern': '{prefix}_{epoch}.keras',
            'checkpoint_custom_epoch': None,
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch}.keras'},
            'training_results_dir': str(root / 'models'),
            'training_metrics_csv_path': str(root / 'models' / 'training_history.csv'),
            'training_history_json_path': str(root / 'models' / 'training_history.json'),
            'validation_loss_filename': 'validation_loss.txt',
            'training_metrics_filename': 'training_metrics.txt',
            'patch_size': [64, 64, 1],
        }
    )
    cfg['create_dataset'].update(
        {
            'metadata_filepath': str(metadata_csv),
            'dataset_dir': cfg['paths']['data_dir'],
            'split_dirs': [cfg['paths']['training_path'], cfg['paths']['eval_path'], cfg['paths']['test_path']],
            'exp_column': 'sci_actual_duration',
            'id_column': 'sci_data_set_name',
            'url_column': 'drz_URL',
            'allowed_survey': ['IR', 'GRISM1024'],
            'low': 1.0,
            'high': 1e6,
            'size': 12,
            'ps': 64,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'location_col': 'location',
            'cropped_stats_output_file': str(metadata_csv),
            'noisy_filtered_metadata_output_file': str(root / 'data' / 'noisy_filtered.csv'),
        }
    )
    cfg['evaluation'].update(
        {
            'models_dir': cfg['paths']['models_dir'],
            'patch_size': [64, 64, 1],
            'stride': [32, 32, 1],
            'batch_size': 2,
            'scaling': 'min_max',
            'gaussian_sigma': 8,
            'weighting': 'average',
            'type_of_image': 'SCI',
            'uncropped_patch_size': [64, 64, 1],
            'uncropped_stride': [32, 32, 1],
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 2,
            'combined_images_dir': str(root / 'metrics' / 'combined'),
            'png_dir': str(root / 'metrics' / 'png'),
            'org_dir': str(root / 'metrics' / 'org'),
            'noisy_dir': str(root / 'metrics' / 'noisy'),
            'rec_dir': str(root / 'metrics' / 'rec'),
            'use_mosaic': False,
        }
    )
    cfg['evaluation']['kwargs_source'] = {
        'sigma': 3,
        'maxiters': 5,
        'nsigma': 1.5,
        'npixels': 3,
        'nlevels': 8,
        'contrast': 0.001,
        'footprint_radius': 2,
        'deblend': False,
        'deblend_timeout': 5,
        'bkg_box_size': 16,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k1': 0.01,
        'k2': 0.03,
        'win_size': 7,
        'win_sigma': 1.0,
        'uncropped_patch_size': (64, 64, 1),
        'uncropped_stride': (32, 32, 1),
        'uncropped_weighting': 'average',
        'uncropped_batch_size': 2,
        'sigma_key': 'bkg_sigma',
        'noise_fn': cfg['data']['noise_fn'],
        'type_of_image': 'SCI',
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'func': None,
        'thresh': 1.5,
        'org_thresh': 1.5,
        'radius_factor': 6.0,
        'PHOT_FLUXFRAC': 0.5,
        'r_min': 3.0,
        'elongation_fraction': 1.2,
        'PHOT_AUTOPARAMS': 2.5,
        'maskthresh': 0.0,
        'minarea': 3,
        'org_minarea': 3,
        'filter_type': 'matched',
        'deblend_nthresh': 16,
        'deblend_cont': 0.005,
        'clean': True,
        'clean_param': 1.0,
    }
    cfg['visualization']['prepare_images']['scaling'] = cfg['training']['scaling']
    cfg['visualization']['prepare_images']['metadata_filepath'] = str(metadata_csv)
    cfg['training']['data_kwargs'] = copy.deepcopy(cfg['data'])
    cfg['training']['network_kwargs'] = copy.deepcopy(cfg['network'])
    cfg['training']['discriminator_kwargs'] = copy.deepcopy(cfg['discriminator'])
    cfg['training']['gan_kwargs'] = copy.deepcopy(cfg['gan'])
    cfg['model_type'] = 'unet'
    cfg['input_shape'] = (64, 64, 1)
    cfg['checkpoint_info'] = {
        'model_type': 'UNET',
        'scaling': cfg['training']['scaling'],
        'config': {'training': {'scaling': cfg['training']['scaling']}},
    }
    return cfg


@pytest.fixture
def tiny_unet(mock_cfg: dict) -> tf.keras.Model:
    model = network(mock_cfg['input_shape'], **mock_cfg['training']['network_kwargs'])
    model.compile(
        optimizer=_instantiate_optimizer(mock_cfg['training']['optimizer'], {'learning_rate': mock_cfg['training']['learning_rate']}),
        loss=mock_cfg['training']['g_loss_fn'],
    )
    return model


@pytest.fixture
def tiny_discriminator(mock_cfg: dict) -> tf.keras.Model:
    return get_discriminator(mock_cfg['input_shape'], **mock_cfg['training']['discriminator_kwargs'])


@pytest.fixture
def tiny_gan(tiny_unet: tf.keras.Model, tiny_discriminator: tf.keras.Model, mock_cfg: dict) -> GAN:
    model = GAN(
        generator=tiny_unet,
        discriminator=tiny_discriminator,
        g_optimizer=tf.keras.optimizers.Adam(1e-3),
        d_optimizer=tf.keras.optimizers.Adam(1e-3),
        adversarial_loss_fn=tf.keras.losses.BinaryCrossentropy(),
        reconstruction_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        adversarial_loss_weight=mock_cfg['gan']['adversarial_loss_weight'],
        reconstruction_loss_weight=mock_cfg['gan']['reconstruction_loss_weight'],
        label_smoothing=0.0,
        name='tiny_gan',
    )
    model.compile()
    return model


@pytest.fixture
def tiny_catalog_df() -> pd.DataFrame:
    values = np.linspace(5.0, 45.0, 20)
    return pd.DataFrame(
        {
            'x': values,
            'y': values[::-1],
            'flux': np.linspace(100.0, 200.0, 20),
            'flux_err': np.linspace(1.0, 4.0, 20),
        }
    )


@pytest.fixture
def tiny_checkpoint_path(tmp_path: Path, tiny_unet: tf.keras.Model, mock_cfg: dict) -> Path:
    checkpoint_path = tmp_path / 'tiny_unet.keras'
    save_checkpoint_model(tiny_unet, str(checkpoint_path), mock_cfg['checkpoint_info'])
    return checkpoint_path
