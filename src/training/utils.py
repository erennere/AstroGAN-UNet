"""Shared training/evaluation filesystem and checkpoint utility helpers."""

import json
import logging
import os
import re
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
from src.models.network import GAN
from src.training.math_helpers import (
    evenly_spaced_numbers,
    create_ratios,
    _sigma_kernel_from_fit_wrapper,
    _sigma_kernel_from_row_wrapper,
    _simulated_image_from_exposure,
    _simulated_image_from_poisson,
    _stats_name_from_sigma,
    _stats_name_from_exposure,
    _stats_name_from_exp_ratio
)


############ FILE I/O ########################################################

def _normalize_runtime_filepath(filepath):
    """Normalize cross-platform filepaths at runtime.

    In WSL/Linux runs, metadata can still contain Windows absolute paths like
    ``D:\\...``. Convert those to ``/mnt/d/...`` so FITS files are readable.
    """
    if not isinstance(filepath, str) or filepath == '':
        return filepath

    if os.name == 'nt':
        return filepath

    windows_abs = re.match(r'^([A-Za-z]):[\\/](.*)$', filepath)
    if windows_abs:
        drive = windows_abs.group(1).lower()
        rest = windows_abs.group(2).replace('\\', '/').lstrip('/')
        return f'/mnt/{drive}/{rest}'

    if '\\' in filepath:
        return filepath.replace('\\', '/')

    return filepath

def ensure_directory_exists(directory_path):
    """Create *directory_path* (and parents) when it does not exist.

    Parameters
    ----------
    directory_path : str or None
        Directory path to create.

    Returns
    -------
    str or None
        The input path when successful, otherwise ``None``.
    """
    if not directory_path:
        return None
    try:
        os.makedirs(directory_path, exist_ok=True)
        return directory_path
    except Exception as error:
        logging.warning('Failed to create directory %s: %s', directory_path, error)
        return None


def ensure_parent_dir_exists(filepath):
    """Ensure the parent directory of *filepath* exists.

    Parameters
    ----------
    filepath : str or None
        Target file path.

    Returns
    -------
    str or None
        Parent directory path (or empty string for cwd), ``None`` on failure.
    """
    if not filepath:
        return None
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        return ensure_directory_exists(parent)
    return ''

def is_not_nan(value):
    """
    Check whether a scalar is finite and not NaN.

    Parameters
    ----------
    value : Any
        Value to validate.

    Returns
    -------
    bool
        True when value can be converted to a finite float.
    """
    if pd.isna(value):
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def open_fits(filepath, ratio=None, type_of_image='SCI', low=None, high=None):
    """Open a FITS image HDU and optionally filter by scaled exposure.

    Parameters
    ----------
    filepath : str
        FITS file path.
    ratio : float or None, optional
        Exposure ratio multiplier used for bound checks, by default None.
    type_of_image : str, optional
        HDU name to select when ratio is provided, by default 'SCI'.
    low : float or None, optional
        Inclusive lower bound for exptime*ratio, by default None.
    high : float or None, optional
        Inclusive upper bound for exptime*ratio, by default None.

    Returns
    -------
    numpy.ndarray or tuple or None
        Image array when ratio is None, otherwise (image, exposure_time, ratio), or None.
    """
    normalized_filepath = _normalize_runtime_filepath(filepath)

    try:
        with fits.open(normalized_filepath, memmap=False) as hdus:
            selected_hdu = None
            target_name = str(type_of_image).strip().lower()

            for hdu in hdus:
                if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
                    continue
                if not isinstance(hdu.data, np.ndarray):
                    continue
                if str(hdu.name).strip().lower() == target_name:
                    selected_hdu = hdu
                    break

            if selected_hdu is None:
                logging.warning(f'No image HDU with name "{type_of_image}" found in {normalized_filepath}.')
                return None

            image = np.array(selected_hdu.data, copy=True)

            if ratio is None:
                return image

            try:
                ratio_value = float(ratio)
            except (TypeError, ValueError):
                logging.warning(f'Invalid ratio "{ratio}" for {normalized_filepath}.')
                return None

            if 'EXPTIME' not in selected_hdu.header:
                logging.warning(f'No EXPTIME found in HDU "{type_of_image}" of {normalized_filepath}.')
                return None
            exposure_time = selected_hdu.header['EXPTIME']

            if not isinstance(exposure_time, (int, float, str, np.number)):
                logging.warning(f'Invalid EXPTIME type "{type(exposure_time).__name__}" in {normalized_filepath}.')
                return None

            try:
                exposure_time = float(exposure_time)
            except (TypeError, ValueError):
                logging.warning(f'Invalid EXPTIME value "{exposure_time}" in {normalized_filepath}.')
                return None

            scaled_exposure = exposure_time * ratio_value
            if low is not None and scaled_exposure < float(low):
                return None
            if high is not None and scaled_exposure > float(high):
                return None

            return image, exposure_time, ratio_value
    except Exception as error:
        logging.warning(f'Error occurred while reading FITS file {normalized_filepath}: {error}')
        return None
    
def save_fits(image, name, path, type_of_image='SCI'):
    """
    Save an image array as a FITS file.

    Parameters
    ----------
    image : numpy.ndarray
        Image data to write.
    name : str
        Output filename.
    path : str
        Output directory.

    Returns
    -------
    bool or None
        True on success, otherwise None.
    """
    try:
        if ensure_directory_exists(path) is None:
            return None

        output_path = os.path.join(path, name)
        image_array = np.asarray(image)

        # Keep a standard PRIMARY HDU and store science image in a named extension.
        primary_hdu = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(data=image_array, name=str(type_of_image))
        fits.HDUList([primary_hdu, image_hdu]).writeto(output_path, overwrite=True)
        return True
    except Exception as error:
        logging.warning(f'An error occurred while saving {name}: {error}')
        return None

############ CHECKPOINT HELPERS ##############################################

CHECKPOINT_INFO_FILENAME = 'checkpoint_info.json'

def build_checkpoint_filename(checkpoint_prefix, epoch, filename_pattern):
    """Build a checkpoint filename from one strict pattern.

    Parameters
    ----------
    checkpoint_prefix : str
        Prefix such as ``best_model`` or ``model``.
    epoch : int
        Epoch identifier to encode in the filename.
    filename_pattern : str
        Pattern containing both ``{prefix}`` and ``{epoch...}`` placeholders.

    Returns
    -------
    str
        Formatted checkpoint filename.
    """
    if not isinstance(filename_pattern, str):
        raise TypeError('filename_pattern must be a string.')
    if '{prefix}' not in filename_pattern or re.search(r'\{epoch[^}]*\}', filename_pattern) is None:
        raise ValueError("filename_pattern must include '{prefix}' and '{epoch...}'.")
    return filename_pattern.format(prefix=checkpoint_prefix, epoch=epoch)

def save_checkpoint_model(model, checkpoint_path, checkpoint_info=None):
    """Save a model and store small checkpoint metadata inside the `.keras` file."""
    model.save(checkpoint_path)
    if checkpoint_info is None or not zipfile.is_zipfile(checkpoint_path):
        return
    with zipfile.ZipFile(checkpoint_path, 'a') as archive:
        archive.writestr(
            CHECKPOINT_INFO_FILENAME,
            json.dumps(checkpoint_info, indent=2, default=str),
        )

def read_checkpoint_info(checkpoint_path):
    """Read checkpoint metadata stored inside a `.keras` file."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return {}
    if not zipfile.is_zipfile(checkpoint_path):
        return {}
    with zipfile.ZipFile(checkpoint_path, 'r') as archive:
        if CHECKPOINT_INFO_FILENAME not in archive.namelist():
            return {}
        with archive.open(CHECKPOINT_INFO_FILENAME) as f:
            data = json.load(f)
    return data if isinstance(data, dict) else {}

def load_checkpoint_model(checkpoint_path, **loader_kwargs):
    """Load a model and attach the stored checkpoint metadata to it."""
    model = tf.keras.models.load_model(checkpoint_path, **loader_kwargs)
    checkpoint_info = read_checkpoint_info(checkpoint_path)
    model.checkpoint_info = checkpoint_info
    if hasattr(model, 'generator'):
        model.generator.checkpoint_info = checkpoint_info
    if hasattr(model, 'discriminator'):
        model.discriminator.checkpoint_info = checkpoint_info
    return model

def restore_model(
    checkpoint_dir,
    checkpoint_prefix,
    epoch,
    filename_pattern,
    loader_kwargs,
):
    """
    Restore a model checkpoint for a specific epoch.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints.
    checkpoint_prefix : str
        Prefix used in checkpoint file names.
    epoch : int
        Epoch number to restore.
    filename_pattern : str
        Filename pattern with prefix and epoch fields.
    loader_kwargs : dict
        Extra keyword arguments for ``tf.keras.models.load_model``.

    Returns
    -------
    tuple or None
        (model, epoch) when successful, otherwise None.
    """
    if not isinstance(loader_kwargs, dict):
        raise TypeError('loader_kwargs must be a dict.')

    filename = build_checkpoint_filename(checkpoint_prefix, epoch, filename_pattern)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(checkpoint_path):
        logging.warning(f'Checkpoint file not found: {checkpoint_path}')
        return None

    try:
        loaded_model = load_checkpoint_model(checkpoint_path, **loader_kwargs)
        logging.warning(f'Restored model checkpoint {filename} at epoch {epoch}.')
        return loaded_model, epoch
    except Exception as error:
        logging.warning(f'Failed to restore epoch {epoch}: {error}')
        return None

def load_model(checkpoint_dir, start_from_best, start_from_last, custom_epoch, restore_kwargs):
    """
    Restore a model from best, last, or custom checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints.
    start_from_best : bool
        Whether to prefer best_model checkpoints.
    start_from_last : bool
        Whether to prefer model checkpoints.
    custom_epoch : int or None
        Exact epoch to try first.
    restore_kwargs : dict
        Must include ``filename_pattern`` and may include extra loader kwargs.

    Returns
    -------
    tuple
        (model, epoch) on success, otherwise (None, 0).
    """
    def _extract_checkpoint_epochs(checkpoint_dir, checkpoint_prefix, filename_pattern):
        """
        Collect checkpoint epochs that match the configured filename pattern.

        Parameters
        ----------
        checkpoint_dir : str
            Directory containing checkpoint files.
        checkpoint_prefix : str
            Prefix to match.
        filename_pattern : str
            Pattern used to construct checkpoint names.

        Returns
        -------
        list of int
        Sorted unique epoch values.
        """
        if not isinstance(filename_pattern, str):
            raise TypeError('filename_pattern must be a string.')

        normalised = filename_pattern.replace('{prefix}', '__PREFIX__')
        normalised = re.sub(r'\{epoch[^}]*\}', '__EPOCH__', normalised)

        if '__PREFIX__' not in normalised or '__EPOCH__' not in normalised:
            raise ValueError("filename_pattern must include '{prefix}' and '{epoch...}'.")

        regex_pattern = '^' + re.escape(normalised).replace('__PREFIX__', re.escape(checkpoint_prefix)).replace('__EPOCH__', r'(\d+)') + '$'
        filename_pattern_re = re.compile(regex_pattern)
        discovered_epochs = []

        for filename in os.listdir(checkpoint_dir):
            match = filename_pattern_re.match(filename)
            if match:
                discovered_epochs.append(int(match.group(1)))
        return sorted(set(discovered_epochs))

    start_epoch = 0
    if not isinstance(restore_kwargs, dict):
        raise TypeError('restore_kwargs must be a dict.')
    if 'filename_pattern' not in restore_kwargs:
        raise ValueError("restore_kwargs must include 'filename_pattern'.")

    active_filename_pattern = restore_kwargs['filename_pattern']
    active_loader_kwargs = {k: v for k, v in restore_kwargs.items() if k != 'filename_pattern'}

    if start_from_best and start_from_last:
        raise ValueError("Cannot start from both 'best' and 'last' checkpoints simultaneously.")

    if not start_from_best and not start_from_last and custom_epoch is None:
        logging.warning("Starting from scratch because no checkpoint source was selected.")
        return None, start_epoch

    if not os.path.exists(checkpoint_dir):
        return None, start_epoch

    best_epochs = _extract_checkpoint_epochs(checkpoint_dir, 'best_model', active_filename_pattern)
    regular_epochs = _extract_checkpoint_epochs(checkpoint_dir, 'model', active_filename_pattern)

    if custom_epoch is not None:
        if custom_epoch in best_epochs:
            restored = restore_model(checkpoint_dir, 'best_model', custom_epoch, active_filename_pattern, active_loader_kwargs)
            if restored is not None:
                return restored
        if custom_epoch in regular_epochs:
            restored = restore_model(checkpoint_dir, 'model', custom_epoch, active_filename_pattern, active_loader_kwargs)
            if restored is not None:
                return restored
        logging.warning(f'No checkpoint found for custom epoch: {custom_epoch}.')

    if best_epochs and start_from_best:
        for epoch_value in reversed(best_epochs):
            restored = restore_model(checkpoint_dir, 'best_model', epoch_value, active_filename_pattern, active_loader_kwargs)
            if restored is not None:
                return restored
        logging.warning('No valid best checkpoint found, trying last checkpoints next.')

    for epoch_value in reversed(regular_epochs):
        restored = restore_model(checkpoint_dir, 'model', epoch_value, active_filename_pattern, active_loader_kwargs)
        if restored is not None:
            return restored

    logging.warning('No valid checkpoint found, starting from scratch.')
    return None, start_epoch


def build_checkpoint_custom_objects():
    """Build the custom_objects mapping used for .keras checkpoint loading."""
    return {
        'GAN': GAN,
        'scale_invariant_mae': scale_invariant_mae,
        'log_cosh_loss': log_cosh_loss,
        'ssim_loss': ssim_loss,
    }


############ LOSS FUNCTIONS ###################################################

@tf.keras.utils.register_keras_serializable(package='astroUnets')
def scale_invariant_mae(y_true, y_pred):
    """
    Compute MAE normalized by target dynamic range.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth tensor.
    y_pred : tf.Tensor
        Predicted tensor.

    Returns
    -------
    tf.Tensor
        Scalar scale-invariant MAE.
    """
    target_range = tf.maximum(
        tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True),
        tf.constant(1e-6, dtype=y_true.dtype),
    )
    mae_per_sample = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))
    return tf.reduce_mean(mae_per_sample / target_range)

@tf.keras.utils.register_keras_serializable(package='astroUnets')
def log_cosh_loss(y_true, y_pred):
    """
    Compute log-cosh loss normalized by target dynamic range.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth tensor.
    y_pred : tf.Tensor
        Predicted tensor.

    Returns
    -------
    tf.Tensor
        Scalar normalized log-cosh loss.
    """
    target_range = (
        tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True)
        + 1e-10
    )
    residual = y_pred - y_true
    log_two = tf.math.log(tf.constant(2.0, dtype=residual.dtype))
    log_cosh_per_sample = tf.reduce_mean(
        residual + tf.nn.softplus(-2.0 * residual) - log_two,
        axis=(1, 2, 3),
    )
    return tf.reduce_mean(log_cosh_per_sample / target_range)

@tf.keras.utils.register_keras_serializable(package='astroUnets')
def ssim_loss(y_true, y_pred, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03):
    """Compute the SSIM-based loss ``(1 - mean_SSIM)`` using a Gaussian sliding window.

    Fully differentiable via TensorFlow ops; compatible with
    ``model.compile()``. The dynamic range ``L`` is estimated per sample from
    *y_true* so that the stability constants ``c1``/``c2`` scale with the
    actual signal, matching the Wang et al. 2004 definition.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth image batch of shape ``(B, H, W, 1)``.
    y_pred : tf.Tensor
        Predicted image batch of shape ``(B, H, W, 1)``.
    win_size : int, optional
        Gaussian kernel width in pixels (must be odd). Standard value is 11.
        Rule of thumb: ~1–2 % of the shorter image dimension, rounded to the
        nearest odd number. Default is 11.
    win_sigma : float, optional
        Standard deviation of the Gaussian kernel. Standard value is 1.5.
        Default is 1.5.
    k1 : float, optional
        Stability constant for the luminance term. Default is 0.01.
    k2 : float, optional
        Stability constant for the contrast term. Default is 0.03.

    Returns
    -------
    tf.Tensor
        Scalar loss in ``[0, 2]`` (0 when the images are identical).
    """
    # --- build 2-D Gaussian kernel -------------------------------------------
    coords = tf.cast(tf.range(win_size) - win_size // 2, tf.float32)
    g1d = tf.exp(-0.5 * (coords / win_sigma) ** 2)
    g1d = g1d / tf.reduce_sum(g1d)
    kernel = g1d[:, None] * g1d[None, :]          # (win_size, win_size)
    kernel = kernel[:, :, None, None]               # (kH, kW, in_ch, out_ch)

    def local_mean(t):
        # t: (B, H, W, 1)  ->  (B, H, W, 1)
        return tf.nn.conv2d(t, kernel, strides=1, padding='SAME')

    # --- local statistics ----------------------------------------------------
    mu_x  = local_mean(y_true)
    mu_y  = local_mean(y_pred)
    mu_xx = local_mean(y_true * y_true) - mu_x * mu_x
    mu_yy = local_mean(y_pred * y_pred) - mu_y * mu_y
    mu_xy = local_mean(y_true * y_pred) - mu_x * mu_y

    sigma_x_sq = tf.maximum(mu_xx, 0.0)
    sigma_y_sq = tf.maximum(mu_yy, 0.0)
    sigma_xy   = mu_xy

    # --- dynamic range per sample -------------------------------------------
    L  = (tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
         - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True))
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # --- SSIM map ------------------------------------------------------------
    numerator   = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map    = numerator / (denominator + 1e-10)

    return 1.0 - tf.reduce_mean(ssim_map)

############ DATASET HELPERS #################################################

def create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32, scaling=None, augment=True):
    """
    Create an augmented tf.data.Dataset from a sample generator.

    Parameters
    ----------
    images : list
        Image file path list.
    sample_generator : callable
        Generator yielding (x, y, metadata).
    generator_kwargs : dict
        Keyword arguments passed to sample_generator.
    batch_size : int, optional
        Batch size, by default 32.
    scaling : str or None, optional
        Scaling mode controlling metadata size.
    augment : bool, optional
        Whether to apply random flips and rotations before batching.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset with random flips/rotations.
    """
    metadata_shape = 1
    if scaling in ['log_min_max']:
        metadata_shape = 7
    elif scaling in ['z_scale', 'min_max']:
        metadata_shape = 5

    def generator():
        yield from sample_generator(images=images, kwargs_data=generator_kwargs, scaling=scaling)

    patch_size = generator_kwargs['ps']
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_size, patch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(patch_size, patch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(metadata_shape,), dtype=tf.string),
        ),
    )

    def apply_augmentations(noisy_image, clean_image, metadata):
        """Apply random 90-degree rotations and flips to noisy/clean image pairs."""
        rotation_steps = tf.random.uniform([], 0, 4, dtype=tf.int32)
        noisy_image = tf.image.rot90(noisy_image, rotation_steps)
        clean_image = tf.image.rot90(clean_image, rotation_steps)

        flip_left_right = tf.random.uniform([], 0, 1) > 0.5
        flip_up_down = tf.random.uniform([], 0, 1) > 0.5

        if flip_left_right:
            noisy_image = tf.image.flip_left_right(noisy_image)
            clean_image = tf.image.flip_left_right(clean_image)

        if flip_up_down:
            noisy_image = tf.image.flip_up_down(noisy_image)
            clean_image = tf.image.flip_up_down(clean_image)
        return noisy_image, clean_image, metadata

    if augment:
        dataset = dataset.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

############ PLUGGABLE AUGMENT STRATEGIES ###################################

def is_relevant_crop(stats, delta=0.5):
    """Return ``True`` when a crop's absolute mean exceeds a threshold.

    Parameters
    ----------
    stats : dict or pandas.Series
        Must contain ``'full_abs_mean'`` (whole-image absolute mean) and
        ``'crop_abs_mean'`` (crop absolute mean).
    delta : float, optional
        Fraction of *full_abs_mean* used as the acceptance threshold. Default
        is 0.5.

    Returns
    -------
    bool
        ``True`` when ``crop_abs_mean > delta * full_abs_mean``.
    """
    abs_uncropped = stats['full_abs_mean']
    abs_crop = stats['crop_abs_mean']
    abs_threshold = abs_uncropped * delta
    return abs_crop > abs_threshold

def filtering_df(data, n_samples, delta, id_, abs_, base=None, exponent=None, noise_ratio=None):
    """
    Select a diversified subset of candidate rows after relevance filtering.

    This helper first removes rows whose `cropped_abs_mean` is too small relative to
    `full_abs_mean` using `is_relevant_crop`. It then repeatedly sorts by
    grouping key (`id_`) plus either `(base, exponent)` or `noise_ratio`, and
    takes one representative per group based on the largest `abs_` value.
    The loop continues until `n_samples` rows are accumulated or input rows are
    exhausted, then shuffles the final selection for training.

    Parameters
    ----------
    data : pandas.DataFrame
        Candidate metadata table.
    n_samples : int
        Requested number of output rows.
    delta : float
        Relevance threshold applied to `cropped_abs_mean` vs `full_abs_mean`.
    id_ : str
        Column used as grouping identifier (for example `dataset`).
    abs_ : str
        Magnitude column used for de-dup and descending preference.
    base : str or None, optional
        Base-digit column used with `exponent` for sigma-range strategy.
    exponent : str or None, optional
        Exponent-difference column used with `base`.
    noise_ratio : str or None, optional
        Ratio column used for exposure-ratio strategy.

    Returns
    -------
    pandas.DataFrame
        Filtered and sampled rows with randomized order.
    """
    if n_samples >= len(data):
        return data

    data['valid'] = data.apply(lambda row: is_relevant_crop(row, delta=delta), axis=1)
    data = data[data['valid']]
    if n_samples >= len(data):
        return data

    old_n_samples = n_samples

    if base is not None and exponent is not None:
        sort_keys = [id_, abs_, base, exponent]
        sort_ascending = [True, False, False, True]
    elif noise_ratio is not None:
        sort_keys = [id_, abs_, noise_ratio]
        sort_ascending = [True, False, False]
    else:
        raise ValueError('Must provide either (base, exponent) or noise_ratio for sorting.')

    data.loc[:, 'temp_index'] = np.arange(0, len(data))
    dfs = []
    already_in = 0
    while True:
        n_samples -= already_in
        sorted_df = data.sort_values(by=sort_keys, ascending=sort_ascending)

        result = sorted_df.groupby(id_).apply(
            lambda group: group.drop_duplicates(subset=abs_, keep='first')
        )
        if len(result) >= n_samples:
            dfs.append(result.sort_values(by=[abs_], ascending=False, ignore_index=True).iloc[:n_samples])
            break
        dfs.append(result)
        data = data[~data['temp_index'].isin(result['temp_index'])]
        already_in += len(result)
        if data.empty:  # pragma: no cover - defensive guard; loop normally exits via n_samples condition first
            break
    return pd.concat(dfs, ignore_index=True).iloc[:old_n_samples].sample(frac=1).reset_index(drop=True)

def filtering_df_v2(data, n_samples, col_A, col_B, col_C, col_D, occurrences_per_col_D, quantiles, percentages):
    """
    Select rows using explicit ordering and staged quantile-based sampling.

    Ordering rules:
    - sort by `col_A` and `col_B` in descending order
    - sort by `col_C` and `col_D` in ascending order

    Selection rules:
     1. Scan from top to bottom and keep selecting rows while each unique
         `col_D` value has been chosen fewer than `occurrences_per_col_D` times.
         In other words, the initial pass keeps the first X occurrences of each
         unique `col_D` value in sorted order.
    2. Split the remaining ordered rows into percentile-position bins defined
       by `quantiles`, allocate picks using `percentages`, and within each bin
         prefer as many different `col_D` values as possible while still
         scanning top to bottom.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table to order and sample from.
    n_samples : int
        Total number of rows to select.
    col_A : str
        First ordering column, descending.
    col_B : str
        Second ordering column, descending.
    col_C : str
        Third ordering column, ascending.
    col_D : str
        Fourth ordering column, ascending, and grouping key for diversity.
    occurrences_per_col_D : int
        Number of top-to-bottom occurrences to keep for each unique `col_D`
        value during the initial pass.
    quantiles : tuple[int | float]
        Increasing upper percentile bounds, for example `(20, 40, 60, 80, 100)`.
    percentages : tuple[int | float]
        Allocation weights for each quantile bin.

    Returns
    -------
    pandas.DataFrame
        Selected rows in the order they were chosen.
    """
    # Validate the requested sample plan before touching the data.
    if n_samples < 0:
        raise ValueError('n_samples must be non-negative.')
    if occurrences_per_col_D < 0:
        raise ValueError('occurrences_per_col_D must be non-negative.')
    if len(quantiles) == 0:
        raise ValueError('quantiles must not be empty.')
    if len(quantiles) != len(percentages):
        raise ValueError('quantiles and percentages must have the same length.')

    previous_quantile = 0
    for quantile in quantiles:
        if quantile <= previous_quantile or quantile > 100:
            raise ValueError('quantiles must be strictly increasing and end at 100.')
        previous_quantile = quantile
    if quantiles[-1] != 100:
        raise ValueError('The last quantile must be 100.')

    total_percentage = float(sum(percentages))
    if total_percentage <= 0:
        raise ValueError('percentages must sum to a positive value.')

    if data.empty or n_samples == 0:
        return data.iloc[0:0].copy()

    # Establish one global ordering that is reused by every later selection step.
    sorted_df = data.sort_values(
        by=[col_A, col_B, col_C, col_D],
        ascending=[False, False, True, True],
        kind='mergesort',
    ).reset_index(drop=True)

    target_n = min(n_samples, len(sorted_df))
    selected_positions = []
    selected_set = set()
    selected_group_counts = {}
    nan_group = object()

    def normalize_group_value(value):
        # Treat all NaN values in col_D as one logical group.
        return nan_group if pd.isna(value) else value

    def pick_from_positions(positions, count):
        # Work only with rows that were not already selected elsewhere.
        available_positions = [position for position in positions if position not in selected_set]
        picked = []

        # Repeated passes over the slice prefer different col_D values first.
        while len(picked) < count and available_positions:
            groups_seen_in_pass = set()
            next_available_positions = []

            for position in available_positions:
                group_value = normalize_group_value(sorted_df.at[position, col_D])
                if group_value in groups_seen_in_pass:
                    # Defer repeated groups in this pass so other groups get a chance first.
                    next_available_positions.append(position)
                    continue

                picked.append(position)
                groups_seen_in_pass.add(group_value)
                if len(picked) == count:
                    return picked

            available_positions = next_available_positions

        return picked

    all_positions = list(range(len(sorted_df)))

    if occurrences_per_col_D > 0:
        # Stage 1: keep the first X occurrences of each col_D in global order.
        for position in all_positions:
            if len(selected_positions) >= target_n:
                break
            group_value = normalize_group_value(sorted_df.at[position, col_D])
            current_group_count = selected_group_counts.get(group_value, 0)
            if current_group_count >= occurrences_per_col_D:
                continue
            selected_positions.append(position)
            selected_set.add(position)
            selected_group_counts[group_value] = current_group_count + 1

    remaining_needed = target_n - len(selected_positions)
    if remaining_needed <= 0:
        return sorted_df.iloc[selected_positions].reset_index(drop=True)

    remainder_positions = [position for position in all_positions if position not in selected_set]
    if not remainder_positions:  # pragma: no cover - unreachable after target_n=min(n_samples,len(sorted_df))
        return sorted_df.iloc[selected_positions].reset_index(drop=True)

    # Convert percentage weights into integer pick counts that sum exactly to the remainder.
    raw_allocations = [remaining_needed * (float(weight) / total_percentage) for weight in percentages]
    allocation_counts = [int(np.floor(value)) for value in raw_allocations]
    leftover = remaining_needed - sum(allocation_counts)

    if leftover > 0:
        ranked_fractional_parts = sorted(
            range(len(raw_allocations)),
            key=lambda index: (raw_allocations[index] - allocation_counts[index]),
            reverse=True,
        )
        for index in ranked_fractional_parts[:leftover]:
            allocation_counts[index] += 1

    # Stage 2: split the unselected tail by percentile position in the ordered remainder.
    remainder_count = len(remainder_positions)
    lower_quantile = 0
    for upper_quantile, allocation_count in zip(quantiles, allocation_counts):
        start = int(np.floor((lower_quantile / 100.0) * remainder_count))
        end = int(np.floor((upper_quantile / 100.0) * remainder_count))
        quantile_positions = remainder_positions[start:end]

        # Within each quantile band, keep the same top-down order but maximize col_D diversity.
        chosen_positions = pick_from_positions(quantile_positions, allocation_count)

        for position in chosen_positions:
            if position in selected_set:  # pragma: no cover - pick_from_positions already excludes selected_set
                continue
            selected_positions.append(position)
            selected_set.add(position)
            group_value = normalize_group_value(sorted_df.at[position, col_D])
            selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1

        lower_quantile = upper_quantile

    if len(selected_positions) < target_n:
        # Final backfill: if some bins could not satisfy their quota, fill from the full remainder.
        fill_positions = pick_from_positions(remainder_positions, target_n - len(selected_positions))
        for position in fill_positions:
            if position in selected_set:  # pragma: no cover - pick_from_positions already excludes selected_set
                continue
            selected_positions.append(position)
            selected_set.add(position)
            group_value = normalize_group_value(sorted_df.at[position, col_D])
            selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1

    return sorted_df.iloc[selected_positions[:target_n]].reset_index(drop=True)

def _augment_samples_based_on_ratio(original_exposure, original_sigma, exposure_ratio):
    """Compute simulated sigma and exposure time from an exposure-ratio factor.

    Parameters
    ----------
    original_exposure : float
        Exposure time of the original image (seconds).
    original_sigma : float
        Background noise standard deviation of the original image.
    exposure_ratio : float
        Ratio ``original_exposure / simulated_exposure``; must be positive.

    Returns
    -------
    simulated_sigma : float
        Noise standard deviation for the simulated shorter exposure
        (scales as ``sqrt(ratio) * original_sigma``).
    simulated_exposure : float
        Simulated exposure time in seconds; 0.0 when *exposure_ratio* <= 0.
    """
    if exposure_ratio <= 0.0:
        return 0.0, 0.0

    simulated_exposure = original_exposure / exposure_ratio
    simulated_sigma = np.sqrt(original_exposure / simulated_exposure) * original_sigma
    return simulated_sigma, simulated_exposure

def _augment_samples_based_on_range(sigma, lowest_power=-4, highest_power=5, n_samples_per_magnitude=3):
    """Generate candidate sigma values by sweeping base/exponent bins above *sigma*.

    Starting from the exponent and base of *sigma*, iterates upward through
    exponents up to *highest_power*, sampling up to *n_samples_per_magnitude*
    evenly-spaced base digits per order of magnitude.

    Parameters
    ----------
    sigma : float
        Input noise sigma used as the lower starting point.
    lowest_power : int, optional
        Minimum exponent to process. Default is -4.
    highest_power : int, optional
        Maximum exponent (inclusive). Default is 5.
    n_samples_per_magnitude : int, optional
        Maximum number of base digits to sample per order of magnitude.
        Default is 3.

    Returns
    -------
    list of tuple
        Each element is ``(sigma_value, base, original_exponent,
        exponent_diff)``.
    """
    sigmas = []
    if is_not_nan(sigma) and sigma != 0.0:
        exponent = np.floor(np.log10(abs(sigma))).astype(int)
        base = np.floor(sigma / (10.0**exponent))
        if lowest_power <= exponent <= highest_power:
            for x in range(exponent, highest_power + 1):
                start = base if x == exponent else 1
                stop = 9
                for b in evenly_spaced_numbers(start, stop, n_samples_per_magnitude):
                    sigma_value = b * 10.0**x
                    sigmas.append((sigma_value, b, exponent, x - exponent))
    return sigmas

def candidates_based_on_range(row, kwargs_data):
    """
    Expand one metadata row into sigma-range candidate rows.

    The function reads configured source columns (name, exposure, means, and
    location) and generates target `combined_sigma` values by enumerating
    base/exponent bins around the original sigma. For each target value it
    computes `diff_sigma`, `new
    _exp_time`, and `exp_ratio` and emits a row that
    downstream filters/samplers can consume. This path is used when
    `candidates_fn` is `sigma_range`.

    Parameters
    ----------
    row : pandas.Series
        One metadata row from the source CSV.
    kwargs_data : dict
        Config dictionary containing column names and range hyperparameters.

    Returns
    -------
    list[dict]
        Candidate rows with sigma and exposure-derived fields.
    """
    name_col = kwargs_data['name_col']
    location_col = kwargs_data['location_col']
    dataset_col = kwargs_data['dataset']
    exposure_col = kwargs_data['exposure_col']
    lowest_power = kwargs_data['lowest_power']
    highest_power = kwargs_data['highest_power']
    n_samples_per_magnitude = kwargs_data['n_samples_per_magnitude']

    scm = kwargs_data['stats_column_map']
    prefix = kwargs_data['original_stats_prefix']
    org_scm = {key: f'{prefix}{value}' for key, value in scm.items()}

    sigma = row[scm['std_bkg']]
    if not is_not_nan(sigma):
        sigma = row[org_scm['std_bkg']]
    if not is_not_nan(sigma):
        return []
    sigma = float(sigma)

    exposure_value = row[exposure_col]
    if not is_not_nan(exposure_value):
        return []
    exposure_value = float(exposure_value)

    result = []
    for new_sigma, base, exponent, exponent_difference in _augment_samples_based_on_range(
        sigma,
        lowest_power,
        highest_power,
        n_samples_per_magnitude,
    ):
        if not is_not_nan(new_sigma):
            continue
        diff_sigma = np.sqrt(np.abs(new_sigma**2 - sigma**2))
        new_exp_time = exposure_value * (new_sigma / sigma)**2 if sigma != 0 else 0.0
        entry = {
            'name': row[name_col], 'base': base, 'exponent': exponent, 'exponent_diff': exponent_difference,
            'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma,
            'location': row[location_col], 'dataset': row[dataset_col],
            'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], 
            'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']],
            'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']],
            'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']],
            'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']],
            'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']],
            'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']],
            'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']],
            'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0,
            'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0,
            'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0,
            'exp_time': exposure_value, 'new_exp_time': new_exp_time, 'exp_ratio': exposure_value/new_exp_time if new_exp_time > 0 else 0.0
        }
        result.append(entry)
    return result

def candidates_based_on_ratio(row, kwargs_data):
    """
    Expand one metadata row into exposure-ratio candidate rows.

    This strategy builds geometric ratios using `ratio_initial`,
    `ratio_count`, and `ratio_growth`, then converts each ratio to
    `(combined_sigma, new_exp_time)` with `_augment_samples_based_on_ratio`.
    It preserves row-level context fields (`dataset`, `location`, means,
    median) and adds `diff_sigma` and `exp_ratio` for later filtering.

    Parameters
    ----------
    row : pandas.Series
        One metadata row from the source CSV.
    kwargs_data : dict
        Config dictionary with column names and ratio hyperparameters.

    Returns
    -------
    list[dict]
        Candidate rows for the exposure-ratio pipeline.
    """
    name_col = kwargs_data['name_col']
    location_col = kwargs_data['location_col']
    dataset_col = kwargs_data['dataset']
    exposure_col = kwargs_data['exposure_col']

    scm = kwargs_data['stats_column_map']
    prefix = kwargs_data['original_stats_prefix']
    org_scm = {key: f'{prefix}{value}' for key, value in scm.items()}

    ratio_initial = kwargs_data['ratio_initial']
    ratio_count = kwargs_data['ratio_count']
    ratio_growth = kwargs_data['ratio_growth']
    ratios = create_ratios(ratio_initial, ratio_count, ratio_growth)

    sigma = row[scm['std_bkg']]
    if not is_not_nan(sigma):
        sigma = row[org_scm['std_bkg']]
    if not is_not_nan(sigma):
        return []
    sigma = float(sigma)
    if not is_not_nan(row[exposure_col]):
        return []
    exp = float(row[exposure_col])

    result = []
    for ratio in ratios:
        new_sigma, new_exp = _augment_samples_based_on_ratio(exp, sigma, ratio)
        diff_sigma = np.sqrt(np.abs(new_sigma**2 - sigma**2))
        result.append({
            'name': row[name_col],
            'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma,
            'location': row[location_col], 'dataset': row[dataset_col], 
            'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], 
            'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']],
            'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']],
            'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']],
            'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']],
            'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']],
            'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']],
            'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']],
            'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0,
            'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0,
            'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0,
            'exp_time': exp, 'new_exp_time': new_exp, 'exp_ratio': ratio
        })
    return result

def post_filter(info, kwargs_data): 
    """
    Apply quality cuts and compute NSR-derived ranking columns.

    The filter enforces positive-valued sigma/means/exposure rows, requires
    `combined_sigma > sigma`, and then bounds `exp_ratio` with
    `min_exp_ratio`/`max_exp_ratio` from config. It computes `NSR_org`,
    `NSR_simulated`, and `NSR_ratio` to support ratio-based sampling in
    `sample_ratio`. This keeps only physically plausible candidates for
    training/evaluation generation.

    Parameters
    ----------
    info : pandas.DataFrame
        Candidate table produced by a candidates strategy.
    kwargs_data : dict
        Config dictionary containing ratio bounds.

    Returns
    -------
    pandas.DataFrame
        Filtered table with NSR helper columns.
    """
    min_ratio = float(kwargs_data['min_exp_ratio'])
    max_ratio = float(kwargs_data['max_exp_ratio'])
    min_exp_time = float(kwargs_data['min_exp_time'])

    info = info.dropna().reset_index(drop=True)
    numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio']
    for column_name in numeric_columns:
        if column_name in info:
            info[column_name] = pd.to_numeric(info[column_name], errors='coerce')
    info = info.dropna(subset=[column for column in numeric_columns if column in info]).reset_index(drop=True)

    info = info[info['combined_sigma'] > info['org_sigma']]
    info = info[info['exp_time'] > info['new_exp_time']]
    info = info[info['exp_ratio'] >= min_ratio]
    info = info[info['exp_ratio'] <= max_ratio]
    info = info[info['new_exp_time'] >= min_exp_time]
    return info

def sample_range(info, x, kwargs_data):
    """Sample candidates for sigma-range mode using base/exponent ordering.

    Parameters
    ----------
    info : pandas.DataFrame
        Candidate table from :func:`post_filter`.
    x : int
        Requested number of output rows.
    kwargs_data : dict
        Config dictionary; must contain ``'delta'``.

    Returns
    -------
    pandas.DataFrame
        Sampled rows.
    """
    return filtering_df(info, x, kwargs_data['delta'], 'dataset', 'abs_mean',
                        base='base', exponent='exponent_diff', noise_ratio=None)
def sample_ratio(info, x, kwargs_data):
    """Sample candidates for exposure-ratio mode using NSR_ratio ordering.

    Parameters
    ----------
    info : pandas.DataFrame
        Candidate table from :func:`post_filter`.
    x : int
        Requested number of output rows.
    kwargs_data : dict
        Config dictionary; must contain ``'delta'``.

    Returns
    -------
    pandas.DataFrame
        Sampled rows.
    """
    return filtering_df(info, x, kwargs_data['delta'], 'dataset', 'abs_mean',
                        base=None, exponent=None, noise_ratio='exp_ratio')

def sample_range_v2(info, x, kwargs_data):
    """Sample candidates using the quantile-based v2 strategy.

    Parameters
    ----------
    info : pandas.DataFrame
        Candidate table from :func:`post_filter`.
    x : int
        Requested number of output rows.
    kwargs_data : dict
        Config dictionary; must contain ``'occurrences_per_col_D'``,
        ``'quantiles'``, and ``'percentages'``.

    Returns
    -------
    pandas.DataFrame
        Sampled rows.
    """
    return filtering_df_v2(info, x, col_A='exp_ratio',
                            col_B='sm_peak_NSR', col_C='dataset', col_D='location', 
                            occurrences_per_col_D=kwargs_data['occurrences_per_col_D'],
                            quantiles=kwargs_data['quantiles'], percentages=kwargs_data['percentages']
                            )

def resolve_registry_function(category, name):
    """
    Resolve a strategy function by category/name from local registries.

    Categories cover candidate generation, post filtering, sampling, sigma
    kernel selection, noise simulation, and stats filename formatting.
    The function validates that `name` exists in the chosen category registry
    and raises a descriptive `ValueError` otherwise. This keeps
    `new_train.data_augment_pluggable` fully config-driven while avoiding
    hardcoded function branching.

    Parameters
    ----------
    category : str
        Registry family key (for example `sample_fn` or `noise_fn`).
    name : str
        Concrete strategy name within that family.

    Returns
    -------
    callable
        Resolved strategy function.
    """
    # DONE
    if category == 'candidates_fn':
        candidates_registry = {
            'sigma_range': candidates_based_on_range,
            'exposure_ratio': candidates_based_on_ratio,
        }
        if name not in candidates_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return candidates_registry[name]

    # DONE
    if category == 'post_filter_fn':
        post_filter_registry = {
            'default': post_filter,
        }
        if name not in post_filter_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return post_filter_registry[name]

    # DONE
    if category == 'sample_fn':
        sample_registry = {
            'sigma_range': sample_range,
            'exposure_ratio': sample_ratio,
            'exposure_ratio_v2': sample_range_v2,
        }
        if name not in sample_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return sample_registry[name]

    # DONE
    if category == 'noise_fn':
        noise_registry = {
            '_simulated_image_from_exposure': _simulated_image_from_exposure,
            '_simulated_image_from_poisson': _simulated_image_from_poisson,
        }
        if name not in noise_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return noise_registry[name]

    # DONE
    if category == 'sigma_kernel_fn':
        sigma_kernel_registry = {
            'fit': _sigma_kernel_from_fit_wrapper,
            'row': _sigma_kernel_from_row_wrapper,
        }
        if name not in sigma_kernel_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return sigma_kernel_registry[name]

    # DONE
    if category == 'stats_name_fn':
        stats_name_registry = {
            'sigma': _stats_name_from_sigma,
            'exposure': _stats_name_from_exposure,
            'exp_ratio': _stats_name_from_exp_ratio,
        }
        if name not in stats_name_registry:
            raise ValueError(f'Unsupported {category}: {name}')
        return stats_name_registry[name]

    raise ValueError(f'Unsupported strategy category: {category}')
