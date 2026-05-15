"""Main training entrypoint and dataset/model wiring for AstroGAN-UNet."""

import os, logging, json, sys, inspect
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import numpy as np
import pandas as pd

# Avoid intermittent oneDNN/MKL conv_transpose primitive crashes on Windows CPU.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from astropy.io import fits
from src.models.network import network, GAN, get_discriminator
from src.training.callback import Callback
from src.training.utils import (
    open_fits,
    save_fits,
    create_tf_dataset,
    load_model,
    build_checkpoint_custom_objects,
    scale_invariant_mae,
    log_cosh_loss,
    ssim_loss,
    ensure_parent_dir_exists,
)
from src.training.math_helpers import (
    apply_scaling_and_stats,
)
from starter import load_config, parse_config_overrides  #sym:parse_config_overrides

MEM_CACHED = None
MEM_CACHED_EVAL = None
MEM_CACHED_TEST = None
INFO_CACHED = None
INFO_CACHED_EVAL = None
INFO_CACHED_TEST = None


def _instantiate_optimizer(optimizer_factory, candidate_kwargs):
    """Instantiate an optimizer using only kwargs accepted by the factory."""
    if not callable(optimizer_factory):
        raise ValueError(f'Optimizer must be callable, got: {type(optimizer_factory)!r}')

    supported_kwargs = dict(candidate_kwargs or {})
    try:
        signature = inspect.signature(optimizer_factory)
        supported_keys = set(signature.parameters.keys())
        supported_kwargs = {k: v for k, v in supported_kwargs.items() if k in supported_keys}
    except (TypeError, ValueError):
        # Builtins/C extensions may not expose signatures; best effort.
        pass

    try:
        return optimizer_factory(**supported_kwargs)
    except TypeError:
        if supported_kwargs:
            logging.warning(
                'Optimizer %s rejected kwargs %s; falling back to default constructor.',
                getattr(optimizer_factory, '__name__', str(optimizer_factory)),
                sorted(supported_kwargs.keys()),
            )
        return optimizer_factory()

#######################ORIGINAL DATA AUGMENTATION (NO LONGER USED) ###############################
##################################################################################################
##################################################################################################


def poisson_noise_with_extra_components(img_data, ratio=0.5, exp_time=1, ron=3, dk=7, save=False,
                   sv_name=None, path=None, type_of_image='SCI'):
    """
    Add Poisson, readout, and dark-current noise to an input patch.

    The function treats `img_data` as count-rate data, converts it to counts
    with `exp_time`, simulates a shorter exposure using `ratio`, and injects
    three noise sources: Poisson shot noise, Gaussian readout noise (`ron`),
    and Gaussian dark-current noise scaled by `dk` and simulated exposure.
    Pixels that are exactly zero in the original image are forced back to zero
    after simulation to preserve detector masks/background cutouts.

    Parameters
    ----------
    img_data : numpy.ndarray
        2D image array.
    ratio : float, optional
        Exposure scaling ratio; simulated exposure is `exp_time * ratio`.
    exp_time : float, optional
        Original exposure time in seconds.
    ron : float, optional
        Readout-noise standard deviation.
    dk : float, optional
        Dark-current rate parameter.
    save : bool, optional
        If True, write generated image to disk.
    sv_name : str or None, optional
        Output filename used when `save` is True.
    path : str or None, optional
        Output directory used when `save` is True.

    Returns
    -------
    numpy.ndarray or None
        Simulated noisy image, or None for invalid input/zero simulated exposure.
    """

    if img_data is None or img_data.shape[0] < 2:
        return None
    height, width = img_data.shape[:2]
    simulated_time = exp_time * ratio # in seconds
    if simulated_time == 0.:
        return None
    img = img_data * exp_time
    
    # Generate noise
    dark_current_noise = np.random.normal(0, np.sqrt(dk * simulated_time / (60 * 60)), (height, width))  
    readout_noise = np.random.normal(0, ron, (height, width))
    poisson_noise_img = np.random.poisson(img * ratio)
    noisy_img = (poisson_noise_img + readout_noise + dark_current_noise) / simulated_time
    noisy_img[img_data == 0.0] = 0.0
    if save:
        save_fits(noisy_img, sv_name, path, type_of_image=type_of_image)
    return noisy_img

def black_level(H, W, image, ps=256, steps=100):
    """
    Select a `ps x ps` crop with the lowest fraction of zero-valued pixels.

    This is used for the original-image augmentation path to avoid mostly blank
    detector regions. The function samples `steps` random candidate crops,
    computes zero-pixel percentages for each, and returns the crop with the
    minimum zero fraction. Negative values in the selected crop are clipped to
    zero before returning.

    Parameters
    ----------
    H : int
        Image height.
    W : int
        Image width.
    image : numpy.ndarray
        Input image array.
    ps : int, optional
        Patch size in pixels.
    steps : int, optional
        Number of random candidate crops.

    Returns
    -------
    numpy.ndarray or None
        Selected crop, or None when crop extraction is impossible.
    """
    if H < ps or W < ps or ps == 0:
        return None
    if ps == 0:
        logging.warning("Patch size `ps` is zero. Returning None.")
        return None
    
    xx = np.random.randint(0, H - ps, steps)
    yy = np.random.randint(0, W - ps, steps)

    patch_area = ps * ps
    best_idx = 0

    patches = np.array([image[x:x + ps, y:y + ps] for x, y in zip(xx, yy)])
    zero_counts = np.sum(patches == 0., axis=(1, 2))
    zero_percents = zero_counts / patch_area
    best_idx = np.argmin(zero_percents)
    xx = xx[best_idx]
    yy = yy[best_idx]
    image = image[xx:xx+ps, yy:yy+ps]
    return image

def prepare_patch_pair(gt_patch, func, kwargs_data):
    """
    Transform one clean patch into a `(noisy, clean)` training pair.

    The function applies `func(gt_patch, **kwargs_data)` to synthesize a noisy
    version of the patch, then appends a channel axis to both arrays so they
    match the expected `(H, W, 1)` dataset shape. It returns None when the input
    patch is missing or when synthesis fails, allowing callers to skip invalid
    samples without breaking the generator loop.

    Parameters
    ----------
    gt_patch : numpy.ndarray
        Clean patch.
    func : callable
        Noise/transformation function used to generate the input patch.
    kwargs_data : dict
        Keyword arguments forwarded to `func`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or None
        `(gt_patch, in_patch)` with channel axis, or None on failure.
    """
    if gt_patch is None:
        return None
    try:        
        in_patch = func(gt_patch, **kwargs_data)
        gt_patch = np.expand_dims(gt_patch, axis=(-1))
        in_patch = np.expand_dims(in_patch, axis=(-1))
    except Exception as err:
        logging.warning(f'an error occurred while image handling: {err}')
        return None
    return gt_patch, in_patch

def data_augment(images, kwargs_data):
    """
    Yield augmented patch pairs from raw FITS files using ratio sweeps.

    For each input filepath and each integer ratio in `[start, stop)`, the
    function opens the FITS image with exposure bounds, extracts a low-black
    patch via `black_level`, and generates a noisy counterpart with
    `poisson_noise_with_extra_components`. The output format matches the
    training pipeline: `(input_patch, target_patch, [filepath])`, where patches
    already include a channel axis.

    Parameters
    ----------
    images : list[str]
        FITS file paths.
    kwargs_data : dict
        Augmentation parameters (`start`, `stop`, `ps`, `steps`, `type_of_image`,
        `low`, `high`, and noise kwargs).

    Yields
    ------
    tuple
        `(in_patch, gt_patch, [filepath])` for each valid generated sample.
    """
        
    kwargs_data = {k:v for k, v in kwargs_data.items()}
    patch_pair_kwargs = {k: v for k, v in kwargs_data.items() if k not in ['ps', 'steps', 'high', 'low']}
    
    if kwargs_data['start'] >= kwargs_data['stop']:
        raise ValueError("`start` must be less than `stop`.")
    data = [(filepath, i) for filepath in images for i in range(kwargs_data['start'], kwargs_data['stop'])]
    random.shuffle(data)
    
    for filepath, ratio_ in data:
        try:
            result = open_fits(filepath, ratio_, type_of_image=kwargs_data['type_of_image'],
                                low=kwargs_data['low'], high=kwargs_data['high'])
            if result is None:
                logging.debug(f"could not open file {filepath} due to invalid result.")
                continue
            
            image, exp_time, ratio_ = result
            gt_patch = black_level(image.shape[0], image.shape[1], image, ps=kwargs_data['ps'], steps=kwargs_data['steps'])
            patch_pair_kwargs['ratio'] = ratio_
            patch_pair_kwargs['exp_time'] = exp_time
            result = prepare_patch_pair(gt_patch=gt_patch, func=poisson_noise_with_extra_components, 
                                  kwargs_data=patch_pair_kwargs)
            if result is None:
                logging.info(f"Skipping file {filepath} due to error in prepare_patch_pair processing.")
                continue
            gt_patch, in_patch = result
            yield (in_patch, gt_patch, [filepath])
        except Exception as err:
            logging.error(f"An error occurred while processing file {filepath}: {err}")
            continue

################### PREPARE_DATA STRATEGIES ###################################
###############################################################################

def prepare_data(
    sampled_data,
    fit_data,
    training,
    sigma_kernel_fn,
    noise_fn,
    stats_name_fn,
    type_of_image,
    preprocess_nan_value,
    preprocess_posinf_value,
    preprocess_neginf_value,
    sigma_key,
    scaling=None,
    info_cached_df=None,
    max_workers=8,
):
    """
    Load sampled FITS rows and emit scaled `(noisy, clean, stats)` tensors.

    This generator consumes rows from `sampled_data` and reads FITS files from
    each row's `location` path. For each row it computes a sigma kernel through
    `sigma_kernel_fn(row, fit_data, sigma_key)`, synthesizes noise via
    `noise_fn(file, row, sigma_kernel)`, and builds metadata names
    through `stats_name_fn`. The final arrays are optionally scaled with
    `apply_scaling_and_stats` and returned as `(H, W, 1)` tensors.

    Parameters
    ----------
    sampled_data : pandas.DataFrame
        Selected metadata rows containing at least `name` and strategy fields.
    fit_data : pandas.DataFrame or None
        Fit table used by fit-based sigma kernels.
    training : bool
        Training-mode flag kept for generator compatibility.
    sigma_kernel_fn : callable
        Strategy used to compute per-row noise sigma.
    noise_fn : callable
        Strategy used to synthesize noisy image from clean image + row metadata.
    stats_name_fn : callable
        Strategy used to generate stats filename/token.
    preprocess_nan_value, preprocess_posinf_value, preprocess_neginf_value : float
        Numeric cleanup controls applied before/after simulation.
    sigma_key : str
        Row key consumed by row-based sigma strategy.
    scaling : str or None, optional
        Scaling mode forwarded to `apply_scaling_and_stats`.
    info_cached_df : pandas.DataFrame or None, optional
        Retry candidate pool used to top up failed rows until the generator
        reaches `len(sampled_data)`. When provided, retry rows are sampled from
        this DataFrame while excluding locations that already yielded output;
        otherwise retries are sampled randomly from `sampled_data`.
    max_workers : int, optional
        Max concurrent FITS loads for grouped locations.

    Yields
    ------
    tuple
        `(simulated_image, clean_image, stats)` with channel-expanded images.
    """
    if 'location' not in sampled_data:
        logging.warning('prepare_data: sampled_data has no "location" column; yielding nothing.')
        return

    required_strategies = {
        'sigma_kernel_fn': sigma_kernel_fn,
        'noise_fn': noise_fn,
        'stats_name_fn': stats_name_fn,
    }
    missing_strategies = [name for name, fn in required_strategies.items() if fn is None]
    if missing_strategies:
        raise ValueError(f'Missing required strategy functions: {", ".join(missing_strategies)}')

    target_size = len(sampled_data)
    if target_size <= 0:
        logging.info('prepare_data: sampled_data is empty; yielding nothing.')
        return

    max_workers = max(1, int(max_workers))

    def iter_processed_rows(rows_df):
        grouped_rows = [(filepath, group) for filepath, group in rows_df.groupby('location', sort=False) if pd.notna(filepath)]
        if not grouped_rows:
            return

        with ThreadPoolExecutor(max_workers=min(max_workers, len(grouped_rows))) as executor:
            futures = {
                executor.submit(open_fits, filepath, type_of_image=type_of_image): (filepath, group)
                for filepath, group in grouped_rows
            }

            for future in as_completed(futures):
                filepath, group = futures[future]
                file = future.result()
                if file is None:
                    logging.debug('prepare_data: could not open file %s; skipping group.', filepath)
                    continue

                clean_base_image = np.nan_to_num(
                    file,
                    nan=preprocess_nan_value,
                    posinf=preprocess_posinf_value,
                    neginf=preprocess_neginf_value,
                )

                for row in group.itertuples(index=False):
                    sigma_kernel = sigma_kernel_fn(row, fit_data, sigma_key)
                    sigma_alias = f"{sigma_kernel:.5f}".replace('.', '_')
                    simulated_image = noise_fn(clean_base_image, row, sigma_kernel)
                    simulated_image = np.nan_to_num(
                        simulated_image,
                        nan=preprocess_nan_value,
                        posinf=preprocess_posinf_value,
                        neginf=preprocess_neginf_value,
                    )

                    stats_name = stats_name_fn(filepath, row, sigma_alias)
                    scaled = apply_scaling_and_stats(simulated_image, clean_base_image, scaling, stats_name)
                    if scaled is None:
                        logging.debug('prepare_data: scaling returned None for %s; skipping.', filepath)
                        continue

                    simulated_image, clean_image, stats = scaled
                    yield filepath, (np.expand_dims(simulated_image, -1), np.expand_dims(clean_image, -1), stats)

    count = 0
    yielded_locations = set()
    logging.info(
        'prepare_data: processing %d rows (training=%s, scaling=%s)',
        target_size, training, scaling,
    )
    for filepath, payload in iter_processed_rows(sampled_data):
        yielded_locations.add(filepath)
        yield payload
        count += 1
        if count % 100 == 0:
            logging.debug('prepare_data: yielded %d / %d samples.', count, target_size)

    if count >= target_size:
        return

    if info_cached_df is not None:
        retry_pool = info_cached_df
        if 'location' not in retry_pool:
            logging.warning('prepare_data: info_cached_df has no "location" column; using sampled_data for retries.')
            info_cached_df = None
        else:
            retry_pool = retry_pool[~retry_pool['location'].isin(yielded_locations)].copy()
            while count < target_size and not retry_pool.empty:
                retry_index = random.choice(retry_pool.index.tolist())
                retry_candidate = retry_pool.loc[[retry_index]]
                retry_pool = retry_pool[retry_pool['location'] != retry_candidate.iloc[0]['location']]
                result = next(iter_processed_rows(retry_candidate), None)
                if result is None:
                    continue
                filepath, payload = result
                yielded_locations.add(filepath)
                yield payload
                count += 1
                if count % 100 == 0:
                    logging.debug('prepare_data: yielded %d / %d samples.', count, target_size)

    if info_cached_df is None:
        attempts = 0
        max_attempts = max(target_size * 10, 1)
        while count < target_size and attempts < max_attempts:
            attempts += 1
            retry_index = random.choice(sampled_data.index.tolist())
            result = next(iter_processed_rows(sampled_data.loc[[retry_index]]), None)
            if result is None:
                continue
            _, payload = result
            yield payload
            count += 1
            if count % 100 == 0:
                logging.debug('prepare_data: yielded %d / %d samples.', count, target_size)

    if count < target_size:
        logging.warning(
            'prepare_data: exhausted retry candidates after yielding %d / %d samples.',
            count, target_size,
        )

def data_augment_pluggable(images, kwargs_data, scaling=None):
    """
    Build metadata-driven samples using resolver-selected strategy functions.

    The function reads `metadata_filepath`, applies exposure bounds, expands each
    row with `candidates_fn`, filters with `post_filter_fn`, and selects a final
    set with `sample_fn`. It supports in-memory/raw-CSV caching via
    `MEM_CACHED`/`MEM_CACHED_EVAL`/`MEM_CACHED_TEST`, stores the branch-specific
    filtered candidate tables in `INFO_CACHED`/`INFO_CACHED_EVAL`/
    `INFO_CACHED_TEST`, and supports optional sub-sampling through
    `sub_sample_train` and `sub_sample_eval`. Selected rows are then passed to
    `prepare_data`, which performs FITS loading, noise synthesis, and scaling.

    Parameters
    ----------
    images : list
        Unused placeholder for generator API compatibility.
    kwargs_data : dict
        Configuration dict containing metadata paths, strategy names, sampling
        sizes, preprocessing values, and cache file paths.
    scaling : str or None, optional
        Scaling mode forwarded to `prepare_data`.

    Yields
    ------
    tuple
        Items yielded by `prepare_data`.
    """
    metadata_filepath = kwargs_data['metadata_filepath']
    exposure_col = kwargs_data['exposure_col']
    times = float(kwargs_data['times'])
    low = float(kwargs_data['low'])
    high = float(kwargs_data['high'])
    training = kwargs_data['training']
    test = kwargs_data.get('test', False)
    samples = int(float(kwargs_data['samples']))
    val_samples = int(float(kwargs_data['val_samples']))

    candidates_fn = kwargs_data['candidates_fn']
    post_filter_fn = kwargs_data['post_filter_fn']
    sample_fn = kwargs_data['sample_fn']
    sigma_kernel_fn = kwargs_data['sigma_kernel_fn']
    noise_fn = kwargs_data['noise_fn']
    stats_name_fn = kwargs_data['stats_name_fn']

    cache_raw_metadata = kwargs_data['cache_raw_metadata']
    sub_sample_train = kwargs_data['sub_sample_train']
    sub_sample_eval = kwargs_data['sub_sample_eval']
    preprocess_nan_value = kwargs_data['nan_value']
    preprocess_posinf_value = kwargs_data['posinf_value']
    preprocess_neginf_value = kwargs_data['neginf_value']
    sigma_key = kwargs_data['sigma_key']
    max_workers = kwargs_data.get('max_workers', 8)

    fit_data = None
    if kwargs_data.get('sigma_kernel_requires_fit_data', False):
        fit_data_filepath = kwargs_data['fit_data_filepath']
        if not os.path.exists(fit_data_filepath):
            raise FileNotFoundError(f'Fit data CSV not found: {fit_data_filepath}')
        fit_data = pd.read_csv(fit_data_filepath)
        if fit_data.empty:
            raise ValueError(f'Fit data CSV is empty: {fit_data_filepath}')

    global MEM_CACHED, MEM_CACHED_EVAL, MEM_CACHED_TEST
    global INFO_CACHED, INFO_CACHED_EVAL, INFO_CACHED_TEST

    sampled_data = None
    if cache_raw_metadata:
        if test and MEM_CACHED_TEST is not None:
            logging.info('data_augment_pluggable: using cached test metadata (%d rows).', len(MEM_CACHED_TEST))
            sampled_data = MEM_CACHED_TEST
        elif training and MEM_CACHED is not None:
            logging.info('data_augment_pluggable: using cached training metadata (%d rows).', len(MEM_CACHED))
            sampled_data = MEM_CACHED
        elif not training and not test and MEM_CACHED_EVAL is not None:
            logging.info('data_augment_pluggable: using cached eval metadata (%d rows).', len(MEM_CACHED_EVAL))
            sampled_data = MEM_CACHED_EVAL

    if sampled_data is None:
        df = pd.read_csv(metadata_filepath)
        logging.info('data_augment_pluggable: loaded metadata CSV (%d rows) from %s', len(df), metadata_filepath)
        df = df[(df[exposure_col] >= times * low) & (df[exposure_col] <= high)]
        logging.info('data_augment_pluggable: %d rows after exposure filter (times=%s, low=%s, high=%s)', len(df), times, low, high)

        info = []
        for _, row in df.iterrows():
            info.extend(candidates_fn(row, kwargs_data))
        info = pd.DataFrame(info).dropna()
        info = post_filter_fn(info, kwargs_data)
        x = min(int(float(kwargs_data['test_samples'])), len(info)) if test else min(samples if training else val_samples, len(info))

        logging.info('data_augment_pluggable: %d candidates after post-filter; sampling %d (training=%s).', len(info), x, training)

        if not test:
            if training:
                info = info[info['location'].apply(
                    lambda loc: 'training' in str(loc) if pd.notnull(loc) else False)]
                INFO_CACHED = info.copy()
            else:
                info = info[info['location'].apply(
                    lambda loc: 'eval' in str(loc) if pd.notnull(loc) else False)]
                INFO_CACHED_EVAL = info.copy()
        else:
            info = info[info['location'].apply(
                lambda loc: 'test' in str(loc) if pd.notnull(loc) else False)]
            INFO_CACHED_TEST = info.copy()

        sampled_data = sample_fn(info, x, kwargs_data)
        logging.info('data_augment_pluggable: %d rows selected by sample_fn.', len(sampled_data))

        if cache_raw_metadata and test:
            MEM_CACHED_TEST = sampled_data
            ensure_parent_dir_exists(kwargs_data['test_cache_filepath'])
            sampled_data.reset_index(drop=True).to_csv(kwargs_data['test_cache_filepath'], index=False)
            logging.info('data_augment_pluggable: test metadata cached to %s.', kwargs_data['test_cache_filepath'])
        elif cache_raw_metadata and training:
            MEM_CACHED = sampled_data
            ensure_parent_dir_exists(kwargs_data['training_cache_filepath'])
            sampled_data.reset_index(drop=True).to_csv(kwargs_data['training_cache_filepath'], index=False)
            logging.info('data_augment_pluggable: training metadata cached to %s.', kwargs_data['training_cache_filepath'])
        elif cache_raw_metadata and not training:
            MEM_CACHED_EVAL = sampled_data
            ensure_parent_dir_exists(kwargs_data['eval_cache_filepath'])
            sampled_data.reset_index(drop=True).to_csv(kwargs_data['eval_cache_filepath'], index=False)
            logging.info('data_augment_pluggable: eval metadata cached to %s.', kwargs_data['eval_cache_filepath'])

        ensure_parent_dir_exists(kwargs_data['info_filepath'])
        info.to_csv(kwargs_data['info_filepath'], index=False)

    frac = sub_sample_train if training else sub_sample_eval
    if frac is not None:
        n = int(len(sampled_data) * frac)
        sampled_data = sampled_data.sample(n=n).reset_index(drop=True)
        logging.info('data_augment_pluggable: sub-sampled to %d rows (frac=%s).', len(sampled_data), frac)

    unique_locations = sampled_data.location.unique().tolist()
    random.shuffle(unique_locations)
    location_order = {location: idx for idx, location in enumerate(unique_locations)}
    sampled_data = (
        sampled_data
        .assign(_location_order=sampled_data['location'].map(location_order))
        .sort_values('_location_order', kind='stable')
        .drop(columns=['_location_order'])
        .reset_index(drop=True)
    )
    yield from prepare_data(
        sampled_data=sampled_data,
        fit_data=fit_data,
        training=training,
        scaling=scaling,
        sigma_kernel_fn=sigma_kernel_fn,
        noise_fn=noise_fn,
        stats_name_fn=stats_name_fn,
        type_of_image=kwargs_data['type_of_image'],
        preprocess_nan_value=preprocess_nan_value,
        preprocess_posinf_value=preprocess_posinf_value,
        preprocess_neginf_value=preprocess_neginf_value,
        sigma_key=sigma_key,
        info_cached_df=INFO_CACHED if training else INFO_CACHED_EVAL if not training and not test else INFO_CACHED_TEST,
        max_workers=max_workers,
    )

################TRAIN THE MODEL ################################
def train_network(input_shape, n_epochs, kwargs_data, kwargs_network, data_generator, batch_size=32,
         optimizer=Adam, change_learning_rate=[(0, 1e-4), (2000, 1e-5)], G_loss_fn: Any = tf.keras.losses.MeanAbsoluteError(),
         learning_rate=None, beta_1=None,
         start_from_best=False, start_from_last=True,
         save_freq=500, eval_save_percentage=20, ds_save_percentage=30, scaling=None,
         discriminator_kwargs=None, gan_kwargs=None, use_gan=True,
         training_results_dir='./results', training_metrics_csv_path='./training_history.csv',
         training_history_json_path='./training_history.json',
         validation_loss_filename='validation_loss.txt',
         training_metrics_filename='training_metrics.txt',
         config=None):
    """
    Train GAN components using FITS-derived tf.data batches and callback logging.

    The function creates output directories, builds the generator network from
    `kwargs_network`, constructs datasets through `create_tf_dataset`, and
    optionally restores checkpoints via `load_model` using
    `checkpoint_custom_epoch` and `checkpoint_restore_kwargs`. It then builds a
    discriminator + `GAN` wrapper, runs `fit`, and persists training history to
    JSON. Validation sampling behavior is controlled by cloning `kwargs_data`
    with `training=False` for callback-driven evaluation snapshots.

    Parameters
    ----------
    input_shape : tuple
        Input tensor shape for generator/discriminator.
    n_epochs : int
        Total epoch target.
    kwargs_data : dict
        Data configuration including training/eval paths and checkpoint options.
    kwargs_network : dict
        Generator architecture kwargs.
    data_generator : callable
        Sample generator used by `create_tf_dataset`.
    batch_size : int, optional
        Dataset batch size.
    optimizer : callable, optional
        Optimizer factory for callback scheduling context.
    change_learning_rate : list, optional
        Epoch-based LR schedule consumed by callback.
    G_loss_fn : callable, optional
        Generator loss callable (validated as callable).
    learning_rate : float or None, optional
        Shared optimizer learning rate used in both UNet-only and GAN modes.
    beta_1 : float or None, optional
        Shared optimizer beta_1 (for Adam-like optimizers) used in both modes.
    start_from_best, start_from_last : bool, optional
        Checkpoint restore preferences.
    save_freq, eval_save_percentage, ds_save_percentage : int, optional
        Callback save/eval cadence parameters.
    scaling : str or None, optional
        Scaling mode passed into dataset generator.
    discriminator_kwargs, gan_kwargs : dict or None, optional
        Discriminator and GAN-construction kwargs.
    use_gan : bool, optional
        If True, train adversarially with GAN; if False, train generator only.
    training_results_dir : str, optional
        Artifact output location for generated previews/validation logs.
    training_metrics_csv_path, training_history_json_path : str, optional
        Metrics/history output files.
    """

    logging.info('Starting train_network for %s epochs (scaling=%s, use_gan=%s).', n_epochs, scaling, use_gan)

    # Ensures that the dateset exists and checks
    if not callable(G_loss_fn):
        raise ValueError("The 'G_loss_fn' must be callable.")

    if len(tf.config.list_physical_devices('GPU')) == 0 and batch_size > 2:
        logging.warning('No GPU detected; reducing batch_size from %d to %d for CPU stability.', batch_size, 2)
        batch_size = 2
    
    #Reading the imge filepaths from files residing in the respective folders
    training_path = kwargs_data['training_path']
    eval_path = kwargs_data['eval_path']
    results_path = kwargs_data['results_path']

    if training_path is not None and os.path.exists(training_path):
        train_data = [os.path.join(training_path, f) for f in os.listdir(training_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {training_path}')
    if eval_path is not None and os.path.exists(eval_path):
        eval_data = [os.path.join(eval_path, f) for f in os.listdir(eval_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {eval_path}')
    logging.info('Discovered %d training and %d evaluation FITS files.', len(train_data), len(eval_data))

    # Model definition
    start_epoch = 0
    generator = network(input_shape, **kwargs_network)
    optimizer_factory = optimizer
    optimizer_kwargs = {}
    if learning_rate is not None:
        optimizer_kwargs['learning_rate'] = float(learning_rate)
    if beta_1 is not None:
        optimizer_kwargs['beta_1'] = float(beta_1)
    optimizer = _instantiate_optimizer(optimizer_factory, optimizer_kwargs)

    # Restoring prior epochs
    checkpoint_custom_epoch = kwargs_data['checkpoint_custom_epoch']
    checkpoint_restore_kwargs = dict(kwargs_data['checkpoint_restore_kwargs'])
    checkpoint_restore_kwargs['custom_objects'] = build_checkpoint_custom_objects()
    result = load_model(
        results_path,
        start_from_best,
        start_from_last,
        custom_epoch=checkpoint_custom_epoch,
        restore_kwargs=checkpoint_restore_kwargs,
    )
    if result:
        restored_model, start_epoch = result
        if restored_model is not None:
            if use_gan and hasattr(restored_model, 'generator'):
                generator = restored_model.generator
            else:
                generator = restored_model
            logging.info('Restored model state; resuming from epoch %d.', start_epoch)
        else:
            start_epoch = 0
            logging.info('Checkpoint lookup returned no model; training starts from scratch.')
    else:
        logging.info('No checkpoint restored; training starts from scratch.')
    #model.compile(optimizer=optimizer, loss=G_loss_fn)

    if use_gan:
        if discriminator_kwargs is None:
            discriminator_kwargs = {}
        if gan_kwargs is None:
            gan_kwargs = {}
        discriminator = get_discriminator(input_shape=input_shape, **discriminator_kwargs)

        generator_optimizer_kwargs = dict(optimizer_kwargs)

        discriminator_optimizer_kwargs = {}
        if 'd_learning_rate' in gan_kwargs:
            discriminator_optimizer_kwargs['learning_rate'] = float(gan_kwargs['d_learning_rate'])
        elif 'learning_rate' in generator_optimizer_kwargs:
            discriminator_optimizer_kwargs['learning_rate'] = generator_optimizer_kwargs['learning_rate']
        if 'beta_1' in generator_optimizer_kwargs:
            discriminator_optimizer_kwargs['beta_1'] = generator_optimizer_kwargs['beta_1']

        train_model = GAN(
            generator, discriminator,
            g_optimizer=_instantiate_optimizer(optimizer_factory, generator_optimizer_kwargs),
            d_optimizer=_instantiate_optimizer(optimizer_factory, discriminator_optimizer_kwargs),
            adversarial_loss_fn=gan_kwargs['loss_fn'],
            reconstruction_loss_fn=G_loss_fn,
            adversarial_loss_weight=gan_kwargs['adversarial_loss_weight'],
            reconstruction_loss_weight=gan_kwargs['reconstruction_loss_weight'],
            label_smoothing=gan_kwargs['label_smoothing'],
        )
        # Custom Model subclasses still need compile() before fit().
        train_model.compile()
        callback_optimizer = train_model.g_optimizer
    else:
        generator.compile(optimizer=optimizer, loss=G_loss_fn)
        train_model = generator
        callback_optimizer = optimizer
        logging.info('Configured generator-only training mode (no discriminator).')

    if 'training' in kwargs_data:
        kwargs_val = dict(kwargs_data)
        kwargs_val['training'] = False
    else:
        kwargs_val = kwargs_data

    train_dataset = create_tf_dataset(train_data,
                                    data_generator,
                                    generator_kwargs=kwargs_data,
                                    batch_size=batch_size, scaling=scaling, augment=True)
    logging.info('Training dataset created with batch_size=%d.', batch_size)

    validation_dataset = create_tf_dataset(eval_data,
                                        data_generator,
                                        generator_kwargs=kwargs_val,
                                        batch_size=batch_size, scaling=scaling, augment=False)
    logging.info('Validation dataset created with batch_size=%d.', batch_size)

    train_dataset_size = 0
    train_batches = 0
    for x_batch, _, _ in train_dataset:
        train_batches += 1
        train_dataset_size += int(x_batch.shape[0])

    validation_dataset_size = 0
    valid_batches = 0
    for x_batch, _, _ in validation_dataset:
        valid_batches += 1
        validation_dataset_size += int(x_batch.shape[0])

    if train_batches <= 0:
        raise ValueError('Training dataset produced zero batches. Check data filtering/sampling settings.')
    if valid_batches <= 0:
        raise ValueError('Validation dataset produced zero batches. Check data filtering/sampling settings.')

    callback = Callback(
        dataset=train_dataset,
        dataset_size=train_dataset_size,
        validation_dataset=validation_dataset,
        validation_dataset_size=validation_dataset_size,
        epoch=n_epochs,
        start_epoch=start_epoch,
        eval_save_percentage=eval_save_percentage,
        ds_save_percentage=ds_save_percentage,
        save_freq=save_freq,
        sample_generator=data_generator,
        kwargs_validation=kwargs_val,
        optimizer=callback_optimizer,
        change_learning_rate=change_learning_rate,
        batch_size=batch_size,
        scaling=scaling,
        use_gan=use_gan,
        training_results_dir=training_results_dir,
        checkpoints_dir=results_path,
        training_metrics_csv_path=training_metrics_csv_path,
        validation_loss_filename=validation_loss_filename,
        training_metrics_filename=training_metrics_filename,
        config=config,
    )
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #Train the model
    logging.info('Starting %s.fit for %d epochs (effective=%d).', 'GAN' if use_gan else 'network', n_epochs, n_epochs - start_epoch)
    fit_dataset = train_dataset.repeat()
    history = train_model.fit(
        fit_dataset.map(lambda x, y, filepath: (x, y)),
        epochs=n_epochs-start_epoch,
        steps_per_epoch=train_batches,
        callbacks=[callback],
    )
    try:
        ensure_parent_dir_exists(training_history_json_path)
        with open(training_history_json_path, 'w') as f:
            json.dump(history.history, f)
        logging.info('Saved training history JSON to %s.', training_history_json_path)
    except Exception as err:
        logging.warning(f'an error occurred while saving the history: {err}')

def main():
    """Load YAML config, prepare merged runtime kwargs, and launch training."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    overrides = parse_config_overrides()  # parses sys.argv by default
    config_values = load_config(**overrides)
    logging.info('Loaded configuration and switched cwd to project root.')

    training_config = dict(config_values['training'])
    data_config = dict(training_config['data_kwargs'])
    network_config = dict(training_config['network_kwargs'])
    discriminator_config = dict(training_config['discriminator_kwargs'])
    gan_config = dict(training_config['gan_kwargs'])

    logging.info('Resolved paths: training=%s eval=%s models=%s', data_config['training_path'], data_config['eval_path'], data_config['results_path'])

    logging.warning(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    train_network(
        tuple(training_config['patch_size']),
        training_config['n_epochs'],
        data_config,
        network_config,
        data_generator=training_config['data_generator'],
        batch_size=training_config['batch_size'],
        optimizer=training_config['optimizer'],
        change_learning_rate=training_config['change_learning_rate'],
        G_loss_fn=training_config['g_loss_fn'],
        learning_rate=training_config['learning_rate'],
        beta_1=training_config['beta_1'],
        start_from_best=training_config['start_from_best'],
        start_from_last=training_config['start_from_last'],
        save_freq=training_config['save_freq'],
        eval_save_percentage=training_config['eval_save_percentage'],
        ds_save_percentage=training_config['ds_save_percentage'],
        scaling=training_config['scaling'],
        discriminator_kwargs=discriminator_config,
        gan_kwargs=gan_config,
        use_gan=training_config['use_gan'],
        training_results_dir=training_config['training_results_dir'],
        training_metrics_csv_path=training_config['training_metrics_csv_path'],
        training_history_json_path=training_config['training_history_json_path'],
        validation_loss_filename=training_config['validation_loss_filename'],
        training_metrics_filename=training_config['training_metrics_filename'],
        config=config_values,
    )
    
if __name__ == "__main__":
    main()
