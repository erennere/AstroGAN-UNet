"""Dataset creation pipeline for FITS-based training and evaluation splits."""

import os, logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_sources, detect_threshold
from photutils.background import Background2D, SExtractorBackground
from photutils.utils import circular_footprint
from src.data.mast import download_images
from src.training.utils import open_fits, save_fits, ensure_directory_exists, ensure_parent_dir_exists
from starter import load_config, parse_config_overrides  #sym:parse_config_overrides

####### HELPER FUNCTIONS###########################
def create_dir(save_dir):
    """Create a directory if it does not already exist.

    Parameters
    ----------
    save_dir : str
        Path to the directory to create.

    Returns
    -------
    bool
        ``True`` if the directory was created or already exists, ``False`` if
        an error occurred.
    """
    try:
        return ensure_directory_exists(save_dir) is not None
    except Exception as err:
        logging.warning(f'an error occurred while creating the directory: {save_dir}, {err}')
        return False
        
def plot_histogram(df, exp_column, filepath):
    """Plot a histogram for one column of a DataFrame and save to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    exp_column : str
        Column name to plot.
    filepath : str
        Output path where the histogram image is saved.

    Returns
    -------
    None
    """
    mean_val = round(df[exp_column].mean(), 2)
    median_val = round(df[exp_column].median(), 2)
    std_val = round(df[exp_column].std(), 2)

    plt.figure(figsize=(10, 6))
    plt.hist(df[exp_column], bins=30, color='skyblue', edgecolor='black', label=f'N: {len(df)}')
    plt.title(f'Histogram of {exp_column}')
    plt.xlabel(exp_column)
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_val}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='dashed', linewidth=2)

    plt.legend()
    ensure_parent_dir_exists(filepath)
    plt.savefig(filepath)
    plt.close()

################### IMAGE CROPPING AND STATISTICS #########################
def _iter_image_crops(image, ps=256):
    """Yield ``(i, j, crop)`` for each padded non-overlapping patch."""
    if not isinstance(image, np.ndarray):
        logging.warning("the image is not an 'numpy.ndarray' but %s", type(image))
        return

    h, w = image.shape
    if h < ps or w < ps:
        logging.warning('the size of the image %s is smaller than %s', (h, w), ps)
        return

    pad_h = (ps - (h % ps)) % ps
    pad_w = (ps - (w % ps)) % ps
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        h, w = image.shape

    n_h = h // ps
    n_w = w // ps
    for i in range(n_h):
        for j in range(n_w):
            yield i, j, image[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]


def crop_image_generator(image, ps=256):
    """Yield padded non-overlapping square crops from an image."""
    for _, _, cropped_image in _iter_image_crops(image, ps=ps):
        yield cropped_image


def crop_image(image, filepath, save_dir, ps=256, crop_name_separator='_', output_extension='.fits', type_of_image='SCI'):
    """Crop an image into patches and save each patch as a FITS file."""
    if filepath is None:
        logging.warning('crop_image: filepath is None; cannot derive patch filename')
        return

    if save_dir is None:
        logging.warning('crop_image: save_dir is None; cannot save cropped files')
        return

    create_dir(save_dir)
    base_name = os.path.basename(filepath).split('.')[0]
    for i, j, cropped_image in _iter_image_crops(image, ps=ps):
        new_filename = f'{base_name}{crop_name_separator}{i}{crop_name_separator}{j}{output_extension}'
        save_fits(cropped_image, new_filename, save_dir, type_of_image=type_of_image)

def process_image_cropping(filepath, save_dir, ps, type_of_image, crop_name_separator, output_extension):
    """
    Opens a FITS file and delegates to crop_image to save all patches.

    Parameters:
        filepath (str): Path to the FITS file to open and crop.
        save_dir (str): Directory where cropped patches are saved.
        ps (int): Patch size in pixels.
        type_of_image (str): HDU name used when opening the FITS file (e.g. 'DRZ', 'SCI').
        crop_name_separator (str): Separator used between base name and patch indices.
        output_extension (str): File extension for saved patches (e.g. '.fits').

    Returns:
        None

    Notes:
        - Returns early with a warning if the file cannot be opened.
        - Patch filenames follow the pattern:
          `<base_name><sep><row_index><sep><col_index><output_extension>`.
    """
    img = open_fits(filepath, type_of_image=type_of_image)
    if img is None:
        logging.warning(f'could not open the image at: {filepath}')
        return
    crop_image(
        img,
        filepath,
        save_dir,
        ps=ps,
        crop_name_separator=crop_name_separator,
        output_extension=output_extension,
        type_of_image=type_of_image,
    )
    return logging.debug(f'cropping of the image at {filepath} is done')

def calculate_image_stats(data: np.ndarray, sigma: float = 3.0, n_sigma: float = 2.0,
                          n_pixels: int = 10, footprint_radius: int = 10, maxiters: int = 10,
                          step: int = 5, bkg_box_size: int = 64, exclude_percentile: float = 10.0):
    """
    Calculate sigma-clipped statistics and additional background statistics for an image.

    Parameters:
        data (np.ndarray): Input 2D image data.
        sigma (float, optional): Sigma value for sigma-clipping during background statistics calculation. Defaults to 3.0.
        n_sigma (float, optional): Threshold level for object detection. Defaults to 2.0.
       n_pixels (int, optional): Minimum number of connected pixels required to detect a source. Defaults to 10.
        footprint_radius (int, optional): Radius of the circular footprint used for masking detected sources. Defaults to 10.
        maxiters (int, optional): Maximum number of iterations for sigma-clipping. Defaults to 10.
        step (int, optional): Step size for percentile calculations. Defaults to 5.
        bkg_box_size (int, optional): Background2D box size in pixels. Defaults to 64.
        exclude_percentile (float, optional): Percentage threshold used by Background2D
            to reject boxes with too few unmasked/finite pixels. Defaults to 10.0.

    Returns:
        tuple:
            - A tuple containing the following:
                - mean_bkg (float): Sigma-clipped mean of the background.
                - median_bkg (float): Sigma-clipped median of the background.
                - std_bkg (float): Sigma-clipped standard deviation of the background.
                - abs_mean (float): Absolute mean of the entire image.
                - abs_median (float): Absolute median of the entire image.
                - mean_src (float): Sigma-clipped mean of the light source regions.
                - median_src (float): Sigma-clipped median of the light source regions.
                - std_src (float): Sigma-clipped standard deviation of the light source regions.
            - mask (np.ndarray): A boolean mask indicating non-source regions (background).
            - percentage_stats (dict): Percentile-based statistics for the image.

    Notes:
        - `abs_mean` is the plain arithmetic mean of the full image (not the mean of absolute values).
          It is computed before source masking and serves as a whole-image reference.
        - The mask returned marks the source regions (True = source pixel); background stats are
          computed on the masked-out (background) pixels, source stats on the unmasked pixels.
        - Percentile-based statistics are calculated using the `classify_data` function.
        - Returns None (not default NaN values) if source detection or masking fails.

    Example:
         stats, mask, percentile_stats = calculate_image_stats(image_data)
    """
    if data is None:
        return
    
    percentage_stats = {}
    try:
        abs_mean = np.mean(data)
        abs_median = np.median(data)
        percentage_stats = classify_data(data, step)
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        background_model = None

        try:
            bkg = Background2D(data, box_size=bkg_box_size, sigma_clip=sigma_clip,  # type: ignore[arg-type]
                               bkg_estimator=SExtractorBackground(),
                               exclude_percentile=exclude_percentile)
            threshold = bkg.background + n_sigma * bkg.background_rms
            background_model = np.asarray(bkg.background, dtype=float)
        except Exception as err:
            logging.warning(f'Background2D estimation failed: {err}. Falling back to global sigma-clipped stats for thresholding.')
            threshold = detect_threshold(data, n_sigma=n_sigma, sigma_clip=sigma_clip)  # type: ignore[arg-type]

        segment_img = detect_sources(data, threshold, n_pixels=n_pixels)
        if segment_img is None:
            logging.warning('No sources detected; returning')
            return None
        if not hasattr(segment_img, 'make_source_mask'):
            logging.warning('Segmentation object does not expose make_source_mask; returning')
            return None
        
        footprint = circular_footprint(radius=footprint_radius)
        mask = segment_img.make_source_mask(footprint=footprint)  # type: ignore[attr-defined]
        mask = np.asarray(mask, dtype=bool)

        if background_model is None:
            mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(data, sigma=sigma, mask=mask)
            max_bkg = np.max(data[~mask]) if np.any(~mask) else np.nan
            source_data = data - median_bkg
        else:
            mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(background_model, sigma=sigma, mask=mask)
            max_bkg = np.max(background_model[~mask]) if np.any(~mask) else np.nan
            source_data = data - background_model

        mean_src, median_src, std_src = sigma_clipped_stats(source_data, sigma=sigma, mask=~mask)
        max_src = np.max(source_data[mask]) if np.any(mask) else np.nan
        return (mean_bkg, median_bkg, abs(std_bkg), max_bkg, mean_src, median_src, abs(std_src), max_src, abs_mean, abs_median), np.array(mask).reshape(data.shape), percentage_stats
    except Exception as err:
        logging.warning(f'an error occurred while masking the light source: {err}')
        return None
    
def classify_data(data, step):
    """Classify data into percentile bins and compute summary statistics per bin.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    step : int
        Step size for percentile boundaries (e.g. 10 for deciles).

    Returns
    -------
    dict
        Keys ``'<p>_mean'``, ``'<p>_median'``, ``'<p>_std'``, ``'<p>_max'``
        for each percentile boundary *p* in ``range(0, 101, step)``.
    """
    data = data.copy()
    data = np.abs(np.asarray(data)).flatten()
    data = np.sort(data)

    if step <= 0:
        step = 1

    percentiles = np.arange(0, 101, step, dtype=int)
    if percentiles[-1] != 100:
        percentiles = np.append(percentiles, 100)

    stats = {}
    for p in percentiles:
        cutoff = len(data) * p // 100
        subset = data[:cutoff]

        if subset.size == 0:
            stats[f'{p}_mean'] = np.nan
            stats[f'{p}_median'] = np.nan
            stats[f'{p}_std'] = np.nan
            stats[f'{p}_max'] = np.nan
            continue

        stats[f'{p}_mean'] = np.mean(subset)
        stats[f'{p}_median'] = np.median(subset)
        stats[f'{p}_std'] = np.abs(np.std(subset))
        stats[f'{p}_max'] = np.max(subset)

    return stats


####################FILTERING AND DOWNLOADING DATA################################

def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000,
                        seed=42, temp_index_column='temp_index', max_iterations=1, filter_surveys=True, 
                        filter_by_last_name=False, last_name_filter_value=[], 
                        last_name_col=None):
    """
    Filters and samples metadata from a CSV file based on specific conditions.

    Parameters:
        filepath (str): The path to the CSV file containing the metadata.
        col (str): The column name to filter based on allowed survey values.
        exp_column (str): The column name for applying range filtering and sampling.
        allowed_survey (list): A list of allowed values for the `col` column.
        size (int, optional): The desired size of the final filtered dataset. Defaults to 1000.
        low (int, optional): The minimum acceptable value for the `exp_column`. Defaults to 100.
        high (int, optional): The maximum acceptable value for the `exp_column`. Defaults to 10000.
        seed (int, optional): The random seed for shuffling and sampling. Defaults to 42.
        temp_index_column (str, optional): Temporary helper column name used while sampling. Defaults to 'temp_index'.
        max_iterations (int, optional): Maximum number of resampling iterations after the unique-pass seed set. Defaults to 1.
        filter_surveys (bool, optional): If True, filter rows by `allowed_survey`. If False, skip survey filtering. Defaults to True.
        filter_by_last_name (bool, optional): If True, filter rows by PI last name. Defaults to False.
        last_name_filter_value (list, optional): List of last names to filter by if `filter_by_last_name` is True. Defaults to [].
        last_name_col (str, optional): The column name for PI last names. Required if `filter_by_last_name` is True. Defaults to None.

    Returns:
        pd.DataFrame: A filtered and sampled DataFrame, sorted by `exp_column`.
                      Returns `None` if errors occur during processing.

    Notes:
        - The function filters the data to include only rows where `col` has a value in `allowed_survey` 
          and `exp_column` values lie within the specified range (`low` to `high`), unless filter_surveys=False.
        - It ensures the resulting dataset contains unique values in `exp_column` and attempts 
          to reach the desired size (`size`) by adding samples based on a normal distribution 
          if the initial filtering does not yield enough rows.
    """
    if not os.path.exists(filepath):
        logging.warning(f'no file found at: {filepath}')
        return
    try:
        df = pd.read_csv(filepath)
    except Exception as err:
        logging.warning(f'{filepath} is not a valid csv file: {err}')
        return
    logging.info('Loaded metadata CSV: %d rows from %s', len(df), filepath)

    if col not in df:
        logging.warning(f'{col} is not in the csv file')
        return  
    if exp_column not in df:
        logging.warning(f'{exp_column} is not in the csv file')
        return  
    
    if filter_surveys:
        df = df[df[col].isin(allowed_survey)]
        logging.info('After survey filter (%s in %s): %d rows remain', col, allowed_survey, len(df))
    else:
        logging.info('Survey filtering disabled; keeping all %d rows', len(df))
    if filter_by_last_name:
        if last_name_col is None:
            logging.warning('last_name_col must be specified when filter_by_last_name is True')
            return
        df = df[df[last_name_col].isin(last_name_filter_value)].reset_index(drop=True)
        logging.info('After last name filter (%s contains any of %s): %d rows remain', last_name_col, last_name_filter_value, len(df))
    
    df = df[(df[exp_column] <= high) & (df[exp_column] >= low)]
    logging.info('After exposure filter (%s in [%s, %s]): %d rows remain', exp_column, low, high, len(df))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df[temp_index_column] = np.arange(0, len(df))
    part1 = df.drop_duplicates(subset=[exp_column], keep='first')
    part1.reset_index(drop=True, inplace=True)

    results = [part1]
    already_taken = set(part1[temp_index_column].tolist())
    iterations = 0
    np.random.seed(seed)
    while len(already_taken) < size and iterations < max_iterations:
        subdf = df[~df[temp_index_column].isin(already_taken)].copy()
        subdf.sort_values(by=[exp_column], inplace=True)
        subdf.reset_index(drop=True, inplace=True)

        if subdf.empty:
            break
        additional_size = size - len(already_taken)
        if additional_size <= 0:
            break

        indices = np.random.normal(loc=len(subdf) // 2, scale=len(subdf) // 4, size=additional_size)
        unique_indices = np.unique(indices)
        unique_indices = np.unique(np.clip(unique_indices, 0, len(subdf) - 1).astype(int))

        part2 = subdf.iloc[unique_indices]
        results.append(part2)
        already_taken.update(part2[temp_index_column].tolist())
        iterations += 1

    results = pd.concat(results, ignore_index=True)
    results.drop([temp_index_column], inplace=True, axis=1)
    results.sort_values(by=[exp_column], inplace=True)
    logging.info('Metadata filtering complete: %d rows selected (target size=%d)', len(results), size)
    return results

def test_train_validation_split(dataset_metadata, url_column, split=(60, 20, 20), seed=42):
    """
    Splits a dataset into training, testing, and validation sets.

    Parameters:
        dataset_metadata (pd.DataFrame): Metadata DataFrame containing at least `url_column`.
        url_column (str): Column in `dataset_metadata` whose values are used as split keys
            (typically image URLs).
        split (tuple, optional): Three integers (train%, test%, val%) that must sum to 100.
            Defaults to (60, 20, 20).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray] | None:
            A (training_urls, test_urls, val_urls) tuple of arrays, or None if
            `url_column` is not present in the DataFrame.

    Notes:
        - No check is performed that split percentages sum to 100.
        - Data is shuffled before splitting; the shuffle and sampling both use `seed`.
    """
    np.random.seed(seed)
    if url_column not in dataset_metadata:
        logging.warning(f'{url_column} is not in the metadata provided')
        return

    data = dataset_metadata[url_column].to_numpy()
    np.random.shuffle(data)

    total_len = len(data)
    test_size = total_len * split[1] // 100
    val_size = total_len * split[2] // 100

    test_indices = np.random.choice(np.arange(total_len), size=test_size, replace=False)
    rest_indices = [i for i in range(total_len) if i not in test_indices]
    val_indices = np.random.choice(rest_indices, size=val_size, replace=False)
    train_indices = [i for i in range(total_len) if i not in test_indices and i not in val_indices]
    logging.info(
        'Train/test/val split: %d / %d / %d (total=%d, split=%s)',
        len(train_indices), len(test_indices), len(val_indices), total_len, split,
    )
    return data[train_indices], data[test_indices], data[val_indices]

def download_dataset(metadata, id_column, url_column, save_dir, max_requests=5, reset_after=10):
    """
    Download images from URLs provided in a metadata DataFrame.

    Parameters:
        metadata (pd.DataFrame): The metadata containing image IDs and URLs.
        id_column (str): The column name in the metadata that contains unique image IDs.
        url_column (str): The column name in the metadata that contains URLs for downloading images.
        save_dir (str): Directory where the downloaded images will be saved.
        max_requests (int, optional): Maximum number of concurrent requests. Defaults to 5.
        reset_after (int, optional): Reset the connection pool after this many requests. Defaults to 10.

    Returns:
        Any: Returns the result of the asynchronous image download operation.
    
    Notes:
        - The function validates the presence of `id_column` and `url_column` in the metadata.
        - Downloads images asynchronously using the `download_images` coroutine.
        - Uses `asyncio.run()` to execute the asynchronous download logic.
    """
    if id_column not in metadata:
        logging.warning(f'{id_column} is not in the metadata provided')
        return  
    if url_column not in metadata:
        logging.warning(f'{url_column} is not in the metadata provided')
        return  
    logging.info('Starting async download of %d images to %s (max_requests=%d)', len(metadata), save_dir, max_requests)
    result = asyncio.run(download_images(metadata[id_column], metadata[url_column], save_dir, max_requests, reset_after))
    logging.info('Download complete for %s', save_dir)
    return result

def process_image_stats(filepath, type_of_image, sigma, n_sigma,n_pixels, footprint_radius, maxiters, step,
                        bkg_box_size, exclude_percentile, save,
                        masked_images_dirname, masked_filename_prefix, filename_column, location_col,
                        stats_column_map, nan_value, posinf_value, neginf_value,
                        original_filename_column=None, crop_name_separator='_',
                        crop_prefix_parts=1, original_filename_suffix='_drz.fits'):
    """
    Open a FITS file, replace non-finite values, compute sigma-clipped background and
    source statistics via calculate_image_stats, and return a flat dict of results.

    Parameters:
        filepath (str): Path to the FITS file to process.
        type_of_image (str): HDU name used when opening the FITS file (e.g. 'DRZ', 'SCI').
        sigma (float): Sigma threshold for sigma-clipping.
        n_sigma (float): Detection threshold in sigma units for source detection.
       n_pixels (int): Minimum connected pixels to constitute a detected source.
        footprint_radius (int): Radius of the circular footprint used to dilate the source mask.
        maxiters (int): Maximum sigma-clipping iterations.
        step (int): Percentile step passed to classify_data.
        bkg_box_size (int): Background2D box size in pixels.
        exclude_percentile (float): Background2D exclude_percentile value.
        save (bool): If True and masked_images_dirname is set, save the background-only
            (source-masked) image to disk.
        masked_images_dirname (str | None): Directory where masked images are written.
            Ignored when save is False.
        masked_filename_prefix (str | None): Prefix prepended to the filename when saving the
            masked image. Pass None or '' for no prefix.
        filename_column (str): Key used for the bare filename in the returned dict.
        location_col (str): Key used for the full filepath in the returned dict.
        stats_column_map (dict): Maps stat keys ('mean_bkg', 'median_bkg', 'std_bkg',
            'abs_mean', 'mean_stc', 'median_stc', 'std_stc') to output column names.
        nan_value (float): Replacement for NaN pixels before statistics.
        posinf_value (float): Replacement for +inf pixels before statistics.
        neginf_value (float): Replacement for -inf pixels before statistics.
        original_filename_column (str | None): If provided, the returned dict includes this
            key mapped to the reconstructed original FITS filename (crop → original linkage).
        crop_name_separator (str, optional): Separator used in cropped patch filenames.
            Defaults to '_'.
        crop_prefix_parts (int, optional): Number of leading name parts kept when reconstructing
            the original filename. Defaults to 1.
        original_filename_suffix (str, optional): Suffix appended to the reconstructed original
            filename. Defaults to '_drz.fits'.

    Returns:
        dict | None: A flat dictionary containing filename, location, all stats from
            stats_column_map, percentile statistics from classify_data, and optionally the
            original filename column. Returns None if the file cannot be opened or if
            source masking fails.

    Notes:
        - Non-finite pixel values are replaced before any statistics are computed.
        - The source mask marks source pixels as True; background stats use the complement.
    """
        
    img = open_fits(filepath, type_of_image=type_of_image)
    if img is None:
        logging.warning(f'could not open the image at: {filepath}')
        return None
    img = np.nan_to_num(np.asarray(img), nan=nan_value, posinf=posinf_value, neginf=neginf_value)

    stats = calculate_image_stats(
        img, sigma, n_sigma, n_pixels, footprint_radius, maxiters, step,
        bkg_box_size=bkg_box_size, exclude_percentile=exclude_percentile,
    )
    if stats is None:
        logging.warning(f'skipping file {filepath} because light source masking failed')
        return None
    
    (mean_bkg, median_bkg, std_bkg, max_bkg, mean_src, median_src, std_src, max_src, abs_mean, abs_median), mask, percentage_stats = stats
    if save and masked_images_dirname:
        create_dir(masked_images_dirname)
        masked_prefix = masked_filename_prefix or ''
        save_fits(
            img[mask],
            f'{masked_prefix}{os.path.basename(filepath)}',
            masked_images_dirname,
            type_of_image=type_of_image,
        )
    
    to_be_returned = {
        filename_column: os.path.basename(filepath),
        location_col: filepath,
        stats_column_map['mean_bkg']: mean_bkg,
        stats_column_map['median_bkg']: median_bkg,
        stats_column_map['std_bkg']: std_bkg,
        stats_column_map['max_bkg']: max_bkg,
        stats_column_map['abs_mean']: abs_mean,
        stats_column_map['abs_median']: abs_median,
        stats_column_map['mean_src']: mean_src,
        stats_column_map['median_src']: median_src,
        stats_column_map['std_src']: std_src,
        stats_column_map['max_src']: max_src,
    }

    if original_filename_column:
        original_filename = build_original_filename_from_crop(
            os.path.basename(filepath), crop_name_separator, crop_prefix_parts, original_filename_suffix
        )
        to_be_returned[original_filename_column] = original_filename

    to_be_returned.update(percentage_stats)
    logging.debug('Stats computed for %s', os.path.basename(filepath))
    return to_be_returned

def extract_filename_from_url(url, url_filename_split_token):
    """Extract a filename token from a URL using a configurable split marker."""
    if url is None or pd.isna(url) or url.strip() == '':
        logging.warning('URL is None or NaN; cannot extract filename')
        return None
    basename = url.split(url_filename_split_token)[-1]
    if basename == url:
        return os.path.basename(url)
    return basename

def build_original_filename_from_crop(filename, crop_name_separator, crop_prefix_parts, original_filename_suffix):
    """Map a cropped filename back to its original FITS filename using configurable naming rules."""
    base_parts = filename.split(crop_name_separator)
    base_name = crop_name_separator.join(base_parts[:crop_prefix_parts])
    return f'{base_name}{original_filename_suffix}'

def control_flow(dataset_dir, metadata_filepath, survey_column, exp_column, id_column, url_column, allowed_survey,
                 split_dirs, originals_subdir, masked_images_dirname, file_extension,
                 filtered_metadata_output_file, noisy_filtered_metadata_output_file, cropped_stats_output_file,
                 url_filename_split_token, crop_name_separator, crop_prefix_parts, original_filename_suffix,
                 stats_column_tokens, temp_index_column, filename_column, original_filename_column, location_col,
                 masked_filename_prefix, stats_column_map, original_stats_prefix, max_iterations,
                 nan_value, posinf_value, neginf_value,
                 download=False, cropping=True, stats_on_crops=False, save=True,
                 size=1000, low=100, high=1000, seed=42, split=(60, 20, 20),
                 max_requests=5, reset_after=10, type_of_image='SCI', sigma=3, n_sigma=2,
                n_pixels=10, footprint_radius=10, maxiters=10, bkg_box_size=64,
                 exclude_percentile=10.0, ps=256, max_workers=16, step=5, filter_surveys=True,
                 filter_by_last_name=False, last_name_filter_value=[], last_name_col=None):
    """
    Run the full dataset preparation pipeline in up to three optional phases.

    Phase 1 (download=True):
        Filter the raw metadata CSV, split URLs into train/test/eval subsets, download
        the FITS originals, and crop each original into ps×ps patches.

    Phase 2 (cropping=True):
        For each split, compute sigma-clipped background/source statistics on the
        original images and write an enriched metadata CSV. When stats_on_crops is
        also True, the same stats are computed for every cropped patch in that pass.

    Phase 3 (stats_on_crops=True):
        Merge crop-patch stats with the original-image stats (prefixed with
        original_stats_prefix), filter out patches whose originals have no stats, and
        write the final cropped-image stats CSV.
        Requires cropping=True to have run first (noisy_filtered_metadata_output_file
        must exist and cropped_image_attributes must have been populated).

    Parameters:
        dataset_dir (str): Root output directory for all generated data.
        metadata_filepath (str): Path to the raw MAST metadata CSV.
        survey_column (str): Column used to filter rows by survey label.
        exp_column (str): Column containing exposure time; used for range filtering and
            as the fallback when FITS header exposure values are unavailable.
        id_column (str): Metadata column with unique dataset IDs (passed to downloader).
        url_column (str): Metadata column with FITS download URLs.
        allowed_survey (list[str]): Survey labels to keep during metadata filtering (when filter_surveys=True).
        filter_surveys (bool, optional): Enable survey label filtering. Defaults to True.
        split_dirs (list[str]): Directory names for the dataset splits (e.g. ['training', 'test', 'eval']).
        originals_subdir (str): Sub-directory under each split directory where original FITS are stored.
        masked_images_dirname (str): Sub-directory name under each split directory where
            source-masked images are written (when save=True).
        file_extension (str): Extension used when scanning directories for FITS files (e.g. '.fits').
        filtered_metadata_output_file (str): Output path for the filtered metadata CSV (Phase 1).
        noisy_filtered_metadata_output_file (str): Output path for the metadata CSV enriched
            with per-image stats (Phase 2).
        cropped_stats_output_file (str): Output path for the final cropped-patch stats CSV (Phase 3).
        url_filename_split_token (str): Token used to extract the bare filename from a URL
            (e.g. '%2F' for URL-encoded paths).
        crop_name_separator (str): Separator between base name and patch indices in crop filenames.
        crop_prefix_parts (int): Number of leading name parts used to reconstruct the original filename.
        original_filename_suffix (str): Suffix appended when reconstructing original filenames.
        stats_column_tokens (list[str]): Substrings used to identify stat columns when selecting
            columns for the original-stats prefix rename (e.g. ['mean', 'median', 'std']).
        temp_index_column (str): Temporary column name used internally during metadata sampling.
        filename_column (str): Column name emitted for the bare filename in stats output.
        original_filename_column (str): Column name emitted for the original FITS filename in
            crop-stats output; also used as join key in Phase 3.
        location_col (str): Column name emitted for the full filepath in stats output.
        masked_filename_prefix (str): Prefix added to filenames when saving masked images.
        stats_column_map (dict): Maps internal stat keys to output column names.
        original_stats_prefix (str): Prefix added to original-image stat columns in the final CSV
            to distinguish them from crop-level stat columns.
        max_iterations (int): Maximum resampling iterations in filter_out_metadata.
        nan_value (float): Replacement for NaN pixels in images.
        posinf_value (float): Replacement for +inf pixels.
        neginf_value (float): Replacement for -inf pixels.
        download (bool, optional): Enable Phase 1. Defaults to False.
        cropping (bool, optional): Enable Phase 2. Defaults to True.
        stats_on_crops (bool, optional): Enable cropped-patch stats within Phase 2 and
            the Phase 3 merge. Requires cropping=True. Defaults to False.
        save (bool, optional): Save source-masked FITS files during Phase 2. Defaults to True.
        size (int, optional): Target number of rows after metadata sampling. Defaults to 1000.
        low (int, optional): Minimum exposure time kept during filtering. Defaults to 100.
        high (int, optional): Maximum exposure time kept during filtering. Defaults to 1000.
        seed (int, optional): Random seed for shuffling and sampling. Defaults to 42.
        split (tuple, optional): (train%, test%, val%) split percentages. Defaults to (60, 20, 20).
        max_requests (int, optional): Max concurrent HTTP requests for downloading. Defaults to 5.
        reset_after (int, optional): Reset the HTTP session after this many requests. Defaults to 10.
        type_of_image (str, optional): HDU name for image loading. Defaults to 'SCI'.
        sigma (float, optional): Sigma-clipping threshold. Defaults to 3.
        n_sigma (float, optional): Source detection threshold in sigma. Defaults to 2.
       n_pixels (int, optional): Min connected pixels for source detection. Defaults to 10.
        footprint_radius (int, optional): Footprint radius for source mask dilation. Defaults to 10.
        maxiters (int, optional): Max sigma-clipping iterations. Defaults to 10.
        bkg_box_size (int, optional): Background2D box size in pixels. Defaults to 64.
        exclude_percentile (float, optional): Background2D exclude_percentile threshold.
            Defaults to 10.0.
        ps (int, optional): Patch size in pixels for cropping. Defaults to 256.
        max_workers (int, optional): Thread pool size for parallel image processing. Defaults to 16.
        step (int, optional): Percentile step for classify_data. Defaults to 5.

    Returns:
        None
    """
    # Ensure the root dataset output directory exists before anything is written.
    create_dir(dataset_dir)
    logging.info(
        'control_flow started (download=%s, cropping=%s, stats_on_crops=%s)',
        download, cropping, stats_on_crops,
    )

    # These dicts/lists accumulate results across all split folders so they can
    # be written to CSV in one go at the end of each phase.
    noise_attributes = []        # one dict per original image: background/source stats
    cropped_image_attributes = []  # one dict per cropped patch: stats + original-file linkage

    filtered_metadata = None

    # ── PHASE 1: filter → split → download → crop ────────────────────────────
    if download:
        logging.info('── Phase 1: filter / download / crop ──')
        # Read the raw MAST metadata CSV and keep only rows whose survey label
        # is in `allowed_survey` and whose exposure time is between low and high.
        # The function also samples down to `size` rows using a normal distribution.
        filtered_metadata = filter_out_metadata(metadata_filepath, survey_column, exp_column, 
                                                allowed_survey, size=size, low=low, high=high,
                                                seed=seed, temp_index_column=temp_index_column,
                                                max_iterations=max_iterations, filter_surveys=filter_surveys, 
                                                filter_by_last_name=filter_by_last_name, 
                                                last_name_filter_value=last_name_filter_value, last_name_col=last_name_col)
        if filtered_metadata is None:
            logging.warning(f'no metadata available after filtering with the given conditions: {metadata_filepath}')
            return

        # Persist the filtered metadata so the cropping phase can reload it
        # independently (even if this script is re-run with download=False).
        filtered_output_dir = os.path.dirname(filtered_metadata_output_file)
        create_dir(filtered_output_dir)
        filtered_metadata.to_csv(filtered_metadata_output_file, index=False)
        logging.info('Filtered metadata saved to %s (%d rows)', filtered_metadata_output_file, len(filtered_metadata))

        # Shuffle and divide the URLs into training / test / eval subsets.
        result = test_train_validation_split(filtered_metadata, url_column, split, seed)
        if result is None:
            logging.warning('train-test-validation split failed')
            return
        training_data, test_data, validation_data = result

        # Download originals and crop them for each split directory in turn.
        for data, directory in zip([training_data, test_data, validation_data], split_dirs):
            # `data` is the array of URLs assigned to this split; filter metadata to match.
            temp_data = filtered_metadata[filtered_metadata[url_column].isin(data)]

            split_dir = os.path.join(dataset_dir, directory)          # e.g. dataset/training
            save_dir  = os.path.join(split_dir, originals_subdir)     # e.g. dataset/training/originals
            if not create_dir(save_dir):
                continue

            # Download all FITS files for this split asynchronously.
            result = download_dataset(temp_data, id_column, url_column, save_dir, max_requests=max_requests, reset_after=reset_after)
            if result is None:
                logging.warning(f'image downloading failed for the {directory} split')
                continue

            # Crop every downloaded FITS into ps×ps patches and save them
            # directly inside `split_dir` (one level up from originals).
            fits_to_crop = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(file_extension)]
            logging.info('Cropping %d FITS files for split "%s" (ps=%d)', len(fits_to_crop), directory, ps)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_image_cropping, file, split_dir,
                        ps, type_of_image, crop_name_separator,
                        file_extension,)
                    for file in fits_to_crop
                ]
                for future in as_completed(futures):
                    future.result()  # re-raise any worker exceptions
            logging.info('Cropping complete for split "%s"', directory)

    # ── PHASE 2: compute background/source stats for original images ──────────
    if cropping:
        logging.info('── Phase 2: original-image statistics ──')
        for split_name in split_dirs:
            folder = os.path.join(dataset_dir, split_name)       # e.g. dataset/training
            save_dir = os.path.join(folder, originals_subdir)      # e.g. dataset/training/originals
            if not os.path.exists(folder) or not os.path.exists(save_dir):
                logging.debug('Skipping split "%s": directory not found', split_name)
                continue  # skip if this split was never downloaded

            # For each cropped FITS: detect sources, compute sigma-clipped
            # background and source stats, optionally save the masked image.
            crops_to_process = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(file_extension)]
            logging.info('Computing stats for %d original images in split "%s"', len(crops_to_process), save_dir)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_image_stats, file, type_of_image,
                        sigma, n_sigma,n_pixels, footprint_radius,
                        maxiters, step, bkg_box_size, exclude_percentile, save,
                        os.path.join(folder, masked_images_dirname),  # where to write masked FITS
                        masked_filename_prefix, filename_column,
                        location_col, stats_column_map,
                        nan_value, posinf_value, neginf_value,
                    )
                    for file in crops_to_process
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        noise_attributes.append(result)  # collect stats row
            logging.info('Collected stats for %d/%d original images in split "%s"', len(noise_attributes), len(crops_to_process), save_dir)

            # Optionally run the same stats pipeline on the cropped patches.
            # Unlike originals, no masked image is written (save=False) and
            # original_filename_column is filled so rows can be joined back later.
            if stats_on_crops:
                crops_to_process = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_extension)]
                logging.info('Computing stats for %d cropped patches in split "%s"', len(crops_to_process), folder)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            process_image_stats, file, type_of_image,
                            sigma, n_sigma,n_pixels, footprint_radius,
                            maxiters, step, bkg_box_size, exclude_percentile,
                            False,   # save=False: never write masked FITS for crops
                            None,                    # no masked_images_dirname needed
                            None, filename_column,   # no prefix needed
                            location_col, stats_column_map,
                            nan_value, posinf_value, neginf_value,
                            original_filename_column, crop_name_separator,  # enables original-file linkage
                            crop_prefix_parts, original_filename_suffix,
                        )
                        for file in crops_to_process
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            cropped_image_attributes.append(result)  # collect crop stats row
                logging.info('Collected stats for %d/%d cropped patches in split "%s"', len(cropped_image_attributes), len(crops_to_process), folder)

        # Reload the filtered metadata CSV (written in phase 1) and attach the
        # per-image stats collected above via a left-join on the filename column.
        noise_attributes = pd.DataFrame(noise_attributes)
        filtered_metadata = pd.read_csv(filtered_metadata_output_file)
        # Derive the bare filename from the URL (URL-encoded path, split on token).
        filtered_metadata[filename_column] = filtered_metadata[url_column].apply(
            lambda url: extract_filename_from_url(url, url_filename_split_token)
        )

        filtered_metadata = filtered_metadata.merge(noise_attributes, on=filename_column, how='left')
        # Write the enriched metadata (original images + stats) to disk.
        noisy_output_dir = os.path.dirname(noisy_filtered_metadata_output_file)
        create_dir(noisy_output_dir)
        filtered_metadata.to_csv(noisy_filtered_metadata_output_file, index=False)
        logging.info('Phase 2 complete: enriched metadata saved to %s (%d rows)', noisy_filtered_metadata_output_file, len(filtered_metadata))

    # ── PHASE 3: build final cropped-image stats CSV ──────────────────────────
    if stats_on_crops:
        logging.info('── Phase 3: merge crop-patch stats with original-image stats ──')
        # Phase 3 depends on Phase 2 having populated cropped_image_attributes and
        # written noisy_filtered_metadata_output_file.  Guard against misconfiguration.
        if not cropping:
            logging.error(
                'stats_on_crops=True but cropping=False: Phase 2 was skipped so '
                'noisy_filtered_metadata_output_file does not exist and cropped stats '
                'were not collected. Set cropping=True or run Phase 2 first.'
            )
            return
        cropped_image_attributes = pd.DataFrame(cropped_image_attributes)

        # Reload baseline filtered metadata and map each row to its original filename
        # so it can be joined with the noisy (original-image) stats below.
        filtered_metadata = pd.read_csv(filtered_metadata_output_file)
        filtered_metadata[original_filename_column] = filtered_metadata[url_column].apply(
            lambda url: extract_filename_from_url(url, url_filename_split_token)
        )
        # Load the noisy (original-image) stats CSV and add original_filename so it
        # can be joined with the cropped-image rows.
        noisy_filtered_metadata = pd.read_csv(noisy_filtered_metadata_output_file)
        noisy_filtered_metadata[original_filename_column] = noisy_filtered_metadata[url_column].apply(
            lambda url: extract_filename_from_url(url, url_filename_split_token)
        )
        # Keep only the statistical columns (those whose name contains a token like
        # 'mean', 'median', 'std') plus the join key, then rename them with the
        # original_stats_prefix so they don't collide with the crop-level stats.
        stats_columns = [
            col for col in noisy_filtered_metadata.columns
            if any(token in col for token in stats_column_tokens)
        ]
        cols = [original_filename_column]
        cols.extend(stats_columns)
        noisy_filtered_metadata = noisy_filtered_metadata[cols]
        noisy_filtered_metadata.rename(
            columns={f'{col}': f'{original_stats_prefix}{col}' for col in cols if col != original_filename_column},
            inplace=True,
        )

        # Attach original-image stats to filtered_metadata via original_filename join.
        filtered_metadata = filtered_metadata.merge(noisy_filtered_metadata, on=original_filename_column, how='left')

        # Final join: for every surviving crop patch, attach the full metadata row
        # of its parent original image (including the prefixed original-image stats).
        cropped_image_attributes = filtered_metadata.merge(
            cropped_image_attributes,
            on=original_filename_column,
            how='left',
        )

        cropped_output_dir = os.path.dirname(cropped_stats_output_file)
        create_dir(cropped_output_dir)
        cropped_image_attributes.to_csv(cropped_stats_output_file, index=False)
        logging.info('Phase 3 complete: crop stats CSV saved to %s (%d rows)', cropped_stats_output_file, len(cropped_image_attributes))

def main():
    """Load YAML config and run the full dataset preparation pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    overrides = parse_config_overrides()  # parses sys.argv by default
    cfg = load_config(**overrides)
    dataset_cfg = cfg['create_dataset']

    control_flow(
        dataset_dir=dataset_cfg['dataset_dir'],
        metadata_filepath=dataset_cfg['metadata_filepath'],
        survey_column=dataset_cfg['survey_column'],
        exp_column=dataset_cfg['exp_column'],
        id_column=dataset_cfg['id_column'],
        url_column=dataset_cfg['url_column'],
        allowed_survey=dataset_cfg['allowed_survey'],
        filter_surveys=dataset_cfg['filter_surveys'],
        split_dirs=dataset_cfg['split_dirs'],
        originals_subdir=dataset_cfg['originals_subdir'],
        masked_images_dirname=dataset_cfg['masked_images_dirname'],
        file_extension=dataset_cfg['file_extension'],
        filtered_metadata_output_file=dataset_cfg['filtered_metadata_output_file'],
        noisy_filtered_metadata_output_file=dataset_cfg['noisy_filtered_metadata_output_file'],
        cropped_stats_output_file=dataset_cfg['cropped_stats_output_file'],
        url_filename_split_token=dataset_cfg['url_filename_split_token'],
        crop_name_separator=dataset_cfg['crop_name_separator'],
        crop_prefix_parts=dataset_cfg['crop_prefix_parts'],
        original_filename_suffix=dataset_cfg['original_filename_suffix'],
        stats_column_tokens=dataset_cfg['stats_column_tokens'],
        temp_index_column=dataset_cfg['temp_index_column'],
        filename_column=dataset_cfg['filename_column'],
        original_filename_column=dataset_cfg['original_filename_column'],
        location_col=dataset_cfg['location_col'],
        masked_filename_prefix=dataset_cfg['masked_filename_prefix'],
        stats_column_map=dataset_cfg['stats_column_map'],
        original_stats_prefix=dataset_cfg['original_stats_prefix'],
        max_iterations=dataset_cfg['max_iterations'],
        nan_value=dataset_cfg['nan_value'],
        posinf_value=dataset_cfg['posinf_value'],
        neginf_value=dataset_cfg['neginf_value'],
        download=dataset_cfg['download'],
        cropping=dataset_cfg['cropping'],
        stats_on_crops=dataset_cfg['stats_on_crops'],
        save=dataset_cfg['save'],
        size=dataset_cfg['size'],
        low=dataset_cfg['low'],
        high=dataset_cfg['high'],
        seed=dataset_cfg['seed'],
        split=tuple(dataset_cfg['split']),
        max_requests=dataset_cfg['max_requests'],
        reset_after=dataset_cfg['reset_after'],
        type_of_image=dataset_cfg['type_of_image'],
        sigma=dataset_cfg['sigma'],
        n_sigma=dataset_cfg['nsigma'],
        n_pixels=dataset_cfg['npixels'],
        footprint_radius=dataset_cfg['footprint_radius'],
        maxiters=dataset_cfg['maxiters'],
        bkg_box_size=dataset_cfg['bkg_box_size'],
        exclude_percentile=dataset_cfg['exclude_percentile'],
        ps=dataset_cfg['ps'],
        max_workers=dataset_cfg['max_workers'],
        step=dataset_cfg['step'],
        filter_by_last_name=dataset_cfg['filter_by_last_name'],
        last_name_filter_value=dataset_cfg['last_name_filter_value'],
        last_name_col=dataset_cfg['last_name_col'],
    )

if __name__ == "__main__":
    main()