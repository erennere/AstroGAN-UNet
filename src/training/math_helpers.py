"""Math and simulation helpers used by data generation and evaluation."""

import os
import numpy as np
import pandas as pd

from src.config_parsing import parse_required_int


LOG_DOMAIN_CLIP_MAX = 80.0


def set_log_domain_clip_max(value):
    """Configure the clipping ceiling used in inverse adaptive-log reconstruction."""
    if value is None:
        raise ValueError('log_domain_clip_max cannot be None.')
    clip_value = float(value)
    if not np.isfinite(clip_value):
        raise ValueError('log_domain_clip_max must be finite.')
    global LOG_DOMAIN_CLIP_MAX
    LOG_DOMAIN_CLIP_MAX = clip_value


def _row_get(row, key):
    """Read a field from either mapping-like rows or attribute-like rows."""
    if isinstance(row, dict):
        return row[key]
    try:
        return row[key]
    except (TypeError, KeyError, IndexError):
        return getattr(row, key)


############ BASIC MATH ######################################################


def linear_function(x, slope, intercept):
    """
    Evaluate the linear expression slope*x + intercept.

    Parameters
    ----------
    x : float or array-like
        Input value or values.
    slope : float
        Linear slope.
    intercept : float
        Linear intercept.

    Returns
    -------
    float or numpy.ndarray
        Evaluated linear value(s).
    """
    return slope * x + intercept


def power_law(x, exponent, amplitude):
    """
    Evaluate the power-law expression amplitude*x**exponent.

    Parameters
    ----------
    x : float or array-like
        Input value or values.
    exponent : float
        Power-law exponent.
    amplitude : float
        Power-law amplitude.

    Returns
    -------
    float or numpy.ndarray
        Evaluated power-law value(s).
    """
    return amplitude * np.power(x, exponent)

def evenly_spaced_numbers(start, stop, count):
    """
    Return up to count unique integers distributed in (start, stop].

    Parameters
    ----------
    start : int
        Lower reference bound.
    stop : int
        Inclusive upper bound.
    count : int
        Maximum number of integers to return.

    Returns
    -------
    list of int
        Sorted unique integer values.
    """
    if stop <= start or count <= 0:
        return []

    lower_bound = start if start == 1 else start + 1
    if lower_bound > stop:
        return []

    sample_count = min(int(count), stop - lower_bound + 1)
    if sample_count <= 0:
        return []

    spaced_values = np.rint(np.linspace(lower_bound, stop, sample_count)).astype(int)
    return np.unique(spaced_values).tolist()


############ DISTRIBUTION / RANGE HELPERS ####################################


def _exp_bounds(series):
    """
    Compute min and max floor(log10(.)) exponents for positive finite values.

    Parameters
    ----------
    series : pandas.Series
        Input numeric series.

    Returns
    -------
    tuple of int
        Minimum and maximum exponent; returns (0, 0) when empty.
    """
    numeric_values = pd.to_numeric(series, errors='coerce').to_numpy()
    finite_abs_values = np.abs(numeric_values[np.isfinite(numeric_values)])
    positive_values = finite_abs_values[finite_abs_values > 0]
    if positive_values.size == 0:
        return 0, 0

    min_exponent = int(np.floor(np.log10(positive_values.min())))
    max_exponent = int(np.floor(np.log10(positive_values.max())))
    return min_exponent, max_exponent


def find_distribution(dataframe, column_name):
    """
    Build per-(base, exponent) mean and std bins for a numeric column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Source dataframe.
    column_name : str
        Numeric column to bin.

    Returns
    -------
    pandas.DataFrame
        Columns are base, exponent, mean, and std.
    """
    min_exponent, max_exponent = _exp_bounds(dataframe[column_name])
    abs_values = dataframe[column_name].abs()
    distribution_rows = []

    for exponent_value in range(min_exponent, max_exponent + 1):
        for base_value in range(1, 10):
            bin_start = base_value * 10.0 ** exponent_value
            bin_end = (base_value + 1) * 10.0 ** exponent_value
            selection_mask = (abs_values >= bin_start) & (abs_values < bin_end)
            selected_values = abs_values[selection_mask]
            distribution_rows.append(
                {
                    'base': base_value,
                    'exponent': exponent_value,
                    'mean': selected_values.mean() if len(selected_values) else np.nan,
                    'std': selected_values.std() if len(selected_values) else np.nan,
                }
            )

    return pd.DataFrame(distribution_rows)


def find_distribution_only_exp(dataframe, column_name):
    """
    Build per-exponent mean and std bins for a numeric column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Source dataframe.
    column_name : str
        Numeric column to bin.

    Returns
    -------
    pandas.DataFrame
        Columns are exponent, mean, and std.
    """
    min_exponent, max_exponent = _exp_bounds(dataframe[column_name])
    abs_values = dataframe[column_name].abs()
    distribution_rows = []

    for exponent_value in range(min_exponent, max_exponent + 1):
        bin_start = 10.0 ** exponent_value
        bin_end = 10.0 ** (exponent_value + 1)
        selection_mask = (abs_values >= bin_start) & (abs_values < bin_end)
        selected_values = abs_values[selection_mask]
        distribution_rows.append(
            {
                'exponent': exponent_value,
                'mean': selected_values.mean() if len(selected_values) else np.nan,
                'std': selected_values.std() if len(selected_values) else np.nan,
            }
        )

    return pd.DataFrame(distribution_rows)


def _apply_range(range_dataframe, row, use_base=True, n_sigma=3):
    """
    Check whether combined_sigma lies inside the bin mean +/- n_sigma*std interval.

    Parameters
    ----------
    range_dataframe : pandas.DataFrame
        Range table with exponent and optional base columns.
    row : pandas.Series
        Row containing combined_sigma and bin identifiers.
    use_base : bool, optional
        If True also match by base, by default True.
    n_sigma : float, optional
        Number of standard deviations for the acceptance band, by default 3.

    Returns
    -------
    bool
        True when combined_sigma is inside the selected interval.
    """
    combined_sigma = abs(_row_get(row, 'combined_sigma'))
    matching_mask = range_dataframe['exponent'].eq(_row_get(row, 'sigma_exponent'))
    if use_base:
        matching_mask &= range_dataframe['base'].eq(_row_get(row, 'sigma_base'))

    selected_rows = range_dataframe.loc[matching_mask, ['mean', 'std']]
    if selected_rows.empty:
        return False

    mean_value, std_value = selected_rows.iloc[0]
    if pd.isna(mean_value) or pd.isna(std_value):
        return False

    lower_bound = mean_value - n_sigma * std_value
    upper_bound = mean_value + n_sigma * std_value
    return lower_bound <= combined_sigma <= upper_bound


############ EXPOSURE / SIMULATION HELPERS ###################################
def _sigma_kernel_from_fit_wrapper(row, fit_data, sigma_key):
    """Bind standard fit keys and delegate to :func:`sigma_kernel_from_fit`.

    Parameters
    ----------
    row : pandas.Series
        Metadata row containing at least ``'exp_time'``.
    fit_data : pandas.DataFrame
        Single-row DataFrame with columns ``'t'``, ``'a'``, ``'dt'``,
        ``'da'``.
    sigma_key : str
        Unused; kept for a consistent wrapper signature.

    Returns
    -------
    float
        Estimated noise sigma from the power-law fit.
    """
    return sigma_kernel_from_fit(row, fit_data, 't', 'a', 'dt', 'da', 'exp_time')
def sigma_kernel_from_fit(row, fit_data, t_key, a_key, dt_key, da_key, exp_time_key):
    """Estimate the noise kernel magnitude from fitted power-law parameters.

    Reads one fit row containing the power-law exponent and amplitude together
    with their uncertainty bounds, samples random perturbations within those
    bounds, and evaluates the power-law at the exposure time stored in *row*.

    Parameters
    ----------
    row : pandas.Series
        Metadata row; must contain the key *exp_time_key*.
    fit_data : pandas.DataFrame
        Single-row DataFrame with power-law fit results.
    t_key : str
        Column name for the power-law exponent in *fit_data*.
    a_key : str
        Column name for the power-law amplitude in *fit_data*.
    dt_key : str
        Column name for the exponent uncertainty in *fit_data*.
    da_key : str
        Column name for the amplitude uncertainty in *fit_data*.
    exp_time_key : str
        Column name for the exposure time in *row*.

    Returns
    -------
    float
        Estimated noise sigma from the perturbed power-law.
    """
    t = fit_data[t_key].values[0]
    a = fit_data[a_key].values[0]
    dt = np.random.uniform(-fit_data[dt_key].values[0], fit_data[dt_key].values[0])
    da = np.random.uniform(-fit_data[da_key].values[0], fit_data[da_key].values[0])
    t += dt
    a += da
    simulated_sigma = power_law(_row_get(row, exp_time_key), t, a)
    return simulated_sigma

def _sigma_kernel_from_row_wrapper(row, fit_data, sigma_key):
    """Forward row-based sigma selection to :func:`sigma_kernel_from_row`.

    Parameters
    ----------
    row : pandas.Series
        Metadata row containing the sigma field.
    fit_data : object
        Unused; kept for a consistent wrapper signature.
    sigma_key : str
        Column name in *row* holding the precomputed sigma value.

    Returns
    -------
    float
        Precomputed sigma value from *row[sigma_key]*.
    """
    return sigma_kernel_from_row(row, sigma_key)
def sigma_kernel_from_row(row, sigma_key):
    """Return the precomputed sigma value from a metadata row.

    Parameters
    ----------
    row : pandas.Series
        Metadata row.
    sigma_key : str
        Column name holding the precomputed sigma value.

    Returns
    -------
    float
        Value of ``row[sigma_key]``.
    """
    return _row_get(row, sigma_key)

def create_ratios(initial_ratio, ratio_count, growth_factor):
    """
    Build exposure ratios using ceil-rounded geometric progression.

    Parameters
    ----------
    initial_ratio : int or float
        First ratio value.
    ratio_count : int
        Number of ratios to generate.
    growth_factor : float
        Multiplicative growth factor per step.

    Returns
    -------
    list of int or float
        Sequence like [r0, ceil(r0*g), ceil(ceil(r0*g)*g), ...].
    """
    try:
        current_ratio = float(initial_ratio)
    except (TypeError, ValueError) as err:
        raise ValueError(f'initial_ratio must be numeric, got {initial_ratio!r}') from err

    ratio_count_int = parse_required_int(ratio_count, 'ratio_count')

    try:
        growth_factor_float = float(growth_factor)
    except (TypeError, ValueError) as err:
        raise ValueError(f'growth_factor must be numeric, got {growth_factor!r}') from err

    if ratio_count_int <= 0:
        return []

    ratio_values = []
    for _ in range(ratio_count_int):
        ratio_values.append(current_ratio)
        current_ratio = float(np.ceil(current_ratio * growth_factor_float))
    return ratio_values

def create_simulated_image_poisson(image, original_exposure, exposure_ratio):
    """
    Simulate a new exposure realization via Poisson photon statistics.

    The input is interpreted as count-rate data, clipped at 1e-12, and
    scaled by the simulated exposure `original_exposure / exposure_ratio` to
    obtain expected counts. A Poisson sample is drawn in count space and then
    converted back to count-rate units by dividing by the simulated exposure.
    This mirrors exposure-time changes while preventing non-positive rates.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in count-rate units.
    original_exposure : float
        Exposure of the input image.
    exposure_ratio : float
        Ratio original_exposure/simulated_exposure.

    Returns
    -------
    numpy.ndarray
        Simulated noisy image.
    """
    if exposure_ratio <= 0.0 or original_exposure <= 0.0:
        return image 
    simulated_exposure = original_exposure / exposure_ratio
    exposure_scaled_signal = image * simulated_exposure
    poisson_sample = np.random.poisson(exposure_scaled_signal) / simulated_exposure
    return poisson_sample

def create_simulated_image_gaussian(image, median_background, new_sigma):
    """
    Simulate a noisy image by injecting Gaussian background fluctuations.

    The signal component is formed as `image - median_background`, then a
    Gaussian background with mean `median_background` and standard deviation
    `new_sigma` is sampled per pixel and added back. This keeps the global
    background level while controlling noise width through `new_sigma`, which is
    useful for exposure-driven augmentation paths.

    Parameters
    ----------
    image : numpy.ndarray
        Input image values.
    median_background : float
        Background median level.
    new_sigma : float
        Target Gaussian noise standard deviation.

    Returns
    -------
    numpy.ndarray
        Simulated noisy image.
    """
    gaussian_noise = np.random.normal(
        loc=median_background,
        scale=new_sigma,
        size=image.shape,
    )
    simulated_image = image + gaussian_noise 
    return simulated_image

def _simulated_image_from_exposure(file, row, new_sigma):
    """Gaussian noise simulation using the background median from *row*.

    Resolver-compatible wrapper around
    :func:`create_simulated_image_gaussian` that extracts
    ``row['full_median_bkg']`` as the background level.

    Parameters
    ----------
    file : numpy.ndarray
        Input image array.
    row : pandas.Series
        Metadata row; must contain ``'full_median_bkg'``.
    new_sigma : float
        Target Gaussian noise standard deviation.

    Returns
    -------
    numpy.ndarray
        Simulated noisy image.
    """
    return create_simulated_image_gaussian(file, _row_get(row, 'full_median_bkg'), new_sigma)
    
def _simulated_image_from_poisson(file, row, new_sigma):
    """Poisson noise simulation using exposure time and ratio from *row*.

    Resolver-compatible wrapper around
    :func:`create_simulated_image_poisson` that extracts
    ``row['exp_time']`` and ``row['exp_ratio']``.

    Parameters
    ----------
    file : numpy.ndarray
        Input image array in count-rate units.
    row : pandas.Series
        Metadata row; must contain ``'exp_time'`` and ``'exp_ratio'``.
    new_sigma : float
        Unused; kept for a consistent wrapper signature.

    Returns
    -------
    numpy.ndarray
        Simulated noisy image.
    """
    return create_simulated_image_poisson(file, _row_get(row, 'exp_time'), _row_get(row, 'exp_ratio'))

############ FILENAME MODIFICATIONS ########################################

def _stats_name_from_sigma(filename, row, suffix_value):
    """Build a FITS stats filename using the sigma suffix value.

    Parameters
    ----------
    filename : str
        Original FITS file path.
    row : pandas.Series
        Metadata row (unused; kept for consistent wrapper signature).
    suffix_value : float
        Sigma value appended to the base name (dots replaced with
        underscores).

    Returns
    -------
    str
        Stats filename of the form
        ``<basename>_<sigma_formatted>.fits``.
    """
    return os.path.basename(filename).split('.fits')[0] + f"_{suffix_value:.5f}".replace('.', '_') + ".fits"

def _stats_name_from_exposure(filename, row, suffix_value):
    """Build a FITS stats filename from original and simulated exposure times.

    Parameters
    ----------
    filename : str
        Original FITS file path.
    row : pandas.Series
        Metadata row; must contain ``'exp_time'`` and ``'new_exp_time'``.
    suffix_value : float
        Unused; kept for a consistent wrapper signature.

    Returns
    -------
    str
        Stats filename of the form
        ``<basename>_<exp_time>__<new_exp_time>.fits``.
    """
    return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits"

def _stats_name_from_exp_ratio(filename, row, suffix_value):
    """Build a FITS stats filename from the exposure ratio.

    Parameters
    ----------
    filename : str
        Original FITS file path.
    row : pandas.Series
        Metadata row; must contain ``'exp_ratio'``.
    suffix_value : float
        Unused; kept for a consistent wrapper signature.

    Returns
    -------
    str
        Stats filename of the form
        ``<basename>_<exp_ratio>.fits``.
    """
    return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits"

############ NORMALIZATION HELPERS ###########################################


def min_max_normalization(data):
    """
    Apply min-max normalization to [0, 1].

    Parameters
    ----------
    data : numpy.ndarray
        Input array.

    Returns
    -------
    tuple or None
        (normalized_data, min_value, max_value), or None for zero or near-zero range.
    """
    min_value = np.min(data)
    max_value = np.max(data)
    value_range = max_value - min_value
    if not np.isfinite(min_value) or not np.isfinite(max_value) or value_range <= 1e-6:
        return None

    normalized_data = (data - min_value) / value_range
    return normalized_data, min_value, max_value


def inverse_min_max_normalization(normalized_data, min_value, max_value):
    """
    Reconstruct values from min-max normalized data.

    Parameters
    ----------
    normalized_data : numpy.ndarray
        Normalized input values.
    min_value : float
        Original minimum.
    max_value : float
        Original maximum.

    Returns
    -------
    numpy.ndarray or None
        Denormalized data, or None when bounds are invalid.
    """
    if not np.isfinite(min_value) or not np.isfinite(max_value) or (max_value - min_value) <= 1e-6:
        return None
    return (normalized_data * (max_value - min_value)) + min_value


def zscore_normalization(data):
    """
    Apply z-score normalization.

    Parameters
    ----------
    data : numpy.ndarray
        Input array.

    Returns
    -------
    tuple or None
        (normalized_data, mean_value, std_value), or None for zero or near-zero std.
    """
    mean_value = np.mean(data)
    std_value = np.std(data)
    if not np.isfinite(std_value) or std_value <= 1e-6:
        return None

    normalized_data = (data - mean_value) / std_value
    return normalized_data, mean_value, std_value


def inverse_zscore_normalization(normalized_data, mean_value, std_value):
    """
    Reconstruct values from z-score normalized data.

    Parameters
    ----------
    normalized_data : numpy.ndarray
        Normalized input values.
    mean_value : float
        Original mean.
    std_value : float
        Original standard deviation.

    Returns
    -------
    numpy.ndarray or None
        Denormalized data, or None when std_value is zero, near-zero, or invalid.
    """
    if not np.isfinite(mean_value) or not np.isfinite(std_value) or std_value <= 1e-6:
        return None
    return (normalized_data * std_value) + mean_value


def adaptive_log_transform_and_normalize(data):
    """
    Apply adaptive log transform followed by min-max normalization.

    Parameters
    ----------
    data : numpy.ndarray
        Input array.

    Returns
    -------
    tuple
        (normalized_data, min_value, max_value, shift_value), or (None, None, None, None).
    """
    if not np.all(np.isfinite(data)):
        return None, None, None, None

    min_data_value = np.min(data)
    # Shift keeps the log argument strictly positive.
    shift_value = -min_data_value + 1 if min_data_value < 0 else 1

    log_data = np.log(data + shift_value)
    min_log_value = np.min(log_data)
    max_log_value = np.max(log_data)
    if not np.isfinite(min_log_value) or not np.isfinite(max_log_value):
        return None, None, None, None
    if (max_log_value - min_log_value) <= 1e-6:
        return None, None, None, None

    normalized_data = (log_data - min_log_value) / (max_log_value - min_log_value)
    return normalized_data.astype(np.float32), min_log_value, max_log_value, shift_value


def inverse_adaptive_log_transform_and_denormalize(
    predicted_data,
    original_min_value,
    original_max_value,
    shift_value,
):
    """
    Undo adaptive log transform plus min-max normalization.

    Parameters
    ----------
    predicted_data : numpy.ndarray
        Normalized log-domain values.
    original_min_value : float
        Min value used in log-domain min-max normalization.
    original_max_value : float
        Max value used in log-domain min-max normalization.
    shift_value : float
        Positive shift used before log transform.

    Returns
    -------
    numpy.ndarray or None
        Reconstructed linear-domain data, or None when bounds are missing.
    """
    if original_min_value is None or original_max_value is None:
        return None
    if not np.isfinite(original_min_value) or not np.isfinite(original_max_value) or not np.isfinite(shift_value):
        return None

    log_domain_data = (
        predicted_data * (original_max_value - original_min_value) + original_min_value
    )
    log_domain_data = np.clip(log_domain_data, a_min=None, a_max=LOG_DOMAIN_CLIP_MAX)
    linear_domain_data = np.exp(log_domain_data) - shift_value
    return linear_domain_data.astype(np.float32)


############ SCALING PIPELINE ################################################


def apply_scaling_and_stats(noisy_image, clean_image, scaling, output_name):
    """
    Apply configured scaling to noisy/clean images and package inverse metadata.

    The same scaling function is applied independently to noisy and clean images
    (`log_min_max`, `z_scale`, or `min_max`) to preserve per-image statistics.
    When scaling succeeds, the function serializes clean stats first, then noisy
    stats, after `output_name`; this ordering matches the callback-side inverse
    decoding logic used during evaluation/visualization. Returning `None` signals
    invalid scaling output (for example zero-range or missing stats).

    Parameters
    ----------
    noisy_image : numpy.ndarray
        Noisy input image.
    clean_image : numpy.ndarray
        Clean target image.
    scaling : str or None
        Scaling mode: log_min_max, z_scale, min_max, or None.
    output_name : str
        Identifier prepended to returned metadata.

    Returns
    -------
    tuple or None
        (scaled_noisy, scaled_clean, stats_list), or None on invalid scaling output.
    """
    stats_values = [output_name]
    scaling_function_map = {
        'log_min_max': adaptive_log_transform_and_normalize,
        'z_scale': zscore_normalization,
        'min_max': min_max_normalization,
    }
    scaling_function = scaling_function_map[scaling] if scaling in scaling_function_map else None

    if scaling_function is None:
        return noisy_image, clean_image, stats_values

    noisy_scaling_result = scaling_function(noisy_image)
    clean_scaling_result = scaling_function(clean_image)
    if noisy_scaling_result is None or clean_scaling_result is None:
        return None

    if any(value is None for value in noisy_scaling_result[1:]):
        return None
    if any(value is None for value in clean_scaling_result[1:]):
        return None

    scaled_noisy_image, *noisy_stats = noisy_scaling_result
    scaled_clean_image, *clean_stats = clean_scaling_result
    if not np.all(np.isfinite(scaled_noisy_image)) or not np.all(np.isfinite(scaled_clean_image)):
        return None
    if any(not np.isfinite(value) for value in clean_stats):
        return None
    if any(not np.isfinite(value) for value in noisy_stats):
        return None

    # Keep metadata order compatible with the callback inverse decoding logic.
    stats_values.extend(str(value) for value in clean_stats)
    stats_values.extend(str(value) for value in noisy_stats)
    return scaled_noisy_image, scaled_clean_image, stats_values
