from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from src.training.math_helpers import (
    adaptive_log_transform_and_normalize,
    evenly_spaced_numbers,
    find_distribution,
    find_distribution_only_exp,
    inverse_adaptive_log_transform_and_denormalize,
    inverse_min_max_normalization,
    inverse_zscore_normalization,
    linear_function,
    min_max_normalization,
    power_law,
    zscore_normalization,
)
from src.training.utils import (
    CHECKPOINT_INFO_FILENAME,
    _normalize_runtime_filepath,
    build_checkpoint_filename,
    ensure_directory_exists,
    is_not_nan,
    open_fits,
    read_checkpoint_info,
)


@pytest.mark.unit
def test_inverse_scalers_roundtrip_random_arrays():
    rng = np.random.default_rng(20260515)
    data = rng.normal(loc=3.0, scale=7.5, size=(32, 32)).astype(np.float32)
    data += rng.uniform(-15.0, 15.0, size=(32, 32)).astype(np.float32)

    normalized_minmax, min_value, max_value = min_max_normalization(data)
    restored_minmax = inverse_min_max_normalization(normalized_minmax, min_value, max_value)
    normalized_z, mean_value, std_value = zscore_normalization(data)
    restored_z = inverse_zscore_normalization(normalized_z, mean_value, std_value)
    log_normalized, log_min, log_max, shift_value = adaptive_log_transform_and_normalize(data)
    restored_log = inverse_adaptive_log_transform_and_denormalize(log_normalized, log_min, log_max, shift_value)

    assert np.allclose(restored_minmax, data, atol=5e-6)
    assert np.allclose(restored_z, data, atol=5e-6)
    assert np.allclose(restored_log, data, atol=5e-5)


@pytest.mark.unit
def test_inverse_scalers_reject_invalid_stats():
    sample = np.array([[0.25, 0.75]], dtype=np.float32)

    assert inverse_min_max_normalization(sample, 1.0, 1.0) is None
    assert inverse_zscore_normalization(sample, 0.0, 0.0) is None
    assert inverse_adaptive_log_transform_and_denormalize(sample, None, 1.0, 1.0) is None
    assert inverse_adaptive_log_transform_and_denormalize(sample, 0.0, 1.0, np.nan) is None


@pytest.mark.unit
def test_numeric_helpers_and_spacing_behave_consistently_with_random_inputs():
    rng = np.random.default_rng(314159)
    x = rng.uniform(0.5, 5.0, size=20)
    slope = rng.normal()
    intercept = rng.normal()
    exponent = rng.uniform(-2.0, 2.0)
    amplitude = rng.uniform(0.1, 3.0)

    linear_values = linear_function(x, slope, intercept)
    power_values = power_law(x, exponent, amplitude)
    spaced = evenly_spaced_numbers(2, 17, 5)

    assert np.allclose(linear_values, slope * x + intercept)
    assert np.allclose(power_values, amplitude * np.power(x, exponent))
    assert spaced == sorted(set(spaced))
    assert len(spaced) <= 5
    assert all(2 < value <= 17 for value in spaced)
    assert evenly_spaced_numbers(5, 5, 3) == []
    assert evenly_spaced_numbers(5, 4, 3) == []


@pytest.mark.unit
def test_distribution_helpers_produce_expected_bin_metadata_for_random_log_data():
    rng = np.random.default_rng(271828)
    magnitudes = np.power(10.0, rng.uniform(-2.0, 3.0, size=400))
    signs = rng.choice([-1.0, 1.0], size=400)
    sigma_values = magnitudes * signs
    df = pd.DataFrame({'sigma': sigma_values})

    per_base = find_distribution(df, 'sigma')
    per_exp = find_distribution_only_exp(df, 'sigma')

    assert set(per_base.columns) == {'base', 'exponent', 'mean', 'std'}
    assert set(per_exp.columns) == {'exponent', 'mean', 'std'}
    assert per_base['base'].between(1, 9).all()
    assert per_base['exponent'].min() == -2
    assert per_base['exponent'].max() == 2
    assert per_exp['exponent'].tolist() == [-2, -1, 0, 1, 2]

    populated_base = per_base.dropna(subset=['mean', 'std'])
    populated_exp = per_exp.dropna(subset=['mean', 'std'])
    assert not populated_base.empty
    assert not populated_exp.empty
    assert (populated_base['std'] >= 0).all()
    assert (populated_exp['std'] >= 0).all()

    for _, row in populated_base.iterrows():
        lower = row['base'] * 10.0 ** row['exponent']
        upper = (row['base'] + 1) * 10.0 ** row['exponent']
        assert lower <= row['mean'] < upper


@pytest.mark.unit
def test_normalize_runtime_filepath_covers_windows_and_slash_cases():
    windows_path = r'D:\astro\data\example.fits'
    mixed_slashes = r'relative\folder\example.fits'

    normalized_windows = _normalize_runtime_filepath(windows_path)
    normalized_mixed = _normalize_runtime_filepath(mixed_slashes)

    if os.name == 'nt':
        assert normalized_windows == windows_path
    else:
        assert normalized_windows == '/mnt/d/astro/data/example.fits'
    assert normalized_mixed == ('relative/folder/example.fits' if os.name != 'nt' else mixed_slashes)
    assert _normalize_runtime_filepath('') == ''
    assert _normalize_runtime_filepath(None) is None


@pytest.mark.unit
def test_ensure_directory_exists_and_is_not_nan_handle_common_edge_cases(tmp_path: Path):
    nested_dir = tmp_path / 'a' / 'b' / 'c'

    returned = ensure_directory_exists(str(nested_dir))

    assert returned == str(nested_dir)
    assert nested_dir.is_dir()
    assert ensure_directory_exists(None) is None
    assert is_not_nan(1.25)
    assert is_not_nan('2.5')
    assert not is_not_nan(np.nan)
    assert not is_not_nan(float('inf'))
    assert not is_not_nan('not-a-number')


@pytest.mark.unit
def test_open_fits_handles_ratio_bounds_and_invalid_metadata(tmp_path: Path):
    rng = np.random.default_rng(424242)
    image = rng.normal(size=(8, 8)).astype(np.float32)
    valid_path = tmp_path / 'valid.fits'
    missing_exptime_path = tmp_path / 'missing_exptime.fits'

    sci = fits.ImageHDU(data=image, name='SCI')
    sci.header['EXPTIME'] = 120.0
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(valid_path, overwrite=True)

    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=image, name='SCI')]).writeto(missing_exptime_path, overwrite=True)

    raw = open_fits(str(valid_path), type_of_image='SCI')
    ratio_payload = open_fits(str(valid_path), ratio=2.0, type_of_image='SCI', low=200.0, high=300.0)

    assert np.allclose(raw, image)
    assert ratio_payload is not None
    loaded_image, exposure_time, ratio_value = ratio_payload
    assert np.allclose(loaded_image, image)
    assert exposure_time == pytest.approx(120.0)
    assert ratio_value == pytest.approx(2.0)
    assert open_fits(str(valid_path), ratio=2.0, type_of_image='SCI', low=300.1) is None
    assert open_fits(str(valid_path), ratio='bad', type_of_image='SCI') is None
    assert open_fits(str(missing_exptime_path), ratio=2.0, type_of_image='SCI') is None
    assert open_fits(str(valid_path), type_of_image='DOES_NOT_EXIST') is None


@pytest.mark.unit
def test_checkpoint_filename_and_info_helpers_validate_edge_cases(tmp_path: Path):
    assert build_checkpoint_filename('model', 7, '{prefix}_epoch{epoch:03d}.keras') == 'model_epoch007.keras'

    with pytest.raises(TypeError):
        build_checkpoint_filename('model', 1, None)
    with pytest.raises(ValueError):
        build_checkpoint_filename('model', 1, '{prefix}.keras')
    with pytest.raises(ValueError):
        build_checkpoint_filename('model', 1, '{epoch}.keras')

    plain_file = tmp_path / 'plain.txt'
    plain_file.write_text('not a zip archive', encoding='utf-8')
    assert read_checkpoint_info(str(plain_file)) == {}
    assert read_checkpoint_info(str(tmp_path / 'missing.keras')) == {}

    no_info_zip = tmp_path / 'no_info.keras'
    with zipfile.ZipFile(no_info_zip, 'w') as archive:
        archive.writestr('weights.txt', 'content')
    assert read_checkpoint_info(str(no_info_zip)) == {}

    bad_info_zip = tmp_path / 'bad_info.keras'
    with zipfile.ZipFile(bad_info_zip, 'w') as archive:
        archive.writestr(CHECKPOINT_INFO_FILENAME, '[1, 2, 3]')
    assert read_checkpoint_info(str(bad_info_zip)) == {}
