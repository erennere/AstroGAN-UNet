"""Tests for src/visualization/prepare_plots.py pure helpers and data-manipulation functions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.prepare_plots import (
    add_columns,
    add_summary_metrics,
    apply_pre_window,
    build_mask,
    build_norm,
    build_panel_norm,
    calculate_abmag,
    clip_aligned_paired_series,
    clip_separate_paired_series,
    clip_single_series,
    convert_to_angstrom,
    convert_to_jansky,
    create_table,
    finite_mask,
    get_exp_ratio_subset,
    get_gridsize,
    get_legend_value,
    positive_mask,
    prepare_paired_panel,
    prepare_single_panel,
    safe_divide,
    transform_limits,
    transform_series,
)


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_safe_divide_scalar_zero_denominator():
    assert np.isnan(safe_divide(5.0, 0))
    assert safe_divide(10.0, 2) == pytest.approx(5.0)


@pytest.mark.unit
def test_safe_divide_series_zero_denominator():
    num = pd.Series([4.0, 6.0, 8.0])
    den = pd.Series([2.0, 0.0, 4.0])
    result = safe_divide(num, den)
    assert result.iloc[0] == pytest.approx(2.0)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)


@pytest.mark.unit
def test_safe_divide_array_zero_denominator():
    rng = np.random.default_rng(42)
    num = rng.uniform(1.0, 10.0, size=20)
    den = np.where(np.arange(20) % 4 == 0, 0.0, rng.uniform(1.0, 3.0, size=20))
    result = safe_divide(num, den)
    assert result.shape == (20,)
    assert np.all(np.isnan(result[np.arange(20) % 4 == 0]))
    assert np.all(np.isfinite(result[np.arange(20) % 4 != 0]))


# ---------------------------------------------------------------------------
# finite_mask / positive_mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_finite_mask_excludes_nan_and_inf():
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        'a': rng.uniform(0, 1, 10).tolist(),
        'b': rng.uniform(0, 1, 10).tolist(),
    })
    df.loc[0, 'a'] = np.nan
    df.loc[3, 'b'] = np.inf
    df.loc[7, 'b'] = -np.inf

    mask = finite_mask(df, 'a', 'b')
    assert not mask.iloc[0]
    assert not mask.iloc[3]
    assert not mask.iloc[7]
    assert mask.sum() == 7


@pytest.mark.unit
def test_positive_mask_requires_positive_and_finite():
    df = pd.DataFrame({'x': [1.0, 0.0, -1.0, np.nan, 2.5]})
    mask = positive_mask(df, 'x')
    assert mask.tolist() == [True, False, False, False, True]


# ---------------------------------------------------------------------------
# add_summary_metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_add_summary_metrics_computes_precision_recall_f_measure():
    rng = np.random.default_rng(111)
    n = 20
    df = pd.DataFrame({
        'TP': rng.integers(10, 50, size=n).astype(float),
        'FP': rng.integers(0, 20, size=n).astype(float),
        'FN': rng.integers(0, 20, size=n).astype(float),
        'IOU_sum': rng.uniform(0, 1, n),
        'union': rng.integers(20, 100, size=n).astype(float),
        'SNR_org_sum': rng.uniform(10, 50, n),
        'SNR_rec_sum': rng.uniform(10, 50, n),
        'RFE_sum': rng.uniform(0, 5, n),
    })
    result = add_summary_metrics(df)

    assert 'Precision' in result.columns
    assert 'Recall' in result.columns
    assert 'F-measure' in result.columns
    assert 'IoU' in result.columns

    precision = result['TP'] / (result['TP'] + result['FP'])
    recall = result['TP'] / (result['TP'] + result['FN'])
    assert np.allclose(result['Precision'], precision, equal_nan=True)
    assert np.allclose(result['Recall'], recall, equal_nan=True)


@pytest.mark.unit
def test_add_summary_metrics_handles_zero_denominators():
    df = pd.DataFrame({'TP': [0.0], 'FP': [0.0], 'FN': [0.0],
                        'IOU_sum': [0.0], 'union': [0.0],
                        'SNR_org_sum': [0.0], 'SNR_rec_sum': [0.0], 'RFE_sum': [0.0]})
    result = add_summary_metrics(df)
    assert np.isnan(result['Precision'].iloc[0]) or result['Precision'].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# convert_to_jansky / convert_to_angstrom / calculate_abmag
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_convert_to_jansky_positive_flux():
    rng = np.random.default_rng(777)
    flux = rng.uniform(0.1, 100.0, 50)
    result = convert_to_jansky(flux)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)


@pytest.mark.unit
def test_convert_to_angstrom_scales_by_photflam():
    flux = np.array([1.0, 2.0, 10.0])
    PHOTFLAM = 1.92756031304868e-20
    result = convert_to_angstrom(flux)
    assert np.allclose(result, PHOTFLAM * flux)


@pytest.mark.unit
def test_calculate_abmag_correct_formula_and_edge_cases():
    # Known value: abmag = -2.5*log10(1e-26) + 8.9 ≈ some finite value
    flux_jansky = 1e-3  # 1 mJy
    abmag = calculate_abmag(flux_jansky)
    assert np.isfinite(abmag)
    expected = -2.5 * np.log10(flux_jansky) + 8.9
    assert abmag == pytest.approx(expected)

    assert np.isnan(calculate_abmag(0.0))
    assert np.isnan(calculate_abmag(-1.0))
    assert np.isnan(calculate_abmag(np.nan))


# ---------------------------------------------------------------------------
# get_exp_ratio_subset / get_gridsize
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_exp_ratio_subset_filters_and_returns_total():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({'exp_ratio': [2, 2, 3, 3, 5], 'val': rng.uniform(size=5)})

    subset, label = get_exp_ratio_subset(df, 2, 'all')
    assert len(subset) == 2
    assert '2' in label or '2' in str(label)

    total, total_label = get_exp_ratio_subset(df, 0, 'all')
    assert len(total) == len(df)
    assert total_label == 'all'


@pytest.mark.unit
def test_get_gridsize_min_100_and_scales_with_count():
    # min is 100; int(sqrt(1/10))=0, max(100,0)=100
    assert get_gridsize([1]) == 100
    # still 100 for small lists
    assert get_gridsize(list(range(100))) >= 100
    # for a very large list (200 000 items) sqrt(200000/10)=~141 > 100
    large_result = get_gridsize(list(range(200_000)))
    assert large_result > 100


# ---------------------------------------------------------------------------
# build_norm
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_norm_returns_normalize_with_finite_bounds():
    from matplotlib.colors import Normalize
    rng = np.random.default_rng(31)
    values = pd.Series(rng.uniform(1.0, 100.0, 200))
    norm = build_norm(values, (5, 95))
    assert isinstance(norm, Normalize)
    assert np.isfinite(norm.vmin)
    assert np.isfinite(norm.vmax)
    assert norm.vmin < norm.vmax


@pytest.mark.unit
def test_build_norm_raises_on_empty_series():
    with pytest.raises(ValueError):
        build_norm(pd.Series([], dtype=float), (5, 95))


@pytest.mark.unit
def test_build_norm_raises_when_bounds_collapse():
    with pytest.raises(ValueError):
        build_norm(pd.Series([1.0] * 50), (5, 95))


# ---------------------------------------------------------------------------
# clip_single_series
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clip_single_series_reduces_outliers():
    rng = np.random.default_rng(55)
    x = pd.Series(rng.normal(0, 1, 500))
    y = pd.Series(rng.normal(0, 1, 500))
    result = clip_single_series(x, y)
    assert result is not None
    x_out, y_out, (xl, xh), (yl, yh) = result
    assert len(x_out) < len(x)
    assert xl <= x_out.min()
    assert xh >= x_out.max()


@pytest.mark.unit
def test_clip_single_series_returns_none_for_empty():
    result = clip_single_series(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert result is None


# ---------------------------------------------------------------------------
# clip_aligned_paired_series / clip_separate_paired_series
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clip_aligned_paired_series_returns_shared_y_window():
    rng = np.random.default_rng(88)
    x_rec = pd.Series(rng.normal(0, 1, 400))
    y_rec = pd.Series(rng.normal(5, 2, 400))
    x_noise = pd.Series(rng.normal(0, 1, 400))
    y_noise = pd.Series(rng.normal(5, 2, 400))
    result = clip_aligned_paired_series(x_rec, y_rec, x_noise, y_noise)
    assert result is not None
    xr, yr, xn, yn, (yl, yh) = result
    assert len(xr) == len(yr) == len(xn) == len(yn)
    assert yr.min() >= yl
    assert yn.max() <= yh


@pytest.mark.unit
def test_clip_separate_paired_series_returns_independent_windows():
    rng = np.random.default_rng(77)
    x_rec = pd.Series(rng.normal(0, 1, 300))
    y_rec = pd.Series(rng.normal(5, 2, 300))
    x_noise = pd.Series(rng.normal(10, 1, 300))
    y_noise = pd.Series(rng.normal(2, 0.5, 300))
    result = clip_separate_paired_series(x_rec, y_rec, x_noise, y_noise)
    assert result is not None
    xr, yr, xn, yn, _ = result
    assert len(xr) > 0 and len(xn) > 0


# ---------------------------------------------------------------------------
# transform_series
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transform_series_identity_log10_ratio_delta():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({'flux': rng.uniform(1.0, 100.0, 50), 'ref': rng.uniform(1.0, 10.0, 50)})
    cols = {'flux': 'flux', 'ref': 'ref'}

    identity = transform_series(df, cols, {'key': 'flux', 'transform': 'identity'})
    log10 = transform_series(df, cols, {'key': 'flux', 'transform': 'log10'})
    ratio = transform_series(df, cols, {'key': 'flux', 'transform': 'ratio', 'reference': 'ref'})
    delta = transform_series(df, cols, {'key': 'flux', 'transform': 'delta', 'reference': 'ref'})

    assert np.allclose(identity.values, df['flux'].values)
    assert np.allclose(log10.values, np.log10(df['flux'].values))
    assert np.allclose(ratio.values, df['flux'].values / df['ref'].values)
    assert np.allclose(delta.values, df['flux'].values - df['ref'].values)


@pytest.mark.unit
def test_transform_series_raises_on_unsupported_transform():
    df = pd.DataFrame({'x': [1.0]})
    with pytest.raises(ValueError, match='Unsupported transform'):
        transform_series(df, {'x': 'x'}, {'key': 'x', 'transform': 'invalid'})


# ---------------------------------------------------------------------------
# transform_limits
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transform_limits_identity_and_log10_and_none():
    assert transform_limits(None, {'transform': 'identity'}) is None
    assert transform_limits((1.0, 100.0), {'transform': 'identity'}) == (1.0, 100.0)
    result = transform_limits((1.0, 1000.0), {'transform': 'log10'})
    assert result == pytest.approx((0.0, 3.0))


@pytest.mark.unit
def test_transform_limits_raises_on_unsupported():
    with pytest.raises(ValueError):
        transform_limits((1.0, 10.0), {'transform': 'bad'})


# ---------------------------------------------------------------------------
# get_legend_value
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_legend_value_strategies():
    rng = np.random.default_rng(17)
    values = pd.Series(rng.uniform(1.0, 100.0, 200))
    assert get_legend_value(values, 'median') == pytest.approx(np.median(values))
    assert get_legend_value(values, 'min') == pytest.approx(values.min())
    assert get_legend_value(values, 'unknown') is None


# ---------------------------------------------------------------------------
# build_mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_mask_positive_finite_and_missing_rules():
    df = pd.DataFrame({
        'flux': [1.0, -1.0, 2.0, np.nan, 3.0],
        'kron': [0.5, 0.5, np.inf, 0.5, 0.5],
        'noise': [1.0, np.nan, 1.0, 1.0, np.nan],
    })
    columns = {'flux': 'flux', 'kron': 'kron', 'noise': 'noise'}

    mask_none = build_mask(df, columns, None)
    assert mask_none.all()

    mask_pos = build_mask(df, columns, {'positive': ['flux']})
    assert mask_pos.tolist() == [True, False, True, False, True]

    mask_fin = build_mask(df, columns, {'finite': ['kron']})
    assert not mask_fin.iloc[2]

    mask_miss = build_mask(df, columns, {'missing': ['noise']})
    assert mask_miss.tolist() == [False, True, False, False, True]


# ---------------------------------------------------------------------------
# apply_pre_window
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_apply_pre_window_trims_to_percentile_window():
    rng = np.random.default_rng(61)
    df = pd.DataFrame({'x': rng.normal(0, 1, 200)})
    columns = {'x': 'x'}

    trimmed, limits = apply_pre_window(df, columns, {'key': 'x', 'positive': False, 'percentiles': (5, 95)})
    assert trimmed is not None
    assert len(trimmed) < len(df)
    assert limits is not None
    assert limits[0] <= limits[1]


@pytest.mark.unit
def test_apply_pre_window_returns_none_when_no_finite_values():
    df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
    columns = {'x': 'x'}
    result, _ = apply_pre_window(df, columns, {'key': 'x', 'positive': False, 'percentiles': (5, 95)})
    assert result is None


# ---------------------------------------------------------------------------
# build_panel_norm
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_panel_norm_column_and_dynamic_and_none_modes():
    rng = np.random.default_rng(77)
    df = pd.DataFrame({'val': rng.uniform(1.0, 100.0, 200)})
    columns = {'val': 'val'}
    x = pd.Series(rng.uniform(1.0, 10.0, 200))
    y = pd.Series(rng.uniform(1.0, 10.0, 200))

    norm_col = build_panel_norm(df, columns, {'norm_mode': 'column', 'norm_key': 'val'}, (5, 95))
    assert norm_col is not None

    norm_dyn = build_panel_norm(df, columns, {'norm_mode': 'dynamic'}, (5, 95), x_values=x, y_values=y)
    assert norm_dyn is not None

    norm_none = build_panel_norm(df, columns, {}, (5, 95))
    assert norm_none is None


# ---------------------------------------------------------------------------
# prepare_single_panel
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prepare_single_panel_returns_none_when_no_data():
    df = pd.DataFrame({'x': [np.nan] * 20, 'y': [np.nan] * 20})
    columns = {'x': 'x', 'y': 'y'}
    spec = {
        'mask': {'finite': ['x', 'y']},
        'x': {'key': 'x'},
        'y': {'key': 'y'},
        'norm_mode': None,
    }
    result = prepare_single_panel(df, spec, columns, (5, 95))
    assert result is None


@pytest.mark.unit
def test_prepare_single_panel_returns_data_dict_with_valid_input():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'x': rng.uniform(1.0, 10.0, 300), 'y': rng.uniform(1.0, 5.0, 300)})
    columns = {'x': 'x', 'y': 'y'}
    spec = {
        'mask': {'finite': ['x', 'y']},
        'x': {'key': 'x'},
        'y': {'key': 'y'},
        'norm_mode': None,
    }
    result = prepare_single_panel(df, spec, columns, (5, 95))
    assert result is not None
    assert 'x' in result and 'y' in result
    assert len(result['x']) > 0


# ---------------------------------------------------------------------------
# prepare_paired_panel
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prepare_paired_panel_returns_rec_noise_arrays():
    rng = np.random.default_rng(123)
    n = 300
    df = pd.DataFrame({
        'x_rec': rng.uniform(1, 10, n),
        'y_rec': rng.uniform(1, 5, n),
        'x_noise': rng.uniform(1, 10, n),
        'y_noise': rng.uniform(1, 5, n),
    })
    columns = {'x_rec': 'x_rec', 'y_rec': 'y_rec', 'x_noise': 'x_noise', 'y_noise': 'y_noise'}
    spec = {
        'pre_window': None,
        'mask_mode': 'shared',
        'mask': {'finite': ['x_rec', 'y_rec']},
        'rec': {'x': {'key': 'x_rec'}, 'y': {'key': 'y_rec'}},
        'noise': {'x': {'key': 'x_noise'}, 'y': {'key': 'y_noise'}},
        'clip_mode': 'aligned',
    }
    result = prepare_paired_panel(df, spec, columns)
    assert result is not None
    assert 'x_rec' in result and 'x_noise' in result
    assert len(result['x_rec']) > 0


# ---------------------------------------------------------------------------
# create_table
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_table_aggregates_per_ratio_and_overall(tmp_path: Path):
    rng = np.random.default_rng(2024)

    # Create a synthetic metadata CSV with eval/test location entries
    metadata = pd.DataFrame({
        'location': [f'/data/eval/img{i}.fits' for i in range(5)],
        'sci_data_set_name': ['ds1', 'ds2', 'ds3', 'ds4', 'ds5'],
    })
    metadata_csv = tmp_path / 'metadata.csv'
    metadata.to_csv(metadata_csv, index=False)

    # Create results CSV with required columns
    n = 60
    exp_times = rng.choice([60.0, 120.0, 300.0], size=n)
    new_exp_times = exp_times / rng.choice([2, 4, 8], size=n).astype(float)
    datasets = rng.choice(['ds1', 'ds2'], size=n)
    results = pd.DataFrame({
        'image_id': [f'{ds}_{i}' for ds, i in zip(datasets, range(n))],
        'org_exp_time': exp_times,
        'new_exp_time': new_exp_times,
        'TP': rng.integers(5, 30, n).astype(float),
        'FP': rng.integers(0, 10, n).astype(float),
        'FN': rng.integers(0, 10, n).astype(float),
        'IoU': rng.uniform(0, 1, n),
        'union': rng.integers(20, 100, n).astype(float),
        'SNR_org': rng.uniform(5, 20, n),
        'SNR_rec': rng.uniform(5, 20, n),
        'RFE': rng.uniform(0, 1, n),
        'PSNR_rec': rng.uniform(20, 40, n),
        'PSNR_noisy': rng.uniform(15, 30, n),
        'SSIM_rec': rng.uniform(0.5, 1.0, n),
        'SSIM_noisy': rng.uniform(0.3, 0.8, n),
    })
    results_csv = tmp_path / 'results.csv'
    results.to_csv(results_csv, index=False)

    table = create_table(str(metadata_csv), str(results_csv))

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert 'Precision' in table.columns
    assert 'Recall' in table.columns
    assert 'F-measure' in table.columns
    # The last row is the global total (exp_ratio == 0)
    assert (table['exp_ratio'] == 0).any()


# ---------------------------------------------------------------------------
# add_columns
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_add_columns_computes_exp_ratio_and_derived_flux_columns():
    rng = np.random.default_rng(5050)
    n = 20
    base_cols = {
        'flux_x_org': rng.uniform(1, 10, n), 'flux_y_org': rng.uniform(1, 10, n),
        'flux_x_rec': rng.uniform(1, 10, n), 'flux_y_rec': rng.uniform(1, 10, n),
        'flux_x_noise': rng.uniform(1, 10, n), 'flux_y_noise': rng.uniform(1, 10, n),
        'cflux_org': rng.uniform(1, 10, n), 'cflux_rec': rng.uniform(1, 10, n), 'cflux_noise': rng.uniform(1, 10, n),
        'flux_err_org': rng.uniform(0.1, 1, n), 'flux_err_rec': rng.uniform(0.1, 1, n), 'flux_err_noise': rng.uniform(0.1, 1, n),
        'exp_time_rec': rng.uniform(100, 500, n),
        'new_exp_time_rec': rng.uniform(50, 100, n),
        'exp_time_noise': rng.uniform(100, 500, n),
        'new_exp_time_noise': rng.uniform(50, 100, n),
    }
    df = pd.DataFrame(base_cols)

    result = add_columns(df)

    assert 'exp_ratio' in result.columns
    assert (result['exp_ratio'] > 0).all()
    assert 'j_flux_x_org' in result.columns
    assert 'abmag_flux_x_org' in result.columns
    assert 'a_flux_x_org' in result.columns
    assert 'delta_flux_x_org' in result.columns
