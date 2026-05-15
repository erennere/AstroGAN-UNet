"""Tests for additional training utility helpers in src/training/utils.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.utils import (
    candidates_based_on_range,
    candidates_based_on_ratio,
    ensure_parent_dir_exists,
    filtering_df,
    filtering_df_v2,
    is_relevant_crop,
)


# ---------------------------------------------------------------------------
# ensure_parent_dir_exists
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ensure_parent_dir_exists_creates_nested_dirs(tmp_path: Path):
    target_file = tmp_path / 'a' / 'b' / 'c' / 'output.txt'
    ensure_parent_dir_exists(str(target_file))
    assert (tmp_path / 'a' / 'b' / 'c').exists()


@pytest.mark.unit
def test_ensure_parent_dir_exists_is_idempotent(tmp_path: Path):
    target_file = tmp_path / 'subdir' / 'file.txt'
    ensure_parent_dir_exists(str(target_file))
    ensure_parent_dir_exists(str(target_file))  # must not raise


# ---------------------------------------------------------------------------
# is_relevant_crop
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_is_relevant_crop_returns_true_when_crop_exceeds_threshold():
    stats = {'full_abs_mean': 100.0, 'crop_abs_mean': 60.0}
    assert is_relevant_crop(stats, delta=0.5) is True


@pytest.mark.unit
def test_is_relevant_crop_returns_false_when_crop_below_threshold():
    stats = {'full_abs_mean': 100.0, 'crop_abs_mean': 30.0}
    assert is_relevant_crop(stats, delta=0.5) is False


@pytest.mark.unit
def test_is_relevant_crop_custom_delta():
    stats = {'full_abs_mean': 100.0, 'crop_abs_mean': 15.0}
    assert is_relevant_crop(stats, delta=0.1) is True
    assert is_relevant_crop(stats, delta=0.2) is False


# ---------------------------------------------------------------------------
# filtering_df
# ---------------------------------------------------------------------------

def _make_filtering_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    datasets = rng.choice(['A', 'B', 'C'], size=n)
    return pd.DataFrame({
        'id': [f'img_{i}' for i in range(n)],
        'dataset': datasets,
        'abs_mean': rng.uniform(1.0, 100.0, n),
        'full_abs_mean': np.full(n, 50.0),
        'crop_abs_mean': rng.uniform(30.0, 80.0, n),  # all > delta*50=25
        'noise_ratio': rng.uniform(2.0, 10.0, n),
    })


@pytest.mark.unit
def test_filtering_df_returns_at_most_n_samples():
    rng = np.random.default_rng(1)
    df = _make_filtering_df(200, rng)
    result = filtering_df(df.copy(), n_samples=50, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(result) <= 50


@pytest.mark.unit
def test_filtering_df_returns_full_df_when_n_samples_larger():
    rng = np.random.default_rng(2)
    df = _make_filtering_df(10, rng)
    result = filtering_df(df.copy(), n_samples=500, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(result) == len(df)


@pytest.mark.unit
def test_filtering_df_raises_when_no_sort_key_provided():
    rng = np.random.default_rng(3)
    df = _make_filtering_df(50, rng)
    with pytest.raises(ValueError):
        filtering_df(df.copy(), n_samples=10, delta=0.5, id_='dataset', abs_='abs_mean')


# ---------------------------------------------------------------------------
# filtering_df_v2
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_filtering_df_v2_returns_correct_count():
    rng = np.random.default_rng(10)
    n = 200
    df = pd.DataFrame({
        'col_A': rng.uniform(0, 1, n),
        'col_B': rng.uniform(0, 1, n),
        'col_C': rng.uniform(0, 1, n),
        'col_D': rng.choice(['x', 'y', 'z'], size=n),
    })
    result = filtering_df_v2(df, n_samples=50, col_A='col_A', col_B='col_B', col_C='col_C', col_D='col_D',
                              occurrences_per_col_D=2, quantiles=(20, 40, 60, 80, 100), percentages=(1, 1, 1, 1, 1))
    assert len(result) <= 50


@pytest.mark.unit
def test_filtering_df_v2_empty_df_returns_empty():
    df = pd.DataFrame({'col_A': pd.Series(dtype=float), 'col_B': pd.Series(dtype=float),
                        'col_C': pd.Series(dtype=float), 'col_D': pd.Series(dtype=str)})
    result = filtering_df_v2(df, n_samples=10, col_A='col_A', col_B='col_B', col_C='col_C', col_D='col_D',
                              occurrences_per_col_D=1, quantiles=(100,), percentages=(1,))
    assert len(result) == 0


@pytest.mark.unit
def test_filtering_df_v2_raises_on_invalid_quantiles():
    df = pd.DataFrame({'a': [1], 'b': [1], 'c': [1], 'd': ['x']})
    with pytest.raises(ValueError):
        filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', 1, quantiles=(80, 60, 100), percentages=(1, 1, 1))


@pytest.mark.unit
def test_filtering_df_v2_n_samples_zero_returns_empty():
    rng = np.random.default_rng(99)
    df = pd.DataFrame({'a': rng.uniform(0, 1, 20), 'b': rng.uniform(0, 1, 20),
                        'c': rng.uniform(0, 1, 20), 'd': ['x'] * 20})
    result = filtering_df_v2(df, n_samples=0, col_A='a', col_B='b', col_C='c', col_D='d',
                              occurrences_per_col_D=1, quantiles=(100,), percentages=(1,))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# candidates_based_on_range
# ---------------------------------------------------------------------------

def _make_kwargs_range():
    scm = {
        'std_bkg': 'std_bkg', 'abs_mean': 'abs_mean', 'abs_median': 'abs_median',
        'median_bkg': 'median_bkg', 'mean_bkg': 'mean_bkg', 'max_bkg': 'max_bkg',
        'median_src': 'median_src', 'mean_src': 'mean_src', 'max_src': 'max_src',
    }
    return {
        'name_col': 'name', 'location_col': 'location', 'dataset': 'dataset',
        'exposure_col': 'exp_time',
        'lowest_power': -2, 'highest_power': 3, 'n_samples_per_magnitude': 2,
        'stats_column_map': scm,
        'original_stats_prefix': 'orig_',
    }


def _make_row(rng: np.random.Generator) -> pd.Series:
    return pd.Series({
        'name': 'img_001', 'location': '/data/img_001.fits', 'dataset': 'DS1',
        'exp_time': 300.0,
        'std_bkg': 5.0, 'abs_mean': 50.0, 'abs_median': 48.0,
        'median_bkg': 3.0, 'mean_bkg': 3.2, 'max_bkg': 10.0,
        'median_src': 20.0, 'mean_src': 22.0, 'max_src': 100.0,
        'orig_std_bkg': 5.0, 'orig_abs_mean': 50.0, 'orig_abs_median': 48.0,
        'orig_median_bkg': 3.0, 'orig_mean_bkg': 3.2, 'orig_max_bkg': 10.0,
        'orig_median_src': 20.0, 'orig_mean_src': 22.0, 'orig_max_src': 100.0,
    })


@pytest.mark.unit
def test_candidates_based_on_range_returns_list_of_dicts():
    rng = np.random.default_rng(20)
    row = _make_row(rng)
    kwargs = _make_kwargs_range()
    result = candidates_based_on_range(row, kwargs)
    assert isinstance(result, list)
    assert len(result) > 0
    for entry in result:
        assert 'combined_sigma' in entry
        assert 'new_exp_time' in entry
        assert 'exp_ratio' in entry


@pytest.mark.unit
def test_candidates_based_on_range_returns_empty_for_nan_sigma():
    rng = np.random.default_rng(21)
    row = _make_row(rng)
    row['std_bkg'] = np.nan
    row['orig_std_bkg'] = np.nan
    kwargs = _make_kwargs_range()
    result = candidates_based_on_range(row, kwargs)
    assert result == []


@pytest.mark.unit
def test_candidates_based_on_range_returns_empty_for_nan_exposure():
    rng = np.random.default_rng(22)
    row = _make_row(rng)
    row['exp_time'] = np.nan
    kwargs = _make_kwargs_range()
    result = candidates_based_on_range(row, kwargs)
    assert result == []


# ---------------------------------------------------------------------------
# candidates_based_on_ratio
# ---------------------------------------------------------------------------

def _make_kwargs_ratio():
    scm = {
        'std_bkg': 'std_bkg', 'abs_mean': 'abs_mean', 'abs_median': 'abs_median',
        'median_bkg': 'median_bkg', 'mean_bkg': 'mean_bkg', 'max_bkg': 'max_bkg',
        'median_src': 'median_src', 'mean_src': 'mean_src', 'max_src': 'max_src',
    }
    return {
        'name_col': 'name', 'location_col': 'location', 'dataset': 'dataset',
        'exposure_col': 'exp_time',
        'ratio_initial': 2.0, 'ratio_count': 3, 'ratio_growth': 2.0,
        'stats_column_map': scm,
        'original_stats_prefix': 'orig_',
    }


@pytest.mark.unit
def test_candidates_based_on_ratio_returns_one_entry_per_ratio():
    rng = np.random.default_rng(30)
    row = _make_row(rng)
    kwargs = _make_kwargs_ratio()
    result = candidates_based_on_ratio(row, kwargs)
    assert isinstance(result, list)
    assert len(result) == 3  # ratio_count = 3
    ratios = [e['exp_ratio'] for e in result]
    assert ratios[0] == pytest.approx(2.0)
    assert ratios[1] == pytest.approx(4.0)
    assert ratios[2] == pytest.approx(8.0)


@pytest.mark.unit
def test_candidates_based_on_ratio_returns_empty_for_nan_sigma():
    rng = np.random.default_rng(31)
    row = _make_row(rng)
    row['std_bkg'] = np.nan
    row['orig_std_bkg'] = np.nan
    kwargs = _make_kwargs_ratio()
    result = candidates_based_on_ratio(row, kwargs)
    assert result == []
