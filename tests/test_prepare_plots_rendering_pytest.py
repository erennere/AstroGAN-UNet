"""Tests for rendering helpers in src/visualization/prepare_plots.py."""
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
    binned_median,
    build_hexbin_plot_specs,
    finalize_plot,
    new_metrics,
)


# ---------------------------------------------------------------------------
# finalize_plot
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_finalize_plot_saves_file_to_disk(tmp_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    output_path = str(tmp_path / 'output.png')

    finalize_plot(fig, output_path, suptitle='Test', dpi=50)

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


# ---------------------------------------------------------------------------
# binned_median
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_binned_median_returns_non_empty_arrays():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.uniform(0.1, 100.0, 500))
    y = pd.Series(rng.uniform(0.5, 5.0, 500))
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 15)

    centers, medians, p25, p75 = binned_median(x, y, bins)
    assert len(centers) > 0
    assert len(centers) == len(medians) == len(p25) == len(p75)
    assert np.all(p25 <= medians)
    assert np.all(medians <= p75)


@pytest.mark.unit
def test_binned_median_ignores_nan_and_non_positive():
    rng = np.random.default_rng(7)
    x = pd.Series(np.concatenate([rng.uniform(1.0, 10.0, 100), [np.nan, 0.0, -1.0]]))
    y = pd.Series(np.concatenate([rng.uniform(0.5, 2.0, 100), [1.0, 1.0, 1.0]]))
    bins = 10
    centers, medians, _, _ = binned_median(x, y, bins)
    assert np.all(np.isfinite(centers))
    assert np.all(np.isfinite(medians))


# ---------------------------------------------------------------------------
# new_metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_new_metrics_computes_tp_fp_fn_and_overall():
    rng = np.random.default_rng(99)
    n = 100
    # TP: both present; FP: org missing, rec present; FN: org present, rec missing
    flux_org = np.where(rng.uniform(size=n) > 0.2, rng.uniform(1, 10, n), np.nan)
    flux_rec = np.where(rng.uniform(size=n) > 0.2, rng.uniform(1, 10, n), np.nan)
    df = pd.DataFrame({
        'exp_ratio': rng.choice([2, 4, 8], size=n),
        'flux_x_org': flux_org,
        'flux_x_rec': flux_rec,
    })

    result = new_metrics(df, exp_time_col='exp_ratio')

    assert isinstance(result, pd.DataFrame)
    assert 'Precision' in result.columns
    assert 'Recall' in result.columns
    assert 'F-measure' in result.columns
    # Last row should be the overall (exp_ratio==0)
    assert (result['exp_ratio'] == 0).any()


@pytest.mark.unit
def test_new_metrics_precision_recall_in_valid_range():
    df = pd.DataFrame({
        'exp_ratio': [2, 2, 2, 4, 4],
        'flux_x_org': [1.0, np.nan, 1.0, 1.0, np.nan],
        'flux_x_rec': [1.0, 1.0, np.nan, 1.0, 1.0],
    })
    result = new_metrics(df)
    precision_vals = result['Precision'].dropna()
    recall_vals = result['Recall'].dropna()
    assert (precision_vals >= 0).all() and (precision_vals <= 1).all()
    assert (recall_vals >= 0).all() and (recall_vals <= 1).all()


# ---------------------------------------------------------------------------
# build_hexbin_plot_specs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_hexbin_plot_specs_returns_list_of_dicts():
    specs = build_hexbin_plot_specs('viridis', 'plasma')
    assert isinstance(specs, list)
    assert len(specs) > 0
    for spec in specs:
        assert 'layout' in spec
        assert spec['layout'] in ('single', 'paired')
        assert 'output_filename' in spec
        assert 'suptitle' in spec
