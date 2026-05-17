"""Additional branch coverage boosts for math_helpers and training utils."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import math_helpers as mh
from src.training import utils as ut


@pytest.mark.unit
def test_row_get_dict_and_evenly_spaced_guard_paths():
    assert mh._row_get({'k': 9}, 'k') == 9
    assert mh.evenly_spaced_numbers(5, 4, 2) == []
    assert mh.evenly_spaced_numbers(3, 3, 2) == []


@pytest.mark.unit
def test_apply_range_false_and_true_paths():
    ranges = pd.DataFrame(
        {
            'base': [2, 2, 2],
            'exponent': [1, 1, 2],
            'mean': [10.0, np.nan, 20.0],
            'std': [1.0, 1.0, np.nan],
        }
    )

    row_missing = {'combined_sigma': 1.0, 'sigma_base': 9, 'sigma_exponent': 9}
    assert mh._apply_range(ranges, row_missing, use_base=True, n_sigma=3) is False

    row_nan = {'combined_sigma': 20.0, 'sigma_base': 2, 'sigma_exponent': 2}
    assert mh._apply_range(ranges, row_nan, use_base=True, n_sigma=3) is False

    row_ok = {'combined_sigma': 10.5, 'sigma_base': 2, 'sigma_exponent': 1}
    assert mh._apply_range(ranges, row_ok, use_base=True, n_sigma=2) is True


@pytest.mark.unit
def test_create_ratios_validation_branches():
    with pytest.raises(ValueError):
        mh.create_ratios('x', 2, 2)
    with pytest.raises(ValueError):
        mh.create_ratios(2, 'x', 2)
    with pytest.raises(ValueError):
        mh.create_ratios(2, 2, 'x')
    assert mh.create_ratios(2, 0, 2) == []


@pytest.mark.unit
def test_create_tf_dataset_augmentations_branches(monkeypatch: pytest.MonkeyPatch):
    def fake_generator(images, kwargs_data, scaling):
        noisy = np.ones((2, 2, 1), dtype=np.float32)
        clean = np.ones((2, 2, 1), dtype=np.float32) * 2
        metadata = np.array(['a', 'b', 'c', 'd', 'e'], dtype=object)
        yield noisy, clean, metadata

    def fake_uniform(shape, minval, maxval=None, dtype=None):
        if dtype == tf.int32:
            return tf.constant(1, dtype=tf.int32)
        return tf.constant(0.9, dtype=tf.float32)

    monkeypatch.setattr(ut.tf.random, 'uniform', fake_uniform)

    ds = ut.create_tf_dataset(
        images=['x'],
        sample_generator=fake_generator,
        generator_kwargs={'ps': 2},
        scaling='min_max',
        batch_size=1,
        augment=True,
    )

    batch = next(iter(ds))
    assert len(batch) == 3
    assert batch[0].shape == (1, 2, 2, 1)


@pytest.mark.unit
def test_filtering_df_v2_final_backfill_branch():
    df = pd.DataFrame(
        {
            'A': [10, 9, 8, 7, 6, 5],
            'B': [10, 9, 8, 7, 6, 5],
            'C': [1, 1, 1, 1, 1, 1],
            'D': ['g1', 'g2', 'g3', 'g4', 'g5', 'g6'],
        }
    )

    out = ut.filtering_df_v2(
        data=df,
        n_samples=3,
        col_A='A',
        col_B='B',
        col_C='C',
        col_D='D',
        occurrences_per_col_D=0,
        quantiles=(10, 100),
        percentages=(100, 0),
    )
    assert len(out) == 3
