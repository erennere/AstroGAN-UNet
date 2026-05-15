"""Tests for remaining utility functions: sample_ratio/range_v2/local_mean (training/utils.py),
attention_gate/conv_batch_maxpooling (models/network.py), and log_range (uncropped_metrics.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# log_range (src/evaluation/uncropped_metrics.py)
# ---------------------------------------------------------------------------

from src.evaluation.uncropped_metrics import log_range


@pytest.mark.unit
def test_log_range_returns_sorted_array():
    result = log_range(-2, 2)
    assert isinstance(result, np.ndarray)
    assert np.all(result[:-1] <= result[1:])  # sorted


@pytest.mark.unit
def test_log_range_spans_decades():
    result = log_range(0, 1)
    # From 10^0 to 10^1: 1,2,...,9,10,20,...,90 = 18 values
    assert len(result) == 18


@pytest.mark.unit
def test_log_range_all_positive():
    result = log_range(-3, 3)
    assert np.all(result > 0)


# ---------------------------------------------------------------------------
# attention_gate (src/models/network.py)
# ---------------------------------------------------------------------------

def test_attention_gate_output_shape_matches_skip():
    import tensorflow as tf
    from src.models.network import attention_gate

    ps = 32
    batch = 2
    skip = tf.random.normal((batch, ps, ps, 16))
    gating = tf.random.normal((batch, ps, ps, 32))

    # Build within a Functional API model so Keras layers get proper graphs
    inp_skip = tf.keras.Input(shape=(ps, ps, 16))
    inp_gate = tf.keras.Input(shape=(ps, ps, 32))
    out = attention_gate(inp_skip, inp_gate, n_intermediate_filters=8)
    model = tf.keras.Model(inputs=[inp_skip, inp_gate], outputs=out)

    result = model([skip, gating])
    assert result.shape == (batch, ps, ps, 16)


def test_attention_gate_weights_are_in_zero_one():
    import tensorflow as tf
    from src.models.network import attention_gate

    ps = 16
    inp_skip = tf.keras.Input(shape=(ps, ps, 8))
    inp_gate = tf.keras.Input(shape=(ps, ps, 8))
    out = attention_gate(inp_skip, inp_gate, n_intermediate_filters=4)
    model = tf.keras.Model(inputs=[inp_skip, inp_gate], outputs=out)

    skip_val = tf.ones((1, ps, ps, 8))
    gating_val = tf.zeros((1, ps, ps, 8))
    result = model([skip_val, gating_val])
    # After sigmoid gating and multiplying with skip=1, values should be in [0, 1]
    assert float(tf.reduce_min(result)) >= 0.0
    assert float(tf.reduce_max(result)) <= 1.0


# ---------------------------------------------------------------------------
# conv_batch_maxpooling (src/models/network.py)
# ---------------------------------------------------------------------------

def test_conv_batch_maxpooling_returns_two_tensors():
    import tensorflow as tf
    from src.models.network import conv_batch_maxpooling

    inp = tf.keras.Input(shape=(64, 64, 1))
    conv, pool = conv_batch_maxpooling(inp, output_channels=16, kernel_size=3, pooling_size=2)
    model = tf.keras.Model(inputs=inp, outputs=[conv, pool])

    x = tf.random.normal((1, 64, 64, 1))
    conv_out, pool_out = model(x)
    assert conv_out.shape == (1, 64, 64, 16)
    assert pool_out.shape == (1, 32, 32, 16)


# ---------------------------------------------------------------------------
# sample_ratio / sample_range_v2 (src/training/utils.py)
# ---------------------------------------------------------------------------

from src.training.utils import sample_ratio, sample_range_v2


def _make_ratio_candidates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    datasets = rng.choice(['A', 'B', 'C'], size=n)
    locations = [f'/data/loc_{i}.fits' for i in range(n)]
    return pd.DataFrame({
        'dataset': datasets,
        'location': locations,
        'abs_mean': rng.uniform(10, 100, n),
        'full_abs_mean': np.full(n, 50.0),
        'crop_abs_mean': rng.uniform(30, 80, n),
        'exp_ratio': rng.uniform(2.0, 16.0, n),
        'sm_peak_NSR': rng.uniform(0.1, 5.0, n),
        'base': rng.integers(1, 9, n),
        'exponent_diff': rng.integers(0, 3, n),
    })


@pytest.mark.unit
def test_sample_ratio_returns_at_most_n_samples():
    rng = np.random.default_rng(100)
    df = _make_ratio_candidates(200, rng)
    result = sample_ratio(df.copy(), 50, {'delta': 0.5})
    assert len(result) <= 50


@pytest.mark.unit
def test_sample_range_v2_returns_at_most_n_samples():
    rng = np.random.default_rng(101)
    df = _make_ratio_candidates(200, rng)
    kwargs = {
        'occurrences_per_col_D': 2,
        'quantiles': (20, 40, 60, 80, 100),
        'percentages': (1, 1, 1, 1, 1),
    }
    result = sample_range_v2(df.copy(), 60, kwargs)
    assert len(result) <= 60


@pytest.mark.unit
def test_sample_range_v2_empty_when_n_zero():
    rng = np.random.default_rng(102)
    df = _make_ratio_candidates(50, rng)
    kwargs = {'occurrences_per_col_D': 1, 'quantiles': (100,), 'percentages': (1,)}
    result = sample_range_v2(df.copy(), 0, kwargs)
    assert len(result) == 0
