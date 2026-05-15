import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.training.math_helpers import (
    _sigma_kernel_from_fit_wrapper,
    adaptive_log_transform_and_normalize,
    apply_scaling_and_stats,
    create_ratios,
    create_simulated_image_poisson,
    min_max_normalization,
    sigma_kernel_from_row,
    zscore_normalization,
)
from src.training.utils import log_cosh_loss, scale_invariant_mae, ssim_loss


@pytest.mark.unit
def test_scale_invariant_mae_properties():
    y_true = tf.reshape(tf.linspace(0.0, 10.0, 64 * 64), (1, 64, 64, 1))
    y_pred = y_true + 2.0
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    loss = scale_invariant_mae(y_true, y_pred)
    constant_loss = scale_invariant_mae(tf.ones((1, 8, 8, 1)), tf.zeros((1, 8, 8, 1)))

    assert loss.shape == ()
    assert float(scale_invariant_mae(y_true, y_true)) == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(float(constant_loss))
    assert float(loss) < float(mae)


@pytest.mark.unit
def test_log_cosh_loss_properties():
    y_true = tf.reshape(tf.linspace(0.0, 1.0, 64), (1, 8, 8, 1))
    small_pred = y_true + 0.01
    large_pred = y_true + 4.0
    small_loss = float(log_cosh_loss(y_true, small_pred))
    large_loss = float(log_cosh_loss(y_true, large_pred))
    normalized_mse = float(tf.reduce_mean(tf.square(small_pred - y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true) + 1e-10))
    normalized_mae = float(tf.reduce_mean(tf.abs(large_pred - y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true) + 1e-10))

    assert log_cosh_loss(y_true, y_true).shape == ()
    assert float(log_cosh_loss(y_true, y_true)) == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(float(log_cosh_loss(tf.ones((1, 8, 8, 1)), tf.zeros((1, 8, 8, 1)))))
    assert small_loss == pytest.approx(normalized_mse / 2.0, rel=0.2)
    assert large_loss == pytest.approx(normalized_mae, rel=0.2)


@pytest.mark.unit
def test_ssim_loss_properties():
    smooth = tf.reshape(tf.linspace(0.0, 1.0, 64 * 64), (1, 64, 64, 1))
    noisy = tf.random.uniform((1, 64, 64, 1), dtype=tf.float32)
    small_true = tf.reshape(tf.linspace(0.0, 1.0, 64), (1, 8, 8, 1))
    small_pred = tf.reverse(small_true, axis=[1])
    noise_loss = float(ssim_loss(smooth, noisy))
    small_loss = float(ssim_loss(small_true, small_pred, win_size=5, win_sigma=1.0))

    assert 0.0 <= float(ssim_loss(smooth, smooth)) <= 1.0
    assert float(ssim_loss(smooth, smooth)) == pytest.approx(0.0, abs=1e-7)
    assert 0.0 <= noise_loss <= 1.1
    assert noise_loss > 0.5
    assert 0.0 <= small_loss <= 1.1


@pytest.mark.unit
@pytest.mark.parametrize('loss_fn', [scale_invariant_mae, log_cosh_loss, ssim_loss])
def test_losses_are_differentiable(loss_fn):
    x = tf.ones((1, 16, 16, 1), dtype=tf.float32)
    weight = tf.Variable(0.5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        y_pred = x * weight
        loss = loss_fn(x, y_pred)
    grad = tape.gradient(loss, weight)

    assert grad is not None
    assert np.isfinite(float(grad.numpy()))


@pytest.mark.unit
def test_min_max_normalization_output_range_and_constant_handling():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalized, min_value, max_value = min_max_normalization(arr)

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert min_value == pytest.approx(1.0)
    assert max_value == pytest.approx(4.0)
    assert min_max_normalization(np.ones((4, 4), dtype=np.float32)) is None


@pytest.mark.unit
def test_zscore_normalization_behavior_and_zero_std_handling():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalized, mean_value, std_value = zscore_normalization(arr)

    assert float(np.mean(normalized)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(normalized)) == pytest.approx(1.0, abs=1e-6)
    assert mean_value == pytest.approx(np.mean(arr))
    assert std_value == pytest.approx(np.std(arr))
    assert zscore_normalization(np.ones((4, 4), dtype=np.float32)) is None


@pytest.mark.unit
def test_adaptive_log_transform_preserves_range_and_order():
    arr = np.array([[0.5, 1.0], [2.0, 10.0]], dtype=np.float32)
    normalized, *_ = adaptive_log_transform_and_normalize(arr)

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert list(np.argsort(arr.flatten())) == list(np.argsort(normalized.flatten()))


@pytest.mark.unit
@pytest.mark.parametrize('scaling', [None, 'min_max', 'z_scale', 'log_min_max'])
def test_apply_scaling_and_stats_returns_expected_shapes_and_dtypes(scaling):
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) + 1.0
    scaled = apply_scaling_and_stats(arr, arr * 1.5, scaling, 'synthetic')

    assert scaled is not None
    scaled_noisy, scaled_clean, stats = scaled
    assert scaled_noisy.shape == arr.shape
    assert scaled_clean.shape == arr.shape
    assert scaled_noisy.dtype == np.float32
    assert scaled_clean.dtype == np.float32
    assert stats[0] == 'synthetic'


@pytest.mark.unit
def test_create_simulated_image_poisson_properties():
    np.random.seed(123)
    clean = np.full((32, 32), 5.0, dtype=np.float32)
    simulated = create_simulated_image_poisson(clean, original_exposure=120.0, exposure_ratio=2.0)

    assert simulated.shape == clean.shape
    assert np.all(simulated >= 0.0)
    assert not np.array_equal(simulated, clean)


@pytest.mark.unit
def test_sigma_kernel_helpers_return_positive_scalars(tiny_metadata_df):
    row = tiny_metadata_df.iloc[0]
    fit_data = pd.DataFrame([{'t': 0.5, 'a': 1.5, 'dt': 0.01, 'da': 0.01}])
    fit_sigma = _sigma_kernel_from_fit_wrapper(row, fit_data, 'unused')
    row_sigma = sigma_kernel_from_row(row, 'bkg_sigma')

    assert fit_sigma > 0
    assert row_sigma > 0
    assert row_sigma == pytest.approx(row['bkg_sigma'])


@pytest.mark.unit
def test_create_ratios_returns_expected_progression():
    assert create_ratios(2.0, 4, 1.5) == [2.0, 3.0, 5.0, 8.0]
