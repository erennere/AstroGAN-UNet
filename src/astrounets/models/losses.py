"""Reconstruction losses (functions of y_true, y_pred -> scalar)."""
from __future__ import annotations

import tensorflow as tf


def _range(y_true):
    return tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True)


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def simae(y_true, y_pred):
    """MAE divided by each target's dynamic range (scale-invariant MAE)."""
    per_sample = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3), keepdims=True)
    return tf.reduce_mean(per_sample / tf.maximum(_range(y_true), 1e-6))


def log_cosh(y_true, y_pred):
    """log(cosh(residual)), stable form, divided by each target's dynamic range."""
    r = y_pred - y_true
    per_sample = tf.reduce_mean(r + tf.nn.softplus(-2.0 * r) - tf.math.log(2.0), axis=(1, 2, 3), keepdims=True)
    return tf.reduce_mean(per_sample / tf.maximum(_range(y_true), 1e-6))


def ssim(y_true, y_pred, win_size: int = 11, win_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03):
    """1 - mean SSIM with a Gaussian window (Wang et al. 2004); dynamic range taken from y_true."""
    coords = tf.cast(tf.range(win_size) - win_size // 2, tf.float32)
    g = tf.exp(-0.5 * (coords / win_sigma) ** 2)
    g = g / tf.reduce_sum(g)
    kernel = (g[:, None] * g[None, :])[:, :, None, None]

    def blur(t):
        return tf.nn.conv2d(t, kernel, strides=1, padding="SAME")

    mu_x, mu_y = blur(y_true), blur(y_pred)
    var_x = tf.maximum(blur(y_true * y_true) - mu_x ** 2, 0.0)
    var_y = tf.maximum(blur(y_pred * y_pred) - mu_y ** 2, 0.0)
    cov = blur(y_true * y_pred) - mu_x * mu_y
    span = _range(y_true)
    c1, c2 = (k1 * span) ** 2, (k2 * span) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + 1e-12)
    return 1.0 - tf.reduce_mean(s)


LOSSES = {"mae": mae, "mse": mse, "simae": simae, "log_cosh": log_cosh, "ssim": ssim}
