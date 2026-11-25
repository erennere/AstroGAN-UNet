"""Simulated shorter exposures.

A drizzled image in e-/s with pixel noise sigma, taken with exposure t, is turned into an
exposure t / ratio by adding zero-mean Gaussian noise of sigma_add = sigma * sqrt(ratio - 1), so
the result has noise sigma * sqrt(ratio). Empty (0) pixels stay empty.

Limitation (stated, not modelled): drizzled noise is spatially correlated; the added noise is white.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from astrounets.config import Noise


def max_ratio(exptime, noise: Noise):
    """Largest usable ratio for an exposure: the simulated exposure must stay >= min_new_exposure."""
    return np.minimum(noise.ratio_max, np.asarray(exptime, float) / noise.min_new_exposure)


def usable(crops: pd.DataFrame, noise: Noise) -> pd.DataFrame:
    """Crops whose exposure allows at least ratio_min."""
    return crops[max_ratio(crops["exptime"], noise) >= noise.ratio_min].reset_index(drop=True)


def sample_ratio(rng: np.random.Generator, exptime: float, noise: Noise) -> float:
    """Log-uniform ratio in [ratio_min, max_ratio(exptime)]."""
    hi = float(max_ratio(exptime, noise))
    return float(np.exp(rng.uniform(np.log(noise.ratio_min), np.log(hi))))


def added_sigma(sigma: float, ratio: float) -> float:
    return float(sigma) * float(np.sqrt(ratio - 1.0))


def noisy_sigma(sigma: float, ratio: float) -> float:
    """Pixel noise of the simulated exposure."""
    return float(sigma) * float(np.sqrt(ratio))


def add_noise(clean: np.ndarray, sigma: float, ratio: float, rng: np.random.Generator) -> np.ndarray:
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    noisy = clean + rng.normal(0.0, added_sigma(sigma, ratio), clean.shape).astype(np.float32)
    noisy[clean == 0] = 0.0
    return noisy.astype(np.float32)
