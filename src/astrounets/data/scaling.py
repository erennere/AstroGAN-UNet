"""Input scaling for the network.

The parameters are fitted on the NOISY image only (the one image available at inference) and the
same parameters are applied to the clean target and used to invert the prediction. A perfect
prediction therefore maps back to the true flux. (The previous code scaled clean and noisy images
with their own statistics and inverted with the noisy ones, which biased fluxes by sigma_noisy/sigma_clean.)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


class Scaler:
    def __init__(self, mode: str, log_clip_max: float = 80.0):
        self.mode = mode
        self.log_clip_max = log_clip_max

    def fit(self, noisy: np.ndarray) -> tuple[float, ...] | None:
        """Parameters from the noisy image, or None when it is (nearly) constant."""
        x = np.asarray(noisy, np.float64)
        if self.mode == "none":
            return ()
        if self.mode == "z_scale":
            mean, std = float(x.mean()), float(x.std())
            return None if std < 1e-9 else (mean, std)
        if self.mode == "min_max":
            lo, hi = float(x.min()), float(x.max())
            return None if hi - lo < 1e-9 else (lo, hi)
        if self.mode == "log_min_max":
            shift = 1.0 - float(x.min())
            logs = np.log(x + shift)
            lo, hi = float(logs.min()), float(logs.max())
            return None if hi - lo < 1e-9 else (shift, lo, hi)
        raise ValueError(f"unknown scaling mode {self.mode!r}")

    def forward(self, x: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.mode == "none":
            y = x
        elif self.mode == "z_scale":
            mean, std = params
            y = (x - mean) / std
        elif self.mode == "min_max":
            lo, hi = params
            y = (x - lo) / (hi - lo)
        else:
            shift, lo, hi = params
            y = (np.log(np.maximum(x + shift, EPS)) - lo) / (hi - lo)
        return y.astype(np.float32)

    def inverse(self, y: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
        y = np.asarray(y, np.float64)
        if self.mode == "none":
            x = y
        elif self.mode == "z_scale":
            mean, std = params
            x = y * std + mean
        elif self.mode == "min_max":
            lo, hi = params
            x = y * (hi - lo) + lo
        else:
            shift, lo, hi = params
            x = np.exp(np.minimum(y * (hi - lo) + lo, self.log_clip_max)) - shift
        return x.astype(np.float32)
