"""Input scaling for the network.

The parameters are fitted on the NOISY image only (the one image available at inference) and the
same parameters are applied to the clean target and used to invert the prediction. A perfect
prediction therefore maps back to the true flux. (The previous code scaled clean and noisy images
with their own statistics and inverted with the noisy ones, which biased fluxes by sigma_noisy/sigma_clean.)

robust_z is z_scale with the median and the normalised MAD (1.4826 x median absolute deviation)
instead of mean and standard deviation, taken over the crop's valid pixels only. A bright star then no
longer sets the scale of the whole crop: with z_scale, one star can raise the standard deviation 70x
above the sky noise, the noise all but vanishes in scaled units and the crop is barely denoised. The
other modes use every pixel (no-data pixels hold the sky level), exactly as their runs were trained.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


class Scaler:
    def __init__(self, mode: str, log_clip_max: float = 80.0):
        self.mode = mode
        self.log_clip_max = log_clip_max

    def fit(self, noisy: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, ...] | None:
        """Parameters from the noisy image, or None when it is (nearly) constant. `valid` (pixels with
        data) is used by robust_z only."""
        x = np.asarray(noisy, np.float64)
        if self.mode == "none":
            return ()
        if self.mode == "robust_z":
            v = x[np.asarray(valid, bool)] if valid is not None and np.any(valid) else x.ravel()
            center = float(np.median(v))
            scale = float(1.4826 * np.median(np.abs(v - center)))
            return None if scale < 1e-9 else (center, scale)
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
        elif self.mode in ("z_scale", "robust_z"):
            center, scale = params
            y = (x - center) / scale
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
        elif self.mode in ("z_scale", "robust_z"):
            center, scale = params
            x = y * scale + center
        elif self.mode == "min_max":
            lo, hi = params
            x = y * (hi - lo) + lo
        else:
            shift, lo, hi = params
            x = np.exp(np.minimum(y * (hi - lo) + lo, self.log_clip_max)) - shift
        return x.astype(np.float32)
