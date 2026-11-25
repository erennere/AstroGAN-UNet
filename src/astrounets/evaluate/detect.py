"""Source detection and photometry with SEP, using one absolute threshold for every image."""
from __future__ import annotations

import numpy as np
import pandas as pd
import sep

from astrounets.config import Eval

COLUMNS = ["x", "y", "flux", "a", "b", "theta", "radius", "ellipticity"]
sep.set_extract_pixstack(2_000_000)
sep.set_sub_object_limit(4096)


def measure(image: np.ndarray, threshold: float, e: Eval) -> pd.DataFrame:
    """Detect sources above `threshold` (e-/s per pixel, absolute) after local background removal.

    Returns positions, Kron flux, shape (a, b, theta), half-light radius and ellipticity 1 - b/a.
    """
    data = np.ascontiguousarray(image, dtype=np.float64)
    try:
        bkg = sep.Background(data, bw=e.bkg_box_size, bh=e.bkg_box_size)
        sub = data - bkg.back()
        objects = sep.extract(sub, threshold, minarea=e.detect_minarea,
                              deblend_nthresh=e.deblend_nthresh, deblend_cont=e.deblend_cont)
    except Exception:  # SEP buffer overflows on pathological images: report no detections
        return pd.DataFrame(columns=COLUMNS)
    if len(objects) == 0:
        return pd.DataFrame(columns=COLUMNS)
    x, y = objects["x"].astype(float), objects["y"].astype(float)
    a, b, theta = objects["a"].astype(float), objects["b"].astype(float), objects["theta"].astype(float)
    a, b = np.maximum(a, 1e-3), np.maximum(b, 1e-3)
    kron, _ = sep.kron_radius(sub, x, y, a, b, theta, e.kron_radius_factor)
    kron = np.where(np.isfinite(kron) & (kron > 0), kron, 0.0)
    use_circle = kron * np.sqrt(a * b) < e.min_radius
    flux = np.zeros(len(x))
    if (~use_circle).any():
        k = ~use_circle
        flux[k], _, _ = sep.sum_ellipse(sub, x[k], y[k], a[k], b[k], theta[k], e.kron_factor * kron[k], subpix=1)
    if use_circle.any():
        k = use_circle
        flux[k], _, _ = sep.sum_circle(sub, x[k], y[k], e.min_radius, subpix=1)
    radius, _ = sep.flux_radius(sub, x, y, e.kron_radius_factor * a, 0.5, normflux=np.where(flux > 0, flux, np.nan), subpix=5)
    return pd.DataFrame({"x": x, "y": y, "flux": flux, "a": a, "b": b, "theta": theta,
                         "radius": radius, "ellipticity": 1.0 - b / a})
