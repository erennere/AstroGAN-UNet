"""Source detection and photometry with SEP, using one absolute threshold for every image."""
from __future__ import annotations

import numpy as np
import pandas as pd
import sep

from astrounets.config import Eval

COLUMNS = ["x", "y", "flux", "a", "b", "theta", "radius", "kron_radius", "ellipticity"]
sep.set_extract_pixstack(2_000_000)
sep.set_sub_object_limit(4096)


def background(image: np.ndarray, e: Eval) -> np.ndarray | None:
    """SEP background map of an image, or None if SEP cannot model it."""
    try:
        return sep.Background(np.ascontiguousarray(image, dtype=np.float64), bw=e.bkg_box_size, bh=e.bkg_box_size).back()
    except Exception:
        return None


def measure(image: np.ndarray, threshold: float, e: Eval, back: np.ndarray | None = None) -> pd.DataFrame:
    """Detect sources above `threshold` (e-/s per pixel, absolute) after subtracting a background.

    `back` is the background map to subtract; by default the image's own. To compare images of the
    same sky, pass one map (from the deep image) to every call: separately estimated backgrounds
    absorb more of the faint outer light of sources the noisier an image is, which biases fluxes.
    Returns positions, Kron flux, shape (a, b, theta), half-light radius, circularised Kron radius
    (pixels) and ellipticity 1 - b/a.
    """
    data = np.ascontiguousarray(image, dtype=np.float64)
    back = background(data, e) if back is None else back
    if back is None:
        return pd.DataFrame(columns=COLUMNS)
    sub = data - back
    try:
        objects = sep.extract(sub, threshold, minarea=e.detect_minarea,
                              deblend_nthresh=e.deblend_nthresh, deblend_cont=e.deblend_cont)
    except Exception:  # SEP buffer overflows on pathological images: report no detections
        return pd.DataFrame(columns=COLUMNS)
    if len(objects) == 0:
        return pd.DataFrame(columns=COLUMNS)
    objects = objects[np.isfinite(objects["x"]) & np.isfinite(objects["y"])]
    x, y = objects["x"].astype(float), objects["y"].astype(float)
    a, b = objects["a"].astype(float), objects["b"].astype(float)
    # SEP stores theta as float32: +-pi/2 in float32 lies just outside SEP's own float64 range check
    theta = np.clip(objects["theta"].astype(float), -np.pi / 2, np.pi / 2)
    shape_ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(theta) & (a > 0) & (b > 0)
    a, theta = np.where(shape_ok, a, 1.0), np.where(shape_ok, theta, 0.0)
    b = np.where(shape_ok, np.minimum(b, a), 1.0)
    kron, _ = sep.kron_radius(sub, x, y, a, b, theta, e.kron_radius_factor)
    kron = np.where(np.isfinite(kron) & (kron > 0), kron, 0.0)
    use_circle = ~shape_ok | (kron * np.sqrt(a * b) < e.min_radius)  # degenerate shapes: circular aperture
    flux = np.zeros(len(x))
    if (~use_circle).any():
        k = ~use_circle
        flux[k], _, _ = sep.sum_ellipse(sub, x[k], y[k], a[k], b[k], theta[k], e.kron_factor * kron[k], subpix=1)
    if use_circle.any():
        k = use_circle
        flux[k], _, _ = sep.sum_circle(sub, x[k], y[k], e.min_radius, subpix=1)
    radius, _ = sep.flux_radius(sub, x, y, e.kron_radius_factor * a, 0.5, normflux=np.where(flux > 0, flux, np.nan), subpix=5)
    # circularised Kron radius in pixels (the aperture used for the flux): kron x sqrt(a b), or the circle
    kron_radius = np.where(use_circle, e.min_radius, kron * np.sqrt(a * b))
    a, b = np.where(shape_ok, a, np.nan), np.where(shape_ok, b, np.nan)
    return pd.DataFrame({"x": x, "y": y, "flux": flux, "a": a, "b": b, "theta": theta,
                         "radius": radius, "kron_radius": kron_radius, "ellipticity": 1.0 - b / a})
