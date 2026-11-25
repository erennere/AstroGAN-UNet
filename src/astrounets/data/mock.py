"""Synthetic WFC3/IR-like sky for smoke runs and tests.

Images contain exponential-disk galaxies and Gaussian stars at known positions and fluxes,
Gaussian pixel noise with a known sigma, an empty rotated drizzle border and a few NaN pixels.
They are written as real FITS files with the headers the pipeline reads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.data.fits_io import write_image

PHOTFLAM = 1.9275602e-20
PHOTPLAM = 15369.176
PI_NAMES = ("FABER", "MOMCHEVA", "SMITH")
N_FIELDS = 6  # well-separated sky tiles


def noise_sigma(exptime: float) -> float:
    """Pixel noise (e-/s) of a mock drizzled image; falls as 1/sqrt(exposure)."""
    return 0.02 * np.sqrt(1000.0 / exptime)


def make_image(rng: np.random.Generator, size: int, exptime: float, n_sources: int | None = None,
               border: bool = True) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Return (image in e-/s, true pixel sigma, source catalog)."""
    sigma = noise_sigma(exptime)
    image = np.zeros((size, size), dtype=np.float64)
    n_sources = n_sources if n_sources is not None else int(size * size / 1500)
    yy, xx = np.mgrid[0:size, 0:size]
    rows = []
    for k in range(n_sources):
        x0, y0 = rng.uniform(8, size - 8, 2)
        flux = float(np.exp(rng.uniform(np.log(0.05), np.log(50.0))))
        star = rng.random() < 0.3
        if star:
            scale, q, theta = 1.0, 1.0, 0.0
        else:
            scale, q, theta = rng.uniform(1.0, 4.0), rng.uniform(0.4, 1.0), rng.uniform(0, np.pi)
        r = int(np.ceil(8 * scale)) + 2
        ys, xs = slice(max(0, int(y0) - r), min(size, int(y0) + r + 1)), slice(max(0, int(x0) - r), min(size, int(x0) + r + 1))
        dx, dy = xx[ys, xs] - x0, yy[ys, xs] - y0
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = (-dx * np.sin(theta) + dy * np.cos(theta)) / q
        if star:
            profile = np.exp(-0.5 * (u ** 2 + v ** 2) / scale ** 2)
        else:
            profile = np.exp(-np.hypot(u, v) / scale)
        image[ys, xs] += flux * profile / profile.sum()
        rows.append({"x": x0, "y": y0, "flux": flux, "star": star, "scale": scale, "q": q})
    image += rng.normal(0.0, sigma, image.shape)
    if border:
        angle = rng.uniform(0.15, 0.5)
        c = (size - 1) / 2
        u = (xx - c) * np.cos(angle) + (yy - c) * np.sin(angle)
        v = -(xx - c) * np.sin(angle) + (yy - c) * np.cos(angle)
        inside = (np.abs(u) < 0.42 * size) & (np.abs(v) < 0.42 * size)
        image[~inside] = 0.0
        bad = rng.integers(0, size, (5, 2))
        image[bad[:, 0], bad[:, 1]] = np.nan
    return image.astype(np.float32), float(sigma), pd.DataFrame(rows)


def mock_mast(cfg) -> pd.DataFrame:
    """Write synthetic images to data/mast/originals and return the metadata table (canonical columns)."""
    m = cfg.mast
    rng = np.random.default_rng(cfg.seed)
    originals = cfg.mast_dir / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    rows = []
    size = m.tile_arcmin / 60.0
    for i in range(m.mock_images):
        field = i % N_FIELDS
        # each mock field sits in the middle of its own sky tile, far from the others
        band = int((-20.0 + 5.0 * field + 90.0) / size)
        dec = -90.0 + (band + 0.5) * size
        width = size / np.cos(np.radians(dec))
        ra = (int((30.0 + 40.0 * field) / width) + 0.5) * width
        ra, dec = ra + rng.normal(0, 0.1 * width), dec + rng.normal(0, 0.1 * size)
        exptime = float(rng.choice([500.0, 1000.0, 2500.0, 5000.0]))
        image, _, _ = make_image(rng, m.mock_image_size, exptime)
        image_id = f"MOCK{i:04d}"
        path = originals / f"{image_id.lower()}_{m.product_suffix}"
        write_image(path, image, cfg.dataset.hdu,
                    {"EXPTIME": exptime, "PHOTFLAM": PHOTFLAM, "PHOTPLAM": PHOTPLAM, "FILTER": "F160W"})
        rows.append({
            "id": image_id, "exposure": exptime, "ra": ra, "dec": dec, "aperture": "IR",
            "pi": PI_NAMES[(i // N_FIELDS) % len(PI_NAMES)], "target": f"FIELD-{field}",
            "url": f"mock://{image_id}", "path": f"originals/{path.name}",
        })
    return pd.DataFrame(rows)
