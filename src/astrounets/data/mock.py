"""Synthetic WFC3/IR-like sky for smoke runs and tests.

Images contain exponential-disk galaxies and Gaussian stars at known positions and fluxes on a sky
background, Gaussian pixel noise with a known sigma, an empty rotated drizzle border and a few NaN
pixels. Crowded images add thousands of faint stars and unresolved light, like a field in M31. The
noise follows the detector model of the dataset stage (photon + dark + read noise, lowered by the
drizzle correlation factor MOCK_K). Images are written as real FITS files with the headers the
pipeline reads.
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
SKY = 0.6               # e-/s per pixel, F160W-like
UNRESOLVED = 3.0        # extra e-/s per pixel of unresolved starlight in crowded images
DARK, READ_NOISE = 0.05, 15.0  # as in configs/default.yaml
MOCK_K = 0.75           # drizzle correlation: measured / uncorrelated noise


def noise_sigma(exptime: float, back: float = SKY) -> float:
    """Pixel noise (e-/s) of a mock drizzled single exposure with background `back`."""
    return float(MOCK_K * np.sqrt((back + DARK) / exptime + READ_NOISE ** 2 / exptime ** 2))


def _add_source(image, xx, yy, x0, y0, flux, scale, q, theta, star):
    size = image.shape[0]
    r = int(np.ceil(8 * scale)) + 2
    ys, xs = slice(max(0, int(y0) - r), min(size, int(y0) + r + 1)), slice(max(0, int(x0) - r), min(size, int(x0) + r + 1))
    dx, dy = xx[ys, xs] - x0, yy[ys, xs] - y0
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = (-dx * np.sin(theta) + dy * np.cos(theta)) / q
    profile = np.exp(-0.5 * (u ** 2 + v ** 2) / scale ** 2) if star else np.exp(-np.hypot(u, v) / scale)
    image[ys, xs] += flux * profile / profile.sum()


def make_image(rng: np.random.Generator, size: int, exptime: float, n_sources: int | None = None,
               border: bool = True, crowded: bool = False) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Return (image in e-/s, true pixel sigma, source catalog)."""
    back = SKY + (UNRESOLVED if crowded else 0.0)
    sigma = noise_sigma(exptime, back)
    image = np.full((size, size), back, dtype=np.float64)
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
        _add_source(image, xx, yy, x0, y0, flux, scale, q, theta, star)
        rows.append({"x": x0, "y": y0, "flux": flux, "star": star, "scale": scale, "q": q})
    if crowded:  # a resolved stellar population: one faint star per 10 pixels
        for k in range(size * size // 10):
            x0, y0 = rng.uniform(0, size, 2)
            flux = float(np.exp(rng.uniform(np.log(0.01), np.log(3.0))))
            _add_source(image, xx, yy, x0, y0, flux, 1.0, 1.0, 0.0, True)
            rows.append({"x": x0, "y": y0, "flux": flux, "star": True, "scale": 1.0, "q": 1.0})
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


def is_crowded(index: int) -> bool:
    """Mock image `index` is a crowded field: one image per field in every fourth round, so every split has some."""
    return (index // N_FIELDS) % 4 == 2


def add_dead_patch(rng: np.random.Generator, image: np.ndarray, weight: np.ndarray, sigma: float):
    """A disc of near-zero drizzle weight filled with strong noise, like the WFC3/IR "death star"."""
    size = image.shape[0]
    radius = size / 12
    cy, cx = rng.uniform(radius + 2, size - radius - 2, 2)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
    image, weight = image.copy(), weight.copy()
    image[disc] = rng.normal(0.0, 20 * sigma, int(disc.sum()))
    weight[disc] = 0.02 * weight.max()
    return image, weight


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
        image, sigma, _ = make_image(rng, m.mock_image_size, exptime, crowded=is_crowded(i))
        weight = np.where(np.isfinite(image) & (image != 0), exptime, 0.0).astype(np.float32)
        if i % 2 == 0:  # a WFC3/IR "death star": almost no weight, noise-amplified junk in the image
            image, weight = add_dead_patch(rng, image, weight, sigma)
        image_id = f"MOCK{i:04d}"
        path = originals / f"{image_id.lower()}_{m.product_suffix}"
        write_image(path, image, cfg.dataset.hdu,
                    {"EXPTIME": exptime, "PHOTFLAM": PHOTFLAM, "PHOTPLAM": PHOTPLAM, "FILTER": "F160W"},
                    weight=weight, weight_hdu=cfg.dataset.weight_hdu)
        rows.append({
            "id": image_id, "exposure": exptime, "ra": ra, "dec": dec, "aperture": "IR",
            "pi": PI_NAMES[(i // N_FIELDS) % len(PI_NAMES)], "target": f"FIELD-{field}",
            "url": f"mock://{image_id}", "path": f"originals/{path.name}",
        })
    return pd.DataFrame(rows)
