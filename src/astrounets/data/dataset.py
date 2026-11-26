"""Stage `dataset`: select images, measure the pixel noise, and list every usable crop.

No crop files are written. Output is data/<dataset.name>/crops.parquet with one row per crop
(image path relative to data/mast, position, split, sky tile, exposure, photometric keywords, pixel
sigma, sky, crowding) and summary.json. Crops are read straight from the original FITS files when needed.

Pixel noise of a crop (README, "Noise"):
- measured: the robust spread (1.4826 x MAD) of the crop's empty-sky pixels around the local
  background. Empty sky = valid pixels outside the source mask (pixels `source_nsigma` above the
  background after 2-px Gaussian smoothing, grown by `source_grow` pixels).
- expected: photon + dark + read noise of the drizzled pixel,
  sigma_exp^2 = (B + dark) / t + n * read_noise^2 / t^2, with B the local background (sky plus
  unresolved light, e-/s), t the pixel's exposure (drizzle weight, EXP weighting) and n the exposures
  per pixel. Drizzling correlates neighbouring pixels, which lowers the per-pixel spread by a factor
  k < 1: k is the median of measured / expected over the dataset's crops (iteratively clipped).
- used: the measured value, unless the crop has too little empty sky or its measured noise exceeds
  max_noise_excess x k x expected. Then the spread is structure (unresolved stars, nebulae, the body of
  a galaxy), not noise, and k x expected is used instead.
A crop is "crowded" if less than `crowded_sky_fraction` of it is empty sky or its noise came from the
expectation. (The first version took the spread of the smooth background model, about 70x too small;
the second took the clipped spread of all pixels, up to 15x too large in crowded fields.)
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.config import Config, Dataset, dataset_key, with_dataset
from astrounets.files import write_json, write_parquet
from astrounets.data.fits_io import header_info, read_image, read_weight

log = logging.getLogger(__name__)


def select_images(meta: pd.DataFrame, d: Dataset, mast_dir: Path, seed: int) -> pd.DataFrame:
    """Rows of the MAST table that belong to this dataset and have a downloaded file."""
    rows = meta[meta["path"].notna() & meta["split"].isin(["train", "val", "test"])].copy()
    rows["abspath"] = [str(mast_dir / p) for p in rows["path"]]
    rows = rows[[Path(p).is_file() for p in rows["abspath"]]]
    if d.apertures:
        rows = rows[rows["aperture"].isin(d.apertures)]
    if d.pi_names:
        wanted = {name.upper() for name in d.pi_names}
        rows = rows[rows["pi"].astype(str).str.upper().isin(wanted)]
    rows = rows[(rows["exposure"] >= d.exposure_min) & (rows["exposure"] <= d.exposure_max)]
    if d.max_images is not None and len(rows) > d.max_images:
        rows = rows.sample(n=d.max_images, random_state=seed)
    return rows.sort_values("id").reset_index(drop=True)


def measure_background(image: np.ndarray, d: Dataset) -> tuple[np.ndarray, np.ndarray] | None:
    """(background, background_rms) maps of a whole image; empty pixels are masked."""
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground, StdBackgroundRMS

    invalid = ~np.isfinite(image) | (image == 0)
    if invalid.all():
        return None
    data = np.where(invalid, 0.0, image)
    try:
        bkg = Background2D(
            data, d.bkg_box_size, filter_size=d.bkg_filter_size, mask=invalid,
            sigma_clip=SigmaClip(sigma=d.bkg_sigma), bkg_estimator=MedianBackground(),
            bkg_rms_estimator=StdBackgroundRMS(), exclude_percentile=d.bkg_exclude_percentile,
        )
    except ValueError as err:  # e.g. every box mostly empty
        log.warning("background estimation failed: %s", err)
        return None
    return np.asarray(bkg.background, np.float32), np.asarray(bkg.background_rms, np.float32)


MIN_SKY_PIXELS = 1000  # fewer empty-sky pixels in a crop: its noise is not measured


def source_mask(image: np.ndarray, back: np.ndarray, rms: np.ndarray, invalid: np.ndarray, d: Dataset) -> np.ndarray:
    """Pixels belonging to sources: d.source_nsigma x rms above the background after 2-px Gaussian
    smoothing (at least 5 connected pixels), grown by d.source_grow pixels."""
    from astropy.convolution import convolve
    from photutils.segmentation import detect_sources, make_2dgaussian_kernel
    from photutils.utils import circular_footprint

    residual = np.where(invalid, 0.0, image - back)
    smooth = convolve(residual, make_2dgaussian_kernel(2.0, size=5), normalize_kernel=True)
    segments = detect_sources(smooth, d.source_nsigma * rms, n_pixels=5, mask=invalid)
    if segments is None:
        return np.zeros(image.shape, bool)
    if d.source_grow == 0:
        return segments.data > 0
    return segments.make_source_mask(footprint=circular_footprint(d.source_grow))


def expected_sigma(back: float, exposure: float, n_exposures: float, d: Dataset) -> float:
    """Photon + dark + read noise (e-/s) of a drizzled pixel, before drizzle correlation."""
    if not (exposure > 0) or not np.isfinite(back):
        return float("nan")
    return float(np.sqrt((max(back, 0.0) + d.dark_current) / exposure + n_exposures * d.read_noise ** 2 / exposure ** 2))


def image_crops(row: dict, d: Dataset) -> tuple[list[dict], str | None]:
    """All crops of one image with their measured and expected noise (the noise used, and whether a
    crop is crowded, is decided per dataset in assign_noise). Returns (rows, reason skipped or None)."""
    info = header_info(row["abspath"], d.hdu)
    if str(info["filter"]).strip().upper() != d.filter.upper():
        return [], f"filter {info['filter']}"
    if not info["exptime"] or not info["photflam"] or not info["photplam"]:
        return [], "missing EXPTIME/PHOTFLAM/PHOTPLAM"
    image = read_image(row["abspath"], d.hdu)
    weight = read_weight(row["abspath"], d.weight_hdu)
    if weight is None and d.min_weight_fraction > 0:
        return [], f"no {d.weight_hdu} extension"
    weight_cut = 0.0
    if weight is not None and d.min_weight_fraction > 0:
        covered = weight[weight > 0]
        if covered.size == 0:
            return [], "zero weight everywhere"
        weight_cut = float(d.min_weight_fraction * np.median(covered))
        image = np.where(weight < weight_cut, 0.0, image).astype(np.float32)  # low weight = no data
    maps = measure_background(image, d)
    if maps is None:
        return [], "no background"
    bkg, rms = maps
    valid = np.isfinite(image) & (image != 0)
    sources = source_mask(np.where(valid, image, 0.0), bkg, rms, ~valid, d)
    residual = image - bkg
    exptime = float(info["exptime"])
    n_total = float(info.get("ndrizim") or 1)
    ps = d.patch_size
    out = []
    for y in range(0, image.shape[0] - ps + 1, ps):
        for x in range(0, image.shape[1] - ps + 1, ps):
            box = (slice(y, y + ps), slice(x, x + ps))
            ok = valid[box]
            fraction = float(ok.mean())
            if fraction < d.min_valid_fraction:
                continue
            sky = ok & ~sources[box]
            values = residual[box][sky]
            measured = (float(1.4826 * np.median(np.abs(values - np.median(values))))
                        if values.size >= MIN_SKY_PIXELS else float("nan"))
            back = float(np.median(bkg[box][ok]))
            # exposure of the crop's pixels: the drizzle weight (EXP weighting), never more than the
            # header total (which for a mosaic counts every pointing)
            exposure = min(float(np.median(weight[box][ok])), exptime) if weight is not None else exptime
            n = max(1.0, n_total * exposure / exptime)
            out.append({
                "image_id": row["id"], "path": row["path"], "y": y, "x": x,
                "split": row["split"], "tile": int(row["tile"]), "pi": row["pi"],
                "exptime": exposure, "exptime_header": exptime, "photflam": float(info["photflam"]),
                "photplam": float(info["photplam"]), "sigma_measured": measured,
                "sigma_expected": expected_sigma(back, exposure, n, d), "sky": back,
                "sky_fraction": float(sky.sum() / ok.sum()), "valid_fraction": fraction,
                "weight_cut": weight_cut,
            })
    return out, None


def noise_scale(measured: np.ndarray, expected: np.ndarray, max_excess: float) -> float:
    """k: median of measured / expected over crops whose noise is plain noise (iteratively clipped at
    max_excess x k, so crowded fields do not pull it up)."""
    ratio = np.asarray(measured, float) / np.asarray(expected, float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        raise RuntimeError("no crop has enough empty sky to measure the noise")
    k = float(np.median(ratio))
    for _ in range(20):
        new = float(np.median(ratio[ratio <= max_excess * k]))
        if abs(new - k) < 1e-6 * k:
            break
        k = new
    return k


def assign_noise(table: pd.DataFrame, d: Dataset) -> tuple[pd.DataFrame, float]:
    """Choose each crop's noise (measured or k x expected) and mark crowded crops; drops crops
    without any usable noise value. Returns (table, k)."""
    k = noise_scale(table["sigma_measured"], table["sigma_expected"], d.max_noise_excess)
    measured, expected = table["sigma_measured"].to_numpy(float), table["sigma_expected"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        plain = np.isfinite(measured) & (~np.isfinite(expected) | (measured <= d.max_noise_excess * k * expected))
    table = table.assign(noise_measured=plain, sigma=np.where(plain, measured, k * expected))
    table["crowded"] = (table["sky_fraction"] < d.crowded_sky_fraction) | ~table["noise_measured"]
    table = table[np.isfinite(table["sigma"]) & (table["sigma"] > 0)].reset_index(drop=True)
    return table, k


def _image_crops_job(args):
    return image_crops(*args)


def build_dataset(cfg: Config) -> Path:
    """Run the dataset stage; returns the crops table path."""
    d = cfg.dataset
    meta_path = cfg.mast_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run `astrounets fetch` first")
    images = select_images(pd.read_parquet(meta_path), d, cfg.mast_dir, cfg.seed)
    if images.empty:
        raise RuntimeError("no downloaded image matches the dataset filters")
    jobs = [(row, d) for row in images.to_dict("records")]
    if d.workers > 1:
        with ProcessPoolExecutor(max_workers=d.workers) as pool:
            results = list(pool.map(_image_crops_job, jobs, chunksize=4))
    else:
        results = [image_crops(*job) for job in jobs]
    crops = [c for rows, _ in results for c in rows]
    skipped: dict[str, int] = {}
    for _, reason in results:
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
    table = pd.DataFrame(crops)
    if table.empty:
        raise RuntimeError(f"no usable crops; skipped images: {skipped}")
    table, k = assign_noise(table, d)
    if table.empty:
        raise RuntimeError("no crop has a usable noise estimate")
    table.insert(0, "crop_id", np.arange(len(table)))
    counts = table["split"].value_counts().to_dict()
    missing = [s for s in ("train", "val", "test") if counts.get(s, 0) == 0]
    if missing:
        raise RuntimeError(f"dataset '{d.name}' has no crops in split(s) {missing}: its images cover too few sky tiles")
    out_dir = cfg.dataset_dir()
    summary = {
        "dataset": d.name, "settings": dataset_key(cfg), "patch_size": d.patch_size, "images_selected": len(images), "images_used": int(table["image_id"].nunique()),
        "skipped_images": skipped, "crops": int(len(table)),
        "crops_per_split": {s: int(counts.get(s, 0)) for s in ("train", "val", "test")},
        "tiles_per_split": {s: int(table.loc[table["split"] == s, "tile"].nunique()) for s in ("train", "val", "test")},
        "sigma_median": float(table["sigma"].median()),
        "noise_scale_k": k,
        "noise_from_expectation": int((~table["noise_measured"]).sum()),
        "crowded_per_split": {s: int(table.loc[table["split"] == s, "crowded"].sum()) for s in ("train", "val", "test")},
    }
    for stale in out_dir.glob("eval_*.parquet"):  # frozen evaluation sets of an earlier build
        stale.unlink()
    write_json(out_dir / "summary.json", summary)
    write_parquet(table, out_dir / "crops.parquet")  # written last: its presence means "built"
    log.info("dataset %s: %s", d.name, summary)
    return out_dir / "crops.parquet"


def load_crops(cfg: Config, name: str | None = None) -> pd.DataFrame:
    """The crop table of dataset `name` (default: the config's), checked against the config's settings."""
    target = with_dataset(cfg, name or cfg.dataset.name)
    path = target.dataset_dir() / "crops.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `astrounets dataset --set dataset.name={target.dataset.name}` first")
    built = json.loads((path.parent / "summary.json").read_text(encoding="utf-8")).get("settings")
    if built != dataset_key(target):
        raise RuntimeError(f"{path.parent} was built with other dataset settings than this config; "
                           f"delete the folder and build it again")
    return pd.read_parquet(path)
