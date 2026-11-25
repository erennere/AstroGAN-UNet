"""Stage `dataset`: select images, measure the pixel noise, and list every usable crop.

No crop files are written. Output is data/<dataset.name>/crops.parquet with one row per crop
(image path relative to data/mast, position, split, sky tile, exposure, photometric keywords, pixel sigma, sky) and
summary.json. Crops are read straight from the original FITS files when needed.

The pixel noise sigma is the median of Background2D's background_rms map inside the crop, i.e. the
sigma-clipped spread of the pixels around the local background. (The previous code took the spread
of the smooth background model itself, which is about 70x too small.)
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
from astrounets.data.fits_io import header_info, read_image

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


def image_crops(row: dict, d: Dataset) -> tuple[list[dict], str | None]:
    """All usable crops of one image. Returns (rows, reason the image was skipped or None)."""
    info = header_info(row["abspath"], d.hdu)
    if str(info["filter"]).strip().upper() != d.filter.upper():
        return [], f"filter {info['filter']}"
    if not info["exptime"] or not info["photflam"] or not info["photplam"]:
        return [], "missing EXPTIME/PHOTFLAM/PHOTPLAM"
    image = read_image(row["abspath"], d.hdu)
    maps = measure_background(image, d)
    if maps is None:
        return [], "no background"
    bkg, rms = maps
    valid = np.isfinite(image) & (image != 0)
    ps = d.patch_size
    out = []
    for y in range(0, image.shape[0] - ps + 1, ps):
        for x in range(0, image.shape[1] - ps + 1, ps):
            ok = valid[y:y + ps, x:x + ps]
            fraction = float(ok.mean())
            if fraction < d.min_valid_fraction:
                continue
            sigma = float(np.median(rms[y:y + ps, x:x + ps][ok]))
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            out.append({
                "image_id": row["id"], "path": row["path"], "y": y, "x": x,
                "split": row["split"], "tile": int(row["tile"]), "pi": row["pi"],
                "exptime": float(info["exptime"]), "photflam": float(info["photflam"]),
                "photplam": float(info["photplam"]), "sigma": sigma,
                "sky": float(np.median(bkg[y:y + ps, x:x + ps][ok])), "valid_fraction": fraction,
            })
    return out, None


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
