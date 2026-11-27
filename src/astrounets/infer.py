"""A trained model on real images: `astrounets denoise` and `TrainedModel`.

A model folder is a run folder (checkpoints/best.keras) or the report's copy of one
(runs/_workflow/report/models/<dataset>/<run>/best.keras); both hold the run's config.yaml. Only the
settings the model depends on are read from it (patch size, scaling, the no-data and background
settings of its dataset), so a model keeps loading when the configuration gains fields later.
`denoise.model: best` is the study's final model (workflow.final).

An image of any size is reconstructed tile by tile, exactly as the evaluation reconstructs a crop:
1. no data = empty (0) or non-finite pixels, and pixels whose drizzle weight is below the training
   cut (dataset.min_weight_fraction x the image's median weight; only if the file has a weight map);
2. the background map of the dataset stage (same Background2D settings) gives each tile its sky
   level, which fills the tile's no-data pixels (no noise), as in training;
3. the image is mirrored outward by overlap / 2 pixels (and up to one patch if it is smaller), so its
   border never lies at a tile border; tiles of the patch size start every patch - overlap pixels,
   the last one flush with the edge;
4. each tile is scaled with parameters fitted on itself, passed through the model and scaled back to
   e-/s (a constant tile is passed through unchanged); with tta, averaged over the 8 rotations and
   flips the training augmented with;
5. tiles are blended with a weight that is flat in the middle and falls off over the overlap
   (cosine taper), normalised by the summed weights;
6. no-data pixels get their input values back.
With an identity model this returns the input to float precision, whatever the image size.

Input in e-/s as in training; images in electrons (BUNIT) are divided by EXPTIME for the model and the
reconstruction is written in the input's units. Scaling none and log_min_max depend on the units,
z_scale and min_max do not.
"""
from __future__ import annotations

import glob
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrounets.config import Config, read_yaml
from astrounets.data.pairs import fill_no_data, no_data
from astrounets.data.scaling import Scaler

log = logging.getLogger(__name__)

FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
REC_SUFFIX = "_rec"
PIXEL_SCALE_TOLERANCE = 0.05  # relative difference from the training pixel scale that triggers a warning


# ------------------------------------------------------------------------------------ the model
def _model_files(folder: Path) -> tuple[Path, Path, Path]:
    """(best.keras, best.json, config.yaml) of a run folder, a report copy or a .keras file."""
    folder = Path(folder)
    if folder.is_file() and folder.suffix == ".keras":
        keras_file = folder
        folder = folder.parent.parent if folder.parent.name == "checkpoints" else folder.parent
    elif (folder / "checkpoints" / "best.keras").exists():
        keras_file = folder / "checkpoints" / "best.keras"
    elif (folder / "best.keras").exists():
        keras_file = folder / "best.keras"
    else:
        raise FileNotFoundError(f"{folder} holds no trained model (checkpoints/best.keras or best.keras)")
    config = folder / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(f"{config} not found: a model folder needs the config it was trained with")
    return keras_file, keras_file.with_name("best.json"), config


def _setting(data: dict, dotted: str, source: Path):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{source} has no {dotted}; was the model trained with an older astrounets?")
        node = node[part]
    return node


@dataclass
class TrainedModel:
    folder: Path
    name: str                    # <training dataset>/<run folder>
    model: object                # the Keras generator
    patch_size: int
    scaler: Scaler
    data: SimpleNamespace        # hdu, weight_hdu, min_weight_fraction, bkg_* and filter of its dataset
    instrument: str | None
    epoch: int | None

    @classmethod
    def load(cls, folder: str | Path) -> "TrainedModel":
        import keras

        keras_file, best_json, config = _model_files(Path(folder))
        data = read_yaml(config)
        keys = ("hdu", "weight_hdu", "min_weight_fraction", "bkg_box_size", "bkg_filter_size", "bkg_sigma",
                "bkg_exclude_percentile", "filter")
        dataset = SimpleNamespace(**{k: _setting(data, f"dataset.{k}", config) for k in keys})
        scaler = Scaler(_setting(data, "scaling.mode", config), float(_setting(data, "scaling.log_clip_max", config)))
        query = (data.get("mast") or {}).get("query") or {}
        epoch = json.loads(best_json.read_text(encoding="utf-8")).get("epoch") if best_json.exists() else None
        model = keras.models.load_model(keras_file, compile=False)
        run_folder = config.parent
        return cls(run_folder, f"{run_folder.parent.name}/{run_folder.name}", model,
                   int(_setting(data, "dataset.patch_size", config)), scaler, dataset,
                   query.get("sci_instrume"), epoch)

    # ---------------------------------------------------------------------------- crops
    def reconstruct(self, noisy: np.ndarray, batch_size: int = 16, tta: bool = False,
                    valid: np.ndarray | None = None) -> list[np.ndarray | None]:
        """Reconstructions (e-/s) of a stack of crops (n, patch, patch) in e-/s whose no-data pixels are
        already filled (`valid`: the pixels with data, for robust_z); None for a crop the scaling cannot
        handle (constant). The one code path of evaluation and denoising."""
        params = [self.scaler.fit(crop, None if valid is None else valid[i]) for i, crop in enumerate(noisy)]
        index = [i for i, p in enumerate(params) if p is not None]
        out: list[np.ndarray | None] = [None] * len(noisy)
        for start in range(0, len(index), batch_size):
            chunk = index[start:start + batch_size]
            x = np.stack([self.scaler.forward(noisy[i], params[i]) for i in chunk])[..., None]
            if not tta:
                pred = [np.asarray(self.model(x, training=False))[..., 0]]
            else:
                pred = []
                for k in range(4):
                    for flip in (False, True):
                        t = np.rot90(x, k, axes=(1, 2))
                        t = t[:, :, ::-1] if flip else t
                        p = np.asarray(self.model(np.ascontiguousarray(t), training=False))
                        p = p[:, :, ::-1] if flip else p
                        pred.append(np.rot90(p, -k, axes=(1, 2))[..., 0])
            for j, i in enumerate(chunk):
                out[i] = np.mean([self.scaler.inverse(p[j], params[i]) for p in pred], axis=0).astype(np.float32)
        return out

    # ---------------------------------------------------------------------------- images
    def no_data_mask(self, image: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
        hole = no_data(image)
        if weight is not None and self.data.min_weight_fraction > 0:
            covered = weight[np.isfinite(weight) & (weight > 0)]
            if covered.size:
                hole |= ~(weight >= self.data.min_weight_fraction * float(np.median(covered)))
        return hole

    def denoise(self, image: np.ndarray, weight: np.ndarray | None = None, overlap: int = 64,
                batch_size: int = 16, tta: bool = False) -> np.ndarray:
        """Reconstruction of a whole image (e-/s, any size); no-data pixels keep their input values."""
        from astrounets.data.dataset import measure_background

        ps = self.patch_size
        if not 0 <= 2 * overlap <= ps:
            raise ValueError(f"overlap must be between 0 and half the patch size ({ps // 2}), got {overlap}")
        image = np.asarray(image, np.float32)
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D image, got shape {image.shape}")
        hole = self.no_data_mask(image, weight)
        if hole.all():
            log.warning("the image has no valid pixels; returned unchanged")
            return image.copy()
        data = np.where(hole, 0.0, image).astype(np.float32)
        maps = measure_background(data, self.data)
        if maps is None:  # too small or too empty for the background mesh: one level for the whole image
            log.warning("background map failed; every tile uses the image's median as its sky level")
            back = np.full(image.shape, np.median(data[~hole]), np.float32)
        else:
            back = maps[0]
        pad = overlap // 2
        height, width = image.shape
        widths = ((pad, pad + max(0, ps - height - 2 * pad)), (pad, pad + max(0, ps - width - 2 * pad)))
        data, hole_p, back = (np.pad(a, widths, mode="reflect") for a in (data, hole, back))
        weights = blend_window(ps, overlap)
        total = np.zeros(data.shape, np.float32)
        norm = np.zeros(data.shape, np.float32)
        stride = ps - overlap
        places = [(y, x) for y in tile_starts(data.shape[0], ps, stride) for x in tile_starts(data.shape[1], ps, stride)]
        for start in range(0, len(places), 4 * batch_size):
            tiles, valid, kept = [], [], []
            for y, x in places[start:start + 4 * batch_size]:
                h = hole_p[y:y + ps, x:x + ps]
                if h.all():
                    continue
                sky = float(np.median(back[y:y + ps, x:x + ps][~h]))
                tiles.append(fill_no_data(data[y:y + ps, x:x + ps], sky, h)[0])
                valid.append(~h)
                kept.append((y, x))
            if not tiles:
                continue
            recs = self.reconstruct(np.stack(tiles), batch_size, tta, np.stack(valid))
            for (y, x), tile, rec in zip(kept, tiles, recs):
                total[y:y + ps, x:x + ps] += (tile if rec is None else rec) * weights
                norm[y:y + ps, x:x + ps] += weights
        out = np.divide(total, norm, out=data.copy(), where=norm > 0)[pad:pad + height, pad:pad + width]
        out[hole] = image[hole]
        return out


def tile_starts(n: int, size: int, stride: int) -> list[int]:
    """Start positions of tiles of `size` along an axis of n >= size pixels: every `stride` pixels,
    the last tile flush with the end."""
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


def blend_window(size: int, overlap: int) -> np.ndarray:
    """2-D tile weight: 1 in the middle, a cosine taper over `overlap` pixels at each edge (> 0 everywhere).
    Neighbouring tiles `size - overlap` apart sum to 1 in their overlap."""
    w = np.ones(size, np.float32)
    if overlap > 0:
        taper = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
        w[:overlap] = taper
        w[size - overlap:] = taper[::-1]
    return np.outer(w, w).astype(np.float32)


# -------------------------------------------------------------------------------------- files
def resolve_model(cfg: Config) -> Path:
    """denoise.model as a folder: `best` is the study's final model."""
    if cfg.denoise.model.strip().lower() == "best":
        from astrounets.workflow import final_model

        return final_model(cfg)
    return cfg.path(cfg.denoise.model)


def find_images(items: list[str]) -> list[Path]:
    """FITS files from files, folders (every FITS file in it, earlier reconstructions excepted) and
    patterns (expanded here, so they work in any shell)."""
    out: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            found = [p for p in sorted(path.iterdir()) if p.is_file() and p.name.lower().endswith(FITS_SUFFIXES)
                     and not _stem(p).endswith(REC_SUFFIX)]
            if not found:
                raise FileNotFoundError(f"no FITS files in {path}")
            out += found
        elif path.is_file():
            out.append(path)
        else:
            matches = sorted(Path(p) for p in glob.glob(item) if Path(p).is_file())
            if not matches:
                raise FileNotFoundError(f"{item}: no such file")
            out += matches
    return list(dict.fromkeys(p.resolve() for p in out))


def _stem(path: Path) -> str:
    name = path.name
    for suffix in FITS_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def read_fits(path: Path, hdu: str, weight_hdu: str) -> dict:
    """The science image (the named HDU, else the first 2-D image), its weight map if the file has one
    of the same shape, and the headers."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        index = None
        if hdu in hdul and hdul[hdu].data is not None:
            index = hdul.index_of(hdu)
        else:
            for i, h in enumerate(hdul):
                if h.data is not None and h.is_image and np.squeeze(h.data).ndim == 2:
                    index = i
                    break
        if index is None:
            raise ValueError(f"{path}: no 2-D image found")
        image = np.squeeze(np.asarray(hdul[index].data, np.float32))
        if image.ndim != 2:
            raise ValueError(f"{path}: HDU {index} is not a 2-D image (shape {hdul[index].data.shape})")
        weight = None
        if weight_hdu in hdul and hdul[weight_hdu].data is not None:
            w = np.squeeze(np.asarray(hdul[weight_hdu].data, np.float32))
            weight = w if w.shape == image.shape else None
        return {"image": image, "weight": weight, "index": index, "shape": hdul[index].data.shape,
                "header": hdul[index].header.copy(), "primary": hdul[0].header.copy()}


def _keyword(f: dict, key: str):
    return f["header"].get(key, f["primary"].get(key))


def to_electrons_per_second(f: dict) -> tuple[float, list[str]]:
    """(factor that turns the image into e-/s, warnings), from BUNIT and EXPTIME."""
    unit = str(_keyword(f, "BUNIT") or "").strip().upper().replace(" ", "")
    if unit in ("ELECTRONS/S", "ELECTRON/S", "E-/S", "E/S", "ELECTRONS/SEC"):
        return 1.0, []
    if unit in ("ELECTRONS", "ELECTRON", "E-", "E"):
        exptime = _keyword(f, "EXPTIME")
        if exptime and float(exptime) > 0:
            return 1.0 / float(exptime), []
        return 1.0, [f"BUNIT {unit} without a positive EXPTIME: used as e-/s"]
    return 1.0, [f"BUNIT '{unit or 'missing'}' is not e-/s: used as e-/s as it is (the model was trained on e-/s)"]


def domain_warnings(m: TrainedModel, f: dict) -> list[str]:
    """Where the image differs from what the model was trained on."""
    from astrounets.plots import PIXEL_SCALE

    out = []
    filt = str(_keyword(f, "FILTER") or "").strip()
    if filt.upper() != str(m.data.filter).upper():
        out.append(f"filter {filt or 'unknown'}, the model was trained on {m.data.filter}")
    instrument = str(_keyword(f, "INSTRUME") or "").strip()
    if m.instrument and instrument.upper() != str(m.instrument).upper():
        out.append(f"instrument {instrument or 'unknown'}, the model was trained on {m.instrument.upper()}")
    try:
        import warnings

        from astropy.wcs import WCS, FITSFixedWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(f["header"])
        if wcs.has_celestial:
            scales = [v.to_value("arcsec") if hasattr(v, "to_value") else float(v) * 3600.0
                      for v in wcs.celestial.proj_plane_pixel_scales()]
            if any(abs(v / PIXEL_SCALE - 1) > PIXEL_SCALE_TOLERANCE for v in scales):
                shown = " x ".join(f"{v:.4f}" for v in scales)
                out.append(f"pixel scale {shown}\", the model was trained on {PIXEL_SCALE}\"")
    except Exception:  # a broken WCS is no reason to stop
        pass
    return out


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("astrounets")
    except PackageNotFoundError:
        return "unknown"


def write_reconstruction(path: Path, source: Path, f: dict, rec: np.ndarray, m: TrainedModel, overlap: int,
                         tta: bool) -> Path:
    """A copy of the input file with the reconstruction in place of the science image: every other HDU
    (weight, context, tables) and every header unchanged, provenance keywords added (DN*)."""
    from astropy.io import fits

    from astrounets.files import atomic

    with fits.open(source, memmap=False) as hdul:
        hdus = fits.HDUList([h.copy() for h in hdul])
    sci = hdus[f["index"]]
    for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
        sci.header.remove(key, ignore_missing=True)
    sci.data = rec.astype(np.float32).reshape(f["shape"])  # the input's shape, e.g. (1, ny, nx)
    header = sci.header
    header["DNMODEL"] = (m.name, "astrounets model")
    header["DNEPOCH"] = (m.epoch if m.epoch is not None else -1, "checkpoint epoch")
    header["DNSCALE"] = (m.scaler.mode, "scaling, fitted per tile")
    header["DNTILE"] = (m.patch_size, "tile size, pixels")
    header["DNOVERLP"] = (overlap, "tile overlap, pixels")
    header["DNTTA"] = (bool(tta), "averaged over 8 rotations/flips")
    header["DNVERS"] = (_version(), "astrounets version")
    header.add_history("astrounets denoise: reconstruction; no-data pixels keep their input values")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic(path) as tmp:
        hdus.writeto(tmp, output_verify="silentfix")
    return path


def denoise_file(m: TrainedModel, path: Path, out_dir: Path, overlap: int, batch_size: int, tta: bool,
                 figure: bool, zoom: int, centres: list[tuple[int, int]] | None = None, dpi: int = 150,
                 show: bool = False) -> list[Path]:
    """Reconstruct one FITS image: <out_dir>/<name>_rec.fits and (figure) <name>_compare.png."""
    start = time.time()
    path = Path(path)
    f = read_fits(path, m.data.hdu, m.data.weight_hdu)
    factor, warnings = to_electrons_per_second(f)
    warnings += domain_warnings(m, f)
    if f["weight"] is None and m.data.min_weight_fraction > 0:
        warnings.append(f"no {m.data.weight_hdu} weight map: only empty and non-finite pixels count as no data")
    for text in warnings:
        log.warning("%s: %s", path.name, text)
    rec = m.denoise(f["image"] * factor, f["weight"], overlap, batch_size, tta) / factor
    hole = m.no_data_mask(f["image"], f["weight"])
    rec[hole] = f["image"][hole]  # exactly the input values, also after the unit round trip
    stem = _stem(path)
    written = [write_reconstruction(out_dir / f"{stem}{REC_SUFFIX}.fits", path, f, rec, m, overlap, tta)]
    if figure or show:
        from astrounets.plots import compare_figure, show_side_by_side

        title = f"{path.name}  |  model {m.name}" + (f"  |  {'; '.join(warnings)}" if warnings else "")
        if figure:
            written.append(compare_figure(out_dir / f"{stem}_compare.png", f["image"], rec, hole, zoom,
                                          centres or [], dpi, title))
        if show:
            show_side_by_side(f["image"], rec, hole, title)
    log.info("%s: %d x %d pixels in %.1f s -> %s", path.name, *f["image"].shape, time.time() - start, written[0])
    return written


def denoise_files(cfg: Config, items: list[str], centres: list[tuple[int, int]] | None = None,
                  show: bool = False) -> list[Path]:
    """`astrounets denoise`: every image with the model and settings of cfg.denoise."""
    n = cfg.denoise
    images = find_images(items)
    folder = resolve_model(cfg)
    m = TrainedModel.load(folder)
    if 2 * n.overlap > m.patch_size:
        raise ValueError(f"denoise.overlap {n.overlap} is more than half the model's patch size {m.patch_size} "
                         "(each tile tapers over the overlap on both sides)")
    log.info("model %s (%s, %d px tiles, checkpoint epoch %s) on %d image(s)", m.name, m.scaler.mode,
             m.patch_size, m.epoch, len(images))
    out_dir = cfg.path(n.out)
    written = []
    for path in images:
        written += denoise_file(m, path, out_dir, n.overlap, n.batch_size, n.tta, n.figure, n.zoom, centres,
                                cfg.plots.dpi, show)
    return written
