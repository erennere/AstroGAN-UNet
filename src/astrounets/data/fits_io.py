"""FITS reading: header values and memory-mapped crop windows."""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from astropy.io import fits

HEADER_KEYS = ("EXPTIME", "PHOTFLAM", "PHOTPLAM", "FILTER", "NDRIZIM")


def header_info(path: str | Path, hdu: str) -> dict:
    """EXPTIME, PHOTFLAM, PHOTPLAM, FILTER and NDRIZIM (exposures combined; None if absent), taken from
    the science HDU or else the primary HDU."""
    with fits.open(path, memmap=True) as hdul:
        headers = [hdul[hdu].header, hdul[0].header]
        info = {}
        for key in HEADER_KEYS:
            value = next((h[key] for h in headers if key in h), None)
            info[key.lower()] = value
    return info


def read_image(path: str | Path, hdu: str) -> np.ndarray:
    """Whole science image as native float32 (NaN and inf kept)."""
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[hdu].data, dtype=np.float32)


def read_weight(path: str | Path, hdu: str) -> np.ndarray | None:
    """Whole drizzle weight map as float32, or None if the file has no such extension."""
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[hdu].data, dtype=np.float32) if hdu in hdul else None


def write_image(path: str | Path, data: np.ndarray, hdu: str, header: dict,
                weight: np.ndarray | None = None, weight_hdu: str = "WHT") -> None:
    """Write a FITS file shaped like an HST drizzled product: empty primary + science (+ weight) HDU."""
    primary = fits.PrimaryHDU()
    for key in ("EXPTIME", "FILTER"):
        if key in header:
            primary.header[key] = header[key]
    image = fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name=hdu)
    for key, value in header.items():
        image.header[key] = value
    hdus = [primary, image]
    if weight is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(weight, dtype=np.float32), name=weight_hdu))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def clean(window: np.ndarray) -> np.ndarray:
    """Empty and non-finite pixels set to 0 (the drizzle convention for no data)."""
    out = np.array(window, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out


class CropStore:
    """Read square windows from many FITS files, keeping a small set of them memory-mapped.

    Thread-safe; only the requested rows are read from disk. With a weight cut, pixels whose drizzle
    weight is below it are set to 0 (no data), like empty and non-finite pixels.
    """

    def __init__(self, base_dir: str | Path, hdu: str, weight_hdu: str = "WHT", max_open: int = 32):
        self.base_dir = Path(base_dir)
        self.hdu, self.weight_hdu = hdu, weight_hdu
        self.max_open = max_open
        self._open: OrderedDict[str, tuple[fits.HDUList, np.ndarray, np.ndarray | None]] = OrderedDict()
        self._lock = threading.Lock()

    def _arrays(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        with self._lock:
            if path in self._open:
                self._open.move_to_end(path)
                return self._open[path][1:]
            hdul = fits.open(path, memmap=True)
            data = hdul[self.hdu].data
            weight = hdul[self.weight_hdu].data if self.weight_hdu in hdul else None
            self._open[path] = (hdul, data, weight)
            while len(self._open) > self.max_open:
                # Dropped, not closed: another thread may still be slicing it; GC closes the file.
                self._open.popitem(last=False)
            return data, weight

    def crop(self, path: str, y: int, x: int, size: int, weight_cut: float = 0.0) -> np.ndarray:
        """Window of `size` pixels at (y, x); `path` is relative to base_dir (as stored in crops.parquet)."""
        data, weight = self._arrays(str(self.base_dir / path))
        window = clean(data[y:y + size, x:x + size])
        if weight_cut > 0:
            if weight is None:
                raise ValueError(f"{path} has no {self.weight_hdu} extension to apply the weight cut")
            window[np.asarray(weight[y:y + size, x:x + size]) < weight_cut] = 0.0
        return window

    def close(self) -> None:
        with self._lock:
            for hdul, *_ in self._open.values():
                hdul.close()
            self._open.clear()
