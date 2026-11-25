"""(noisy, clean) pairs for training, validation and evaluation, all from one code path."""
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.config import Config
from astrounets.data.fits_io import CropStore
from astrounets.data.noise import add_noise, max_ratio, sample_ratio, usable
from astrounets.data.scaling import Scaler
from astrounets.files import write_parquet


def noisy_pair(store: CropStore, row: dict, ratio: float, seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) crop in e-/s; identical for identical (row, ratio, seed)."""
    clean = store.crop(row["path"], int(row["y"]), int(row["x"]), size)
    noisy = add_noise(clean, row["sigma"], ratio, np.random.default_rng(seed))
    return clean, noisy


def augment(rng: np.random.Generator, *arrays: np.ndarray) -> list[np.ndarray]:
    """Same random 90-degree rotation and flips for every array."""
    k = int(rng.integers(4))
    flip_lr, flip_ud = rng.random() < 0.5, rng.random() < 0.5
    out = []
    for a in arrays:
        a = np.rot90(a, k)
        if flip_lr:
            a = a[:, ::-1]
        if flip_ud:
            a = a[::-1, :]
        out.append(np.ascontiguousarray(a))
    return out


def scaled(scaler: Scaler, clean: np.ndarray, noisy: np.ndarray):
    """(x, y, params) with both images scaled by parameters fitted on the noisy one; None if degenerate."""
    params = scaler.fit(noisy)
    if params is None:
        return None
    return scaler.forward(noisy, params)[..., None], scaler.forward(clean, params)[..., None], params


def training_stream(rows: pd.DataFrame, store: CropStore, cfg: Config, start_epoch: int):
    """Endless stream of scaled (x, y) training pairs; epoch k uses seed (cfg.seed, k)."""
    scaler = Scaler(cfg.scaling.mode, cfg.scaling.log_clip_max)
    records = rows.to_dict("records")
    size, block = cfg.dataset.patch_size, 4 * cfg.train.batch_size

    def make(item):
        index, ratio, seed = item
        rng = np.random.default_rng(seed)
        clean, noisy = noisy_pair(store, records[index], ratio, seed, size)
        if cfg.train.augment:
            clean, noisy = augment(rng, clean, noisy)
        return scaled(scaler, clean, noisy)

    with ThreadPoolExecutor(max_workers=cfg.train.loader_workers) as pool:
        for epoch in itertools.count(start_epoch):
            rng = np.random.default_rng([cfg.seed, epoch])
            n = cfg.train.samples_per_epoch
            index = rng.integers(0, len(records), n)
            ratios = [sample_ratio(rng, records[i]["exptime"], cfg.noise) for i in index]
            seeds = rng.integers(0, 2**63 - 1, n)
            items = list(zip(index, ratios, seeds))
            for start in range(0, n, block):
                for result in pool.map(make, items[start:start + block]):
                    if result is not None:
                        yield result[0], result[1]


def fixed_set(rows: pd.DataFrame, store: CropStore, cfg: Config, n: int, seed: int) -> dict:
    """A deterministic set of scaled pairs (validation): same arrays every time it is built."""
    scaler = Scaler(cfg.scaling.mode, cfg.scaling.log_clip_max)
    rng = np.random.default_rng(seed)
    records = rows.to_dict("records")
    xs, ys, params, cleans, noisys = [], [], [], [], []
    tries = 0
    while len(xs) < n and tries < 10 * n:
        tries += 1
        row = records[int(rng.integers(len(records)))]
        ratio = sample_ratio(rng, row["exptime"], cfg.noise)
        clean, noisy = noisy_pair(store, row, ratio, int(rng.integers(2**63 - 1)), cfg.dataset.patch_size)
        result = scaled(scaler, clean, noisy)
        if result is None:
            continue
        xs.append(result[0]); ys.append(result[1]); params.append(result[2])
        cleans.append(clean); noisys.append(noisy)
    if not xs:
        raise RuntimeError("could not build a single validation pair")
    return {"x": np.stack(xs), "y": np.stack(ys), "params": params, "clean": np.stack(cleans), "noisy": np.stack(noisys)}


def eval_table(crops: pd.DataFrame, cfg: Config, split: str, dataset_dir: Path) -> pd.DataFrame:
    """Frozen evaluation set: n_crops crops x eval ratios, each with a fixed noise seed.

    Stored as data/<dataset>/eval_<split>.parquet and rebuilt only when its settings change, so every
    model evaluated on a split sees byte-identical noisy inputs.
    """
    e = cfg.eval
    path = dataset_dir / f"eval_{split}.parquet"
    settings = f"seed={cfg.seed};n={e.n_crops};ratios={list(map(float, e.ratios))};noise={cfg.noise}"
    if path.exists():
        table = pd.read_parquet(path)
        if len(table) and table["settings"].iloc[0] == settings:
            return table
    pool = usable(crops[crops["split"] == split], cfg.noise)
    if pool.empty:
        raise RuntimeError(f"no usable crops in split '{split}'")
    rng = np.random.default_rng([cfg.seed, 7])
    chosen = pool.sample(n=min(e.n_crops, len(pool)), random_state=int(rng.integers(2**31))).reset_index(drop=True)
    rows = []
    for _, crop in chosen.iterrows():
        limit = float(max_ratio(crop["exptime"], cfg.noise))
        for ratio in e.ratios:
            if ratio <= limit:
                rows.append({**crop.to_dict(), "ratio": float(ratio), "noise_seed": int(rng.integers(2**63 - 1))})
    table = pd.DataFrame(rows)
    table.insert(0, "sample_id", np.arange(len(table)))
    table["settings"] = settings
    write_parquet(table, path)  # parallel jobs may build the same (identical) table at the same time
    return table
