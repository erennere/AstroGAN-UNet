"""Stage `fetch`: query MAST once, cut the sky into tiles, assign splits, resolve URLs, download.

Output: data/mast/metadata.parquet (one row per image, canonical column names, sky tile, split) and
data/mast/originals/*.fits. Every dataset is cut from this one table, so a tile that is in the test
split is in the test split for every dataset. Images too close to another split are marked 'buffer'
and never used.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from astrounets.config import Config
from astrounets.files import write_parquet

log = logging.getLogger(__name__)
CANONICAL = ("id", "exposure", "ra", "dec", "aperture", "pi", "target")


# ------------------------------------------------------------------------------------ MAST query
def query_metadata(cfg: Config) -> pd.DataFrame:
    """All rows matching mast.query, paged, renamed to canonical columns."""
    from astropy.table import vstack
    from astroquery.mast import MastMissions

    m = cfg.mast
    missions = MastMissions(mission=m.mission)
    pages, offset = [], 0
    while True:
        page = missions.query_criteria(limit=m.page_size, offset=offset, **m.query)
        n = 0 if page is None else len(page)
        log.info("MAST page at offset %d: %d rows", offset, n)
        if n == 0:
            break
        pages.append(page)
        offset += n
        if n < m.page_size:
            break
    if not pages:
        raise RuntimeError("MAST query returned no rows; check mast.query")
    table = vstack(pages).to_pandas()
    columns = {getattr(m.columns, key): key for key in CANONICAL}
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise RuntimeError(f"MAST result lacks column(s) {missing}; fix mast.columns")
    table = table.rename(columns=columns)[list(CANONICAL)]
    table["id"] = table["id"].astype(str).str.upper()
    return table.drop_duplicates("id").reset_index(drop=True)


def resolve_urls(ids: list[str], cfg: Config) -> dict[str, str]:
    """Dataset id -> download URL of its drizzled science product (chunked bulk queries with retries)."""
    from astroquery.mast import Observations

    m = cfg.mast
    urls: dict[str, str] = {}
    for start in range(0, len(ids), m.resolve_chunk):
        chunk = ids[start:start + m.resolve_chunk]
        for attempt in range(1, m.retries + 1):
            try:
                obs = Observations.query_criteria(obs_id=chunk)
                products = Observations.get_product_list(obs).to_pandas()
                break
            except Exception as err:  # network errors: retry, then give up loudly
                if attempt == m.retries:
                    raise RuntimeError(f"URL resolution failed for ids {chunk[:3]}...: {err}") from err
                time.sleep(2.0 * attempt)
        products["obs_id"] = products["obs_id"].astype(str).str.upper()
        name = products["productFilename"].astype(str).str.lower()
        products = products[name.str.endswith(m.product_suffix)]
        for obs_id, group in products.groupby("obs_id"):
            if obs_id in chunk:
                exact = group[group["productFilename"].str.lower() == f"{obs_id.lower()}_{m.product_suffix}"]
                chosen = exact if len(exact) else group  # prefer the association's own drizzled product
                urls[obs_id] = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + str(chosen["dataURI"].iloc[0])
        log.info("resolved %d / %d URLs", len(urls), len(ids))
    return urls


async def _download_all(jobs: list[tuple[str, Path]], workers: int, timeout: float) -> dict[str, bool]:
    import aiohttp

    semaphore = asyncio.Semaphore(workers)

    async def one(session, url: str, dest: Path) -> bool:
        if dest.exists():
            return True
        async with semaphore:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        log.warning("download %s: HTTP %d", url, response.status)
                        return False
                    part = dest.with_suffix(dest.suffix + ".part")
                    part.write_bytes(await response.read())
                    part.replace(dest)
                    return True
            except Exception as err:
                log.warning("download %s failed: %s", url, err)
                return False

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        results = await asyncio.gather(*(one(session, url, dest) for url, dest in jobs))
    return {str(dest): ok for (_, dest), ok in zip(jobs, results)}


def download(urls: dict[str, str], folder: Path, workers: int, timeout: float) -> dict[str, Path]:
    """Download in parallel (existing files are kept). Returns id -> path of files present."""
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(url, folder / url.rsplit("/", 1)[-1].rsplit(":", 1)[-1].lower()) for url in urls.values()]
    status = asyncio.run(_download_all(jobs, workers, timeout))
    present = {}
    for image_id, (_, dest) in zip(urls, jobs):
        if status[str(dest)]:
            present[image_id] = dest
    log.info("downloads present: %d / %d", len(present), len(urls))
    return present


# ------------------------------------------------------------------------------ tiles, splits
def _unit_vectors(ra, dec) -> np.ndarray:
    ra_r, dec_r = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])


def _chord(arcmin: float) -> float:
    return 2.0 * np.sin(np.radians(arcmin / 60.0) / 2.0)


def assign_tiles(ra: np.ndarray, dec: np.ndarray, tile_arcmin: float) -> np.ndarray:
    """Sky tile of each image centre: declination bands of tile_arcmin, each cut into RA cells of
    (roughly) the same angular width."""
    size = tile_arcmin / 60.0
    dec = np.asarray(dec, float)
    band = np.floor((dec + 90.0) / size).astype(np.int64)
    centre = np.radians(-90.0 + (band + 0.5) * size)
    width = size / np.maximum(np.cos(centre), 1e-3)
    cell = np.floor(np.mod(np.asarray(ra, float), 360.0) / width).astype(np.int64)
    return band * 1_000_000 + cell


def assign_splits(groups: np.ndarray, shares: dict[str, int], seed: int) -> np.ndarray:
    """Give whole groups (tiles) to train/val/test, biggest first, always to the split furthest below
    its target share. Deterministic for a given seed."""
    groups = np.asarray(groups)
    ids, counts = np.unique(groups, return_counts=True)
    if len(ids) < 3:
        raise ValueError(f"only {len(ids)} sky tile(s); need at least 3 to fill train, val and test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    order = order[np.argsort(-counts[order], kind="stable")]
    total = counts.sum()
    names = list(shares)
    target = {s: shares[s] / 100.0 * total for s in names}
    filled = {s: 0 for s in names}
    group_split = {}
    for k, idx in enumerate(order):
        empty = [s for s in names if filled[s] == 0]
        if empty and len(order) - k <= len(empty):  # make sure no split ends up without a tile
            choice = empty[0]
        else:
            choice = max(names, key=lambda s: ((target[s] - filled[s]) / target[s], target[s]))
        group_split[ids[idx]] = choice
        filled[choice] += counts[idx]
    return np.array([group_split[g] for g in groups], dtype=object)


def apply_buffer(ra, dec, split: np.ndarray, overlap_arcmin: float) -> np.ndarray:
    """Mark as 'buffer' every val image within overlap_arcmin of a train image and every test image
    within overlap_arcmin of a train or val image, so no patch of sky is in two splits."""
    split = np.asarray(split, dtype=object).copy()
    if overlap_arcmin <= 0:
        return split
    points = _unit_vectors(ra, dec)
    tree = cKDTree(points)
    original = split.copy()
    earlier = {"val": {"train"}, "test": {"train", "val"}}
    for i in np.where(np.isin(original, ["val", "test"]))[0]:
        near = tree.query_ball_point(points[i], _chord(overlap_arcmin))
        if earlier[original[i]] & set(original[near]):
            split[i] = "buffer"
    return split


def _selected(pool: pd.DataFrame, d) -> pd.Series:
    """Rows of the MAST table that dataset settings `d` would use (the same filters as select_images)."""
    keep = (pool["exposure"] >= d.exposure_min) & (pool["exposure"] <= d.exposure_max)
    if d.apertures:
        keep &= pool["aperture"].isin(d.apertures)
    if d.pi_names:
        keep &= pool["pi"].astype(str).str.upper().isin({n.upper() for n in d.pi_names})
    return keep


def choose_downloads(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rows to download. Without mast.download_limit: every image that any entry of `datasets` can use
    (images in buffer, outside the exposure range or with other apertures are never downloaded). With
    a limit: a small subset for dataset.name, spread over splits (by their shares) and over tiles."""
    from astrounets.config import with_dataset

    limit = cfg.mast.download_limit
    pool = table[table["split"].isin(["train", "val", "test"])]
    if limit is None:
        usable = pd.Series(False, index=pool.index)
        for name in cfg.datasets:
            usable |= _selected(pool, with_dataset(cfg, name).dataset)
        return pool[usable]
    pool = pool[_selected(pool, cfg.dataset)]
    rng = np.random.default_rng([cfg.seed, 11])
    shares = {"train": cfg.mast.split.train, "val": cfg.mast.split.val, "test": cfg.mast.split.test}
    chosen = []
    for name, share in shares.items():
        quota = max(1, round(limit * share / 100))
        rows = pool[pool["split"] == name].sample(frac=1.0, random_state=int(rng.integers(2**31)))
        rank_in_tile = rows.groupby("tile").cumcount()  # one image per tile first, then a second, ...
        chosen.append(rows.assign(_r=rank_in_tile).sort_values("_r", kind="stable").head(quota).drop(columns="_r"))
    return pd.concat(chosen)


# ------------------------------------------------------------------------------------------ stage
def fetch(cfg: Config, refresh: bool = False) -> Path:
    """Run the fetch stage; returns the metadata table path. Re-running only downloads what is missing."""
    out = cfg.mast_dir / "metadata.parquet"
    cfg.mast_dir.mkdir(parents=True, exist_ok=True)
    m = cfg.mast
    if m.mock:
        from astrounets.data.mock import mock_mast

        table = mock_mast(cfg)
    elif out.exists() and not refresh:
        table = pd.read_parquet(out)
        log.info("using existing %s (%d rows); pass --refresh to query MAST again", out, len(table))
    else:
        table = query_metadata(cfg)
        table["url"], table["path"] = None, None
    table["tile"] = assign_tiles(table["ra"], table["dec"], m.tile_arcmin)
    split = assign_splits(table["tile"], {"train": m.split.train, "val": m.split.val, "test": m.split.test}, cfg.seed)
    table["split"] = apply_buffer(table["ra"], table["dec"], split, m.overlap_arcmin)
    if m.download and not m.mock:
        todo = choose_downloads(table, cfg)
        missing = todo.loc[todo["url"].isna(), "id"].tolist()
        if missing:
            urls = resolve_urls(missing, cfg)
            table["url"] = table["url"].where(table["url"].notna(), table["id"].map(urls))
        todo = table.loc[table["id"].isin(todo["id"]) & table["url"].notna()]
        present = download(dict(zip(todo["id"], todo["url"])), cfg.mast_dir / "originals", m.download_workers, m.http_timeout)
        known = table["path"].where(table["path"].notna(), None)
        table["path"] = [f"originals/{present[i].name}" if i in present else p for i, p in zip(table["id"], known)]
    write_parquet(table, out)
    summary = table.groupby("split").agg(images=("id", "size"), tiles=("tile", "nunique"),
                                        downloaded=("path", lambda p: int(p.notna().sum())))
    log.info("fetch done: %d images in %d sky tiles\n%s", len(table), table["tile"].nunique(), summary)
    return out
