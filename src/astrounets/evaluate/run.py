"""Stage `evaluate`: best checkpoint of a run on the frozen evaluation set of a dataset split.

Output: <run>/eval/<dataset>_<split>/ with samples.parquet (one row per noisy input),
sources.parquet (one row per deep-image source, with its matches), summary.json and examples.npz.
summary.json holds the measures for all crops ("overall") and separately for sparse and crowded crops
("groups"; crowded as marked by the dataset stage).
Every model evaluated on the same dataset split sees byte-identical noisy inputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.config import Config
from astrounets.data.dataset import load_crops
from astrounets.data.fits_io import CropStore
from astrounets.data.noise import noisy_sigma
from astrounets.data.pairs import eval_table, noisy_pair
from astrounets.evaluate.metrics import compare, summary
from astrounets.files import atomic, write_json, write_parquet

log = logging.getLogger(__name__)


def example_ids(table: pd.DataFrame, n: int) -> set:
    """Samples kept for examples.png: crops from both groups (sparse and crowded), each crop at another
    exposure ratio, from the lowest to the highest."""
    crowded = table["crowded"].to_numpy(bool) if "crowded" in table else np.zeros(len(table), bool)
    parts = [part for part in (table[~crowded], table[crowded]) if len(part)]
    chosen = set()
    for g, part in enumerate(parts):
        k = n // len(parts) + (1 if g < n % len(parts) else 0)
        crops = part["crop_id"].drop_duplicates().head(k).tolist()
        ratios = sorted(part["ratio"].unique())
        for j, crop in enumerate(crops):
            want = ratios[round(j * (len(ratios) - 1) / max(len(crops) - 1, 1))]
            rows = part[part["crop_id"] == crop]
            chosen.add(int(rows.iloc[int(np.argmin(np.abs(rows["ratio"].to_numpy(float) - want)))]["sample_id"]))
    return chosen


def eval_dir(run_dir: Path, dataset: str, split: str) -> Path:
    return Path(run_dir) / "eval" / f"{dataset}_{split}"


def evaluate(cfg: Config, run_dir: Path | None = None, dataset: str | None = None, split: str | None = None) -> Path:
    from astrounets.infer import TrainedModel

    run_dir = Path(run_dir or cfg.run_dir)
    dataset, split = dataset or cfg.dataset.name, split or cfg.eval.split
    best = run_dir / "checkpoints" / "best.keras"
    if not best.exists():
        raise FileNotFoundError(f"{best} not found; train the run first")
    if not (run_dir / "checkpoints" / "best.json").exists():
        raise FileNotFoundError(f"{run_dir}: no best checkpoint record")
    trained = TrainedModel.load(run_dir)  # with the scaling and patch size it was trained with
    ps = trained.patch_size
    crops = load_crops(cfg, dataset)
    built_ps = json.loads((cfg.dataset_dir(dataset) / "summary.json").read_text(encoding="utf-8"))["patch_size"]
    if built_ps != ps:
        raise ValueError(f"dataset '{dataset}' was cut into {built_ps}px crops but the model expects {ps}px")
    table = eval_table(crops, cfg, split, cfg.dataset_dir(dataset))
    store = CropStore(cfg.mast_dir, cfg.dataset.hdu, cfg.dataset.weight_hdu)
    e = cfg.eval
    samples, sources = [], []
    examples = {"noisy": [], "rec": [], "clean": [], "ratio": [], "crowded": [], "sample_id": []}
    keep = example_ids(table, e.n_examples)
    records = table.to_dict("records")
    for start in range(0, len(records), e.batch_size):
        batch = [(row, *noisy_pair(store, row, row["ratio"], row["noise_seed"], ps))
                 for row in records[start:start + e.batch_size]]
        recs = trained.reconstruct(np.stack([noisy for _, _, noisy, _ in batch]), e.batch_size,
                                   valid=np.stack([~hole for _, _, _, hole in batch]))
        for (row, clean, noisy, hole), rec in zip(batch, recs):
            if rec is None:  # constant crop: the scaling cannot handle it
                continue
            rec[hole] = clean[hole]  # no-data pixels: the same flat background in all three images
            threshold = e.detect_nsigma * noisy_sigma(row["sigma"], row["ratio"])
            sample, src = compare(clean, noisy, rec, threshold, e, row["photflam"], row["photplam"])
            ids = {"sample_id": row["sample_id"], "crop_id": row["crop_id"], "image_id": row["image_id"], "ratio": row["ratio"]}
            samples.append({**ids, "sigma": row["sigma"], "crowded": bool(row.get("crowded", False)),
                            "threshold": threshold, **sample})
            src = src.assign(**ids)
            sources.append(src)
            if row["sample_id"] in keep:
                for key, value in (("noisy", noisy), ("rec", rec), ("clean", clean), ("ratio", row["ratio"]),
                                   ("crowded", bool(row.get("crowded", False))), ("sample_id", row["sample_id"])):
                    examples[key].append(value)
    store.close()
    samples_df = pd.DataFrame(samples)
    sources_df = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    out = eval_dir(run_dir, dataset, split)
    write_parquet(samples_df, out / "samples.parquet")
    write_parquet(sources_df, out / "sources.parquet")
    with atomic(out / "examples.npz") as tmp:
        images = {k: np.stack(examples[k]) if examples[k] else np.zeros((0, ps, ps), np.float32)
                  for k in ("noisy", "rec", "clean")}
        labels = {k: np.asarray(examples[k]) for k in ("ratio", "crowded", "sample_id")}
        np.savez_compressed(tmp, **images, **labels)
    result = summary(samples_df, sources_df, e.mag_bins)
    best_info = json.loads((run_dir / "checkpoints" / "best.json").read_text(encoding="utf-8"))
    result.update({"run": run_dir.name, "dataset": dataset, "split": split, "checkpoint_epoch": best_info["epoch"]})
    write_json(out / "summary.json", result)  # written last: its presence means "evaluated"
    for group, o in result["groups"].items():
        log.info("evaluate %s on %s/%s, %s crops (%d samples): completeness %.3f (noisy %.3f), flux scatter %.3f mag, "
                 "SSIM %.3f", run_dir.name, dataset, split, group, o["n_samples"], o["completeness_rec"],
                 o["completeness_noisy"], o["flux_scatter_rec"], o["ssim_rec"])
    return out
