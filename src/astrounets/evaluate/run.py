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

from astrounets.config import Config, from_dict, read_yaml
from astrounets.data.dataset import load_crops
from astrounets.data.fits_io import CropStore
from astrounets.data.noise import noisy_sigma
from astrounets.data.pairs import eval_table, noisy_pair
from astrounets.data.scaling import Scaler
from astrounets.evaluate.metrics import compare, summary
from astrounets.files import atomic, write_json, write_parquet

log = logging.getLogger(__name__)


def eval_dir(run_dir: Path, dataset: str, split: str) -> Path:
    return Path(run_dir) / "eval" / f"{dataset}_{split}"


def evaluate(cfg: Config, run_dir: Path | None = None, dataset: str | None = None, split: str | None = None) -> Path:
    import keras

    run_dir = Path(run_dir or cfg.run_dir)
    dataset, split = dataset or cfg.dataset.name, split or cfg.eval.split
    best = run_dir / "checkpoints" / "best.keras"
    if not best.exists():
        raise FileNotFoundError(f"{best} not found; train the run first")
    run_cfg = from_dict(read_yaml(run_dir / "config.yaml"), cfg.root)  # scaling + patch size the model was trained with
    ps = run_cfg.dataset.patch_size
    crops = load_crops(cfg, dataset)
    built_ps = json.loads((cfg.dataset_dir(dataset) / "summary.json").read_text(encoding="utf-8"))["patch_size"]
    if built_ps != ps:
        raise ValueError(f"dataset '{dataset}' was cut into {built_ps}px crops but the model expects {ps}px")
    table = eval_table(crops, cfg, split, cfg.dataset_dir(dataset))
    if not (run_dir / "checkpoints" / "best.json").exists():
        raise FileNotFoundError(f"{run_dir}: no best checkpoint record")
    model = keras.models.load_model(best, compile=False)
    scaler = Scaler(run_cfg.scaling.mode, run_cfg.scaling.log_clip_max)
    store = CropStore(cfg.mast_dir, cfg.dataset.hdu, cfg.dataset.weight_hdu)
    e = cfg.eval
    samples, sources, examples = [], [], {"noisy": [], "rec": [], "clean": []}
    records = table.to_dict("records")
    for start in range(0, len(records), e.batch_size):
        batch = []
        for row in records[start:start + e.batch_size]:
            clean, noisy, hole = noisy_pair(store, row, row["ratio"], row["noise_seed"], ps)
            params = scaler.fit(noisy)
            if params is not None:
                batch.append((row, clean, noisy, hole, params))
        if not batch:
            continue
        x = np.stack([scaler.forward(noisy, params)[..., None] for _, _, noisy, _, params in batch])
        pred = np.asarray(model(x, training=False))[..., 0]
        for (row, clean, noisy, hole, params), p in zip(batch, pred):
            rec = scaler.inverse(p, params)
            rec[hole] = clean[hole]  # no-data pixels: the same flat background in all three images
            threshold = e.detect_nsigma * noisy_sigma(row["sigma"], row["ratio"])
            sample, src = compare(clean, noisy, rec, threshold, e, row["photflam"], row["photplam"])
            ids = {"sample_id": row["sample_id"], "crop_id": row["crop_id"], "image_id": row["image_id"], "ratio": row["ratio"]}
            samples.append({**ids, "sigma": row["sigma"], "crowded": bool(row.get("crowded", False)),
                            "threshold": threshold, **sample})
            src = src.assign(**ids)
            sources.append(src)
            if len(examples["rec"]) < e.n_examples:
                examples["noisy"].append(noisy); examples["rec"].append(rec); examples["clean"].append(clean)
    store.close()
    samples_df = pd.DataFrame(samples)
    sources_df = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    out = eval_dir(run_dir, dataset, split)
    write_parquet(samples_df, out / "samples.parquet")
    write_parquet(sources_df, out / "sources.parquet")
    with atomic(out / "examples.npz") as tmp:
        np.savez_compressed(tmp, **{k: np.stack(v) if v else np.zeros((0, ps, ps)) for k, v in examples.items()})
    result = summary(samples_df, sources_df, e.mag_bins)
    best_info = json.loads((run_dir / "checkpoints" / "best.json").read_text(encoding="utf-8"))
    result.update({"run": run_dir.name, "dataset": dataset, "split": split, "checkpoint_epoch": best_info["epoch"]})
    write_json(out / "summary.json", result)  # written last: its presence means "evaluated"
    for group, o in result["groups"].items():
        log.info("evaluate %s on %s/%s, %s crops (%d samples): completeness %.3f (noisy %.3f), flux scatter %.3f mag, "
                 "SSIM %.3f", run_dir.name, dataset, split, group, o["n_samples"], o["completeness_rec"],
                 o["completeness_noisy"], o["flux_scatter_rec"], o["ssim_rec"])
    return out
