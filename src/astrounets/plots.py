"""Stage `plots`: figures drawn only from files the other stages wrote."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path, dpi: int) -> Path:
    from astrounets.files import atomic

    fig.tight_layout()
    with atomic(path) as tmp:
        fig.savefig(tmp, dpi=dpi)
    _pyplot().close(fig)
    return path


def _limits(image: np.ndarray) -> tuple[float, float]:
    from astropy.visualization import ZScaleInterval

    finite = image[np.isfinite(image)]
    if finite.size == 0 or np.ptp(finite) == 0:
        return 0.0, 1.0
    return ZScaleInterval().get_limits(finite)


def save_triptychs(path: Path, rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], dpi: int,
                   titles: tuple[str, str, str] = ("noisy", "reconstruction", "clean")) -> None:
    """One row per example: noisy | reconstruction | clean, all with the clean image's display range."""
    plt = _pyplot()
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for r, images in enumerate(rows):
        lo, hi = _limits(images[2])
        for c, (image, title) in enumerate(zip(images, titles)):
            ax = axes[r][c]
            ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
    _save(fig, path, dpi)


def plot_history(run_dir: Path, dpi: int) -> Path | None:
    path = run_dir / "history.csv"
    if not path.exists():
        return None
    history = pd.read_csv(path)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["rec_loss"], label="train")
    ax.plot(history["epoch"], history["val_rec_loss"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction loss"); ax.set_yscale("log"); ax.legend()
    return _save(fig, run_dir / "plots" / "history.png", dpi)


def _binned(x: np.ndarray, y: np.ndarray, bins: list[float]):
    centers, med, scatter = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(y)
        if sel.sum() >= 3:
            v = y[sel]
            centers.append((lo + hi) / 2)
            med.append(np.median(v))
            scatter.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return np.array(centers), np.array(med), np.array(scatter)


def plot_eval(eval_dir: Path, mag_bins: list[float], dpi: int) -> list[Path]:
    plt = _pyplot()
    sources = pd.read_parquet(eval_dir / "sources.parquet")
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    out_dir = eval_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. magnitude error vs true magnitude
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind in zip(axes, ("rec", "noisy")):
        sel = sources[f"matched_{kind}"] & np.isfinite(sources[f"dmag_{kind}"])
        ax.scatter(sources.loc[sel, "mag_true"], sources.loc[sel, f"dmag_{kind}"], s=3, alpha=0.3)
        c, m, s = _binned(sources.loc[sel, "mag_true"].to_numpy(), sources.loc[sel, f"dmag_{kind}"].to_numpy(), mag_bins)
        ax.errorbar(c, m, yerr=s, color="k", fmt="o-", capsize=3, label="median ± robust scatter")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("reconstruction" if kind == "rec" else "noisy input")
        ax.set_xlabel("true AB magnitude")
    axes[0].set_ylabel("Δmag (measured − true)"); axes[0].set_ylim(-1.5, 1.5); axes[0].legend()
    written.append(_save(fig, out_dir / "dmag_vs_mag.png", dpi))

    # 2. completeness vs magnitude
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, label in (("rec", "reconstruction"), ("noisy", "noisy input")):
        frac = []
        for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
            sel = (sources["mag_true"] >= lo) & (sources["mag_true"] < hi)
            frac.append(sources.loc[sel, f"matched_{kind}"].mean() if sel.any() else np.nan)
        ax.plot((np.array(mag_bins[:-1]) + np.array(mag_bins[1:])) / 2, frac, "o-", label=label)
    ax.set_xlabel("true AB magnitude"); ax.set_ylabel("completeness"); ax.set_ylim(0, 1.05); ax.legend()
    written.append(_save(fig, out_dir / "completeness.png", dpi))

    # 3. summary measures vs exposure ratio
    per_ratio = pd.DataFrame(summary["per_ratio"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(axes.flat, ("completeness", "flux_scatter", "ssim", "psnr"),
                              ("completeness", "flux scatter [mag]", "SSIM", "PSNR [dB]")):
        for kind, name in (("rec", "reconstruction"), ("noisy", "noisy input")):
            ax.plot(per_ratio["ratio"], per_ratio[f"{key}_{kind}"], "o-", label=name)
        ax.set_xscale("log"); ax.set_xlabel("exposure ratio"); ax.set_ylabel(label)
        ax.set_xticks(per_ratio["ratio"]); ax.set_xticklabels([f"{r:g}" for r in per_ratio["ratio"]]); ax.minorticks_off()
    axes[0][0].legend()
    written.append(_save(fig, out_dir / "by_ratio.png", dpi))

    # 4. examples
    examples = eval_dir / "examples.npz"
    if examples.exists():
        data = np.load(examples)
        rows = [(data["noisy"][i], data["rec"][i], data["clean"][i]) for i in range(len(data["noisy"]))]
        if rows:
            path = out_dir / "examples.png"
            save_triptychs(path, rows, dpi, ("noisy", "reconstruction", "deep image"))
            written.append(path)
    return written


def make_plots(cfg, run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    written = [p for p in [plot_history(run_dir, cfg.plots.dpi)] if p]
    for eval_dir in sorted((run_dir / "eval").glob("*")):
        if (eval_dir / "summary.json").exists():
            written += plot_eval(eval_dir, cfg.eval.mag_bins, cfg.plots.dpi)
    log.info("plots written: %d", len(written))
    return written
