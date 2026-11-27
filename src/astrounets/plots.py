"""Stage `plots`: figures of one run, drawn only from files the other stages wrote.

Per run:      plots/history.png                    training and validation loss per epoch
Per eval set: eval/<dataset>_<split>/plots/
  examples.png       noisy | reconstruction | deep image, crops from both groups at several ratios
  completeness.png   completeness vs true AB magnitude, sparse and crowded crops
  dmag_vs_mag.png    measured - true magnitude vs true magnitude, sparse and crowded crops
  by_ratio.png       completeness, flux scatter, SSIM, PSNR vs exposure ratio on the same crops
  hexbin/<group>/<subset>_<name>.png   (detailed=True) the thesis hexbin family, see HEXBINS:
      flux ratio, Kron radius, Delta Kron radius and flux against true AB magnitude, flux and Kron
      radius; for sources found in each image ("combined"), in both ("combined_both") and only in
      the reconstruction ("rec"); groups overall / sparse / crowded; subsets all and bright (AB < 25).

Per denoised image (astrounets denoise): <out>/<image>_compare.png, original | reconstruction |
difference, the whole image and full-resolution cutouts.

Group = sparse or crowded crops (dataset stage); every panel compares the reconstruction and the noisy
input on the same samples. Fluxes are Kron fluxes in e-/s; radii in arcsec (drizzled pixel PIXEL_SCALE).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PIXEL_SCALE = 0.12825       # arcsec per pixel of the WFC3/IR drizzled products (header D001SCAL)
BRIGHT_MAG = 25.0           # the "bright" subset of the hexbin figures, as in the thesis
COMMON_MAX_RATIO = 16.0     # by_ratio.png: ratios up to this, on crops evaluated at every one of them
GROUP_COLORS = {"overall": "0.3", "sparse": "tab:blue", "crowded": "tab:orange"}
KINDS = {"rec": "reconstruction", "noisy": "noisy input"}
CMAPS = {"rec": "viridis", "noisy": "cividis"}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path, dpi: int) -> Path:
    from astrounets.files import atomic

    path.parent.mkdir(parents=True, exist_ok=True)
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
                   titles: tuple[str, str, str] = ("noisy", "reconstruction", "clean"),
                   labels: list[str] | None = None) -> None:
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
        if labels:
            axes[r][0].set_ylabel(labels[r])
    _save(fig, path, dpi)


# ------------------------------------------------------------------------------ denoised images
NO_DATA_COLOR = "#4c72b0"   # no-data pixels in the comparison figures
OVERVIEW_PIXELS = 1200      # the whole-image row is block-averaged to at most this many pixels a side


def _block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks, ignoring non-finite pixels (all non-finite: NaN)."""
    if factor <= 1:
        return image
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    blocks = image[:h, :w].reshape(h // factor, factor, w // factor, factor)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _cutouts(shape: tuple[int, int], size: int, centres: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) boxes of `size` pixels around the centres (x, y), inside the image; default: the centre."""
    height, width = shape
    boxes = []
    for x, y in centres or [(width // 2, height // 2)]:
        sy, sx = min(size, height), min(size, width)
        y0 = int(np.clip(y - sy // 2, 0, height - sy))
        x0 = int(np.clip(x - sx // 2, 0, width - sx))
        boxes.append((y0, y0 + sy, x0, x0 + sx))
    return boxes


def compare_rows(original, rec, diff, boxes, factor) -> list[list[np.ndarray]]:
    """The images of each figure row: the whole image (block-averaged), then one row per cutout box."""
    rows = [[_block_mean(a, factor) for a in (original, rec, diff)]]
    for y0, y1, x0, x1 in boxes:
        rows.append([a[y0:y1, x0:x1] for a in (original, rec, diff)])
    return rows


def compare_figure(path: Path, original: np.ndarray, rec: np.ndarray, hole: np.ndarray, zoom: int,
                   centres: list[tuple[int, int]], dpi: int, title: str) -> Path:
    """original | reconstruction | original - reconstruction: the whole image (block-averaged if large)
    and full-resolution cutouts (`zoom` pixels; around `centres` (x, y), default the image centre).
    Original and reconstruction share the display range (zscale of the original)."""
    plt = _pyplot()
    original = np.where(hole, np.nan, original).astype(np.float64)
    rec = np.where(hole, np.nan, rec).astype(np.float64)
    diff = original - rec
    lo, hi = _limits(original)
    d = diff[np.isfinite(diff)]
    spread = float(1.4826 * np.median(np.abs(d - np.median(d)))) if d.size else 0.0
    dlim = 5 * spread if spread > 0 else 1.0
    boxes = _cutouts(original.shape, zoom, centres)
    factor = int(np.ceil(max(original.shape) / OVERVIEW_PIXELS))
    rows = compare_rows(original, rec, diff, boxes, factor)
    gray = plt.get_cmap("gray").with_extremes(bad=NO_DATA_COLOR)
    div = plt.get_cmap("RdBu_r").with_extremes(bad=NO_DATA_COLOR)
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)), squeeze=False)
    for r, images in enumerate(rows):
        if r == 0:
            extent = (-0.5, original.shape[1] - 0.5, -0.5, original.shape[0] - 0.5)
            label = "whole image" + (f" ({factor}x{factor} block mean)" if factor > 1 else "")
        else:
            y0, y1, x0, x1 = boxes[r - 1]
            extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
            label = f"x {x0}-{x1 - 1}, y {y0}-{y1 - 1} (full resolution)"
        for c, (image, name) in enumerate(zip(images, ("original", "reconstruction", "original - reconstruction"))):
            ax = axes[r][c]
            if c < 2:
                im = ax.imshow(image, origin="lower", cmap=gray, vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
            else:
                im = ax.imshow(image, origin="lower", cmap=div, vmin=-dlim, vmax=dlim, extent=extent, interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name}\n{label}", fontsize=9)
            if r == 0 and c < 2:
                for y0, y1, x0, x1 in boxes:
                    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="tab:orange", lw=1))
    fig.suptitle(f"{title}\nsame display range for original and reconstruction (zscale of the original); "
                 "blue: no data", fontsize=9)
    return _save(fig, path, dpi)


def show_side_by_side(original: np.ndarray, rec: np.ndarray, hole: np.ndarray, title: str) -> None:
    """Original and reconstruction in a window with linked axes (zoom and pan move both)."""
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        plt.switch_backend(matplotlib.rcParamsDefault["backend"])
    except Exception as err:  # no display
        log.warning("cannot open a window (%s); the comparison figure is written instead", err)
        return
    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "pgf", "template"):
        log.warning("no interactive matplotlib backend here; the comparison figure is written instead")
        return
    original = np.where(hole, np.nan, original)
    rec = np.where(hole, np.nan, rec)
    lo, hi = _limits(original)
    gray = plt.get_cmap("gray").with_extremes(bad=NO_DATA_COLOR)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
    for ax, image, name in zip(axes, (original, rec), ("original", "reconstruction")):
        ax.imshow(image, origin="lower", cmap=gray, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(name)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    matplotlib.use("Agg")


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


# ------------------------------------------------------------------------------------- data
def load_eval(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, sources) of an evaluation; sources get the sample's ratio and group."""
    samples = pd.read_parquet(eval_dir / "samples.parquet")
    sources = pd.read_parquet(eval_dir / "sources.parquet")
    if "crowded" not in samples:  # evaluations from before the sparse/crowded split
        samples["crowded"] = False
    if len(sources):
        sources = sources.drop(columns=[c for c in ("ratio",) if c in sources]).merge(
            samples[["sample_id", "ratio", "crowded"]], on="sample_id", how="left")
        sources["crowded"] = sources["crowded"].fillna(False).astype(bool)
    return samples, sources


def groups_of(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """overall, and sparse / crowded where present."""
    out = {"overall": frame}
    crowded = frame["crowded"].to_numpy(bool) if "crowded" in frame else np.zeros(len(frame), bool)
    for name, mask in (("sparse", ~crowded), ("crowded", crowded)):
        if mask.any():
            out[name] = frame[mask]
    return out


def mag_bins(mags) -> np.ndarray:
    """1-mag bins covering the sources (0.5 to 99.5 percentile), from the bright end."""
    m = np.asarray(mags, float)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return np.arange(20.0, 29.0)
    lo, hi = np.floor(np.percentile(m, 0.5)), np.ceil(np.percentile(m, 99.5))
    return np.arange(lo, max(hi, lo + 1) + 1.0)


def _binned(x: np.ndarray, y: np.ndarray, bins) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, med, scatter = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(y)
        if sel.sum() >= 5:
            v = y[sel]
            centers.append((lo + hi) / 2)
            med.append(np.median(v))
            scatter.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return np.array(centers), np.array(med), np.array(scatter)


# --------------------------------------------------------------------------- summary figures
def plot_completeness(sources: pd.DataFrame, path: Path, dpi: int) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = mag_bins(sources["mag_true"])
    centers = (bins[:-1] + bins[1:]) / 2
    for group, part in groups_of(sources).items():
        if group == "overall" and len(groups_of(sources)) > 2:
            continue  # with both groups, show them separately
        for kind, style in (("rec", "-"), ("noisy", "--")):
            frac = [part.loc[(part["mag_true"] >= lo) & (part["mag_true"] < hi), f"matched_{kind}"].mean()
                    if ((part["mag_true"] >= lo) & (part["mag_true"] < hi)).sum() >= 5 else np.nan
                    for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(centers, frac, style, marker="o", ms=3, color=GROUP_COLORS[group], label=f"{group}, {KINDS[kind]}")
    ax.set_xlabel("true AB magnitude"); ax.set_ylabel("completeness"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path, dpi)


def plot_dmag(sources: pd.DataFrame, path: Path, dpi: int) -> Path:
    plt = _pyplot()
    groups = {k: v for k, v in groups_of(sources).items() if k != "overall" or len(groups_of(sources)) == 1}
    fig, axes = plt.subplots(len(groups), 2, figsize=(11, 3.8 * len(groups)), squeeze=False, sharex=True, sharey=True)
    bins = mag_bins(sources["mag_true"])
    for r, (group, part) in enumerate(groups.items()):
        for c, kind in enumerate(("rec", "noisy")):
            ax = axes[r][c]
            sel = part[f"matched_{kind}"].to_numpy(bool) & np.isfinite(part[f"dmag_{kind}"].to_numpy(float))
            x, y = part["mag_true"].to_numpy(float)[sel], part[f"dmag_{kind}"].to_numpy(float)[sel]
            ax.scatter(x, y, s=2, alpha=0.2, color=GROUP_COLORS[group], rasterized=True)
            cen, med, sc = _binned(x, y, bins)
            ax.errorbar(cen, med, yerr=sc, color="k", fmt="o-", ms=3, capsize=3, label="median ± robust scatter")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(f"{KINDS[kind]}, {group} crops (N={sel.sum()})")
            ax.set_ylim(-1.5, 1.5); ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Δmag (measured − true)")
            if r == len(groups) - 1:
                ax.set_xlabel("true AB magnitude")
    axes[0][0].legend(fontsize=8)
    return _save(fig, path, dpi)


def _ratio_measures(samples: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    from astrounets.evaluate.metrics import summarise

    rows = []
    for (group, ratio), part in [((g, r), p) for g, gp in groups_of(samples).items() for r, p in gp.groupby("ratio")]:
        src = sources[sources["sample_id"].isin(part["sample_id"])] if len(sources) else sources
        rows.append({"group": group, "ratio": float(ratio), "crops": int(part["crop_id"].nunique()), **summarise(part, src)})
    return pd.DataFrame(rows)


def plot_by_ratio(samples: pd.DataFrame, sources: pd.DataFrame, path: Path, dpi: int) -> Path | None:
    """Measures vs exposure ratio on the SAME crops at every ratio (those evaluated at all ratios up to
    COMMON_MAX_RATIO), so the curves show the effect of noise, not of which crops allow which ratio."""
    ratios = sorted(r for r in samples["ratio"].unique() if r <= COMMON_MAX_RATIO)
    if not ratios:
        return None
    per_crop = samples[samples["ratio"].isin(ratios)].groupby("crop_id")["ratio"].nunique()
    common = per_crop.index[per_crop == len(ratios)]
    part = samples[samples["crop_id"].isin(common) & samples["ratio"].isin(ratios)]
    if part.empty:
        return None
    table = _ratio_measures(part, sources)
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(axes.flat, ("completeness", "flux_scatter", "ssim", "psnr"),
                              ("completeness", "flux scatter [mag]", "SSIM", "PSNR [dB]")):
        for group, rows in table.groupby("group", sort=False):
            if group == "overall" and table["group"].nunique() > 2:
                continue
            for kind, style in (("rec", "-"), ("noisy", "--")):
                ax.plot(rows["ratio"], rows[f"{key}_{kind}"], style, marker="o", ms=3, color=GROUP_COLORS[group],
                        label=f"{group}, {KINDS[kind]} ({int(rows['crops'].iloc[0])} crops)")
        ax.set_xscale("log"); ax.set_xlabel("exposure ratio"); ax.set_ylabel(label); ax.grid(alpha=0.3)
        ax.set_xticks(ratios); ax.set_xticklabels([f"{r:g}" for r in ratios]); ax.minorticks_off()
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"same crops at every ratio (crops evaluated at all ratios up to {COMMON_MAX_RATIO:g})", fontsize=10)
    return _save(fig, path, dpi)


def plot_examples(eval_dir: Path, path: Path, dpi: int) -> Path | None:
    examples = eval_dir / "examples.npz"
    if not examples.exists():
        return None
    data = np.load(examples)
    rows = [(data["noisy"][i], data["rec"][i], data["clean"][i]) for i in range(len(data["noisy"]))]
    if not rows:
        return None
    labels = None
    if "ratio" in data and "crowded" in data:
        labels = [f"γ = {r:g}, {'crowded' if c else 'sparse'}" for r, c in zip(data["ratio"], data["crowded"])]
    save_triptychs(path, rows, dpi, ("noisy", "reconstruction", "deep image"), labels)
    return path


# ------------------------------------------------------------------------- thesis hexbins
def _quantity(sources: pd.DataFrame, name: str, kind: str | None = None) -> np.ndarray:
    """Plotted quantities: mag (true AB), logflux_<kind>, ratio_<kind>, kron_<kind>, dkron_<kind>."""
    f = lambda col: sources[col].to_numpy(float)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "mag":
            return f("mag_true")
        if name == "logflux":
            return np.log10(np.where(f(f"flux_{kind}") > 0, f(f"flux_{kind}"), np.nan))
        if name == "ratio":
            return f(f"flux_{kind}") / f("flux_true")
        if name == "kron":
            return f(f"kron_radius_{kind}") * PIXEL_SCALE
        if name == "dkron":
            return (f(f"kron_radius_{kind}") - f("kron_radius_true")) * PIXEL_SCALE
    raise ValueError(name)


AXES = {
    "mag": "true AB magnitude",
    "logflux": r"$\log_{10}$ flux [e$^-$ s$^{-1}$]",
    "ratio": "flux ratio (measured / true)",
    "kron": "Kron radius [arcsec]",
    "dkron": r"$\Delta$Kron radius (measured − true) [arcsec]",
}

# name: (x, y, reference line). x of "logflux"/"kron" are the true values; y is per image.
QUANTITIES = {
    "flux_ratio_mag": ("mag", "ratio", "one"),
    "flux_ratio": ("logflux", "ratio", "one"),
    "delta_kron_mag": ("mag", "dkron", "zero"),
    "delta_kron": ("kron", "dkron", "zero"),
    "kron": ("kron", "kron", "diagonal"),
    "flux": ("logflux", "logflux", "diagonal"),
}
SETS = {  # file suffix: (which sources, title)
    "combined": ("each", "sources found in each image"),
    "combined_both": ("both", "sources found in both images"),
    "rec": ("rec_only", "sources found only in the reconstruction"),
}


def _selection(sources: pd.DataFrame, which: str, kind: str) -> np.ndarray:
    rec, noisy = sources["matched_rec"].to_numpy(bool), sources["matched_noisy"].to_numpy(bool)
    if which == "each":
        return rec if kind == "rec" else noisy
    if which == "both":
        return rec & noisy
    return rec & ~noisy


def _panel_values(sources: pd.DataFrame, x: str, y: str, which: str, kind: str):
    sel = _selection(sources, which, kind)
    xv = _quantity(sources, x, "true") if x != "mag" else _quantity(sources, "mag")
    yv = _quantity(sources, y, kind)
    ok = sel & np.isfinite(xv) & np.isfinite(yv)
    return xv[ok], yv[ok]


def _hexbin(fig, ax, x, y, cmap: str, limits, reference: str, label: str) -> None:
    ax.set_box_aspect(1)
    if len(x) < 2:
        ax.text(0.5, 0.5, "no sources", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=9)
        return
    (x0, x1), (y0, y1) = limits
    gridsize = 100 if len(x) > 20000 else 50
    hb = ax.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=2, cmap=cmap, linewidths=0.1,
                   extent=(x0, x1, y0, y1))
    if len(hb.get_array()):  # no cell with 2+ sources: nothing to colour
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.scatter(x, y, s=4, color="k")
    edges = np.linspace(x0, x1, 16)
    cen, med, _ = _binned(x, y, edges)
    ax.plot(cen, med, color="k", lw=1.2, label="running median")
    if reference == "one":
        ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.6)
    elif reference == "zero":
        ax.axhline(0.0, color="red", ls="--", lw=0.8, alpha=0.6)
    else:
        lo, hi = max(x0, y0), min(x1, y1)
        ax.plot([lo, hi], [lo, hi], color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=0.3)
    ax.set_title(f"{label}, N={len(x)}", fontsize=9)


def _limits_of(values: list[np.ndarray], lo: float = 1, hi: float = 99) -> tuple[float, float]:
    v = np.concatenate([a for a in values if len(a)]) if any(len(a) for a in values) else np.array([0.0, 1.0])
    a, b = np.percentile(v, [lo, hi])
    if a == b:
        a, b = a - 0.5, b + 0.5
    pad = 0.03 * (b - a)
    return a - pad, b + pad


def plot_hexbin_family(sources: pd.DataFrame, out_dir: Path, dpi: int, title: str) -> list[Path]:
    """The 18 thesis hexbin figures for one set of sources: one panel per exposure ratio, the
    reconstruction next to the noisy input (one column only for sources found only in the reconstruction)."""
    plt = _pyplot()
    written = []
    has_kron = "kron_radius_true" in sources
    ratios = sorted(sources["ratio"].dropna().unique())
    for qname, (x, y, reference) in QUANTITIES.items():
        if not has_kron and "kron" in (x, y) or not has_kron and y == "dkron":
            continue
        for suffix, (which, set_title) in SETS.items():
            kinds = ["rec"] if which == "rec_only" else ["rec", "noisy"]
            per_ratio = {r: sources[sources["ratio"] == r] for r in ratios}
            values = {(r, k): _panel_values(part, x, y, which, k) for r, part in per_ratio.items() for k in kinds}
            xlim = _limits_of([v[0] for v in values.values()])
            ylim = _limits_of([v[1] for v in values.values()], 2, 98) if reference != "diagonal" else xlim
            if reference == "diagonal":
                xlim = ylim = _limits_of([v[0] for v in values.values()] + [v[1] for v in values.values()])
            pairs_per_row = 2 if len(ratios) > 3 else 1
            n_rows = int(np.ceil(len(ratios) / pairs_per_row))
            n_cols = pairs_per_row * len(kinds)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.9 * n_rows), squeeze=False)
            for i, r in enumerate(ratios):
                row, pair = i // pairs_per_row, i % pairs_per_row
                for j, k in enumerate(kinds):
                    ax = axes[row][pair * len(kinds) + j]
                    xv, yv = values[(r, k)]
                    _hexbin(fig, ax, xv, yv, CMAPS[k], (xlim, ylim), reference, f"{KINDS[k]}, γ = {r:g}")
                    ax.set_xlabel(AXES[x] if x == "mag" else AXES[x].replace("Kron radius", "true Kron radius")
                                  .replace("flux [", "true flux ["), fontsize=8)
                    ax.set_ylabel(AXES[y], fontsize=8)
            for k in range(len(ratios), n_rows * pairs_per_row):  # empty slots
                for j in range(len(kinds)):
                    axes[k // pairs_per_row][(k % pairs_per_row) * len(kinds) + j].set_visible(False)
            fig.suptitle(f"{title}: {set_title}", fontsize=11)
            written.append(_save(fig, out_dir / f"{qname}_{suffix}.png", dpi))
    return written


def plot_hexbins(sources: pd.DataFrame, out_dir: Path, dpi: int) -> list[Path]:
    """The hexbin family for each group (overall, sparse, crowded) and subset (all, bright)."""
    written = []
    if sources.empty:
        return written
    for group, part in groups_of(sources).items():
        for subset, sel in (("all", np.ones(len(part), bool)),
                            ("bright", part["mag_true"].to_numpy(float) < BRIGHT_MAG)):
            if sel.sum() < 2:
                continue
            title = f"{group} crops, {'all sources' if subset == 'all' else f'AB < {BRIGHT_MAG:g}'}"
            written += [p for p in plot_hexbin_family(part[sel], out_dir / group / subset, dpi, title)]
    return written


# ------------------------------------------------------------------------------------- entry
def plot_eval(eval_dir: Path, dpi: int, detailed: bool = False) -> list[Path]:
    samples, sources = load_eval(eval_dir)
    out_dir = eval_dir / "plots"
    written = [p for p in [plot_examples(eval_dir, out_dir / "examples.png", dpi)] if p]
    if len(sources):
        written.append(plot_completeness(sources, out_dir / "completeness.png", dpi))
        written.append(plot_dmag(sources, out_dir / "dmag_vs_mag.png", dpi))
        written += [p for p in [plot_by_ratio(samples, sources, out_dir / "by_ratio.png", dpi)] if p]
        if detailed:
            written += plot_hexbins(sources, out_dir / "hexbin", dpi)
    return written


def make_plots(cfg, run_dir: Path, detailed: bool = False) -> list[Path]:
    run_dir = Path(run_dir)
    written = [p for p in [plot_history(run_dir, cfg.plots.dpi)] if p]
    for eval_dir in sorted((run_dir / "eval").glob("*")):
        if (eval_dir / "summary.json").exists():
            written += plot_eval(eval_dir, cfg.plots.dpi, detailed)
    log.info("plots written: %d", len(written))
    return written
