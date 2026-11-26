"""Image and catalogue metrics, per sample and aggregated."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from astrounets.config import Eval
from astrounets.evaluate.detect import background, measure
from astrounets.evaluate.match import match

JY_PER_FLAM_A2 = 1e23 / 2.99792458e18  # f_nu[Jy] = f_lambda * lambda^2 * this


def ab_mag(flux_e_s, photflam: float, photplam: float):
    """AB magnitude of a count rate (e-/s) from the PHOTFLAM/PHOTPLAM header keywords."""
    flux = np.asarray(flux_e_s, float)
    jansky = flux * photflam * photplam ** 2 * JY_PER_FLAM_A2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(jansky > 0, -2.5 * np.log10(jansky) + 8.90, np.nan)


def ssim(x: np.ndarray, y: np.ndarray, win_size: int = 11, win_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> float:
    """Mean SSIM with a Gaussian window (Wang et al. 2004); dynamic range from the reference x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = x.max() - x.min()
    if span == 0:
        return float("nan")
    c1, c2 = (k1 * span) ** 2, (k2 * span) ** 2

    def blur(a):
        return gaussian_filter(a, win_sigma, truncate=(win_size // 2) / win_sigma)

    mx, my = blur(x), blur(y)
    vx, vy = np.maximum(blur(x * x) - mx ** 2, 0), np.maximum(blur(y * y) - my ** 2, 0)
    cov = blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(np.mean(s))


def compare(clean: np.ndarray, noisy: np.ndarray, rec: np.ndarray, threshold: float, e: Eval,
            photflam: float, photplam: float) -> tuple[dict, pd.DataFrame]:
    """Detect in all three images with the same absolute threshold and the same background (estimated
    on the deep image) and match to the deep image."""
    back = background(clean, e)
    true = measure(clean, threshold, e, back)
    found = {"rec": measure(rec, threshold, e, back), "noisy": measure(noisy, threshold, e, back)}
    sources = pd.DataFrame({
        "x": true["x"].to_numpy(float), "y": true["y"].to_numpy(float),
        "flux_true": true["flux"].to_numpy(float), "radius_true": true["radius"].to_numpy(float),
        "ellipticity_true": true["ellipticity"].to_numpy(float),
    })
    sources["mag_true"] = ab_mag(sources["flux_true"], photflam, photplam)
    sample = {"n_true": len(true), "range": float(clean.max() - clean.min())}
    for kind, cat in found.items():
        i_true, i_found = match(true[["x", "y"]].to_numpy(), cat[["x", "y"]].to_numpy(), e.match_radius)
        for col in ("flux", "radius", "ellipticity"):
            values = np.full(len(sources), np.nan)
            values[i_true] = cat[col].to_numpy(float)[i_found]
            sources[f"{col}_{kind}"] = values
        matched = np.zeros(len(sources), bool)
        matched[i_true] = True
        sources[f"matched_{kind}"] = matched
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sources[f"flux_{kind}"] / sources["flux_true"]
            sources[f"dmag_{kind}"] = np.where((ratio > 0) & matched, -2.5 * np.log10(ratio), np.nan)
        image = rec if kind == "rec" else noisy
        sample.update({
            f"n_{kind}": len(cat), f"tp_{kind}": len(i_true),
            f"mse_{kind}": float(np.mean((image.astype(float) - clean) ** 2)),
            f"ssim_{kind}": ssim(clean, image),
        })
    return sample, sources


def _robust(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med)))


def summarise(samples: pd.DataFrame, sources: pd.DataFrame) -> dict:
    """Measures for the reconstruction and, for reference, the noisy input."""
    out = {"n_samples": int(len(samples)), "n_true_sources": int(samples["n_true"].sum())}
    for kind in ("rec", "noisy"):
        tp, n_true, n_found = samples[f"tp_{kind}"].sum(), samples["n_true"].sum(), samples[f"n_{kind}"].sum()
        bias, scatter = _robust(sources[f"dmag_{kind}"].to_numpy(float)) if len(sources) else (float("nan"),) * 2
        m = sources[f"matched_{kind}"].to_numpy(bool) if len(sources) else np.zeros(0, bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_err = np.abs(sources[f"radius_{kind}"].to_numpy(float)[m] / sources["radius_true"].to_numpy(float)[m] - 1)
            e_err = np.abs(sources[f"ellipticity_{kind}"].to_numpy(float)[m] - sources["ellipticity_true"].to_numpy(float)[m])
        # PSNR per crop (its own deep-image range), then the median: a few bright crops with large
        # absolute errors no longer decide the number for all crops
        with np.errstate(divide="ignore", invalid="ignore"):
            psnr = 10 * np.log10(samples["range"].to_numpy(float) ** 2 / samples[f"mse_{kind}"].to_numpy(float))
        psnr = psnr[np.isfinite(psnr)]
        out.update({
            f"completeness_{kind}": float(tp / n_true) if n_true else float("nan"),
            f"purity_{kind}": float(tp / n_found) if n_found else float("nan"),
            f"flux_bias_{kind}": bias,
            f"flux_scatter_{kind}": scatter,
            f"radius_error_{kind}": float(np.nanmedian(r_err)) if np.isfinite(r_err).any() else float("nan"),
            f"ellipticity_error_{kind}": float(np.nanmedian(e_err)) if np.isfinite(e_err).any() else float("nan"),
            f"ssim_{kind}": float(samples[f"ssim_{kind}"].mean()),
            f"psnr_{kind}": float(np.median(psnr)) if psnr.size else float("nan"),
        })
    return out


def crop_groups(samples: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row masks of the evaluation groups: every crop, sparse crops and crowded crops (empty groups left out)."""
    crowded = samples["crowded"].to_numpy(bool) if "crowded" in samples else np.zeros(len(samples), bool)
    groups = {"overall": np.ones(len(samples), bool), "sparse": ~crowded, "crowded": crowded}
    return {name: mask for name, mask in groups.items() if name == "overall" or mask.any()}


def summary(samples: pd.DataFrame, sources: pd.DataFrame, mag_bins: list[float]) -> dict:
    groups = {}
    for name, mask in crop_groups(samples).items():
        part = samples[mask]
        groups[name] = summarise(part, sources[sources["sample_id"].isin(part["sample_id"])] if len(sources) else sources)
    overall = groups["overall"]
    per_ratio = []
    for ratio, group in samples.groupby("ratio"):
        row = summarise(group, sources[sources["sample_id"].isin(group["sample_id"])])
        per_ratio.append({"ratio": float(ratio), **row})
    per_mag = []
    for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
        sel = (sources["mag_true"] >= lo) & (sources["mag_true"] < hi)
        row = {"mag_lo": lo, "mag_hi": hi, "n": int(sel.sum())}
        for kind in ("rec", "noisy"):
            row[f"completeness_{kind}"] = float(sources.loc[sel, f"matched_{kind}"].mean()) if sel.any() else float("nan")
            row[f"flux_bias_{kind}"], row[f"flux_scatter_{kind}"] = _robust(sources.loc[sel, f"dmag_{kind}"].to_numpy(float))
        per_mag.append(row)
    # ranking measures per group: <measure>_<group>, guard_ok_<group>, n_samples_<group>
    ranking = {}
    for name, g in groups.items():
        ranking.update({
            f"flux_scatter_{name}": g["flux_scatter_rec"],
            f"flux_bias_{name}": abs(g["flux_bias_rec"]),
            f"ssim_{name}": g["ssim_rec"],
            f"radius_error_{name}": g["radius_error_rec"],
            f"ellipticity_error_{name}": g["ellipticity_error_rec"],
            f"guard_ok_{name}": bool(g["completeness_rec"] >= g["completeness_noisy"] and g["purity_rec"] >= g["purity_noisy"]),
            f"n_samples_{name}": g["n_samples"],
        })
    return {"overall": overall, "groups": groups, "per_ratio": per_ratio, "per_mag": per_mag, "ranking": ranking}
