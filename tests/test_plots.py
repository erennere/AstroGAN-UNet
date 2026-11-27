"""Figures: drawn from evaluation files only, for every group, subset and exposure ratio."""
import json

import numpy as np
import pandas as pd

from astrounets.plots import BRIGHT_MAG, QUANTITIES, SETS, load_eval, plot_eval


def _fake_eval(path, rng, n_per_ratio=400, ratios=(2.0, 8.0, 32.0)):
    """samples.parquet / sources.parquet / summary.json shaped like an evaluation (sparse and crowded)."""
    samples, sources = [], []
    sample_id = 0
    for crop in range(12):
        crowded = crop >= 8
        for ratio in ratios:
            samples.append({"sample_id": sample_id, "crop_id": crop, "image_id": f"I{crop}", "ratio": ratio,
                            "crowded": crowded, "n_true": n_per_ratio // 12, "n_rec": 30, "n_noisy": 30, "tp_rec": 25,
                            "tp_noisy": 24, "mse_rec": 1e-4, "mse_noisy": 2e-4, "ssim_rec": 0.95, "ssim_noisy": 0.9, "range": 10.0})
            k = n_per_ratio // 12
            mag = rng.uniform(19, 28, k)
            flux = 10 ** (-0.4 * (mag - 25))
            kron = rng.uniform(2, 8, k)
            row = {"sample_id": sample_id, "crop_id": crop, "image_id": f"I{crop}", "ratio": ratio, "mag_true": mag,
                   "flux_true": flux, "kron_radius_true": kron, "radius_true": kron / 2, "ellipticity_true": rng.uniform(0, .5, k)}
            for kind, noise in (("rec", 0.05), ("noisy", 0.1)):
                found = rng.random(k) < 0.8
                row |= {f"flux_{kind}": np.where(found, flux * (1 + noise * rng.normal(size=k)), np.nan),
                        f"kron_radius_{kind}": np.where(found, kron + noise * rng.normal(size=k), np.nan),
                        f"radius_{kind}": kron / 2, f"ellipticity_{kind}": 0.2, f"matched_{kind}": found,
                        f"dmag_{kind}": np.where(found, noise * rng.normal(size=k), np.nan)}
            sources.append(pd.DataFrame(row))
            sample_id += 1
    path.mkdir(parents=True)
    pd.DataFrame(samples).to_parquet(path / "samples.parquet")
    pd.concat(sources, ignore_index=True).to_parquet(path / "sources.parquet")
    (path / "summary.json").write_text(json.dumps({"overall": {}}))
    return path


def test_every_figure_is_drawn_for_both_groups_and_subsets(tmp_path):
    eval_dir = _fake_eval(tmp_path / "eval" / "all_test", np.random.default_rng(0))
    written = plot_eval(eval_dir, dpi=30, detailed=True)
    names = {p.relative_to(eval_dir / "plots").as_posix() for p in written}
    assert {"completeness.png", "dmag_vs_mag.png", "by_ratio.png"} <= names
    per_folder = len(QUANTITIES) * len(SETS)  # 6 quantities x 3 source sets = the 18 thesis figures
    for group in ("overall", "sparse", "crowded"):
        for subset in ("all", "bright"):
            folder = [n for n in names if n.startswith(f"hexbin/{group}/{subset}/")]
            assert len(folder) == per_folder, (group, subset, len(folder))
    _, sources = load_eval(eval_dir)
    assert sources["crowded"].any() and (~sources["crowded"]).any() and (sources["mag_true"] < BRIGHT_MAG).any()


def test_figures_skip_kron_panels_of_old_evaluations(tmp_path):
    eval_dir = _fake_eval(tmp_path / "eval" / "all_val", np.random.default_rng(1), ratios=(4.0,))
    sources = pd.read_parquet(eval_dir / "sources.parquet")
    sources.drop(columns=[c for c in sources if c.startswith("kron_radius")]).to_parquet(eval_dir / "sources.parquet")
    written = plot_eval(eval_dir, dpi=30, detailed=True)
    assert written and not any("kron" in p.name for p in written)
