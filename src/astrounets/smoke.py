"""`astrounets smoke`: the whole workflow (configs/smoke.yaml) on a synthetic sky, in the foreground.

fetch (mock) -> datasets mock and mock-faber -> stage 1 (U-Net and GAN) -> ranking -> tuning
(two learning rates) -> stage 2 on both datasets, tested on both -> results.csv
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from astrounets.config import find_root, load_config


def smoke(fresh: bool = True, workdir: Path | None = None) -> str:
    from astrounets.launch import start
    from astrounets.workflow import results_path

    root = find_root(Path(__file__).parent)
    config = root / "configs" / "smoke.yaml"
    workdir = Path(workdir) if workdir is not None else root / "smoke"
    overrides = [f"paths.data={(workdir / 'data').as_posix()}", f"paths.runs={(workdir / 'runs').as_posix()}"]
    if fresh and workdir.exists():
        shutil.rmtree(workdir)
    code = start(config, overrides, local=True, foreground=True)
    if code != 0:
        raise RuntimeError(f"the smoke workflow failed (exit code {code}); see the output above")
    results = pd.read_csv(results_path(load_config(config, overrides)))
    columns = ["setup", "train_on", "test_on", "flux_scatter_sparse", "flux_scatter_crowded", "ssim_overall",
               "completeness_rec", "completeness_noisy"]
    return results[[c for c in columns if c in results]].to_string(index=False)
