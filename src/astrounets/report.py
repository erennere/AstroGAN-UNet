"""Stage `report`: the study in one folder, runs/_workflow/report/ (the workflow's last step).

  ranking.png        stage 1: score of every setup (weighted mean rank, lower is better), top setups marked
  tuning.png         tuning: validation loss per epoch for each tuned setup and learning rate, the choice
  results_<measure>.png   stage 2: trained on (rows) x tested on (columns), sparse and crowded crops,
                     one block per setup; flux scatter, |flux bias|, completeness gain, SSIM
  tables/            ranking.csv, choice.csv, results.csv, results_by_group.csv; final.json (the final
                     model, `astrounets denoise` by default) and final_ranking.csv (how it was chosen)
  models/<training dataset>/<setup>/   best generator (best.keras), its config and best.json; each
                     folder can be given to `astrounets denoise --model`
  index.txt          what is where, including each final model's evaluation figures

Everything is drawn from files the stages wrote; `astrounets report` can be run again at any time.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.config import Config
from astrounets.files import write_csv, write_text
from astrounets.plots import _pyplot, _save, make_plots
from astrounets.workflow import Plan, choice_path, final_path, ranking_path, report_index, results_path

log = logging.getLogger(__name__)

MEASURES = {  # summary key (per group) -> (label, transform)
    "flux_scatter": ("flux scatter [mag]", lambda g: g["flux_scatter_rec"]),
    "flux_bias": ("|flux bias| [mag]", lambda g: abs(g["flux_bias_rec"])),
    "completeness_gain": ("completeness gain (reconstruction − noisy)", lambda g: g["completeness_rec"] - g["completeness_noisy"]),
    "ssim": ("SSIM", lambda g: g["ssim_rec"]),
}


def report_dir(cfg: Config) -> Path:
    return report_index(cfg).parent


def _leaves(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_leaves(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _short(setup) -> str:
    """A setup as 'type=gan, attention=True, mode=z_scale, loss=log_cosh'."""
    setup = json.loads(setup) if isinstance(setup, str) else setup
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in _leaves(setup).items()) or "default"


def plot_ranking(cfg: Config, out: Path) -> Path | None:
    path = ranking_path(cfg)
    if not path.exists():
        return None
    table = pd.read_csv(path)
    ranked = table[table["rank"].notna()].sort_values("rank")
    unranked = table[table["rank"].isna()]
    plt = _pyplot()
    n = len(ranked) + len(unranked)
    fig, ax = plt.subplots(figsize=(9, 0.28 * n + 1.5))
    labels = [_short(s) for s in ranked["setup"]] + [f"{_short(s)} ({st})" for s, st in zip(unranked["setup"], unranked["status"])]
    y = np.arange(n)[::-1]
    top = cfg.workflow.tune.top
    for k, (score, rank, lab) in enumerate(zip(ranked["score"], ranked["rank"], labels)):
        ax.barh(y[k], score, color="tab:red" if "type=gan" in lab else "tab:blue", alpha=1.0 if rank <= top else 0.4)
    for yy in y[len(ranked):]:
        ax.text(0.01, yy, "not ranked (guard or diverged)", va="center", fontsize=7, color="0.4", transform=ax.get_yaxis_transform())
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"score: weighted mean rank over {', '.join(f'{g} ({w:g})' for g, w in cfg.ranking.groups.items())} (lower is better)")
    ax.set_title(f"stage 1: {len(ranked)} ranked setups; the top {top} (solid) go on; red = GAN, blue = U-Net", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out / "ranking.png", cfg.plots.dpi)


def plot_tuning(cfg: Config, out: Path) -> Path | None:
    if not choice_path(cfg).exists():
        return None
    runs = Plan(cfg).runs("tune")
    choice = pd.read_csv(choice_path(cfg))
    setups = list(dict.fromkeys(json.dumps(r.setup, sort_keys=True) for r in runs))
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(setups), figsize=(5.5 * len(setups), 4.2), squeeze=False)
    for ax, setup in zip(axes[0], setups):
        for run in sorted((r for r in runs if json.dumps(r.setup, sort_keys=True) == setup), key=lambda r: r.cfg.train.learning_rate):
            history = run.cfg.run_dir / "history.csv"
            if history.exists():
                h = pd.read_csv(history)
                ax.plot(h["epoch"], h["val_rec_loss"], label=f"lr {run.cfg.train.learning_rate:g}")
        chosen = choice[choice["setup"] == setup]
        if len(chosen):
            c = chosen.iloc[0]
            ax.axvline(c["best_epoch"], color="k", ls="--", lw=0.8)
            ax.set_title(f"{_short(setup)}\nchosen lr {c['learning_rate']:g}, best epoch {c['best_epoch']}, "
                         f"stage 2: {c['epochs']} epochs{' (still improving)' if c['still_improving'] else ''}", fontsize=8)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation reconstruction loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, out / "tuning.png", cfg.plots.dpi)


def results_by_group(cfg: Config) -> pd.DataFrame:
    """One row per stage-2 run x test set x crop group, from the evaluation summaries."""
    from astrounets.evaluate.run import eval_dir

    rows = []
    for run in Plan(cfg).runs("datasets"):
        for dataset, split in run.targets:
            path = eval_dir(run.cfg.run_dir, dataset, split) / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            for group, values in summary.get("groups", {"overall": summary["overall"]}).items():
                row = {"setup": _short(run.setup), "train_on": run.cfg.dataset.name, "test_on": f"{dataset}:{split}",
                       "group": group, "run": run.key}
                row.update(values)
                row.update({k: fn(values) for k, (_, fn) in MEASURES.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def plot_results(cfg: Config, table: pd.DataFrame, out: Path) -> list[Path]:
    if table.empty:
        return []
    plt = _pyplot()
    written = []
    setups = list(dict.fromkeys(table["setup"]))
    groups = [g for g in ("sparse", "crowded") if g in set(table["group"])] or ["overall"]
    train_on = list(dict.fromkeys(table["train_on"]))
    test_on = list(dict.fromkeys(table["test_on"]))
    for key, (label, _) in MEASURES.items():
        fig, axes = plt.subplots(len(setups), len(groups), figsize=(1.6 * len(test_on) * len(groups) + 2, 1.3 * len(train_on) * len(setups) + 1.5), squeeze=False)
        for i, setup in enumerate(setups):
            for j, group in enumerate(groups):
                ax = axes[i][j]
                part = table[(table["setup"] == setup) & (table["group"] == group)]
                grid = part.pivot_table(index="train_on", columns="test_on", values=key).reindex(index=train_on, columns=test_on)
                im = ax.imshow(grid.to_numpy(float), cmap="viridis_r" if key != "ssim" and key != "completeness_gain" else "viridis")
                for (r, c), v in np.ndenumerate(grid.to_numpy(float)):
                    if np.isfinite(v):
                        red, green, blue, _ = im.cmap(im.norm(v))
                        dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.5
                        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8, color="w" if dark else "k")
                ax.set_xticks(range(len(test_on))); ax.set_xticklabels(test_on, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(len(train_on))); ax.set_yticklabels(train_on, fontsize=8)
                ax.set_title(f"{setup}\n{group} crops", fontsize=8)
                if j == 0:
                    ax.set_ylabel("trained on")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"stage 2: {label}", fontsize=10)
        written.append(_save(fig, out / f"results_{key}.png", cfg.plots.dpi))
    return written


def copy_models(cfg: Config, out: Path) -> list[str]:
    """Each final model with its config: models/<training dataset>/<run folder>/."""
    lines = []
    for run in Plan(cfg).runs("datasets"):
        src = run.cfg.run_dir
        if not (src / "checkpoints" / "best.keras").exists():
            continue
        dest = out / "models" / run.cfg.dataset.name / src.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints/best.keras", "checkpoints/best.json", "config.yaml", "train.json", "history.csv"):
            if (src / name).exists():
                shutil.copy2(src / name, dest / Path(name).name)
        lines.append(f"{dest.relative_to(out)}  <- {src}  (figures: {src / 'eval'}/*/plots)")
    return lines


def make_report(cfg: Config) -> Path:
    out = report_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    for path, name in ((ranking_path(cfg), None), (choice_path(cfg), None), (results_path(cfg), None),
                       (final_path(cfg), None), (final_path(cfg).parent / "ranking.csv", "final_ranking.csv")):
        if path.exists():
            shutil.copy2(path, tables / (name or path.name))
    final = []
    if final_path(cfg).exists():
        chosen = json.loads(final_path(cfg).read_text(encoding="utf-8"))
        final = [f"  models/{chosen['run']}  (best of {chosen['candidates']} on {chosen['selected_on']}, "
                 f"score {chosen['score']:.3f}; tables/final_ranking.csv)"]
    figures = [p for p in [plot_ranking(cfg, out), plot_tuning(cfg, out)] if p]
    models = []
    if results_path(cfg).exists():
        by_group = results_by_group(cfg)
        write_csv(by_group, tables / "results_by_group.csv")
        figures += plot_results(cfg, by_group, out)
        if cfg.plots.hexbins:  # the detailed figures of the final models, where a job has not drawn them
            for run in Plan(cfg).runs("datasets"):
                if (run.cfg.run_dir / "eval").exists() and not list((run.cfg.run_dir / "eval").glob("*/plots/hexbin")):
                    make_plots(cfg, run.cfg.run_dir, detailed=True)
        models = copy_models(cfg, out)
    index = ["Study report (astrounets report); see src/astrounets/report.py for what each file shows.", "",
             "figures:", *[f"  {p.relative_to(out)}" for p in figures], "",
             "tables:", *[f"  tables/{p.name}" for p in sorted(tables.glob("*.csv"))], "",
             "final model (what `astrounets denoise` uses unless given --model):", *(final or ["  not chosen yet"]), "",
             "final models (stage 2) and where their evaluation figures are:", *[f"  {line}" for line in models]]
    write_text(report_index(cfg), "\n".join(index) + "\n")
    log.info("report written to %s", out)
    return report_index(cfg)
