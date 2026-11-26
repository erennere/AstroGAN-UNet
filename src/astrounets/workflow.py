"""The study as three stages of runs. workflow/Snakefile schedules them (`astrounets workflow`).

models    stage 1: every valid combination of workflow.models.grid, trained on the config's dataset
          and ranked on the first workflow.models.evaluate target by ranking.weights (ranks averaged
          over the crop groups in ranking.groups with their weights: sparse fields first)
tune      the `top` best setups of stage 1, each trained once per rate in workflow.tune.learning_rates
          until its validation loss stops improving (patience). Per setup, the rate with the lowest
          validation loss wins, and the epochs it needed (times epoch_margin, within min_epochs and
          max_epochs) become the setup's epoch budget
datasets  stage 2: every tuned setup trained on each dataset in train_on, with the chosen rate and
          epochs, and tested on every workflow.datasets.evaluate target

A job trains one run (resuming from its checkpoint), evaluates it on its targets and draws its
plots. Runs whose losses turn NaN are recorded as diverged and never ranked.

Files under runs/_workflow/
  config.yaml                         the configuration the workflow runs with (frozen at start)
  models/ranking.csv                  stage-1 ranking
  tune/choice.csv                     chosen learning rate and epochs per setup
  datasets/results.csv                final table: setup x training dataset x test dataset
  <stage>/done/<dataset>/<run>.json   one marker per finished job
  logs/                               workflow logs (`astrounets status` shows the latest)
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from astrounets.config import (CROP_GROUPS, Config, ConfigError, RANKING_MEASURES, _deep_merge, dataset_key, fetch_key,
                               from_dict, parse_overrides, to_dict, with_dataset)
from astrounets.files import write_csv, write_json

log = logging.getLogger(__name__)
STAGES = ("models", "tune", "datasets")
HIGHER_IS_BETTER = {"ssim"}


@dataclass
class Run:
    stage: str
    key: str                                 # "<dataset>/<run folder>", unique within a stage
    cfg: Config
    setup: dict                              # the grid values that define the model setup
    targets: list[tuple[str, str]] = field(default_factory=list)  # (dataset, split) to evaluate on


# ------------------------------------------------------------------------------------- locations
def workflow_dir(cfg: Config) -> Path:
    return cfg.runs_dir / "_workflow"


def ranking_path(cfg: Config) -> Path:
    return workflow_dir(cfg) / "models" / "ranking.csv"


def choice_path(cfg: Config) -> Path:
    return workflow_dir(cfg) / "tune" / "choice.csv"


def results_path(cfg: Config) -> Path:
    return workflow_dir(cfg) / "datasets" / "results.csv"


def fetch_marker(cfg: Config) -> Path:
    """data/mast/fetched-<key>.json: written when the fetch stage is done for this download plan."""
    return cfg.mast_dir / f"fetched-{fetch_key(cfg)}.json"


def mark_fetched(cfg: Config) -> Path:
    table = pd.read_parquet(cfg.mast_dir / "metadata.parquet", columns=["id", "path"])
    return write_json(fetch_marker(cfg), {"key": fetch_key(cfg), "images": int(len(table)),
                                         "downloaded": int(table["path"].notna().sum()),
                                         "at": datetime.now().isoformat(timespec="seconds")})


def marker_path(cfg: Config, stage: str, key: str) -> Path:
    return workflow_dir(cfg) / stage / "done" / f"{key}.json"


def run_key(cfg: Config) -> str:
    return f"{cfg.dataset.name}/{cfg.run_dir.name}"


def targets(cfg: Config, items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        dataset, _, split = item.partition(":")
        out.append((cfg.dataset.name if dataset == "own" else dataset, split or cfg.eval.split))
    return out


def _nested(dotted: dict) -> dict:
    return parse_overrides([f"{k}={json.dumps(v)}" for k, v in dotted.items()])


def _setup_text(setup: dict) -> str:
    return json.dumps(setup, sort_keys=True)


# ---------------------------------------------------------------------------------- stage runs
def stage_runs(cfg: Config, stage: str) -> tuple[list[Run], list[tuple[dict, str]]]:
    """The runs of a stage and the grid combinations skipped as invalid (models only).

    tune needs models/ranking.csv and datasets needs tune/choice.csv."""
    base = to_dict(cfg)
    runs: list[Run] = []
    skipped: list[tuple[dict, str]] = []
    w = cfg.workflow
    if stage == "models":
        stage_base = _deep_merge(base, _nested(w.models.set))
        keys = list(w.models.grid)
        for combo in itertools.product(*[w.models.grid[k] for k in keys]):
            setup = _nested(dict(zip(keys, combo)))
            try:
                run_cfg = from_dict(_deep_merge(stage_base, setup), cfg.root)
            except ConfigError as err:
                skipped.append((setup, str(err).split("\n  - ", 1)[-1]))
                continue
            runs.append(Run(stage, run_key(run_cfg), run_cfg, setup, targets(run_cfg, w.models.evaluate)))
    elif stage == "tune":
        t = w.tune
        for setup in _top_setups(cfg):
            for rate in t.learning_rates:
                tuning = {"train": {"learning_rate": rate, "epochs": t.max_epochs, "patience": t.patience, "lr_drops": []}}
                run_cfg = from_dict(_deep_merge(_deep_merge(base, setup), tuning), cfg.root)
                runs.append(Run(stage, run_key(run_cfg), run_cfg, setup, []))
    elif stage == "datasets":
        for setup, tuned in _tuned_setups(cfg):
            for name in w.datasets.train_on:
                overrides = _deep_merge(_deep_merge(setup, tuned), {"dataset": {"name": name}})
                run_cfg = from_dict(_deep_merge(base, overrides), cfg.root)
                runs.append(Run(stage, run_key(run_cfg), run_cfg, setup, targets(run_cfg, w.datasets.evaluate)))
    else:
        raise ConfigError(f"unknown stage '{stage}'; stages are {STAGES}")
    _check_unique(runs, stage)
    _check_datasets(cfg, runs)
    return runs, skipped


def _check_unique(runs: list[Run], stage: str) -> None:
    seen: dict[str, Run] = {}
    for run in runs:
        if run.key in seen:
            raise ConfigError(f"stage {stage}: setups {_setup_text(seen[run.key].setup)} and "
                              f"{_setup_text(run.setup)} are the same run")
        seen[run.key] = run


def _check_datasets(cfg: Config, runs: list[Run]) -> None:
    """Datasets are built once from the workflow config, so no run may change dataset settings."""
    expected: dict[str, str] = {}
    for run in runs:
        for name in {run.cfg.dataset.name, *(d for d, _ in run.targets)}:
            want = expected.setdefault(name, dataset_key(with_dataset(cfg, name)))
            if dataset_key(with_dataset(run.cfg, name)) != want:
                raise ConfigError(f"setup {_setup_text(run.setup)} changes the settings of dataset '{name}'; "
                                  "datasets are built once from the main config, so change them there")


def _top_setups(cfg: Config) -> list[dict]:
    path = ranking_path(cfg)
    if not path.exists():
        raise ConfigError(f"{path} not found: the models stage has to finish first")
    table = pd.read_csv(path)
    ranked = table[table["rank"].notna()].sort_values("rank").head(cfg.workflow.tune.top)
    if ranked.empty:
        raise ConfigError("no stage-1 run passed the completeness/purity guard")
    return [json.loads(text) for text in ranked["setup"]]


def _tuned_setups(cfg: Config) -> list[tuple[dict, dict]]:
    path = choice_path(cfg)
    if not path.exists():
        raise ConfigError(f"{path} not found: the tune stage has to finish first")
    table = pd.read_csv(path).sort_values("stage1_rank")
    return [(json.loads(row.setup), {"train": {"learning_rate": float(row.learning_rate), "epochs": int(row.epochs)}})
            for row in table.itertuples()]


class Plan:
    """Stage runs for the Snakefile, rebuilt only when the file a stage depends on changes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache: dict[str, tuple[float | None, list[Run]]] = {}

    def _stamp(self, stage: str) -> float | None:
        source = {"tune": ranking_path(self.cfg), "datasets": choice_path(self.cfg)}.get(stage)
        return source.stat().st_mtime if source is not None and source.exists() else None

    def runs(self, stage: str) -> list[Run]:
        stamp = self._stamp(stage)
        cached = self._cache.get(stage)
        if cached is None or cached[0] != stamp:
            cached = (stamp, stage_runs(self.cfg, stage)[0])
            self._cache[stage] = cached
        return cached[1]

    def run(self, stage: str, key: str) -> Run:
        for run in self.runs(stage):
            if run.key == key:
                return run
        raise ConfigError(f"stage {stage} has no run {key}; did the configuration change since the workflow started?")

    def markers(self, stage: str) -> list[str]:
        return [marker_path(self.cfg, stage, run.key).as_posix() for run in self.runs(stage)]

    def job_inputs(self, stage: str, key: str) -> list[str]:
        run = self.run(stage, key)
        names = sorted({run.cfg.dataset.name, *(d for d, _ in run.targets)})
        return [(self.cfg.dataset_dir(n) / "crops.parquet").as_posix() for n in names]


# ------------------------------------------------------------------------------------------ jobs
def _lock_is_stale(path: Path, stale_after: float) -> bool:
    import psutil

    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True  # vanished: try again
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return age > 10  # just created and not written yet, or broken
    if owner.get("host") == socket.gethostname() and not psutil.pid_exists(int(owner.get("pid", -1))):
        return True
    return age > stale_after


@contextmanager
def run_lock(run_dir: Path, stale_after: float = 600.0, poll: float = 30.0, beat: float = 60.0):
    """Hold <run>/run.lock while a job works on a run folder.

    If the workflow is restarted while a job of the earlier start still runs (say the workflow job
    was cancelled), Snakemake submits the run again; the new job waits here until the old one is done
    instead of writing into the same folder. A lock whose holder died is taken over once it is
    `stale_after` seconds old (at once if the holder was a process on this machine)."""
    path = Path(run_dir) / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    me = {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_is_stale(path, stale_after):
                path.unlink(missing_ok=True)
                continue
            log.info("%s is in use by another job; waiting", run_dir)
            time.sleep(poll)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(me))
        break
    done = threading.Event()

    def heartbeat():
        while not done.wait(beat):
            try:
                os.utime(path)
            except OSError:
                pass

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
        path.unlink(missing_ok=True)


def run_job(run: Run) -> dict:
    """Train (or resume), evaluate on every target not evaluated yet, draw plots."""
    with run_lock(run.cfg.run_dir):
        return _run_job(run)


def _run_job(run: Run) -> dict:
    from astrounets.evaluate.run import eval_dir, evaluate
    from astrounets.plots import make_plots
    from astrounets.train import train, train_result

    cfg, run_dir = run.cfg, run.cfg.run_dir
    train(cfg)
    result = train_result(run_dir)
    evaluations = []
    if result["status"] != "diverged":
        for dataset, split in run.targets:
            if not (eval_dir(run_dir, dataset, split) / "summary.json").exists():
                evaluate(cfg, run_dir, dataset, split)
            evaluations.append(f"{dataset}:{split}")
    make_plots(cfg, run_dir)
    return {"run": run.key, "train": result, "evaluated": evaluations}


def job(cfg: Config, stage: str, key: str) -> Path:
    """One Snakemake job: run `key` of `stage` and write its done-marker."""
    run = Plan(cfg).run(stage, key)
    out = marker_path(cfg, stage, key)
    write_json(out, run_job(run))
    return out


def run_one(cfg: Config, items: list[str]) -> Path:
    """A single run outside the workflow (`astrounets run`)."""
    run_job(Run("single", run_key(cfg), cfg, {}, targets(cfg, items)))
    return cfg.run_dir


def call(*args: str) -> None:
    """Run `python -m astrounets <args>` with this interpreter (used by the Snakefile)."""
    command = [sys.executable, "-m", "astrounets", *args]
    print("+ astrounets " + " ".join(args), flush=True)
    subprocess.run(command, check=True)


def _marker(cfg: Config, stage: str, run: Run) -> dict:
    path = marker_path(cfg, stage, run.key)
    if not path.exists():
        raise RuntimeError(f"stage {stage} is not complete ({run.key} has not run); start `astrounets workflow`")
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_summary(run: Run, dataset: str, split: str) -> dict:
    from astrounets.evaluate.run import eval_dir

    return json.loads((eval_dir(run.cfg.run_dir, dataset, split) / "summary.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- stage results (select)
def select(cfg: Config, stage: str) -> Path:
    """Summarise a finished stage: models -> ranking.csv, tune -> choice.csv, datasets -> results.csv."""
    return {"models": rank_models, "tune": choose_tuning, "datasets": collect_results}[stage](cfg)


def rank_models(cfg: Config) -> Path:
    rows = []
    for run in Plan(cfg).runs("models"):
        status = _marker(cfg, "models", run)["train"]["status"]
        row = {"key": run.key, "setup": _setup_text(run.setup), "status": status}
        if status != "diverged":
            row.update(_eval_summary(run, *run.targets[0])["ranking"])
        rows.append(row)
    table = rank_table(pd.DataFrame(rows), cfg.ranking.weights, cfg.ranking.guard, cfg.ranking.groups)
    out = write_csv(table, ranking_path(cfg))
    shown = table.drop(columns=["setup"]).to_string(index=False)
    log.info("stage-1 ranking written to %s\n%s", out, shown)
    if table["rank"].isna().all():
        raise RuntimeError("no stage-1 run passed the guard (completeness and purity at least those of the noisy "
                           "input); see the table above. Train longer or set ranking.guard to false.")
    return out


def rank_table(table: pd.DataFrame, weights: dict[str, float], guard: bool = True,
               groups: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted mean of per-measure ranks over runs that trained and pass the guard; the others get no rank.

    Columns are <measure>_<group> and guard_ok_<group>. A measure's rank is the weighted mean of its
    ranks within the groups (`groups`: name -> weight) that the evaluation set has, so a group's say
    does not depend on how many crops or sources it holds; the guard must hold in each of them."""
    table = table.copy()
    groups = groups or {"overall": 1.0}
    present = {g: w for g, w in groups.items() if w > 0 and f"n_samples_{g}" in table
               and pd.to_numeric(table[f"n_samples_{g}"]).fillna(0).gt(0).any()}
    if not present:
        present = {"overall": 1.0}
    for g in present:
        for measure in (*RANKING_MEASURES, "guard_ok"):
            if f"{measure}_{g}" not in table:
                table[f"{measure}_{g}"] = np.nan
    passing = table["status"].ne("diverged") if "status" in table else pd.Series(True, index=table.index)
    if guard:
        for g in present:
            passing &= table[f"guard_ok_{g}"].map(lambda v: v is True or v == 1.0)
    score = np.zeros(len(table))
    total = sum(weights.get(m, 0.0) for m in RANKING_MEASURES)
    for measure in RANKING_MEASURES:
        w = weights.get(measure, 0.0)
        if w == 0:
            continue
        ranks = [table[f"{measure}_{g}"].astype(float).where(passing)
                 .rank(ascending=measure not in HIGHER_IS_BETTER, na_option="bottom").to_numpy() for g in present]
        score += w * np.average(ranks, axis=0, weights=list(present.values()))
    table["score"] = np.where(passing, score / total, np.nan)
    table = table.sort_values("score", na_position="last").reset_index(drop=True)
    table.insert(0, "rank", [i + 1 if ok else None for i, ok in enumerate(table["score"].notna())])
    return table


def choose_tuning(cfg: Config) -> Path:
    t = cfg.workflow.tune
    rows = []
    for run in Plan(cfg).runs("tune"):
        result = _marker(cfg, "tune", run)["train"]
        rows.append({"setup": _setup_text(run.setup), "learning_rate": run.cfg.train.learning_rate, **result})
    table = pd.DataFrame(rows)
    choices = []
    for rank, setup in enumerate(_top_setups(cfg), start=1):
        group = table[(table["setup"] == _setup_text(setup)) & table["status"].ne("diverged")
                      & table["best_val_rec_loss"].notna()]
        if group.empty:
            log.warning("every tuning run of setup %s diverged; the setup is dropped", _setup_text(setup))
            continue
        best = group.loc[group["best_val_rec_loss"].astype(float).idxmin()]
        epochs = int(min(max(math.ceil(best["best_epoch"] * t.epoch_margin), t.min_epochs), t.max_epochs))
        losses = {f"{r:g}": float(v) for r, v in zip(group["learning_rate"], group["best_val_rec_loss"])}
        # best loss in the very last epoch of a run that was not stopped early: it may still be improving
        still_improving = best["status"] == "finished" and int(best["best_epoch"]) >= t.max_epochs
        if still_improving:
            log.warning("setup %s: validation loss still improving after workflow.tune.max_epochs = %d; its "
                        "stage-2 budget is capped there (raise max_epochs for a fair budget)", _setup_text(setup), t.max_epochs)
        choices.append({"stage1_rank": rank, "setup": _setup_text(setup), "learning_rate": float(best["learning_rate"]),
                        "best_epoch": int(best["best_epoch"]), "epochs": epochs, "still_improving": bool(still_improving),
                        "val_loss_by_rate": json.dumps(losses)})
    if not choices:
        raise RuntimeError("every tuning run diverged; lower workflow.tune.learning_rates")
    table = pd.DataFrame(choices)
    out = write_csv(table, choice_path(cfg))
    log.info("tuning choice written to %s\n%s", out, table.drop(columns=["setup"]).to_string(index=False))
    return out


def collect_results(cfg: Config) -> Path:
    rows = []
    for run in Plan(cfg).runs("datasets"):
        marker = _marker(cfg, "datasets", run)
        base = {"setup": run.cfg.run_name, "train_on": run.cfg.dataset.name, "learning_rate": run.cfg.train.learning_rate,
                "epochs": run.cfg.train.epochs, "status": marker["train"]["status"], "run": run.key}
        for dataset, split in run.targets:
            row = {**base, "test_on": f"{dataset}:{split}"}
            if marker["train"]["status"] != "diverged":
                summary = _eval_summary(run, dataset, split)
                row.update(summary["ranking"])
                row.update({f"{k}": v for k, v in summary["overall"].items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    out = write_csv(table, results_path(cfg))
    log.info("results written to %s", out)
    for group in CROP_GROUPS:
        if f"flux_scatter_{group}" in table and table[f"flux_scatter_{group}"].notna().any():
            pivot = table.pivot_table(index=["setup", "train_on"], columns="test_on", values=f"flux_scatter_{group}")
            log.info("flux scatter [mag], %s crops (rows: trained on, columns: tested on)\n%s", group, pivot.round(3).to_string())
    return out


# ---------------------------------------------------------------------------------------- status
def progress(cfg: Config) -> list[tuple[str, str]]:
    """(stage, text) lines for `astrounets status`."""
    lines = []
    waiting = {"tune": "waiting for the models ranking", "datasets": "waiting for the tuning choice"}
    final = {"models": ranking_path(cfg), "tune": choice_path(cfg), "datasets": results_path(cfg)}
    for stage in STAGES:
        try:
            runs, skipped = stage_runs(cfg, stage)
        except ConfigError as err:
            lines.append((stage, waiting.get(stage, str(err))))
            continue
        done = [r for r in runs if marker_path(cfg, stage, r.key).exists()]
        diverged = sum(1 for r in done if json.loads(marker_path(cfg, stage, r.key).read_text(encoding="utf-8"))
                       ["train"]["status"] == "diverged")
        text = f"{len(done)}/{len(runs)} runs done"
        if diverged:
            text += f", {diverged} diverged"
        if skipped:
            text += f" ({len(skipped)} invalid combinations skipped)"
        if final[stage].exists():
            text += f"; {final[stage].name} written"
        lines.append((stage, text))
    return lines
