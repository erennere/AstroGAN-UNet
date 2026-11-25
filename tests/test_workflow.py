import json

import pandas as pd
import pytest

from astrounets.config import ConfigError, load_config
from astrounets.workflow import (Plan, choice_path, choose_tuning, marker_path, progress, rank_table, ranking_path,
                                 stage_runs, write_json)
from conftest import DEFAULT, SMOKE


def _study(tmp_path, *overrides):
    return load_config(DEFAULT, [f"paths.data={tmp_path / 'data'}", f"paths.runs={tmp_path / 'runs'}", *overrides])


def _fake_ranking(cfg, order=None):
    """A stage-1 ranking in the given order of the models runs (default: as listed)."""
    runs = stage_runs(cfg, "models")[0]
    order = order or list(range(len(runs)))
    rows = [{"rank": i + 1, "key": runs[j].key, "setup": json.dumps(runs[j].setup, sort_keys=True), "status": "finished"}
            for i, j in enumerate(order)]
    path = ranking_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return [runs[j] for j in order]


def test_stage1_is_every_valid_combination(tmp_path):
    cfg = _study(tmp_path)
    runs, skipped = stage_runs(cfg, "models")
    assert len(runs) == 56 and len(skipped) == 8
    assert all(setup["train"]["loss"] == "ssim" for setup, _ in skipped)
    assert len({r.key for r in runs}) == 56 and all(r.key.startswith("all/") for r in runs)
    assert all(r.cfg.train.epochs == 20 and r.targets == [("all", "val")] for r in runs)


def test_tune_stage_tries_every_rate_for_the_top_setups(tmp_path):
    cfg = _study(tmp_path)
    with pytest.raises(ConfigError, match="models stage has to finish"):
        stage_runs(cfg, "tune")
    ranked = _fake_ranking(cfg, order=[5, 2, 9, 0])
    runs = stage_runs(cfg, "tune")[0]
    t = cfg.workflow.tune
    assert len(runs) == t.top * len(t.learning_rates) == 9
    assert [r.setup for r in runs[::3]] == [r.setup for r in ranked[:3]]
    for run in runs:
        train = run.cfg.train
        assert train.epochs == t.max_epochs and train.patience == t.patience and train.lr_drops == []
        assert run.targets == [] and run.cfg.dataset.name == "all"
    assert sorted({r.cfg.train.learning_rate for r in runs}) == sorted(t.learning_rates)


def test_tuning_choice_picks_the_rate_with_the_lowest_validation_loss(tmp_path):
    cfg = _study(tmp_path)
    _fake_ranking(cfg)
    runs = stage_runs(cfg, "tune")[0]
    losses = {0.0001: 0.30, 0.0002: 0.20, 0.0005: 0.25}
    for i, run in enumerate(runs):
        rate = run.cfg.train.learning_rate
        status = "diverged" if i >= 6 else "stopped_early"  # every rate of the third setup diverged
        write_json(marker_path(cfg, "tune", run.key), {"run": run.key, "train": {
            "status": status, "epochs": 40, "best_epoch": 30 if rate == 0.0002 else 10,
            "best_val_rec_loss": None if status == "diverged" else losses[rate]}, "evaluated": []})
    choose_tuning(cfg)
    choice = pd.read_csv(choice_path(cfg))
    assert len(choice) == 2, "a setup whose tuning runs all diverged is dropped"
    assert choice["learning_rate"].tolist() == [0.0002, 0.0002]
    assert choice["epochs"].tolist() == [36, 36]  # ceil(30 * 1.2), within [min_epochs, max_epochs]

    stage2 = stage_runs(cfg, "datasets")[0]
    assert len(stage2) == 2 * 3
    assert {r.cfg.dataset.name for r in stage2} == {"all", "faber", "momcheva"}
    assert all(r.cfg.train.epochs == 36 and r.cfg.train.learning_rate == 0.0002 for r in stage2)
    assert all(r.cfg.train.lr_drops == [0.75] and r.cfg.train.patience is None for r in stage2)
    assert stage2[0].targets == [("all", "test"), ("faber", "test"), ("momcheva", "test")]
    faber = next(r for r in stage2 if r.cfg.dataset.name == "faber")
    assert faber.cfg.dataset.pi_names == ["FABER"]


def test_epoch_budget_respects_min_and_max(tmp_path):
    cfg = _study(tmp_path, "workflow.tune.top=1", "workflow.tune.learning_rates=[0.0002]")
    _fake_ranking(cfg)
    for best_epoch, expected in ((2, 20), (90, 100)):
        run = stage_runs(cfg, "tune")[0][0]
        write_json(marker_path(cfg, "tune", run.key), {"run": run.key, "evaluated": [], "train": {
            "status": "finished", "epochs": 100, "best_epoch": best_epoch, "best_val_rec_loss": 0.1}})
        choose_tuning(cfg)
        assert pd.read_csv(choice_path(cfg))["epochs"].tolist() == [expected]


def test_setups_may_not_change_dataset_settings(tmp_path):
    with pytest.raises(ConfigError, match="changes the settings of dataset 'all'"):
        stage_runs(_study(tmp_path, "workflow.models.grid={dataset.min_valid_fraction: [0.5, 0.9]}"), "models")


def test_setups_that_are_the_same_run_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="are the same run"):
        stage_runs(_study(tmp_path, "workflow.models.grid={train.loader_workers: [1, 2]}"), "models")


def test_plan_follows_a_new_ranking(tmp_path):
    cfg = _study(tmp_path, "workflow.tune.top=1")
    plan = Plan(cfg)
    first = _fake_ranking(cfg, order=[0, 1])[0]
    assert {r.setup == first.setup for r in plan.runs("tune")} == {True}
    import os, time
    second = _fake_ranking(cfg, order=[1, 0])[0]
    os.utime(ranking_path(cfg), (time.time() + 5, time.time() + 5))
    assert {r.setup == second.setup for r in plan.runs("tune")} == {True}
    key = plan.runs("tune")[0].key
    assert plan.job_inputs("tune", key) == [(cfg.dataset_dir("all") / "crops.parquet").as_posix()]
    with pytest.raises(ConfigError, match="has no run"):
        plan.run("tune", "all/nothing-12345678")


def test_ranking_uses_mean_rank_guard_and_skips_diverged_runs():
    base = {"setup": "{}", "status": "finished"}
    table = pd.DataFrame([
        {**base, "key": "a", "flux_scatter": 0.10, "flux_bias": 0.01, "ssim": 0.90, "radius_error": 0.1, "ellipticity_error": 0.1, "guard_ok": True},
        {**base, "key": "b", "flux_scatter": 0.20, "flux_bias": 0.02, "ssim": 0.80, "radius_error": 0.2, "ellipticity_error": 0.2, "guard_ok": True},
        {**base, "key": "c", "flux_scatter": 0.01, "flux_bias": 0.00, "ssim": 0.99, "radius_error": 0.0, "ellipticity_error": 0.0, "guard_ok": False},
        {"setup": "{}", "key": "d", "status": "diverged"},
    ])
    weights = {"flux_scatter": 1, "flux_bias": 1, "ssim": 1, "radius_error": 0.5, "ellipticity_error": 0.5}
    ranked = rank_table(table, weights)
    assert ranked["key"].tolist()[:2] == ["a", "b"] and ranked["rank"].tolist()[:2] == [1, 2]
    assert ranked["rank"].iloc[2:].isna().all()
    unguarded = rank_table(table, weights, guard=False)
    assert unguarded["key"].tolist()[:3] == ["c", "a", "b"] and pd.isna(unguarded["rank"].iloc[3])


def test_progress_lists_every_stage(tmp_path):
    cfg = load_config(SMOKE, [f"paths.data={tmp_path / 'data'}", f"paths.runs={tmp_path / 'runs'}"])
    lines = dict(progress(cfg))
    assert lines["models"].startswith("0/2 runs done")
    assert lines["tune"] == "waiting for the models ranking" and lines["datasets"] == "waiting for the tuning choice"


def test_run_lock_makes_a_second_job_wait(tmp_path):
    import threading
    import time

    from astrounets.workflow import run_lock

    order = []

    def second():
        with run_lock(tmp_path, poll=0.05):
            order.append("second")

    with run_lock(tmp_path, beat=0.05):
        thread = threading.Thread(target=second)
        thread.start()
        time.sleep(0.3)
        order.append("first")
    thread.join(5)
    assert order == ["first", "second"] and not (tmp_path / "run.lock").exists()


def test_run_lock_takes_over_from_a_dead_holder(tmp_path):
    import socket

    from astrounets.workflow import run_lock

    (tmp_path / "run.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": 2 ** 22 + 99}))
    with run_lock(tmp_path, poll=0.05):
        pass
    (tmp_path / "run.lock").write_text(json.dumps({"host": "another-node", "pid": 1}))
    import os
    os.utime(tmp_path / "run.lock", (0, 0))  # its heartbeat stopped long ago
    with run_lock(tmp_path, poll=0.05, stale_after=60):
        pass
