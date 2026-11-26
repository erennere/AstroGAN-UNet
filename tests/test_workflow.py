import json

import pandas as pd
import pytest

from astrounets.config import ConfigError, load_config
from astrounets.workflow import (Plan, choice_path, choose_tuning, marker_path, progress, rank_table, ranking_path,
                                 stage_runs, write_json)
from conftest import DEFAULT, ROOT, SMOKE


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
    for best_epoch, status, expected, improving in ((2, "stopped_early", 20, False), (90, "stopped_early", 100, False),
                                                    (100, "finished", 100, True)):
        run = stage_runs(cfg, "tune")[0][0]
        write_json(marker_path(cfg, "tune", run.key), {"run": run.key, "evaluated": [], "train": {
            "status": status, "epochs": max(best_epoch, 12), "best_epoch": best_epoch, "best_val_rec_loss": 0.1}})
        choose_tuning(cfg)
        choice = pd.read_csv(choice_path(cfg))
        assert choice["epochs"].tolist() == [expected] and choice["still_improving"].tolist() == [improving]


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


def _measures(group, scatter, bias, ssim, radius, ellipticity, guard, n=10):
    return {f"flux_scatter_{group}": scatter, f"flux_bias_{group}": bias, f"ssim_{group}": ssim,
            f"radius_error_{group}": radius, f"ellipticity_error_{group}": ellipticity,
            f"guard_ok_{group}": guard, f"n_samples_{group}": n}


def test_ranking_uses_mean_rank_guard_and_skips_diverged_runs():
    base = {"setup": "{}", "status": "finished"}
    table = pd.DataFrame([
        {**base, "key": "a", **_measures("overall", 0.10, 0.01, 0.90, 0.1, 0.1, True)},
        {**base, "key": "b", **_measures("overall", 0.20, 0.02, 0.80, 0.2, 0.2, True)},
        {**base, "key": "c", **_measures("overall", 0.01, 0.00, 0.99, 0.0, 0.0, False)},
        {"setup": "{}", "key": "d", "status": "diverged"},
    ])
    weights = {"flux_scatter": 1, "flux_bias": 1, "ssim": 1, "radius_error": 0.5, "ellipticity_error": 0.5}
    ranked = rank_table(table, weights)
    assert ranked["key"].tolist()[:2] == ["a", "b"] and ranked["rank"].tolist()[:2] == [1, 2]
    assert ranked["rank"].iloc[2:].isna().all()
    unguarded = rank_table(table, weights, guard=False)
    assert unguarded["key"].tolist()[:3] == ["c", "a", "b"] and pd.isna(unguarded["rank"].iloc[3])


def test_ranking_weighs_crop_groups():
    """'a' is a little better on sparse crops, 'b' much better on crowded ones. Overall measures follow the
    crowded crops (they hold most sources); group weights decide how much each kind of field counts."""
    base = {"setup": "{}", "status": "finished"}
    weights = {"flux_scatter": 1, "flux_bias": 0, "ssim": 0, "radius_error": 0, "ellipticity_error": 0}
    table = pd.DataFrame([
        {**base, "key": "a", **_measures("sparse", 0.10, 0, 0.9, 0, 0, True), **_measures("crowded", 0.50, 0, 0.9, 0, 0, True),
         **_measures("overall", 0.45, 0, 0.9, 0, 0, True)},
        {**base, "key": "b", **_measures("sparse", 0.11, 0, 0.9, 0, 0, True), **_measures("crowded", 0.30, 0, 0.9, 0, 0, True),
         **_measures("overall", 0.28, 0, 0.9, 0, 0, True)},
        {**base, "key": "c", **_measures("sparse", 0.12, 0, 0.9, 0, 0, True), **_measures("crowded", 0.90, 0, 0.9, 0, 0, True),
         **_measures("overall", 0.80, 0, 0.9, 0, 0, True)},
    ])
    assert rank_table(table, weights, groups={"sparse": 1})["key"].tolist() == ["a", "b", "c"]
    assert rank_table(table, weights, groups={"overall": 1})["key"].tolist() == ["b", "a", "c"]
    equal = rank_table(table, weights, groups={"sparse": 1, "crowded": 1})
    assert equal["score"].tolist()[:2] == [1.5, 1.5]  # a and b: one win each
    sparse_first = rank_table(table, weights, groups={"sparse": 3, "crowded": 1})
    assert sparse_first["key"].tolist() == ["a", "b", "c"]
    assert sparse_first["score"].tolist() == [1.25, 1.75, 3.0]  # 0.75 x sparse rank + 0.25 x crowded rank
    # the guard must hold in every weighted group
    table.loc[0, "guard_ok_crowded"] = False
    guarded = rank_table(table, weights, groups={"sparse": 3, "crowded": 1})
    assert guarded["key"].tolist()[:2] == ["b", "c"] and pd.isna(guarded["rank"].iloc[2])
    # an evaluation set without crowded crops is ranked on the sparse ones alone
    sparse_only = table.drop(columns=[c for c in table if c.endswith("_crowded")])
    assert rank_table(sparse_only, weights, groups={"sparse": 3, "crowded": 1})["key"].tolist() == ["a", "b", "c"]


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


def test_full_study_fetches_again_after_a_pilot(tmp_path):
    """A pilot downloads a few images into the shared data/mast; the full study has another download plan,
    so its fetch marker differs and Snakemake fetches again (downloading only what is missing)."""
    from astrounets.workflow import fetch_marker

    paths = [f"paths.data={tmp_path / 'data'}", f"paths.runs={tmp_path / 'runs'}"]
    full = load_config(ROOT / "configs" / "default.yaml", paths)
    pilot = load_config(ROOT / "configs" / "pilot.yaml", paths)
    assert fetch_marker(full) != fetch_marker(pilot)
    assert fetch_marker(full).parent == fetch_marker(pilot).parent == full.mast_dir
    # settings that do not change what is listed or downloaded keep the marker
    same = load_config(ROOT / "configs" / "default.yaml", paths + ["mast.download_workers=3", "train.epochs=7"])
    assert fetch_marker(same) == fetch_marker(full)
    other = load_config(ROOT / "configs" / "default.yaml", paths + ["datasets.faber.pi_names=[FABER, LOTZ]"])
    assert fetch_marker(other) != fetch_marker(full)
