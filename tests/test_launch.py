import json
import os
import subprocess
import sys
import time

import pytest

from astrounets.config import load_config
from astrounets.launch import (_spawn_detached, choose_profile, freeze, is_alive, running_pid, snakemake_command,
                               start, status, stop)
from astrounets.workflow import workflow_dir
from conftest import ROOT, SMOKE


def _overrides(tmp_path):
    return [f"paths.data={(tmp_path / 'data').as_posix()}", f"paths.runs={(tmp_path / 'runs').as_posix()}"]


def test_profile_follows_sbatch(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
    assert choose_profile(local=False) == "slurm" and choose_profile(local=True) == "local"
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert choose_profile(local=False) == "local"


def test_frozen_config_has_absolute_paths(tmp_path):
    cfg = load_config(SMOKE, ["paths.data=relative/data", f"paths.runs={(tmp_path / 'runs').as_posix()}"])
    frozen = load_config(freeze(cfg))
    assert frozen.data_dir == ROOT / "relative" / "data" and frozen.runs_dir == tmp_path / "runs"
    assert frozen.run_dir == cfg.run_dir


def test_snakemake_command(tmp_path):
    frozen = freeze(load_config(SMOKE, _overrides(tmp_path)))
    command = snakemake_command(frozen, "slurm", "models", dry_run=True)
    assert command[:3] == [sys.executable, "-m", "snakemake"]
    assert command[command.index("--profile") + 1].endswith("workflow/profiles/slurm")
    assert command[command.index("--directory") + 1] == (tmp_path / "runs" / "_workflow").as_posix()
    assert command[-2:] == ["--config", f"astrounets_config={frozen.as_posix()}"], "--config swallows later words"
    assert command[3].endswith("models/ranking.csv") and "--dry-run" in command
    data = snakemake_command(frozen, "local", "data", dry_run=False)
    assert [c.rsplit("/", 2)[-2] for c in data[3:5]] == ["mock", "mock-faber"]


@pytest.mark.parametrize("until, rules, absent", [
    (None, ("fetch", "dataset", "job", "ranking", "choice", "results"), ()),
    ("data", ("fetch", "dataset"), ("job", "ranking")),
    ("models", ("fetch", "dataset", "job", "ranking"), ("choice", "results")),
])
def test_dry_run_lists_the_jobs(tmp_path, until, rules, absent):
    extra = ["--until", until] if until else []
    out = subprocess.run([sys.executable, "-m", "astrounets", "workflow", "-c", str(SMOKE), "--dry-run", *extra,
                          *[a for o in _overrides(tmp_path) for a in ("--set", o)]],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    table = out.stdout[out.stdout.index("Job stats"):].split("-------  -------\n", 1)[1].split("\n\n", 1)[0]
    listed = {line.split()[0] for line in table.splitlines() if line.strip()}
    assert set(rules) <= listed and not set(absent) & listed, listed
    assert not (tmp_path / "data").exists(), "a dry run creates no data"


def test_detached_process_survives_and_stops(tmp_path):
    log = tmp_path / "bg.log"
    pid = _spawn_detached([sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)", "snakemake"],
                          log, dict(os.environ), tmp_path)
    try:
        time.sleep(1)
        assert is_alive(pid) and "up" in log.read_text()
        if os.name != "nt":
            assert os.getsid(pid) == pid, "the workflow runs in its own session, like nohup"
        cfg = load_config(SMOKE, _overrides(tmp_path))
        state = workflow_dir(cfg) / "workflow.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"pid": pid, "started": "now", "profile": "local", "log": log.as_posix()}))
        assert running_pid(cfg) == pid
        assert start(SMOKE, _overrides(tmp_path)) == 1, "a second workflow on the same runs folder is refused"
        assert "workflow running" in status(SMOKE, _overrides(tmp_path))
        assert stop(SMOKE, _overrides(tmp_path), wait=10) == 0
        assert not is_alive(pid) and running_pid(cfg) is None
    finally:
        if is_alive(pid):
            os.kill(pid, 9)


def test_is_alive_ignores_other_processes():
    assert not is_alive(os.getpid()), "the test runner is not a snakemake process"
    assert not is_alive(2 ** 22 + 12345)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="the fake SLURM uses POSIX shells")
def test_slurm_profile_runs_the_whole_workflow(tmp_path, monkeypatch):
    """The Helix profile against a stand-in SLURM: every run is an sbatch job with its own TMPDIR."""
    import fake_slurm
    import pandas as pd

    state = tmp_path / "slurm"
    fake_slurm.install(tmp_path / "bin", state)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ASTROUNETS_ROOT", str(ROOT))
    assert choose_profile(local=False) == "slurm"
    cfg = load_config(SMOKE, _overrides(tmp_path))
    command = snakemake_command(freeze(cfg), "slurm", None, dry_run=False)
    command += ["--slurm-init-seconds-before-status-checks", "2"]
    out = subprocess.run(command, cwd=workflow_dir(cfg), capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]

    jobs = [json.loads(p.read_text()) for p in sorted(state.glob("*.json"))]
    gpu = [j for j in jobs if j["partition"] == "gpu-single"]
    assert len(gpu) == 6 and all(j["gres"] == "gpu:A40:1" for j in gpu)  # 2 models + 2 tune + 2 datasets runs
    assert len([j for j in jobs if j["partition"] == "cpu-single"]) == 2  # the two datasets
    assert all(j["state"] == "COMPLETED" for j in jobs)
    assert any((state / "tmp" / str(j["id"]) / "astrounets" / "originals").is_dir() for j in gpu), \
        "training jobs read their images from the job's TMPDIR"
    results = pd.read_csv(workflow_dir(cfg) / "datasets" / "results.csv")
    assert len(results) == 4 and results["flux_scatter"].notna().all()
    logs = list((workflow_dir(cfg) / ".snakemake" / "slurm_logs").rglob("*.log"))
    assert len(logs) == len(jobs) and any("node scratch" in p.read_text() for p in logs)
