import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from astrounets.config import load_config
from astrounets.launch import (_spawn_detached, choose_profile, freeze, is_alive, running, snakemake_command,
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


@pytest.mark.slow
def test_setups_added_after_the_ranking_are_trained_on_the_next_start(tmp_path):
    """Stage 1 done and ranked, then the grid is extended (a study stopped, updated, started again): the
    next start trains only the new run and ranks again; nothing is fetched or rebuilt. (With the mtime
    trigger alone Snakemake saw nothing to do, since ranking.csv was newer than all its old inputs.)"""
    def workflow(grid, *extra):
        args = [a for o in _overrides(tmp_path) for a in ("--set", o)] + ["--set", f"workflow.models.grid={grid}"]
        return subprocess.run([sys.executable, "-m", "astrounets", "workflow", "-c", str(SMOKE), "--local",
                               "--until", "models", *args, *extra], capture_output=True, text=True, cwd=tmp_path)

    first = workflow("{model.type: [unet]}", "--foreground")
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    dry = workflow("{model.type: [unet, gan]}", "--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    table = dry.stdout[dry.stdout.index("Job stats"):].split("-------  -------\n", 1)[1].split("\n\n", 1)[0]
    counts = {line.split()[0]: int(line.split()[1]) for line in table.splitlines() if line.strip()}
    assert counts == {"job": 1, "ranking": 1, "total": 2}, counts


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
        assert running(cfg)["pid"] == pid
        assert start(SMOKE, _overrides(tmp_path)) == 1, "a second workflow on the same runs folder is refused"
        assert "workflow running" in status(SMOKE, _overrides(tmp_path))
        assert stop(SMOKE, _overrides(tmp_path), wait=10) == 0
        assert not is_alive(pid) and running(cfg) is None
    finally:
        if is_alive(pid):
            os.kill(pid, 9)


def test_is_alive_ignores_other_processes():
    assert not is_alive(os.getpid()), "the test runner is not a snakemake process"
    assert not is_alive(2 ** 22 + 12345)


def _failed_job_logs(state: Path, lines: int = 40) -> str:
    """The end of the log of every failed stand-in SLURM job, for the assertion message."""
    text = ""
    for path in sorted(state.glob("*.json")):
        job = json.loads(path.read_text())
        if job.get("state") == "FAILED" and job.get("log") and Path(job["log"]).exists():
            tail = Path(job["log"]).read_text(errors="replace").splitlines()[-lines:]
            text += f"\n----- job {job['id']} ({job['name']}) -----\n" + "\n".join(tail)
    return text


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="the fake SLURM uses POSIX shells")
def test_slurm_profile_runs_the_whole_workflow(tmp_path, monkeypatch):
    """The Helix profile against a stand-in SLURM: every run is its own sbatch job."""
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
    assert out.returncode == 0, out.stderr[-3000:] + _failed_job_logs(state)

    jobs = [json.loads(p.read_text()) for p in sorted(state.glob("*.json"))]
    gpu = [j for j in jobs if j["partition"] == "gpu-single"]
    assert len(gpu) == 7 and all(j["gres"] == "gpu:A40:1" for j in gpu)  # 2 models + 2 tune + 2 datasets runs + final
    assert len([j for j in jobs if j["partition"] == "cpu-single"]) == 2  # the two datasets
    assert all(j["state"] == "COMPLETED" for j in jobs)
    results = pd.read_csv(workflow_dir(cfg) / "datasets" / "results.csv")
    assert len(results) == 4 and results["flux_scatter_overall"].notna().all()
    logs = list((workflow_dir(cfg) / ".snakemake" / "slurm_logs").rglob("*.log"))
    assert len(logs) == len(jobs)


def test_status_and_stop_reach_a_workflow_running_as_a_slurm_job(tmp_path, monkeypatch):
    import astrounets.launch as launch

    cfg = load_config(SMOKE, _overrides(tmp_path))
    state = workflow_dir(cfg) / "workflow.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"pid": 1, "slurm_job": "4242", "host": "gpu-node", "started": "now",
                                 "profile": "slurm", "log": None}))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(launch, "slurm_job_state", lambda job: "RUNNING" if job == "4242" else None)
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert running(cfg)["slurm_job"] == "4242"
    text = status(SMOKE, _overrides(tmp_path))
    assert "workflow running (SLURM job 4242 on gpu-node" in text and "workflow-4242.log" in text
    assert start(SMOKE, _overrides(tmp_path)) == 1, "a second workflow is refused while the job runs"
    assert stop(SMOKE, _overrides(tmp_path)) == 0
    assert calls == [["scancel", "--signal=USR2", "--batch", "4242"]]
    monkeypatch.setattr(launch, "slurm_job_state", lambda job: None)
    assert running(cfg) is None and "workflow not running" in status(SMOKE, _overrides(tmp_path))


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt" or not (ROOT / ".venv" / "bin" / "astrounets").exists(),
                    reason="needs the project's Linux .venv and POSIX shells")
def test_helix_job_script_resubmits_itself_before_the_time_limit(tmp_path):
    """workflow/helix/run.sbatch under the stand-in SLURM: on the time-limit warning (USR1) Snakemake
    cancels its jobs and the script submits itself again, to start once this job has ended."""
    import fake_slurm
    import yaml

    state = tmp_path / "slurm"
    fake_slurm.install(tmp_path / "bin", state)
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", ASTROUNETS_ROOT=str(ROOT))
    env.pop("SLURM_JOB_ID", None)
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump({"base": str(SMOKE), "paths": {"data": str(tmp_path / "data"),
                                                                     "runs": str(tmp_path / "runs")}}))
    log = tmp_path / "workflow-%j.log"  # not into the repository, where run.sbatch would put it
    job = subprocess.run(["sbatch", f"--output={log}", "workflow/helix/run.sbatch", str(config)], cwd=ROOT,
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    marker = tmp_path / "runs" / "_workflow" / "models" / "done"
    for _ in range(240):  # until the first training jobs are under way
        if any(json.loads(p.read_text()).get("partition") == "gpu-single" for p in state.glob("*.json")):
            break
        time.sleep(0.5)
    subprocess.run(["scancel", "--signal=USR1", "--batch", job], env=env, check=True)
    pending = []
    for _ in range(240):
        pending = [r for r in (json.loads(p.read_text()) for p in state.glob("*.json")) if r.get("dependency")]
        if pending:
            break
        time.sleep(0.5)
    assert pending and pending[0]["dependency"] == f"afterany:{job}"
    assert pending[0]["command"].endswith(f"workflow/helix/run.sbatch {config}")
    for _ in range(120):  # the old controller has ended and cancelled its training jobs
        if json.loads((state / f"{job}.json").read_text())["state"] != "RUNNING":
            break
        time.sleep(0.5)
    jobs = [json.loads(p.read_text()) for p in state.glob("*.json")]
    assert json.loads((state / f"{job}.json").read_text())["state"] != "RUNNING"
    assert all(j["state"] != "RUNNING" for j in jobs), "no training job is left running"
    assert not marker.exists() or len(list(marker.rglob("*.json"))) < 2
    first = json.loads((state / f"{job}.json").read_text())
    assert first["partition"] == "cpu-single"  # from the #SBATCH lines of run.sbatch
    assert (tmp_path / f"workflow-{job}.log").exists() and not (ROOT / f"workflow-{job}.log").exists()
