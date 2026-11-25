"""`astrounets workflow | status | stop`: start Snakemake for the whole study, in the background.

Same command on a laptop and on Helix: with `sbatch` on the PATH the SLURM profile is used (every
run becomes a SLURM job, fetch and the quick summary steps run where the workflow runs), otherwise
the local profile (one job at a time on this machine). The workflow keeps running after the console
is closed (like `nohup ... &`); its log is runs/_workflow/logs/<start time>.log. Starting again
continues where it stopped: finished runs are kept, interrupted runs resume from their checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from astrounets.config import Config, find_root, load_config, save_config
from astrounets.files import write_json
from astrounets.workflow import choice_path, progress, ranking_path, results_path, workflow_dir

UNTIL = ("data", "models", "tune", "datasets")


def _state_path(cfg: Config) -> Path:
    return workflow_dir(cfg) / "workflow.json"


def _read_state(cfg: Config) -> dict | None:
    path = _state_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_alive(pid: int) -> bool:
    """Whether `pid` is a running Snakemake process (guards against reused process ids)."""
    import psutil

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and \
            any("snakemake" in part for part in proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def running_pid(cfg: Config) -> int | None:
    state = _read_state(cfg)
    if state and is_alive(int(state["pid"])):
        return int(state["pid"])
    return None


def freeze(cfg: Config) -> Path:
    """Write the configuration the workflow (and every job) uses, with absolute data and run paths."""
    from astrounets.config import replace

    frozen = replace(cfg, {"paths": {"data": cfg.data_dir.as_posix(), "runs": cfg.runs_dir.as_posix()}})
    path = workflow_dir(cfg) / "config.yaml"
    save_config(frozen, path)
    return path


def snakemake_command(frozen: Path, profile: str, until: str | None, dry_run: bool) -> list[str]:
    root = find_root(Path(__file__).parent)
    cfg = load_config(frozen)
    targets: list[str] = []  # none = everything (rule all)
    if until == "data":
        names = sorted({cfg.dataset.name, *cfg.workflow.datasets.train_on,
                        *(item.split(":")[0] for item in cfg.workflow.datasets.evaluate)} - {"own"})
        targets = [(cfg.dataset_dir(n) / "crops.parquet").as_posix() for n in names]
    elif until is not None:
        targets = [{"models": ranking_path, "tune": choice_path, "datasets": results_path}[until](cfg).as_posix()]
    command = [sys.executable, "-m", "snakemake", *targets,  # targets first: --config takes every later word
               "--snakefile", (root / "workflow" / "Snakefile").as_posix(),
               "--directory", workflow_dir(cfg).as_posix(),
               "--profile", (root / "workflow" / "profiles" / profile).as_posix()]
    if dry_run:
        command.append("--dry-run")
    return command + ["--config", f"astrounets_config={frozen.as_posix()}"]


def choose_profile(local: bool) -> str:
    return "local" if local or shutil.which("sbatch") is None else "slurm"


def _environment(root: Path) -> dict:
    env = dict(os.environ)
    env["ASTROUNETS_ROOT"] = str(root)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    scripts = str(Path(sys.executable).parent)
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    return env


def _spawn_detached(command: list[str], log_file: Path, env: dict, cwd: Path) -> int:
    """Start `command` so that it survives closing the console (nohup on Linux, a hidden console of
    its own on Windows); its output goes to `log_file`."""
    with open(log_file, "ab") as out:
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            try:  # leave the console's job object too, if the console allows it
                proc = subprocess.Popen(command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
            except OSError:
                proc = subprocess.Popen(command, creationflags=flags, **kwargs)
        else:
            proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    return proc.pid


def start(config: Path, overrides: list[str], local: bool = False, foreground: bool = False,
          dry_run: bool = False, until: str | None = None) -> int:
    cfg = load_config(config, overrides)
    pid = running_pid(cfg)
    if pid is not None:
        print(f"the workflow is already running (process {pid}); `astrounets status` shows its progress, "
              "`astrounets stop` stops it")
        return 1
    root = find_root(Path(__file__).parent)
    profile = choose_profile(local)
    frozen = freeze(cfg)
    command = snakemake_command(frozen, profile, until, dry_run)
    env = _environment(root)
    wdir = workflow_dir(cfg)
    print(f"workflow: {'SLURM jobs' if profile == 'slurm' else 'this machine'} (profile {profile}), "
          f"config {Path(config).as_posix()}, results in {cfg.runs_dir.as_posix()}")
    if dry_run:
        return subprocess.run(command, env=env, cwd=wdir).returncode
    # No workflow of ours is running, so a lock left by one that was killed can go.
    subprocess.run(command[:-2] + ["--unlock"] + command[-2:], env=env, cwd=wdir, capture_output=True)
    state = {"started": datetime.now().isoformat(timespec="seconds"), "profile": profile, "until": until,
             "config": Path(config).as_posix(), "command": command}
    if foreground:
        proc = subprocess.Popen(command, env=env, cwd=wdir)
        write_json(_state_path(cfg), {"pid": proc.pid, "log": None, **state})
        try:
            return proc.wait()
        except KeyboardInterrupt:  # Snakemake got the Ctrl-C too and cancels its jobs
            return proc.wait()
    logs = wdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / f"{datetime.now():%Y%m%d-%H%M%S}.log"
    pid = _spawn_detached(command, log_file, env, wdir)
    write_json(_state_path(cfg), {"pid": pid, "log": log_file.as_posix(), **state})
    time.sleep(3)
    if not is_alive(pid):
        print(f"the workflow stopped right away; see {log_file.as_posix()}:\n{_tail(log_file, 30)}")
        return 1
    print(f"running in the background (process {pid}); you can close this console.\n"
          f"  log:      {log_file.as_posix()}\n  progress: astrounets status\n  stop:     astrounets stop")
    return 0


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def status(config: Path, overrides: list[str], lines: int = 15) -> str:
    cfg = load_config(config, overrides)
    frozen = workflow_dir(cfg) / "config.yaml"
    if frozen.exists():
        cfg = load_config(frozen)
    state = _read_state(cfg)
    out = []
    if state is None:
        out.append(f"no workflow started yet for {cfg.runs_dir.as_posix()}")
    else:
        alive = is_alive(int(state["pid"]))
        out.append(f"workflow {'running' if alive else 'not running'} (process {state['pid']}, started "
                   f"{state['started']}, profile {state['profile']})")
    for stage, text in progress(cfg):
        out.append(f"  {stage:<9} {text}")
    if state is not None and state["log"]:
        out.append(f"log {state['log']} (last lines):")
        out.extend("  " + line for line in _tail(Path(state["log"]), lines).splitlines())
    elif state is not None:
        out.append("started in the foreground: its output went to that console")
    return "\n".join(out)


def stop(config: Path, overrides: list[str], wait: float = 60.0) -> int:
    """Stop the background workflow. Snakemake cancels its running jobs (scancel on SLURM); training
    resumes from the last finished epoch when the workflow is started again."""
    import psutil

    cfg = load_config(config, overrides)
    pid = running_pid(cfg)
    if pid is None:
        print("no workflow is running")
        return 0
    proc = psutil.Process(pid)
    children = proc.children(recursive=True)
    if os.name == "nt":
        proc.kill()
    elif os.getpgid(pid) == pid:  # started in the background: its own process group
        os.killpg(pid, signal.SIGINT)  # Snakemake cancels its jobs on Ctrl-C
    else:  # started in the foreground of some console: signal only Snakemake itself
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=wait)
    except psutil.TimeoutExpired:
        proc.kill()
    for child in children:  # local jobs that outlived Snakemake
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    print(f"stopped the workflow (process {pid}); `astrounets workflow` continues where it stopped")
    return 0
