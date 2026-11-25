"""A tiny stand-in for SLURM so the SLURM profile can be tested without a cluster.

`install(bin_dir, state_dir)` writes executables sbatch, srun, sacct, squeue, scancel, scontrol,
sacctmgr and sshare into bin_dir. sbatch runs the job in the background on this machine with
SLURM_JOB_ID and a per-job TMPDIR set (like Helix's node-local scratch); sacct reports job states.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

COMMANDS = ("sbatch", "srun", "sacct", "squeue", "scancel", "scontrol", "sacctmgr", "sshare")


def install(bin_dir: Path, state_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in COMMANDS:
        path = bin_dir / name
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" {name} '
                        f'"{state_dir}" "$@"\n')
        path.chmod(0o755)


def _next_id(state: Path) -> int:
    counter = state / "counter"
    value = int(counter.read_text()) + 1 if counter.exists() else 1000
    counter.write_text(str(value))
    return value


def sbatch(state: Path, args: list[str]) -> int:
    opts, i = {}, 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--wrap="):
            opts["wrap"] = arg[len("--wrap="):]
        elif arg in ("--job-name", "--output", "-D", "-t", "--mem", "-p", "-A", "--comment", "--gres", "--ntasks"):
            opts[arg] = args[i + 1]
            i += 1
        elif arg.startswith("--gres="):
            opts["--gres"] = arg.split("=", 1)[1]
        i += 1
    job_id = _next_id(state)
    log = opts["--output"].replace("%j", str(job_id))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    tmp = state / "tmp" / str(job_id)
    tmp.mkdir(parents=True)
    env = dict(os.environ, SLURM_JOB_ID=str(job_id), TMPDIR=str(tmp), SLURM_CPUS_PER_TASK="2")
    record = {"id": job_id, "name": opts["--job-name"], "state": "RUNNING", "gres": opts.get("--gres"),
              "partition": opts.get("-p"), "command": opts["wrap"]}
    (state / f"{job_id}.json").write_text(json.dumps(record))
    runner = (f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} _run "
              f"{shlex.quote(str(state))} {job_id}")
    with open(log, "ab") as out:
        proc = subprocess.Popen(["/bin/sh", "-c", runner], stdout=out, stderr=subprocess.STDOUT, env=env,
                                cwd=opts.get("-D"), start_new_session=True)
    record["pid"] = proc.pid
    (state / f"{job_id}.json").write_text(json.dumps(record))
    print(job_id)
    return 0


def _run(state: Path, job_id: str) -> int:
    path = state / f"{job_id}.json"
    record = json.loads(path.read_text())
    code = subprocess.call(["/bin/sh", "-c", record["command"]])
    record = json.loads(path.read_text())
    if record["state"] == "RUNNING":
        record["state"] = "COMPLETED" if code == 0 else "FAILED"
    path.write_text(json.dumps(record))
    return code


def srun(args: list[str]) -> int:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 1
    return subprocess.call(args[i:])


def sacct(state: Path, args: list[str]) -> int:
    name = args[args.index("--name") + 1] if "--name" in args else None
    for path in sorted(state.glob("*.json")):
        record = json.loads(path.read_text())
        if name is None or record["name"] == name:
            print(f"{record['id']}|{record['state']}")
    return 0


def scancel(state: Path, args: list[str]) -> int:
    for job_id in args:
        path = state / f"{job_id}.json"
        if path.exists():
            record = json.loads(path.read_text())
            record["state"] = "CANCELLED"
            path.write_text(json.dumps(record))
            try:
                os.killpg(record["pid"], signal.SIGTERM)
            except (ProcessLookupError, KeyError):
                pass
    return 0


def main(argv: list[str]) -> int:
    command, state, args = argv[0], Path(argv[1]), argv[2:]
    if command == "sbatch":
        return sbatch(state, args)
    if command == "_run":
        return _run(state, args[0])
    if command == "srun":
        return srun(args)
    if command == "sacct":
        return sacct(state, args)
    if command == "squeue":  # same "<id>|<state>" lines as sacct
        return sacct(state, args)
    if command == "scancel":
        return scancel(state, args)
    if command == "scontrol":
        print("MinJobAge               = 300 sec")
        return 0
    if command in ("sacctmgr", "sshare"):
        print("fakeaccount")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
