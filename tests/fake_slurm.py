"""A tiny stand-in for SLURM so the SLURM profile can be tested without a cluster.

`install(bin_dir, state_dir)` writes executables sbatch, srun, sacct, squeue, scancel, scontrol,
sacctmgr and sshare into bin_dir. sbatch runs the job in the background on this machine with
SLURM_JOB_ID and a per-job TMPDIR set (as SLURM does); #SBATCH lines in a submitted script are read;
sacct and squeue report job states.
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


ALIASES = {"--partition": "-p", "--time": "-t", "--account": "-A", "--chdir": "-D", "-o": "--output", "-J": "--job-name"}
VALUED = ("--job-name", "--output", "-D", "-t", "--mem", "-p", "-A", "--comment", "--gres", "--ntasks")


def _options(args: list[str]) -> tuple[dict, list[str]]:
    """sbatch options up to the script name; the script name and its arguments are returned as rest."""
    opts, i = {}, 0
    while i < len(args):
        arg = args[i]
        key, _, value = arg.partition("=") if arg.startswith("--") else (arg, "", "")
        key = ALIASES.get(key, key)
        if key == "--wrap":
            opts["wrap"] = value
        elif key == "--dependency":
            opts["dependency"] = value
        elif key in VALUED:
            if not value:
                i += 1
                value = args[i]
            opts[key] = value
        elif not arg.startswith("-"):
            return opts, args[i:]
        i += 1
    return opts, []


def sbatch(state: Path, args: list[str]) -> int:
    opts, rest = _options(args)
    if rest and Path(rest[0]).is_file():  # #SBATCH lines in the script; the command line wins
        lines = Path(rest[0]).read_text().splitlines()
        directives = [w for line in lines if line.startswith("#SBATCH") for w in shlex.split(line[7:])]
        opts = {**_options(directives)[0], **opts}
    job_id = _next_id(state)
    command = opts.get("wrap") or " ".join(shlex.quote(a) for a in ["bash", *rest])
    record = {"id": job_id, "name": opts.get("--job-name", rest[0] if rest else "wrap"), "state": "RUNNING",
              "gres": opts.get("--gres"), "partition": opts.get("-p"), "command": command,
              "dependency": opts.get("dependency")}
    if record["dependency"]:  # a stand-in: dependent jobs are recorded, not run
        record["state"] = "PENDING"
        (state / f"{job_id}.json").write_text(json.dumps(record))
        print(job_id)
        return 0
    log = opts.get("--output", f"slurm-{job_id}.out").replace("%j", str(job_id))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    tmp = state / "tmp" / str(job_id)
    tmp.mkdir(parents=True)
    env = dict(os.environ, SLURM_JOB_ID=str(job_id), TMPDIR=str(tmp), SLURM_CPUS_PER_TASK="2",
               SLURM_SUBMIT_DIR=os.getcwd())
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
    if any("Account" in a for a in args):  # account lookup
        print("fakeaccount")
        return 0
    name = args[args.index("--name") + 1] if "--name" in args else None
    for path in sorted(state.glob("*.json")):
        record = json.loads(path.read_text())
        if name is None or record["name"] == name:
            print(f"{record['id']}|{record['state']}")
    return 0


def scancel(state: Path, args: list[str]) -> int:
    """scancel <id>... cancels; scancel --signal=SIG --batch <id> sends SIG to the job's batch script."""
    import psutil

    sig = next((a.split("=", 1)[1] for a in args if a.startswith("--signal=")), None)
    for job_id in [a for a in args if not a.startswith("-")]:
        path = state / f"{job_id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if sig:  # the batch script is the "bash <script>" process below the job's runner
            for proc in psutil.Process(record["pid"]).children(recursive=True):
                if proc.cmdline()[:1] == ["bash"]:
                    proc.send_signal(getattr(signal, "SIG" + sig))
                    break
            continue
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
    if command == "squeue":
        if "-j" in args:  # squeue -h -j <id> -o %T: the state of one job, nothing once it is over
            path = state / f"{args[args.index('-j') + 1]}.json"
            if path.exists() and json.loads(path.read_text())["state"] in ("RUNNING", "PENDING"):
                print(json.loads(path.read_text())["state"])
            return 0
        return sacct(state, args)  # same "<id>|<state>" lines as sacct
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
