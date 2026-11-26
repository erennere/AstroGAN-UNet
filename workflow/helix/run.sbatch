#!/bin/bash
# The whole study on bwForCluster Helix, as one job (from the repository folder):
#   sbatch workflow/helix/run.sbatch                       uses configs/default.yaml
#   sbatch workflow/helix/run.sbatch configs/pilot.yaml    any other config
# This small CPU job runs Snakemake, which downloads the data and submits every training run as its
# own GPU job and every dataset build as a CPU job (workflow/profiles/slurm/config.yaml). 15 minutes
# before the 120 h limit it stops cleanly and submits itself again; the new job continues where this
# one stopped. Progress: .venv/bin/astrounets status    Stop: .venv/bin/astrounets stop
# Log: workflow-<job id>.log here; one log per SLURM job in runs/_workflow/.snakemake/slurm_logs.
#SBATCH --job-name=astrounets
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=120:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=workflow-%j.log
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
[ -x .venv/bin/astrounets ] || { echo "no .venv: run  sbatch workflow/helix/setup.sbatch  first"; exit 1; }
config="${1:-configs/default.yaml}"
resubmit=0
interrupt() { kill -INT -- "-$pid" 2>/dev/null; }      # Snakemake cancels its jobs and exits
trap 'resubmit=1; interrupt' USR1                       # time limit is near
trap 'interrupt' USR2                                   # astrounets stop
setsid .venv/bin/astrounets workflow -c "$config" --foreground &
pid=$!
while true; do
    wait "$pid"; code=$?
    kill -0 "$pid" 2>/dev/null || break                 # a trap interrupted wait: keep waiting
done
if [ "$resubmit" = 1 ]; then
    sbatch --dependency=afterany:"$SLURM_JOB_ID" "$SLURM_SUBMIT_DIR/workflow/helix/run.sbatch" "$config"
fi
exit "$code"
