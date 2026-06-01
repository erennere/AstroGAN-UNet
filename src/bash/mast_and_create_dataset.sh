#!/bin/bash
#SBATCH --job-name=mast-create-dataset
#SBATCH --output=../logs/mast_and_create_dataset_%j.out
#SBATCH --error=../errs/mast_and_create_dataset_%j.err
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --array=1-3

# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
#   All variables below can be overridden by exporting them before calling this
#   script, e.g.:
#       export AUN_PROJECT_ROOT=/custom/path ; bash bash/mast_and_create_dataset.sh
#   Script assumes it is launched from the src/ directory.
# ═════════════════════════════════════════════════════════════════════════════

set -euo pipefail

source bash/utils.sh

init_aun_paths

# Pipeline stage toggles (set to false to skip a stage).
: "${AUN_RUN_MAST:=true}"
: "${AUN_RUN_CREATE_DATASET:=true}"

# ─────────────────────────────────────────────────────────────────────────────
# Job parameter table
#   Index 0 is unused (arrays are 1-based to match SLURM_ARRAY_TASK_ID).
#   To add a new job: append one value to each array and increment --array.
#   Note: sigma=3 matches the config.yaml default for all jobs; no override needed.
# ─────────────────────────────────────────────────────────────────────────────
#                            job:  0(dummy)  1      2      3
NSIGMA=(                          ""        2      2      2      )
FP_RADIUS=(                       ""        10     10     10     ) 
NPIXELS=(                         ""        8      8      8      )
FILTER_SURVEYS=(                  ""        true   true   true   )
FILTER_BY_LAST_NAME=(             ""        false  true   true   )
LAST_NAME_FILTER=(                ""        ""     ""     MOMCHEVA )

# ─────────────────────────────────────────────────────────────────────────────
# Explicit runtime values (no auto-detection)
# ─────────────────────────────────────────────────────────────────────────────

setup_aun_logging "mast_and_create_dataset"

echo "Project root : ${AUN_PROJECT_ROOT}"
echo "Conda env    : ${AUN_ENV_PATH}"
echo "Log file     : ${AUN_LOG_FILE}"

activate_aun_env
echo "Activated conda environment: ${CONDA_PREFIX}"
require_aun_env_active

# Work from the src/ subdirectory.
enter_aun_src

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build CLI args for job index $1 and run create_dataset
# ─────────────────────────────────────────────────────────────────────────────
run_job() {
    local idx=$1
    local args=(
        --nsigma              "${NSIGMA[$idx]}"
        --footprint-radius    "${FP_RADIUS[$idx]}"
        --npixels             "${NPIXELS[$idx]}"
        --filter-surveys      "${FILTER_SURVEYS[$idx]}"
        --filter-by-last-name "${FILTER_BY_LAST_NAME[$idx]}"
    )
    if [[ -n "${LAST_NAME_FILTER[$idx]}" ]]; then
        args+=(--last-name-filter-value "${LAST_NAME_FILTER[$idx]}")
    fi

    if [[ "${AUN_RUN_MAST}" == "true" ]]; then
        echo "[Job ${idx}] python -m src.data.mast ${args[*]}"
        python -m src.data.mast "${args[@]}"
    fi

    if [[ "${AUN_RUN_CREATE_DATASET}" == "true" ]]; then
        echo "[Job ${idx}] python -m src.data.create_dataset ${args[*]}"
        python -m src.data.create_dataset "${args[@]}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch: use SLURM array task ID when running on HPC, otherwise run all
# jobs sequentially (or a single job if a job index is passed as $1).
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    run_job "${SLURM_ARRAY_TASK_ID}"
elif [[ -n "$1" ]]; then
    run_job "$1"
else
    for idx in "${!NSIGMA[@]}"; do
        [[ $idx -eq 0 ]] && continue
        run_job "$idx"
    done
fi
