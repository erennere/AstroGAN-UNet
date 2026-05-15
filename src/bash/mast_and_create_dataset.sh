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

# Project root. Default assumes current working directory is src/.
: "${AUN_PROJECT_ROOT:=$(cd .. && pwd)}"

# Conda environment path (created by setup_hpc_environment.sh)
: "${AUN_ENV_PATH:=${AUN_PROJECT_ROOT}/.venv}"
: "${AUN_CONDA_BASE:=${HOME}/.local/miniconda3}"

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

# Logging destination (can be overridden). Default: <project_root>/logs
: "${AUN_LOG_DIR:=${AUN_PROJECT_ROOT}/logs}"
mkdir -p "${AUN_LOG_DIR}"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="${AUN_LOG_DIR}/mast_and_create_dataset_${LOG_JOB_ID}_${LOG_TS}.log"

# Mirror stdout/stderr to both console and log file.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Project root : ${AUN_PROJECT_ROOT}"
echo "Conda env    : ${AUN_ENV_PATH}"
echo "Log file     : ${LOG_FILE}"

# Activate conda environment from a prefix path
if [[ -f "${AUN_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${AUN_CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${AUN_ENV_PATH}"
    echo "Activated conda environment: ${CONDA_PREFIX}"
else
    echo "ERROR: Conda base not found at ${AUN_CONDA_BASE}"
    echo "Set AUN_CONDA_BASE or run: bash bash/setup_hpc_environment.sh"
    exit 1
fi

if [[ "${CONDA_PREFIX:-}" != "${AUN_ENV_PATH}" ]]; then
    echo "ERROR: Expected active env ${AUN_ENV_PATH}, got ${CONDA_PREFIX:-<none>}"
    echo "Run: bash bash/setup_hpc_environment.sh"
    exit 1
fi

# Work from the src/ subdirectory.
cd "${AUN_PROJECT_ROOT}/src"
# Keep project root on PYTHONPATH so absolute imports like `from src...` work.
export PYTHONPATH="${AUN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
