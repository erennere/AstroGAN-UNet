#!/bin/bash

AUN_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


init_aun_paths() {
    : "${AUN_PROJECT_ROOT:=$(cd "${AUN_UTILS_DIR}/../.." && pwd)}"
    : "${AUN_ENV_PATH:=${AUN_PROJECT_ROOT}/.venv}"
    : "${AUN_CONDA_BASE:=${HOME}/.local/miniconda3}"
    : "${AUN_LOG_DIR:=${AUN_PROJECT_ROOT}/logs}"
}


setup_aun_logging() {
    local log_name="$1"

    init_aun_paths
    mkdir -p "${AUN_LOG_DIR}"

    local log_ts
    local log_job_id
    local log_file
    log_ts="$(date +%Y%m%d_%H%M%S)"
    log_job_id="${SLURM_JOB_ID:-local}"
    log_file="${AUN_LOG_DIR}/${log_name}_${log_job_id}_${log_ts}.log"
    AUN_LOG_FILE="${log_file}"
    exec > >(tee -a "${log_file}") 2>&1
}


activate_aun_env() {
    init_aun_paths

    if [[ -f "${AUN_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${AUN_CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${AUN_ENV_PATH}"
    else
        echo "ERROR: Conda base not found at ${AUN_CONDA_BASE}"
        exit 1
    fi
}


require_aun_env_active() {
    init_aun_paths

    if [[ "${CONDA_PREFIX:-}" != "${AUN_ENV_PATH}" ]]; then
        echo "ERROR: Expected active env ${AUN_ENV_PATH}, got ${CONDA_PREFIX:-<none>}"
        echo "Run: bash bash/setup_hpc_environment.sh"
        exit 1
    fi
}


enter_aun_src() {
    init_aun_paths
    cd "${AUN_PROJECT_ROOT}/src"
    export PYTHONPATH="${AUN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}


configure_aun_threads() {
    local requested_threads="$1"
    local effective_threads="${requested_threads}"

    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "${requested_threads}" ]]; then
        echo "WARNING: N=${requested_threads} but SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}. Using SLURM_CPUS_PER_TASK."
        effective_threads="${SLURM_CPUS_PER_TASK}"
    fi

    export OMP_NUM_THREADS="${effective_threads}"
    export MKL_NUM_THREADS="${effective_threads}"
    export OPENBLAS_NUM_THREADS="${effective_threads}"
    AUN_EFFECTIVE_THREADS="${effective_threads}"
}


build_data_args() {
    local filter_by_last_name="$1"
    local last_name_filter_value="${2:-}"

    DATA_ARGS=(
        --nsigma "2"
        --footprint-radius "10"
        --npixels "8"
        --filter-surveys "true"
        --filter-by-last-name "${filter_by_last_name}"
    )

    if [[ "${filter_by_last_name}" == "true" ]]; then
        DATA_ARGS+=(--last-name-filter-value "${last_name_filter_value}")
    fi
}


build_default_data_args() {
    build_data_args "true" "MOMCHEVA"
}


build_uncropped_data_args() {
    build_data_args "true" "FABER"
}


build_model_scenario_args() {
    local model="$1"
    local attention="$2"
    local scaling="$3"
    local loss="$4"
    local output_activation="null"

    if [[ "${loss}" == "ssim_loss" ]]; then
        output_activation="sigmoid"
    fi

    MODEL_ARGS=(
        --model-type "${model}"
        --attention "${attention}"
        --scaling "${scaling}"
        --loss-name "${loss}"
        --dropout-rate "0.2"
        --output-activation "${output_activation}"
        --kernel-initializer "he_normal"
        --activation-name "LeakyReLU"
        --discriminator-activation "LeakyReLU"
        --discriminator-output-activation "sigmoid"
    )
}


init_aun_runtime_selectors() {
    : "${AUN_DATA_ALIAS_HEX:=eNpzYWRl-M_AxMDFwAEAClcBXg}"
    : "${AUN_MODEL_ALIAS_HEX:=null}"
    : "${AUN_EPOCH_SELECTOR:=null}"
}