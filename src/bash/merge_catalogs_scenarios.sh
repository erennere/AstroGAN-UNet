#!/bin/bash
#SBATCH --job-name=merge-scenarios
#SBATCH --output=../logs/merge_scenarios_%j.out
#SBATCH --error=../errs/merge_scenarios_%j.err
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=64
#SBATCH --mem=234gb
#SBATCH --time=48:00:00
#SBATCH --array=1-10%10

set -euo pipefail

N=64
EVAL_CONCURRENT_WORKERS=10

source bash/utils.sh

init_aun_paths
setup_aun_logging "merge_scenarios"
activate_aun_env
enter_aun_src
configure_aun_threads "${N}"
N="${AUN_EFFECTIVE_THREADS}"

build_uncropped_data_args
init_aun_runtime_selectors

EVAL_INDEX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
DATA_ALIAS_SELECTOR="${AUN_DATA_ALIAS_HEX}"
MODEL_ALIAS_SELECTOR="${AUN_MODEL_ALIAS_HEX}"
EPOCH_SELECTOR="${AUN_EPOCH_SELECTOR}"

MODEL_OPTIONS=("unet" "gan")
ATTENTION_OPTIONS=("false" "true")
SCALING_OPTIONS=("z_scale" "min_max" "log_min_max" "null")
LOSS_OPTIONS=("log_cosh_loss" "scale_invariant_mae" "MeanAbsoluteError" "ssim_loss")

build_common_args() {
    local model="$1"
    local attn="$2"
    local scaling="$3"
    local loss="$4"

    build_model_scenario_args "${model}" "${attn}" "${scaling}" "${loss}"
    COMMON_ARGS=("${DATA_ARGS[@]}" "${MODEL_ARGS[@]}")
}

run_one() {
    local model="$1"
    local attn="$2"
    local scaling="$3"
    local loss="$4"

    build_common_args "${model}" "${attn}" "${scaling}" "${loss}"

    echo "[merge] idx=${EVAL_INDEX} cw=${EVAL_CONCURRENT_WORKERS} model=${model} attn=${attn} scaling=${scaling} loss=${loss} model_alias=${MODEL_ALIAS_SELECTOR}"
    python -m src.evaluation.merge_catalogs \
        "${EVAL_INDEX}" "${EVAL_CONCURRENT_WORKERS}" \
        "${DATA_ALIAS_SELECTOR}" "${MODEL_ALIAS_SELECTOR}" "${EPOCH_SELECTOR}" \
        "${COMMON_ARGS[@]}"
}

for model in "${MODEL_OPTIONS[@]}"; do
    for attn in "${ATTENTION_OPTIONS[@]}"; do
        for scaling in "${SCALING_OPTIONS[@]}"; do
            for loss in "${LOSS_OPTIONS[@]}"; do
                run_one "${model}" "${attn}" "${scaling}" "${loss}"
            done
        done
    done
done
