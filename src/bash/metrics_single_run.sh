#!/bin/bash
#SBATCH --job-name=metrics-single
#SBATCH --output=../logs/metrics_single_%j.out
#SBATCH --error=../errs/metrics_single_%j.err
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00

set -euo pipefail

source bash/utils.sh

init_aun_paths
activate_aun_env
enter_aun_src

build_default_data_args
init_aun_runtime_selectors

# Single-run selectors
INDEX=0
CONCURRENT_WORKERS=1
DATA_ALIAS_SELECTOR="${AUN_DATA_ALIAS_HEX}"
MODEL_ALIAS_SELECTOR="${AUN_MODEL_ALIAS_HEX}"
EPOCH_SELECTOR="${AUN_EPOCH_SELECTOR}"

python -m src.evaluation.metrics \
    "${INDEX}" "${CONCURRENT_WORKERS}" \
    "${DATA_ALIAS_SELECTOR}" "${MODEL_ALIAS_SELECTOR}" "${EPOCH_SELECTOR}" \
    "${DATA_ARGS[@]}" \
    "$@"
