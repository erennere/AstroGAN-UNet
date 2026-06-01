#!/bin/bash
#SBATCH --job-name=prepare-plots-single
#SBATCH --output=../logs/prepare_plots_single_%j.out
#SBATCH --error=../errs/prepare_plots_single_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=24:00:00

set -euo pipefail

source bash/utils.sh

init_aun_paths
activate_aun_env
enter_aun_src

INDEX=0
CONCURRENT_WORKERS=1
init_aun_runtime_selectors
DATA_ALIAS_SELECTOR="${AUN_DATA_ALIAS_HEX}"
MODEL_ALIAS_SELECTOR="${AUN_MODEL_ALIAS_HEX}"
EPOCH_SELECTOR="${AUN_EPOCH_SELECTOR}"

build_default_data_args

python -m src.visualization.prepare_plots \
    "${INDEX}" "${CONCURRENT_WORKERS}" \
    "${DATA_ALIAS_SELECTOR}" "${MODEL_ALIAS_SELECTOR}" "${EPOCH_SELECTOR}" \
    "${DATA_ARGS[@]}" \
    "$@"
