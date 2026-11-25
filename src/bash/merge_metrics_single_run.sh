#!/bin/bash
#SBATCH --job-name=merge-metrics
#SBATCH --output=../logs/merge_metrics_%j.out
#SBATCH --error=../errs/merge_metrics_%j.err
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=01:00:00

set -euo pipefail

source bash/utils.sh

init_aun_paths
activate_aun_env
enter_aun_src

build_default_data_args
#AUN_DATA_ALIAS_HEX="eNpzYWRl-M_AxMDFwAEAClcBXg"
init_aun_runtime_selectors

DATA_ALIAS_SELECTOR="${AUN_DATA_ALIAS_HEX}"
MODEL_ALIAS_SELECTOR="${AUN_MODEL_ALIAS_HEX}"
EPOCH_SELECTOR="${AUN_EPOCH_SELECTOR}"

python -m src.evaluation.merge_metrics \
    "${DATA_ALIAS_SELECTOR}" "${MODEL_ALIAS_SELECTOR}" "${EPOCH_SELECTOR}" \
    "${DATA_ARGS[@]}" \
    "$@"
