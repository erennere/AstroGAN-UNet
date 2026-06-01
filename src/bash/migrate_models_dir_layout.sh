#!/bin/bash
#SBATCH --job-name=migrate-models-layout
#SBATCH --output=../logs/migrate_models_dir_layout_%j.out
#SBATCH --error=../errs/migrate_models_dir_layout_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=04:00:00

set -euo pipefail

source bash/utils.sh

init_aun_paths
setup_aun_logging "migrate_models_dir_layout"
activate_aun_env
echo "Activated conda environment: ${CONDA_PREFIX}"
require_aun_env_active
enter_aun_src

MODELS_ROOT="${AUN_MIGRATE_MODELS_ROOT:-${1:-}}"
EXECUTE_MODE="${AUN_MIGRATE_EXECUTE:-false}"

if [[ -z "${MODELS_ROOT}" ]]; then
    echo "ERROR: Provide models root via AUN_MIGRATE_MODELS_ROOT or as the first positional argument."
    echo "Example dry-run: sbatch bash/migrate_models_dir_layout.sh /gpfs/lsdf02/sd17f001/eren/network/models"
    echo "Example execute: AUN_MIGRATE_EXECUTE=true sbatch bash/migrate_models_dir_layout.sh /gpfs/lsdf02/sd17f001/eren/network/models"
    exit 2
fi

CMD=(python ../scripts/migrate_models_dir_layout.py --models-root "${MODELS_ROOT}")

if [[ "${EXECUTE_MODE}" == "true" ]]; then
    CMD+=(--execute)
fi

echo "Project root : ${AUN_PROJECT_ROOT}"
echo "Conda env    : ${AUN_ENV_PATH}"
echo "Log file     : ${AUN_LOG_FILE}"
echo "Models root  : ${MODELS_ROOT}"
echo "Execute mode : ${EXECUTE_MODE}"
echo "Command      : ${CMD[*]}"

"${CMD[@]}"