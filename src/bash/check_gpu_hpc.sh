#!/bin/bash
#SBATCH --job-name=check-gpu
#SBATCH --output=../logs/check_gpu_%j.out
#SBATCH --error=../errs/check_gpu_%j.err
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=2gb
#SBATCH --time=00:30:00

set -euo pipefail

# Default assumes submission from src/.
: "${AUN_PROJECT_ROOT:=$(cd .. && pwd)}"
: "${AUN_ENV_PATH:=${AUN_PROJECT_ROOT}/.venv}"
: "${AUN_CONDA_BASE:=${HOME}/.local/miniconda3}"

echo "Project root : ${AUN_PROJECT_ROOT}"
echo "Conda env    : ${AUN_ENV_PATH}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Job ID       : ${SLURM_JOB_ID:-local}"

# Activate conda environment from a prefix path.
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
    exit 1
fi

echo ""
echo "=== System-level GPU check (nvidia-smi) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
    nvidia-smi || true
else
    echo "ERROR: nvidia-smi is not available on PATH."
    exit 1
fi

echo ""
echo "=== Python-level GPU check ==="
cd "${AUN_PROJECT_ROOT}/src"
export PYTHONPATH="${AUN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import importlib
import json
import sys

result = {
    "tensorflow_installed": False,
    "tensorflow_visible_gpus": [],
    "torch_installed": False,
    "torch_cuda_available": False,
    "torch_device_count": 0,
}

# TensorFlow check
try:
    tf = importlib.import_module("tensorflow")
    result["tensorflow_installed"] = True
    gpus = tf.config.list_physical_devices("GPU")
    result["tensorflow_visible_gpus"] = [getattr(g, "name", str(g)) for g in gpus]
except Exception as exc:
    result["tensorflow_error"] = str(exc)

# PyTorch check
try:
    torch = importlib.import_module("torch")
    result["torch_installed"] = True
    result["torch_cuda_available"] = bool(torch.cuda.is_available())
    result["torch_device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    result["torch_error"] = str(exc)

print(json.dumps(result, indent=2))

tf_ok = len(result["tensorflow_visible_gpus"]) > 0
torch_ok = result["torch_cuda_available"] and result["torch_device_count"] > 0

if not (tf_ok or torch_ok):
    print("ERROR: No GPU detected by TensorFlow or PyTorch.")
    sys.exit(2)

print("SUCCESS: GPU is recognized by at least one ML framework.")
PY

echo ""
echo "GPU check completed successfully."
