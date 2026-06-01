#!/bin/bash
#SBATCH --job-name=check-gpu
#SBATCH --output=../logs/check_gpu_%j.out
#SBATCH --error=../errs/check_gpu_%j.err
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=2gb
#SBATCH --time=00:30:00

set -euo pipefail

source bash/utils.sh

init_aun_paths

echo "Project root : ${AUN_PROJECT_ROOT}"
echo "Conda env    : ${AUN_ENV_PATH}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Job ID       : ${SLURM_JOB_ID:-local}"

activate_aun_env
echo "Activated conda environment: ${CONDA_PREFIX}"
require_aun_env_active

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
enter_aun_src

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
