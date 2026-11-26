#!/bin/bash
# One-time setup on bwForCluster Helix, as a job (no modules; the job downloads Python packages):
#   sbatch workflow/helix/setup.sbatch          (from the repository folder)
# Creates .venv with TensorFlow for the A40s, checks that TensorFlow sees the GPU and runs the tests.
# Log: setup-<job id>.log in the repository folder.
#SBATCH --job-name=astrounets-setup
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=setup-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f pyproject.toml ] || { echo "submit this from the repository folder"; exit 1; }
bash setup_env.sh
.venv/bin/python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); assert gpus, 'TensorFlow sees no GPU'"
.venv/bin/python -m pytest
echo "setup done: submit the study with  sbatch workflow/helix/run.sbatch"
