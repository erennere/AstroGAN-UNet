#!/bin/bash
################################################################################
# SLURM Job Submission Wrapper - Python + TensorFlow GPU Setup
#
# Submits the environment setup as a SLURM batch job on HPC cluster.
# Builds Python 3.11.13 and TensorFlow GPU stack in isolated venv.
#
# Usage (run from src/):
#   sbatch bash/setup_hpc_environment_slurm.sh
#   sbatch bash/setup_hpc_environment_slurm.sh --prefix /custom/path
#
# Output:
#   SLURM stdout: ../logs/setup_astro_hpc_${SLURM_JOB_ID}.out
#   SLURM stderr: ../errs/setup_astro_hpc_${SLURM_JOB_ID}.err
#   Build artifacts: ~/.local/astro/
#   Venv: ../.venv/
#
################################################################################

#SBATCH --job-name=setup-astro-hpc
#SBATCH --output=../logs/setup_astro_hpc_%j.out
#SBATCH --error=../errs/setup_astro_hpc_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

set -e

echo "========================================================================"
echo "SLURM Setup Job Started"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Partition: $SLURM_JOB_PARTITION"
echo "========================================================================"

# Assumes submission from src/ directory.
SRC_DIR="$PWD"
PROJECT_ROOT="$(cd .. && pwd)"

cd "$SRC_DIR" || {
  echo "Failed to change to src directory: $SRC_DIR"
  exit 1
}

# Verify setup script exists
if [[ ! -f "bash/setup_hpc_environment.sh" ]]; then
  echo "ERROR: bash/setup_hpc_environment.sh not found in $SRC_DIR"
  echo "Make sure you run: sbatch bash/setup_hpc_environment_slurm.sh from src"
  exit 1
fi

# Pass through any command-line arguments to the actual setup script
echo ""
echo "Running setup_hpc_environment.sh..."
echo ""

bash bash/setup_hpc_environment.sh "$@"

SETUP_EXIT=$?

echo ""
echo "========================================================================"
if [[ $SETUP_EXIT -eq 0 ]]; then
  echo "✓ Setup completed successfully"
else
  echo "✗ Setup failed with exit code $SETUP_EXIT"
fi
echo "========================================================================"
echo "Output: ../logs/setup_astro_hpc_${SLURM_JOB_ID}.out"
echo "Errors: ../errs/setup_astro_hpc_${SLURM_JOB_ID}.err"
echo ""

exit $SETUP_EXIT
