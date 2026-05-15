# Helix HPC GPU Environment Setup (TensorFlow + Miniconda)

## Overview

**Simple, reproducible setup using Miniconda:** Zero compilation, pre-built binaries, all extensions included.

CUDA + cuDNN are auto-bundled by `tensorflow==2.20.0` (conda-forge).  
NVIDIA drivers are pre-installed on Helix compute nodes (sysadmin responsibility).

**Result:** Guaranteed compatibility, ~3GB disk, ~2-3 min setup.

---

## Stack

| Component | Version | Source |
|-----------|---------|--------|
| **Python** | 3.11 | Pre-compiled (Miniconda) |
| **Miniconda** | 24-11 | conda-forge (no compilation) |
| **TensorFlow** | 2.20.0 | conda-forge |
| **CUDA** | 12.4 (bundled) | TensorFlow[and-cuda] |
| **cuDNN** | 9.4 (bundled) | TensorFlow[and-cuda] |

---

## Quick Start

### Prerequisites

Minimal requirements (likely already available):

```bash
# Verify you have wget and tar (almost always available)
command -v wget tar bash
```

That's it! No gcc, make, or system package manager needed.

---

## Setup (One-Time)

**From the project root:**

```bash
cd /mnt/sds-hd/sd17f001/eren/network
bash src/bash/setup_hpc_environment.sh
```

**What it does:**
1. Downloads Miniconda3 installer (~150MB)
2. Installs to `~/.local/miniconda3`
3. Creates conda environment `.venv/` from `environment.yml`
4. Verifies all dependencies (numpy, astropy, tensorflow, photutils, etc.)

**Expected output:**
```
[INFO] Installing Miniconda to: /home/hd/hd_hd/hd_nk194/.local/miniconda3
[OK] Miniconda installed
[INFO] Creating conda environment from: .../environment.yml
[OK] Conda environment created: /path/to/.venv
[OK] Installation verified
[OK] Setup succeeded
```

**Time:** ~2-3 minutes (depends on internet speed)

---

## Usage

### Local Shell

```bash
# Activate environment
source ~/.local/miniconda3/etc/profile.d/conda.sh
conda activate .venv

# Verify
which python
python --version
python -c "import tensorflow; print(tensorflow.__version__)"

# Deactivate when done
conda deactivate
```

### SLURM Jobs

**Both data pipeline and training scripts automatically activate the conda environment:**

```bash
# Submit data download/preprocessing (CPU)
sbatch src/bash/mast_and_create_dataset.sh

# Submit 64-scenario training (GPU, array job)
sbatch src/bash/train_model_scenarios.sh

# Monitor
squeue -u $USER
```

### Verify GPU Access

```bash
# Interactive allocation
salloc -p gpu4 -N 1 --gres=gpu:1 -t 5:00

# Then on compute node
source ~/.local/miniconda3/etc/profile.d/conda.sh
conda activate .venv
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs found: {len(gpus)}')
for gpu in gpus:
    print(f'  {gpu}')
"
```

---

## Environment Details

**Location:** `.venv/` (project root)

**Includes:**
- Python 3.11 (all standard modules)
- numpy, scipy, pandas, scikit-learn, scikit-image
- astropy, photutils (astronomy)
- tensorflow 2.20 + CUDA 12.4 + cuDNN 9.4
- matplotlib, plotly (visualization)
- jupyter, ipython (interactive)

**Declared in:** `environment.yml`

**Reproducible:** Run same setup on any HPC cluster with wget/bash

---

## Troubleshooting

---

## Troubleshooting

### "Miniconda not found" error

The setup script automatically downloads Miniconda. If you see errors:

```bash
# Check if miniconda was downloaded
ls -lh ~/.local/miniconda3/

# If missing, verify internet access and retry
bash src/bash/setup_hpc_environment.sh
```

### "conda: command not found" in SLURM job

This usually means the environment path is wrong. Verify:

```bash
# Check environment location
ls -la .venv/bin/python

# If missing, setup wasn't completed
bash src/bash/setup_hpc_environment.sh
```

### "ModuleNotFoundError: No module named '_bz2'" (now fixed!)

This **shouldn't happen** with Miniconda (all modules pre-built). If it does:

```bash
# Verify Python has bz2
python -c "import bz2; print('✓ bz2 works')"

# If fails, environment is corrupted. Recreate:
rm -rf .venv
bash src/bash/setup_hpc_environment.sh
```

### GPU not detected on compute node

Expected on login nodes. Test interactively:

```bash
salloc -p gpu4 -N 1 --gres=gpu:1 -t 5:00
source ~/.local/miniconda3/etc/profile.d/conda.sh
conda activate .venv

python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Found {len(gpus)} GPU(s)')
else:
    print('No GPUs detected')
"

exit
```

If no GPUs found on compute node:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA availability: `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"`
3. Contact HPC support

### Environment takes too long to create

Miniconda usually finishes in 2-3 minutes. If slower:

- Check internet speed: `wget -O /dev/null https://repo.anaconda.com/miniconda/Miniconda3-24-11.6-Linux-x86_64.sh`
- Consider running setup during off-peak hours or via SLURM job
- If it times out, it's safe to re-run (idempotent)

### Disk space issues

If you run out of space during setup:

```bash
# Check available space
df -h ~/.local/

# Clean Miniconda package cache
conda clean --all --yes

# Or manually delete
rm -rf ~/.local/miniconda3/  # Frees ~500MB
```

Then re-run setup.

---

## Files Reference

- **[setup_hpc_environment.sh](src/bash/setup_hpc_environment.sh)** - Main setup (Miniconda)
- **[environment.yml](environment.yml)** - Conda environment specification
- **[mast_and_create_dataset.sh](src/bash/mast_and_create_dataset.sh)** - Data pipeline (auto-activates)
- **[train_model_scenarios.sh](src/bash/train_model_scenarios.sh)** - Training (auto-activates, 64 scenarios)
- **[MINICONDA_MIGRATION.md](MINICONDA_MIGRATION.md)** - What changed and why

---

## Summary

| Question | Answer |
|----------|--------|
| **How long?** | ~2-3 minutes |
| **How much disk?** | ~3GB |
| **Do I need to compile?** | No |
| **Do I need sudo?** | No |
| **Will it work on any HPC?** | Yes (just need wget/bash) |
| **Can I reproduce it?** | Yes (environment.yml is declarative) |
| **What if something fails?** | Re-run setup; it's safe and idempotent |




