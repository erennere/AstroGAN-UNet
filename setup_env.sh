#!/usr/bin/env bash
# Create .venv in this folder and install astrounets into it (Linux, WSL, bwForCluster Helix).
#   bash setup_env.sh            GPU build of TensorFlow (also runs on CPU)
#   CPU_ONLY=1 bash setup_env.sh smaller CPU-only TensorFlow
# On Helix, load a Python 3.11 or 3.12 module first (see `module avail devel/python`).
set -eo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), f"need Python 3.11-3.13, found {sys.version}"'
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "${CPU_ONLY:-0}" = "1" ]; then extra="cpu"; else extra="gpu"; fi
.venv/bin/python -m pip install -e ".[${extra},dev]"
.venv/bin/astrounets --help > /dev/null
echo "Done. Activate with: source .venv/bin/activate"
