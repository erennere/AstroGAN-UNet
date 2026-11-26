#!/usr/bin/env bash
# Create .venv in this folder (or update it) and install astrounets into it (Linux, Helix).
# No modules needed. On Helix submit it as a job:  sbatch workflow/helix/setup.sbatch
#   bash setup_env.sh                GPU build of TensorFlow (also runs on CPU)
#   CPU_ONLY=1 bash setup_env.sh     smaller CPU-only TensorFlow
# Python 3.11-3.13 comes from an existing .venv, else $PYTHON, else python3.13/3.12/3.11/python3 on
# the PATH, else uv downloads one into .tools/ (also forced with ASTROUNETS_USE_UV=1). Needs internet.
set -eo pipefail
cd "$(dirname "$0")"
UV_VERSION=0.13.10

fits() {  # is $1 a Python 3.11-3.13?
    "$1" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null
}

uv_bin() {  # a uv executable in .tools/uv, installed on first use
    if [ ! -x .tools/uv/bin/uv ]; then
        local base
        base="$(command -v python3 || true)"
        if [ -n "$base" ] && "$base" -m pip --version >/dev/null 2>&1; then
            "$base" -m pip install --quiet --target .tools/uv "uv==${UV_VERSION}" >&2
        else
            curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
                | env UV_INSTALL_DIR="$PWD/.tools/uv/bin" UV_NO_MODIFY_PATH=1 sh >&2
        fi
    fi
    echo "$PWD/.tools/uv/bin/uv"
}

if [ -x .venv/bin/python ] && fits .venv/bin/python; then
    echo "reusing .venv ($(.venv/bin/python -V))"
else
    rm -rf .venv
    python=""
    if [ "${ASTROUNETS_USE_UV:-0}" != "1" ]; then
        for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3; do
            if [ -n "$candidate" ] && command -v "$candidate" >/dev/null && fits "$(command -v "$candidate")"; then
                python="$(command -v "$candidate")"
                break
            fi
        done
    fi
    if [ -n "$python" ]; then
        echo "creating .venv with $python ($("$python" -V))"
        "$python" -m venv .venv
    else
        echo "uv ${UV_VERSION} installs Python 3.11 into .tools/python (none on the PATH, or ASTROUNETS_USE_UV=1)"
        UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" "$(uv_bin)" venv --managed-python --python 3.11 --seed .venv
    fi
fi
.venv/bin/python -m pip install --quiet --upgrade pip
if [ "${CPU_ONLY:-0}" = "1" ]; then extra="cpu"; else extra="gpu"; fi
.venv/bin/python -m pip install -e ".[${extra},dev]"
.venv/bin/astrounets --help > /dev/null
echo "Done: $(.venv/bin/python -V) in .venv"
