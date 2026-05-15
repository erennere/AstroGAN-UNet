#!/bin/bash
################################################################################
# HPC Environment Bootstrap Script - Miniconda + TensorFlow GPU Stack
# 
# Purpose: Setup isolated conda environment with TensorFlow GPU support.
# Everything pre-compiled with ALL extensions included (no libffi/bz2 issues).
#
# Requirements:
#   - NVIDIA drivers already installed on compute nodes
#   - Internet access to download miniconda + packages
#   - ~8GB free disk space
#   - Standard Unix tools: wget, tar, bash (NO gcc/make needed!)
#
# Usage (run from src/):
#   bash bash/setup_hpc_environment.sh [--prefix PATH] [--clean-start]
#
# Examples (from src/):
#   bash bash/setup_hpc_environment.sh                    # Defaults: ~/.local/miniconda3
#   bash bash/setup_hpc_environment.sh --prefix /opt/conda-astro
#   bash bash/setup_hpc_environment.sh --clean-start      # Remove old miniconda + .venv, then reinstall
#
################################################################################

set -e  # Exit on any error
set -o pipefail  # Preserve failures in piped commands (e.g., command | tee)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Assumes this script is run from src/ directory.
PROJECT_ROOT="$(cd .. && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

# Miniconda paths
CONDA_INSTALL_PREFIX="${HOME}/.local/miniconda3"
ENV_NAME="astro-unets"
ENV_PATH="${PROJECT_ROOT}/.venv"  # Use .venv for compatibility with existing scripts
CLEAN_START=0

# Miniconda download
MINICONDA_VERSION="latest"  # or use specific version like "24.11.2-0"
MINICONDA_OS="Linux"
MINICONDA_ARCH="x86_64"
MINICONDA_INSTALLER="Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-${MINICONDA_ARCH}.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"

# Logging
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/setup_hpc_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --prefix)
      CONDA_INSTALL_PREFIX="$2"
      shift 2
      ;;
    --clean-start)
      CLEAN_START=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ============================================================================
# LOGGING
# ============================================================================

log() {
  local level="$1"
  shift
  local msg="$*"
  local ts=$(date '+%Y-%m-%d %H:%M:%S')
  local line="[$ts] [$level] $msg"
  echo "$line" >> "$LOG_FILE"
  echo "$line" >&2
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_ok() { log "OK" "$@"; }

# ============================================================================
# CLEAN START
# ============================================================================

clean_start() {
  log_warn "Clean start requested. Removing existing environment artifacts..."

  if [[ -d "$ENV_PATH" ]]; then
    log_info "Removing environment: $ENV_PATH"
    rm -rf "$ENV_PATH"
  else
    log_info "Environment not present: $ENV_PATH"
  fi

  if [[ -d "$CONDA_INSTALL_PREFIX" ]]; then
    log_info "Removing miniconda prefix: $CONDA_INSTALL_PREFIX"
    rm -rf "$CONDA_INSTALL_PREFIX"
  else
    log_info "Miniconda prefix not present: $CONDA_INSTALL_PREFIX"
  fi

  log_ok "Clean start removal complete"
}

# ============================================================================
# DOWNLOAD MINICONDA
# ============================================================================

download_miniconda() {
  # Keep installer cache outside the install prefix so clean installs do not
  # pre-create the target directory and confuse installer mode selection.
  local download_dir="${HOME}/.cache/astrounets/miniconda"
  mkdir -p "$download_dir"
  
  local installer_path="$download_dir/$MINICONDA_INSTALLER"
  
  if [[ -f "$installer_path" ]]; then
    # Validate existing file
    local file_size=$(stat -c%s "$installer_path" 2>/dev/null || echo "0")
    if [[ $file_size -gt 50000000 ]]; then  # Should be ~150MB
      log_info "Miniconda already downloaded: $installer_path"
      echo "$installer_path"
      return 0
    else
      log_warn "Existing file too small ($file_size bytes), re-downloading..."
      rm -f "$installer_path"
    fi
  fi
  
  log_info "Downloading Miniconda from: $MINICONDA_URL"
  
  local wget_output
  if ! wget_output=$(wget -O "$installer_path" "$MINICONDA_URL" 2>&1); then
    log_error "Failed to download miniconda: $wget_output"
    rm -f "$installer_path"
    return 1
  fi
  
  # Validate downloaded file
  local file_size=$(stat -c%s "$installer_path" 2>/dev/null || echo "0")
  if [[ ! -f "$installer_path" ]] || [[ $file_size -lt 50000000 ]]; then
    log_error "Downloaded file invalid or too small ($file_size bytes, need ~150MB)"
    rm -f "$installer_path"
    return 1
  fi
  
  log_ok "Downloaded: $installer_path ($((file_size / 1024 / 1024))MB)"
  echo "$installer_path"
}

# ============================================================================
# INSTALL MINICONDA
# ============================================================================

install_miniconda() {
  log_info "Installing Miniconda to: $CONDA_INSTALL_PREFIX"
  
  # Check if already installed
  if [[ -d "$CONDA_INSTALL_PREFIX" ]] && [[ -f "$CONDA_INSTALL_PREFIX/bin/conda" ]]; then
    log_info "Miniconda already installed"
    return 0
  fi
  
  local installer
  installer=$(download_miniconda) || {
    log_error "Miniconda download failed"
    return 1
  }
  
  # Validate installer file is executable shell script
  if ! file "$installer" 2>/dev/null | grep -q "shell script\|Bourne\|bash"; then
    log_error "Downloaded file is not a valid shell script: $installer"
    log_error "File type: $(file "$installer" 2>/dev/null)"
    return 1
  fi
  
  # Handle partially existing prefix directories by removing them and doing a
  # clean install only (no in-place update mode).
  if [[ -d "$CONDA_INSTALL_PREFIX" ]] && [[ ! -f "$CONDA_INSTALL_PREFIX/bin/conda" ]]; then
    log_warn "Prefix exists but conda binary is missing: $CONDA_INSTALL_PREFIX"
    log_info "Removing broken prefix and reinstalling cleanly..."
    rm -rf "$CONDA_INSTALL_PREFIX"
    if ! bash "$installer" -b -p "$CONDA_INSTALL_PREFIX" 2>&1 | tee -a "$LOG_FILE"; then
      log_error "Miniconda installation command failed"
      return 1
    fi
  else
    # Run installer with -b (batch mode, no user input)
    log_info "Running installer (this may take 1-2 minutes)..."
    if ! bash "$installer" -b -p "$CONDA_INSTALL_PREFIX" 2>&1 | tee -a "$LOG_FILE"; then
      log_error "Miniconda installation command failed"
      return 1
    fi
  fi
  
  # Verify installation succeeded
  if [[ ! -f "$CONDA_INSTALL_PREFIX/bin/conda" ]]; then
    log_error "Installation completed but conda binary not found at: $CONDA_INSTALL_PREFIX/bin/conda"
    return 1
  fi
  
  log_ok "Miniconda installed: $CONDA_INSTALL_PREFIX"
}

# ============================================================================
# CREATE CONDA ENVIRONMENT
# ============================================================================

create_environment() {
  log_info "Creating conda environment: $ENV_NAME at $ENV_PATH"
  
  local conda_bin="$CONDA_INSTALL_PREFIX/bin/conda"
  local env_yml="$PROJECT_ROOT/environment.yml"

  if [[ ! -x "$conda_bin" ]]; then
    log_error "Conda binary not found or not executable: $conda_bin"
    return 1
  fi
  
  if [[ ! -f "$env_yml" ]]; then
    log_error "environment.yml not found at: $env_yml"
    return 1
  fi

  # Accept Anaconda Terms of Service in non-interactive HPC jobs when the
  # command is available. Safe to run repeatedly.
  if "$conda_bin" tos --help >/dev/null 2>&1; then
    log_info "Accepting Conda ToS for default Anaconda channels (non-interactive)..."
    "$conda_bin" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    "$conda_bin" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  fi
  
  # Check if environment already exists
  if [[ -d "$ENV_PATH" ]]; then
    log_info "Environment already exists at: $ENV_PATH"
    log_info "To recreate, run: rm -rf $ENV_PATH"
    return 0
  fi
  
  # Create environment from environment.yml
  log_info "Creating environment from: $env_yml"
  if ! "$conda_bin" env create -f "$env_yml" -p "$ENV_PATH" 2>&1 | tee -a "$LOG_FILE"; then
    log_error "Environment creation failed"
    return 1
  fi
  
  log_ok "Conda environment created: $ENV_PATH"
}

# ============================================================================
# VERIFICATION
# ============================================================================

verify_installation() {
  log_info "Verifying installation..."
  
  local python_bin="$ENV_PATH/bin/python"
  
  if [[ ! -f "$python_bin" ]]; then
    log_error "Python binary not found: $python_bin"
    return 1
  fi
  
  # Check Python version
  local py_version=$("$python_bin" --version)
  log_ok "Python: $py_version"
  
  # Check critical imports
  local checks=(
    "import bz2; print('✓ bz2')"
    "import ssl; print('✓ ssl')"
    "import sqlite3; print('✓ sqlite3')"
    "import numpy; print('✓ numpy')"
    "import astropy; print('✓ astropy')"
    "import photutils; print('✓ photutils')"
    "import tensorflow; print('✓ tensorflow')"
  )
  
  for check in "${checks[@]}"; do
    if "$python_bin" -c "$check" 2>&1 | tee -a "$LOG_FILE"; then
      :
    else
      log_warn "Check failed: $check"
    fi
  done
  
  log_ok "Installation verified"
}

print_summary() {
  log_info ""
  log_info "=========================================="
  log_info "SETUP COMPLETE"
  log_info "=========================================="
  log_info ""
  log_info "Paths:"
  log_info "  Miniconda: $CONDA_INSTALL_PREFIX"
  log_info "  Environment: $ENV_PATH"
  log_info "  Project: $PROJECT_ROOT"
  log_info ""
  log_info "Activation (for shell/SLURM jobs):"
  log_info "  source $CONDA_INSTALL_PREFIX/etc/profile.d/conda.sh"
  log_info "  conda activate $ENV_PATH"
  log_info ""
  log_info "Deactivation:"
  log_info "  conda deactivate"
  log_info ""
  log_info "Verification (on compute node with GPU):"
  log_info "  salloc -p gpu4 -N 1 --gres=gpu:1 -t 5:00"
  log_info "  source $CONDA_INSTALL_PREFIX/etc/profile.d/conda.sh"
  log_info "  conda activate $ENV_PATH"
  log_info "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
  log_info ""
  log_info "Submit training:"
  log_info "  sbatch bash/train_model_scenarios.sh"
  log_info ""
  log_info "Setup log: $LOG_FILE"
  log_info "=========================================="
}

# ============================================================================
# MAIN
# ============================================================================

main() {
  log_info "================================================================"
  log_info "HPC Environment Setup - Miniconda + TensorFlow GPU"
  log_info "================================================================"
  log_info "Conda prefix: $CONDA_INSTALL_PREFIX"
  log_info "Environment: $ENV_PATH"
  log_info "Clean start: $CLEAN_START"
  log_info "Log: $LOG_FILE"
  log_info ""

  if [[ "$CLEAN_START" -eq 1 ]]; then
    clean_start || return 1
  fi
  
  install_miniconda || return 1
  create_environment || return 1
  verify_installation || return 1
  
  print_summary
  log_ok "Setup succeeded"
}

main
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  log_error "Setup failed (code: $exit_code)"
fi

exit $exit_code
