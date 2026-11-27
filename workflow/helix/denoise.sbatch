#!/bin/bash
# astrounets denoise on a Helix GPU node (from the repository folder); arguments go to the command as
# they are (astrounets denoise --help). Quote patterns so the command expands them, not the shell:
#   sbatch workflow/helix/denoise.sbatch data/new/ --out denoised/
#   sbatch workflow/helix/denoise.sbatch "data/new/*_drz.fits" --model runs/_workflow/report/models/all/<run>
# Without --model: the study's final model. Log: denoise-<job id>.log in the repository folder.
#SBATCH --job-name=astrounets-denoise
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=denoise-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -x .venv/bin/astrounets ] || { echo "no .venv: run  sbatch workflow/helix/setup.sbatch  first"; exit 1; }
exec .venv/bin/astrounets denoise "$@"
