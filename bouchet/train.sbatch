#!/bin/bash
# One training run. Submit with bin/bouchet submit (it sets --job-name and --output), or on Bouchet:
#   sbatch --job-name=names-RUN --output=runs/RUN/slurm-%j.out bouchet/train.sbatch RUN [names_ft.train args]
# Options given to sbatch override these defaults. Gemma 4 12B + LoRA is the same model and GPU Turion trained it
# on (one RTX PRO 6000 Blackwell, 96 GB). No Slurm account is set here: bin/bouchet submit passes
# --account=$NAMES_ACCOUNT (from .env). By hand, pass --account or export SBATCH_ACCOUNT; else Slurm's default applies.
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
RUN=$1
shift
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source bouchet/env.sh
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is), restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# a requeued job (preemption, node failure) resumes from runs/$RUN/last
.venv/bin/python -m names_ft.train --out "runs/$RUN" "$@"
echo "finished $(date -Is)"
