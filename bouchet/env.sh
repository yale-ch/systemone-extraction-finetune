# Sourced by every job and by bin/bouchet on the login node. No user, account or path is hardcoded: set them in
# .env on the laptop (bin/bouchet forwards them over ssh, and jobs inherit them) or in bouchet/env.local.sh on the
# cluster (not synced; sourced below, so it wins). See .env.example.
NAMES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# training only: no hosted model APIs anywhere in this workflow
export GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY= TURION_API_TOKEN=
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1; module load CUDA/12.8.0 >/dev/null 2>&1 || true; fi
if [ -f "$NAMES_DIR/bouchet/env.local.sh" ]; then source "$NAMES_DIR/bouchet/env.local.sh"; fi
# The HF cache can be shared with other projects, so a base model is downloaded once. Deliberately not
# ${HF_HOME:-...}: a HF_HOME from ~/.bashrc may point at a small or nearly full fileset.
export HF_HOME=${NAMES_HF_HOME:-$NAMES_DIR/hf_cache}
# .venv must see only its own packages. EasyBuild bundles (e.g. Python-bundle-PyPI/2024.06, a Python 3.12 module)
# put their site-packages on PYTHONPATH, which outranks the venv's and breaks compiled packages under another Python
unset PYTHONPATH PYTHONHOME EBPYTHONPREFIXES
true
