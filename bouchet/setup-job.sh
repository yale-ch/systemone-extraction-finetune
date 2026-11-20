#!/bin/bash
# Run bouchet/setup.sh on a compute node with srun: the pip install of torch + ~3 GB of CUDA wheels was killed on
# a login node (per-user resource limits). Waits for the allocation and streams the output back; `bin/bouchet setup`
# calls this on the login node. Settings (environment or bouchet/env.local.sh):
#   NAMES_SETUP_PARTITION   default: the first of devel, day that sinfo lists
#   NAMES_ACCOUNT           Slurm account; default: your Slurm default account
#   NAMES_SETUP_MEM         default: 16G
set -euo pipefail
part=${NAMES_SETUP_PARTITION:-}
avail=$(sinfo -h -o %R 2>/dev/null | sort -u || true)
if [ -z "$part" ]; then
  for p in devel day; do
    if printf '%s\n' "$avail" | grep -qx "$p"; then part=$p; break; fi
  done
fi
if [ -z "$part" ]; then
  echo "setup: no devel or day partition found; set NAMES_SETUP_PARTITION (e.g. in bouchet/env.local.sh) to one of:"
  printf '  %s\n' $avail
  exit 1
fi
echo "setup: installing on a compute node (partition $part, account ${NAMES_ACCOUNT:-your default}); waiting for it..."
# the job inherits this shell's environment (bouchet/env.sh, env.local.sh, any loaded Python module)
exec srun --partition="$part" ${NAMES_ACCOUNT:+--account="$NAMES_ACCOUNT"} --job-name=names-setup \
  --cpus-per-task=4 --mem="${NAMES_SETUP_MEM:-16G}" --time=01:00:00 bash bouchet/setup.sh
