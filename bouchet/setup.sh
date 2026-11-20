#!/bin/bash
# Build .venv with pip (no uv needed on the cluster), from requirements-cluster.txt: the exact versions in uv.lock
# that the tests ran on. Run from the repo root with bouchet/env.sh sourced. `bin/bouchet setup` runs it on a
# compute node through bouchet/setup-job.sh (pip installing torch + CUDA wheels was killed on a login node).
#
# Needs Python 3.11 or 3.12. Bouchet's system /usr/bin/python3.11 (RHEL 9) works and needs no module at job time:
#   echo 'export NAMES_PYTHON=/usr/bin/python3.11' >> bouchet/env.local.sh        # on the cluster
# Or load a Python module in bouchet/env.local.sh (sourced after `module purge` by every job too).
set -euo pipefail
unset PYTHONPATH PYTHONHOME EBPYTHONPREFIXES      # as in bouchet/env.sh: no module's packages may shadow .venv's
PY=${NAMES_PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))' 2>/dev/null; then
  echo "setup: need Python 3.11 or 3.12; '$PY' is $("$PY" --version 2>&1 || echo missing) ($(command -v "$PY" || echo not on PATH))."
  echo "setup: e.g. put 'export NAMES_PYTHON=/usr/bin/python3.11' in bouchet/env.local.sh,"
  echo "setup: or 'module load <a Python 3.12 module>' (see 'module spider Python')."
  exit 1
fi
# build from the base interpreter, even if $PY is itself inside a venv (possibly this one, about to be rebuilt)
PY=$("$PY" -c 'import sys; print(getattr(sys, "_base_executable", None) or sys.executable)')
if [ ! -f models/turion-extract-alpha/code/pyproject.toml ]; then
  echo "setup: models/turion-extract-alpha/code is missing here ($(pwd)); run bin/bouchet push-model first."
  exit 1
fi
DONE=.venv/.setup-complete               # written only after a full install and import check
if [ -d .venv ] && [ ! -f "$DONE" ]; then
  echo "setup: .venv is incomplete (an earlier install was interrupted); rebuilding it"
  rm -rf .venv
elif [ -x .venv/bin/python ] && ! .venv/bin/python -c '' 2>/dev/null; then
  echo "setup: .venv is broken (its Python moved or its module is not loaded); rebuilding it"
  rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
# --no-cache-dir: the CUDA wheels are ~3 GB and pip's cache would land in the home directory
.venv/bin/python -m pip install --no-cache-dir -r requirements-cluster.txt
.venv/bin/python -c "import sys, torch, transformers, peft, turion_extract
print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
date '+%Y-%m-%dT%H:%M:%S%z' > "$DONE"
echo "setup: .venv ready"
