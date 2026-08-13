#!/usr/bin/env bash
# Install ALFWorld / TextWorld deps into PYTHONUSERBASE for the job container
# python (/opt/conda, typically 3.11). Run ONCE inside pt-gv791b30 (master is enough
# if both pods share the same AFS user-site).
#
#   bash install_job_deps.sh
set -xeuo pipefail

PYTHON=${PYTHON:-/opt/conda/bin/python}
[[ -x "${PYTHON}" ]] || PYTHON=$(command -v python3)
export PYTHONUSERBASE=${PYTHONUSERBASE:-/mnt/afs_reason/chendongyang/conda/iclr_py311_user}
export PATH="$(dirname "${PYTHON}"):${PATH}"

"${PYTHON}" -m pip install --user -U \
  'alfworld' \
  'textworld' \
  'gymnasium' \
  'networkx' \
  'pyyaml'

STRICT=1 PYTHON="${PYTHON}" bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/check_env.sh"
echo "[install_job_deps] done. PYTHONUSERBASE=${PYTHONUSERBASE}"
