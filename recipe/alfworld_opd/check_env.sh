#!/usr/bin/env bash
# Preflight for Phase 2' ALFWorld native OPD (ATOD stack).
# Safe on login node or inside the job; does not start training.
#
#   bash check_env.sh              # paths + soft import warnings
#   STRICT=1 bash check_env.sh     # also require ray/torch/vllm/textworld (job)
set -euo pipefail

ATOD_ROOT=${ATOD_ROOT:-/mnt/afs_reason/chendongyang/code/refs/ATOD}
ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/afs_reason/chendongyang/.cache/alfworld}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-30B-A3B}
DATA_DIR=${DATA_DIR:-/mnt/afs_reason/chendongyang/code/data/verl-agent/text}
STRICT=${STRICT:-0}

# Prefer job container python; fall back to PATH.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    PYTHON=/opt/conda/bin/python
  else
    PYTHON=$(command -v python3)
  fi
fi

export ATOD_ROOT
export ALFWORLD_DATA
# Only prepend ATOD; do not clobber user-site on login via PYTHONUSERBASE.
export PYTHONPATH="${ATOD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

fail=0
check() {
  local label=$1
  shift
  if "$@"; then
    echo "[ok] ${label}"
  else
    echo "[FAIL] ${label}"
    fail=1
  fi
}

echo "=== alfworld_opd check_env (STRICT=${STRICT}) ==="
echo "PYTHON=${PYTHON}"
echo "ATOD_ROOT=${ATOD_ROOT}"
echo "ALFWORLD_DATA=${ALFWORLD_DATA}"
echo

check "ATOD root" test -d "${ATOD_ROOT}/verl/trainer"
check "main_sod" test -f "${ATOD_ROOT}/verl/trainer/main_sod.py"
check "skills/alfworld" test -d "${ATOD_ROOT}/skills/alfworld"
check "ALFWORLD json_2.1.1/train" test -d "${ALFWORLD_DATA}/json_2.1.1/train"
check "ALFWORLD logic" test -f "${ALFWORLD_DATA}/logic/alfred.pddl"
check "student config.json" test -f "${STUDENT_MODEL}/config.json"
check "teacher config.json" test -f "${TEACHER_MODEL}/config.json"

if [[ -f "${DATA_DIR}/train.parquet" && -f "${DATA_DIR}/test.parquet" ]]; then
  echo "[ok] stub parquet present under ${DATA_DIR}"
else
  echo "[warn] stub parquet missing under ${DATA_DIR} (run_native_opd.sh will generate)"
fi

# Soft imports on login; STRICT=1 on the job (container + user site).
export CHECK_STRICT="${STRICT}"
"${PYTHON}" - <<'PY' || fail=1
import importlib, os, sys

strict = os.environ.get("CHECK_STRICT", "0") == "1"
hard = ["pyarrow"]  # needed for offline stub prep / dataset IO
soft = ["ray", "torch", "vllm", "omegaconf", "hydra", "gymnasium", "yaml", "textworld"]

def try_mod(name):
    try:
        importlib.import_module(name)
        return None
    except Exception as e:
        return str(e)

hard_miss, soft_miss = [], []
for m in hard:
    err = try_mod(m)
    if err:
        hard_miss.append(f"{m}: {err}")
for m in soft:
    err = try_mod(m)
    if err:
        soft_miss.append(f"{m}: {err}")

if hard_miss:
    print("[FAIL] required imports:\n  " + "\n  ".join(hard_miss))
    sys.exit(1)
print("[ok] required imports (pyarrow)")

if soft_miss:
    tag = "FAIL" if strict else "warn"
    print(f"[{tag}] training imports missing (ok on login; need on job):\n  " + "\n  ".join(soft_miss))
    if strict:
        sys.exit(1)
else:
    print("[ok] training imports (ray/torch/vllm/...)")

sys.path.insert(0, os.environ.get("ATOD_ROOT", "."))
try:
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment  # noqa: F401
    print("[ok] ATOD alfworld get_environment import")
except Exception as e:
    # May fail without textworld/gymnasium on login
    tag = "FAIL" if strict else "warn"
    print(f"[{tag}] ATOD alfworld import: {e}")
    if strict:
        sys.exit(1)
PY

if [[ "${fail}" -ne 0 ]]; then
  echo "=== check_env FAILED ==="
  exit 1
fi
echo "=== check_env PASSED ==="
