#!/usr/bin/env bash
# Resume the two valid_seen jobs the master suite never reached.
#   ROLE=master -> OPD student @200 × valid_seen
#   ROLE=worker -> vanilla 30B-A3B  × valid_seen
set -xeuo pipefail

if [[ -z "${OPYTHON:-}" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    OPYTHON=/opt/conda/bin/python
  else
    OPYTHON=$(command -v python3)
  fi
fi
export OPYTHON
export CODE_DIR=${CODE_DIR:-/mnt/afs_reason/chendongyang/code/iclr/verl}
export DATA_ROOT=${DATA_ROOT:-/mnt/afs_reason/chendongyang/code/data}
export PYTHONUSERBASE=${PYTHONUSERBASE:-/mnt/afs_reason/chendongyang/conda/iclr_py311_user}
export PYTHONPATH=${CODE_DIR}:${PYTHONUSERBASE}/lib/python3.11/site-packages:${PYTHONPATH:-}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/afs_reason/chendongyang/.cache/alfworld}

EVAL_SH="${CODE_DIR}/recipe/alfworld_opd/eval_inrepo.sh"
TEACHER_BASE=${TEACHER_BASE:-${DATA_ROOT}/models/Qwen3-30B-A3B}
MERGED=${MERGED_HF:-/mnt/afs_reason/chendongyang/code/iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b/merged_hf/step200}
ROLE=${ROLE:-master}
SEEN_ROWS=${SEEN_ROWS:-140}

if [[ "${ROLE}" == "worker" ]]; then
  tag=teacher_base
  path=${TEACHER_BASE}
else
  tag=student_opd_step200
  path=${MERGED}
fi

echo "[remaining-seen] role=${ROLE} host=$(hostname) tag=${tag} rows=${SEEN_ROWS}"
MODEL_PATH="${path}" \
  MODEL_TAG="${tag}" \
  EVAL_SPLIT=eval_in_distribution \
  VAL_ROWS="${SEEN_ROWS}" \
  EXPERIMENT_NAME="eval_${tag}_eval_in_distribution_thinking" \
  bash "${EVAL_SH}"
echo "[remaining-seen] DONE role=${ROLE} tag=${tag}"
