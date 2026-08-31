#!/usr/bin/env bash
# Re-eval valid_seen with a 28k response budget so long rooms are not
# cut mid-episode. Writes to eval_*_resp28k so the 20k numbers stay.
#
#   ROLE=master -> vanilla 4B, then OPD @200
#   ROLE=worker -> OPD @100, then vanilla 30B-A3B
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
STUDENT_BASE=${STUDENT_BASE:-${DATA_ROOT}/models/Qwen3-4B}
TEACHER_BASE=${TEACHER_BASE:-${DATA_ROOT}/models/Qwen3-30B-A3B}
CKPT_ROOT=${CKPT_ROOT:-/mnt/afs_reason/chendongyang/code/iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b}
MERGED_100=${CKPT_ROOT}/merged_hf/step100
MERGED_200=${CKPT_ROOT}/merged_hf/step200
ROLE=${ROLE:-master}
SEEN_ROWS=${SEEN_ROWS:-140}
RESP=${MAX_RESPONSE_LENGTH:-28672}

# 2048 + 28672 + 1 = 30721; keep actor token budget above that.
export MAX_RESPONSE_LENGTH="${RESP}"
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

run_one() {
  local tag="$1" path="$2"
  echo "[seen-28k $(date +%H:%M:%S)] START tag=${tag} rows=${SEEN_ROWS} resp=${RESP} path=${path}"
  MODEL_PATH="${path}" \
    MODEL_TAG="${tag}" \
    EVAL_SPLIT=eval_in_distribution \
    VAL_ROWS="${SEEN_ROWS}" \
    EXPERIMENT_NAME="eval_${tag}_eval_in_distribution_resp28k_thinking" \
    bash "${EVAL_SH}"
  echo "[seen-28k $(date +%H:%M:%S)] DONE tag=${tag}"
}

echo "[seen-28k] role=${ROLE} host=$(hostname) resp=${RESP}"
if [[ "${ROLE}" == "worker" ]]; then
  run_one student_opd_step100 "${MERGED_100}"
  run_one teacher_base "${TEACHER_BASE}"
else
  run_one student_base "${STUDENT_BASE}"
  run_one student_opd_step200 "${MERGED_200}"
fi
echo "[seen-28k] ALL_DONE role=${ROLE}"
