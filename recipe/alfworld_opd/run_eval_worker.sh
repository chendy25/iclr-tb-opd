#!/usr/bin/env bash
# Remaining ALFWorld evals assigned to the idle worker node.
# Isolated Ray head (do not join master's cluster).
#
#   OPD student @ step200 × valid_unseen
#   vanilla 30B-A3B       × valid_unseen
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
SUITE_LOG=${SUITE_LOG:-/mnt/afs_reason/chendongyang/code/iclr/logs/verl_agent_alfworld_eval/suite-worker-$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${SUITE_LOG}")"

[[ -f "${MERGED}/config.json" ]] || { echo "merged HF missing: ${MERGED}"; exit 1; }

log() { echo "[suite-worker $(date +%H:%M:%S)] $*" | tee -a "${SUITE_LOG}"; }

UNSEEN_ROWS=${UNSEEN_ROWS:-134}
log "worker host=$(hostname) unseen_rows=${UNSEEN_ROWS} merged=${MERGED}"

run_one() {
  local tag="$1" path="$2" split="$3" rows="$4"
  log "START tag=${tag} split=${split} rows=${rows} path=${path}"
  MODEL_PATH="${path}" \
  MODEL_TAG="${tag}" \
  EVAL_SPLIT="${split}" \
  VAL_ROWS="${rows}" \
  EXPERIMENT_NAME="eval_${tag}_${split}_thinking" \
    bash "${EVAL_SH}"
  log "DONE tag=${tag} split=${split}"
}

run_one student_opd_step200 "${MERGED}" eval_out_of_distribution "${UNSEEN_ROWS}"
run_one teacher_base "${TEACHER_BASE}" eval_out_of_distribution "${UNSEEN_ROWS}"

log "WORKER_ALFWORLD_EVALS_DONE"
