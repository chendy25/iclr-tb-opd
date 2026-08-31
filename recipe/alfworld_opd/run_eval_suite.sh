#!/usr/bin/env bash
# Merge the OPD student ckpt, then eval 3 models x 2 ALFWorld test splits.
#
# Models:
#   student_base  = vanilla Qwen3-4B
#   teacher_base  = vanilla Qwen3-30B-A3B (rolled out as the actor)
#   student_opd   = FSDP ckpt @ global_step_200, merged to HF
# Splits: valid_seen + full valid_unseen
#
# Intended to run on the job master after Ray is up (see _relaunch_alfworld_eval.sh).
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
FSDP_ACTOR=${FSDP_ACTOR:-/mnt/afs_reason/chendongyang/code/iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b/ckpt/global_step_200/actor}
MERGED=${MERGED_HF:-/mnt/afs_reason/chendongyang/code/iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b/merged_hf/step200}
SUITE_LOG=${SUITE_LOG:-/mnt/afs_reason/chendongyang/code/iclr/logs/verl_agent_alfworld_eval/suite-$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${SUITE_LOG}")" "${MERGED}"

log() { echo "[suite $(date +%H:%M:%S)] $*" | tee -a "${SUITE_LOG}"; }

# ---- merge OPD student FSDP -> HF (needed by vLLM) ----
if [[ -f "${MERGED}/config.json" ]] && compgen -G "${MERGED}/*.safetensors" > /dev/null; then
  log "skip merge, already have ${MERGED}"
else
  log "merging FSDP actor ${FSDP_ACTOR} -> ${MERGED}"
  "${OPYTHON}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${FSDP_ACTOR}" \
    --target_dir "${MERGED}" \
    --use_cpu_initialization
  [[ -f "${MERGED}/config.json" ]]
  ls "${MERGED}"/*.safetensors >/dev/null
  log "merge done"
fi

# ---- count registered games (solvable filter) ----
SEEN_ROWS=140
UNSEEN_ROWS=134
if count_out=$("${OPYTHON}" "${CODE_DIR}/recipe/alfworld_opd/count_split_games.py" 2>/dev/null); then
  log "game counts:"$'\n'"${count_out}"
  seen_n=$(echo "${count_out}" | awk '/valid_seen/{print $NF}')
  unseen_n=$(echo "${count_out}" | awk '/valid_unseen/{print $NF}')
  [[ -n "${seen_n}" && "${seen_n}" -gt 0 ]] && SEEN_ROWS=${seen_n}
  [[ -n "${unseen_n}" && "${unseen_n}" -gt 0 ]] && UNSEEN_ROWS=${unseen_n}
else
  log "count_split_games failed; using defaults seen=${SEEN_ROWS} unseen=${UNSEEN_ROWS}"
fi
log "VAL_ROWS seen=${SEEN_ROWS} unseen=${UNSEEN_ROWS}"

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

# 4B first (cheaper), then OPD student, then 30B (full-node TP).
run_one student_base "${STUDENT_BASE}" eval_in_distribution "${SEEN_ROWS}"
run_one student_base "${STUDENT_BASE}" eval_out_of_distribution "${UNSEEN_ROWS}"
run_one student_opd_step200 "${MERGED}" eval_in_distribution "${SEEN_ROWS}"
run_one student_opd_step200 "${MERGED}" eval_out_of_distribution "${UNSEEN_ROWS}"
run_one teacher_base "${TEACHER_BASE}" eval_in_distribution "${SEEN_ROWS}"
run_one teacher_base "${TEACHER_BASE}" eval_out_of_distribution "${UNSEEN_ROWS}"

log "ALL_ALFWORLD_EVALS_DONE seen=${SEEN_ROWS} unseen=${UNSEEN_ROWS}"
log "logs under ${DATA_ROOT}/../iclr/logs/verl_agent_alfworld_eval/"
