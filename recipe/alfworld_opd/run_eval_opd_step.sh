#!/usr/bin/env bash
# Eval one OPD student FSDP ckpt on one ALFWorld split.
#   STEP=100 SPLIT=eval_in_distribution VAL_ROWS=140 bash run_eval_opd_step.sh
#   STEP=100 SPLIT=eval_out_of_distribution VAL_ROWS=134 bash run_eval_opd_step.sh
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

STEP=${STEP:?set STEP (e.g. 100 for 1 epoch)}
SPLIT=${SPLIT:-eval_out_of_distribution}
VAL_ROWS=${VAL_ROWS:-134}
DO_MERGE=${DO_MERGE:-0}

CKPT_ROOT=${CKPT_ROOT:-/mnt/afs_reason/chendongyang/code/iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b}
FSDP_ACTOR=${FSDP_ACTOR:-${CKPT_ROOT}/ckpt/global_step_${STEP}/actor}
MERGED=${MERGED_HF:-${CKPT_ROOT}/merged_hf/step${STEP}}
EVAL_SH="${CODE_DIR}/recipe/alfworld_opd/eval_inrepo.sh"
tag="student_opd_step${STEP}"

if [[ "${DO_MERGE}" == "1" ]]; then
  mkdir -p "${MERGED}"
  if [[ -f "${MERGED}/config.json" ]] && compgen -G "${MERGED}/*.safetensors" > /dev/null; then
    echo "[opd-step] skip merge, already have ${MERGED}"
  else
    echo "[opd-step] merging ${FSDP_ACTOR} -> ${MERGED}"
    "${OPYTHON}" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${FSDP_ACTOR}" \
      --target_dir "${MERGED}" \
      --use_cpu_initialization
    [[ -f "${MERGED}/config.json" ]]
    ls "${MERGED}"/*.safetensors >/dev/null
    echo "[opd-step] merge done"
  fi
else
  echo "[opd-step] waiting for merged HF ${MERGED}"
  for i in $(seq 1 180); do
    if [[ -f "${MERGED}/config.json" ]] && compgen -G "${MERGED}/*.safetensors" > /dev/null; then
      echo "[opd-step] merge ready"
      break
    fi
    sleep 5
  done
  [[ -f "${MERGED}/config.json" ]] || { echo "merged HF missing: ${MERGED}"; exit 1; }
fi

echo "[opd-step] host=$(hostname) tag=${tag} split=${SPLIT} rows=${VAL_ROWS} path=${MERGED}"
MODEL_PATH="${MERGED}" \
  MODEL_TAG="${tag}" \
  EVAL_SPLIT="${SPLIT}" \
  VAL_ROWS="${VAL_ROWS}" \
  EXPERIMENT_NAME="eval_${tag}_${SPLIT}_thinking" \
  bash "${EVAL_SH}"
echo "[opd-step] DONE tag=${tag} split=${SPLIT}"
