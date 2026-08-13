#!/usr/bin/env bash
# =====================================================================
# Agentic TB-OPD — SOD-suite evaluation (val_only).
#
# Benchmarks (Open-AgentRL-Eval, TIR tool-agent parquet):
#   AIME2024 / AIME2025 / GPQA-Diamond / LiveCodeBench-v6
# Metrics (verl process_validation_metrics):
#   mean@N (= avg@N / acc), best@N ≈ pass@N, maj@N
#
# Protocol aligned with SOD examples/SOD/eval/run_eval_*.sh:
#   n=32, temp=1.0, top_p=0.6, max_turns=16, response=20480
# Sandbox defaults to local subprocess (same as Phase-1' training).
#
# Usage:
#   MODEL_PATH=.../Qwen3-4B bash recipe/agentic_tbopd/eval_sod_suite.sh
#   MODEL_PATH=.../SOD-GRPO_teacher-4B MODEL_TAG=sod_teacher_4b \
#       bash recipe/agentic_tbopd/eval_sod_suite.sh
# =====================================================================
set -xeuo pipefail

# ---- env / code ----
if [[ -z "${OPYTHON:-}" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    OPYTHON=/opt/conda/bin/python
  else
    OPYTHON=/mnt/afs_reason/chendongyang/conda/opd/bin/python3.12
  fi
fi
export OPYTHON
export CODE_DIR=${CODE_DIR:-/mnt/afs_reason/chendongyang/code/iclr/verl}
export DATA_ROOT=${DATA_ROOT:-/mnt/afs_reason/chendongyang/code/data}
export PYTHONUSERBASE=${PYTHONUSERBASE:-/mnt/afs_reason/chendongyang/conda/iclr_py311_user}
export PYTHONPATH=${CODE_DIR}:${PYTHONUSERBASE}/lib/python3.11/site-packages:${PYTHONPATH:-}
export PATH=/opt/conda/bin:${CUDA_HOME:-/usr/local/cuda}/bin:${PYTHONUSERBASE}/bin:${PATH}

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}
export MODELING_BACKEND=${MODELING_BACKEND:-hf}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}

# Sandbox: generation tool + LCB reward execution (local subprocess default).
export SANDBOX_BACKEND=${SANDBOX_BACKEND:-local}
export SANDBOX_EXEC_BACKEND=${SANDBOX_EXEC_BACKEND:-local}
export LOCAL_EXEC_MAX_CONCURRENT=${LOCAL_EXEC_MAX_CONCURRENT:-16}
export CODE_REWARD_TOTAL_TIMEOUT=${CODE_REWARD_TOTAL_TIMEOUT:-120}
export E2B_API_KEY=${E2B_API_KEY:-}
export E2B_DOMAIN=${E2B_DOMAIN:-ap-beijing.tencentags.com}
export E2B_VALIDATE_API_KEY=${E2B_VALIDATE_API_KEY:-false}

cd "${CODE_DIR}"

# ---- model ----
MODEL_PATH=${MODEL_PATH:-}
if [[ $# -gt 0 && "${1}" != *=* ]]; then
  MODEL_PATH="$1"
  shift
fi
[[ -n "${MODEL_PATH}" && -d "${MODEL_PATH}" ]] || {
  echo "MODEL_PATH missing or not a dir: '${MODEL_PATH}'"
  exit 1
}
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH}")}

# ---- data ----
TIR_ROOT=${TIR_ROOT:-${DATA_ROOT}/preprocessed/open_agentrl_tir}
AIME_2024=${AIME_2024:-${TIR_ROOT}/test_aime2024.parquet}
AIME_2025=${AIME_2025:-${TIR_ROOT}/test_aime2025.parquet}
GPQA=${GPQA:-${TIR_ROOT}/test_gpqa_diamond.parquet}
LCB=${LCB:-${TIR_ROOT}/test_livecodebench_v6.parquet}
EVAL_SETS=${EVAL_SETS:-all}   # all | aime | gpqa | lcb

case "${EVAL_SETS}" in
  all)
    EVAL_FILES="['${AIME_2024}','${AIME_2025}','${GPQA}','${LCB}']"
    ;;
  aime)
    EVAL_FILES="['${AIME_2024}','${AIME_2025}']"
    ;;
  gpqa)
    EVAL_FILES="['${GPQA}']"
    ;;
  lcb)
    EVAL_FILES="['${LCB}']"
    ;;
  *)
    # Allow raw hydra list override, e.g. EVAL_SETS="['/path/a.parquet']"
    EVAL_FILES="${EVAL_SETS}"
    ;;
esac

for f in "${AIME_2024}" "${AIME_2025}" "${GPQA}" "${LCB}"; do
  # Only require files that appear in EVAL_FILES.
  if [[ "${EVAL_FILES}" == *"${f}"* ]]; then
    [[ -f "${f}" ]] || { echo "Missing eval parquet: ${f} (run prepare_open_agentrl.py --eval_only)"; exit 1; }
  fi
done

# ---- tool ----
if [[ -z "${TOOL_CONFIG:-}" ]]; then
  if [[ "${SANDBOX_BACKEND}" == "sandbox_fusion" ]]; then
    TOOL_CONFIG=${CODE_DIR}/recipe/agentic_tbopd/config/sandbox_fusion_tool_config.yaml
  elif [[ "${SANDBOX_BACKEND}" == "e2b" ]]; then
    TOOL_CONFIG=${CODE_DIR}/recipe/agentic_tbopd/config/e2b_tool_config.yaml
  else
    TOOL_CONFIG=${CODE_DIR}/recipe/agentic_tbopd/config/local_tool_config.yaml
  fi
fi
[[ -f "${TOOL_CONFIG}" ]] || { echo "Missing tool config: ${TOOL_CONFIG}"; exit 1; }

# ---- eval hyperparams (SOD defaults) ----
MAX_TURNS=${MAX_TURNS:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
VAL_N=${VAL_N:-32}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.6}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.70}
ROLLOUT_TP=${ROLLOUT_TP:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
FSDP_OFFLOAD=${FSDP_OFFLOAD:-True}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1 ))}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}

PROJECT_NAME=${PROJECT_NAME:-agentic_tb_opd_eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${MODEL_TAG}_${EVAL_SETS}_n${VAL_N}}
RESULT_DIR=${RESULT_DIR:-${CODE_DIR}/output/${PROJECT_NAME}/${EXPERIMENT_NAME}}
CKPT_DIR=${RESULT_DIR}/checkpoints
EVAL_DUMP_DIR=${RESULT_DIR}/eval_generations
mkdir -p "${CKPT_DIR}" "${EVAL_DUMP_DIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a __gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS=${#__gpu_list[@]}
else
  NUM_GPUS=${NUM_GPUS:-${NGPUS_PER_NODE}}
fi

LOGGER_CFG=${LOGGER_CFG:-'["console","swanlab"]'}

echo "AGENTIC EVAL: model=${MODEL_PATH} tag=${MODEL_TAG}"
echo "  eval_files=${EVAL_FILES}"
echo "  n=${VAL_N} temp=${VAL_TEMPERATURE} top_p=${VAL_TOP_P} turns=${MAX_TURNS}"
echo "  resp=${MAX_RESPONSE_LENGTH} tp=${ROLLOUT_TP} gpus=${NUM_GPUS}"
echo "  sandbox=${SANDBOX_BACKEND} tool=${TOOL_CONFIG}"
echo "  out=${RESULT_DIR}"

# ---- Ray ----
ray stop --force 2>/dev/null || true
pkill -9 -f 'vllm' 2>/dev/null || true
sleep 2
ray start --head --port=${MASTER_PORT:-6379} --disable-usage-stats --include-dashboard=false
sleep 5

# ---- launch (no distillation; val_only) ----
${OPYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${EVAL_FILES}" \
    data.val_files="${EVAL_FILES}" \
    data.return_raw_chat=True \
    data.train_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OFFLOAD} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 )) \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    distillation.enabled=False \
    trainer.logger="${LOGGER_CFG}" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=0 \
    trainer.test_freq=0 \
    trainer.save_freq=-1 \
    trainer.log_val_generations=20 \
    trainer.validation_data_dir="${EVAL_DUMP_DIR}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    "$@" \
    2>&1 | tee "${RESULT_DIR}/eval-$(date +%Y%m%d_%H%M%S).log"

ray stop --force 2>/dev/null || true
echo "EVAL_DONE model=${MODEL_TAG} out=${RESULT_DIR}"
