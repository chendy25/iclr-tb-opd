#!/usr/bin/env bash
# =====================================================================
# ALFWorld val_only eval on the in-repo agent loop (no distillation).
#
# One model, one split. The stub's extra_info.index seeds the game, so
# VAL_ROWS should equal the split's registered game count for full coverage.
#
# Usage:
#   MODEL_PATH=.../Qwen3-4B EVAL_SPLIT=eval_out_of_distribution VAL_ROWS=134 \
#       bash recipe/alfworld_opd/eval_inrepo.sh
#   EVAL_SPLIT=eval_in_distribution  -> valid_seen
#   EVAL_SPLIT=eval_out_of_distribution -> valid_unseen
# =====================================================================
set -xeuo pipefail

export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6379}

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
export PATH=/opt/conda/bin:${CUDA_HOME:-/usr/local/cuda}/bin:${PYTHONUSERBASE}/bin:${PATH}

export ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/afs_reason/chendongyang/.cache/alfworld}
export ALFWORLD_TRAIN_EVAL=${ALFWORLD_TRAIN_EVAL:-train}
export ALFWORLD_EVAL_SPLIT=${EVAL_SPLIT:-${ALFWORLD_EVAL_SPLIT:-eval_out_of_distribution}}

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-Z33thTEDb9YisL16DCoL7}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export MODELING_BACKEND=${MODELING_BACKEND:-hf}

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to an HF dir}
[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH is not a dir: ${MODEL_PATH}"; exit 1; }
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH}")}

# Official ALFWorld 2.1.1 after AlfredTWEnv's solvable filter: seen=140, unseen=134.
# Override VAL_ROWS after counting on the job if the filter differs.
case "${ALFWORLD_EVAL_SPLIT}" in
  eval_in_distribution|valid_seen)
    export ALFWORLD_EVAL_SPLIT=eval_in_distribution
    default_rows=140
    split_tag=valid_seen
    ;;
  eval_out_of_distribution|valid_unseen)
    export ALFWORLD_EVAL_SPLIT=eval_out_of_distribution
    default_rows=134
    split_tag=valid_unseen
    ;;
  *)
    echo "Unknown EVAL_SPLIT=${ALFWORLD_EVAL_SPLIT}"
    exit 1
    ;;
esac
val_rows=${VAL_ROWS:-${default_rows}}

STUB_DIR=${ALFWORLD_EVAL_STUB_DIR:-${DATA_ROOT}/verl-agent/alfworld_inrepo_eval}
"${OPYTHON}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/prepare_text_stubs.py" \
  --local_dir "${STUB_DIR}/${split_tag}" \
  --train_data_size 1 \
  --val_data_size "${val_rows}"
AGENTRL_TRAIN=${STUB_DIR}/${split_tag}/train.parquet
AGENTRL_VAL=${STUB_DIR}/${split_tag}/test.parquet
train_files="['${AGENTRL_TRAIN}']"
val_files="['${AGENTRL_VAL}']"

# 30B-A3B needs a full node of TP; 4B can shard TP=2 across 8 GPUs.
if [[ "${MODEL_PATH}" == *"30B"* || "${MODEL_PATH}" == *"30b"* ]]; then
  rollout_tp=${ROLLOUT_TP:-8}
  rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.85}
else
  rollout_tp=${ROLLOUT_TP:-2}
  rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.55}
fi

val_n=${VAL_N:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
# Tracks train_inrepo_opd.sh ATOD/TCOD protocol (enable_thinking=False, ~294 tok/step).
# Thinking-protocol evals must override MAX_RESPONSE_LENGTH=30720 and
# PPO_MAX_TOKEN_LEN_PER_GPU=40960 or they measure the cap instead of the checkpoint.
max_response_length=${MAX_RESPONSE_LENGTH:-20480}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
# Must exceed max_num_tokens (22529 here); these two move together.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
alfworld_max_steps=${ALFWORLD_MAX_STEPS:-50}
alfworld_pool_size=${ALFWORLD_POOL_SIZE:-16}
alfworld_max_turn_tokens=${ALFWORLD_MAX_TURN_TOKENS:-512}
alfworld_config_path=${ALFWORLD_CONFIG_PATH:-${CODE_DIR}/verl/experimental/agent_loop/alfworld_env/config_tw.yaml}
# Match the training-time val protocol (val_kwargs.temperature=0.4).
val_temperature=${VAL_TEMPERATURE:-0.4}
# False = ATOD/TCOD: template pre-fills empty <think></think>.
# True = model generates the think tags (pair with MAX_RESPONSE_LENGTH=30720).
enable_thinking=${ENABLE_THINKING:-False}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}

project_name=${PROJECT_NAME:-verl_agent_alfworld_eval}
experiment_name=${EXPERIMENT_NAME:-eval_${MODEL_TAG}_${split_tag}_atodproto}
trainer_logger=${TRAINER_LOGGER:-"['console','swanlab']"}
result_dir=${RESULT_DIR:-${DATA_ROOT}/../iclr/logs/${project_name}/${experiment_name}}
ckpt_dir=${result_dir}/ckpt
rollout_dir=${result_dir}/rollout
mkdir -p "${ckpt_dir}" "${rollout_dir}" "${result_dir}"

echo "ALFWORLD EVAL: model=${MODEL_PATH} tag=${MODEL_TAG}"
echo "  split=${ALFWORLD_EVAL_SPLIT} (${split_tag}) val_rows=${val_rows} n=${val_n} temp=${val_temperature}"
echo "  enable_thinking=${enable_thinking} turn_tokens=${alfworld_max_turn_tokens}"
echo "  tp=${rollout_tp} gpu_mem=${rollout_gpu_mem_util} max_tokens=${max_num_tokens}"
echo "  out=${result_dir}"

# Master suite stays sequential; worker-assigned experiment names are skipped here
# so the two 8-GPU nodes do not launch the same val_only job.
SKIP_ON_MASTER=${EVAL_SKIP_ON_MASTER:-${DATA_ROOT}/../iclr/logs/${project_name}/SKIP_ON_MASTER}
if [[ -f "${SKIP_ON_MASTER}" ]] && hostname | grep -q -- '-master-'; then
  if grep -Fxq "${experiment_name}" "${SKIP_ON_MASTER}"; then
    echo "SKIP ${experiment_name} (assigned to worker via ${SKIP_ON_MASTER})"
    exit 0
  fi
fi

${OPYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${enable_thinking} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${val_temperature} \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${alfworld_max_steps} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${alfworld_max_steps} \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +alfworld.max_steps=${alfworld_max_steps} \
    +alfworld.pool_size=${alfworld_pool_size} \
    +alfworld.max_turn_tokens=${alfworld_max_turn_tokens} \
    +alfworld.config_path="${alfworld_config_path}" \
    +alfworld.train_eval=${ALFWORLD_TRAIN_EVAL} \
    +alfworld.eval_split=${ALFWORLD_EVAL_SPLIT} \
    trainer.balance_batch=False \
    trainer.logger="${trainer_logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=0 \
    trainer.test_freq=0 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${ckpt_dir} \
    trainer.rollout_data_dir=${rollout_dir} \
    trainer.validation_data_dir=${rollout_dir} \
    trainer.v1.sampler.sync_refill_failed_groups=True \
    distillation.enabled=False \
    distillation.nnodes=0 \
    "$@" \
    2>&1 | tee "${result_dir}/eval-$(date +%Y%m%d_%H%M%S).log"
