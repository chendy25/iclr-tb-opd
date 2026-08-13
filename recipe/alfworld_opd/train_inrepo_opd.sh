#!/usr/bin/env bash
# =====================================================================
# Phase 2' ALFWorld native OPD — IN-REPO (iclr/verl), no refs/ATOD.
#
# Env interaction: verl V1 agent loop `alfworld_agent`
#   (verl/experimental/agent_loop/alfworld_agent_loop.py) — one episode per run,
#   assistant tokens masked 1, environment-observation tokens masked 0.
# Learning objective: the repo's own distillation OPD, identical machinery to
#   Phase 1' agentic TB-OPD (B-A0):
#     loss_mode=k1 + use_policy_gradient=True + use_task_rewards=False
#   => per-token PG advantage = -(logπ_S - logπ_T) = (logπ_T - logπ_S)  [pure OPD]
#
# Student: Qwen3-4B     Teacher: Qwen3-30B-A3B
#
# Run on allocated nodes (RANK/MASTER_* from cluster). Requires the alfworld
# runtime on the job python: recipe/alfworld_opd/install_job_deps.sh.
# =====================================================================
set -xeuo pipefail

# ---- cluster ----
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6379}

# ---- env / code ----
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

# ---- models ----
STUDENT_MODEL=${STUDENT_MODEL:-${DATA_ROOT}/models/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-${DATA_ROOT}/models/Qwen3-30B-A3B}

# ---- data (offline stub parquet; games come from $ALFWORLD_DATA) ----
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/verl-agent/text}
AGENTRL_TRAIN=${ALFWORLD_TRAIN:-${DATA_DIR}/train.parquet}
AGENTRL_VAL=${ALFWORLD_VAL:-${DATA_DIR}/test.parquet}
train_files="['${AGENTRL_TRAIN}']"
val_files="['${AGENTRL_VAL}']"

# ---- scale ----
train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
rollout_n=${ROLLOUT_N:-8}                 # GRPO/OPD group size (env.rollout.n)
val_n=${VAL_N:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
actor_lr=${ACTOR_LR:-1e-6}

# ---- alfworld env loop ----
alfworld_max_steps=${ALFWORLD_MAX_STEPS:-30}
alfworld_pool_size=${ALFWORLD_POOL_SIZE:-16}
alfworld_max_turn_tokens=${ALFWORLD_MAX_TURN_TOKENS:-512}
alfworld_config_path=${ALFWORLD_CONFIG_PATH:-${CODE_DIR}/verl/experimental/agent_loop/alfworld_env/config_tw.yaml}

# ---- rollout / teacher parallelism ----
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.55}
teacher_tp=${TEACHER_TP:-8}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}

# ---- OPD hyperparams (native OPD == k1 + PG, no task reward) ----
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# ---- cluster split (student trainer node0, teacher node1) ----
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}
DISTILL_NNODES=${DISTILL_NNODES:-1}
DISTILL_NGPUS_PER_NODE=${DISTILL_NGPUS_PER_NODE:-8}

# ---- trainer bookkeeping ----
project_name=${PROJECT_NAME:-verl_agent_alfworld}
experiment_name=${EXPERIMENT_NAME:-alfworld_inrepo_opd_qwen3_4b_from_30ba3b}
trainer_logger=${TRAINER_LOGGER:-"['console','swanlab']"}
total_epochs=${TOTAL_EPOCHS:-150}
save_freq=${SAVE_FREQ:-25}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-True}
ckpt_dir=${CKPT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}/ckpt}
rollout_dir=${ROLLOUT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}/rollout}
result_dir=${RESULT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}}
mkdir -p "${ckpt_dir}" "${rollout_dir}" "${result_dir}"

${OPYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${alfworld_max_steps} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${alfworld_max_steps} \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +alfworld.max_steps=${alfworld_max_steps} \
    +alfworld.pool_size=${alfworld_pool_size} \
    +alfworld.max_turn_tokens=${alfworld_max_turn_tokens} \
    +alfworld.config_path="${alfworld_config_path}" \
    +alfworld.train_eval=${ALFWORLD_TRAIN_EVAL} \
    trainer.balance_batch=True \
    trainer.logger="${trainer_logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=${ckpt_dir} \
    trainer.rollout_data_dir=${rollout_dir} \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${DISTILL_NGPUS_PER_NODE} \
    distillation.nnodes=${DISTILL_NNODES} \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
    distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
    distillation.distillation_loss.topk=${distillation_topk} \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    distillation.tb_opd.enable=False \
    "$@" \
    2>&1 | tee "${result_dir}/train-$(date +%Y%m%d_%H%M%S).log"
