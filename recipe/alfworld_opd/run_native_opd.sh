#!/usr/bin/env bash
# =====================================================================
# ALFWorld native OPD on verl-family stack (refs/ATOD main_sod).
#
# Algorithm (pure OPD, no GRPO in the loss):
#   A = opd_coef * (log π_T − log π_S)   # mode=uniform, opd_only=true
#
# Student: Qwen3-4B
# Teacher: Qwen3-30B-A3B  (base; not ATOD's GRPO-ALFWorld teacher)
#
# Launch via iclr/logs/_relaunch_alfworld_opd.sh on a free 2-node job.
# Do not invoke from the login node.
# =====================================================================
set -xeuo pipefail

RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATOD_ROOT=${ATOD_ROOT:-/mnt/afs_reason/chendongyang/code/refs/ATOD}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-30B-A3B}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/afs_reason/chendongyang/.cache/alfworld}
# Shared AFS path (not $HOME) so both pods see the same stubs.
DATA_DIR=${DATA_DIR:-/mnt/afs_reason/chendongyang/code/data/verl-agent/text}

if [[ -x /opt/conda/bin/python ]]; then
  PYTHON=${PYTHON:-/opt/conda/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi
export PYTHONUSERBASE=${PYTHONUSERBASE:-/mnt/afs_reason/chendongyang/conda/iclr_py311_user}
export PYTHONPATH="${ATOD_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

# ---- pure OPD knobs ----
SOD_MODE=${SOD_MODE:-uniform}          # uniform | gated | stepwise  (native OPD => uniform)
OPD_ONLY=${OPD_ONLY:-true}
OPD_COEF=${OPD_COEF:-1.0}
USE_EXTERNAL_TEACHER=${USE_EXTERNAL_TEACHER:-true}

# ---- scale (match ATOD sod_trainer defaults; 2-node) ----
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
NNODES=${NNODES:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}
ENGINE=${ENGINE:-vllm}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
SKIP_PREPARE=${SKIP_PREPARE:-0}
SKIP_CHECK=${SKIP_CHECK:-0}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-alfworld_native_opd_qwen3_4b_from_30ba3b}
PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld}

cd "${ATOD_ROOT}"

if [[ "${SKIP_CHECK}" != "1" ]]; then
  STRICT=1 bash "${RECIPE_DIR}/check_env.sh"
fi

# Offline stubs (no HF). Override SKIP_PREPARE=1 to reuse existing parquet as-is.
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  mkdir -p "${DATA_DIR}"
  "${PYTHON}" "${RECIPE_DIR}/prepare_text_stubs.py" \
    --local_dir "${DATA_DIR}" \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"
fi

# Optional W&B (unset => console only still works if logger includes wandb and key missing may warn)
: "${WANDB_API_KEY:=}"

"${PYTHON}" -m verl.trainer.main_sod \
  algorithm.adv_estimator=grpo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size="${TRAIN_DATA_SIZE}" \
  data.val_batch_size="${VAL_DATA_SIZE}" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  +algorithm.sod.use_external_teacher="${USE_EXTERNAL_TEACHER}" \
  +algorithm.sod.mode="${SOD_MODE}" \
  +algorithm.sod.opd_coef="${OPD_COEF}" \
  +algorithm.sod.opd_only="${OPD_ONLY}" \
  +algorithm.sod.skills_dir=skills/alfworld \
  +algorithm.sod.skill_all=false \
  +ray_init.runtime_env.env_vars.PYTHONPATH="${ATOD_ROOT}" \
  +ray_init.runtime_env.env_vars.ALFWORLD_DATA="${ALFWORLD_DATA}" \
  +ray_init.runtime_env.env_vars.PYTHONUSERBASE="${PYTHONUSERBASE}" \
  +ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM=true \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.rollout.n="${GROUP_SIZE}" \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.ray_wait_register_center_timeout=600 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=True \
  "$@"
