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
# Validation rollouts play held-out games: eval_out_of_distribution = valid_unseen
# (unseen rooms), eval_in_distribution = valid_seen.
export ALFWORLD_EVAL_SPLIT=${ALFWORLD_EVAL_SPLIT:-eval_out_of_distribution}

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
# The stub carries no task content: one row == one episode slot, and its
# ``extra_info.index`` seeds ALFWorld's game draw. So #rows = games touched per
# epoch (all ~6k train games are registered regardless; the seed picks among them).
# Kept in its own dir so the legacy ATOD path (verl-agent/text) can't clobber it.
STUB_DIR=${ALFWORLD_STUB_DIR:-${DATA_ROOT}/verl-agent/alfworld_inrepo}
alfworld_train_rows=${ALFWORLD_TRAIN_ROWS:-6400}
alfworld_val_rows=${ALFWORLD_VAL_ROWS:-128}
AGENTRL_TRAIN=${ALFWORLD_TRAIN:-${STUB_DIR}/train.parquet}
AGENTRL_VAL=${ALFWORLD_VAL:-${STUB_DIR}/test.parquet}
train_files="['${AGENTRL_TRAIN}']"
val_files="['${AGENTRL_VAL}']"

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "${OPYTHON}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/prepare_text_stubs.py" \
    --local_dir "${STUB_DIR}" \
    --train_data_size "${alfworld_train_rows}" \
    --val_data_size "${alfworld_val_rows}"
fi

# ---- scale ----
# Native OPD: advantage is per-token (logπ_T − logπ_S), so there is no group to
# compare against -> n=1. (n>1 only buys something for GRPO / TB-OPD branching.)
rollout_n=${ROLLOUT_N:-1}
train_batch_size=${TRAIN_BATCH_SIZE:-64}
# Keep mini == train batch: verl scales both by rollout.n, so this yields exactly
# ONE optimizer step per rollout batch, i.e. genuinely on-policy distillation.
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-${train_batch_size}}
val_n=${VAL_N:-1}
# 2048 matches ATOD. The initial ALFWorld turn (room description + the admissible
# command list, often 50+ entries) runs ~700-1000 tokens, and the agent loop
# *silently left-truncates* anything over rollout.prompt_length, which would eat
# the instructions and room description. Only the first turn lands in the prompt;
# later observations are appended to the response.
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
# Budget for a whole episode transcript (assistant turns + env observations).
# NOT comparable to ATOD's max_response_length=512: there one row is a single
# turn (history re-injected into the prompt), here one row is a whole episode.
# Sized from measured rollouts: ~294 tokens per env step on average, p90 409, so the
# 50-step cap needs ~20k to be reachable by ~90% of episodes. At 12288, 45% of episodes
# were cut off mid-task with ~13 of their 50 steps unused and every one of them scored
# 0, i.e. the token budget rather than the policy was setting the success rate.
max_response_length=${MAX_RESPONSE_LENGTH:-20480}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
actor_lr=${ACTOR_LR:-1e-6}

# ---- alfworld env loop ----
# 50 matches ATOD (env.max_steps) and the usual ALFWorld budget; max_response_length
# above is sized so an episode can actually reach it rather than dying on tokens.
alfworld_max_steps=${ALFWORLD_MAX_STEPS:-50}
# Needs >= ceil(train_batch_size * rollout_n / rollout.agent.num_workers).
alfworld_pool_size=${ALFWORLD_POOL_SIZE:-16}
# ATOD parity. ALFWorld actions are short ("go to shelf 1"), but the cap must leave
# room for the whole turn: truncating before the closing </action> makes the action
# unparseable and silently falls back to a garbage tail slice.
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
# On-policy means sampling from the student's own distribution: keep temp at 1.0.
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
# token-mean matches the plain-OPD (B1) recipe; length-normalizing per sequence
# would under-weight the long episodes we care about.
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}

# ---- cluster split (student trainer node0, teacher node1) ----
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}
DISTILL_NNODES=${DISTILL_NNODES:-1}
DISTILL_NGPUS_PER_NODE=${DISTILL_NGPUS_PER_NODE:-8}

# ---- trainer bookkeeping ----
project_name=${PROJECT_NAME:-verl_agent_alfworld}
experiment_name=${EXPERIMENT_NAME:-alfworld_inrepo_opd_qwen3_4b_from_30ba3b}
trainer_logger=${TRAINER_LOGGER:-"['console','swanlab']"}
# steps/epoch = alfworld_train_rows / train_batch_size (drop_last). With the
# defaults above that is 6400/64 = 100 steps, so a couple of epochs is plenty.
total_epochs=${TOTAL_EPOCHS:-2}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-25}
val_before_train=${VAL_BEFORE_TRAIN:-True}
ckpt_dir=${CKPT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}/ckpt}
rollout_dir=${ROLLOUT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}/rollout}
result_dir=${RESULT_DIR:-${DATA_ROOT}/../iclr/logs/${experiment_name}}
mkdir -p "${ckpt_dir}" "${rollout_dir}" "${result_dir}"

# sync_refill_failed_groups: by default a group whose episode raised keeps its slot in
# the batch and materializes nothing, so the trainer pads the shortfall with synthetic
# rows -- silently, and the run still reports a full batch. Refill instead, so an
# episode that dies costs a retry rather than a real training sample.
# adv_estimator=grpo is required only because ppo_loss() is still invoked when
# use_policy_gradient=True; with use_task_rewards=False that branch is zeroed and
# the OPD gradient comes solely from advantages = -distillation_loss. At n=1 GRPO
# special-cases a singleton group (mean=0, std=1), so no NaN.
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
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${rollout_temperature} \
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
    +alfworld.eval_split=${ALFWORLD_EVAL_SPLIT} \
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
    trainer.v1.sampler.sync_refill_failed_groups=True \
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
