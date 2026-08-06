#!/usr/bin/env bash
# =====================================================================
# Agentic TB-OPD (Phase 1') inner training script.
#   Multi-turn tool-use (code_interpreter / SandboxFusion) On-Policy
#   Distillation, with an optional turn-level fork (TB-OPD-Turn).
#
# Arms (select via env, see run_phase1_*.sh presets):
#   B-A0 : agent OPD baseline        -> TB_ENABLE=False
#   M    : TB-OPD-Turn (this paper)  -> TB_ENABLE=True TB_FORK_UNIT=turn
#
# Teacher SOD-GRPO_teacher-4B -> Student Qwen3-1.7B (defaults), TIR math.
# Requires a running SandboxFusion service; set SANDBOX_FUSION_URL.
#
# Usage (on allocated nodes, RANK/MASTER_* set by cluster):
#   SANDBOX_FUSION_URL=http://host:8080/run_code bash recipe/agentic_tbopd/train_agentic_tbopd.sh
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

# ---- models (local, SOD stack) ----
STUDENT_MODEL=${STUDENT_MODEL:-${DATA_ROOT}/models/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-${DATA_ROOT}/models/SOD-GRPO_teacher-4B}

# ---- data (Open-AgentRL, TIR tool-agent parquet from prepare_open_agentrl.py) ----
TIR_ROOT=${TIR_ROOT:-${DATA_ROOT}/preprocessed/open_agentrl_tir}
AGENTRL_TRAIN=${AGENTRL_TRAIN:-${TIR_ROOT}/train.parquet}
AGENTRL_VAL=${AGENTRL_VAL:-${TIR_ROOT}/test_aime2024.parquet}
train_files="['${AGENTRL_TRAIN}']"
val_files="['${AGENTRL_VAL}']"

# ---- tool / multi-turn ----
TOOL_CONFIG=${TOOL_CONFIG:-${CODE_DIR}/recipe/agentic_tbopd/config/sandbox_fusion_tool_config.yaml}
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://localhost:8080/run_code}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-8}
max_user_turns=${MAX_USER_TURNS:-8}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}
multiturn_format=${MULTITURN_FORMAT:-hermes}

# ---- OPD hyperparams ----
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# ---- TB-OPD-Turn hyperparams ----
tb_enable=${TB_ENABLE:-True}
tb_fork_unit=${TB_FORK_UNIT:-turn}
tb_k=${TB_K:-2}
tb_only_fail=${TB_ONLY_FAIL:-True}
tb_fork_metric=${TB_FORK_METRIC:-hybrid}          # ent | dHtool | disagree | hybrid
tb_correct_threshold=${TB_CORRECT_THRESHOLD:-1.0}
tb_branch_mode=${TB_BRANCH_MODE:-forced_topk}     # forced_topk (M) | resample
tb_resample_temperature=${TB_RESAMPLE_TEMPERATURE:--1.0}
tb_topk_logprobs=${TB_TOPK_LOGPROBS:-20}
tb_turn_first_k=${TB_TURN_FIRST_K:-16}
tb_turn_only_post_tool=${TB_TURN_ONLY_POST_TOOL:-True}
tb_turn_skip_first=${TB_TURN_SKIP_FIRST:-1}
tb_max_branches=${TB_MAX_BRANCHES_PER_TRAJ:-1}
tb_min_fork_signal=${TB_MIN_FORK_SIGNAL:-0.0}
tb_consec_penalty=${TB_CONSEC_PENALTY:-False}
tb_consec_weight=${TB_CONSEC_WEIGHT:-0.5}

# rollout.n: TB-OPD-Turn uses 1 (main) + k (branches); baseline uses GRPO group size.
if [[ "${tb_enable}" == "True" ]]; then
  rollout_n=${ROLLOUT_N:-$((1 + tb_k))}
else
  rollout_n=${ROLLOUT_N:-5}
fi

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.45}

# ---- resources ----
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
DISTILL_NGPUS_PER_NODE=${DISTILL_NGPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}
DISTILL_NNODES=${DISTILL_NNODES:-1}
NNODES=${NNODES:-$((TRAINER_NNODES + DISTILL_NNODES))}
nnodes=${NNODES}; export nnodes

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-25}
total_training_steps=${TOTAL_TRAINING_STEPS:-}
val_n=${VAL_N:-1}
val_before_train=${VAL_BEFORE_TRAIN:-False}
trainer_logger=${TRAINER_LOGGER:-'["console","swanlab"]'}

project_name=${PROJECT_NAME:-agentic_tb_opd}
experiment_name=${EXPERIMENT_NAME:-agentic_tbopd_phase1}

result_dir=${RESULT_DIR:-${CODE_DIR}/output/${project_name}/${experiment_name}}
ckpt_dir=${result_dir}/checkpoints
rollout_dir=${result_dir}/samples
mkdir -p "${ckpt_dir}" "${rollout_dir}"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
need_gpus=$(( TRAINER_NNODES * NGPUS_PER_NODE + DISTILL_NNODES * DISTILL_NGPUS_PER_NODE ))

echo "AGENTIC TB-OPD: tb_enable=${tb_enable} fork_unit=${tb_fork_unit} metric=${tb_fork_metric} branch_mode=${tb_branch_mode} rollout_n=${rollout_n}"
echo "tool_config=${TOOL_CONFIG} sandbox_url=${SANDBOX_FUSION_URL} max_turns=${max_assistant_turns}"
echo "student=${STUDENT_MODEL} teacher=${TEACHER_MODEL}"
echo "train=${train_files} val=${val_files}"

# ---- sanity ----
[[ -d "${STUDENT_MODEL}" ]] || { echo "Missing student: ${STUDENT_MODEL}"; exit 1; }
[[ -d "${TEACHER_MODEL}" ]] || { echo "Missing teacher: ${TEACHER_MODEL}"; exit 1; }
[[ -f "${AGENTRL_TRAIN}" ]] || { echo "Missing train parquet: ${AGENTRL_TRAIN} (run prepare_open_agentrl.py)"; exit 1; }
[[ -f "${TOOL_CONFIG}" ]] || { echo "Missing tool config: ${TOOL_CONFIG}"; exit 1; }
[[ -x "${OPYTHON}" ]] || { echo "Missing python: ${OPYTHON}"; exit 1; }

# ---- cleanup + Ray ----
ray stop --force 2>/dev/null || true
pkill -9 -f 'vllm' 2>/dev/null || true
pkill -9 -f 'main_ppo' 2>/dev/null || true
sleep 3

if [[ "${nnodes}" -gt 1 ]]; then
  if [[ "${RANK}" = "0" ]]; then
    ray start --head --port=6379 --disable-usage-stats
    sleep 5
    echo "Waiting for Ray cluster GPUs >= ${need_gpus} (timeout 300s)..."
    for _ in $(seq 1 60); do
      avail=$("${OPYTHON}" - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
PY
)
      echo "  ray GPUs available=${avail} need=${need_gpus}"
      [[ "${avail}" -ge "${need_gpus}" ]] && break
      sleep 5
    done
    ray status
  else
    ray start --address "${MASTER_ADDR}:6379" --block &
    wait
    exit 0
  fi
else
  ray start --head --port=6379 --disable-usage-stats
  sleep 5
fi

########################### launch ###########################
TRAIN_STEP_ARG=()
if [[ -n "${total_training_steps}" ]]; then
  TRAIN_STEP_ARG=(trainer.total_training_steps="${total_training_steps}")
fi

# TB-OPD block only when enabled (B-A0 leaves standard agent OPD untouched).
TB_ARGS=()
if [[ "${tb_enable}" == "True" ]]; then
  TB_ARGS=(
    distillation.tb_opd.enable=True
    distillation.tb_opd.fork_unit=${tb_fork_unit}
    distillation.tb_opd.k=${tb_k}
    distillation.tb_opd.only_fail=${tb_only_fail}
    distillation.tb_opd.fork_metric=${tb_fork_metric}
    distillation.tb_opd.correct_threshold=${tb_correct_threshold}
    distillation.tb_opd.branch_mode=${tb_branch_mode}
    distillation.tb_opd.resample_temperature=${tb_resample_temperature}
    distillation.tb_opd.topk_logprobs=${tb_topk_logprobs}
    distillation.tb_opd.turn_first_k=${tb_turn_first_k}
    distillation.tb_opd.turn_only_post_tool=${tb_turn_only_post_tool}
    distillation.tb_opd.turn_skip_first=${tb_turn_skip_first}
    distillation.tb_opd.max_branches_per_traj=${tb_max_branches}
    distillation.tb_opd.min_fork_signal=${tb_min_fork_signal}
    distillation.tb_opd.consecutive_high_entropy_penalty=${tb_consec_penalty}
    distillation.tb_opd.consecutive_penalty_weight=${tb_consec_weight}
  )
else
  TB_ARGS=(distillation.tb_opd.enable=False)
fi

${OPYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=${multiturn_format} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_user_turns} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
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
    "${TB_ARGS[@]}" \
    "${TRAIN_STEP_ARG[@]}" \
    "$@" \
    2>&1 | tee "${result_dir}/train-$(date +%Y%m%d_%H%M%S).log"

ray stop --force 2>/dev/null || true
