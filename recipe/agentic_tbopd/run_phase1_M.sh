#!/usr/bin/env bash
# Phase 1' arm M: TB-OPD-Turn (this paper).
#   Only-fail + B=1 + k=2, fork at the highest-uncertainty post-tool turn,
#   forced-topk breakpoint resume, teacher dense KD over the whole turn tree
#   (tool tokens masked out).
# Teacher SOD-GRPO_teacher-4B -> Student Qwen3-1.7B on Open-AgentRL TIR.
#
# Selection signal: ENTROPY-ONLY (ΔH_post-tool). Teacher-at-rollout disagreement
# is not used for now (scoring the main trajectory's turns with the teacher at
# rollout time is not yet available), so we run the pure ARPO-style entropy
# signal rather than the hybrid Soft-OR.
#
# Prereqs:
#   1) python -m recipe.agentic_tbopd.prepare_open_agentrl   # build TIR parquet
#   2) sandbox backend: E2B_API_KEY=... (default) or SANDBOX_BACKEND=sandbox_fusion + SANDBOX_FUSION_URL
set -xeuo pipefail
export TB_ENABLE=True
export TB_FORK_UNIT=turn
export TB_K=${TB_K:-2}
export TB_ONLY_FAIL=${TB_ONLY_FAIL:-True}
# Entropy-only post-tool signal (no disagreement). Use dHtool; 'ent' also valid.
export TB_FORK_METRIC=${TB_FORK_METRIC:-dHtool}
export TB_BRANCH_MODE=${TB_BRANCH_MODE:-forced_topk}
export TB_TURN_ONLY_POST_TOOL=${TB_TURN_ONLY_POST_TOOL:-True}
export TB_MAX_BRANCHES_PER_TRAJ=${TB_MAX_BRANCHES_PER_TRAJ:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agentic_phase1_M}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_agentic_tbopd.sh" "$@"
