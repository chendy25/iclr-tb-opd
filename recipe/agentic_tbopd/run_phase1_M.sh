#!/usr/bin/env bash
# Phase 1' arm M: TB-OPD-Turn (this paper).
#   Only-fail + B=1 + k=2, fork at the highest-uncertainty post-tool turn,
#   forced-topk breakpoint resume, teacher dense KD over the whole turn tree
#   (tool tokens masked out).
# Teacher SOD-GRPO_teacher-4B -> Student Qwen3-1.7B on Open-AgentRL TIR.
#
# Selection signal: the same one the math token arms use
# (iclr_opd_tbopd_rbw_klfork_r16k_e2) -- truncated entropy blended with teacher
# disagreement at fork_alpha=0.5, rank-normalized, no positional prior. The teacher
# IS available at rollout time for fork ranking, so the disagreement term costs
# nothing extra: that forward is needed for the loss regardless.
#
# Variants (each changes one axis):
#   ARPO-style   : TB_FORK_METRIC=dHtool TB_TURN_ONLY_POST_TOOL=True
#   pure entropy : TB_FORK_ALPHA=1.0                  # drop the teacher term
#   ATOD T-DUR   : TB_FORK_FUSE=soft_or TB_FORK_NORMALIZE=minmax TB_FORK_METRIC=dHtool
#   fork inside the reasoning / action span: TB_FORK_ELIGIBILITY=reasoning|action
#
# Prereqs:
#   1) python -m recipe.agentic_tbopd.prepare_open_agentrl   # build TIR parquet
#   2) sandbox backend: E2B_API_KEY=... (default) or SANDBOX_BACKEND=sandbox_fusion + SANDBOX_FUSION_URL
set -xeuo pipefail
export TB_ENABLE=True
export TB_FORK_UNIT=turn
export TB_K=${TB_K:-2}
export TB_ONLY_FAIL=${TB_ONLY_FAIL:-False}
export TB_FORK_METRIC=${TB_FORK_METRIC:-entropy}
export TB_FORK_ALPHA=${TB_FORK_ALPHA:-0.5}
export TB_FORK_FUSE=${TB_FORK_FUSE:-blend}
export TB_FORK_NORMALIZE=${TB_FORK_NORMALIZE:-rank}
export TB_BRANCH_MODE=${TB_BRANCH_MODE:-forced_topk}
export TB_TURN_ONLY_POST_TOOL=${TB_TURN_ONLY_POST_TOOL:-False}
export TB_MAX_BRANCHES_PER_TRAJ=${TB_MAX_BRANCHES_PER_TRAJ:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agentic_phase1_M}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_agentic_tbopd.sh" "$@"
