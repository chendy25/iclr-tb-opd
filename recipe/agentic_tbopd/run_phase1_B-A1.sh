#!/usr/bin/env bash
# Phase 1' arm B-A1: turn-reweight OPD (ATOD T-DUR style; reweight-only).
#   No branch expansion (rollout identical to B-A0). At loss time the KD token
#   loss is scaled up on high-uncertainty *post-tool* assistant turns, using the
#   entropy proxy mean(-logp) from the rollout logprobs. This is the "只重加权"
#   half of the "展开 vs 只重加权" ablation against method M.
# Teacher SOD-GRPO_teacher-4B -> Student Qwen3-1.7B on Open-AgentRL TIR.
#
# Entropy-only: no teacher-at-rollout disagreement is used or needed.
#
# Prereqs:
#   1) python -m recipe.agentic_tbopd.prepare_open_agentrl   # build TIR parquet
#   2) sandbox backend: E2B_API_KEY=... (default) or SANDBOX_BACKEND=sandbox_fusion + SANDBOX_FUSION_URL
set -xeuo pipefail
export TB_ENABLE=False                 # no rollout fork (B-A0 rollout)
export TB_TURN_REWEIGHT=True           # loss-side turn reweighting
export TB_REWEIGHT_METRIC=${TB_REWEIGHT_METRIC:-ent}    # ent | dHtool
export TB_REWEIGHT_ALPHA=${TB_REWEIGHT_ALPHA:-1.0}
export TB_TURN_ONLY_POST_TOOL=${TB_TURN_ONLY_POST_TOOL:-True}
export TB_TURN_SKIP_FIRST=${TB_TURN_SKIP_FIRST:-1}
export TB_TURN_FIRST_K=${TB_TURN_FIRST_K:-16}
export ROLLOUT_N=${ROLLOUT_N:-5}       # same GRPO group size as B-A0
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agentic_phase1_B-A1}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_agentic_tbopd.sh" "$@"
