#!/usr/bin/env bash
# Phase 1' arm B-A0: agent OPD baseline (single trajectory, full-token KD, no fork).
# Teacher SOD-GRPO_teacher-4B -> Student Qwen3-1.7B on Open-AgentRL TIR (code_interpreter).
#
# Prereqs:
#   1) python -m recipe.agentic_tbopd.prepare_open_agentrl   # build TIR parquet
#   2) a running SandboxFusion service -> export SANDBOX_FUSION_URL=http://host:8080/run_code
set -xeuo pipefail
export TB_ENABLE=False
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agentic_phase1_B-A0}
export ROLLOUT_N=${ROLLOUT_N:-5}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_agentic_tbopd.sh" "$@"
