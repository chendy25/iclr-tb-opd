#!/usr/bin/env bash
# ALFWorld TB-OPD-Turn arm.
#
# Same script, same data, same teacher and same OPD loss as the native-OPD baseline
# (train_inrepo_opd.sh with TB_ENABLE unset) -- the ONLY difference is that a failed
# episode spends k extra rollout slots on branches resumed from its most uncertain
# assistant turn. That is what makes the baseline a controlled comparison.
#
# Default scoring axes are the math token arm's, so "math vs agentic" differs only in
# the candidate set (assistant turns / spans instead of individual tokens):
#   truncated top-k entropy, rank-normalized, fork_alpha=1.0 (pure uncertainty),
#   blend fusion, no positional prior, B=1.
#
# Variants (each changes exactly one axis):
#   ARPO-style       TB_FORK_METRIC=dHtool TB_TURN_ONLY_POST_TOOL=True
#   KL-fork          TB_FORK_ALPHA=0.5                    # blend in teacher disagreement
#   ATOD T-DUR       TB_FORK_FUSE=soft_or TB_FORK_NORMALIZE=minmax TB_FORK_METRIC=dHtool
#   branch the think TB_FORK_ELIGIBILITY=reasoning
#   branch the act   TB_FORK_ELIGIBILITY=action
#   wider budget     TB_MAX_BRANCHES_PER_TRAJ=2 TB_K=4
#   skip confident   TB_FORK_MIN_ENTROPY=0.5              # the math arms' floor
#
# Cost: branch slots are played sequentially after the main slot, so a step costs
# roughly (1+k) episodes of wall clock. The env pool does NOT need to grow with k.
#
# Prereq: the thinking-protocol baseline must exist for comparison --
#   bash recipe/alfworld_opd/train_inrepo_opd.sh
set -xeuo pipefail

export TB_ENABLE=True
export TB_K=${TB_K:-2}
export TB_ONLY_FAIL=${TB_ONLY_FAIL:-True}
export TB_FORK_METRIC=${TB_FORK_METRIC:-entropy}
export TB_FORK_ALPHA=${TB_FORK_ALPHA:-1.0}
export TB_FORK_FUSE=${TB_FORK_FUSE:-blend}
export TB_FORK_NORMALIZE=${TB_FORK_NORMALIZE:-rank}
export TB_BRANCH_MODE=${TB_BRANCH_MODE:-forced_topk}
export TB_TURN_ONLY_POST_TOOL=${TB_TURN_ONLY_POST_TOOL:-False}
export TB_MAX_BRANCHES_PER_TRAJ=${TB_MAX_BRANCHES_PER_TRAJ:-1}
export TB_RECORD_ACTION_SPANS=${TB_RECORD_ACTION_SPANS:-True}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_inrepo_opd.sh" "$@"
