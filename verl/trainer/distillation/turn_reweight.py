# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turn-level loss reweighting for agentic OPD (B-A1, ATOD T-DUR style).

This is the *reweight-only* counterpart to the turn-*expansion* method (M):
instead of forking new sub-trajectories at a high-uncertainty assistant turn, we
keep the single trajectory and scale the per-token KD loss up on the uncertain
turns. It is implemented entirely at loss time and shares nothing with the
rollout branch path, so enabling it cannot change method M's behaviour.

Turn boundaries are recovered directly from ``response_mask`` (a turn = a maximal
run of consecutive assistant tokens; tool/observation tokens have mask 0 and
separate turns). The per-turn uncertainty signal is the student NLL proxy
``mean(-logprob)`` over the turn's leading tokens -- the same entropy proxy used
by ``tb_opd.select_fork_turn`` -- so no teacher logprobs are required.
"""

from __future__ import annotations

import torch


def _segment_bounds(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for each run of True in ``mask_row``."""
    idx = torch.nonzero(mask_row, as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    starts = [int(idx[0].item())]
    ends: list[int] = []
    prev = int(idx[0].item())
    for p in idx[1:].tolist():
        if p != prev + 1:
            ends.append(prev + 1)
            starts.append(p)
        prev = p
    ends.append(prev + 1)
    return list(zip(starts, ends))


def compute_turn_reweight(
    response_mask: torch.Tensor,
    logprobs: torch.Tensor,
    *,
    alpha: float = 1.0,
    metric: str = "ent",
    turn_first_k: int = 16,
    only_post_tool: bool = True,
    skip_first: int = 1,
) -> torch.Tensor:
    """Per-token loss multiplier that emphasizes high-uncertainty assistant turns.

    Args:
        response_mask: (B, T) bool/0-1 tensor, 1 on assistant tokens.
        logprobs: (B, T) student log-probs (e.g. rollout ``old_log_probs``).
        alpha: emphasis strength; weight = 1 + alpha * normalized_signal.
        metric: ``ent`` (mean -logp over the turn) or ``dHtool`` (that minus the
            first turn's entropy, i.e. the post-tool entropy *increase*).
        turn_first_k: leading tokens of a turn used for the signal (<=0 = whole turn).
        only_post_tool: if True, the opening turn (index 0) is never emphasized.
        skip_first: number of leading turns excluded from emphasis.

    Returns:
        (B, T) float tensor of per-token weights. Non-response tokens and
        non-eligible turns get weight 1.0, so the mean weight stays ~1.
    """
    mask_bool = response_mask.bool()
    weights = torch.ones_like(logprobs, dtype=torch.float32)
    surprisal = (-logprobs).to(torch.float32)
    thr = max(int(skip_first), 1 if only_post_tool else 0)

    bsz = mask_bool.shape[0]
    for b in range(bsz):
        segs = _segment_bounds(mask_bool[b])
        if len(segs) < 2:
            continue  # single turn: nothing to reweight relative to.

        turn_ent: list[float] = []
        for s, e in segs:
            if turn_first_k and turn_first_k > 0:
                e_sig = min(e, s + turn_first_k)
            else:
                e_sig = e
            seg = surprisal[b, s:e_sig]
            turn_ent.append(float(seg.mean().item()) if seg.numel() > 0 else 0.0)

        base = turn_ent[0]
        signals: list[float] = []
        for j, ent in enumerate(turn_ent):
            signals.append(ent - base if metric == "dHtool" else ent)

        eligible = [j for j in range(len(segs)) if j >= thr]
        if not eligible:
            continue
        elig_sig = [signals[j] for j in eligible]
        lo, hi = min(elig_sig), max(elig_sig)
        span = hi - lo
        for j in eligible:
            norm = (signals[j] - lo) / span if span > 1e-8 else 0.0
            w = 1.0 + alpha * norm
            s, e = segs[j]
            weights[b, s:e] = w

    return weights
