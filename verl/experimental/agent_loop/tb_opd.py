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
"""Pure helpers for token-level branching on-policy distillation (TB-OPD).

Kept free of any ``agent_loop`` imports to avoid a circular dependency; the
worker (``agent_loop.AgentLoopWorker``) owns orchestration and the
``AgentLoopOutput`` construction, while this module provides the stateless
fork-selection and scoring primitives.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _truncated_entropy(logps: list[float]) -> float:
    """Entropy (nats) of the renormalized top-k distribution given top-k logprobs."""
    probs = [math.exp(lp) for lp in logps]
    z = sum(probs)
    if z <= 0.0:
        return 0.0
    h = 0.0
    for p in probs:
        q = p / z
        if q > 0.0:
            h -= q * math.log(q)
    return h


def fork_uncertainty(logps: list[float], metric: str) -> float:
    """Higher return value == more uncertain (better fork candidate).

    Args:
        logps: top-k logprobs at a position, descending by rank (rank-1 first).
        metric: "entropy" or "topk_gap".
    """
    if metric == "topk_gap":
        # Small top1-top2 margin => high uncertainty => larger score.
        return -(logps[0] - logps[1])
    return _truncated_entropy(logps)


def disagreement_window(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    window: int,
) -> list[float]:
    """Forward-looking teacher/student disagreement per response position.

    ``D(t) = log pi_theta(a_t) - log pi_T(a_t)`` is the per-token surprise the OPD
    loss already uses as the (negated) reward. A good fork point is one where the
    *upcoming* span is where the student and teacher part ways, so each position is
    scored by the mean ``D`` over the next ``window`` tokens rather than by ``D(t)``
    alone -- a single token's disagreement is far too noisy to rank on.

    Positive values mean the student is over-confident relative to the teacher,
    which is exactly the region worth spending branches on.
    """
    n = min(len(student_logprobs), len(teacher_logprobs))
    if n == 0:
        return []
    d = [student_logprobs[i] - teacher_logprobs[i] for i in range(n)]
    # Suffix sums so each window mean is O(1).
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + d[i]
    w = max(1, int(window))
    out = [0.0] * n
    for i in range(n):
        j = min(n, i + w)
        out[i] = (suffix[i] - suffix[j]) / (j - i)
    return out


def _normalized_ranks(values: list[float]) -> list[float]:
    """Map values to their rank in ``[0, 1]`` (1.0 == largest). Ties broken by index."""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for r, i in enumerate(order):
        ranks[i] = r / (n - 1)
    return ranks


def branch_weights(
    logprobs: list[Optional[float]],
    *,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> Optional[list[float]]:
    """Rao-Blackwellized weights over the ``k+1`` realized continuations of a fork.

    ``logprobs[j]`` is ``log pi_theta(a_j | s_fork)`` for the token slot ``j`` actually
    took at the fork: slot 0 is the main rollout's own sampled token, slots ``1..k``
    are the forced top-k alternatives.

    Forcing the top-k alternatives and then averaging them uniformly is what makes
    branch-OPD off-policy: the group's gradient estimates
    ``(1/(k+1)) * sum_j f(a_j)`` when the on-policy quantity is
    ``E_{a ~ pi_theta}[f(a)]``. Reweighting by the renormalized student probability
    ``pi_theta(a_j) / sum_i pi_theta(a_i)`` turns the sum back into a conditional
    expectation over the sampled support -- the standard Rao-Blackwell estimator,
    which is unbiased for that support and strictly lower-variance than sampling one
    continuation. Concretely it stops a rank-5 token the student would emit ~1% of the
    time from carrying the same gradient mass as the token it actually chose, which is
    what drives entropy up and the response length with it.

    Weights are rescaled to sum to ``len(logprobs)`` (i.e. mean 1) so the group's total
    contribution -- and therefore the effective learning rate -- matches the uniform
    weighting it replaces. Only the *relative* weighting inside the group changes.

    Args:
        logprobs: per-slot fork-token logprob; ``None`` anywhere disables weighting.
        temperature: softens the weights (``1.0`` = exact RB, larger -> uniform).
        floor: minimum weight per slot before renormalization; keeps a very unlikely
            branch from contributing literally nothing after we paid to generate it.

    Returns:
        Weights summing to ``len(logprobs)``, or ``None`` if they cannot be computed.
    """
    if not logprobs or any(lp is None for lp in logprobs):
        return None
    n = len(logprobs)
    t = max(1e-6, float(temperature))
    scaled = [float(lp) / t for lp in logprobs]
    m = max(scaled)
    probs = [math.exp(s - m) for s in scaled]
    z = sum(probs)
    if not math.isfinite(z) or z <= 0.0:
        return None
    w = [p / z for p in probs]
    f = max(0.0, float(floor))
    if f > 0.0:
        w = [max(x, f) for x in w]
        z2 = sum(w)
        w = [x / z2 for x in w]
    return [x * n for x in w]


def slot_fork_index(slot: int, k: int, n_forks: int) -> tuple[int, int]:
    """Map a rollout slot to ``(fork_index, candidate_index)``.

    Slot 0 is the main trajectory; slot ``1 + b*k + j`` is the j-th forced alternative
    at fork ``b``, so ``rollout.n`` should be ``1 + B*k``. The modulo keeps every slot
    filled when it is not: the minimum-gap filter can return fewer forks than requested,
    and leaving a slot unassigned would break the fixed ``rollout.n`` row contract.
    """
    i = int(slot) - 1
    k = max(1, int(k))
    n_forks = max(1, int(n_forks))
    return (i // k) % n_forks, i % k


def mask_shared_prefix(
    response_mask: list[int],
    eos_sft_mask: Optional[list[float]],
    prefix_len: int,
) -> tuple[list[int], Optional[list[float]]]:
    """Drop a branch's replay of the main trajectory from the loss.

    A branch is generated as ``main[:fork] + forced_token + continuation``, so the
    first ``prefix_len`` tokens are byte-identical to the main trajectory. Left
    unmasked they are trained once per slot, which up-weights early tokens by the
    number of slots and (via ``eos_sft_mask``) can duplicate the learn-EOS
    supervision at the very same index. Zeroing both columns leaves the prefix
    supervised exactly once, by the main slot.
    """
    if prefix_len <= 0:
        return list(response_mask), (list(eos_sft_mask) if eos_sft_mask is not None else None)
    n = min(int(prefix_len), len(response_mask))
    masked = [0] * n + list(response_mask[n:])
    if eos_sft_mask is None:
        return masked, None
    m = min(int(prefix_len), len(eos_sft_mask))
    return masked, [0.0] * m + list(eos_sft_mask[m:])


def normalize_ground_truth(reward_model: Any) -> Optional[str]:
    if reward_model is None:
        return None
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
    else:
        gt = getattr(reward_model, "ground_truth", None)
    return gt


def score_solution(tokenizer, response_ids: list[int], kwargs: dict, threshold: float) -> tuple[float, bool]:
    """Rule-based score of a decoded response. Returns (score, is_correct).

    Failures in the reward function are treated as score 0.0 (i.e. "fail"), so
    that branching still fires -- this keeps TB-OPD robust to reward edge cases.
    """
    from verl.utils.reward_score import default_compute_score

    gt = normalize_ground_truth(kwargs.get("reward_model"))
    if gt is None:
        return 0.0, False
    data_source = kwargs.get("data_source")
    if hasattr(data_source, "item"):
        data_source = data_source.item()
    extra_info = kwargs.get("extra_info")
    solution_str = tokenizer.decode(response_ids, skip_special_tokens=True)
    try:
        res = default_compute_score(data_source, solution_str, gt, extra_info)
    except Exception as e:  # noqa: BLE001 - reward fns can raise on odd outputs
        logger.warning(f"[TB-OPD] reward scoring failed ({e}); treating as fail.")
        return 0.0, False
    if isinstance(res, dict):
        score = float(res.get("score", 0.0))
    else:
        score = float(res)
    return score, score >= threshold


# Single-character math tokens are genuine arithmetic fork points and must never
# be dropped by a length filter. Digits, operators, comparators, brackets, and the
# decimal point all qualify.
_MATH_KEEP_CHARS = frozenset("0123456789+-*/=<>%^()[]{}.√≤≥≠±×÷·")


def _passes_token_filter(
    tokenizer,
    token_id: int,
    special_ids: set,
    min_strip_len: int,
    mode: str = "math_aware",
) -> bool:
    """Decide whether a response position is eligible as a fork point.

    Modes:
      - ``"math_aware"`` (default, recommended for OPD): drop only structurally
        meaningless tokens -- special/control tokens, pure whitespace, and pure
        non-math punctuation (``,`` ``"`` ``—`` ...). Crucially it NEVER drops a
        math-bearing single char (``5`` / ``+`` / ``=``) or a reflection word
        (``Wait``); those are real forking tokens (Beyond-80/20). The
        entropy / top-k-gap metric does the actual selecting.
      - ``"strip_len"`` (legacy CURE-style): keep only tokens whose decoded text,
        stripped of whitespace, is strictly longer than ``min_strip_len``. WARNING:
        with ``min_strip_len=1`` this silently discards every single-char math
        token -- exactly the arithmetic fork points we care about.
    """
    if int(token_id) in special_ids:
        return False
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - decode can fail on odd ids
        return False
    stripped = text.strip()
    if not stripped:
        return False  # pure whitespace / control
    if mode == "math_aware":
        if any(c in _MATH_KEEP_CHARS for c in stripped):
            return True  # keep digits/operators even when single-char
        # Drop tokens that are entirely punctuation/symbols with no alphanumerics
        # (e.g. ',', a lone quote, an em-dash): high-entropy but not real forks.
        return any(c.isalnum() for c in stripped)
    return len(stripped) > min_strip_len


def _collect_fork_candidates(
    response_ids: list[int],
    row_getter,
    *,
    resp_len: int,
    min_tokens: int,
    response_length: int,
    skip_first: int,
    tokenizer,
    special_ids: set,
    min_token_strip_len: int,
    filter_mode: str,
    metric: str,
    disagreement: Optional[list[float]] = None,
    eligible_mask: Optional[list[int]] = None,
) -> tuple[list, int, int]:
    """Score every eligible response position as a fork candidate.

    ``row_getter(p)`` returns ``(logprobs, token_ids)`` -- the top-k logprobs and
    token ids (descending by rank) of the decoding distribution at response
    position ``p`` -- or ``None`` if unavailable. Shared by the second-forward
    (``select_fork``) and Scheme-B (``select_fork_from_topk``) paths so filtering
    and scoring stay identical; only the source of the per-position top-k differs.

    ``eligible_mask`` is the main trajectory's ``response_mask``: positions it zeroes
    are already out of the loss (post-answer span under ``mask_after_answer`` /
    learn-EOS), and forking there yields a branch whose whole continuation is masked
    away while its injected EOS collides with the main trajectory's.

    Each candidate is ``(uncertainty, position, token_ids, logprobs, disagreement)``;
    the last entry is the teacher/student disagreement at that position (0.0 when no
    teacher signal was supplied) and is consumed by ``_finalize_fork``.

    Returns ``(candidates, n_considered, n_mask_skipped)``.
    """
    cands: list[tuple[float, int, list[int], list[float], float]] = []
    n_considered = 0
    n_mask_skipped = 0
    for p in range(resp_len):
        # Skip opener positions (CURE skips position 0).
        if p < skip_first:
            continue
        # Need enough remaining budget to make a branch worthwhile.
        if response_length - p < min_tokens:
            break
        if eligible_mask is not None and (p >= len(eligible_mask) or not eligible_mask[p]):
            n_mask_skipped += 1
            continue
        row = row_getter(p)
        if row is None:
            continue
        row_lp = [x for x in row[0] if x is not None]
        row_id = [x for x in row[1] if x is not None]
        if len(row_lp) < 2 or len(row_id) < 2:
            continue
        if tokenizer is not None and not _passes_token_filter(
            tokenizer, response_ids[p], special_ids, min_token_strip_len, filter_mode
        ):
            continue
        n_considered += 1
        dis = float(disagreement[p]) if disagreement is not None and p < len(disagreement) else 0.0
        cands.append((fork_uncertainty(row_lp, metric), p, row_id, row_lp, dis))
    return cands, n_considered, n_mask_skipped


def _greedy_pick(cands: list, seq: list[int], num_forks: int, min_gap: int) -> list[int]:
    """Take up to ``num_forks`` candidate indices from ``seq``, at least ``min_gap`` apart.

    ``seq`` is walked in order, so the caller decides the policy: ranked by fused score
    for ``argmax``, shuffled for ``topk_uniform``. The gap check is what makes ``B > 1``
    worth the rollouts -- the top-ranked positions are usually neighbours, and two forks
    a few tokens apart share almost their entire prefix, so their branches come out as
    near-duplicates of each other.
    """
    picked: list[int] = []
    for i in seq:
        p = cands[i][1]
        if all(abs(p - cands[j][1]) >= min_gap for j in picked):
            picked.append(i)
            if len(picked) >= num_forks:
                break
    return picked


def _fork_entry(
    cands: list,
    pick: int,
    response_ids: list[int],
    *,
    dedup_main: bool,
    sampled_logprobs: Optional[list[float]],
    fused: list[float],
    set_h: Optional[set],
    set_d: Optional[set],
) -> dict:
    """Build the per-fork payload: position, forced candidates, and their logprobs."""
    best_score, best_pos, best_cands, best_lps, best_dis = cands[pick]

    # Exclude the main-sampled token so each branch is a genuine alternative.
    main_tok = int(response_ids[best_pos])
    lp_by_id = {int(t): float(lp) for t, lp in zip(best_cands, best_lps, strict=False)}
    main_logprob = lp_by_id.get(main_tok)
    if main_logprob is None and sampled_logprobs is not None and best_pos < len(sampled_logprobs):
        # Main token fell outside the top-k row; fall back to its rollout logprob.
        main_logprob = float(sampled_logprobs[best_pos])
    if dedup_main:
        deduped = [(t, lp) for t, lp in zip(best_cands, best_lps, strict=False) if int(t) != main_tok]
        if deduped:
            best_cands = [t for t, _ in deduped]
            best_lps = [lp for _, lp in deduped]

    if set_h is None or set_d is None:
        source = "entropy"
    else:
        in_h, in_d = pick in set_h, pick in set_d
        source = "both" if in_h and in_d else ("disagreement" if in_d else "entropy")

    return {
        "pos": best_pos,
        "cand_token_ids": best_cands,
        "cand_logprobs": [float(x) for x in best_lps],
        "main_logprob": main_logprob,
        "score": float(best_score),
        "fused_score": float(fused[pick]),
        "disagreement": float(best_dis),
        "source": source,
    }


def _finalize_fork(
    cands: list,
    n_considered: int,
    response_ids: list[int],
    *,
    metric: str,
    min_entropy: float,
    select: str,
    topk_positions: int,
    dedup_main: bool,
    fork_alpha: float = 1.0,
    fork_fuse: str = "blend",
    sampled_logprobs: Optional[list[float]] = None,
    n_mask_skipped: int = 0,
    num_forks: int = 1,
    fork_min_gap: int = 0,
) -> dict:
    """Choose fork position(s) from scored candidates and build the result dict.

    Selection is a two-stage process:

    1. **Entropy gate.** ``min_entropy`` *filters* the candidate pool instead of
       vetoing the whole prompt after the fact. Ranking by entropy alone still
       returns a fork on a response whose most uncertain position is a near-certain
       token, where forcing a top-k alternative injects a token the policy would
       essentially never sample; those forks are the ones that push the branch
       off-policy. Dropping them here degrades the prompt to plain rollouts
       (``none_reason="below_min_entropy"``) rather than branching badly.
    2. **How the two signals combine** (``fork_fuse``):
       - ``"blend"`` (default): ``alpha * rank(H) + (1 - alpha) * rank(D)``. A
         position has to score well on *both* to stay in the top-k, so a
         high-entropy / low-disagreement token (or the reverse) is squeezed out.
       - ``"max"``: ``max(rank(H), rank(D))``. Either extreme ranks at the top.
       - ``"union"``: take the top ``ceil(k/2)`` by entropy *and* the top
         ``ceil(k/2)`` by disagreement, then pick from that union. This is the
         mode that actually lets both kinds of position be selected.

    3. **How many forks** (``num_forks``, ``B``). ``B == 1`` returns the single best
       position. ``B > 1`` walks the same ordering greedily, skipping anything within
       ``fork_min_gap`` tokens of a position already taken, and returns them under
       ``forks``. Fork 0's fields are mirrored at the top level for existing readers.

    Each fork carries ``cand_logprobs`` (aligned with ``cand_token_ids``) and
    ``main_logprob`` so the caller can weight each branch by the student's own
    probability of the forced token. ``source`` is ``"entropy"`` /
    ``"disagreement"`` / ``"both"`` for diagnostics.
    """
    if not cands:
        return {
            "pos": None,
            "none_reason": "all_positions_filtered",
            "num_mask_skipped": n_mask_skipped,
            "forks": [],
        }

    n_before_gate = len(cands)
    # Stage 1: entropy gate as a filter (only meaningful for the entropy metric).
    if metric == "entropy" and min_entropy > 0.0:
        gated = [c for c in cands if c[0] >= min_entropy]
        if not gated:
            best = max(c[0] for c in cands)
            return {
                "pos": None,
                "none_reason": "below_min_entropy",
                "score": float(best),
                "num_mask_skipped": n_mask_skipped,
                "forks": [],
            }
        cands = gated

    # Stage 2: combine entropy with teacher-disagreement.
    alpha = min(1.0, max(0.0, float(fork_alpha)))
    fuse = str(fork_fuse)
    use_teacher = fuse in ("max", "union") or alpha < 1.0
    n_h = n_d = 0
    set_h = set_d = None
    if not use_teacher:
        fused = [c[0] for c in cands]
        order = sorted(range(len(cands)), key=lambda i: fused[i], reverse=True)
        pool = order[: max(1, topk_positions)] if select == "topk_uniform" else order[:1]
        ranked = order
        n_h = len(pool)
    else:
        rank_h = _normalized_ranks([c[0] for c in cands])
        rank_d = _normalized_ranks([c[4] for c in cands])
        n_each = max(1, (int(topk_positions) + 1) // 2)
        top_h = sorted(range(len(cands)), key=lambda i: cands[i][0], reverse=True)[:n_each]
        top_d = sorted(range(len(cands)), key=lambda i: cands[i][4], reverse=True)[:n_each]
        set_h, set_d = set(top_h), set(top_d)
        if fuse == "union":
            # Preserve entropy-first then disagreement so ties stay deterministic.
            pool = list(dict.fromkeys(top_h + top_d))
            fused = [max(rank_h[i], rank_d[i]) for i in range(len(cands))]
            ranked = sorted(pool, key=lambda i: fused[i], reverse=True)
            n_h, n_d = len(set_h), len(set_d)
        else:
            if fuse == "max":
                fused = [max(rank_h[i], rank_d[i]) for i in range(len(cands))]
            else:  # blend
                fused = [alpha * rank_h[i] + (1.0 - alpha) * rank_d[i] for i in range(len(cands))]
            order = sorted(range(len(cands)), key=lambda i: fused[i], reverse=True)
            pool = order[: max(1, topk_positions)] if select == "topk_uniform" else order[:1]
            ranked = order
            n_h = sum(1 for i in pool if i in set_h)
            n_d = sum(1 for i in pool if i in set_d)

    # Stage 3: how many fork positions to open. B == 1 keeps the original single-pick
    # semantics byte for byte; B > 1 walks the same ordering greedily under a gap.
    b = max(1, int(num_forks))
    if b == 1:
        picks = [random.choice(pool)] if select == "topk_uniform" else [ranked[0]]
    else:
        seq = random.sample(pool, len(pool)) if select == "topk_uniform" else ranked
        picks = _greedy_pick(cands, seq, b, max(0, int(fork_min_gap)))

    forks = [
        _fork_entry(
            cands,
            i,
            response_ids,
            dedup_main=dedup_main,
            sampled_logprobs=sampled_logprobs,
            fused=fused,
            set_h=set_h,
            set_d=set_d,
        )
        for i in picks
    ]

    # Fork 0's fields stay at the top level so every existing reader (metrics,
    # has_fork checks) keeps working unchanged; ``forks`` carries the full list.
    result = dict(forks[0])
    result.update(
        {
            "forks": forks,
            "num_forks": len(forks),
            "num_positions": n_considered,
            "num_gated": n_before_gate - len(cands),
            "num_mask_skipped": n_mask_skipped,
            "pool_entropy": n_h,
            "pool_disagreement": n_d,
        }
    )
    return result


async def select_fork(
    server_manager,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    topk: int,
    metric: str,
    min_tokens: int,
    response_length: int,
    tokenizer=None,
    special_ids: Optional[set] = None,
    skip_first: int = 1,
    min_token_strip_len: int = 1,
    min_entropy: float = 0.0,
    select: str = "argmax",
    topk_positions: int = 20,
    dedup_main: bool = True,
    filter_mode: str = "math_aware",
    fork_alpha: float = 1.0,
    fork_fuse: str = "blend",
    disagreement: Optional[list[float]] = None,
    sampled_logprobs: Optional[list[float]] = None,
    eligible_mask: Optional[list[int]] = None,
    num_forks: int = 1,
    fork_min_gap: int = 0,
) -> Optional[dict]:
    """Second student forward to locate a high-uncertainty response fork position.

    Runs a ``max_tokens=1`` generation with ``prompt_logprobs=topk`` over
    ``prompt_ids + response_ids`` and reuses vLLM's ``extract_prompt_logprobs``
    output (stored in ``extra_fields``) to read per-position top-k logprobs/ids.

    Candidate positions are filtered (see ``_passes_token_filter``) to skip the
    opener, special tokens, and meaningless punctuation/whitespace, then the fork
    is chosen by ``select`` ("argmax" over uncertainty, or "topk_uniform"). The
    main-sampled token can be removed from the expanded candidate set
    (``dedup_main``).

    Prefer ``select_fork_from_topk`` (Scheme B) when the main rollout already
    carries per-token top-k logprobs -- it avoids this extra forward entirely.

    Always returns a dict. On success it has a non-``None`` ``pos`` plus
    ``cand_token_ids``/``score``/``num_positions``; on failure ``pos`` is
    ``None`` and ``none_reason`` explains why (used for diagnostics).
    """
    if not response_ids:
        return {"pos": None, "none_reason": "empty_response"}

    seq = list(prompt_ids) + list(response_ids)
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": False,
        "prompt_logprobs": int(topk),
        "max_tokens": 1,
    }
    out = await server_manager.generate(
        request_id=uuid4().hex,
        prompt_ids=seq,
        sampling_params=sampling_params,
    )
    pl = out.extra_fields.get("prompt_logprobs")
    pid = out.extra_fields.get("prompt_ids")
    if not pl or not pid:
        return {"pos": None, "none_reason": "no_prompt_logprobs"}

    special_ids = special_ids or set()
    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)

    # extract_prompt_logprobs drops the distribution for the very first token, so
    # the distribution that produced absolute token j lives at index (j - 1).
    def _row(p: int):
        di = (prompt_len + p) - 1
        if di < 0 or di >= len(pl):
            return None
        return pl[di], pid[di]

    cands, n_considered, n_mask_skipped = _collect_fork_candidates(
        response_ids,
        _row,
        resp_len=resp_len,
        min_tokens=min_tokens,
        response_length=response_length,
        skip_first=skip_first,
        tokenizer=tokenizer,
        special_ids=special_ids,
        min_token_strip_len=min_token_strip_len,
        filter_mode=filter_mode,
        metric=metric,
        disagreement=disagreement,
        eligible_mask=eligible_mask,
    )
    return _finalize_fork(
        cands,
        n_considered,
        response_ids,
        metric=metric,
        min_entropy=min_entropy,
        select=select,
        topk_positions=topk_positions,
        dedup_main=dedup_main,
        fork_alpha=fork_alpha,
        fork_fuse=fork_fuse,
        sampled_logprobs=sampled_logprobs,
        n_mask_skipped=n_mask_skipped,
        num_forks=num_forks,
        fork_min_gap=fork_min_gap,
    )


def select_fork_from_topk(
    response_ids: list[int],
    out_logprobs: list,
    out_ids: list,
    *,
    metric: str,
    min_tokens: int,
    response_length: int,
    tokenizer=None,
    special_ids: Optional[set] = None,
    skip_first: int = 1,
    min_token_strip_len: int = 1,
    min_entropy: float = 0.0,
    select: str = "argmax",
    topk_positions: int = 20,
    dedup_main: bool = True,
    filter_mode: str = "math_aware",
    fork_alpha: float = 1.0,
    fork_fuse: str = "blend",
    disagreement: Optional[list[float]] = None,
    sampled_logprobs: Optional[list[float]] = None,
    eligible_mask: Optional[list[int]] = None,
    num_forks: int = 1,
    fork_min_gap: int = 0,
) -> dict:
    """Scheme B: locate the fork from the main rollout's OWN per-token top-k
    logprobs (captured during generation via ``logprobs=k``), with no extra
    student forward.

    ``out_logprobs[p]`` / ``out_ids[p]`` are the top-k logprobs / token ids of the
    decoding distribution that produced response token ``p``. Unlike prompt
    logprobs, these are already position-aligned to ``response_ids`` (no first-token
    offset), so index ``p`` maps directly.

    Because these come from the actual sampling pass (temperature ``T``,
    ``processed_logprobs``), the entropy here is the *decoding* entropy at the
    sampling temperature -- which is the definition of a forking token in
    Beyond-80/20 -- rather than the raw ``T=0`` distribution used by the
    second-forward ``select_fork``. See ``scheme_b_validate`` for an agreement
    check between the two.

    Returns the same dict schema as ``select_fork``.
    """
    if not response_ids:
        return {"pos": None, "none_reason": "empty_response"}
    if not out_logprobs or not out_ids:
        return {"pos": None, "none_reason": "no_output_logprobs"}

    special_ids = special_ids or set()
    resp_len = min(len(response_ids), len(out_logprobs), len(out_ids))

    def _row(p: int):
        if p >= len(out_logprobs) or p >= len(out_ids):
            return None
        return out_logprobs[p], out_ids[p]

    cands, n_considered, n_mask_skipped = _collect_fork_candidates(
        response_ids,
        _row,
        resp_len=resp_len,
        min_tokens=min_tokens,
        response_length=response_length,
        skip_first=skip_first,
        tokenizer=tokenizer,
        special_ids=special_ids,
        min_token_strip_len=min_token_strip_len,
        filter_mode=filter_mode,
        metric=metric,
        disagreement=disagreement,
        eligible_mask=eligible_mask,
    )
    return _finalize_fork(
        cands,
        n_considered,
        response_ids,
        metric=metric,
        min_entropy=min_entropy,
        select=select,
        topk_positions=topk_positions,
        dedup_main=dedup_main,
        fork_alpha=fork_alpha,
        fork_fuse=fork_fuse,
        sampled_logprobs=sampled_logprobs,
        n_mask_skipped=n_mask_skipped,
        num_forks=num_forks,
        fork_min_gap=fork_min_gap,
    )
