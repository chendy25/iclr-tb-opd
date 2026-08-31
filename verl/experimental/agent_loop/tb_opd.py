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


# Environment-provided success flags, checked before falling back to rule-based
# scoring of the decoded text. An agentic task's outcome is decided by the env, not
# by anything recoverable from the transcript.
_ENV_OUTCOME_KEYS = ("alfworld_won", "env_success")


def score_trajectory(tokenizer, output, kwargs: dict, threshold: float) -> tuple[float, bool]:
    """Success of a rollout, preferring an environment-provided outcome.

    ``score_solution`` answers "does the decoded text match the ground truth", which
    is meaningless for an env task: ALFWorld rows carry no ``reward_model``, so it
    would report every episode as a failure and ``only_fail`` would branch even on
    episodes that already won. When the agent loop reports the env outcome, use it.
    """
    extra = getattr(output, "extra_fields", None) or {}
    for key in _ENV_OUTCOME_KEYS:
        if extra.get(key) is not None:
            score = float(extra[key])
            return score, score >= threshold
    reward = getattr(output, "reward_score", None)
    if reward is not None and normalize_ground_truth(kwargs.get("reward_model")) is None:
        score = float(reward)
        return score, score >= threshold
    return score_solution(tokenizer, list(output.response_ids), kwargs, threshold)


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
) -> tuple[list, int]:
    """Score every eligible response position as a fork candidate.

    ``row_getter(p)`` returns ``(logprobs, token_ids)`` -- the top-k logprobs and
    token ids (descending by rank) of the decoding distribution at response
    position ``p`` -- or ``None`` if unavailable. Shared by the second-forward
    (``select_fork``) and Scheme-B (``select_fork_from_topk``) paths so filtering
    and scoring stay identical; only the source of the per-position top-k differs.
    """
    # The row's logprobs ride along with its token ids: they are the student's own
    # probabilities of each candidate, which is exactly what Rao-Blackwell branch
    # weighting needs. Recovering them later would cost a second forward pass.
    cands: list[tuple[float, int, list[int], list[float]]] = []
    n_considered = 0
    for p in range(resp_len):
        # Skip opener positions (CURE skips position 0).
        if p < skip_first:
            continue
        # Need enough remaining budget to make a branch worthwhile.
        if response_length - p < min_tokens:
            break
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
        cands.append((fork_uncertainty(row_lp, metric), p, row_id, row_lp))
    return cands, n_considered


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
) -> dict:
    """Choose a fork position from scored candidates and build the result dict."""
    if not cands:
        return {"pos": None, "none_reason": "all_positions_filtered"}

    if select == "topk_uniform":
        cands.sort(key=lambda c: c[0], reverse=True)
        pool = cands[: max(1, topk_positions)]
        best_score, best_pos, best_cands, best_lps = random.choice(pool)
    else:  # argmax
        best_score, best_pos, best_cands, best_lps = max(cands, key=lambda c: c[0])

    # Entropy gate (only meaningful for the entropy metric).
    if metric == "entropy" and min_entropy > 0.0 and best_score < min_entropy:
        return {"pos": None, "none_reason": "below_min_entropy", "score": float(best_score)}

    # The student's own logprob of the token it sampled, read off the same top-k row so
    # it is on the same scale as the candidates'. Stays None when the sampled token fell
    # outside top-k, which makes RB weighting degrade to uniform for this fork rather
    # than mixing a logprob from a different distribution into the ratio.
    main_tok = int(response_ids[best_pos])
    main_logprob = next((lp for t, lp in zip(best_cands, best_lps, strict=False) if int(t) == main_tok), None)

    # Exclude the main-sampled token so each branch is a genuine alternative. The
    # logprobs are filtered in lockstep -- dropping only the ids would shift every
    # candidate's weight onto its neighbour.
    if dedup_main:
        paired = [(t, lp) for t, lp in zip(best_cands, best_lps, strict=False) if int(t) != main_tok]
        if paired:
            best_cands = [t for t, _ in paired]
            best_lps = [lp for _, lp in paired]

    return {
        "pos": best_pos,
        "cand_token_ids": best_cands,
        "cand_logprobs": best_lps,
        "main_logprob": main_logprob,
        "score": float(best_score),
        "num_positions": n_considered,
    }


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

    cands, n_considered = _collect_fork_candidates(
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
    )


# ---------------------------------------------------------------------------
# Turn-level branching (TB-OPD-Turn).
#
# Token- and turn-level forks are scored by the same functional
#
#     score(c) = Fuse( norm(U(c)), norm(D(c)) )    over candidates c in Eligibility
#
# and differ only in which candidates exist and over which span U and D are
# averaged. ``U`` is student uncertainty: truncated top-k entropy when the rollout
# carried per-position top-k, else the mean-NLL proxy mean(-log p) of the sampled
# tokens; optionally minus a per-trajectory baseline (ARPO's ΔH_post-tool, whose
# baseline is the *first* assistant turn, not the preceding one). ``D`` is
# teacher-student disagreement, by default the same signed forward-window
# log-ratio the token path uses.
#
# Provenance: ARPO (entropy rises right after a tool response, so that is the
# natural decision point), ATOD (Soft-OR of disagreement and uncertainty beats
# entropy alone), AEPO (penalize branching on consecutive high-signal turns).
# See docs/proposals/agentic_tb_opd_research.md.
# ---------------------------------------------------------------------------

# Where a turn fork may land. ``post_tool``/``turn_open`` fork at a turn's first
# token (ARPO); ``reasoning`` and ``action`` need per-turn action spans and fork
# inside the turn, which is what separates "branch the thinking" from "branch the
# tool call / env action". ``all`` imposes no positional prior, which is what the
# token path does.
TURN_ELIGIBILITY = ("post_tool", "turn_open", "reasoning", "action", "all")

# Uncertainty statistic for a turn candidate. Parallel to the token path's
# ``entropy | topk_gap``: the metric names *only* how U is measured. How U and the
# teacher-disagreement D combine is the independent ``fork_alpha`` / ``fork_fuse``
# pair, exactly as on the token path -- collapsing all of that into one enum is what
# made ``hybrid`` silently mean three things at once.
# ``entropy`` is the token path's spelling of the same statistic and is accepted as
# an alias, so one ``fork_metric`` value is valid for both fork units.
TURN_METRICS = ("ent", "entropy", "dHtool")

# Fusions shared with the token path, plus ATOD's Soft-OR. All except ``blend`` with
# ``alpha == 1`` need a teacher signal.
TURN_FUSES = ("blend", "max", "union", "soft_or")


def segment_assistant_turns(response_mask: list[int]) -> list[tuple[int, int, bool]]:
    """Split a flat response into assistant turns from ``response_mask``.

    ``response_mask[p] == 1`` marks an LLM-generated (assistant) token and ``0`` a
    tool/observation (environment-injected) token. Each maximal run of 1s is one
    assistant turn. Returns a list of ``(start, end, post_tool)`` in response
    coordinates, where ``post_tool`` is True iff the run is immediately preceded by
    at least one 0 (i.e. the turn follows a tool response), which is ARPO's natural
    decision point.
    """
    turns: list[tuple[int, int, bool]] = []
    n = len(response_mask)
    p = 0
    while p < n:
        if response_mask[p] != 1:
            p += 1
            continue
        start = p
        while p < n and response_mask[p] == 1:
            p += 1
        end = p  # exclusive
        post_tool = start > 0 and response_mask[start - 1] == 0
        turns.append((start, end, post_tool))
    return turns


def _turn_nll_proxy(logprobs: list[float], start: int, end: int, first_k: int) -> float:
    """Mean(-logp) over the first ``first_k`` tokens of ``[start, end)`` (ATOD h_k).

    A one-sample estimate of the cross-entropy at each position: cheap, but biased
    and noisy next to the true decoding entropy. Used only when the rollout did not
    carry per-position top-k (see ``_span_entropy``).
    """
    hi = end if first_k <= 0 else min(end, start + first_k)
    vals = [logprobs[i] for i in range(start, hi) if i < len(logprobs)]
    if not vals:
        return 0.0
    return sum(-lp for lp in vals) / len(vals)


def _span_entropy(topk_logprobs: list, start: int, end: int, first_k: int) -> float:
    """Mean truncated top-k entropy over the first ``first_k`` tokens of ``[start, end)``.

    Same estimator the token path uses (``_truncated_entropy``), so turn and token
    forks are ranked by the same quantity rather than by two different proxies.
    """
    hi = end if first_k <= 0 else min(end, start + first_k)
    vals = []
    for i in range(start, hi):
        if i >= len(topk_logprobs):
            break
        row = topk_logprobs[i]
        if not row:
            continue
        row = [x for x in row if x is not None]
        if len(row) >= 2:
            vals.append(_truncated_entropy(row))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


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
    which is exactly the region worth spending branches on. Ported verbatim from
    the token path so both granularities share one definition.
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


def _turn_disagreement(
    student_lp: list[float], teacher_lp: list[float], start: int, end: int, first_k: int
) -> float:
    """Mean |teacher_logp - student_logp| over the first ``first_k`` turn tokens (ATOD d_k).

    Unsigned variant; kept for the ``disagreement_signed=False`` ablation. The
    signed form is preferred because ``student - teacher > 0`` isolates the
    "student is over-confident here" direction, whereas the absolute value merges
    it with "student is needlessly timid".
    """
    hi = end if first_k <= 0 else min(end, start + first_k)
    vals = []
    for i in range(start, hi):
        if i < len(student_lp) and i < len(teacher_lp):
            vals.append(abs(teacher_lp[i] - student_lp[i]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _span_mean(values: list[float], start: int, end: int, first_k: int) -> float:
    hi = end if first_k <= 0 else min(end, start + first_k)
    vals = [values[i] for i in range(start, hi) if i < len(values)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo <= 1e-8:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


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


# ---------------------------------------------------------------------------
# Branch weighting and shared-prefix accounting.
#
# Ported from the math token arm so both fork units train their branches under
# the same estimator; the functions are unit-agnostic (they see slots and token
# positions, not tokens vs turns).
# ---------------------------------------------------------------------------


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
    branch-OPD off-policy: the group's gradient estimates ``(1/(k+1)) * sum_j f(a_j)``
    when the on-policy quantity is ``E_{a ~ pi_theta}[f(a)]``. Reweighting by the
    renormalized student probability ``pi_theta(a_j) / sum_i pi_theta(a_i)`` turns the
    sum back into a conditional expectation over the sampled support -- the standard
    Rao-Blackwell estimator, unbiased for that support and strictly lower-variance than
    sampling one continuation. Concretely it stops a rank-5 token the student would emit
    ~1% of the time from carrying the same gradient mass as the token it actually chose.

    Weights are rescaled to sum to ``len(logprobs)`` (mean 1) so the group's total
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


def multifork_branch_weights(
    forks: list[dict],
    assignments: list[tuple[int, int]],
    *,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> Optional[list[float]]:
    """Combine ``B`` per-fork Rao-Blackwell estimators into one weight per slot.

    B forks on a single trajectory are *not* a joint expectation over B tokens -- that
    would need a ``k**B`` tree, not ``1 + B*k`` rows. They are B separate
    Rao-Blackwellizations of the same sample, so the estimator is their average
    ``(1/B) * sum_b g_b``, where each ``g_b`` is exactly the single-fork estimator.

    Averaging is what dissolves the apparent conflict of the main trajectory "needing B
    weights at once": it appears in every ``g_b``, so it carries the *mean* of its
    per-fork weights -- a single scalar. A branch appears in one ``g_b`` only, so it
    carries ``w_j / B``. Pinning the main slot at 1.0 instead would hand its share to
    the forced branches, i.e. reintroduce exactly the off-policy bias RB exists to
    remove.

    Weights are rescaled to sum to ``n_slots`` so the mean stays 1 and the effective
    learning rate is unchanged. At ``B == 1`` the scale is exactly 1 and this reduces to
    the plain conditional expectation.

    Slots that wrap onto the same ``(fork, candidate)`` pair are independent samples of
    one forced branch and split its weight, so a fork's total mass does not depend on
    how many slots happened to land on it.

    ``assignments[i]`` is the ``(fork_index, candidate_index)`` that branch slot ``i+1``
    actually forced. Passing the realized assignment rather than recomputing it from a
    modulo is deliberate: the candidate index wraps on the number of candidates that
    survived dedup, which is not necessarily the configured ``k``, so a derived mapping
    can disagree with what was really generated and silently weight the wrong branch.

    Returns ``None`` only if the weights cannot be normalized at all; a single fork
    missing its logprobs degrades to uniform rather than disabling the group.
    """
    if not forks or not assignments:
        return None
    n_slots = len(assignments) + 1

    # Per-fork weight vectors, sized to the candidates that fork actually owns.
    cand_count: dict[int, int] = {}
    for b, j in assignments:
        cand_count[b] = max(cand_count.get(b, 0), int(j) + 1)

    per_fork: dict[int, list[float]] = {}
    for b, n_cand in cand_count.items():
        fk = forks[b] if 0 <= b < len(forks) else {}
        lps = fk.get("cand_logprobs") or []
        w = None
        if lps:
            slot_lps = [fk.get("main_logprob")] + [lps[j % len(lps)] for j in range(n_cand)]
            w = branch_weights(slot_lps, temperature=temperature, floor=floor)
        # Degrade one unweightable fork to uniform instead of dropping RB for the whole
        # group: with B forks a single missing logprob would otherwise cost all of them,
        # and the odds of hitting one scale with B.
        per_fork[b] = w if w is not None else [1.0] * (n_cand + 1)

    # A (fork, candidate) pair owns several slots whenever the slot budget does not
    # divide evenly. Those slots are independent samples of the SAME forced branch, so
    # they must split that branch's weight -- leaving each at full weight multiplies the
    # fork's mass by its slot count, and after renormalization that drags the main
    # trajectory down, inverting the very thing RB is for.
    counts: dict[tuple[int, int], int] = {}
    for key in assignments:
        counts[key] = counts.get(key, 0) + 1

    # Average only over forks that actually own slots: a fork that never got generated
    # must not raise the main trajectory's weight on the strength of branches that do
    # not exist.
    active = sorted(per_fork)
    if not active:
        return None
    b_active = len(active)

    out = [sum(per_fork[b][0] for b in active) / b_active]
    for b, j in assignments:
        out.append(per_fork[b][1 + j] / (b_active * counts[(b, j)]))

    # Rescale by the realized total rather than a closed form: the two agree when the
    # budget divides evenly, and only the realized total keeps the mean at 1 when it
    # does not.
    total = sum(out)
    if not math.isfinite(total) or total <= 0.0:
        return None
    scale = float(n_slots) / total
    return [x * scale for x in out]


def mask_shared_prefix(response_mask: list[int], prefix_len: int) -> list[int]:
    """Drop a branch's replay of the main trajectory from the loss.

    A branch is generated as ``main[:fork] + forced_token + continuation``, so the first
    ``prefix_len`` tokens are byte-identical to the main trajectory. Left unmasked they
    are trained once per slot, which up-weights early tokens by the number of slots.
    Zeroing them leaves the prefix supervised exactly once, by the main slot.

    On the turn path the prefix is a whole prefix of the episode, so this also zeroes
    the environment-observation tokens inside it -- which were already masked, making
    the operation idempotent there.
    """
    if prefix_len <= 0:
        return list(response_mask)
    n = min(int(prefix_len), len(response_mask))
    return [0] * n + list(response_mask[n:])


def fuse_signals(
    uncertainty: list[float],
    disagreement: Optional[list[float]],
    *,
    alpha: float,
    fuse: str,
    normalize: str = "rank",
) -> list[float]:
    """Combine uncertainty and disagreement into one ranking score.

    ``normalize`` picks the per-trajectory scale: ``rank`` (the token path's choice)
    is robust to the two signals living on different scales; ``minmax`` keeps ATOD's
    Soft-OR interpretable as a probability-like value.

    ``fuse``:
      - ``blend``   : ``alpha * u + (1 - alpha) * d`` -- a candidate must do well on
                      both (alpha=1 -> uncertainty only, no teacher needed)
      - ``max``     : ``max(u, d)`` -- either extreme survives
      - ``union``   : scored like ``max``; the *pool* restriction happens in the
                      caller, which keeps the top half by each signal separately
      - ``soft_or`` : ``1 - (1 - u)(1 - d)`` (ATOD T-DUR)
    """
    norm = _normalized_ranks if normalize == "rank" else _minmax
    u = norm(uncertainty)
    if disagreement is None:
        return u
    d = norm(disagreement)
    if fuse == "soft_or":
        return [1.0 - (1.0 - ui) * (1.0 - di) for ui, di in zip(u, d)]
    if fuse in ("max", "union"):
        return [max(ui, di) for ui, di in zip(u, d)]
    a = float(alpha)
    return [a * ui + (1.0 - a) * di for ui, di in zip(u, d)]


def turn_fork_candidates(
    turns: list[tuple[int, int, bool]],
    *,
    skip_first_turns: int,
    turn_first_k: int,
    action_spans: Optional[list] = None,
    eligibility: str = "reasoning",
) -> list[dict]:
    """Enumerate fork candidates over a segmented multi-turn trajectory.

    One candidate == one *segment* of a turn. A candidate carries ``pos`` (the token
    the branch re-enters at), the ``[span_start, span_end)`` window whose mean ``U``
    and ``D`` score it, and ``first_k`` (how much of that window to average, ``0`` =
    all of it). Separating ``pos`` from the span is what lets one scorer rank "fork
    at the turn opening" (ARPO) against "fork at the tool call".

    Each turn contributes at most one candidate per kind:

      ``reasoning``  fork at the turn's first token, scored over the whole thinking
                     span. The branch re-thinks the turn from scratch.
      ``action``     fork at the first token of the action / tool-call block, scored
                     over that block. The thinking is kept verbatim and only the
                     action changes.
      ``turn_open``  fork at the turn's first token, scored over the whole turn
                     (thinking *and* action), with ARPO's first-``turn_first_k``
                     window. Same fork point as ``reasoning``, coarser score.

    ``turn_open`` and ``reasoning`` fork at the *same* token, so emitting both under
    ``all`` would put one physical fork point in the pool twice and simply take the
    max of two scorings of it. ``all`` therefore emits ``reasoning`` + ``action`` --
    one candidate per segment, so the pool is balanced 1:1 between "change the
    thinking" and "change the action" and a fork_kind histogram means something.
    ``turn_open`` is what the explicit ``turn_open`` / ``post_tool`` (ARPO) arms use.

    ``action_spans[i]`` is the ``(start, end)`` response-coordinate range of turn
    ``i``'s action tokens, or ``None`` when unknown. Without it thinking and action
    cannot be told apart, so the turn falls back to a single whole-turn candidate.
    """
    cands: list[dict] = []
    for ti, (start, end, post_tool) in enumerate(turns):
        if ti < skip_first_turns:
            continue
        span = None
        if action_spans is not None and ti < len(action_spans) and action_spans[ti]:
            a_start, a_end = int(action_spans[ti][0]), int(action_spans[ti][1])
            if start <= a_start < a_end <= end:
                span = (a_start, a_end)

        if eligibility in ("post_tool", "turn_open") and not (
            eligibility == "post_tool" and not post_tool
        ):
            cands.append(
                {
                    "pos": start,
                    "span_start": start,
                    "span_end": end,
                    "first_k": turn_first_k,
                    "turn_index": ti,
                    "post_tool": post_tool,
                    "kind": "turn_open",
                }
            )

        # Emitted in response order (thinking, then action) so that equal scores break
        # toward the earlier fork point rather than toward whichever kind is listed first.
        if eligibility in ("reasoning", "all"):
            r_end = span[0] if span is not None else end
            if r_end > start:
                cands.append(
                    {
                        "pos": start,
                        "span_start": start,
                        "span_end": r_end,
                        # Mean over the entire thinking span, not a leading window:
                        # the segment is the unit being ranked.
                        "first_k": 0,
                        "turn_index": ti,
                        "post_tool": post_tool,
                        # Without an action span this covers the whole turn, so call
                        # it what it is rather than claiming the thinking was isolated.
                        "kind": "reasoning" if span is not None else "turn_open",
                    }
                )

        if eligibility in ("action", "all") and span is not None:
            cands.append(
                {
                    "pos": span[0],
                    "span_start": span[0],
                    "span_end": span[1],
                    # The action block IS the segment, so score all of it.
                    "first_k": 0,
                    "turn_index": ti,
                    "post_tool": post_tool,
                    "kind": "action",
                }
            )
    return cands


def select_fork_turn(
    response_mask: list[int],
    response_logprobs: list[float],
    *,
    metric: str = "entropy",
    teacher_logprobs: Optional[list[float]] = None,
    turn_first_k: int = 16,
    only_post_tool: bool = False,
    skip_first_turns: int = 0,
    min_uncertainty: float = 0.0,
    consecutive_penalty: bool = False,
    consecutive_penalty_weight: float = 0.5,
    eligibility: Optional[str] = None,
    action_spans: Optional[list] = None,
    topk_logprobs: Optional[list] = None,
    fork_alpha: float = 0.5,
    fork_fuse: str = "blend",
    fork_kl_window: int = 128,
    disagreement_signed: bool = True,
    normalize: str = "rank",
    max_forks: int = 1,
    min_turn_gap: int = 1,
    select: str = "argmax",
    topk_positions: int = 20,
) -> dict:
    """Pick fork point(s) in a multi-turn trajectory.

    Operates purely on the main trajectory's per-token student ``response_logprobs``
    (plus optional aligned ``teacher_logprobs`` and per-position ``topk_logprobs``);
    no extra generation. Returns a dict with ``pos`` = the response-coordinate index
    the branch re-enters at, ``forks`` = all selected fork dicts (length
    ``<= max_forks``), plus diagnostics. On failure ``pos`` is ``None`` with a
    ``none_reason``.

    The knobs mirror the token path one-for-one. ``metric`` names only how ``U`` is
    measured:
      - ``ent`` / ``entropy``: the turn's uncertainty (same statistic, and the same
        spelling the token path uses).
      - ``dHtool``: ARPO ΔH_post-tool, i.e. ``ent`` minus the first turn's value.
    How ``U`` and the teacher term ``D`` combine is ``fork_alpha`` / ``fork_fuse``,
    and the teacher is consulted exactly when the token path would consult it:
    ``fork_fuse in ("max", "union", "soft_or") or fork_alpha < 1.0``.

    ``eligibility`` decides *where* a fork may land. Default ``reasoning``: fork
    at the turn's first token, scored over the whole thinking span (the branch
    re-thinks the turn). ``action`` keeps the thinking and forks the tool-call /
    env action; ``all`` pools both 1:1; ``turn_open``/``post_tool`` score the
    whole turn (ARPO). ``None`` derives ``post_tool`` from ``only_post_tool`` and
    ``reasoning`` otherwise. ``action``/``reasoning`` need ``action_spans``.

    ``select`` mirrors the token path's ``fork_select``: ``argmax`` always takes the
    top-scoring candidate, ``topk_uniform`` draws uniformly among the top
    ``topk_positions``. Note the two paths' scores are not on the same scale -- the
    token path ranks single positions, this one ranks span means -- so a candidate
    "rank" means the same thing but the underlying value does not.

    ``min_uncertainty`` (the token path's ``fork_min_entropy``) drops candidates
    whose *raw* ``U`` is below it, so a trajectory the student was confident all the
    way through emits no fork and its extra slots degrade to plain rollouts. Note
    the threshold is in the units of whichever estimator ran: truncated entropy when
    ``topk_logprobs`` is supplied, otherwise the mean-NLL proxy, which is on a
    different scale. ``uncertainty_estimator`` in the result says which.
    """
    if not response_mask or not response_logprobs:
        return {"pos": None, "none_reason": "empty_response"}

    if metric not in TURN_METRICS:
        return {"pos": None, "none_reason": f"unknown_metric:{metric}"}
    if fork_fuse not in TURN_FUSES:
        return {"pos": None, "none_reason": f"unknown_fuse:{fork_fuse}"}
    use_baseline = "first_turn" if metric == "dHtool" else "none"
    fuse = str(fork_fuse)
    alpha = float(fork_alpha)
    want_teacher = fuse in ("max", "union", "soft_or") or alpha < 1.0
    use_teacher = want_teacher and teacher_logprobs is not None

    if eligibility is None:
        eligibility = "post_tool" if only_post_tool else "reasoning"
    if eligibility not in TURN_ELIGIBILITY:
        return {"pos": None, "none_reason": f"unknown_eligibility:{eligibility}"}

    turns = segment_assistant_turns(response_mask)
    if not turns:
        return {"pos": None, "none_reason": "no_assistant_turns"}

    cands = turn_fork_candidates(
        turns,
        eligibility=eligibility,
        skip_first_turns=skip_first_turns,
        turn_first_k=turn_first_k,
        action_spans=action_spans,
    )
    if not cands:
        return {"pos": None, "none_reason": "no_eligible_turns", "num_turns": len(turns)}

    # Uncertainty: truncated top-k entropy when the rollout carried top-k, else the
    # one-sample NLL proxy. Reported so runs are not silently comparing estimators.
    have_topk = bool(topk_logprobs)

    def _u(start: int, end: int, first_k: int) -> float:
        if have_topk:
            return _span_entropy(topk_logprobs, start, end, first_k)
        return _turn_nll_proxy(response_logprobs, start, end, first_k)

    base = 0.0
    if use_baseline == "first_turn":
        base = _u(turns[0][0], turns[0][1], turn_first_k)

    signed_d = (
        disagreement_window(response_logprobs, teacher_logprobs, fork_kl_window)
        if (use_teacher and disagreement_signed)
        else None
    )

    # Raw-uncertainty gate, applied BEFORE normalization -- the token path's
    # fork_min_entropy. It has to happen here: rank and minmax both map the best
    # candidate to 1.0 by construction, so a gate on the *normalized* score can
    # never fire no matter how certain the student was.
    for c in cands:
        c["uncertainty"] = float(_u(c["span_start"], c["span_end"], c["first_k"]))
    if min_uncertainty > 0.0:
        gated = [c for c in cands if c["uncertainty"] >= min_uncertainty]
        if not gated:
            return {
                "pos": None,
                "none_reason": "below_min_entropy",
                "num_turns": len(turns),
                "num_eligible": len(cands),
                "max_uncertainty": float(max(c["uncertainty"] for c in cands)),
                "uncertainty_estimator": "entropy" if have_topk else "nll",
            }
        cands = gated

    u_raw, d_raw = [], []
    for c in cands:
        s, e, fk = c["span_start"], c["span_end"], c["first_k"]
        val = c["uncertainty"]
        u_raw.append(max(0.0, val - base) if use_baseline == "first_turn" else val)
        if not use_teacher:
            d_raw.append(0.0)
        elif signed_d is not None:
            d_raw.append(_span_mean(signed_d, s, e, fk))
        else:
            d_raw.append(_turn_disagreement(response_logprobs, teacher_logprobs, s, e, fk))
        c["disagreement"] = float(d_raw[-1])

    signals = fuse_signals(
        u_raw,
        d_raw if use_teacher else None,
        alpha=alpha,
        fuse=fuse,
        normalize=normalize,
    )

    # AEPO-style consecutive high-signal penalty: if the preceding *turn* is also
    # above-median, down-weight to curb over-branching along one chain. It has to
    # look back by turn, not by list position -- a turn contributes several
    # candidates (thinking, action), so comparing against index i-1 would mostly
    # penalize a turn's action span for its own thinking being uncertain.
    if consecutive_penalty and len(signals) > 1:
        med = sorted(signals)[len(signals) // 2]
        best_by_turn: dict[int, float] = {}
        for c, s in zip(cands, signals):
            ti = int(c["turn_index"])
            best_by_turn[ti] = max(best_by_turn.get(ti, float("-inf")), s)
        penalized = list(signals)
        for i, c in enumerate(cands):
            prev = best_by_turn.get(int(c["turn_index"]) - 1)
            if prev is not None and prev >= med:
                penalized[i] = signals[i] * float(consecutive_penalty_weight)
        signals = penalized

    for c, s in zip(cands, signals):
        c["signal"] = float(s)

    if select not in ("argmax", "topk_uniform"):
        return {"pos": None, "none_reason": f"unknown_select:{select}"}

    order = sorted(range(len(cands)), key=lambda i: signals[i], reverse=True)
    if fuse == "union" and use_teacher and len(cands) > 1:
        # Keep the top half by each signal separately, then rank that union. Blending
        # lets a candidate that is merely decent on both outrank one that is extreme
        # on a single signal; union is the mode where *both* kinds of fork survive.
        half = max(1, (len(cands) + 1) // 2)
        pool = set(sorted(range(len(cands)), key=lambda i: u_raw[i], reverse=True)[:half])
        pool |= set(sorted(range(len(cands)), key=lambda i: d_raw[i], reverse=True)[:half])
        order = [i for i in order if i in pool]

    # CURE-style exploration, the token path's fork_select: instead of always taking
    # the argmax, draw uniformly among the top-N candidates. Always forking the single
    # most uncertain turn makes the branch distribution a deterministic function of the
    # student, so the same few turns get explored every epoch. Shuffling only the
    # top-N leaves the tail as fallback order, which matters when the gap constraint
    # below rejects the whole pool.
    #
    # Watch the scale: the token path draws from thousands of positions, so top-20 is
    # a narrow slice of it. Here the pool is one or two candidates per turn, i.e. tens,
    # so topk_positions=20 can cover all of it and turn selection into a coin flip.
    # select_random_frac reports how much of the ranking got shuffled -- 1.0 means the
    # scoring did nothing and the arm is a uniform-random-fork ablation.
    select_random_frac = 0.0
    if select == "topk_uniform" and len(order) > 1:
        n_pool = min(len(order), max(1, int(topk_positions)))
        select_random_frac = n_pool / len(order)
        pool_ids = order[:n_pool]
        random.shuffle(pool_ids)
        order = pool_ids + order[n_pool:]

    # Greedy pick under a turn-gap constraint so B>1 does not stack every fork on
    # one turn (the over-branching AEPO warns about).
    picked: list[dict] = []
    for i in order:
        c = cands[i]
        if any(abs(c["turn_index"] - p["turn_index"]) < max(0, min_turn_gap) for p in picked):
            continue
        picked.append(c)
        if len(picked) >= max(1, int(max_forks)):
            break

    best = picked[0]
    turn_end = turns[best["turn_index"]][1]
    return {
        "pos": int(best["pos"]),
        "turn_index": int(best["turn_index"]),
        "turn_end": int(turn_end),
        "post_tool": bool(best["post_tool"]),
        "kind": best["kind"],
        "signal": float(best["signal"]),
        "uncertainty": float(best["uncertainty"]),
        "disagreement": float(best["disagreement"]),
        "forks": picked,
        "num_turns": len(turns),
        "num_eligible": len(cands),
        "eligibility": eligibility,
        "used_teacher": bool(use_teacher),
        "uncertainty_estimator": "entropy" if have_topk else "nll",
        "select_random_frac": float(select_random_frac),
    }


async def topk_candidates_at(
    server_manager,
    prompt_ids: list[int],
    response_prefix_ids: list[int],
    *,
    topk: int,
    dedup_token: Optional[int] = None,
) -> tuple[list[int], list[float]]:
    """Top-k candidate token ids -- and their logprobs -- after ``prompt+prefix``.

    Used by forced-topk turn branching to obtain alternative first tokens for the
    forked turn. Runs a single ``max_tokens=1`` generation with
    ``prompt_logprobs=topk`` and reads the distribution for the position that would
    produce the next token (the last extracted prompt-logprob row).

    The logprobs are returned alongside because Rao-Blackwell branch weighting needs
    ``log pi_theta(a_j | s_fork)`` per slot; without them the k forced continuations
    would be averaged uniformly, which is what makes forced-topk branching off-policy.
    They stay index-aligned with the ids through the dedup filter.
    """
    seq = list(prompt_ids) + list(response_prefix_ids)
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
    pid = out.extra_fields.get("prompt_ids")
    if not pid:
        # Fall back to the single greedily-generated next token; no distribution, so no
        # RB weights either (multifork_branch_weights degrades that fork to uniform).
        return list(out.token_ids)[:1], []
    plp = out.extra_fields.get("prompt_logprobs")
    row_ids = pid[-1]
    row_lps = plp[-1] if plp else [None] * len(row_ids)
    pairs = [
        (int(i), float(lp))
        for i, lp in zip(row_ids, row_lps)
        if i is not None and lp is not None
    ]
    if dedup_token is not None:
        pairs = [(i, lp) for i, lp in pairs if i != int(dedup_token)]
    if not pairs:
        return [int(x) for x in row_ids if x is not None], []
    return [i for i, _ in pairs], [lp for _, lp in pairs]


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

    cands, n_considered = _collect_fork_candidates(
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
    )
