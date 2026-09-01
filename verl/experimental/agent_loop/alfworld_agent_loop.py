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
"""Native ALFWorld agent loop for verl's V1 agent-loop rollout.

One ``run`` == one ALFWorld episode. The transcript is accumulated (assistant
tokens masked 1, environment-observation tokens masked 0) exactly like
``ToolAgentLoop``, so it plugs straight into distillation / OPD (the teacher
logprobs and k1 policy-gradient loss operate on ``response_mask``) and into
TB-OPD-Turn later. The episode reward is task success (``won``); with
``distillation.distillation_loss.use_task_rewards=False`` it is used only for
logging/validation, not in the actor loss (pure OPD).

Registered as ``alfworld_agent``; select via
``actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent``.
"""

import hashlib
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.alfworld_env import (
    alfworld_projection,
    build_followup_user_text,
    build_initial_user_text,
    get_env_pool,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _stable_seed(key: str) -> int:
    """Deterministic 31-bit seed from a string (used when only ``uid`` is available)."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _count_assistant_turns(response_mask: list[int]) -> int:
    """Number of maximal runs of 1s (one per assistant turn)."""
    return sum(1 for i, m in enumerate(response_mask) if m == 1 and (i == 0 or response_mask[i - 1] != 1))


def _count_observation_blocks(response_mask: list[int]) -> int:
    """Number of maximal runs of 0s == number of env steps already executed.

    Each env step appends exactly one observation block, so this counts how many
    actions a branch has to replay to reach the fork point -- correct both for forks
    at a turn boundary and for forks inside a turn whose action never ran.
    """
    return sum(1 for i, m in enumerate(response_mask) if m == 0 and (i == 0 or response_mask[i - 1] != 0))


def _to_py(value: Any) -> Any:
    """Normalize numpy 0-d/objects from non_tensor_batch to plain python."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


@register("alfworld_agent")
class AlfWorldAgentLoop(AgentLoopBase):
    """Multi-turn ALFWorld (TextWorld) rollout as a verl agent loop."""

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        # ``tools`` is passed by AgentLoopWorker for all loops; ALFWorld ignores it.
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        alf_cfg = {}
        try:
            alf_cfg = self.config.get("alfworld", {}) or {}
        except Exception:  # noqa: BLE001
            alf_cfg = {}

        def _cfg(key, default):
            val = alf_cfg.get(key, default) if hasattr(alf_cfg, "get") else default
            return default if val is None else val

        self.max_steps = int(_cfg("max_steps", 50))
        self.pool_size = int(_cfg("pool_size", 16))
        self.config_path = _cfg("config_path", None)
        # AlfredTWEnv fixes its game list at construction, so training and validation
        # use two separate pools; ``run`` picks one per episode via ``validate``.
        # Splits: train | eval_in_distribution (valid_seen) | eval_out_of_distribution
        # (valid_unseen). Validation defaults to the unseen rooms.
        self.train_eval = os.environ.get("ALFWORLD_TRAIN_EVAL", str(_cfg("train_eval", "train")))
        self.eval_split = os.environ.get(
            "ALFWORLD_EVAL_SPLIT", str(_cfg("eval_split", "eval_out_of_distribution"))
        )
        # Per-turn generation cap keeps a single turn from consuming the whole budget.
        self.max_turn_tokens = int(_cfg("max_turn_tokens", 512))
        # TB-OPD-Turn with fork_eligibility=reasoning/action needs to know where each
        # turn's <action> block starts. Locating it costs a few tail decodes per turn,
        # so it is opt-in.
        self.record_action_spans = bool(_cfg("record_action_spans", False))
        # ``</think>`` id for the redundant-closing-tag probe. Left at None when the
        # tokenizer has no such token, which disables the probe instead of silently
        # counting the unk id.
        close_id = self.tokenizer.convert_tokens_to_ids("</think>")
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        self._close_think_id = None if close_id is None or close_id == unk_id else int(close_id)
        self._blank_token_cache: dict[int, bool] = {}

    def _is_blank_token(self, tok: int) -> bool:
        """Whether a token decodes to whitespace only (cached; few distinct ids)."""
        cached = self._blank_token_cache.get(tok)
        if cached is None:
            cached = self.tokenizer.decode([tok]).strip() == ""
            self._blank_token_cache[tok] = cached
        return cached

    def _count_leading_close_think(self, ids: list[int]) -> int:
        """Redundant ``</think>`` tokens at the start of one generated turn.

        With ``enable_thinking=False`` the chat template pre-fills a *closed* empty
        think block, so a ``</think>`` the model generates on top of it closes
        nothing. Exactly one is this protocol's steady state. A run of them is a
        repetition attractor: once two are in context the pattern is self-similar
        and the model copies it, the turn never reaches ``<action>``, and the
        episode burns its whole step budget on "Nothing happens". That is what took
        the reward from 0.50 to 0.016 over steps 26-33 of the first 200-step run
        (median 8-36 leading tags, max 136 in one turn), and the count moved two
        steps before the reward did. Whitespace between tags is skipped so
        ``</think>\\n\\n</think>`` counts as two rather than stopping at the newline.
        """
        if self._close_think_id is None:
            return 0
        n = 0
        for tok in ids:
            if tok == self._close_think_id:
                n += 1
            elif not self._is_blank_token(tok):
                break
        return n

    def _char_to_token(self, ids: list[int], char_idx: int) -> int:
        """Smallest ``t`` with ``len(decode(ids[:t])) >= char_idx`` (binary search)."""
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.tokenizer.decode(ids[:mid], skip_special_tokens=True)) >= char_idx:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _action_span(self, ids: list[int]) -> Optional[tuple[int, int]]:
        """Token range of the ``<action>`` block within one assistant turn.

        ALFWorld's protocol puts the action last, so the block always runs to the end
        of the turn and only its start has to be located. Searching an expanding tail
        keeps this to a handful of short decodes instead of scanning the whole turn.
        """
        n = len(ids)
        if n == 0:
            return None
        for k in (64, 128, 256, 512, n):
            k = min(k, n)
            tail = ids[n - k :]
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
            idx = text.rfind("<action>")
            if idx != -1:
                return (n - k + self._char_to_token(tail, idx), n)
            if k == n:
                break
        return None

    def _game_seed(self, kwargs: dict) -> int:
        # Prefer the dataset index: it makes the game a reproducible function of the
        # row, and rollout.n replicas of one prompt share a game (needed for GRPO
        # grouping). Rollout dumps that appeared to show this index broadcast across a
        # batch were actually reading synthetic padding rows, which copy row 0's
        # metadata -- see upsample_batch_to_divisible_size.
        extra_info = _to_py(kwargs.get("extra_info")) or {}
        if isinstance(extra_info, dict) and extra_info.get("index") is not None:
            return int(_to_py(extra_info["index"]))
        uid = _to_py(kwargs.get("uid"))
        if uid is not None:
            return _stable_seed(str(uid))
        return 0

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], validate: bool = False, **kwargs) -> AgentLoopOutput:
        split = self.eval_split if validate else self.train_eval
        return await self._play(sampling_params, split=split, kwargs=kwargs)

    @rollout_trace_op
    async def run_from_prefix(
        self,
        sampling_params: dict[str, Any],
        *,
        base_prompt_ids: list[int],
        prefix_response_ids: list[int],
        prefix_response_mask: list[int],
        prefix_response_logprobs: Optional[list[float]] = None,
        forced_first_token: Optional[int] = None,
        prefix_extra_fields: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AgentLoopOutput:
        """Resume an episode from a mid-trajectory fork point (TB-OPD-Turn E4).

        ALFWorld's backend is a TextWorld process with no snapshot/restore, so the
        branch cannot simply continue from the main trajectory's env state. Instead
        the episode is *replayed*: reset with the same seed (the game is a
        deterministic function of the seed, see ``_game_seed``) and re-issue the
        actions the main trajectory already took before the fork. Replay is exact
        because the env is deterministic given (seed, action sequence); a mismatch
        would show up as ``alfworld_replay_ok=0``.

        The number of actions to replay is the number of observation blocks in the
        prefix: one env step produced each of them. That rule holds both for forks
        at a turn boundary and for forks *inside* a turn (where the partial turn's
        tokens are in the prefix but its action was never executed).
        """
        prefix_extra_fields = prefix_extra_fields or {}
        split = str(prefix_extra_fields.get("alfworld_split") or self.train_eval)
        replay_n = _count_observation_blocks(prefix_response_mask)
        actions = list(prefix_extra_fields.get("alfworld_actions") or [])
        fingerprints = list(prefix_extra_fields.get("alfworld_obs_fingerprints") or [])
        return await self._play(
            sampling_params,
            split=split,
            kwargs=kwargs,
            base_prompt_ids=base_prompt_ids,
            prefix_response_ids=prefix_response_ids,
            prefix_response_mask=prefix_response_mask,
            prefix_response_logprobs=prefix_response_logprobs,
            forced_first_token=forced_first_token,
            replay_actions=actions[:replay_n],
            expected_fingerprints=fingerprints[:replay_n],
        )

    async def _play(
        self,
        sampling_params: dict[str, Any],
        *,
        split: str,
        kwargs: dict[str, Any],
        base_prompt_ids: Optional[list[int]] = None,
        prefix_response_ids: Optional[list[int]] = None,
        prefix_response_mask: Optional[list[int]] = None,
        prefix_response_logprobs: Optional[list[float]] = None,
        forced_first_token: Optional[int] = None,
        replay_actions: Optional[list[str]] = None,
        expected_fingerprints: Optional[list[int]] = None,
    ) -> AgentLoopOutput:
        """One episode, optionally resumed from a shared prefix (see run_from_prefix)."""
        metrics: dict[str, Any] = {}
        seed = self._game_seed(kwargs)
        pool = get_env_pool(self.config_path, split, self.pool_size)
        resumed = base_prompt_ids is not None
        expected_fingerprints = list(expected_fingerprints or [])

        handle = await pool.acquire()
        won = False
        num_env_steps = 0
        invalid_actions = 0
        # Degeneration probes (see _count_leading_close_think). Counted on the raw
        # generation before projection, so they stay meaningful if a rollout-side
        # guard that strips or bans the redundant tag is added later.
        turns_generated = 0
        turns_no_action = 0
        lead_close_first = 0
        lead_close_max = 0
        lead_close_total = 0
        gamefile = None
        replay_ok = True
        env_actions: list[str] = []
        obs_fingerprints: list[int] = []
        action_spans: list[Optional[tuple[int, int]]] = []
        # Tokens of a partially-generated turn carried over from the prefix, so the
        # first resumed projection sees the whole turn (see continuing_turn below).
        carry_ids: list[int] = []
        try:
            obs = await pool.reset(handle, seed)
            gamefile = obs.get("gamefile")

            if resumed:
                # Fast-forward the env to the fork point, then splice the shared
                # prefix back in so the branch's tokens line up with the main slot.
                for i, act in enumerate(replay_actions or []):
                    step = await pool.step(handle, act)
                    num_env_steps += 1
                    env_actions.append(act)
                    fp = _stable_seed(str(step.get("observation", "")))
                    obs_fingerprints.append(fp)
                    if step.get("gamefile") is not None:
                        gamefile = step.get("gamefile")
                    # A branch whose env state does not match its own token prefix would
                    # feed the KD loss a transcript the environment never produced, so
                    # verify rather than assume determinism.
                    if i < len(expected_fingerprints) and fp != expected_fingerprints[i]:
                        replay_ok = False
                        break
                    if bool(step.get("done", False)) or bool(step.get("won", False)):
                        # The prefix is supposed to stop short of episode end; if replay
                        # terminates early the recorded actions did not reproduce.
                        replay_ok = False
                        won = bool(step.get("won", False))
                        break
                running_ids = list(base_prompt_ids) + list(prefix_response_ids)
                response_mask = list(prefix_response_mask)
                response_logprobs = list(prefix_response_logprobs or [0.0] * len(prefix_response_mask))
                have_logprobs = bool(prefix_response_logprobs)
                # The prefix's top-k was consumed by fork selection on the main slot
                # and is not carried across; branches never re-select a fork, so empty
                # rows here only keep the index alignment.
                response_topk = [[] for _ in prefix_response_mask]
                have_topk = False
                assistant_turns = _count_assistant_turns(prefix_response_mask)
                # A fork inside a turn leaves that turn's opening tokens in the
                # prefix, so the next generation *finishes* it rather than starting
                # a new one; only a fork at a turn boundary adds a turn.
                continuing_turn = bool(prefix_response_mask) and prefix_response_mask[-1] == 1
                if forced_first_token is not None:
                    running_ids = running_ids + [int(forced_first_token)]
                    response_mask = response_mask + [1]
                    response_logprobs = response_logprobs + [0.0]
                if continuing_turn:
                    # A fork inside the <action> block leaves the opening tag in the
                    # prefix, so projecting the continuation alone would not parse.
                    # Carry the partial turn's tokens into the first projection.
                    n_partial = 0
                    while n_partial < len(prefix_response_mask) and prefix_response_mask[-1 - n_partial] == 1:
                        n_partial += 1
                    carry_ids = list(running_ids[-(n_partial + (forced_first_token is not None)) :])
            else:
                # Initial user turn: task + first observation.
                user_text = build_initial_user_text(obs["observation"], obs["admissible_commands"])
                running_ids = await self.apply_chat_template([{"role": "user", "content": user_text}])

                response_mask = []
                response_logprobs = []
                response_topk = []
                have_logprobs = False
                have_topk = False
                assistant_turns = 0
                continuing_turn = False

            turn_sampling = dict(sampling_params)
            turn_sampling["max_tokens"] = min(
                int(turn_sampling.get("max_tokens", self.max_turn_tokens) or self.max_turn_tokens),
                self.max_turn_tokens,
            )

            remaining_steps = max(0, self.max_steps - num_env_steps)
            if resumed and not replay_ok:
                remaining_steps = 0
            for _ in range(remaining_steps):
                # ---- assistant turn ----
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=running_ids,
                        sampling_params=turn_sampling,
                    )
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

                assistant_ids = output.token_ids
                if not assistant_ids:
                    break
                # Only a real turn start can carry a redundant closing tag: a fork
                # resumes mid-turn (carry_ids non-empty), where a ``</think>`` may be
                # closing a think block the prefix actually opened.
                if not carry_ids:
                    n_lead_close = self._count_leading_close_think(assistant_ids)
                    lead_close_total += n_lead_close
                    lead_close_max = max(lead_close_max, n_lead_close)
                    if turns_generated == 0:
                        lead_close_first = n_lead_close
                turns_generated += 1
                turn_start = len(response_mask) - len(carry_ids)
                turn_ids = carry_ids + assistant_ids
                carry_ids = []
                running_ids = running_ids + assistant_ids
                response_mask += [1] * len(assistant_ids)
                if output.log_probs:
                    have_logprobs = True
                    response_logprobs += list(output.log_probs)
                else:
                    response_logprobs += [0.0] * len(assistant_ids)
                # Per-position top-k, present only when the caller asked for an integer
                # ``logprobs``. Needed for truncated-entropy fork scoring; without it the
                # turn selector silently falls back to the mean-NLL proxy, which is a
                # different statistic on a different scale.
                turn_topk = output.extra_fields.get("output_logprobs") if output.extra_fields else None
                if turn_topk and len(turn_topk) == len(assistant_ids):
                    have_topk = True
                    response_topk += list(turn_topk)
                else:
                    response_topk += [[] for _ in assistant_ids]
                if continuing_turn:
                    continuing_turn = False
                else:
                    assistant_turns += 1

                assistant_text = self.tokenizer.decode(turn_ids, skip_special_tokens=True)
                action, valid = alfworld_projection(assistant_text)
                if not valid:
                    invalid_actions += 1
                # Narrower than ``valid``, which also fails on Chinese characters.
                # This is the failure the repetition attractor produces: the turn
                # runs out of tokens before it ever emits an action block.
                if "<action>" not in assistant_text:
                    turns_no_action += 1
                if self.record_action_spans:
                    local = self._action_span(turn_ids)
                    action_spans.append(
                        (turn_start + local[0], turn_start + local[1]) if local else None
                    )

                # ---- environment step ----
                with simple_timer("tool_calls", metrics):
                    step = await pool.step(handle, action)
                num_env_steps += 1
                # Recorded so a branch can rebuild this env state by replay; the
                # projected action (not the raw text) is what the env actually saw.
                env_actions.append(action)
                obs_fingerprints.append(_stable_seed(str(step.get("observation", ""))))
                won = bool(step.get("won", False))
                done = bool(step.get("done", False))
                if step.get("gamefile") is not None:
                    gamefile = step.get("gamefile")

                if done or won:
                    break
                if len(response_mask) >= self.response_length:
                    break

                # ---- environment-observation turn (masked 0) ----
                followup = build_followup_user_text(step["observation"], step["admissible_commands"])
                obs_ids = await self.apply_chat_template(
                    [{"role": "user", "content": followup}], remove_system_prompt=True
                )
                obs_ids = self.turn_separator + obs_ids
                if len(response_mask) + len(obs_ids) >= self.response_length:
                    break
                running_ids = running_ids + obs_ids
                response_mask += [0] * len(obs_ids)
                response_logprobs += [0.0] * len(obs_ids)
                response_topk += [[] for _ in obs_ids]
        finally:
            pool.release(handle)

        # Split running sequence into prompt / response (mirror ToolAgentLoop).
        resp_len = len(response_mask)
        if resp_len == 0:
            # Such a row enters the batch fully masked, which downstream shows up as a
            # micro-batch with zero supervised tokens. The only path here is the very
            # first generate() returning no token, so log enough to tell whether the
            # prompt overran the model window or the model emitted EOS immediately.
            logger.warning(
                "[alfworld] degenerate episode: 0 assistant tokens (env_steps=%d, "
                "prompt_len=%d, split=%s, gamefile=%s)",
                num_env_steps,
                len(running_ids),
                split,
                gamefile,
            )
        response_ids = running_ids[-resp_len:] if resp_len else []
        prompt_ids = running_ids[: len(running_ids) - resp_len]

        metrics.setdefault("num_preempted", -1)
        # Diagnostics go under ``reward_extra_info``: that is the only nested key the
        # trainer expands into validation metrics. It is safe to set here because the
        # async reward loop only overwrites it when ``reward_score`` is None, and we
        # always set one. Under pure OPD from an ALFWorld-naive teacher, success rate
        # sits near the floor, so these are the signals that actually discriminate.
        diagnostics: dict[str, Any] = {
            "alfworld_won": float(won),
            "alfworld_num_env_steps": float(num_env_steps),
            "alfworld_invalid_action_frac": float(invalid_actions) / max(1, num_env_steps),
            # Response budget usage: a mean near response_length means episodes are
            # being cut off mid-task, a min of 0 means degenerate rows reached the batch.
            "alfworld_resp_len": float(resp_len),
            # Episode seed == the game index. Dumped so that "how many distinct games
            # did this batch actually play" is checkable against distinct gamefiles.
            "alfworld_seed": float(seed),
            # Redundant-``</think>`` probe (see _count_leading_close_think). Baseline
            # for this protocol is ~1 on the first turn and 0 afterwards; the first
            # run drifted to a steady 2 after its collapse. ``_max`` is what separates
            # a benign excursion (3, seen at steps 44-45 with no reward impact) from
            # the runaway that costs the episode (8+).
            "alfworld_lead_close_think_first": float(lead_close_first),
            "alfworld_lead_close_think_max": float(lead_close_max),
            "alfworld_lead_close_think_per_turn": float(lead_close_total) / max(1, turns_generated),
            # Fraction of generated turns that never emitted an action block. Tracks
            # the collapse one-for-one and, unlike the reward, says *why*.
            "alfworld_turn_no_action_frac": float(turns_no_action) / max(1, turns_generated),
        }
        extra_fields: dict[str, Any] = {
            "alfworld_split": split,
            "turn_scores": [],
            "tool_rewards": [],
            "reward_extra_info": diagnostics,
            # Replay tape for TB-OPD-Turn branches: the exact projected actions the
            # env executed, in order. Without it a branch cannot rebuild env state.
            "alfworld_actions": env_actions,
            # Per-step observation fingerprints, so a branch can verify that replaying
            # those actions really landed in the same state its token prefix describes.
            "alfworld_obs_fingerprints": obs_fingerprints,
            **diagnostics,
        }
        if self.record_action_spans:
            # Response-coordinate <action> spans, one entry per assistant turn, so
            # fork selection can distinguish the reasoning span from the action span.
            # Prefix turns of a resumed episode were not re-decoded, so they are None
            # placeholders that keep the list aligned with segment_assistant_turns().
            prefix_turns = _count_assistant_turns(prefix_response_mask) if resumed else 0
            extra_fields["action_spans"] = [None] * prefix_turns + action_spans
        if resumed:
            extra_fields["alfworld_replay_ok"] = float(replay_ok)
            extra_fields["alfworld_replay_steps"] = float(len(replay_actions or []))
            if not replay_ok:
                logger.warning(
                    "[alfworld] branch replay diverged (seed=%d, split=%s, replayed=%d/%d); "
                    "branch env state does not match its token prefix",
                    seed,
                    split,
                    num_env_steps,
                    len(replay_actions or []),
                )
        if have_topk:
            # Aligned 1:1 with response_ids (observation positions carry empty rows).
            # TB-OPD pops this before the batch is materialized -- storing a per-token
            # top-k distribution for a 20k-token episode would dwarf the episode itself.
            extra_fields["output_logprobs"] = response_topk[: self.response_length]
        if gamefile is not None:
            extra_fields["alfworld_gamefile"] = str(gamefile)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if have_logprobs else None,
            num_turns=assistant_turns * 2 + 1,
            reward_score=float(won),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
