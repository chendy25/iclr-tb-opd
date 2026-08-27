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
        metrics: dict[str, Any] = {}
        seed = self._game_seed(kwargs)
        split = self.eval_split if validate else self.train_eval
        pool = get_env_pool(self.config_path, split, self.pool_size)

        handle = await pool.acquire()
        won = False
        num_env_steps = 0
        invalid_actions = 0
        gamefile = None
        try:
            obs = await pool.reset(handle, seed)
            gamefile = obs.get("gamefile")

            # Initial user turn: task + first observation.
            user_text = build_initial_user_text(obs["observation"], obs["admissible_commands"])
            running_ids = await self.apply_chat_template([{"role": "user", "content": user_text}])

            response_mask: list[int] = []
            response_logprobs: list[float] = []
            have_logprobs = False
            assistant_turns = 0

            turn_sampling = dict(sampling_params)
            turn_sampling["max_tokens"] = min(
                int(turn_sampling.get("max_tokens", self.max_turn_tokens) or self.max_turn_tokens),
                self.max_turn_tokens,
            )

            for _ in range(self.max_steps):
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
                running_ids = running_ids + assistant_ids
                response_mask += [1] * len(assistant_ids)
                if output.log_probs:
                    have_logprobs = True
                    response_logprobs += list(output.log_probs)
                else:
                    response_logprobs += [0.0] * len(assistant_ids)
                assistant_turns += 1

                assistant_text = self.tokenizer.decode(assistant_ids, skip_special_tokens=True)
                action, valid = alfworld_projection(assistant_text)
                if not valid:
                    invalid_actions += 1

                # ---- environment step ----
                with simple_timer("tool_calls", metrics):
                    step = await pool.step(handle, action)
                num_env_steps += 1
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
        }
        extra_fields: dict[str, Any] = {
            "alfworld_split": split,
            "turn_scores": [],
            "tool_rewards": [],
            "reward_extra_info": diagnostics,
            **diagnostics,
        }
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
