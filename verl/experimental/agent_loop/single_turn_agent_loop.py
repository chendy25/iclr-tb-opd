# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.utils import compute_repetition_penalty, keep_len_after_final_answer
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self._rep_newline_ids: set[int] | None = None

    def _newline_token_ids(self) -> set[int]:
        """Single-token ids that decode to a bare newline run (for line stutter split)."""
        if self._rep_newline_ids is None:
            ids: set[int] = set()
            for s in ("\n", "\n\n", "\n\n\n"):
                try:
                    enc = self.tokenizer.encode(s, add_special_tokens=False)
                except Exception:
                    enc = []
                if len(enc) == 1:
                    ids.add(enc[0])
            self._rep_newline_ids = ids
        return self._rep_newline_ids

    def _maybe_rep_penalty(self, response_ids: list[int]) -> list[float] | None:
        """Per-token repetition penalty for advantage shaping, or None if disabled/clean."""
        cfg = self.rollout_config
        if not getattr(cfg, "rep_penalty_enable", False) or not response_ids:
            return None
        newline_ids = self._newline_token_ids() if getattr(cfg, "rep_penalty_line_enable", True) else None
        penalty = compute_repetition_penalty(
            response_ids,
            ngram_ns=tuple(getattr(cfg, "rep_penalty_ngram_ns", (1, 3, 5, 8))),
            min_repeat=int(getattr(cfg, "rep_penalty_min_repeat", 8)),
            min_line_repeat=int(getattr(cfg, "rep_penalty_min_line_repeat", 20)),
            newline_ids=newline_ids,
            lambda_body=float(getattr(cfg, "rep_penalty_lambda_body", 0.5)),
            lambda_entry=float(getattr(cfg, "rep_penalty_lambda_entry", 3.0)),
            mode=str(getattr(cfg, "rep_penalty_mode", "wall")),
            eos_id=self.tokenizer.eos_token_id,
            protect_tail_eos=True,
        )
        # When enabled, always attach a fixed-length vector (zeros if clean) so the
        # batch collation and the TransferQueue column stay uniform across samples
        # (mixing None and tensors would break torch.cat / the TQ schema).
        if penalty is None:
            return [0.0] * len(response_ids)
        return penalty

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        if use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                [],
                [] if output.log_probs else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        # Drop post-answer repetition from the loss: zero the response mask beyond
        # the first complete final answer. Generation is unchanged; only the mask
        # (and therefore the distillation / policy-gradient loss) is affected.
        if getattr(self.rollout_config, "mask_after_answer", False) and response_mask:
            keep = keep_len_after_final_answer(
                self.tokenizer,
                response_ids,
                eos_id=self.tokenizer.eos_token_id,
                post_answer_cap=int(getattr(self.rollout_config, "mask_after_answer_post_cap", 512)),
            )
            if keep is not None and 0 < keep < len(response_mask):
                response_mask = [m if i < keep else 0 for i, m in enumerate(response_mask)]

        response_ids = response_ids[: self.response_length]
        rep_penalty = self._maybe_rep_penalty(response_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            rep_penalty=rep_penalty,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
