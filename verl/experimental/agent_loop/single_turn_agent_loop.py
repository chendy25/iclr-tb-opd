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
from verl.experimental.agent_loop.utils import (
    _first_answer_end_char,
    compute_repetition_penalty,
    keep_len_after_final_answer,
)
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
            ngram_max_period=int(getattr(cfg, "rep_penalty_ngram_max", 64)),
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

    def _apply_learn_eos(
        self,
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
    ) -> tuple[list[int], list[int], list[float] | None, list[float]]:
        """Keep the FULL generation; mask the post-answer refrain out of the main loss
        and teach the model to stop with an EOS after the first answer.

        Unlike the previous behavior this does **not** truncate ``response_ids`` -- the
        entire rollout is preserved for scoring, logging and length metrics so any
        post-answer repetition stays visible. It only:

        1. Locates the first complete final answer (``\\boxed{}`` / ``Answer:``) and zeros
           ``response_mask`` beyond it (same as ``mask_after_answer``; a promptly-emitted
           EOS is folded into ``keep`` so the "answer then stop" signal stays in-loss).
        2. When the model did *not* stop on its own there (it ran into a post-answer
           refrain), overwrites the single first post-answer token with an EOS and marks
           it in ``eos_sft_mask`` so the loss adds a small auxiliary cross-entropy
           teaching ``p(EOS | pre-answer + answer)``. That position is already masked out
           of the main OPD/PG loss, so the injected EOS is that token's only signal, its
           context is exactly ``[pre-answer + answer]``, and the sequence length (hence
           the visible refrain) is unchanged.

        Returns ``(response_ids, response_mask, response_logprobs, eos_sft_mask)`` where
        ``eos_sft_mask`` is always a full-length vector (all zeros unless an EOS was
        injected) so the batch column stays uniform and observable.
        """
        n = len(response_ids)
        eos_sft_mask = [0.0] * n
        eos_id = self.tokenizer.eos_token_id
        keep = keep_len_after_final_answer(
            self.tokenizer,
            response_ids,
            eos_id=eos_id,
            post_answer_cap=int(getattr(self.rollout_config, "mask_after_answer_post_cap", 512)),
        )
        if keep is None or keep <= 0 or keep >= n:
            # No complete answer, or the answer (incl. a prompt EOS) already spans the
            # whole response (nothing after to mask/inject). Full generation is kept.
            return response_ids, response_mask, response_logprobs, eos_sft_mask
        # Drop the post-answer refrain from the main loss (keep pre-answer + answer).
        response_mask = [m if i < keep else 0 for i, m in enumerate(response_mask)]
        stopped = eos_id is not None and response_ids[keep - 1] == eos_id
        if not stopped and eos_id is not None:
            # Model ran on instead of stopping: inject a supervised EOS at the first
            # post-answer position (context == [pre-answer + answer]) and mark it for the
            # aux CE. Overwrite (not append) so the full length / visible refrain stays.
            response_ids = list(response_ids)
            response_ids[keep] = eos_id
            eos_sft_mask[keep] = 1.0
        return response_ids, response_mask, response_logprobs, eos_sft_mask

    async def _generate_early_stop(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images,
        audios,
        videos,
        mm_processor_kwargs,
        priority: int,
    ) -> tuple["TokenOutput", list[int], list[float] | None]:
        """Generate in chunks, stopping once a complete final answer appears.

        Continues generation ``early_stop_chunk_tokens`` at a time (feeding
        prompt + response-so-far, like the multi-turn tool loop), and after each chunk
        checks for a brace-balanced ``\\boxed{...}`` / ``Answer:`` line. When one is
        found the response is truncated just past it (plus ``early_stop_tail_tokens`` to
        keep the closing ``\\]`` / ``$``) and generation stops -- killing the post-answer
        refrain at the source. Also stops on a natural EOS / short chunk, or when the
        ``response_length`` budget is exhausted (a no-answer wall-hit, handled by
        ``mask_truncated_no_answer`` downstream). Returns the last chunk's ``TokenOutput``
        (for metadata) plus the accumulated response ids and logprobs.
        """
        eos_id = self.tokenizer.eos_token_id
        chunk = max(int(getattr(self.rollout_config, "early_stop_chunk_tokens", 2048)), 1)
        tail = max(int(getattr(self.rollout_config, "early_stop_tail_tokens", 4)), 0)
        total = self.response_length
        resp_ids: list[int] = []
        resp_logprobs: list[float] = []
        have_lp = False
        last_output = None
        while len(resp_ids) < total:
            budget = min(chunk, total - len(resp_ids))
            # One continuation sequence per chunk; cap this chunk's new tokens.
            sp = {**sampling_params, "max_tokens": budget, "n": 1}
            out: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + resp_ids,
                sampling_params=sp,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
            last_output = out
            new_ids = out.token_ids or []
            if not new_ids:
                break
            if out.log_probs:
                have_lp = True
                resp_logprobs += out.log_probs
            elif have_lp:
                resp_logprobs += [0.0] * len(new_ids)
            resp_ids += new_ids
            # Stop just past the first complete final answer (drop post-answer refrain).
            keep = keep_len_after_final_answer(self.tokenizer, resp_ids, eos_id=None)
            if keep is not None:
                keep = min(len(resp_ids), keep + tail)
                resp_ids = resp_ids[:keep]
                if have_lp:
                    resp_logprobs = resp_logprobs[:keep]
                break
            # Natural stop: EOS, or vLLM returned fewer tokens than requested.
            if len(new_ids) < budget or (eos_id is not None and new_ids[-1] == eos_id):
                break
        return last_output, resp_ids, (resp_logprobs if have_lp else None)

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
        early_stop = getattr(self.rollout_config, "early_stop_after_answer", False) and not use_continuous_token
        es_response_ids: list[int] | None = None
        es_response_logprobs: list[float] | None = None
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            if early_stop:
                output, es_response_ids, es_response_logprobs = await self._generate_early_stop(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    images=images,
                    audios=audios,
                    videos=videos,
                    mm_processor_kwargs=mm_processor_kwargs,
                    priority=priority,
                )
            else:
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
        elif early_stop:
            response_ids = es_response_ids
            response_mask = [1] * len(response_ids)
            response_logprobs = es_response_logprobs
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        # Drop unambiguously degenerate rollouts from the loss entirely: a sequence
        # that hit ``max_response_length`` (no terminal EOS) *and* never produced a
        # complete final answer (no ``\boxed{}`` / ``Answer:``) is a no-answer refrain
        # / wall-hit (already scored -1). Zero its whole response mask so it does not
        # pollute the token-mean gradient. Generation is unchanged.
        if getattr(self.rollout_config, "mask_truncated_no_answer", False) and response_mask:
            eos_id = self.tokenizer.eos_token_id
            hit_cap = len(response_ids) >= self.response_length
            no_eos = eos_id is None or not response_ids or response_ids[-1] != eos_id
            if hit_cap and no_eos and _first_answer_end_char(self.tokenizer.decode(response_ids)) is None:
                response_mask = [0] * len(response_mask)

        # Learn-EOS: keep the full generation, mask the post-answer refrain out of the
        # main loss, and (when the model did not stop on its own) inject a supervised EOS
        # at the first post-answer token so the loss teaches "answer then stop" in the
        # model itself. Supersedes mask_after_answer (which only masks) when enabled.
        eos_sft_mask: list[float] | None = None
        if getattr(self.rollout_config, "learn_eos_after_answer", False) and response_mask and any(response_mask):
            response_ids, response_mask, response_logprobs, eos_sft_mask = self._apply_learn_eos(
                response_ids, response_mask, response_logprobs
            )
        # Drop post-answer repetition from the loss: zero the response mask beyond
        # the first complete final answer. Generation is unchanged; only the mask
        # (and therefore the distillation / policy-gradient loss) is affected.
        elif getattr(self.rollout_config, "mask_after_answer", False) and response_mask and any(response_mask):
            keep = keep_len_after_final_answer(
                self.tokenizer,
                response_ids,
                eos_id=self.tokenizer.eos_token_id,
                post_answer_cap=int(getattr(self.rollout_config, "mask_after_answer_post_cap", 512)),
            )
            if keep is not None and 0 < keep < len(response_mask):
                response_mask = [m if i < keep else 0 for i, m in enumerate(response_mask)]

        response_ids = response_ids[: self.response_length]
        if eos_sft_mask is not None:
            eos_sft_mask = eos_sft_mask[: self.response_length]
        rep_penalty = self._maybe_rep_penalty(response_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            rep_penalty=rep_penalty,
            eos_sft_mask=eos_sft_mask,
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
