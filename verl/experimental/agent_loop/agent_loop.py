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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.skip import SkipManager
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt, initialize_turn_separator
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    branch_weight: Optional[torch.Tensor] = None
    """Padded per-token TB-OPD branch weight (Rao-Blackwell); 1.0 where unweighted."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        self.continuous_token_builder = None
        self.enable_continuous_token = False
        continuous_token_config = self.data_config.continuous_token
        if continuous_token_config.enable and self.processor is None:
            model_config = self.config.actor_rollout_ref.model
            self.continuous_token_builder = create_continuous_token_builder(
                self.tokenizer,
                model_family=continuous_token_config.model_family,
                model_path=model_config.path,
                tokenizer_name_or_path=model_config.tokenizer_path,
                chat_template_kwargs=self.apply_chat_template_kwargs,
            )
            self.enable_continuous_token = True
            # Continuous Token doesn't use the legacy removable system prompt.
            self.system_prompt = None
            # Continuous Token re-renders non-assistant turns from the full message list, so it does
            # not need the incremental turn separator.
            self.turn_separator = []
        else:
            if continuous_token_config.enable and self.processor is not None:
                logger.warning(
                    "Continuous Token is enabled but processor is set; falling back to legacy multimodal path."
                )
            processing_class = self.processor if self.processor is not None else self.tokenizer
            self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
            # Turn separator dropped when the model stops at the assistant close token; restored at
            # turn boundaries in ``ToolAgentLoop._handle_processing_tools_state``.
            self.turn_separator = initialize_turn_separator(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] = None,
    ) -> list[int]:
        """Build the initial prompt token ids with Continuous Token."""
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.build_initial_tokens(messages, tools=tools),
        )
        return self._cap_text_prompt_length(prompt_ids)

    async def ct_merge_non_assistant_msg(
        self,
        previous_messages: list[dict],
        updated_messages: list[dict],
        runtime_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        tools: list[dict] = None,
    ):
        """Merge appended non-assistant messages into runtime tokens and metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_non_assistant_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result, response_mask, response_logprobs
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    async def ct_merge_assistant_token(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        assistant_logprobs: Optional[list[float]] = None,
    ):
        """Merge assistant-generated tokens and align response metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_assistant_tokens(
                runtime_token_ids,
                assistant_token_ids,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            logger.warning(
                "Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.",
                len(prompt_ids),
                prompt_length,
            )
            return prompt_ids[-prompt_length:]
        return prompt_ids

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        # Mirror the response-side ``response_ids[:response_length]`` cap on the prompt side:
        # every prompt produced by the agent loop must fit in ``rollout.prompt_length`` so that
        # ``_pad_token_ids`` (and downstream ``torch.cat``) can rely on uniform shapes.
        # Multimodal prompts cannot be sliced here because placeholder tokens must remain
        # aligned 1:1 with ``multi_modal_inputs`` features, so we fail loudly instead.
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            if images or videos or audios:
                raise ValueError(
                    f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                    f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                    f"sequences corrupts vision/audio feature alignment, so this is treated "
                    f"as a configuration error. Reduce the multimodal input size "
                    f"(e.g. ``total_pixels`` / ``max_pixels`` / fps / number of frames) or "
                    f"increase ``rollout.prompt_length``."
                )
            prompt_ids = self._cap_text_prompt_length(prompt_ids)

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`, plus
                ``validate`` (bool) marking a validation rollout. Loops that hold
                split-specific state (e.g. an environment pool) should branch on it;
                others may ignore it.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def _get_tb_opd_cfg(config) -> dict:
    """Return the ``distillation.tb_opd`` sub-config as a plain mapping.

    Robust to the subtree being absent so that standard OPD is unaffected.
    """
    try:
        dist = config.get("distillation", None)
    except Exception:  # noqa: BLE001
        return {}
    if dist is None:
        return {}
    tb = dist.get("tb_opd", None) if hasattr(dist, "get") else None
    return tb if tb is not None else {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

        # Token-level branching OPD (TB-OPD). Defensive .get so absence of the
        # config subtree leaves standard OPD (B1) untouched.
        self.tb_opd_cfg = _get_tb_opd_cfg(self.config)
        self.tb_opd_enabled = bool(self.tb_opd_cfg.get("enable", False))

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)

        # TB-OPD: token-level branching. Never branches on validation (eval must
        # stay standard). All other paths are unchanged when tb_opd is disabled.
        if self.tb_opd_enabled and not validate:
            return await self._generate_sequences_tb_opd(batch)

        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=getattr(config, "repetition_penalty", 1.0),
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            # ``validate`` is passed separately rather than merged into ``kwargs``,
            # which is reused below as **kwargs alongside a positional ``validate``.
            output: AgentLoopOutput = await agent_loop.run(
                sampling_params, validate=trajectory["validate"], **kwargs
            )
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    def _make_agent_loop(self, agent_name: str):
        """Instantiate a concrete agent loop (mirrors ``_run_agent_loop``)."""
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered: {_agent_loop_registry.keys()}"
        )
        return hydra.utils.instantiate(
            config=_agent_loop_registry[agent_name],
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            tools=ToolListWrap(self.tools),
        )

    async def _generate_sequences_tb_opd(self, batch: DataProto) -> DataProto:
        """TB-OPD rollout: per-prompt fixed-slot fan-out (main + k branches).

        The batch reaching a worker is already repeated by ``rollout.n`` with
        ``interleave=True``, so rows sharing ``index`` are contiguous groups of
        exactly ``n``. Each group produces exactly ``n`` outputs (slot 0 = main,
        slots 1..k = branches or plain rollouts), preserving row count/order so
        the trainer's ``batch.repeat(n).union(gen)`` remains valid.
        """
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=getattr(config, "repetition_penalty", 1.0),
            logprobs=config.calculate_log_probs,
        )

        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(
                [config.agent.default_agent_loop] * len(batch), dtype=object
            )

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), False
        )

        # Group a prompt's contiguous rollouts. Prefer ``uid`` (guaranteed unique
        # per prompt by the trainer and shared across the n interleaved repeats);
        # ``index`` can collapse for datasets that don't populate extra_info.index.
        if "uid" in batch.non_tensor_batch:
            group_key = list(batch.non_tensor_batch["uid"])
        else:
            group_key = list(index)

        groups: list[list[int]] = []
        start = 0
        for i in range(1, len(group_key) + 1):
            if i == len(group_key) or group_key[i] != group_key[start]:
                groups.append(list(range(start, i)))
                start = i

        default_agent = config.agent.default_agent_loop

        def _row_kwargs(row: int) -> dict:
            # Drop internal keys; ``agent_name`` is consumed separately (keyword-only
            # in the standard path) and must not reach ``run`` / postprocess.
            return {
                k: v[row]
                for k, v in batch.non_tensor_batch.items()
                if k not in ("__do_sample__", "agent_name")
            }

        def _row_agent(row: int) -> str:
            if "agent_name" in batch.non_tensor_batch:
                return batch.non_tensor_batch["agent_name"][row]
            return default_agent

        group_tasks = [
            self._run_tb_opd_group(
                [(row, trajectory_info[row], _row_kwargs(row)) for row in group],
                _row_agent(group[0]),
                dict(sampling_params),
            )
            for group in groups
        ]
        group_outputs = await asyncio.gather(*group_tasks)

        outputs: list[Optional[_InternalAgentLoopOutput]] = [None] * len(batch)
        for group, gout in zip(groups, group_outputs, strict=True):
            for slot, row in enumerate(group):
                outputs[row] = gout[slot]
        assert all(o is not None for o in outputs), "TB-OPD produced a hole in the output batch"

        return self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=False)

    async def _run_tb_opd_group(
        self, rows: list[tuple[int, dict, dict]], agent_name: str, sampling_params: dict
    ) -> list[_InternalAgentLoopOutput]:
        """Coordinate one prompt's ``n`` slots into ``n`` post-processed outputs."""
        # Turn-level branching (TB-OPD-Turn) uses a dedicated multi-turn path.
        if str(self.tb_opd_cfg.get("fork_unit", "token")) == "turn":
            return await self._run_tb_opd_group_turn(rows, agent_name, sampling_params)

        from verl.experimental.agent_loop import tb_opd

        cfg = self.tb_opd_cfg
        only_fail = bool(cfg.get("only_fail", False))
        fork_metric = str(cfg.get("fork_metric", "entropy"))
        topk_logprobs = int(cfg.get("topk_logprobs", 20))
        branch_min_tokens = int(cfg.get("branch_min_tokens", 8))
        correct_threshold = float(cfg.get("correct_threshold", 1.0))
        fork_select = str(cfg.get("fork_select", "argmax"))
        fork_topk_positions = int(cfg.get("fork_topk_positions", 20))
        fork_skip_first = int(cfg.get("fork_skip_first", 1))
        fork_min_token_strip_len = int(cfg.get("fork_min_token_strip_len", 1))
        fork_min_entropy = float(cfg.get("fork_min_entropy", 0.0))
        fork_dedup_main = bool(cfg.get("fork_dedup_main", True))
        fork_token_filter = str(cfg.get("fork_token_filter", "math_aware"))
        # Scheme B: read fork candidates from the main rollout's own top-k logprobs
        # (requested via logprobs=k during generation) instead of a second forward.
        scheme_b = bool(cfg.get("scheme_b", False))
        scheme_b_validate = bool(cfg.get("scheme_b_validate", False))
        branch_mode = str(cfg.get("branch_mode", "forced_topk"))
        resample_temperature = float(cfg.get("resample_temperature", -1.0))
        dedup_shared_prefix = bool(cfg.get("dedup_shared_prefix", True))
        branch_weight_mode = str(cfg.get("branch_weight_mode", "rb"))
        branch_weight_temp = float(cfg.get("branch_weight_temp", 1.0))
        branch_weight_floor = float(cfg.get("branch_weight_floor", 0.0))

        # Cache the tokenizer's special-id set once for the CURE-style filter.
        special_ids = getattr(self, "_tb_opd_special_ids", None)
        if special_ids is None:
            special_ids = {int(x) for x in getattr(self.tokenizer, "all_special_ids", [])}
            self._tb_opd_special_ids = special_ids

        n_slots = len(rows)
        _, _, main_kwargs = rows[0]
        agent_loop = self._make_agent_loop(agent_name)

        # Slot 0: main trajectory (standard single-turn generation). Under Scheme B
        # request per-token top-k logprobs so fork selection can read candidates
        # from this pass alone (no second forward).
        main_sp = dict(sampling_params)
        if scheme_b:
            main_sp["logprobs"] = topk_logprobs
        main_out: AgentLoopOutput = await agent_loop.run(main_sp, **main_kwargs)

        # Reward gate.
        score, is_correct = tb_opd.score_solution(
            self.tokenizer, list(main_out.response_ids), main_kwargs, correct_threshold
        )

        # Consume (and strip) the rollout top-k so the large per-token distribution
        # is never stored/dumped into TransferQueue. Present only under Scheme B.
        out_lp = main_out.extra_fields.pop("output_logprobs", None)
        out_id = main_out.extra_fields.pop("output_ids", None)

        do_branch = (not only_fail) or (not is_correct)
        fork = None
        none_reason = "not_attempted"
        used_scheme_b = False
        if do_branch and n_slots > 1:
            used_scheme_b = scheme_b and out_lp is not None and out_id is not None
            if used_scheme_b:
                # Scheme B: no server round-trip; synchronous selection.
                fork = tb_opd.select_fork_from_topk(
                    list(main_out.response_ids),
                    out_lp,
                    out_id,
                    metric=fork_metric,
                    min_tokens=branch_min_tokens,
                    response_length=agent_loop.response_length,
                    tokenizer=self.tokenizer,
                    special_ids=special_ids,
                    skip_first=fork_skip_first,
                    min_token_strip_len=fork_min_token_strip_len,
                    min_entropy=fork_min_entropy,
                    select=fork_select,
                    topk_positions=fork_topk_positions,
                    dedup_main=fork_dedup_main,
                    filter_mode=fork_token_filter,
                )
            else:
                fork = await tb_opd.select_fork(
                    self.llm_client,
                    list(main_out.prompt_ids),
                    list(main_out.response_ids),
                    topk=topk_logprobs,
                    metric=fork_metric,
                    min_tokens=branch_min_tokens,
                    response_length=agent_loop.response_length,
                    tokenizer=self.tokenizer,
                    special_ids=special_ids,
                    skip_first=fork_skip_first,
                    min_token_strip_len=fork_min_token_strip_len,
                    min_entropy=fork_min_entropy,
                    select=fork_select,
                    topk_positions=fork_topk_positions,
                    dedup_main=fork_dedup_main,
                    filter_mode=fork_token_filter,
                )
            none_reason = "ok" if fork.get("pos") is not None else str(fork.get("none_reason", "unknown"))

            # Validation-only: also run the legacy second-forward selector and record
            # whether it agrees on the fork position. Guards the temperature /
            # logprobs_mode consistency caveat before trusting Scheme B; off by default.
            if used_scheme_b and scheme_b_validate:
                ref = await tb_opd.select_fork(
                    self.llm_client,
                    list(main_out.prompt_ids),
                    list(main_out.response_ids),
                    topk=topk_logprobs,
                    metric=fork_metric,
                    min_tokens=branch_min_tokens,
                    response_length=agent_loop.response_length,
                    tokenizer=self.tokenizer,
                    special_ids=special_ids,
                    skip_first=fork_skip_first,
                    min_token_strip_len=fork_min_token_strip_len,
                    min_entropy=fork_min_entropy,
                    select="argmax",  # deterministic for a meaningful comparison
                    topk_positions=fork_topk_positions,
                    dedup_main=fork_dedup_main,
                    filter_mode=fork_token_filter,
                )
                main_out.extra_fields["tb_opd_schemeb_pos_match"] = float(
                    ref.get("pos") is not None and ref.get("pos") == fork.get("pos")
                )

        # A usable fork needs a position; forced_topk also requires candidate tokens.
        has_fork = fork is not None and fork.get("pos") is not None
        if has_fork and branch_mode != "resample":
            has_fork = bool(fork.get("cand_token_ids"))
        # No valid fork -> degrade to plain rollouts for the extra slots.
        mode = "branch" if has_fork else "plain"

        # TB-OPD diagnostics recorded on the main slot's extra_fields.
        main_out.extra_fields["tb_opd_slot"] = 0
        main_out.extra_fields["tb_opd_is_fail"] = float(not is_correct)
        main_out.extra_fields["tb_opd_score"] = float(score)
        main_out.extra_fields["tb_opd_mode_branch"] = float(mode == "branch")
        main_out.extra_fields["tb_opd_branch_mode"] = branch_mode
        main_out.extra_fields["tb_opd_fork_pos"] = float(fork["pos"]) if has_fork else -1.0
        main_out.extra_fields["tb_opd_fork_score"] = float(fork["score"]) if has_fork else 0.0
        main_out.extra_fields["tb_opd_num_branch"] = float((n_slots - 1) if mode == "branch" else 0.0)
        # Fork-selection diagnostics (main slot only).
        main_out.extra_fields["tb_opd_fork_attempted"] = float(do_branch and n_slots > 1)
        main_out.extra_fields["tb_opd_fork_found"] = float(has_fork)
        main_out.extra_fields["tb_opd_none_reason"] = none_reason
        main_out.extra_fields["tb_opd_scheme_b"] = float(used_scheme_b)
        if has_fork:
            rl = max(1, int(agent_loop.response_length))
            main_out.extra_fields["tb_opd_fork_pos_frac"] = float(fork["pos"]) / rl

        raw_outputs: list[AgentLoopOutput] = [main_out]
        # (fork index, candidate index) each branch slot actually forced. Recorded rather
        # than re-derived: the candidate index wraps on the candidates that survived
        # dedup, so a formula would not always agree with what was generated.
        slot_assignments: list[tuple[int, int]] = []
        for slot in range(1, n_slots):
            if mode == "branch":
                if branch_mode == "resample":
                    branch_out = await self._tb_generate_branch_resample(
                        agent_loop,
                        main_out,
                        fork["pos"],
                        dict(sampling_params),
                        resample_temperature,
                    )
                else:
                    cands = fork["cand_token_ids"]
                    ci = (slot - 1) % len(cands)
                    cand_token = cands[ci]
                    slot_assignments.append((0, ci))
                    branch_out = await self._tb_generate_branch(
                        agent_loop, main_out, fork["pos"], int(cand_token), dict(sampling_params)
                    )
                branch_out.extra_fields["tb_opd_slot"] = slot
                branch_out.extra_fields["tb_opd_is_fail"] = float(not is_correct)
                branch_out.extra_fields["tb_opd_mode_branch"] = 1.0
                branch_out.extra_fields["tb_opd_branch_mode"] = branch_mode
                if dedup_shared_prefix:
                    # The branch re-emits main[:pos] verbatim. Left supervised, those
                    # tokens are trained once per slot, so a prompt's opening would carry
                    # (1+k) times the gradient of its tail purely because it was
                    # regenerated. The forked token at pos differs and stays supervised.
                    branch_out.response_mask = tb_opd.mask_shared_prefix(
                        list(branch_out.response_mask), int(fork["pos"])
                    )
                raw_outputs.append(branch_out)
            else:
                _, _, slot_kwargs = rows[slot]
                plain_out: AgentLoopOutput = await agent_loop.run(dict(sampling_params), **slot_kwargs)
                plain_out.extra_fields["tb_opd_slot"] = slot
                plain_out.extra_fields["tb_opd_mode_branch"] = 0.0
                raw_outputs.append(plain_out)

        # Rao-Blackwell weighting over the k+1 continuations of the single fork point.
        # Forcing top-k alternatives and averaging them uniformly is what makes
        # branch-OPD off-policy: a rank-5 token the student would emit ~1% of the time
        # would otherwise carry the same gradient mass as the token it actually chose.
        if branch_weight_mode == "rb" and mode == "branch" and branch_mode == "forced_topk":
            weights = tb_opd.multifork_branch_weights(
                [fork],
                slot_assignments,
                temperature=branch_weight_temp,
                floor=branch_weight_floor,
            )
            if weights is not None:
                for out, w in zip(raw_outputs, weights, strict=False):
                    out.extra_fields["branch_weight"] = [float(w)] * len(out.response_ids)
                    out.extra_fields["tb_opd_branch_weight"] = float(w)

        results: list[_InternalAgentLoopOutput] = []
        for (row, traj, kwargs), raw in zip(rows, raw_outputs, strict=True):
            results.append(await self._agent_loop_postprocess(raw, traj["validate"], **kwargs))
        return results

    async def _run_tb_opd_group_turn(
        self, rows: list[tuple[int, dict, dict]], agent_name: str, sampling_params: dict
    ) -> list[_InternalAgentLoopOutput]:
        """TB-OPD-Turn: fork at a high-uncertainty assistant turn and re-run the
        remaining tool loop for each branch (E4 breakpoint resume).

        Mirrors ``_run_tb_opd_group`` (fixed-slot fan-out: slot 0 = main, slots
        1..k = turn branches or plain rollouts) but operates on multi-turn tool-use
        trajectories. Fork selection uses per-turn student uncertainty / teacher
        disagreement (``tb_opd.select_fork_turn``); branches re-enter
        ``ToolAgentLoop.run_from_prefix``.
        """
        from verl.experimental.agent_loop import tb_opd

        cfg = self.tb_opd_cfg
        only_fail = bool(cfg.get("only_fail", False))
        fork_metric = str(cfg.get("fork_metric", "hybrid"))
        correct_threshold = float(cfg.get("correct_threshold", 1.0))
        branch_mode = str(cfg.get("branch_mode", "forced_topk"))
        topk_logprobs = int(cfg.get("topk_logprobs", 20))
        turn_first_k = int(cfg.get("turn_first_k", 16))
        only_post_tool = bool(cfg.get("turn_only_post_tool", False))
        turn_skip_first = int(cfg.get("turn_skip_first", 0))
        min_uncertainty = float(cfg.get("fork_min_entropy", 0.0))
        consec_penalty = bool(cfg.get("consecutive_high_entropy_penalty", False))
        consec_weight = float(cfg.get("consecutive_penalty_weight", 0.5))
        resample_temperature = float(cfg.get("resample_temperature", -1.0))
        eligibility = cfg.get("fork_eligibility", None)
        eligibility = None if eligibility is None else str(eligibility)
        fork_alpha = float(cfg.get("fork_alpha", 0.5))
        fork_fuse = str(cfg.get("fork_fuse", "blend"))
        fork_kl_window = int(cfg.get("fork_kl_window", 128))
        fork_normalize = str(cfg.get("fork_normalize", "rank"))
        disagreement_signed = bool(cfg.get("disagreement_signed", True))
        max_branches = int(cfg.get("max_branches_per_traj", 1))
        min_turn_gap = int(cfg.get("fork_min_turn_gap", 1))
        dedup_shared_prefix = bool(cfg.get("dedup_shared_prefix", True))
        branch_weight_mode = str(cfg.get("branch_weight_mode", "rb"))
        branch_weight_temp = float(cfg.get("branch_weight_temp", 1.0))
        branch_weight_floor = float(cfg.get("branch_weight_floor", 0.0))

        n_slots = len(rows)
        _, _, main_kwargs = rows[0]
        agent_loop = self._make_agent_loop(agent_name)

        # Slot 0: main multi-turn trajectory. Ask for per-position top-k (an integer,
        # not a bool) so the turn uncertainty is the same truncated entropy the token
        # path uses rather than the mean-NLL proxy -- the proxy is a different
        # statistic on a different scale, which the fork_min_entropy gate is sensitive
        # to. Loops that ignore it degrade to the proxy; the estimator that actually
        # ran is reported as tb_opd_fork_estimator.
        main_sp = dict(sampling_params)
        main_sp["logprobs"] = topk_logprobs
        main_out: AgentLoopOutput = await agent_loop.run(main_sp, **main_kwargs)

        # Consume (and strip) the rollout top-k so the per-token distribution is never
        # stored or dumped downstream. Mirrors the token path's Scheme B.
        main_topk = main_out.extra_fields.pop("output_logprobs", None)
        main_out.extra_fields.pop("output_ids", None)

        score, is_correct = tb_opd.score_trajectory(self.tokenizer, main_out, main_kwargs, correct_threshold)
        do_branch = (not only_fail) or (not is_correct)

        response_logprobs = list(main_out.response_logprobs) if main_out.response_logprobs else []
        fork = {"pos": None, "none_reason": "not_attempted"}
        if do_branch and n_slots > 1:
            if not response_logprobs:
                fork = {"pos": None, "none_reason": "no_response_logprobs"}
            else:
                # Entropy alone says where the student is *unsure*, which is not where
                # it is *wrong*. Pull the teacher forward for the metrics that use the
                # disagreement term; _compute_teacher_logprobs is idempotent, so this
                # only reorders work that post-processing would do anyway.
                teacher_lp = None
                need_teacher = fork_fuse in ("max", "union", "soft_or") or fork_alpha < 1.0
                if need_teacher:
                    await self._compute_teacher_logprobs(
                        main_out,
                        prompt_ids=main_out.prompt_ids,
                        response_ids=main_out.response_ids,
                        validate=False,
                        sample_kwargs=main_kwargs,
                    )
                    teacher_lp = self._tb_teacher_token_logprobs(main_out, len(main_out.prompt_ids))

                fork = tb_opd.select_fork_turn(
                    list(main_out.response_mask),
                    response_logprobs,
                    metric=fork_metric,
                    teacher_logprobs=teacher_lp,
                    turn_first_k=turn_first_k,
                    only_post_tool=only_post_tool,
                    skip_first_turns=turn_skip_first,
                    min_uncertainty=min_uncertainty,
                    consecutive_penalty=consec_penalty,
                    consecutive_penalty_weight=consec_weight,
                    eligibility=eligibility,
                    action_spans=main_out.extra_fields.get("action_spans"),
                    topk_logprobs=main_topk,
                    fork_alpha=fork_alpha,
                    fork_fuse=fork_fuse,
                    fork_kl_window=fork_kl_window,
                    disagreement_signed=disagreement_signed,
                    normalize=fork_normalize,
                    max_forks=max_branches,
                    min_turn_gap=min_turn_gap,
                )

        has_fork = fork.get("pos") is not None
        mode = "branch" if has_fork else "plain"

        # Budget B: the selector returns up to ``max_branches_per_traj`` fork points,
        # and the k-1 branch slots are dealt round-robin across them, so B>1 spends
        # the same slot budget on several turns instead of stacking on one.
        fork_points: list[dict] = fork.get("forks") or ([fork] if has_fork else [])
        fork_points = fork_points[: max(1, max_branches)] if fork_points else []

        # forced_topk: fetch alternative first tokens for each forked position once.
        # The candidates' logprobs come back with them so the k+1 continuations can be
        # Rao-Blackwell weighted instead of averaged uniformly (see branch_weight_mode).
        cand_tokens_per_fork: list[list[int]] = []
        if has_fork and branch_mode == "forced_topk":
            for fp in fork_points:
                pos = int(fp["pos"])
                main_tok = int(main_out.response_ids[pos]) if pos < len(main_out.response_ids) else None
                cands, cand_lps = await tb_opd.topk_candidates_at(
                    self.llm_client,
                    list(main_out.prompt_ids),
                    list(main_out.response_ids[:pos]),
                    topk=topk_logprobs,
                    dedup_token=main_tok,
                )
                cand_tokens_per_fork.append(cands)
                fp["cand_logprobs"] = cand_lps
                # The student's own logprob of the token it took at the fork -- slot 0's
                # entry in the RB weighting.
                fp["main_logprob"] = (
                    float(response_logprobs[pos]) if pos < len(response_logprobs) else None
                )
            if not any(cand_tokens_per_fork):
                # No usable alternative anywhere -> degrade to resample for the branches.
                branch_mode = "resample"
                cand_tokens_per_fork = []

        # Diagnostics on the main slot.
        main_out.extra_fields["tb_opd_slot"] = 0
        main_out.extra_fields["tb_opd_fork_unit"] = "turn"
        main_out.extra_fields["tb_opd_is_fail"] = float(not is_correct)
        main_out.extra_fields["tb_opd_score"] = float(score)
        main_out.extra_fields["tb_opd_mode_branch"] = float(mode == "branch")
        main_out.extra_fields["tb_opd_branch_mode"] = branch_mode
        main_out.extra_fields["tb_opd_fork_attempted"] = float(do_branch and n_slots > 1)
        main_out.extra_fields["tb_opd_fork_found"] = float(has_fork)
        main_out.extra_fields["tb_opd_none_reason"] = str(fork.get("none_reason", "ok" if has_fork else "unknown"))
        main_out.extra_fields["tb_opd_num_branch"] = float((n_slots - 1) if mode == "branch" else 0.0)
        if has_fork:
            main_out.extra_fields["tb_opd_fork_pos"] = float(fork["pos"])
            main_out.extra_fields["tb_opd_fork_turn"] = float(fork.get("turn_index", -1))
            main_out.extra_fields["tb_opd_fork_signal"] = float(fork.get("signal", 0.0))
            main_out.extra_fields["tb_opd_num_turns"] = float(fork.get("num_turns", 0))
            main_out.extra_fields["tb_opd_fork_kind"] = str(fork.get("kind", "turn_open"))
            main_out.extra_fields["tb_opd_fork_eligibility"] = str(fork.get("eligibility", ""))
            main_out.extra_fields["tb_opd_fork_used_teacher"] = float(fork.get("used_teacher", False))
            main_out.extra_fields["tb_opd_fork_estimator"] = str(fork.get("uncertainty_estimator", ""))
            main_out.extra_fields["tb_opd_num_fork_points"] = float(len(fork_points))

        raw_outputs: list[AgentLoopOutput] = [main_out]
        # (fork index, candidate index) each branch slot actually forced, recorded rather
        # than re-derived: the candidate index wraps on the candidates that survived
        # dedup, so a formula would not always agree with what was generated.
        slot_assignments: list[tuple[int, int]] = []
        for slot in range(1, n_slots):
            if mode == "branch":
                # Fork points cycle fastest so a budget of B spreads over B distinct
                # turns before any turn gets a second branch.
                fi = (slot - 1) % len(fork_points)
                fp = fork_points[fi]
                forced_token = None
                ci = 0
                if branch_mode == "forced_topk" and cand_tokens_per_fork:
                    cands = cand_tokens_per_fork[fi]
                    if cands:
                        # Slots sharing a fork point must force *different* tokens.
                        ci = ((slot - 1) // len(fork_points)) % len(cands)
                        forced_token = int(cands[ci])
                slot_assignments.append((fi, ci))
                branch_out = await self._tb_generate_branch_turn(
                    agent_loop,
                    main_out,
                    int(fp["pos"]),
                    forced_token,
                    dict(sampling_params),
                    main_kwargs,
                    resample_temperature,
                )
                branch_out.extra_fields["tb_opd_slot"] = slot
                branch_out.extra_fields["tb_opd_fork_unit"] = "turn"
                branch_out.extra_fields["tb_opd_is_fail"] = float(not is_correct)
                branch_out.extra_fields["tb_opd_mode_branch"] = 1.0
                branch_out.extra_fields["tb_opd_branch_mode"] = branch_mode
                branch_out.extra_fields["tb_opd_fork_pos"] = float(fp["pos"])
                branch_out.extra_fields["tb_opd_fork_turn"] = float(fp.get("turn_index", -1))
                branch_out.extra_fields["tb_opd_fork_kind"] = str(fp.get("kind", "turn_open"))
                if dedup_shared_prefix:
                    # The branch replays main[:pos] verbatim. Left supervised, those
                    # tokens are trained once per slot, so the early part of an episode
                    # would carry (1+k) times the gradient of the later part purely
                    # because it was regenerated. The forced token at pos differs from
                    # main's and stays supervised.
                    branch_out.response_mask = tb_opd.mask_shared_prefix(
                        list(branch_out.response_mask), int(fp["pos"])
                    )
                raw_outputs.append(branch_out)
            else:
                _, _, slot_kwargs = rows[slot]
                plain_out: AgentLoopOutput = await agent_loop.run(dict(sampling_params), **slot_kwargs)
                plain_out.extra_fields["tb_opd_slot"] = slot
                plain_out.extra_fields["tb_opd_fork_unit"] = "turn"
                plain_out.extra_fields["tb_opd_mode_branch"] = 0.0
                raw_outputs.append(plain_out)

        # Rao-Blackwell weighting over the k+1 continuations. Forcing the top-k
        # alternatives and averaging them uniformly is what makes branch-OPD off-policy:
        # a rank-5 token the student would emit ~1% of the time would otherwise carry the
        # same gradient mass as the token it actually chose.
        if branch_weight_mode == "rb" and mode == "branch" and branch_mode == "forced_topk":
            weights = tb_opd.multifork_branch_weights(
                fork_points,
                slot_assignments,
                temperature=branch_weight_temp,
                floor=branch_weight_floor,
            )
            if weights is not None:
                for out, w in zip(raw_outputs, weights, strict=False):
                    out.extra_fields["branch_weight"] = [float(w)] * len(out.response_ids)
                    out.extra_fields["tb_opd_branch_weight"] = float(w)

        results: list[_InternalAgentLoopOutput] = []
        for (row, traj, kwargs), raw in zip(rows, raw_outputs, strict=True):
            results.append(await self._agent_loop_postprocess(raw, traj["validate"], **kwargs))
        return results

    @staticmethod
    def _tb_teacher_token_logprobs(output: AgentLoopOutput, prompt_len: int) -> Optional[list[float]]:
        """Teacher logprob of each token the student actually emitted, per response position.

        Reads the raw (unpadded) ``(S, K)`` teacher tensors left on ``extra_fields`` by
        ``_compute_teacher_logprobs``; sequence index ``prompt_len + p`` holds the
        distribution that produced response token ``p``. With the ``k1`` loss ``K == 1``
        and the single column is the sampled token; the id lookup keeps this correct for
        top-k teacher modes too.
        """
        teacher_ids = output.extra_fields.get("teacher_ids")
        teacher_logprobs = output.extra_fields.get("teacher_logprobs")
        if teacher_ids is None or teacher_logprobs is None:
            return None
        total = int(teacher_logprobs.shape[0])
        out: list[float] = []
        for p, tok in enumerate(output.response_ids):
            idx = prompt_len + p
            if idx >= total:
                break
            row_ids, row_lp = teacher_ids[idx], teacher_logprobs[idx]
            if int(row_ids[0]) == int(tok):
                out.append(float(row_lp[0]))
                continue
            hit = (row_ids == int(tok)).nonzero()
            # Token outside the teacher's top-k: bound it by the least likely one kept.
            out.append(float(row_lp[int(hit[0][0])]) if hit.numel() else float(row_lp.min()))
        return out

    async def _tb_generate_branch_turn(
        self,
        agent_loop,
        main_out: AgentLoopOutput,
        fork_pos: int,
        forced_token: Optional[int],
        sampling_params: dict,
        row_kwargs: dict,
        resample_temperature: float,
    ) -> AgentLoopOutput:
        """Re-run the tool loop from the forked turn's shared prefix.

        forced_topk forces ``forced_token`` as the turn's first token; resample
        (``forced_token=None``) continues stochastically. Either way the branch runs
        the *remaining* multi-turn tool interaction to completion.
        """
        sp = dict(sampling_params)
        # Request per-token logprobs like the main slot so the branch's
        # response_logprobs cover the newly generated turns (not just the shared
        # prefix); otherwise rollout_log_probs are 0 past the fork point.
        sp["logprobs"] = True
        if forced_token is None and resample_temperature >= 0.0:
            sp["temperature"] = resample_temperature

        prefix_ids = list(main_out.response_ids[:fork_pos])
        prefix_mask = list(main_out.response_mask[:fork_pos])
        prefix_lp = (
            list(main_out.response_logprobs[:fork_pos]) if main_out.response_logprobs else None
        )

        branch_out: AgentLoopOutput = await agent_loop.run_from_prefix(
            sp,
            base_prompt_ids=list(main_out.prompt_ids),
            prefix_response_ids=prefix_ids,
            prefix_response_mask=prefix_mask,
            prefix_response_logprobs=prefix_lp,
            forced_first_token=forced_token,
            # Non-resumable environments (ALFWorld) rebuild their state by replaying
            # the main trajectory's recorded actions; loops that can resume directly
            # ignore this.
            prefix_extra_fields=main_out.extra_fields,
            **row_kwargs,
        )
        return branch_out

    async def _tb_generate_branch(
        self,
        agent_loop,
        main_out: AgentLoopOutput,
        fork_pos: int,
        cand_token: int,
        sampling_params: dict,
    ) -> AgentLoopOutput:
        """Force ``cand_token`` at ``fork_pos`` then continue-generate the branch."""
        prompt_ids = list(main_out.prompt_ids)
        resp_prefix = list(main_out.response_ids[:fork_pos]) + [cand_token]
        response_length = agent_loop.response_length
        remaining = max(1, response_length - len(resp_prefix))

        sp = dict(sampling_params)
        # logprobs flag is consumed by the server as a bool -> keep parity with main.
        sp["max_tokens"] = remaining
        out = await self.llm_client.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids + resp_prefix,
            sampling_params=sp,
        )
        continuation = list(out.token_ids)
        response_ids = (resp_prefix + continuation)[:response_length]

        response_logprobs = None
        if main_out.response_logprobs is not None:
            cont_lp = out.log_probs if out.log_probs is not None else [0.0] * len(continuation)
            response_logprobs = ([0.0] * len(resp_prefix) + list(cont_lp))[:response_length]

        # Preserve weight-version tags for v1 off-policy metrics (TQ tags read these).
        extra_fields: dict[str, Any] = {"turn_scores": [], "tool_rewards": []}
        min_gs = main_out.extra_fields.get("min_global_steps")
        max_gs = main_out.extra_fields.get("max_global_steps")
        out_extra = getattr(out, "extra_fields", None) or {}
        if out_extra.get("min_global_steps") is not None:
            min_gs = out_extra["min_global_steps"] if min_gs is None else min(min_gs, out_extra["min_global_steps"])
        if out_extra.get("max_global_steps") is not None:
            max_gs = out_extra["max_global_steps"] if max_gs is None else max(max_gs, out_extra["max_global_steps"])
        if min_gs is not None:
            extra_fields["min_global_steps"] = min_gs
        if max_gs is not None:
            extra_fields["max_global_steps"] = max_gs

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            multi_modal_data=main_out.multi_modal_data,
            mm_processor_kwargs=main_out.mm_processor_kwargs,
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields,
        )

    async def _tb_generate_branch_resample(
        self,
        agent_loop,
        main_out: AgentLoopOutput,
        fork_pos: int,
        sampling_params: dict,
        resample_temperature: float,
    ) -> AgentLoopOutput:
        """Continue-generate from the shared prefix at ``fork_pos`` (no forced token)."""
        prompt_ids = list(main_out.prompt_ids)
        resp_prefix = list(main_out.response_ids[:fork_pos])
        response_length = agent_loop.response_length
        remaining = max(1, response_length - len(resp_prefix))

        sp = dict(sampling_params)
        sp["max_tokens"] = remaining
        if resample_temperature >= 0.0:
            sp["temperature"] = resample_temperature

        out = await self.llm_client.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids + resp_prefix,
            sampling_params=sp,
        )
        continuation = list(out.token_ids)
        response_ids = (resp_prefix + continuation)[:response_length]

        response_logprobs = None
        if main_out.response_logprobs is not None:
            cont_lp = out.log_probs if out.log_probs is not None else [0.0] * len(continuation)
            response_logprobs = ([0.0] * len(resp_prefix) + list(cont_lp))[:response_length]

        extra_fields: dict[str, Any] = {"turn_scores": [], "tool_rewards": []}
        min_gs = main_out.extra_fields.get("min_global_steps")
        max_gs = main_out.extra_fields.get("max_global_steps")
        out_extra = getattr(out, "extra_fields", None) or {}
        if out_extra.get("min_global_steps") is not None:
            min_gs = out_extra["min_global_steps"] if min_gs is None else min(min_gs, out_extra["min_global_steps"])
        if out_extra.get("max_global_steps") is not None:
            max_gs = out_extra["max_global_steps"] if max_gs is None else max(max_gs, out_extra["max_global_steps"])
        if min_gs is not None:
            extra_fields["min_global_steps"] = min_gs
        if max_gs is not None:
            extra_fields["max_global_steps"] = max_gs

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            multi_modal_data=main_out.multi_modal_data,
            mm_processor_kwargs=main_out.mm_processor_kwargs,
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields,
        )

    def _pad_token_ids(
        self,
        tokens: list[int],
        *,
        max_length: int,
        padding_side: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        """Right/left pad a flat list of token ids to a ``(1, max_length)`` tensor."""
        # tokenizer.pad() with empty input returns dict with list values
        # instead of tensors, which breaks downstream .dim() calls.
        if not tokens:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            result = {"input_ids": torch.full((1, max_length), pad_id, dtype=torch.long)}
            if return_attention_mask:
                result["attention_mask"] = torch.zeros((1, max_length), dtype=torch.long)
            return result
        self.tokenizer.padding_side = padding_side
        padded = self.tokenizer.pad(
            {"input_ids": tokens},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )
        if padded["input_ids"].dim() == 1:
            padded["input_ids"] = padded["input_ids"].unsqueeze(0)
            if return_attention_mask:
                padded["attention_mask"] = padded["attention_mask"].unsqueeze(0)
        return padded

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        prompt_output = self._pad_token_ids(
            output.prompt_ids,
            max_length=self.rollout_config.prompt_length,
            padding_side="left",
            return_attention_mask=True,
        )

        response_output = self._pad_token_ids(
            output.response_ids,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=True,
        )

        response_mask_output = self._pad_token_ids(
            output.response_mask,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=False,
        )

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        # TB-OPD branch weight, per token so it survives the same padding/reordering as
        # the rest of the row. Pads with 1.0 rather than 0.0 -- it multiplies the loss, so
        # the neutral value is one; padding with zero would mute real tokens if the field
        # ever outlived its mask.
        branch_weight = None
        bw = output.extra_fields.pop("branch_weight", None)
        if bw is not None:
            bw = list(bw)[: self.rollout_config.response_length]
            pad_size = self.rollout_config.response_length - len(bw)
            branch_weight = torch.tensor(bw + [1.0] * pad_size, dtype=torch.float32).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            branch_weight=branch_weight,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        # text-only OR non-M-RoPE multimodal (e.g. Gemma4) -> standard 1D positions
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample.

        Idempotent: TB-OPD fork ranking needs the main trajectory's teacher logprobs
        *before* branching, so it calls this early; the guard keeps post-processing
        from issuing a second (identical) teacher forward.
        """
        if "teacher_logprobs" in output.extra_fields and "teacher_ids" in output.extra_fields:
            return
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        # Not keyed off inputs[0]: only forked groups carry a branch weight, so within one
        # batch some rows have it and some do not. The rows that do not are unweighted,
        # which is a weight of 1, so they are filled rather than dropping the column.
        if any(input.branch_weight is not None for input in inputs):
            optional_outputs["branch_weight"] = torch.cat(
                [
                    input.branch_weight
                    if input.branch_weight is not None
                    else torch.ones_like(input.response_ids, dtype=torch.float32)
                    for input in inputs
                ],
                dim=0,
            )
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        # TB-OPD: when enabled we must keep each prompt's n contiguous rows within
        # a single worker chunk so the group can be coordinated (main + branches).
        self.tb_opd_cfg = _get_tb_opd_cfg(config)
        self.tb_opd_enabled = bool(self.tb_opd_cfg.get("enable", False))

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    def _chunk_group_aware(self, prompts: DataProto):
        """Split into contiguous chunks whose boundaries align to ``rollout.n``.

        With ``interleave=True`` repetition, a prompt's ``n`` rollouts are
        contiguous. Aligning chunk boundaries to multiples of ``n`` guarantees a
        prompt group is never split across workers. Returns ``(workers, chunks)``
        with matching lengths (only as many workers as non-empty chunks).
        """
        n = int(self.rollout_config.n)
        total = len(prompts)
        num_workers = len(self.agent_loop_workers)
        if n <= 0 or total % n != 0:
            # Fall back to naive chunking; group coordination handles ragged groups.
            return self.agent_loop_workers, prompts.chunk(num_workers)

        num_groups = total // n
        num_chunks = min(num_workers, num_groups)
        base, rem = divmod(num_groups, num_chunks)
        chunks = []
        g0 = 0
        for c in range(num_chunks):
            gc = base + (1 if c < rem else 0)
            start = g0 * n
            end = (g0 + gc) * n
            chunks.append(prompts.slice(start, end))
            g0 += gc
        return self.agent_loop_workers[:num_chunks], chunks

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Attach per-sample priority to the batch (like ``uid``) so each sample gets
        # a globally-unique priority that flows to vLLM request scheduling. Assigned
        # before chunking so chunks own disjoint ranges without per-worker offsets.
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)

        validate = prompts.meta_info.get("validate", False)
        if self.tb_opd_enabled and not validate:
            workers, chunkes = self._chunk_group_aware(prompts)
        else:
            workers = self.agent_loop_workers
            chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
