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
"""Code-execution tool backed by a SandboxFusion HTTP service.

Ported from the SOD recipe (``SOD/verl/tools/sandbox_fusion_tools.py``) so that
``ToolAgentLoop`` in this fork has a working ``code_interpreter`` tool for
tool-integrated reasoning (TIR) multi-turn rollouts -- the environment used by
Phase 1' of the agentic TB-OPD experiments. The in-tree
``SandboxFusionTool`` reference impl was removed upstream (see
``docs/sglang_multiturn/sandbox_fusion.rst``); this restores it against the
already-present HTTP client in ``verl.utils.reward_score.sandbox_fusion.utils``.
"""

import logging
import os
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray

from verl.tools.base_tool import BaseTool
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Global token-bucket rate limiter shared across all rollout workers."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # only used for observability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    """Ray actor that runs blocking sandbox HTTP calls under a global rate limit."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        # A singleton rate limiter shared by name across the cluster.
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            if self.rate_limit_worker:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    raise NotImplementedError("Process mode is not implemented yet")


class SandboxFusionTool(BaseTool):
    """Execute code in a remote SandboxFusion service.

    Expects a tool schema whose function name is ``code_interpreter`` and that
    accepts a ``code`` string argument. Config keys:

    - ``sandbox_fusion_url`` (required): HTTP endpoint, e.g. ``http://host:8080/run_code``.
    - ``num_workers`` (int): Ray execution-pool concurrency.
    - ``rate_limit`` / ``enable_global_rate_limit``: global token-bucket limit.
    - ``default_timeout`` (int): per-execution wall-clock timeout (seconds).
    - ``default_language`` (str): sandbox language, default ``python``.
    - ``memory_limit_mb`` (int): per-run memory cap.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        logger.info(f"Init SandboxFusionTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics -> return Nones
        return ToolResponse(text=result.text if isinstance(result, ToolResponse) else str(result)), None, None

    def execute_code(self, instance_id, code, timeout=30, language="python") -> ToolResponse:
        # iclr/verl's ``_process_single_case`` signature is positionally compatible
        # with SOD's call: (case_index, stdin, expected_output, url, generation,
        # timeout, memory_limit_mb, language).
        _result_status, metadata = _process_single_case(
            0, None, None, self.sandbox_fusion_url, code, timeout, self.memory_limit_mb, language
        )
        if metadata.get("run_status") == "Finished":
            actual_output = (metadata.get("stdout") or "") + (metadata.get("stderr") or "")
            logger.debug(f"sandbox fusion output for {instance_id}: {actual_output}")
            return ToolResponse(text=actual_output)
        return ToolResponse(text="no stdout here")

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class CustomSandboxFusionTool(SandboxFusionTool):
    """TIR-friendly variant: strip markdown fences and auto-print the last expr.

    Small models frequently emit their code wrapped in ```python ... ``` fences
    and rely on the REPL to echo the final expression. A plain ``exec`` produces
    no stdout in those cases, so the tool feedback is empty and the trajectory
    stalls. This mirrors ``SOD/recipe/demystify/reward.py::CustomSandboxFusionTool``.
    """

    _code_pattern = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        matches = self._code_pattern.findall(code)
        if matches:
            code = matches[0].strip()
        code = self._auto_print_last_line(code)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        return ToolResponse(text=result.text if isinstance(result, ToolResponse) else str(result)), None, None

    @staticmethod
    def _auto_print_last_line(code: str) -> str:
        lines = code.rstrip().split("\n")
        if not lines:
            return code
        last = lines[-1]
        stripped = last.strip()
        # Only wrap a bare expression: no leading indent, not already a statement.
        if (
            stripped
            and last == stripped
            and not stripped.startswith(("print", "import", "from", "#"))
            and not re.match(r"^[\w\.\[\]]+\s*=", stripped)
            and not stripped.endswith(":")
            and not stripped.startswith(("def ", "class ", "for ", "while ", "if ", "elif ", "else", "try", "except", "with ", "return", "raise", "assert", "pass", "break", "continue"))
        ):
            lines[-1] = f"print({stripped})"
        return "\n".join(lines)
