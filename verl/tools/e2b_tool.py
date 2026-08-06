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
"""Code-execution ``code_interpreter`` tool backed by an E2B sandbox.

This is a ``BaseTool`` so it plugs directly into ``ToolAgentLoop`` (the
multi-turn loop the agentic TB-OPD turn-level branching is built on), exactly
like ``sandbox_fusion_tools.CustomSandboxFusionTool``. The sandbox *backend* is
decoupled from the branching logic -- pick E2B or SandboxFusion via the
``tool_config_path`` yaml.

Only E2B's code-execution primitive is reused here (create sandbox / run code /
kill), mirroring ``opd_dev``'s ``recipe_custom/agent/runners/sandbox.py::
E2BSandbox``. The heavier opd_dev "runner + harness" stack is intentionally
*not* adopted: it delegates the whole trajectory to an in-sandbox harness and
returns only a reward, which would hide the per-turn token ids / logprobs /
prefix-resume hooks that turn-level branching needs.

One E2B sandbox is created lazily per trajectory (``instance_id``) on the first
tool call and killed on ``release``. Each ``execute`` runs the snippet as a
fresh ``python3`` process (no cross-turn Python state), which keeps forked
branches independent and deterministic.

Requires ``pip install e2b`` and either ``E2B_API_KEY`` in the environment or
``api_key`` in the tool config (``domain`` / ``E2B_DOMAIN`` for self-hosted
endpoints).
"""

import asyncio
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

try:
    from e2b import AsyncSandbox  # type: ignore[import-untyped]
except ImportError:
    AsyncSandbox = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logging.getLogger("e2b").setLevel(logging.WARNING)

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)
_ASSIGN = re.compile(r"^[\w\.\[\]]+\s*=")
_STATEMENT_STARTS = (
    "def ", "class ", "for ", "while ", "if ", "elif ", "else", "try",
    "except", "with ", "return", "raise", "assert", "pass", "break", "continue",
)


def _strip_fences(code: str) -> str:
    matches = _CODE_FENCE.findall(code)
    if matches:
        return matches[0].strip()
    return code


def _auto_print_last_line(code: str) -> str:
    """Wrap a bare trailing expression in ``print(...)`` so it echoes to stdout.

    Small models often rely on REPL echo; a plain ``python file.py`` prints
    nothing for a bare final expression, leaving empty tool feedback. Mirrors
    ``CustomSandboxFusionTool._auto_print_last_line``.
    """
    lines = code.rstrip().split("\n")
    if not lines:
        return code
    last = lines[-1]
    stripped = last.strip()
    if (
        stripped
        and last == stripped  # no indentation (top-level)
        and not stripped.startswith(("print", "import", "from", "#"))
        and not _ASSIGN.match(stripped)
        and not stripped.endswith(":")
        and not stripped.startswith(_STATEMENT_STARTS)
    ):
        lines[-1] = f"print({stripped})"
    return "\n".join(lines)


class E2BTool(BaseTool):
    """Execute Python code in a per-trajectory E2B sandbox.

    Config keys (all optional unless noted):

    - ``api_key``: E2B API key. Falls back to ``$E2B_API_KEY``.
    - ``domain``: E2B endpoint domain (self-hosted). Falls back to ``$E2B_DOMAIN``.
    - ``template``: sandbox template id/name. Defaults to the account default.
    - ``sandbox_timeout`` (int): sandbox lifetime in seconds (default 300).
    - ``default_timeout`` (int): per-execution command timeout (default 60).
    - ``max_concurrent_sandboxes`` (int): cap on concurrent sandbox creations
      (0 = unlimited, default 0).
    - ``create_max_retries`` (int): retries on sandbox create failure (default 3).
    - ``strip_fences`` (bool): strip ```` ```python ```` fences (default True).
    - ``auto_print`` (bool): auto-print bare trailing expression (default True).
    - ``python_bin`` (str): interpreter to invoke (default ``python3``).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self.api_key = config.get("api_key") or os.getenv("E2B_API_KEY")
        self.domain = config.get("domain") or os.getenv("E2B_DOMAIN")
        self.template = config.get("template")
        self.sandbox_timeout = int(config.get("sandbox_timeout", 300))
        self.default_timeout = int(config.get("default_timeout", 60))
        self.create_max_retries = max(1, int(config.get("create_max_retries", 3)))
        self.strip_fences = bool(config.get("strip_fences", True))
        self.auto_print = bool(config.get("auto_print", True))
        self.python_bin = config.get("python_bin", "python3")

        max_concurrent = int(config.get("max_concurrent_sandboxes", 0))
        self._create_sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

        if AsyncSandbox is None:
            logger.warning(
                "e2b package not importable at init; E2BTool will fail on first "
                "execute. Install with `pip install e2b`."
            )
        logger.info(
            "Init E2BTool: template=%s domain=%s sandbox_timeout=%d default_timeout=%d",
            self.template,
            self.domain,
            self.sandbox_timeout,
            self.default_timeout,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        # Sandbox is created lazily on first execute to avoid spinning up a
        # remote VM for trajectories that never call the tool.
        self._instance_dict[instance_id] = {
            "sandbox": None,
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    async def _ensure_sandbox(self, instance_id: str):
        if AsyncSandbox is None:
            raise RuntimeError(
                "e2b package is required for E2BTool (pip install e2b), and either "
                "E2B_API_KEY in the environment or api_key in the tool config."
            )
        entry = self._instance_dict.setdefault(instance_id, {"sandbox": None, "reward": []})
        if entry.get("sandbox") is not None:
            return entry["sandbox"]

        create_kwargs: dict[str, Any] = {
            "timeout": self.sandbox_timeout,
            "metadata": {"instance_id": instance_id},
        }
        if self.template:
            create_kwargs["template"] = self.template
        if self.api_key:
            create_kwargs["api_key"] = self.api_key
        if self.domain:
            create_kwargs["domain"] = self.domain

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.create_max_retries + 1):
            try:
                if self._create_sem is not None:
                    async with self._create_sem:
                        sandbox = await AsyncSandbox.create(**create_kwargs)
                else:
                    sandbox = await AsyncSandbox.create(**create_kwargs)
                entry["sandbox"] = sandbox
                logger.debug("E2B sandbox created id=%s instance=%s", getattr(sandbox, "sandbox_id", "?"), instance_id)
                return sandbox
            except Exception as error:  # noqa: BLE001
                last_error = error
                logger.warning(
                    "E2B create failed instance=%s attempt=%d/%d: %s",
                    instance_id,
                    attempt,
                    self.create_max_retries,
                    error,
                )
                if attempt < self.create_max_retries:
                    await asyncio.sleep(2.0 * attempt)
        raise RuntimeError(f"E2B sandbox create failed after {self.create_max_retries} attempts: {last_error}")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        if self.strip_fences:
            code = _strip_fences(code)
        if self.auto_print:
            code = _auto_print_last_line(code)
        timeout = int(parameters.get("timeout", self.default_timeout))

        try:
            output = await self._run_in_sandbox(instance_id, code, timeout)
        except Exception as error:  # noqa: BLE001
            logger.warning("E2B execute error instance=%s: %s", instance_id, error)
            output = f"[e2b tool error] {type(error).__name__}: {str(error)[:500]}"
        # No per-step score / metrics from the sandbox.
        return ToolResponse(text=output), None, None

    async def _run_in_sandbox(self, instance_id: str, code: str, timeout: int) -> str:
        sandbox = await self._ensure_sandbox(instance_id)
        path = f"/tmp/_e2b_{uuid4().hex}.py"
        await sandbox.files.write(path, code)
        cmd = f"{self.python_bin} {path}"
        try:
            result = await sandbox.commands.run(cmd, timeout=timeout)
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
        except Exception as error:  # e2b raises on non-zero exit; salvage its streams
            stdout = getattr(error, "stdout", "") or ""
            stderr = getattr(error, "stderr", "") or ""
            if not stdout and not stderr:
                raise
        out = (stdout + stderr).strip()
        return out if out else "no stdout here"

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if not entry:
            return
        sandbox = entry.get("sandbox")
        if sandbox is not None:
            try:
                await sandbox.kill(request_timeout=30)
            except Exception as error:  # noqa: BLE001
                logger.warning("E2B kill failed instance=%s: %s", instance_id, error)
