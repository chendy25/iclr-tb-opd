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
"""Local-subprocess ``code_interpreter`` tool (drop-in for ``E2BTool``).

``E2BTool`` runs each snippet as "a fresh ``python3`` process (no cross-turn
Python state)" inside a remote E2B VM. That remote hop is the whole cost: VM
spin-up, a self-hosted proxy, AGS quota, and -- most painfully -- network stalls
that hang a trajectory (and thus the synchronous rollout step) far longer than
any single exec should take.

Because the execution is **stateless**, a plain local subprocess reproduces the
exact same semantics with none of that: run ``python3 file.py``, capture
stdout/stderr, enforce a hard wall-clock ``timeout`` (kill the whole process
group on expiry), and return the combined output as the tool observation. Forked
turn-level branches stay independent and deterministic, just as before.

Isolation matches the reward-side ``local_exec``: a fresh session per call
(``start_new_session=True``) under POSIX ``setrlimit`` caps (CPU, address space,
file size, #procs). This is standard practice for running model-generated code in
code-RL rollouts.

Select it via the tool yaml:
  actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/agentic_tbopd/config/local_tool_config.yaml
"""

import asyncio
import logging
import os
import re
import shutil
import signal
import tempfile
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

try:  # POSIX-only rlimits applied inside the child.
    import resource  # type: ignore
except Exception:  # noqa: BLE001
    resource = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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
    """Wrap a bare trailing expression in ``print(...)`` so it echoes to stdout."""
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


def _make_preexec(cpu_seconds: int, mem_bytes: int, nproc_limit: int):
    if resource is None:
        return None

    def _limit() -> None:
        # Session/pgroup already created by start_new_session=True before this
        # hook runs; a second setsid() here would raise EPERM and abort the exec.
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except Exception:  # noqa: BLE001
            pass
        if mem_bytes > 0:
            for lim in ("RLIMIT_AS", "RLIMIT_DATA"):
                rl = getattr(resource, lim, None)
                if rl is not None:
                    try:
                        resource.setrlimit(rl, (mem_bytes, mem_bytes))
                    except Exception:  # noqa: BLE001
                        pass
        rf = getattr(resource, "RLIMIT_FSIZE", None)
        if rf is not None:
            try:
                cap = 64 * 1024 * 1024
                resource.setrlimit(rf, (cap, cap))
            except Exception:  # noqa: BLE001
                pass
        if nproc_limit > 0:
            rp = getattr(resource, "RLIMIT_NPROC", None)
            if rp is not None:
                try:
                    resource.setrlimit(rp, (nproc_limit, nproc_limit))
                except Exception:  # noqa: BLE001
                    pass

    return _limit


class LocalPythonTool(BaseTool):
    """Execute Python code in a local subprocess (stateless per call).

    Config keys (all optional):

    - ``default_timeout`` (int): per-execution wall-clock timeout (default 30).
    - ``memory_limit_mb`` (int): address-space cap per exec (default 1024, 0=off).
    - ``max_concurrent`` (int): cap on concurrent subprocesses (0 = unlimited).
    - ``nproc_limit`` (int): RLIMIT_NPROC fork-bomb guard (0 = off).
    - ``max_output_chars`` (int): truncate captured output (default 200000).
    - ``strip_fences`` (bool): strip ```` ```python ```` fences (default True).
    - ``auto_print`` (bool): auto-print bare trailing expression (default True).
    - ``python_bin`` (str): interpreter to invoke (default ``python3``).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self.default_timeout = int(config.get("default_timeout", 30))
        self.memory_limit_mb = int(config.get("memory_limit_mb", 1024))
        self.nproc_limit = int(config.get("nproc_limit", 0))
        self.max_output_chars = int(config.get("max_output_chars", 200_000))
        self.strip_fences = bool(config.get("strip_fences", True))
        self.auto_print = bool(config.get("auto_print", True))
        self.python_bin = config.get("python_bin", "python3")

        max_concurrent = int(config.get("max_concurrent", 0))
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

        logger.info(
            "Init LocalPythonTool: python_bin=%s default_timeout=%d mem_mb=%d max_concurrent=%d",
            self.python_bin,
            self.default_timeout,
            self.memory_limit_mb,
            max_concurrent,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"ground_truth": ground_truth, "reward": []}
        return instance_id, ToolResponse()

    def _truncate(self, s: str) -> str:
        if not s:
            return ""
        if len(s) > self.max_output_chars:
            return s[: self.max_output_chars] + "\n...[truncated]"
        return s

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
            if self._sem is not None:
                async with self._sem:
                    output = await self._run(code, timeout)
            else:
                output = await self._run(code, timeout)
        except Exception as error:  # noqa: BLE001
            logger.warning("LocalPythonTool execute error instance=%s: %s", instance_id, error)
            output = f"[local python tool error] {type(error).__name__}: {str(error)[:500]}"
        return ToolResponse(text=output), None, None

    async def _run(self, code: str, timeout: int) -> str:
        tmpdir = tempfile.mkdtemp(prefix="_local_tool_")
        sol = os.path.join(tmpdir, f"_snippet_{uuid4().hex}.py")
        try:
            with open(sol, "w") as f:
                f.write(code)

            mem_bytes = max(0, self.memory_limit_mb) * 1024 * 1024
            preexec = _make_preexec(max(1, timeout) + 2, mem_bytes, self.nproc_limit)

            proc = await asyncio.create_subprocess_exec(
                self.python_bin,
                sol,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                start_new_session=True,
                preexec_fn=preexec,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                self._kill_group(proc)
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=5)
                except Exception:  # noqa: BLE001
                    pass
                return f"[local python tool] execution timed out after {timeout}s"

            stdout = (stdout_b or b"").decode("utf-8", errors="replace")
            stderr = (stderr_b or b"").decode("utf-8", errors="replace")
            out = (stdout + stderr).strip()
            return self._truncate(out) if out else "no stdout here"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _kill_group(proc: "asyncio.subprocess.Process") -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
