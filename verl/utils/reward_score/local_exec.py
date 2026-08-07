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
"""Local-subprocess drop-in replacement for SandboxFusion's ``call_sandbox_api``.

The agentic TB-OPD reward path scores code via SOD's LiveCodeBench
``code_math.compute_score`` -> ``sandbox_fusion.utils.check_correctness`` ->
``call_sandbox_api``. SOD ran this on a SandboxFusion HTTP service; we later
proxied it through an E2B sandbox, but E2B's network layer stalls/times out
constantly and -- because the reward runs in a separate ``RewardLoopWorker`` not
covered by the rollout ``trajectory_timeout`` -- a single hung sandbox wedges the
whole prompt group indefinitely.

Both the reward path and the rollout ``code_interpreter`` tool run **stateless**
code (a fresh ``python3`` process per call, no cross-call state), so a plain
local subprocess reproduces the exact same behaviour with *zero* network: no VM
spin-up, no proxy, no AGS quota, and a hard wall-clock ``timeout`` that actually
fires. We return the *identical* ``api_response`` shape SandboxFusion/E2B do, so
``_process_single_case``'s harness / output-comparison / status logic stays
byte-for-byte the same (only the executor changes).

Isolation: each snippet runs as its own process in a fresh session
(``start_new_session=True`` so we can kill the whole process group on timeout),
under POSIX ``setrlimit`` caps (CPU time, address space, file size, #procs) to
contain runaway / fork-bomb / OOM code. This mirrors what PRIME / DeepScaler and
most code-RL reward pipelines do when scoring model-generated competitive-
programming solutions locally.

Wiring: ``sandbox_fusion.utils.call_sandbox_api`` delegates here when the
resolved backend is ``local`` (the default when no real HTTP SandboxFusion URL
is configured).
"""

import logging
import os
import subprocess
import tempfile
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # POSIX-only; used inside the child via preexec_fn.
    import resource  # type: ignore
except Exception:  # noqa: BLE001 -- non-POSIX; rlimits simply won't be applied.
    resource = None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Bound how many child processes run at once so a 64-trajectory step scoring many
# cases each cannot fork thousands of interpreters and melt the reward node.
# Reuse the E2B knob if set so existing launch scripts keep working.
_DEFAULT_CONCURRENCY = _int_env(
    "LOCAL_EXEC_MAX_CONCURRENT",
    _int_env("E2B_REWARD_MAX_CONCURRENT", min(32, (os.cpu_count() or 8))),
)
_PYTHON_BIN = os.getenv("LOCAL_EXEC_PYTHON_BIN", os.getenv("E2B_REWARD_PYTHON_BIN", "python3"))
# Cap captured stream size so a chatty/looping program can't balloon reward-worker
# memory before its timeout fires.
_MAX_CAPTURE_CHARS = _int_env("LOCAL_EXEC_MAX_OUTPUT_CHARS", 200_000)
# Hard ceiling on #procs the child user may spawn (fork-bomb guard). 0 disables.
_NPROC_LIMIT = _int_env("LOCAL_EXEC_NPROC_LIMIT", 0)

_sem = threading.BoundedSemaphore(max(1, _DEFAULT_CONCURRENCY))


def _make_preexec(cpu_seconds: int, mem_bytes: int):
    if resource is None:
        return None

    def _limit() -> None:
        # NB: the new session/pgroup is created by Popen(start_new_session=True),
        # which runs setsid() in the child *before* this hook; calling setsid()
        # again here would raise EPERM (already a group leader) and abort the exec.
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
                cap = 64 * 1024 * 1024  # 64 MB of on-disk writes
                resource.setrlimit(rf, (cap, cap))
            except Exception:  # noqa: BLE001
                pass
        if _NPROC_LIMIT > 0:
            rp = getattr(resource, "RLIMIT_NPROC", None)
            if rp is not None:
                try:
                    resource.setrlimit(rp, (_NPROC_LIMIT, _NPROC_LIMIT))
                except Exception:  # noqa: BLE001
                    pass

    return _limit


def _truncate(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) > _MAX_CAPTURE_CHARS:
        return s[:_MAX_CAPTURE_CHARS] + "\n...[truncated]"
    return s


def _run_once(code: str, stdin: Optional[str], run_timeout: int, mem_bytes: int) -> dict[str, Any]:
    """Run ``code`` (python) once as a subprocess; return a SandboxFusion-shaped dict."""
    tag = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"_local_exec_{tag}_")
    sol = os.path.join(tmpdir, "sol.py")
    with open(sol, "w") as f:
        f.write(code)

    # CPU-time rlimit is a hard cap; give it a touch more than the wall timeout so
    # the wall-clock ``communicate(timeout=...)`` is the primary killer and the
    # rlimit only catches busy loops that ignore signals.
    cpu_seconds = max(1, run_timeout)
    preexec = _make_preexec(cpu_seconds + 2, mem_bytes)

    proc = subprocess.Popen(
        [_PYTHON_BIN, sol],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmpdir,
        start_new_session=True,  # new session/pgroup -> killpg on timeout
        preexec_fn=preexec,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=run_timeout)
        return_code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            stdout, stderr = "", ""
        return_code = None
    finally:
        _cleanup_dir(tmpdir)

    if timed_out:
        return {
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "TimeLimitExceeded",
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "return_code": None,
                "execution_time": run_timeout,
            },
        }

    api_status = "Success" if return_code == 0 else "Failed"
    return {
        "status": api_status,
        "compile_result": None,  # python: no separate compile stage
        "run_result": {
            "status": "Finished",
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr),
            "return_code": return_code,
            "execution_time": None,
        },
    }


def _kill_group(proc: "subprocess.Popen") -> None:
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 -- already dead / no pgroup
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _cleanup_dir(path: str) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def local_call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Local-subprocess drop-in for ``sandbox_fusion.utils.call_sandbox_api``.

    Returns ``(api_response, None)`` on a completed run (including wrong-answer /
    runtime-error / timeout, distinguished by ``run_result.status`` +
    ``return_code`` exactly like SandboxFusion), or ``(None, error_msg)`` on an
    executor failure so the case is scored as -1.
    """
    if language != "python":
        return None, f"local executor only supports python, got language={language}"

    mem_bytes = max(0, int(memory_limit_mb)) * 1024 * 1024
    try:
        with _sem:
            return _run_once(code, stdin, int(run_timeout), mem_bytes), None
    except Exception as e:  # noqa: BLE001
        logger.warning("local reward exec error: %s", e)
        return None, f"Local Exec Failed: {e}"
