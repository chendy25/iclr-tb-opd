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
"""E2B-backed drop-in replacement for SandboxFusion's ``call_sandbox_api``.

The agentic TB-OPD reward path scores code via SOD's LiveCodeBench
``code_math.compute_score`` -> ``sandbox_fusion.utils.check_correctness`` ->
``call_sandbox_api``. SOD runs the code on a SandboxFusion HTTP service; we have
no such service here, so this module runs the exact same generated code on an
**E2B** sandbox instead and returns the *identical* ``api_response`` shape so
``_process_single_case``'s harness / output-comparison / status logic stays byte
-for-byte the same as SOD (only the executor changes).

Wiring: set ``SANDBOX_EXEC_BACKEND=e2b`` (plus ``E2B_API_KEY`` / ``E2B_DOMAIN``).
``sandbox_fusion.utils.call_sandbox_api`` then delegates here.

Concurrency: ``check_correctness`` fans every test case out over a large thread
pool. Creating one E2B VM per case would melt the AGS quota, so we keep a small
bounded pool of reusable **synchronous** ``Sandbox`` objects (size
``E2B_REWARD_MAX_CONCURRENT``, default 8). A borrower holds one sandbox
exclusively for one execution, runs the snippet as a fresh ``python3`` process
(stdin/stdout/stderr/exit-code via file redirection), then returns it. Dead
sandboxes are recreated transparently.
"""

import atexit
import logging
import os
import queue
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from e2b import Sandbox  # sync client; mirrors tools.e2b_tool's AsyncSandbox
except Exception:  # noqa: BLE001
    Sandbox = None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_MAX_CONCURRENT = _int_env("E2B_REWARD_MAX_CONCURRENT", 8)
_SANDBOX_LIFETIME = _int_env("E2B_REWARD_SANDBOX_TIMEOUT", 600)
_CREATE_RETRIES = max(1, _int_env("E2B_REWARD_CREATE_RETRIES", 3))
_PYTHON_BIN = os.getenv("E2B_REWARD_PYTHON_BIN", "python3")


def _template() -> Optional[str]:
    # Match the code_interpreter tool template unless overridden.
    return os.getenv("E2B_REWARD_TEMPLATE") or os.getenv("E2B_TEMPLATE") or "node-python-openclaw"


class _SandboxPool:
    """Bounded pool of reusable synchronous E2B sandboxes."""

    def __init__(self, max_size: int):
        self._max = max(1, max_size)
        self._idle: "queue.Queue[Any]" = queue.Queue()
        self._created = 0
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _create(self) -> Any:
        if Sandbox is None:
            raise RuntimeError("e2b package not importable (pip install e2b) for reward-side code exec")
        create_kwargs: dict[str, Any] = {"timeout": _SANDBOX_LIFETIME}
        tmpl = _template()
        if tmpl:
            create_kwargs["template"] = tmpl
        api_key = os.getenv("E2B_API_KEY")
        if api_key:
            create_kwargs["api_key"] = api_key
        domain = os.getenv("E2B_DOMAIN")
        if domain:
            create_kwargs["domain"] = domain

        last_err: Optional[BaseException] = None
        for attempt in range(1, _CREATE_RETRIES + 1):
            try:
                if hasattr(Sandbox, "create"):
                    return Sandbox.create(**create_kwargs)
                return Sandbox(**create_kwargs)
            except Exception as err:  # noqa: BLE001
                last_err = err
                logger.warning("e2b reward sandbox create failed %d/%d: %s", attempt, _CREATE_RETRIES, err)
        raise RuntimeError(f"e2b reward sandbox create failed after {_CREATE_RETRIES}: {last_err}")

    def acquire(self) -> Any:
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self._max:
                self._created += 1
                make_new = True
            else:
                make_new = False
        if make_new:
            try:
                return self._create()
            except Exception:
                with self._lock:
                    self._created -= 1
                raise
        # Pool saturated: block for a returned sandbox.
        return self._idle.get()

    def release(self, sandbox: Any, healthy: bool = True) -> None:
        if sandbox is None:
            return
        if healthy:
            self._idle.put(sandbox)
            return
        # Drop the dead sandbox and free its slot so a fresh one can be made.
        with self._lock:
            self._created = max(0, self._created - 1)
        try:
            sandbox.kill()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        while True:
            try:
                sbx = self._idle.get_nowait()
            except queue.Empty:
                break
            try:
                sbx.kill()
            except Exception:  # noqa: BLE001
                pass


_POOL: Optional[_SandboxPool] = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> _SandboxPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = _SandboxPool(_MAX_CONCURRENT)
    return _POOL


def _read_file(sandbox: Any, path: str) -> str:
    try:
        data = sandbox.files.read(path)
    except Exception:  # noqa: BLE001 -- file may not exist (e.g. timeout before creation)
        return ""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    return data if isinstance(data, str) else str(data)


def _run_once(sandbox: Any, code: str, stdin: Optional[str], run_timeout: int) -> dict[str, Any]:
    """Run ``code`` (as python3) once in ``sandbox``; return SandboxFusion-shaped dict.

    Raises on infra/transport errors so the caller can recycle the sandbox and
    surface an ``error_msg`` (mapped to result_status -1 by ``_process_single_case``).
    """
    tag = uuid.uuid4().hex
    sol = f"/tmp/_e2b_sol_{tag}.py"
    out = f"/tmp/_e2b_out_{tag}.txt"
    err = f"/tmp/_e2b_err_{tag}.txt"
    rc = f"/tmp/_e2b_rc_{tag}.txt"
    sandbox.files.write(sol, code)

    redir_in = ""
    if stdin is not None:
        inp = f"/tmp/_e2b_in_{tag}.txt"
        sandbox.files.write(inp, stdin)
        redir_in = f" < {inp}"

    # e2b runs the command through a shell, so redirection + `;` work directly.
    cmd = f"{_PYTHON_BIN} {sol}{redir_in} > {out} 2> {err}; echo $? > {rc}"

    timed_out = False
    # Give the transport a little slack beyond the code's own run timeout.
    request_timeout = run_timeout + 15
    try:
        sandbox.commands.run(cmd, timeout=run_timeout + 5, request_timeout=request_timeout)
    except TypeError:
        # Older/newer e2b signatures may not accept request_timeout.
        try:
            sandbox.commands.run(cmd, timeout=run_timeout + 5)
        except Exception as e:  # noqa: BLE001
            timed_out = _looks_like_timeout(e)
            if not timed_out:
                raise
    except Exception as e:  # noqa: BLE001
        timed_out = _looks_like_timeout(e)
        if not timed_out:
            raise

    if timed_out:
        return {
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "TimeLimitExceeded",
                "stdout": _read_file(sandbox, out),
                "stderr": _read_file(sandbox, err),
                "return_code": None,
                "execution_time": run_timeout,
            },
        }

    stdout = _read_file(sandbox, out)
    stderr = _read_file(sandbox, err)
    rc_raw = _read_file(sandbox, rc).strip()
    try:
        return_code = int(rc_raw) if rc_raw != "" else 1
    except ValueError:
        return_code = 1

    api_status = "Success" if return_code == 0 else "Failed"
    return {
        "status": api_status,
        "compile_result": None,  # python: no separate compile stage
        "run_result": {
            "status": "Finished",
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "execution_time": None,
        },
    }


def _looks_like_timeout(err: BaseException) -> bool:
    name = type(err).__name__.lower()
    msg = str(err).lower()
    return "timeout" in name or "timeout" in msg or "timedout" in name


def e2b_call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """E2B drop-in for ``sandbox_fusion.utils.call_sandbox_api`` (same signature).

    Returns ``(api_response, None)`` on a completed run (including wrong-answer /
    runtime-error / timeout, distinguished by ``run_result.status`` +
    ``return_code`` exactly like SandboxFusion), or ``(None, error_msg)`` on an
    infra/transport failure so the case is scored as -1.
    """
    if language != "python":
        return None, f"e2b reward executor only supports python, got language={language}"

    pool = _get_pool()
    sandbox = None
    healthy = True
    try:
        sandbox = pool.acquire()
        return _run_once(sandbox, code, stdin, run_timeout), None
    except Exception as e:  # noqa: BLE001
        healthy = False  # recycle the sandbox; it may be in a bad state
        logger.warning("e2b reward exec error: %s", e)
        return None, f"E2B Call Failed: {e}"
    finally:
        if sandbox is not None:
            pool.release(sandbox, healthy=healthy)
