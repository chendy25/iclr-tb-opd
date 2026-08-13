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
"""Per-process ALFWorld (TextWorld) environment pool for the agent loop.

The ATOD stack builds ``train_batch_size * rollout.n`` Ray actors, each holding one
TextWorld env, driven by a central env manager. In verl's agent-loop model each
episode is an independent coroutine, so we instead keep a per-process pool of
env handles and hand one out per episode:

  * A single ``base_env`` (``AlfredTWEnv``) is built once per process per split and
    scans the game files. ``base_env.init_env(batch_size=1)`` then cheaply yields
    independent single-game handles (mirrors ATOD ``AlfworldWorker``).
  * ``get_env_pool(...).acquire()`` returns a handle; the caller seeds it with a
    per-prompt game seed (``extra_info.index``) so a prompt's ``rollout.n`` replicas
    all play the *same* game (valid GRPO / OPD grouping), and different prompts play
    different games.
  * TextWorld calls are blocking C/Python; they run in a thread executor so many
    episodes progress concurrently on one event loop.

The heavy vendored ``alfworld`` package (TextWorld games, PDDL) is a runtime
dependency (``pip install alfworld``), not vendored into verl.
"""

import asyncio
import logging
import os
import threading
from typing import Any, Optional

import yaml

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config_tw.yaml")

# Per-process singletons keyed by (config_path, train_eval).
_BASE_ENVS: dict[tuple, Any] = {}
_BASE_LOCK = threading.Lock()
_POOLS: dict[tuple, "AlfredEnvPool"] = {}
_POOLS_LOCK = threading.Lock()


def _expand_vars(obj: Any) -> Any:
    """Recursively expand ``$ENV`` / ``${ENV}`` in config strings (e.g. ``$ALFWORLD_DATA``)."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(v) for v in obj]
    return obj


def load_alfworld_config(config_path: Optional[str] = None) -> dict:
    """Load and env-expand the ALFWorld TextWorld config."""
    path = config_path or _DEFAULT_CONFIG
    if not os.path.exists(path):
        raise FileNotFoundError(f"ALFWorld config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return _expand_vars(config)


def _get_base_env(config_path: str, train_eval: str):
    """Build (once per process) the shared ``AlfredTWEnv`` for a split."""
    key = (config_path, train_eval)
    with _BASE_LOCK:
        if key not in _BASE_ENVS:
            # Imported lazily so importing this module (e.g. for registration) does
            # not require the heavy alfworld package on the login node.
            from alfworld.agents.environment import get_environment

            config = load_alfworld_config(config_path)
            env_type = config["env"]["type"]  # AlfredTWEnv
            if env_type != "AlfredTWEnv":
                raise ValueError(
                    f"alfworld agent loop only supports AlfredTWEnv (text), got {env_type!r}"
                )
            logger.warning("[alfworld] building base env type=%s split=%s", env_type, train_eval)
            _BASE_ENVS[key] = get_environment(env_type)(config, train_eval=train_eval)
    return _BASE_ENVS[key]


def _extract0(infos: dict, key: str, default=None):
    """Batched TextWorld infos are dict[str, list]; return the first element."""
    val = infos.get(key, None)
    if val is None:
        return default
    try:
        return val[0]
    except (TypeError, IndexError):
        return val


class AlfredEnvHandle:
    """A single-game TextWorld env handle (batch_size=1). Blocking; call from executor."""

    def __init__(self, base_env):
        self.env = base_env.init_env(batch_size=1)

    def reset(self, seed: int) -> dict:
        """Seed then reset so a given ``seed`` deterministically selects a game."""
        try:
            self.env.seed(int(seed))
        except Exception as e:  # noqa: BLE001 - some backends seed lazily
            logger.warning("[alfworld] seed(%s) failed: %s", seed, e)
        obs, infos = self.env.reset()
        return {
            "observation": obs[0],
            "admissible_commands": _extract0(infos, "admissible_commands", []),
            "won": bool(_extract0(infos, "won", False)),
            "gamefile": _extract0(infos, "extra.gamefile", None),
        }

    def step(self, action: str) -> dict:
        obs, scores, dones, infos = self.env.step([action])
        return {
            "observation": obs[0],
            "admissible_commands": _extract0(infos, "admissible_commands", []),
            "won": bool(_extract0(infos, "won", False)),
            "done": bool(_extract0(dones if isinstance(dones, list) else [dones], 0, False)),
            "gamefile": _extract0(infos, "extra.gamefile", None),
        }


class AlfredEnvPool:
    """Async pool of ``AlfredEnvHandle`` for one process/split.

    Handles are created lazily up to ``max_size`` and reused across episodes. All
    blocking env calls are offloaded to a thread executor.
    """

    def __init__(self, config_path: str, train_eval: str, max_size: int):
        self.config_path = config_path
        self.train_eval = train_eval
        self.max_size = max(1, int(max_size))
        self._queue: "asyncio.Queue[AlfredEnvHandle]" = asyncio.Queue()
        self._created = 0
        self._grow_lock = asyncio.Lock()

    async def _make_handle(self) -> AlfredEnvHandle:
        loop = asyncio.get_event_loop()
        base_env = await loop.run_in_executor(None, _get_base_env, self.config_path, self.train_eval)
        return await loop.run_in_executor(None, AlfredEnvHandle, base_env)

    async def acquire(self) -> AlfredEnvHandle:
        # Reuse an idle handle if available.
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        # Otherwise grow the pool until max_size, then block for a free handle.
        async with self._grow_lock:
            if self._created < self.max_size:
                handle = await self._make_handle()
                self._created += 1
                return handle
        return await self._queue.get()

    def release(self, handle: AlfredEnvHandle) -> None:
        self._queue.put_nowait(handle)

    async def reset(self, handle: AlfredEnvHandle, seed: int) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handle.reset, seed)

    async def step(self, handle: AlfredEnvHandle, action: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handle.step, action)


def get_env_pool(config_path: Optional[str], train_eval: str, max_size: int) -> AlfredEnvPool:
    """Return the per-process pool for ``(config_path, train_eval)`` (created once)."""
    path = config_path or _DEFAULT_CONFIG
    key = (path, train_eval)
    with _POOLS_LOCK:
        if key not in _POOLS:
            _POOLS[key] = AlfredEnvPool(path, train_eval, max_size)
        return _POOLS[key]
