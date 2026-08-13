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
"""Native ALFWorld (TextWorld) environment support for verl's agent loop.

This is the in-repo replacement for the ATOD ``agent_system`` env stack: instead
of a separate env-manager + TrajectoryCollector, a single ALFWorld episode is one
``AgentLoopBase.run`` invocation (see ``alfworld_agent_loop.AlfWorldAgentLoop``),
so it plugs directly into verl's V1 agent-loop rollout + distillation (OPD).
"""

from .env_pool import AlfredEnvHandle, get_env_pool
from .projection import alfworld_projection
from .prompts import build_followup_user_text, build_initial_user_text

__all__ = [
    "AlfredEnvHandle",
    "get_env_pool",
    "alfworld_projection",
    "build_initial_user_text",
    "build_followup_user_text",
]
