# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
"""ALFWorld prompt templates for the agent-loop rollout.

Unlike ATOD's per-turn prompt (which re-injects a short observation/action
history each turn because each turn is a separate training row), the agent loop
accumulates the full transcript, so the model already sees prior turns. We
therefore only render:
  - an instruction-bearing initial user turn (task + first observation), and
  - a compact follow-up user turn carrying the new observation each step.
Both keep the ATOD ``<think>``/``<action>`` answer contract so the same teacher /
projection / reward semantics apply.
"""

from typing import List

ALFWORLD_INITIAL_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

ALFWORLD_FOLLOWUP_TEMPLATE = """Your new observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step within <think> </think> tags, then present one admissible action within <action> </action> tags."""


def _format_admissible(admissible_actions: List[str]) -> str:
    """Render admissible commands the way ATOD does (drop ``help``, quote each)."""
    return "\n ".join(f"'{s}'" for s in admissible_actions if s != "help")


def build_initial_user_text(observation: str, admissible_actions: List[str]) -> str:
    """First user turn: instructions + task-bearing initial observation."""
    return ALFWORLD_INITIAL_TEMPLATE.format(
        current_observation=observation,
        admissible_actions=_format_admissible(admissible_actions),
    )


def build_followup_user_text(observation: str, admissible_actions: List[str]) -> str:
    """Subsequent user turns: new observation + admissible actions."""
    return ALFWORLD_FOLLOWUP_TEMPLATE.format(
        current_observation=observation,
        admissible_actions=_format_admissible(admissible_actions),
    )
