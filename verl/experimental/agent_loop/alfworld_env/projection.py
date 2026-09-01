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
"""Parse an assistant turn into an ALFWorld env action.

Ported from ATOD ``env_package/alfworld/projection.py`` (single-sample form).
Validity is a well-formed ``<action>`` block and no Chinese characters.

ATOD also requires a ``<think>...</think>`` pair in the *generated* string.
That check is dropped: with ``enable_thinking=False`` Qwen3 pre-fills an empty
think into the prompt, and the model writes its reasoning *after* ``</think>``
without opening a new tag. Checking the generated tokens for ``<think>`` would
pin ``valid`` to 0 on every turn (measured ``invalid_action_frac=1.0``) while
the extracted ``<action>`` is still well-formed.
"""

import re


def alfworld_projection(text_action: str) -> tuple[str, int]:
    """Extract an ALFWorld action string and its validity from one assistant turn.

    Args:
        text_action: Raw assistant response text.

    Returns:
        (action, valid): ``action`` is the lowercased command to send to the env
        (a best-effort tail slice when no valid ``<action>`` block is found);
        ``valid`` is 1 when the turn has a well-formed ``<action>`` block and
        no Chinese characters; else 0. The ATOD ``<think>`` check is omitted
        (see module docstring).
    """
    original_str = text_action
    lowered = text_action.lower()

    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)

    valid = 0
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        action = lowered[-30:]
    else:
        action = lowered[start_idx + len(start_tag) : end_idx].strip().lower()
        valid = 1

    # Reject Chinese characters (matches ATOD).
    if re.search(r"[\u4e00-\u9fff]", original_str):
        valid = 0

    return action, valid
