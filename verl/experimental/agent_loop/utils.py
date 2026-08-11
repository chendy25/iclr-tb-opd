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

import os
import re
from typing import Any, Optional

# First complete final answer marker: a balanced ``\boxed{...}`` or an
# ``Answer:`` line. ``_ANSWER_LINE_RE`` matches the value that follows so we can
# cut at the end of that line.
_ANSWER_LINE_RE = re.compile(r"(?im)^[ \t]*Answer[ \t]*:[ \t]*\S[^\n]*")


def _first_boxed_end(text: str) -> Optional[int]:
    """Return the char offset just past the first balanced ``\\boxed{...}``.

    Returns ``None`` if there is no complete (brace-balanced) boxed expression.
    """
    start = text.find("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None  # unterminated boxed -> treat as no complete answer


def _first_answer_end_char(text: str) -> Optional[int]:
    """Char offset of the end of the first complete final answer in ``text``.

    Prefers whichever of ``\\boxed{...}`` / ``Answer:`` line ends earlier. Returns
    ``None`` when neither a balanced boxed nor an ``Answer:`` line is present.
    """
    ends: list[int] = []
    boxed_end = _first_boxed_end(text)
    if boxed_end is not None:
        ends.append(boxed_end)
    m = _ANSWER_LINE_RE.search(text)
    if m is not None:
        ends.append(m.end())
    if not ends:
        return None
    return min(ends)


def keep_len_after_final_answer(tokenizer, response_ids: list[int]) -> Optional[int]:
    """Number of leading response tokens to keep so the loss ignores post-answer text.

    Decodes ``response_ids`` and locates the first complete final answer
    (``\\boxed{...}`` or an ``Answer:`` line). Returns the smallest token count
    ``k`` whose decoded prefix already contains that answer, so callers can zero
    the response mask beyond ``k`` and drop trailing repetition. Returns ``None``
    when no complete answer is found (leave the mask untouched).
    """
    if not response_ids:
        return None
    text = tokenizer.decode(response_ids)
    char_end = _first_answer_end_char(text)
    if char_end is None:
        return None
    # Monotone: len(decode(ids[:k])) is non-decreasing in k, so binary-search the
    # smallest k whose decoded prefix already covers ``char_end``.
    lo, hi = 1, len(response_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(response_ids[:mid])) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return lo


def resolve_config_path(config_path: str) -> str:
    """Resolve agent loop configuration file path.

    In multi-node Ray training, relative paths may not resolve correctly
    because the working directory on remote nodes can differ from the driver node.
    This function resolves relative paths by checking multiple locations in order:
    1. If already absolute, return as-is
    2. Try current working directory
    3. Try relative to verl package installation (project root)

    Args:
        config_path: Configuration file path (relative or absolute)

    Returns:
        Absolute path to the configuration file

    Raises:
        FileNotFoundError: If the configuration file cannot be found
    """
    # Return absolute paths unchanged
    if os.path.isabs(config_path):
        return config_path

    # Try current working directory first
    cwd = os.path.abspath(os.getcwd())
    cwd_path = os.path.abspath(os.path.join(cwd, config_path))
    if (cwd_path == cwd or cwd_path.startswith(cwd + os.sep)) and os.path.exists(cwd_path):
        return cwd_path

    # Try relative to verl project root (where verl package is installed)
    try:
        import verl

        verl_package_dir = os.path.abspath(os.path.dirname(verl.__file__))

        # Strategy 1: For development/editable installs.
        project_root = os.path.dirname(verl_package_dir)
        dev_path = os.path.abspath(os.path.join(project_root, config_path))
        if (dev_path == project_root or dev_path.startswith(project_root + os.sep)) and os.path.exists(dev_path):
            return dev_path

        # Strategy 2: For standard package installations.
        install_path = os.path.abspath(os.path.join(verl_package_dir, config_path))
        if (install_path == verl_package_dir or install_path.startswith(verl_package_dir + os.sep)) and os.path.exists(
            install_path
        ):
            return install_path
    except (ImportError, AttributeError):
        pass  # verl not installed or __file__ not available

    # File not found - raise clear error
    raise FileNotFoundError(
        f"Agent loop configuration file not found: {config_path}. Tried current directory and verl project root."
    )


# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"


def build_gpt_oss_tool_response_text(messages: list[dict[str, Any]], tool_call_names: list[str]) -> str:
    """Build gpt-oss tool response text (manual formatting + generation prompt)."""
    tool_response_texts: list[str] = []
    for i, tool_msg in enumerate(messages):
        actual_tool_name = tool_call_names[i]
        formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
        tool_response_texts.append(formatted)
    return add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
