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


def _mark_ngram_runs(
    ids: list[int],
    covered: list[float],
    *,
    ngram_ns: tuple[int, ...],
    min_repeat: int,
    lambda_body: float,
    lambda_entry: float,
    mode: str,
) -> None:
    """Mark consecutive (period-``p``) n-gram runs in ``ids`` into ``covered``.

    A run is a maximal region where the ``p``-token block starting at ``i`` repeats
    ``>= min_repeat`` times back-to-back (period-``p`` repetition). This catches the
    observed stutter units: single-token spam, ``$$`` / ``\\[`` markers, repeated
    ``\\boxed{x}`` lines, ``Answer: 1`` loops -- all short repeating token n-grams,
    detected on ids without any decoding.
    """
    t = len(ids)
    for p in ngram_ns:
        if p <= 0 or p > t:
            continue
        i = 0
        while i + p <= t:
            block = ids[i : i + p]
            reps = 1
            j = i + p
            while j + p <= t and ids[j : j + p] == block:
                reps += 1
                j += p
            if reps >= min_repeat:
                # body of the repeated region gets lambda_body ...
                for k in range(i, j):
                    if lambda_body > covered[k]:
                        covered[k] = lambda_body
                # ... and the entry token an extra wall (mode="wall").
                if mode == "wall":
                    entry_w = lambda_body + lambda_entry
                    if entry_w > covered[i]:
                        covered[i] = entry_w
                i = j  # skip past the run we just consumed
            else:
                i += 1


def _mark_line_runs(
    ids: list[int],
    covered: list[float],
    newline_ids: set[int],
    *,
    min_line_repeat: int,
    lambda_body: float,
    lambda_entry: float,
    mode: str,
) -> None:
    """Mark newline-delimited *line* stutter (consecutive identical lines).

    Lines are delimited by ``newline_ids`` on the token stream -- no text decoding.
    A run of ``>= min_line_repeat`` identical consecutive line-id sequences is marked.
    """
    t = len(ids)
    # Build (start, end) spans for each line; the trailing newline stays with its line.
    lines: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for idx, tok in enumerate(ids):
        if tok in newline_ids:
            lines.append((start, idx + 1, tuple(ids[start : idx + 1])))
            start = idx + 1
    if start < t:
        lines.append((start, t, tuple(ids[start:t])))

    n = len(lines)
    li = 0
    while li < n:
        key = lines[li][2]
        # skip trivial empty/bare-newline lines (avoid flagging normal blank spacing)
        if len(key) <= 1:
            li += 1
            continue
        lj = li + 1
        while lj < n and lines[lj][2] == key:
            lj += 1
        if lj - li >= min_line_repeat:
            span_start = lines[li][0]
            span_end = lines[lj - 1][1]
            for k in range(span_start, span_end):
                if lambda_body > covered[k]:
                    covered[k] = lambda_body
            if mode == "wall":
                entry_w = lambda_body + lambda_entry
                if entry_w > covered[span_start]:
                    covered[span_start] = entry_w
            li = lj
        else:
            li += 1


def compute_repetition_penalty(
    response_ids: list[int],
    *,
    ngram_ns: tuple[int, ...] = (1, 3, 5, 8),
    min_repeat: int = 8,
    min_line_repeat: int = 20,
    newline_ids: Optional[set[int]] = None,
    lambda_body: float = 0.5,
    lambda_entry: float = 3.0,
    mode: str = "wall",
    eos_id: Optional[int] = None,
    protect_tail_eos: bool = True,
) -> Optional[list[float]]:
    """Per-token repetition penalty weights for advantage shaping (no decoding).

    Detects pathological repetition directly on ``response_ids`` and returns a list
    of non-negative weights (same length as ``response_ids``) to be *subtracted* from
    the token-level advantage in the distillation policy loss. ``0.0`` means no
    penalty. The entry token of each repeated span additionally receives
    ``lambda_entry`` when ``mode="wall"`` (a gradient "wall" at the decision to start
    repeating); ``mode="penalize"`` applies only ``lambda_body`` uniformly.

    Only long, clearly-degenerate repetition is flagged (``>= min_repeat`` back-to-back
    n-gram blocks, ``>= min_line_repeat`` identical lines) so ordinary repeated math
    (aligned equations, a couple of ``$$``) is left untouched. The terminal EOS/stop
    token is never penalized so the model keeps learning to stop.

    Returns ``None`` when nothing is flagged (caller may skip attaching the tensor).
    """
    t = len(response_ids)
    if t == 0:
        return None
    covered = [0.0] * t
    _mark_ngram_runs(
        response_ids,
        covered,
        ngram_ns=tuple(ngram_ns),
        min_repeat=min_repeat,
        lambda_body=lambda_body,
        lambda_entry=lambda_entry,
        mode=mode,
    )
    if newline_ids:
        _mark_line_runs(
            response_ids,
            covered,
            set(newline_ids),
            min_line_repeat=min_line_repeat,
            lambda_body=lambda_body,
            lambda_entry=lambda_entry,
            mode=mode,
        )
    # Never penalize a clean terminal stop signal.
    if protect_tail_eos and eos_id is not None and response_ids[-1] == eos_id:
        covered[-1] = 0.0
    if not any(covered):
        return None
    return covered


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
