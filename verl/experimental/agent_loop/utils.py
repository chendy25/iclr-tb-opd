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

# A ``\boxed{...}`` content that is a single bare latin variable (``N``, ``x``,
# ``k`` ...) is treated as a placeholder, not a final value.
_BARE_VAR_RE = re.compile(r"^[A-Za-z]$")

# Unwrap text-formatting commands (``\text{?}`` -> ``?``) the same way the reward
# extractor (``math_dapo.normalize_final_answer``) does, so a placeholder RHS like
# ``\text{?}`` reduces to bare punctuation and is rejected below.
_TEXT_WRAP_RE = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}")


def _iter_boxed(text: str):
    """Yield ``(start, end, content)`` for each brace-balanced ``\\boxed{...}``.

    ``start`` is the offset of the ``\\`` and ``end`` is one past the matching
    ``}``; ``content`` is the text between the outer braces. Stops at the first
    unterminated ``\\boxed{`` (no complete boxed can follow it either).
    """
    tok = "\\boxed{"
    search = 0
    while True:
        start = text.find(tok, search)
        if start < 0:
            return
        i = start + len(tok)
        depth = 1
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return  # unterminated boxed -> no complete answer from here on
        yield start, i, text[start + len(tok) : i - 1]
        search = i


def _strip_math_wrappers(s: str) -> str:
    """Strip surrounding whitespace and a single math delimiter layer (``$``/``\\(``/``\\[``)."""
    s = s.strip()
    for left, right in (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if len(s) >= len(left) + len(right) and s.startswith(left) and s.endswith(right):
            s = s[len(left) : len(s) - len(right)].strip()
            break
    return s


def _is_answer_box(content: str) -> bool:
    """Heuristic: does a ``\\boxed{...}`` content look like a real final answer?

    Conservative on purpose -- only rejects boxes that are *clearly* not a final
    value so we never skip a genuine answer (any ambiguous case is accepted). This
    fixes the "fake first box" anchor bug where a mid-reasoning placeholder box
    (``\\boxed{N}``, ``\\boxed{}``, a nested ``\\boxed``) was mistaken for the final
    answer, masking real reasoning out of the loss (and teaching EOS at the wrong
    spot).

    Equation-form answers are NOT rejected: a box like ``1 + 8 = 9`` or ``x = 5`` is
    a real commitment whose value is the right-hand side. We reduce it to that RHS
    the way the reward extractor does (``math_dapo.normalize_final_answer`` reduces
    via ``split("=")[-1]`` and unwraps ``\\text{...}``) and judge the reduced value,
    instead of dropping every box that contains ``=``. Empirically ~180/207
    equation-first boxes in a full run were the actual answer written out (reward
    scored them correct), so the old "reject any ``=``" rule wrongly skipped real
    answers (and let format-shaping mis-penalize them).

    Nested ``\\boxed`` is unwrapped to its innermost box rather than rejected: a real
    pattern is ``\\boxed{288 + 143 = \\boxed{431}}`` whose actual answer is the inner
    ``431`` -- exactly what the reward extractor picks (``last_boxed_only_string`` takes
    the LAST ``\\boxed``). We recurse to that innermost value before judging.

    Rejected patterns (after unwrapping nesting and reducing an equation to its RHS):
      * empty box / RHS empty (a trailing ``=`` with nothing after) / empty inner box
      * a single bare latin variable (``N`` / ``x`` / ``k``)
      * no alphanumeric char at all (pure punctuation, e.g. ``?`` / ``\\text{?}``)
    """
    inner = _strip_math_wrappers(content)
    # Nested \boxed: the real value is the innermost box (reward uses the LAST \boxed).
    while "\\boxed" in inner:
        inner_c = None
        for _s, _e, c in _iter_boxed(inner):
            inner_c = c  # keep the last (innermost when singly nested)
        if inner_c is None:
            return False  # unbalanced nested \boxed -> placeholder, not a value
        inner = _strip_math_wrappers(inner_c)
    if "=" in inner:
        # Value is the RHS of the last '=', mirroring the reward extractor.
        inner = inner.split("=")[-1]
    # Unwrap \text{...} etc so a placeholder like \text{?} reduces to bare punctuation.
    inner = _TEXT_WRAP_RE.sub(r"\1", inner).replace("$", "").strip()
    if not inner:
        return False
    if _BARE_VAR_RE.match(inner):
        return False
    if not any(ch.isalnum() for ch in inner):
        return False
    return True


def _first_boxed_end(text: str) -> Optional[int]:
    """Char offset just past the first *answer-shaped* balanced ``\\boxed{...}``.

    Skips fake / placeholder boxes (see ``_is_answer_box``) so the anchor lands on
    a real final answer instead of a mid-reasoning box. Returns ``None`` when no
    complete answer-shaped boxed expression exists.
    """
    for _start, end, content in _iter_boxed(text):
        if _is_answer_box(content):
            return end
    return None


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


def keep_len_after_final_answer(
    tokenizer,
    response_ids: list[int],
    eos_id: Optional[int] = None,
    post_answer_cap: int = 512,
) -> Optional[int]:
    """Number of leading response tokens to keep so the loss ignores post-answer text.

    Decodes ``response_ids`` and locates the first complete final answer
    (``\\boxed{...}`` or an ``Answer:`` line). Returns the smallest token count
    ``k`` whose decoded prefix already contains that answer, so callers can zero
    the response mask beyond ``k`` and drop trailing repetition. Returns ``None``
    when no complete answer is found (leave the mask untouched).

    EOS retention: if the model stopped *promptly* after answering -- i.e. an
    ``eos_id`` appears within ``post_answer_cap`` tokens of the answer end -- the
    kept region is extended through that EOS so the "answer then stop" signal
    (including the terminal EOS) stays in the loss and the model keeps learning to
    stop. When the model instead ran on into a long refrain (no EOS within the
    cap, e.g. it hit ``max_response_length``), the kept region stays at the answer
    end so only the pre-answer + answer tokens contribute and the refrain is
    dropped. This fixes the earlier behavior where the answer-tail EOS was always
    masked out (which removed the stop signal entirely).
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
    keep = lo
    # Extend through a prompt-close EOS so the terminal stop stays in the loss.
    if eos_id is not None:
        limit = min(len(response_ids), keep + max(int(post_answer_cap), 0))
        for i in range(keep, limit):
            if response_ids[i] == eos_id:
                return i + 1
    return keep


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
    ngram_max_period: int = 64,
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

    Periods are scanned *densely* from 1 up to ``ngram_max_period`` (not a sparse set)
    so sentence-length refrains such as ``"Let's compute the value:\\n\\n"`` -- observed
    as ~9-token units that a sparse ``(1,3,5,8)`` set misses entirely -- are caught.
    ``min_repeat`` back-to-back copies keeps the bar high enough that ordinary math is
    untouched (e.g. a 9-token block must repeat 8x = 72 tokens to trigger).

    Returns ``None`` when nothing is flagged (caller may skip attaching the tensor).
    """
    t = len(response_ids)
    if t == 0:
        return None
    covered = [0.0] * t
    max_p = max(int(ngram_max_period), (max(ngram_ns) if ngram_ns else 1))
    periods = tuple(range(1, max_p + 1))
    _mark_ngram_runs(
        response_ids,
        covered,
        ngram_ns=periods,
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
