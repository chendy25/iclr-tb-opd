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
"""SOD-homologous reward for Gen-Verse/Open-AgentRL (agentic TB-OPD).

This mirrors ``SOD/recipe/demystify/reward.py::compute_score`` so the agentic
TB-OPD run is scored with the *same* rubric as the SOD teacher was trained
with:

- code (``'code' in data_source`` or a code-harness ground_truth): scored by
  ``livecodebench.code_math.compute_score`` (LiveCodeBench, ported verbatim from
  SOD). SOD executes the tests on a SandboxFusion HTTP service; we have none, so
  ``code_math`` -> ``check_correctness`` -> ``call_sandbox_api`` transparently
  runs the *identical* generated code on an **E2B** sandbox instead
  (``SANDBOX_EXEC_BACKEND=e2b``). Same api_response shape => same scores.
- math / science-MCQ: ``math_dapo.compute_score(strict_box_verify=True)`` -- the
  final answer must be inside ``\\boxed{...}`` (SOD's strict rubric), not merely
  present in free text.

Tool-call shaping (SOD): a *wrong* trajectory (``score < 0``) gets a small
credit for having used the tool, so tool use is not discouraged on failures::

    tool_call_reward = (num_turns - 2) / 2 * 0.1
    score = min(-0.6, score + tool_call_reward)

``num_turns`` is the rollout turn count threaded in via the reward manager
(``extra_info["num_turns"]`` <- ``__num_turns__``). If it is unavailable (e.g.
non-agentic caller), the shaping term is skipped so the score cannot dip below
the raw ``code_math`` / ``math_dapo`` value.

Returns the SOD dict ``{score, acc, pred}`` which ``default_compute_score``
accepts.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _looks_like_code_harness(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "inputs" in obj or "fn_name" in obj or "test_code" in obj or "import_prefix" in obj
    )


def _unwrap_code_gt(ground_truth: Any) -> Any:
    """Return the code test harness as a JSON *string* for ``code_math``.

    Open-AgentRL train-code rows store the harness as a JSON string, but some eval
    sets (e.g. LiveCodeBench_v6) nest it under ``{"ground_truth": {...}}``.
    ``code_math.compute_score`` does ``test_cases = str(test_cases)`` then
    ``json.loads(test_cases)``: it therefore needs a *JSON string*, not a Python
    dict. Passing a dict makes ``str(dict)`` emit single-quoted Python repr, which
    ``json.loads`` cannot parse -> ``in_outs`` degrades to a str and
    ``check_correctness`` raises "string indices must be integers". So always hand
    back a canonical ``json.dumps`` of the top-level harness dict (unwrapping one
    level of nesting if present).
    """
    obj: Any = ground_truth
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
        except (ValueError, TypeError):
            return ground_truth  # not JSON; hand the raw string through
    if _looks_like_code_harness(obj):
        return json.dumps(obj)
    if isinstance(obj, dict):
        inner = obj.get("ground_truth")
        if _looks_like_code_harness(inner):
            return json.dumps(inner)
        return json.dumps(obj)
    # Non-dict (already a JSON string handled above, or something odd): pass through.
    return ground_truth if isinstance(ground_truth, str) else json.dumps(obj)


def _is_code(data_source: str, ground_truth: Any) -> bool:
    if "code" in str(data_source or "").lower():
        return True
    # Fallback on ground_truth shape (covers odd/absent data_source labels).
    probe = ground_truth
    if isinstance(probe, str):
        try:
            probe = json.loads(probe)
        except (ValueError, TypeError):
            probe = None
    if _looks_like_code_harness(probe):
        return True
    if isinstance(probe, dict) and _looks_like_code_harness(probe.get("ground_truth")):
        return True
    return False


def _apply_tool_call_shaping(result: dict, extra_info: Optional[dict]) -> dict:
    """SOD tool-call shaping on failed trajectories (in-place-safe)."""
    if result.get("score", 0.0) >= 0:
        if result.get("pred") is None:
            result["pred"] = ""
        return result

    num_turns = None
    if isinstance(extra_info, dict):
        num_turns = extra_info.get("num_turns", None)
    if num_turns is not None:
        try:
            nt = int(num_turns)
            tool_call_reward = (nt - 2) / 2 * 0.1
            result["score"] = float(min(-0.6, result["score"] + tool_call_reward))
        except (TypeError, ValueError):
            pass  # malformed num_turns -> leave the raw negative score untouched
    if result.get("pred") is None:
        result["pred"] = ""
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Score one Open-AgentRL row, SOD-homologously. Returns {score, acc, pred}."""
    if _is_code(data_source, ground_truth):
        # Lazy: livecodebench/code_math pulls sandbox + heavy deps; skip on math rows.
        from .livecodebench import code_math

        gt = _unwrap_code_gt(ground_truth)
        try:
            result = code_math.compute_score(solution_str, gt)
        except Exception as e:  # noqa: BLE001 -- one bad sample must not sink the group
            logger.warning("open_agentrl code_math error (%s); scoring 0.", e)
            result = {"score": -1.0, "acc": False, "pred": None}
    else:
        # math / science MCQ: strict boxed answer (SOD rubric).
        from . import math_dapo

        gt_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        result = math_dapo.compute_score(
            solution_str=solution_str, ground_truth=gt_str, strict_box_verify=True
        )

    if not isinstance(result, dict):
        result = {"score": float(result), "acc": bool(result and result > 0), "pred": None}

    return _apply_tool_call_shaping(result, extra_info)
