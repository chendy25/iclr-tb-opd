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
"""Reward scoring for Gen-Verse/Open-AgentRL-30K's mixed data sources.

Open-AgentRL rows carry heterogeneous ``data_source`` values and two different
code-test encodings inside ``reward_model.ground_truth``:

- math (``math_dapo``, ``train-math-*``, ``test-math-*``) and science MCQ
  (``mega-science``): a short boxed answer -> scored by ``math_dapo`` (boxed
  extraction + normalized comparison).
- code (``train-code-taco-*``, ``train-code-leetcode-*`` ...): ground_truth is a
  JSON object in one of two shapes:
    * TACO / APPS functional  : {"fn_name", "inputs", "outputs"}  -> prime_code
    * LeetCode assert harness  : {"entry_point", "import_prefix", "test_code"}
      -> executed locally (import_prefix + solution + test_code + check(entry)).

``compute_score`` returns a float in [0, 1] for code and the ``math_dapo`` dict
(``{score, acc, pred}``) for math/science, both of which ``default_compute_score``
accepts.
"""

import json
import logging
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _extract_code(completion: str) -> str:
    """Pull the last fenced python block from a completion, else return as-is."""
    if not isinstance(completion, str):
        completion = str(completion)
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion and completion.count("```") >= 2:
        return completion.split("```")[-2]
    return completion


def _parse_gt(ground_truth: Any) -> Optional[dict]:
    obj: Any = None
    if isinstance(ground_truth, dict):
        obj = ground_truth
    elif isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, dict):
        return None
    # Some eval sets (e.g. LiveCodeBench_v6) nest the harness under "ground_truth".
    inner = obj.get("ground_truth")
    if isinstance(inner, dict) and ("inputs" in inner or "test_code" in inner or "fn_name" in inner):
        return inner
    return obj


def _score_leetcode(completion: str, gt: dict, timeout: int = 12) -> float:
    """Run the LeetCode-style assert harness in a subprocess; 1.0 iff it exits 0."""
    solution = _extract_code(completion)
    prefix = gt.get("import_prefix", "") or ""
    test_code = gt.get("test_code", "") or ""
    entry = (gt.get("entry_point", "") or "").strip()
    if not test_code or not entry:
        return 0.0
    program = f"{prefix}\n{solution}\n{test_code}\ncheck({entry})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 1.0 if r.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:  # noqa: BLE001
        logger.warning("open_agentrl leetcode exec error: %s", e)
        return 0.0


def _score_taco(completion: str, gt: dict) -> float:
    """Score TACO/APPS functional tests via prime_code (handles fn_name).

    ``prime_code`` imports ``pyext`` at module load, which is not installable on
    Python 3.11. Guard the import so a missing/broken dependency degrades to a 0
    reward for this one sample instead of crashing the whole trajectory group's
    reward postprocess.
    """
    try:
        from . import prime_code
    except Exception as e:  # noqa: BLE001  -- ModuleNotFoundError('pyext') etc.
        logger.warning("open_agentrl taco scorer unavailable (%s); returning 0.0", e)
        return 0.0

    try:
        success, _ = prime_code.compute_score(completion, gt, continuous=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("open_agentrl taco exec error: %s", e)
        return 0.0
    if isinstance(success, bool):
        return 1.0 if success else 0.0
    try:
        return float(success)
    except (TypeError, ValueError):
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
):
    """Route an Open-AgentRL row to the right scorer based on source + gt shape."""
    src = str(data_source or "")
    is_code = src.startswith("train-code-") or src.startswith("test-code-") or "code" in src.split("-")[:2]

    parsed = _parse_gt(ground_truth)
    if is_code or (parsed is not None and ("test_code" in parsed or "inputs" in parsed or "fn_name" in parsed)):
        if parsed is None:
            return 0.0
        if "test_code" in parsed or "entry_point" in parsed:
            return _score_leetcode(solution_str, parsed)
        if "inputs" in parsed or "fn_name" in parsed:
            return _score_taco(solution_str, parsed)
        return 0.0

    # math / science MCQ (mega-science) -> boxed answer match.
    if ground_truth is None or (isinstance(ground_truth, str) and ground_truth == ""):
        return {"score": 0.0, "acc": False, "pred": None}
    from . import math_dapo

    return math_dapo.compute_score(solution_str, str(ground_truth))
