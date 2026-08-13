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
"""Convert Gen-Verse/Open-AgentRL parquet into tool-agent (code_interpreter) format.

SOD-aligned. We reproduce the prompt/reward rubric the SOD teacher was trained
under (``SOD/recipe/demystify/reward.py::CustomRLHFDataset``) so the agentic
TB-OPD student sees the same task framing:

Prompt (per SOD ``map_fn* ``):
  * No system prompt -- all instructions live in the user turn; the
    ``code_interpreter`` tool is exposed via the chat template's tool section.
  * math / science: ``math_prompt_1 + problem + math_prompt_2 + agent_prompt +
    answer_format`` then a units clarifier -> forces a final ``\\boxed{...}``.
  * code: ``... + agent_prompt`` then a "submit your code within ```python```"
    clarifier.
  Rows already carrying the SOD scaffolding (Open-AgentRL-30K's ``math_dapo`` /
  ``mega-science`` / ``train-code-*`` were pre-wrapped upstream) only get the
  small trailing clarifier appended (idempotent). Bare rows (``train-math-*``)
  are wrapped in full -- without this they never get the ``\\boxed{}``
  instruction and would score ~0 under ``strict_box_verify=True``.

Reward routing:
  Every row is tagged ``extra_info["agentic_reward"]=True`` (+ ``need_tools_kwargs``)
  so ``verl.utils.reward_score.default_compute_score`` sends it to the
  SOD-homologous ``open_agentrl.compute_score`` (strict boxed math + LiveCodeBench
  code executed on E2B + num_turns tool-call shaping). ``data_source`` is kept
  intact (no filtering / remapping).

Usage:
    python -m recipe.agentic_tbopd.prepare_open_agentrl \
        --hf_root /mnt/afs_reason/chendongyang/code/data/hf_datasets \
        --out_dir /mnt/afs_reason/chendongyang/code/data/preprocessed/open_agentrl_tir
"""

import argparse
import json
import os

import pandas as pd

# --- SOD prompt fragments (verbatim from SOD/recipe/demystify/reward.py) ---
ANSWER_FORMAT = (
    "\nRemember once you make sure the current answer is your final answer, do not "
    "call the tools again and directly output the final answer in the following text "
    "format, the answer format must be: \\boxed{'The final answer goes here.'}."
)
MATH_PROMPT_1 = "Analyze and solve the following math problem step by step. \n\n"
MATH_PROMPT_2 = (
    "\n\nThe tool could be used for more precise and efficient calculation and could "
    "help you to verify your result before you reach the final answer."
)
AGENT_PROMPT = (
    "\n\n**Note: You should first analyze the problem and form a high-level solution "
    "strategy, then utilize the tools to help you solve the problem.**"
)
UNITS_NOTE = (
    "\nDo not put units of the final answer inside \\boxed{}. The content of \\boxed{} "
    "should be the numerical value of the final answer only, without any units."
)
CODE_SUBMIT_NOTE = (
    "\nBefore sumbit your code, you can utilize tools to check the correctness of your "
    "code, once you make sure the current code is correct, do not call the tools again "
    "and submit your code within ```python\n# YOUR CODE HERE\n```."
)


def _is_code_source(data_source: str) -> bool:
    return "code" in str(data_source or "").lower()


def _sod_wrap(content: str, is_code: bool) -> str:
    """Apply SOD's per-source prompt scaffolding, idempotently."""
    content = content or ""
    if is_code:
        # Ensure the code-submission clarifier + agent note are present.
        if "submit your code within" not in content:
            already_agent = "high-level solution" in content
            content = content + ("" if already_agent else AGENT_PROMPT) + CODE_SUBMIT_NOTE
        return content

    # math / science
    has_box_instr = "\\boxed" in content or "answer format must be" in content
    if not has_box_instr:
        # Bare problem (e.g. train-math-*): full SOD scaffold.
        content = MATH_PROMPT_1 + content + MATH_PROMPT_2 + AGENT_PROMPT + ANSWER_FORMAT + UNITS_NOTE
    elif "without any units" not in content:
        # Already SOD-framed upstream: only add the units clarifier.
        content = content + UNITS_NOTE
    return content


def _to_message_list(prompt_field):
    """Normalize the ``prompt`` field into a list of {role, content} messages."""
    if hasattr(prompt_field, "tolist"):  # numpy ndarray
        prompt_field = prompt_field.tolist()
    if isinstance(prompt_field, (list, tuple)):
        return [dict(m) for m in prompt_field]
    return [{"role": "user", "content": str(prompt_field)}]


def _to_dict(value) -> dict:
    """Coerce a possibly-numpy / None / JSON-string parquet cell into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return dict(obj)
            except (ValueError, TypeError):
                pass
        return {}
    if hasattr(value, "item"):  # numpy scalar wrapping a python object
        try:
            inner = value.item()
            if isinstance(inner, dict):
                return dict(inner)
            if isinstance(inner, str):
                return _to_dict(inner)
        except (ValueError, AttributeError):
            pass
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _first_user_index(messages: list) -> int:
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            return i
    return len(messages) - 1 if messages else -1


def _extract_problem_text(example: dict) -> str:
    """Pull the raw problem text from train (``prompt``) or Eval (``Problem``/``problem``)."""
    prompt = example.get("prompt")
    if prompt is not None:
        # Train rows already have chat messages; keep them.
        if isinstance(prompt, (list, tuple)) or hasattr(prompt, "tolist"):
            messages = _to_message_list(prompt)
            for m in messages:
                if m.get("role") == "user" and m.get("content"):
                    return str(m["content"])
            if messages:
                return str(messages[0].get("content", "") or "")
        text = str(prompt).strip()
        if text and text.lower() != "none":
            return text

    for key in ("Problem", "problem", "question", "Question"):
        if key in example and example[key] is not None:
            text = str(example[key]).strip()
            if text and text.lower() != "none":
                return text
    return ""


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` succeeds."""
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_reward_model(example: dict, is_code: bool) -> dict:
    """Build a verl ``reward_model`` dict with a usable ``ground_truth``.

    Open-AgentRL-Eval raw schemas differ from the 30K train parquet:
      * AIME24: ``Answer`` (int/str)
      * AIME25: ``answer``
      * GPQA:   ``solution`` (letter)
      * LCB:    ``reward_model`` JSON string ``{"ground_truth": {...harness...}}``

    Code harnesses are stored as a JSON *string* (not a nested dict) so parquet
    round-trips do not turn lists into numpy arrays that break ``json.dumps``
    inside ``open_agentrl._unwrap_code_gt``.
    """
    rm = _to_dict(example.get("reward_model"))
    gt = rm.get("ground_truth")
    if gt not in (None, ""):
        if is_code or isinstance(gt, (dict, list)):
            return {"ground_truth": json.dumps(_jsonable(gt)), "style": rm.get("style", "rule")}
        return {"ground_truth": str(gt).strip(), "style": rm.get("style", "rule")}

    for key in ("Answer", "answer", "solution", "final_answer", "label"):
        if key in example and example[key] is not None:
            val = example[key]
            # Prefer short MCQ / numeric answers over long free-text "solution".
            if key == "solution" and is_code:
                continue
            if key == "solution" and isinstance(val, str) and len(val) > 8 and "\\boxed" not in val:
                # Long AIME "Solution" prose — skip; wait for Answer/answer.
                if any(k in example and example[k] is not None for k in ("Answer", "answer")):
                    continue
            return {"ground_truth": str(val).strip(), "style": "rule"}

    return {"ground_truth": "", "style": "rule"}


def _normalize_example(example: dict) -> dict:
    """Unify train / Eval row schemas before SOD wrapping."""
    data_source = example.get("data_source") or "math_dapo"
    is_code = _is_code_source(data_source) or (
        "livecodebench" in str(data_source).lower()
    )
    # LCB rows also count as code even if ability says so.
    if str(example.get("ability", "")).lower() == "code":
        is_code = True

    problem = _extract_problem_text(example)
    # Rebuild a single-user chat prompt from the bare problem when needed.
    if example.get("prompt") is not None and isinstance(example.get("prompt"), (list, tuple)):
        messages = _to_message_list(example.get("prompt"))
        # Repair rows whose user content collapsed to the literal "None".
        ui = _first_user_index(messages)
        if ui >= 0:
            content = str(messages[ui].get("content", "") or "")
            if (not content) or content.strip().lower() == "none" or content.lstrip().startswith("None\n"):
                messages = [{"role": "user", "content": problem}]
    else:
        messages = [{"role": "user", "content": problem}]

    reward_model = _extract_reward_model(example, is_code=is_code)
    ability = example.get("ability") or ("code" if is_code else "MATH")
    extra_info = _to_dict(example.get("extra_info"))

    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
        "_is_code": is_code,
    }


def _process_row(example: dict, idx: int, split: str, wrap_prompt: bool) -> dict:
    norm = _normalize_example(example)
    messages = [m for m in norm["prompt"] if m.get("role") != "system"]
    is_code = bool(norm["_is_code"])

    if wrap_prompt and messages:
        ui = _first_user_index(messages)
        if ui >= 0:
            messages[ui]["content"] = _sod_wrap(messages[ui].get("content", ""), is_code)

    extra_info = dict(norm["extra_info"])
    extra_info.setdefault("index", idx)
    extra_info["split"] = split

    rm = norm["reward_model"]
    gt = rm.get("ground_truth", "")
    # tools_kwargs create() only needs a string marker; keep code harness out of it.
    gt_for_tool = gt if isinstance(gt, str) else ""

    extra_info["need_tools_kwargs"] = True
    extra_info["agentic_reward"] = True  # -> open_agentrl.compute_score (SOD rubric)
    extra_info["tools_kwargs"] = {"code_interpreter": {"create_kwargs": {"ground_truth": gt_for_tool}}}
    extra_info["tool_selection"] = ["code_interpreter"]

    return {
        "data_source": norm["data_source"],
        "agent_name": "tool_agent",
        "prompt": messages,
        "ability": norm["ability"],
        "reward_model": rm,
        "extra_info": extra_info,
    }


def _convert(parquet_path: str, split: str, wrap_prompt: bool) -> pd.DataFrame:
    # Read/write parquet directly with pandas/pyarrow to avoid the HF datasets
    # cache/filelock machinery (which is brittle on shared filesystems).
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    rows = [_process_row(rec, i, split, wrap_prompt) for i, rec in enumerate(records)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf_root",
        default="/mnt/afs_reason/chendongyang/code/data/hf_datasets",
        help="Root containing Gen-Verse__Open-AgentRL-* dirs.",
    )
    ap.add_argument(
        "--out_dir",
        default="/mnt/afs_reason/chendongyang/code/data/preprocessed/open_agentrl_tir",
    )
    ap.add_argument(
        "--no_prompt_wrap",
        action="store_true",
        help="Do not apply SOD per-source prompt scaffolding (debug only).",
    )
    ap.add_argument(
        "--eval_names",
        nargs="*",
        default=["aime2024", "aime2025", "gpqa-diamond", "livecodebench-v6"],
        help="Subdirs of Open-AgentRL-Eval to convert as validation sets.",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Only rebuild test_*.parquet (skip the 30K train conversion).",
    )
    args = ap.parse_args()
    wrap_prompt = not args.no_prompt_wrap
    os.makedirs(args.out_dir, exist_ok=True)

    # Train: Open-AgentRL-30K.
    if not args.eval_only:
        train_src = os.path.join(args.hf_root, "Gen-Verse__Open-AgentRL-30K", "Open-AgentRL-30K.parquet")
        if not os.path.exists(train_src):
            raise FileNotFoundError(f"train parquet not found: {train_src}")
        train_ds = _convert(train_src, "train", wrap_prompt)
        train_out = os.path.join(args.out_dir, "train.parquet")
        train_ds.to_parquet(train_out)
        print(f"[train] {len(train_ds)} rows -> {train_out}")
    else:
        print("[train] skipped (--eval_only)")

    # Eval: each subdir under Open-AgentRL-Eval that has a parquet.
    eval_root = os.path.join(args.hf_root, "Gen-Verse__Open-AgentRL-Eval")
    for name in args.eval_names:
        subdir = os.path.join(eval_root, name)
        if not os.path.isdir(subdir):
            print(f"[skip] eval subdir missing: {subdir}")
            continue
        parquets = [f for f in os.listdir(subdir) if f.endswith(".parquet")]
        if not parquets:
            print(f"[skip] no parquet in {subdir}")
            continue
        src = os.path.join(subdir, parquets[0])
        ds = _convert(src, name, wrap_prompt)
        out = os.path.join(args.out_dir, f"test_{name.replace('-', '_')}.parquet")
        ds.to_parquet(out)
        print(f"[eval:{name}] {len(ds)} rows -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()
