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

Open-AgentRL-30K is already in verl schema (``prompt`` chat list, ``reward_model``
with ``ground_truth``). It mixes math (``math_dapo`` / ``train-math-*``), science
MCQ (``mega-science``) and code (``train-code-taco-*`` / ``train-code-leetcode-*``)
sources. For the agentic TB-OPD Phase 1' tool-integrated-reasoning (TIR) env we:

  1. tag each row with ``agent_name="tool_agent"`` so ``ToolAgentLoop`` runs it, and
  2. prepend a system prompt that instructs the model to use the ``code_interpreter``
     tool during reasoning.

No source is filtered or remapped: every ``data_source`` keeps its original value and
is scored by its own reward in ``verl.utils.reward_score.default_compute_score``
(math/science -> ``math_dapo``; code -> ``open_agentrl`` -> prime_code / local exec).

Usage:
    python -m recipe.agentic_tbopd.prepare_open_agentrl \
        --hf_root /mnt/afs_reason/chendongyang/code/data/hf_datasets \
        --out_dir /mnt/afs_reason/chendongyang/code/data/preprocessed/open_agentrl_tir
"""

import argparse
import os

import pandas as pd

TOOL_SYSTEM_PROMPT = (
    "You are a math expert who solves problems with the help of a Python code "
    "interpreter. Reason step by step. Whenever a calculation, verification, or "
    "search over cases would help, call the `code_interpreter` tool with a short "
    "Python snippet and read its stdout before continuing. You may call the tool "
    "multiple times. When you are confident, put your final answer inside "
    "\\boxed{...}."
)


def _to_message_list(prompt_field):
    """Normalize the ``prompt`` field into a list of {role, content} messages.

    pandas reads a parquet list-of-struct column as a numpy object array, so accept
    any sequence of mapping-like items (dict / numpy struct) as well as a raw string.
    """
    if hasattr(prompt_field, "tolist"):  # numpy ndarray
        prompt_field = prompt_field.tolist()
    if isinstance(prompt_field, (list, tuple)):
        return [dict(m) for m in prompt_field]
    return [{"role": "user", "content": str(prompt_field)}]


def _to_dict(value) -> dict:
    """Coerce a possibly-numpy / None parquet cell into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):  # numpy scalar wrapping a python object
        try:
            inner = value.item()
            if isinstance(inner, dict):
                return dict(inner)
        except (ValueError, AttributeError):
            pass
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _process_row(example: dict, idx: int, split: str, add_system: bool) -> dict:
    messages = _to_message_list(example.get("prompt"))
    if add_system:
        has_system = messages and messages[0].get("role") == "system"
        if has_system:
            messages[0]["content"] = TOOL_SYSTEM_PROMPT + "\n\n" + messages[0].get("content", "")
        else:
            messages = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}] + messages

    extra_info = _to_dict(example.get("extra_info"))
    extra_info.setdefault("index", idx)
    extra_info["split"] = split

    # Ground truth for the tool's create() (also keeps the parquet struct non-empty,
    # since pyarrow cannot serialize a struct with no child fields).
    rm = _to_dict(example.get("reward_model"))
    gt = str(rm.get("ground_truth", "") or "")

    # Route this sample to the code_interpreter tool.
    extra_info["need_tools_kwargs"] = True
    extra_info["tools_kwargs"] = {"code_interpreter": {"create_kwargs": {"ground_truth": gt}}}
    extra_info["tool_selection"] = ["code_interpreter"]

    # Keep the original data_source untouched. Open-AgentRL-30K mixes math
    # (``math_dapo`` / ``train-math-*``), science-MCQ (``mega-science``) and code
    # (``train-code-taco-*`` / ``train-code-leetcode-*``) rows; reward routing lives
    # in ``verl.utils.reward_score.default_compute_score`` (+ ``open_agentrl``), so
    # every source is scored with its own compute_score -- no filtering / remapping.
    return {
        "data_source": example.get("data_source", "math_dapo"),
        "agent_name": "tool_agent",
        "prompt": messages,
        "ability": example.get("ability", "MATH"),
        "reward_model": example.get("reward_model"),
        "extra_info": extra_info,
    }


def _convert(parquet_path: str, split: str, add_system: bool) -> pd.DataFrame:
    # Read/write parquet directly with pandas/pyarrow to avoid the HF datasets
    # cache/filelock machinery (which is brittle on shared filesystems).
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    rows = [_process_row(rec, i, split, add_system) for i, rec in enumerate(records)]
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
    ap.add_argument("--no_system", action="store_true", help="Do not inject the tool-use system prompt.")
    ap.add_argument(
        "--eval_names",
        nargs="*",
        default=["aime2024", "aime2025", "gpqa-diamond", "livecodebench-v6"],
        help="Subdirs of Open-AgentRL-Eval to convert as validation sets.",
    )
    args = ap.parse_args()
    add_system = not args.no_system
    os.makedirs(args.out_dir, exist_ok=True)

    # Train: Open-AgentRL-30K.
    train_src = os.path.join(args.hf_root, "Gen-Verse__Open-AgentRL-30K", "Open-AgentRL-30K.parquet")
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"train parquet not found: {train_src}")
    train_ds = _convert(train_src, "train", add_system)
    train_out = os.path.join(args.out_dir, "train.parquet")
    train_ds.to_parquet(train_out)
    print(f"[train] {len(train_ds)} rows -> {train_out}")

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
        ds = _convert(src, name, add_system)
        out = os.path.join(args.out_dir, f"test_{name.replace('-', '_')}.parquet")
        ds.to_parquet(out)
        print(f"[eval:{name}] {len(ds)} rows -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()
