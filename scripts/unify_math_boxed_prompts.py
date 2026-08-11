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
"""Unify math prompts to the ``\\boxed{}`` answer format (with backup).

Some math datasets (DAPO-Math-17k, AIME) prompt for the final answer on an
``Answer:`` line, while others (MATH-500) prompt for ``\\boxed{}``. Mixing the
two formats makes post-answer detection and scoring inconsistent. This script
rewrites every prompt to the ``\\boxed{}`` format:

  - strips the DAPO ``Answer:`` scaffolding (prefix + suffix) back to the raw
    problem, then appends the boxed instruction;
  - leaves prompts that already contain ``\\boxed`` untouched (idempotent);
  - the ``reward_model.ground_truth`` is never modified.

Every source file is backed up before being overwritten (in place by default),
or written to ``--out-dir`` instead. A ``.example.json`` with a before/after
sample is emitted next to each output for manual inspection.

Example:
  python scripts/unify_math_boxed_prompts.py \
      --inputs /path/preprocessed/dapo_math_17k/train.parquet \
               /path/preprocessed/aime2024/test.parquet \
               /path/preprocessed/aime2025/test.parquet \
               /path/preprocessed/math500/test.parquet
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Keep these in sync with scripts/download_and_prep_math_data.py.
BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

DAPO_ANSWER_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_ANSWER_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _strip_dapo_scaffold(content: str) -> str:
    """Remove the DAPO ``Answer:`` prefix/suffix, returning the raw problem."""
    text = content
    if text.startswith(DAPO_ANSWER_PREFIX):
        text = text[len(DAPO_ANSWER_PREFIX) :]
    if text.endswith(DAPO_ANSWER_SUFFIX):
        text = text[: -len(DAPO_ANSWER_SUFFIX)]
    return text.strip()


def to_boxed_content(content: str) -> str:
    """Convert a single prompt string to the ``\\boxed{}`` format (idempotent)."""
    if "\\boxed" in content:
        return content
    raw_problem = _strip_dapo_scaffold(content)
    return f"{raw_problem} {BOXED_INSTRUCTION}"


def _convert_prompt(prompt: Any) -> tuple[Any, Optional[str], Optional[str]]:
    """Convert the user-message content in a verl ``prompt`` list.

    Returns ``(new_prompt, before, after)`` where ``before``/``after`` are the
    first user message content (for logging), or ``None`` if nothing changed.
    """
    # Normalize numpy arrays / tuples to a list of plain dicts.
    messages = [dict(m) for m in list(prompt)]
    before = after = None
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            before = msg["content"]
            after = to_boxed_content(before)
            msg["content"] = after
            break
    return messages, before, after


def convert_file(src: Path, out: Path, backup_dir: Path, dry_run: bool) -> dict:
    df = pd.read_parquet(src)
    if "prompt" not in df.columns:
        raise ValueError(f"{src}: no 'prompt' column (cols={list(df.columns)})")

    new_prompts = []
    n_changed = 0
    example: Optional[dict] = None
    for prompt in df["prompt"].tolist():
        new_prompt, before, after = _convert_prompt(prompt)
        new_prompts.append(new_prompt)
        if before is not None and after is not None and before != after:
            n_changed += 1
            if example is None:
                example = {"before": before, "after": after}
    df = df.copy()
    df["prompt"] = new_prompts

    stats = {
        "src": str(src),
        "out": str(out),
        "rows": len(df),
        "changed": n_changed,
        "already_boxed": len(df) - n_changed,
    }
    if dry_run:
        stats["dry_run"] = True
        if example is not None:
            stats["example"] = example
        return stats

    # Back up the original before writing (never clobber an existing backup).
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / src.name
    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_path}. Refusing to overwrite a prior backup; "
            "remove it or pick a different --backup-dir."
        )
    shutil.copy2(src, backup_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    if example is not None:
        with open(out.with_suffix(".boxed_example.json"), "w") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)

    stats["backup"] = str(backup_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Parquet files to convert.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write converted files to (mirrors input filenames). "
        "Default: overwrite each input in place (after backup).",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Directory to store backups of the original files. "
        "Default: <parent>/_prompt_backup_answer_<timestamp>/ next to each input.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_stats = []
    for path_str in args.inputs:
        src = Path(path_str).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        out = Path(args.out_dir).resolve() / src.name if args.out_dir else src
        backup_dir = (
            Path(args.backup_dir).resolve()
            if args.backup_dir
            else src.parent / f"_prompt_backup_answer_{ts}"
        )
        stats = convert_file(src, out, backup_dir, args.dry_run)
        all_stats.append(stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    total = sum(s["rows"] for s in all_stats)
    changed = sum(s["changed"] for s in all_stats)
    print(f"\n[unify_math_boxed_prompts] files={len(all_stats)} rows={total} changed={changed} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
