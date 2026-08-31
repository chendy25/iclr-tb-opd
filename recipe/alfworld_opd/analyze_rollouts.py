#!/usr/bin/env python3
"""Offline analysis of ALFWorld rollout dumps -- no GPU, no model, stdlib only.

Two things this answers, depending on what the dump contains.

**Any dump** -- protocol health. The one that matters right now is whether the
thinking switch actually took: a dump made under ``enable_thinking=False`` still
contains ``<think>`` in 100% of episodes, because Qwen3's chat template pre-fills an
*empty* pair. So the honest metric is the fraction of turns whose think block has
content, which separates the two protocols cleanly where a substring check does not.

**TB-OPD dumps** (those carrying ``tb_opd_slot``) -- the three-arm comparison. Under
the fixed-slot fan-out one group is ``main + k branches`` sharing a sample uid, so
the recovery rate (main lost, some branch won) is computable straight from the dump.
Point this at the forced-topk arm, the resample arm and a plain ``n=1+k`` run and the
three rows are the comparison:

    fork      TB_BRANCH_MODE=forced_topk   branches resume at the selected turn
    resample  TB_BRANCH_MODE=resample      branches resume at the same turn, plain sampling
    continue  TB_ENABLE=False ROLLOUT_N=3  independent episodes, no fork

Only the first differs from the second in *where* the branch diverges, and only the
second differs from the third in whether a prefix is shared, so the pair of gaps
separates "branching helps" from "more samples help".

Usage
-----
    python3 recipe/alfworld_opd/analyze_rollouts.py DUMP [DUMP ...]
    python3 recipe/alfworld_opd/analyze_rollouts.py fork/ resample/ continue/ --json out.json

DUMP may be a ``.jsonl`` file, a directory of them, or an experiment directory
(``rollout/`` underneath is found automatically). Later steps of a training run are
concatenated; pass ``--step`` to restrict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


# --------------------------------------------------------------------------- io


def find_jsonl(path: str) -> list[str]:
    """Resolve a file / dump dir / experiment dir to the .jsonl files under it."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    for cand in (path, os.path.join(path, "rollout")):
        if not os.path.isdir(cand):
            continue
        hits = sorted(
            os.path.join(cand, f) for f in os.listdir(cand) if f.endswith(".jsonl")
        )
        if hits:
            return hits
    # Experiment dir holding several eval dirs: recurse one level.
    out: list[str] = []
    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub):
            out.extend(find_jsonl(sub))
    return out


def load_rows(paths: Iterable[str], step: Optional[int]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if step is not None and int(row.get("step", 0)) != step:
                    continue
                rows.append(row)
    return rows


# ---------------------------------------------------------------------- helpers


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fnum(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sample_uid(row: dict) -> str:
    """Group key: the uid is ``{sample}_{rollout}_{output}``; the sample part is the group."""
    uid = str(row.get("uid", ""))
    parts = uid.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else uid


def won(row: dict) -> bool:
    v = fnum(row, "alfworld_won")
    if v is None:
        v = fnum(row, "score") or 0.0
    return v > 0.5


# -------------------------------------------------------------------- protocol


def protocol_stats(rows: list[dict], max_response_length: int) -> dict[str, Any]:
    n = len(rows)
    think_turns = think_nonempty = action_turns = 0
    think_lens: list[float] = []
    for row in rows:
        text = row.get("output") or ""
        blocks = THINK_RE.findall(text)
        think_turns += len(blocks)
        for b in blocks:
            b = b.strip()
            if b:
                think_nonempty += 1
                think_lens.append(len(b))
        action_turns += len(ACTION_RE.findall(text))

    steps = [v for v in (fnum(r, "alfworld_num_env_steps") for r in rows) if v is not None]
    lens = [v for v in (fnum(r, "alfworld_resp_len") for r in rows) if v is not None]
    invalid = [v for v in (fnum(r, "alfworld_invalid_action_frac") for r in rows) if v is not None]
    seeds = {fnum(r, "alfworld_seed") for r in rows}

    return {
        "episodes": n,
        "success_pct": pct(sum(won(r) for r in rows), n),
        "distinct_seeds": len([s for s in seeds if s is not None]),
        "steps_mean": mean(steps),
        "steps_p90": quantile(steps, 0.9),
        "step_capped_pct": pct(sum(1 for v in steps if v >= 50), len(steps)),
        "resp_len_mean": mean(lens),
        "budget_saturated_pct": pct(
            sum(1 for v in lens if v >= 0.95 * max_response_length), len(lens)
        ),
        "invalid_action_pct": 100.0 * mean(invalid) if invalid else float("nan"),
        "think_blocks": think_turns,
        "action_blocks": action_turns,
        # THE protocol discriminator: an empty-prefilled dump scores ~0 here while
        # containing <think> in every episode.
        "think_nonempty_pct": pct(think_nonempty, think_turns),
        "think_chars_mean": mean(think_lens),
    }


# ----------------------------------------------------------------------- TB-OPD


def tb_stats(rows: list[dict]) -> Optional[dict[str, Any]]:
    if not any("tb_opd_slot" in r for r in rows):
        return None

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[sample_uid(row)].append(row)

    n_groups = 0
    main_won = 0
    recovered = 0            # main lost, some branch won
    branch_only_win = 0      # groups whose ONLY win came from a branch
    forked_groups = 0
    branch_rows = 0
    branch_wins = 0
    squandered = 0           # main won yet branches were still spent
    fork_kind: Counter = Counter()
    estimator: Counter = Counter()
    branch_mode: Counter = Counter()
    none_reason: Counter = Counter()
    replay_bad = replay_seen = 0
    rel_depth: list[float] = []

    for _, members in groups.items():
        mains = [r for r in members if (fnum(r, "tb_opd_slot") or 0) == 0]
        branches = [r for r in members if (fnum(r, "tb_opd_slot") or 0) > 0]
        if not mains:
            continue
        n_groups += 1
        main = mains[0]
        m_won = won(main)
        main_won += m_won

        if fnum(main, "tb_opd_mode_branch") == 1.0:
            forked_groups += 1
        reason = main.get("tb_opd_none_reason")
        if reason:
            none_reason[str(reason)] += 1
        est = main.get("tb_opd_fork_estimator")
        if est:
            estimator[str(est)] += 1
        mode = main.get("tb_opd_branch_mode")
        if mode:
            branch_mode[str(mode)] += 1

        turn = fnum(main, "tb_opd_fork_turn")
        turns_total = fnum(main, "alfworld_num_env_steps")
        if turn is not None and turn >= 0 and turns_total:
            rel_depth.append(turn / turns_total)

        b_won = False
        for b in branches:
            branch_rows += 1
            kind = b.get("tb_opd_fork_kind")
            if kind:
                fork_kind[str(kind)] += 1
            ok = fnum(b, "alfworld_replay_ok")
            if ok is not None:
                replay_seen += 1
                replay_bad += ok < 0.5
            if won(b):
                branch_wins += 1
                b_won = True

        if not m_won and b_won:
            recovered += 1
            branch_only_win += 1
        if m_won and branches:
            squandered += 1

    return {
        "groups": n_groups,
        "main_success_pct": pct(main_won, n_groups),
        # Headline: of the episodes the student lost, how many did a branch rescue.
        "recovery_pct": pct(recovered, n_groups - main_won),
        "any_success_pct": pct(main_won + branch_only_win, n_groups),
        "forked_pct": pct(forked_groups, n_groups),
        "branch_rows": branch_rows,
        "branch_success_pct": pct(branch_wins, branch_rows),
        # only_fail should keep this at 0; anything else means slots were burned on
        # episodes that had already succeeded.
        "squandered_pct": pct(squandered, n_groups),
        "fork_depth_mean": mean(rel_depth),
        "replay_diverged_pct": pct(replay_bad, replay_seen),
        "fork_kind": dict(fork_kind.most_common()),
        "estimator": dict(estimator.most_common()),
        "branch_mode": dict(branch_mode.most_common()),
        "no_fork_reason": dict(none_reason.most_common(5)),
    }


# ---------------------------------------------------------------------- report


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.1f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={n}" for k, n in v.items()) if v else "-"
    return str(v)


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


PROTOCOL_ROWS = [
    ("episodes", "episodes"),
    ("success_pct", "success %"),
    ("distinct_seeds", "distinct games"),
    ("steps_mean", "env steps (mean)"),
    ("step_capped_pct", "hit 50-step cap %"),
    ("resp_len_mean", "resp tokens (mean)"),
    ("budget_saturated_pct", "budget saturated %"),
    ("invalid_action_pct", "invalid action %"),
    ("think_nonempty_pct", "NON-EMPTY think %"),
    ("think_chars_mean", "think chars/turn"),
]

TB_ROWS = [
    ("groups", "groups"),
    ("main_success_pct", "main success %"),
    ("recovery_pct", "RECOVERY % (of losses)"),
    ("any_success_pct", "any-slot success %"),
    ("forked_pct", "groups forked %"),
    ("branch_success_pct", "branch success %"),
    ("squandered_pct", "squandered (main won) %"),
    ("fork_depth_mean", "fork depth (frac of episode)"),
    ("replay_diverged_pct", "replay diverged %"),
    ("estimator", "U estimator"),
    ("fork_kind", "fork kind"),
    ("branch_mode", "branch mode"),
    ("no_fork_reason", "no-fork reasons"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="jsonl file, rollout dir, or experiment dir")
    ap.add_argument("--label", action="append", default=None, help="name per dump (repeatable)")
    ap.add_argument("--step", type=int, default=None, help="only this training step")
    ap.add_argument("--max-response-length", type=int, default=20480)
    ap.add_argument("--json", default=None, help="also write the metrics as JSON")
    args = ap.parse_args()

    labels = args.label or []
    results: dict[str, dict[str, Any]] = {}
    for i, dump in enumerate(args.dumps):
        label = labels[i] if i < len(labels) else os.path.basename(os.path.normpath(dump))
        paths = find_jsonl(dump)
        if not paths:
            print(f"[warn] no .jsonl under {dump}", file=sys.stderr)
            continue
        rows = load_rows(paths, args.step)
        if not rows:
            print(f"[warn] {dump}: no rows (step filter too strict?)", file=sys.stderr)
            continue
        results[label] = {
            "files": len(paths),
            "protocol": protocol_stats(rows, args.max_response_length),
            "tb": tb_stats(rows),
        }

    if not results:
        print("nothing to report", file=sys.stderr)
        return 1

    names = list(results)
    print("\n=== protocol / episode health ===")
    print(
        table(
            ["metric"] + names,
            [[lbl] + [fmt(results[n]["protocol"].get(key)) for n in names] for key, lbl in PROTOCOL_ROWS],
        )
    )
    print(
        "\nNON-EMPTY think % is the protocol discriminator: an enable_thinking=False dump\n"
        "still contains <think> in every episode (template-prefilled, empty), so a plain\n"
        "substring check cannot tell the two protocols apart."
    )

    tb_names = [n for n in names if results[n]["tb"]]
    if tb_names:
        print("\n=== TB-OPD branching (three-arm comparison) ===")
        print(
            table(
                ["metric"] + tb_names,
                [[lbl] + [fmt(results[n]["tb"].get(key)) for n in tb_names] for key, lbl in TB_ROWS],
            )
        )
        print(
            "\nRECOVERY % is the headline: of the episodes the main slot lost, the share a\n"
            "branch rescued. Compare forced_topk vs resample to isolate *where* the branch\n"
            "diverges, and resample vs a plain n=1+k run to isolate the shared prefix from\n"
            "the extra samples."
        )
    else:
        print("\n(no tb_opd_slot column in any dump -- these are plain rollouts, so the")
        print(" three-arm comparison needs a TB_ENABLE=True run first.)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
