#!/usr/bin/env python
"""Probe whether ``seed(s)`` + ``reset()`` actually selects a distinct ALFWorld game.

The in-repo agent loop seeds each episode with its dataset row index so that a batch
of N rows plays N different games. Rollout dumps showed only 1-8 distinct games per
64-row batch, so this checks the seed -> gamefile mapping directly, both on a single
reused handle (what the pool does) and on independent handles.
"""

import os
import sys

sys.path.insert(0, os.environ.get("CODE_DIR", "/mnt/afs_reason/chendongyang/code/iclr/verl"))

from verl.experimental.agent_loop.alfworld_env.env_pool import (  # noqa: E402
    AlfredEnvHandle,
    _get_base_env,
)

CONFIG = os.environ.get(
    "ALFWORLD_CONFIG_PATH",
    "/mnt/afs_reason/chendongyang/code/iclr/verl/verl/experimental/agent_loop/alfworld_env/config_tw.yaml",
)
SEEDS = [0, 1, 2, 3, 100, 1000, 3552, 3553]


def short(path):
    # Keep the trial dir: one task template holds many trials, so dropping it makes
    # genuinely different games look identical.
    parts = str(path).split("/")
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(path)


def main():
    base = _get_base_env(CONFIG, "train")

    print("--- one reused handle (mirrors the pool) ---")
    handle = AlfredEnvHandle(base)
    reused = []
    for s in SEEDS:
        reused.append(short(handle.reset(s)["gamefile"]))
        print(f"  seed={s:2d} -> {reused[-1]}")
    print(f"  distinct: {len(set(reused))}/{len(SEEDS)}")

    print("--- same seed twice on one handle (determinism) ---")
    a = short(handle.reset(0)["gamefile"])
    b = short(handle.reset(0)["gamefile"])
    print(f"  seed=0 -> {a}\n  seed=0 -> {b}\n  deterministic: {a == b}")

    print("--- fresh handle per seed ---")
    fresh = []
    for s in SEEDS[:4]:
        fresh.append(short(AlfredEnvHandle(base).reset(s)["gamefile"]))
        print(f"  seed={s:2d} -> {fresh[-1]}")
    print(f"  distinct: {len(set(fresh))}/{len(fresh)}")


if __name__ == "__main__":
    main()
