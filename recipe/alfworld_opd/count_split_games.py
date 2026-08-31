#!/usr/bin/env python3
"""Print AlfredTWEnv registered game counts for valid_seen / valid_unseen."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("CODE_DIR", "/mnt/afs_reason/chendongyang/code/iclr/verl"))

from verl.experimental.agent_loop.alfworld_env.env_pool import _get_base_env  # noqa: E402

CONFIG = os.environ.get(
    "ALFWORLD_CONFIG_PATH",
    "/mnt/afs_reason/chendongyang/code/iclr/verl/verl/experimental/agent_loop/alfworld_env/config_tw.yaml",
)


def main() -> None:
    for key, name in (
        ("eval_in_distribution", "valid_seen"),
        ("eval_out_of_distribution", "valid_unseen"),
    ):
        env = _get_base_env(CONFIG, key)
        n = len(getattr(env, "game_files", []) or [])
        print(f"{name} {key} {n}")


if __name__ == "__main__":
    main()
