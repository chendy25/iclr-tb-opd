#!/usr/bin/env python
"""Check that concurrent ALFWorld env calls no longer corrupt TextWorld's PDDL parser.

Training showed episodes dying inside ``tatsu`` (``IndexError`` / ``FailedToken`` /
``'NoneType' object is not iterable``) while a sequential probe ran clean. ALFWorld
parses PDDL through a parser holding non-per-instance state, and with ``batch_size=1``
TextWorld uses SyncBatchEnv, so that parse runs in whichever executor thread the env
pool picked. ``env_pool`` now serializes reset *and* step behind one process lock.

Runs a reset+step workload sequentially (control) and threaded (the shipped path) and
reports failures for each; threaded must reach zero for the fix to hold.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("CODE_DIR", "/mnt/afs_reason/chendongyang/code/iclr/verl"))

from verl.experimental.agent_loop.alfworld_env.env_pool import (  # noqa: E402
    AlfredEnvHandle,
    _get_base_env,
)

CONFIG = os.environ.get(
    "ALFWORLD_CONFIG_PATH",
    "/mnt/afs_reason/chendongyang/code/iclr/verl/verl/experimental/agent_loop/alfworld_env/config_tw.yaml",
)
N_HANDLES = 8
N_EPISODES = 24
N_STEPS = 6


def run(handles, seeds, threaded):
    errors: dict[str, int] = {}

    def one(i):
        handle = handles[i % len(handles)]
        try:
            obs = handle.reset(seeds[i])
            for _ in range(N_STEPS):
                cmds = [c for c in obs.get("admissible_commands", []) if c != "help"]
                if not cmds:
                    break
                obs = handle.step(cmds[i % len(cmds)])
                if obs.get("done"):
                    break
            return None
        except Exception as e:  # noqa: BLE001 - counting failure modes is the point
            return type(e).__name__

    if threaded:
        with ThreadPoolExecutor(max_workers=len(handles)) as ex:
            results = list(ex.map(one, range(len(seeds))))
    else:
        results = [one(i) for i in range(len(seeds))]

    for r in results:
        if r is not None:
            errors[r] = errors.get(r, 0) + 1
    label = "threaded" if threaded else "sequential"
    print(f"  {label:12s} failures={sum(errors.values())}/{len(seeds)}  {errors}")


def main():
    base = _get_base_env(CONFIG, "train")
    print(f"building {N_HANDLES} handles ...")
    handles = [AlfredEnvHandle(base) for _ in range(N_HANDLES)]
    seeds = list(range(100, 100 + N_EPISODES))
    print(f"{N_EPISODES} episodes (reset + {N_STEPS} steps) over {N_HANDLES} handles:")
    run(handles, seeds, threaded=False)
    run(handles, seeds, threaded=True)


if __name__ == "__main__":
    main()
