#!/usr/bin/env python
"""Measure the cost of restoring ALFWorld state by replaying actions from reset.

TB-OPD forks a trajectory at a mid-episode turn and regenerates the rest. In the
tool-agent setting the fork only needs the token prefix, but ALFWorld carries a
stateful simulator that has already advanced past the fork point.

TextWorld offers no snapshot/restore: PddlEnv keeps its state inside a Fast Downward
shared library reached through ctypes (``apply_operator`` / ``get_state`` /
``check_goal``), and ``load_lib`` hands each env its own *copy* of that library, so
the state is neither deep-copyable nor addressable from Python. The only faithful
restore is therefore to reset a handle onto the same game and replay the actions.

Reports per-call reset/step latency and the end-to-end cost of restoring to a given
turn, which is what decides whether post-hoc branching is affordable.
"""

import os
import sys
import time

sys.path.insert(0, os.environ.get("CODE_DIR", "/mnt/afs_reason/chendongyang/code/iclr/verl"))

from verl.experimental.agent_loop.alfworld_env.env_pool import (  # noqa: E402
    AlfredEnvHandle,
    _get_base_env,
)

CONFIG = os.environ.get(
    "ALFWORLD_CONFIG_PATH",
    "/mnt/afs_reason/chendongyang/code/iclr/verl/verl/experimental/agent_loop/alfworld_env/config_tw.yaml",
)
SEEDS = [101, 202, 303, 404, 505]
FORK_TURNS = [10, 20, 30]


def _rollout(handle, seed, n_steps):
    """Walk a fixed pseudo-policy and record the exact actions sent to the env."""
    obs = handle.reset(seed)
    actions = []
    for i in range(n_steps):
        cmds = [c for c in obs.get("admissible_commands", []) if c != "help"]
        if not cmds:
            break
        a = cmds[i % len(cmds)]
        actions.append(a)
        obs = handle.step(a)
        if obs.get("done"):
            break
    return actions


def main():
    base = _get_base_env(CONFIG, "train")
    handle = AlfredEnvHandle(base)
    replay_handle = AlfredEnvHandle(base)

    resets, steps = [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        obs = handle.reset(seed)
        resets.append(time.perf_counter() - t0)
        for i in range(10):
            cmds = [c for c in obs.get("admissible_commands", []) if c != "help"]
            if not cmds:
                break
            t0 = time.perf_counter()
            obs = handle.step(cmds[i % len(cmds)])
            steps.append(time.perf_counter() - t0)
            if obs.get("done"):
                break

    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    print(f"reset : n={len(resets):3d} mean={mean(resets)*1000:8.1f} ms  max={max(resets)*1000:8.1f} ms")
    print(f"step  : n={len(steps):3d} mean={mean(steps)*1000:8.1f} ms  max={max(steps)*1000:8.1f} ms")

    print("\nrestore-to-turn cost (reset + replay of N actions), and state agreement:")
    for fork in FORK_TURNS:
        costs, agree = [], []
        for seed in SEEDS:
            actions = _rollout(handle, seed, fork)
            if len(actions) < fork:
                continue
            live = handle.step(actions[-1]) if False else None  # keep handle untouched
            # Reference: what the original run saw at the fork point.
            ref_actions = _rollout(handle, seed, fork)
            ref_obs = handle.step(ref_actions[-1]) if False else None

            t0 = time.perf_counter()
            obs = replay_handle.reset(seed)
            for a in actions:
                obs = replay_handle.step(a)
            costs.append(time.perf_counter() - t0)

            # Re-run the same prefix on the original handle and compare the state the
            # branch would continue from: identical admissible sets means the replay
            # landed on the same logical state.
            obs2 = handle.reset(seed)
            for a in actions:
                obs2 = handle.step(a)
            agree.append(
                sorted(obs.get("admissible_commands", [])) == sorted(obs2.get("admissible_commands", []))
                and obs.get("observation") == obs2.get("observation")
            )
        if costs:
            print(
                f"  fork@turn {fork:3d}: mean={mean(costs)*1000:8.1f} ms  "
                f"max={max(costs)*1000:8.1f} ms  identical_state={sum(agree)}/{len(agree)}"
            )


if __name__ == "__main__":
    main()
