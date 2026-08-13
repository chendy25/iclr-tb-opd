#!/usr/bin/env python3
"""Job-side preflight for the in-repo ALFWorld agent loop.

Run inside the job container (needs torch/ray/vllm + alfworld installed):

    ALFWORLD_DATA=/.../alfworld python3 recipe/alfworld_opd/preflight_inrepo.py

Checks, in order:
  1. ``alfworld_agent`` is registered in verl's agent-loop registry (import wiring).
  2. A base AlfredTWEnv builds, and one handle can reset() then step() a game.
Neither step launches training or touches the GPU.
"""

import os
import sys


def check_registry() -> None:
    from verl.experimental import agent_loop  # noqa: F401  (triggers registration)
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    assert "alfworld_agent" in _agent_loop_registry, (
        f"alfworld_agent not registered; have: {sorted(_agent_loop_registry.keys())}"
    )
    print("[preflight] registry OK: alfworld_agent ->", _agent_loop_registry["alfworld_agent"])


def check_env_roundtrip() -> None:
    from verl.experimental.agent_loop.alfworld_env.env_pool import _get_base_env, AlfredEnvHandle

    if not os.environ.get("ALFWORLD_DATA"):
        print("[preflight] WARN: ALFWORLD_DATA unset; skipping env round-trip")
        return

    base = _get_base_env(None, os.environ.get("ALFWORLD_TRAIN_EVAL", "train"))
    handle = AlfredEnvHandle(base)
    reset = handle.reset(seed=0)
    print("[preflight] reset OK | gamefile=", reset.get("gamefile"))
    print("[preflight] obs head:", (reset["observation"] or "")[:120].replace("\n", " "))
    n_adm = len(reset.get("admissible_commands") or [])
    print("[preflight] #admissible=", n_adm)
    action = "look"
    if reset.get("admissible_commands"):
        action = next((a for a in reset["admissible_commands"] if a != "help"), "look")
    step = handle.step(action)
    print(f"[preflight] step('{action}') OK | won={step['won']} done={step['done']}")


def main() -> int:
    check_registry()
    try:
        check_env_roundtrip()
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] env round-trip FAILED: {type(e).__name__}: {e}")
        return 1
    print("[preflight] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
