# Agentic TB-OPD (Turn-Branching On-Policy Distillation) — Phase 1'

Turn-level instance of the TB-OPD operator: at the highest-uncertainty *post-tool*
assistant turn of a multi-turn tool-use trajectory, expand candidate sub-trajectories
(forked rollouts that re-enter the real tool loop) and supervise the whole turn tree
with teacher dense KD. See `docs/proposals/agentic_tb_opd_research.md` for the full
proposal (§9 execution plan).

## What's implemented here

| Piece | Location |
|---|---|
| Code tool — E2B backend (`code_interpreter`, TIR) | `verl/tools/e2b_tool.py` (`E2BTool`) |
| Code tool — SandboxFusion backend | `verl/tools/sandbox_fusion_tools.py` (`SandboxFusionTool`, `CustomSandboxFusionTool`) |
| Tool configs | `config/e2b_tool_config.yaml`, `config/sandbox_fusion_tool_config.yaml` |
| Turn config (`fork_unit=turn`, hybrid metric, budget B, AEPO penalty) | `verl/workers/config/distillation.py` (`TBOPDConfig`), `verl/trainer/config/distillation/distillation.yaml` |
| Turn fork selection (ATOD Soft-OR + ARPO ΔH_post-tool, NLL proxy) | `verl/experimental/agent_loop/tb_opd.py` (`select_fork_turn`, `segment_assistant_turns`, `topk_candidates_at`) |
| Breakpoint resume through the tool loop (E4) | `verl/experimental/agent_loop/tool_agent_loop.py` (`ToolAgentLoop.run_from_prefix`) |
| Worker fan-out for turn branches | `verl/experimental/agent_loop/agent_loop.py` (`_run_tb_opd_group_turn`, `_tb_generate_branch_turn`) |
| Tool-token loss mask (E5) | inherited: tool tokens have `response_mask=0`; `verl/trainer/distillation/losses.py` masks by `response_mask` |
| Turn-reweight OPD (B-A1, reweight-only) | `verl/trainer/distillation/turn_reweight.py` (`compute_turn_reweight`), injected in `losses.py::distillation_loss` |
| Data prep | `prepare_open_agentrl.py` |
| Training arms | `run_phase1_B-A0.sh`, `run_phase1_B-A1.sh`, `run_phase1_M.sh`, `train_agentic_tbopd.sh` |

## Provenance (borrowed, verified)

- **ATOD** (`refs/ATOD`, `verl/trainer/ppo/atod_utils.py::compute_tip_weights`): the
  Soft-OR `1-(1-nd)(1-ne)` combination of teacher–student disagreement `d_k` and the
  NLL entropy proxy `h_k = mean(-logp)`, and the lesson that entropy-only selection is
  weaker than the two-signal hybrid. We reuse the *signal math*, not ATOD's advantage-
  based OPD (we keep this fork's own KD loss).
- **ARPO/AEPO** (`refs/ARPO/ARPO/verl_arpo_entropy/.../vllm_rollout_with_tools.py`): the
  ΔH_post-tool idea (entropy spike after a tool response is the natural decision point,
  measured against an initial-entropy baseline) and per-step branch rollout. We reuse the
  *selection timing* and the *branch-rollout pattern*, not ARPO's RL outcome objective.

## Sandbox backend (decoupled from branching)

The `code_interpreter` tool is a `BaseTool` on `ToolAgentLoop`; the turn-level
branching (`select_fork_turn` + `run_from_prefix`) is **backend agnostic**. Two
interchangeable backends ship here, selected by `SANDBOX_BACKEND` (which just
picks the `tool_config_path` yaml):

| `SANDBOX_BACKEND` | tool | needs |
|---|---|---|
| `e2b` (default) | `E2BTool` | `pip install e2b` + `E2B_API_KEY` (+ `E2B_DOMAIN` for self-hosted) |
| `sandbox_fusion` | `CustomSandboxFusionTool` | a running SandboxFusion svc at `SANDBOX_FUSION_URL` |

Why not opd_dev's E2B *runner*: that stack (`recipe_custom/agent/runners`) runs a
whole-trajectory harness inside the sandbox and returns only a reward, hiding the
per-turn token ids / logprobs / prefix-resume hooks turn-branching needs. `E2BTool`
reuses only E2B's create/run/kill primitive (cf. `opd_dev .../sandbox.py::E2BSandbox`)
as a per-turn tool, so branching stays inside verl's `ToolAgentLoop`.

## How to run (SOD stack)

```bash
# 1. Build the tool-agent TIR parquet from Open-AgentRL-30K + Eval.
python -m recipe.agentic_tbopd.prepare_open_agentrl

# 2a. Default backend E2B: just export your key (self-hosted also sets E2B_DOMAIN).
export E2B_API_KEY=...          # export E2B_DOMAIN=e2b.your.host  (optional)

# 2b. Or use SandboxFusion instead:
#     docker run -p 8080:8080 volcengine/sandbox-fusion:server-20250609
#     export SANDBOX_BACKEND=sandbox_fusion SANDBOX_FUSION_URL=http://localhost:8080/run_code

# 3. Launch an arm (RANK/MASTER_* set by the cluster; 1 node student + 1 node teacher).
bash recipe/agentic_tbopd/run_phase1_B-A0.sh     # agent OPD baseline (no fork, no reweight)
bash recipe/agentic_tbopd/run_phase1_B-A1.sh     # turn-reweight OPD (reweight-only, no fork)
bash recipe/agentic_tbopd/run_phase1_M.sh        # TB-OPD-Turn: turn expansion (this paper)
```

The three arms are the **"展开 vs 只重加权"** ablation: B-A0 (neither), B-A1
(reweight the KD loss on uncertain turns, no expansion), M (expand branches at the
uncertain turn). B-A1 is a pure loss-side flag (`tb_opd.turn_reweight=True`,
`enable=False`) and shares no code with M's rollout path.

Defaults: Teacher `data/models/SOD-GRPO_teacher-4B`, Student `data/models/Qwen3-1.7B`.

## Known limitations / remaining work

- **Teacher-at-rollout disagreement is intentionally off for now.** Both selection
  (M) and reweighting (B-A1) use the **entropy-only** signal (`ΔH_post-tool` /
  `mean(-logp)`); `fork_metric`/`reweight_metric` are set to `dHtool`/`ent`. Scoring
  the main trajectory's turns with the teacher at rollout time is not yet available,
  so the two-signal Soft-OR (`disagree`/`hybrid`) is deferred. The code paths for
  `disagree`/`hybrid` exist and degrade gracefully, but are not used in the arms.
- **End-to-end training not yet run.** Needs `pip install e2b` + `E2B_API_KEY`
  (default backend) or a SandboxFusion service, plus the teacher/student checkpoints
  under `data/models/`.
- **forced_topk at a turn** forces only the turn's first token as an alternative; a forced
  token that is itself a tool-trigger is not re-parsed on the first resumed step (content-
  level alternatives are unaffected). `branch_mode=resample` avoids this entirely.
