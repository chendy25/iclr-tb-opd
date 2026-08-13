# ALFWorld Native OPD (Phase 2')

Cross-env **ALFWorld** on-policy distillation (OPD), implemented **natively in this
repo** (`iclr/verl`). No `refs/ATOD` at train time.

- **Env interaction:** verl V1 agent loop `alfworld_agent`
  (`verl/experimental/agent_loop/alfworld_agent_loop.py`) — one ALFWorld episode
  per `run`. Assistant tokens masked `1`, environment-observation tokens masked `0`.
- **Objective:** the repo's own distillation OPD, identical machinery to Phase 1'
  agentic TB-OPD arm B-A0:
  `loss_mode=k1` + `use_policy_gradient=True` + `use_task_rewards=False`
  ⇒ per-token PG advantage `= logπ_T − logπ_S` (**pure OPD**, no task reward).
- This keeps ALFWorld on the same V1 stack / TB-OPD main line, so token- and
  turn-level branching (TB-OPD / TB-OPD-Turn) can be enabled later with the same
  `distillation.tb_opd.*` knobs.

## Status

| Item | Value |
|------|--------|
| Job | `pt-gv791b30` (2×8, `agentic-rl`) — **launch only when free** |
| Student | `Qwen3-4B` |
| Teacher | `Qwen3-30B-A3B` (base) |
| Train code | **in-repo** `python -m verl.trainer.main_ppo` (`recipe/alfworld_opd/train_inrepo_opd.sh`) |
| Rollout | agent loop `alfworld_agent` (`rollout.mode=async`) |
| OPD | `distillation.enabled=True`, `loss_mode=k1`, `use_policy_gradient=True`, `use_task_rewards=False` |
| Env data | `ALFWORLD_DATA=~/.cache/alfworld` (`json_2.1.1` + `logic/`) |
| Stub parquet | `code/data/verl-agent/text/{train,test}.parquet` (offline, drives batch size only) |

The stub parquet carries no task content — ALFWorld games are sampled from
`$ALFWORLD_DATA`. `extra_info.index` seeds game selection so a prompt's `rollout.n`
replicas play the **same** game (valid GRPO/OPD grouping).

## Files

```
verl/experimental/agent_loop/
  alfworld_agent_loop.py        # AlfWorldAgentLoop (@register("alfworld_agent"))
  alfworld_env/
    __init__.py
    config_tw.yaml              # AlfredTWEnv config ($ALFWORLD_DATA expanded)
    env_pool.py                 # per-process TextWorld env pool (blocking -> executor)
    projection.py               # <think>/<action> parse (ATOD contract)
    prompts.py                  # initial + follow-up user turns
  __init__.py                   # imports AlfWorldAgentLoop to register it

recipe/alfworld_opd/
  train_inrepo_opd.sh           # PRIMARY: main_ppo + k1 PG OPD (4B <- 30B-A3B)
  preflight_inrepo.py           # job-side: registry + one env reset/step
  prepare_text_stubs.py         # offline parquet (no HF geometry3k)
  install_job_deps.sh           # pip --user alfworld/textworld/... on the job python
  check_env.sh                  # paths + imports (STRICT=1 on job)
  run_native_opd.sh             # legacy: refs/ATOD main_sod fallback
  README.md
iclr/logs/_relaunch_alfworld_opd.sh          # STACK=inrepo (default) | atod
iclr/logs/_launch_alfworld_opd_via_sco.sh    # DRY_RUN=1 by default
iclr/scripts/run_alfworld_native_opd_on_debug_16p.sh
```

## Preflight

```bash
# login node (paths + soft import warnings)
bash .../recipe/alfworld_opd/check_env.sh

# first time on the job (shared AFS user-site; master pod is enough)
#   python3 iclr/logs/_sco_exec.py --worker master --wait-s 600 -- \
#     'bash .../recipe/alfworld_opd/install_job_deps.sh'

# on the job: full stack + real env round-trip (no GPU, no training)
ALFWORLD_DATA=~/.cache/alfworld python3 .../recipe/alfworld_opd/preflight_inrepo.py

# optional: regenerate stubs only
python3 .../prepare_text_stubs.py \
  --local_dir /mnt/afs_reason/chendongyang/code/data/verl-agent/text \
  --train_data_size 16 --val_data_size 128
```

`textworld` / `alfworld` are not in the debug job's `/opt/conda` (py3.11) +
`iclr_py311_user` by default, so run `install_job_deps.sh` once before the first
ALFWorld train.

## Launch (when `pt-gv791b30` is free)

`STACK=inrepo` is the default.

```bash
# dry-run (default)
bash /mnt/afs_reason/chendongyang/code/iclr/logs/_launch_alfworld_opd_via_sco.sh

# live
DRY_RUN=0 bash /mnt/afs_reason/chendongyang/code/iclr/logs/_launch_alfworld_opd_via_sco.sh
```

Or manually (in-repo stack):

```bash
cd /mnt/afs_reason/chendongyang/code/iclr/logs
python3 _sco_exec.py --worker master --wait-s 240 -- \
  'STACK=inrepo ROLE=master MASTER_ADDR=10.120.3.173 bash /mnt/afs_reason/chendongyang/code/iclr/logs/_relaunch_alfworld_opd.sh'
python3 _sco_exec.py --worker worker --wait-s 240 -- \
  'STACK=inrepo ROLE=worker MASTER_ADDR=10.120.3.173 bash /mnt/afs_reason/chendongyang/code/iclr/logs/_relaunch_alfworld_opd.sh'
```

Logs: `iclr/logs/alfworld_inrepo_opd_qwen3_4b_from_30ba3b/`.

## Knobs (in-repo stack)

| Env | Default | Meaning |
|-----|---------|---------|
| `ROLLOUT_N` | `8` | `rollout.n` (GRPO/OPD group; replicas share a game) |
| `TRAIN_BATCH_SIZE` | `16` | prompts per step (= stub train rows) |
| `MAX_RESPONSE_LENGTH` | `4096` | full episode transcript budget |
| `ALFWORLD_MAX_STEPS` | `30` | max env steps (assistant turns) per episode |
| `ALFWORLD_POOL_SIZE` | `16` | TextWorld envs per rollout process |
| `ALFWORLD_MAX_TURN_TOKENS` | `512` | per-turn generation cap |
| `ALFWORLD_TRAIN_EVAL` | `train` | env split (`train` / `eval_out_of_distribution`) |
| `DISTILLATION_LOSS_MODE` | `k1` | reverse-KL OPD |
| `USE_POLICY_GRADIENT` | `True` | KD-as-advantage (native OPD) |
| `DISTILLATION_TOPK` | `64` | teacher top-k logits |
| `TEACHER_TP` / `ROLLOUT_TP` | `8` / `2` | teacher / student rollout TP |

## Notes / follow-ups

- **Eval split:** rollout currently always uses the `train` games. A proper unseen
  eval (`eval_out_of_distribution`) needs the core agent-loop to thread `validate`
  into `AgentLoopBase.run`; until then override with `ALFWORLD_TRAIN_EVAL`.
- **Legacy ATOD path:** `STACK=atod` (`run_native_opd.sh`, `main_sod` +
  `mode=uniform`/`opd_only=true`) is kept only as a fallback.
