# Token-Level Branching for On-Policy Distillation

## 0. 一句话定位

在 online policy distillation（OPD）中，于学生高熵/高概率分叉点系统展开 top-k 候选为共享前缀的轨迹树，用教师对树节点做 token-level 监督——把 Unmasking 式 enrichment 从诊断变成训练算子，把 CURE 式分支从 RL 迁到 dense KD。

**建议方法名**：TB-OPD（Tree-Branching On-Policy Distillation）或 ForkKD。

**实验唯一设定**：训练 `DAPO-Math-17K`；Teacher→Student `Qwen3-8B → Qwen3-4B`；评测 MATH-500 / AIME24 / AIME25（avg@16）；等生成 token 预算对比。不设小模型短跑轨道。

**冻结长度（与 B1 对齐）**：`max_response_tokens = 16384`，`max_model_len = 18433`（2048 prompt + 16384 + 1）。Phase 0 / 训练 / 评测不得用更短截断冒充诊断（v1 Phase 0 曾因 2k/1k 截断把 P_succ 压到地板）。

**默认方法 M\***：Only-fail + B=1 + k=2；`fork_metric=entropy`（对照 topk_gap / hybrid）；`fork_token_filter=math_aware`；选点 **Scheme B**（主生成带 top-k，省二次 forward）；整树 reverse KL。

**代码落点**：`iclr/verl`（`distillation.tb_opd.*`，默认 `enable=False`，B1 零影响）；诊断脚本 `iclr/scripts/phase0_psucc_diagnose.py`；训练脚本 `iclr/scripts/train_*.sh` / `srun_*.sh`。

---

## 1. 核心研究问题

**假设**：学生分布上真正值得花采样预算的位置，是少数高熵 forking tokens；在这些位置对 top-k 候选做显式分支，比（a）单轨迹 OPD、（b）等量独立多轨迹、（c）仅对高熵 token 加权但不分支，更能提升蒸馏效率与最终准确率。

**待验证的方法**：

1. 学生按当前策略生成主轨迹，记录每步熵与 top-k；
2. 在满足分叉准则的位置，强制展开 top-k 个候选 token，各自续写到 EOS（或截断预算）；
3. 教师对树中所有（或加权后的）轨迹提供 token-level KD；
4. 共享前缀只算一次 forward（工程上可选，科学上先保证逻辑正确）。

---

## 2. 与已有工作的关系（修正后的 novelty）

### 2.1 精确组合：尚未见完整训练方法

下列零件都已被验证，但 **「OPD 训练 + 系统 top-k 全展开 + 教师 KD」** 未见成文方法：


| 零件                                        | 代表工作                                                 | 已证明什么                                 | 与我们的差距                    |
| ----------------------------------------- | ---------------------------------------------------- | ------------------------------------- | ------------------------- |
| 高熵 forking tokens 驱动推理学习                  | Beyond 80/20 (2506.01939), ReasonMaxxer (2605.06241) | 稀疏高熵位置承载大部分 RL 收益                     | 不展开分支，只做稀疏更新              |
| 关键 token 处分支再训练                           | CURE (2508.11016)                                    | 分支+联合优化可防熵塌、提准确率                      | RL + 重采样，非 top-k 全展开，非 KD |
| OPD 底座                                    | GKD (2306.13649), MiniLLM                            | 学生 rollout + 教师 logits 有效             | 默认单/独立轨迹，无分支策略            |
| OPD 中按熵/分歧选监督位置                           | TIP (2604.14084), SEAD (2606.28562), EGRSD/DASD      | fork 点应特殊处理 loss                      | **选/加权**，不生成额外分支          |
| OPD 中建 generation tree + targeted rollout | Unmasking OPD (2605.10889, Apple)                    | 分支点可估 P_{\text{succ}}；enrichment 可扩展树 | **诊断框架**，明确把训练用法列为未来工作    |


**可主张的差异（写进 contribution，勿写“大片空白”）**：

1. 训练算子：分支轨迹直接进入 OPD，而非 Unmasking 诊断；
2. 分支策略：系统展开 top-k，而非 CURE 随机重采样；
3. 信号：教师 dense KD（可加 outcome 加权），而非纯 RLVR；
4. 效率叙事：共享前缀树 vs 独立 BoN（等 token 预算对比）。

### 2.2 关键邻近工作（必须 cite）

- **Beyond the 80/20 Rule** (Wang et al., 2506.01939)：~20% 高熵 token 是 forking tokens；只更新它们可匹配/超过全 token RL。
- **CURE** (2508.11016)：高熵点截断→重生成→与原轨迹联合 GRPO/DAPO。
- **Unmasking OPD** (2605.10889)：generation tree + targeted enrichment；发现教师信号在**错误轨迹**上更可靠。
- **TIP** (2604.14084)：高熵 + **低熵高分歧**都重要——提醒我们不能只盯高熵。
- **SEAD / EGRSD / DASD**：高熵 fork 处强跟教师可能伤探索；fork 处宜用 mode-covering（FKL）或降权。
- **GKD / 标准 OPD**：主基线。
- 次要：BOND、ReST、ToT、rStar-Math、SpecInfer（概念/效率参考）。

### 2.3 谱系（更新）

```
独立多序列采样 ← ReST, BoN, STILL-2
       ↓
OPD 单/独立轨迹 ← GKD
       ↓
OPD 内熵路由（不分支）← TIP, SEAD, EGRSD, DASD
       ↓
高熵 forking 前提 ← 80/20, ReasonMaxxer
       ↓
关键 token 分支 (RL) ← CURE
       ↓
OPD 中建树+补采样 (诊断) ← Unmasking OPD
       ↓
【TB-OPD】系统 top-k 展开 + OPD 训练  ← 窄缝隙，需等 compute 证明
```

---

## 3. 方法草图（实验前先钉死接口）

### 3.1 分叉准则（主：熵；对照：top-k gap / hybrid）

对位置 t，学生分布 \pi_\theta(\cdot|s_t)：

- **Entropy gate**（默认）：H_t \ge \tau 或 batch / 轨迹内 top-p% 熵；可加绝对地板 `min_fork_entropy`；
- **Top-k gap**：p_{(1)}-p_{(2)} < \delta（多候选接近；压「均匀噪声高熵」）；
- **Hybrid**：高熵且 top-2 概率均超过 \epsilon（避免均匀噪声词）。

每个 prompt **最多分支 B 次**（默认 B=1，消融可试 2），每次展开宽度 k\in{2,3,4}（默认 2）。否则算力爆炸。

**Fork token 过滤（默认 math_aware）**：优先在数学/推理相关 token 上分支；不 blacklist `Wait`/`Hmm` 等犹豫词，也不误伤单字符数学 token。`strip_len` 仅作 negative control（GSM8K smoke：与 math_aware 差异很小）。

**选点实现**：

- **Scheme A**：主轨迹后再跑一次 student `prompt_logprobs`（二次 forward）；
- **Scheme B（默认）**：主生成时直接要 top-k logprobs，省一次 forward；smoke 上与 A 的 `scheme_b_pos_match` ≈ 88–91%。

### 3.2 展开方式（相对 CURE 的关键差异）

- **TB-OPD（本方法）**：取 top-k token，**每个强制走一条分支**（确定性展开候选集合）。
- **CURE-style 对照（B4）**：同一分叉点 **温度采样** k 次续写（随机重采样）。
- 两者等分支预算对比，才能证明“系统展开”有额外价值。

**工程：固定槽位 fan-out（不改 ray_trainer）**：设 `rollout.n = 1+k`。槽 0 = 主轨迹；槽 1..k = 分支（Only-fail 且主轨迹正确时退化为普通独立 rollout）。行数恒定，与 `batch.repeat(n).union(gen)` 兼容。AgentLoopManager 分片按 n 对齐，保证同 prompt 组不被切断。

### 3.3 损失（先简后繁）

令树中轨迹集合为 \mathcal{T}，教师 \pi_T：


\[\mathcal{L} = \sum_{\tau\in\mathcal{T}} w(\tau)\,
\frac{1}{|\tau|}\sum_t D\big(\pi_T(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t)\big)\]


推荐 D 先用 **reverse KL / JSD**（跟 GKD 对齐）。权重 w(\tau) 三档消融：

1. **Uniform**：w=1；
2. **Outcome**：正确轨迹 w=1+\alpha，错误 w=1 或按 group advantage；
3. **Fork-aware**（受 SEAD/DASD 启发）：分叉点用 forward KL（保多峰），其余用 reverse KL；或高熵点降权教师、只靠 outcome 选优分支。

**重要约束（来自 Unmasking）**：默认变体只在**主轨迹最终错误**时触发分支；正确轨迹只做标准 OPD（避免把噪声教师信号灌进已对路径）。

### 3.4 预算控制（实验科学性的核心）

固定以下之一作为主公平轴（论文主表选一个，附录报另一个）：

- **等生成 token 数**（推荐主表）：所有方法每个 prompt 生成的总 token 相同；
- **等教师 forward 次数**：教师打分成本对齐。

TB-OPD 的“免费午餐”只能来自分支把预算花在更高信息位置，而不是多花算力。

---

## 4. 实验设计

> **唯一主设定**：`DAPO-Math-17K` + `Qwen3-8B → Qwen3-4B` + `avg@16`。不再设 1.7B / 小模型短跑轨道；诊断与主表训练都在该设定上完成。

### 4.0 冻结协议（全文唯一）


| 项                 | 取值                                                               |
| ----------------- | ---------------------------------------------------------------- |
| 训练数据              | **DAPO-Math-17K**                                                |
| Teacher → Student | **Qwen3-8B → Qwen3-4B**（teacher：Instruct 或开源 GRPO ckpt，与 TIP 对齐） |
| OPD loss          | reverse KL（主）；附录可报 JSD                                           |
| 训练采样              | T=1.0；**max_response=16384**；max_model_len=18433；max_prompt=2048 |
| 评测                | MATH-500、AIME24、AIME25；主报 **avg@16**；附录可加 AMC23 / Avg@8 / Pass@8 |
| 公平轴               | 每 prompt **matched generation tokens** C                         |
| 默认方法 M\*          | Only-fail，B=1，k=2，entropy + math_aware，Scheme B，整树 reverse KL   |


### Phase 0 — 诊断 v2（主设定上做，可与 B1 并行）

**目的（问题已修正）**：验证「在高熵点 **强制 top-k 展开**，是否比 **等预算不强制重采样 / 单轨迹续写** 更能救回错误主轨迹」；并用 **hi vs lo 候选方差** 支撑「高熵点才真有分叉价值」。用作 Figure 1 / 机制段，**不挡主实验开工**。

> **v1 教训（已弃用）**：曾用 `max_response=2048` / `branch=1024` +「高熵位置 fork vs 随机位置 fork」。截断导致 `main_correct≈0.33%`、P_succ 落在零死区，Δ(高熵−随机)≈0.008 无信息。旧指标只回答「选点」，不回答方法主 claim「要不要分支」。v1 结果备份在 `iclr_phase0_psucc_dapo/v1_backup/`。

#### 4.0.1 Phase 0 v2 设置（与 B1 长度对齐）

| 项 | 取值 |
| --- | --- |
| 学生 | **Qwen3-4B**（与主实验相同） |
| 数据 | DAPO-Math-17K 抽 **300** 题 |
| 主轨迹 | G=6；T=1.0；**max_response=16384**；logprobs_k=100 |
| 触发 | **only_fail=1**（主轨迹错才分支，对齐 M\*） |
| 选点 | 同轨迹：hi = 熵 ≥80 分位且 ≥ `min_fork_entropy`；lo = 熵 ≤20 分位（方差对照） |
| **臂 A fork** | hi 点强制 top-k（k=2）各续写 **m=4** 次 |
| **臂 C resample** | 同 hi 前缀、**不强制**、k×m 次自然续写（等预算 no-fork） |
| **臂 D continue** | 同 hi 前缀贪心续写 1 次（单轨迹参考） |
| 臂 A@lo | lo 点 fork，m_lo=2（仅方差 / 回归，不是旧 random 臂） |
| 分支预算 | **自适应**：`min(16384 − fork_pos, 18433 − prompt − prefix − 1)`，硬顶 16384 |
| 脚本 | `iclr/scripts/phase0_psucc_diagnose.py`；launcher `run_phase0_on_debug_4gpu.sh` |

#### 4.0.2 Phase 0 v2 主指标（替代旧 entropy−random）

| 旧指标（v1，弃用） | 新指标（v2） | 现在证的是 |
| --- | --- | --- |
| 高熵 − 随机 的 P_succ 差 | **`recover.fork − recover.resample`** | 等预算下强制展开 vs 不强制重采样谁更能救回 |
| （无「要不要分支」） | **`recover.fork − success_continue`** | 分支 vs 单轨迹贪心续写 |
| 高熵 − 随机 的 ΔP 差 | **`cand_var_hi − cand_var_lo`** + `regression_var_on_entropy` | 高熵点候选间成功率是否更「分叉」 |
| 二值 ΔP（每候选 1 次） | **m 次续写估计的真实率** + `delta_top1_top2`（仅在 hi） | top1 vs top2 是否有实质差距 |

**`recover(arm)`**：仅在主轨迹错误子集上，P(≥1 条分支正确 | main wrong)。

**统计**：配对符号检验；按 prompt 聚类的 bootstrap 95% CI；`PASS_HINTS`（fork>resample / fork>continue / CI 排除 0 / hi 方差更大）。

**通过标准（建议）**：

1. `recover.fork > recover.resample` 且 bootstrap CI 倾向排除 0（主 claim）；
2. `cand_var_hi > cand_var_lo`（选点机制）；
3. `main_solve_rate` 回到合理区间（不再是截断死区的 ~0）。

若不成立：主实验仍可跑完，但机制叙事需收缩；「只是多采样」威胁变大。

### Phase 1 — 主实验（直接开跑）


| ID    | 方法                    | 是否必跑   | 说明                                   |
| ----- | --------------------- | ------ | ------------------------------------ |
| B0    | SFT on teacher traces | 建议     | 下界                                   |
| B1    | 标准 OPD（全 token）       | **必跑** | 现代 OPD 底座                            |
| B2    | OPD-Indep-N           | **必跑** | N 条独立 rollout，等总 token               |
| B3    | OPD-EntSelect         | **必跑** | 单轨迹 + top-50% 高熵 token 才算 KD（TIP 简化） |
| B4    | OPD-Resample          | **必跑** | 高熵点**重采样** k 支 + KD（CURE→蒸馏）         |
| **M** | **TB-OPD-Fail**       | **必跑** | Only-fail + B{=}1 + k{=}2 全展开        |
| M1    | TB-OPD-Always         | 主表有信号后 | 对照：不限错误轨迹                            |


**主结论三问**：

1. 等 token 预算下 M 是否打过 B1/B2？
2. 系统展开是否打过重采样 B4？（vs CURE）
3. 分支是否打过只选 token 的 B3？（vs TIP）

### Phase 2 — 消融（主表有正向信号后）

> **触发条件**：Table A 上 `Avg(M) > Avg(B1) ∧ Avg(M) > Avg(B2)` 至少成立一项；否则只做 A0/A1 的诊断性消融，不扩全表。
>
> **协议**：与 §4.0 相同（DAPO-Math-17K，Qwen3-8B→4B，matched tokens C，avg@16）。默认锚点 **M\*** = Only-fail + B=1 + k=2 + `fork_metric=entropy` + `fork_token_filter=math_aware` + reverse KL 整树。每次消融只改一个轴。

#### 4.2.1 必做消融（主表正向后优先）


| ID  | 轴              | 变体                                                        | 配置入口 / 备注                                                                 | 要回答的问题                                |
| --- | -------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------- |
| A0  | 分叉准则           | `entropy` / `topk_gap` / **`hybrid`**                     | `TB_FORK_METRIC`；hybrid = 高熵且 top-2 概率均 ≥ ε（代码待实现）                        | 选点靠“不确定”还是“多峰接近”？hybrid 是否更稳？        |
| A1  | 阈值             | entropy: τ∈{batch top-p%, 固定 τ}；topk_gap: δ；hybrid: (τ,ε) | `min_fork_entropy` 等                                                       | 过松→噪声分支，过紧→几乎不分支                      |
| A2  | 树宽 k           | k∈{2,3,4}（**等 token 预算**截断续写）                             | `TB_K`                                                                     | 加宽是否还有收益，还是预算被稀释？                     |
| A3  | 触发策略           | **Only-fail** / Always / mid-difficulty（可选）               | `only_fail`                                                                | Unmasking 叙事：是否只该在错轨迹上分支？             |
| A4  | 系统展开 vs 重采样    | M\*（top-k 全展开）vs B4（同点温度重采样）                               | 已是 Table A 的 M vs B4；消融表可复述                                                | 相对 CURE 的贡献是否成立？                     |
| A5  | 轨迹权重           | Uniform / Outcome(+α) / FKL@fork（或高熵降权教师）                 | loss 侧                                                                    | 收益来自采样还是来自 fork 处 loss 形态？（SEAD/DASD） |
| A6  | 续写长度           | 全长续写 vs 分叉后截断（固定剩余预算）                                      | max_response / branch budget                                               | 截断是否够用（对齐 “Full Rollouts?”）           |
| A7  | KD 作用域         | 整树 KD vs **仅分叉点及之后**                                       | mask / loss window                                                         | 共享前缀重复监督是否浪费？                         |


#### 4.2.2 工程 / 过滤消融（可先 smoke，再在主设定复验）


| ID  | 轴            | 变体                                       | 状态（截至 2026-07-30）                                                                                          | 主设定是否还要跑                            |
| --- | ------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| E1  | fork token 过滤 | `math_aware` vs `strip_len`              | **GSM8K smoke 25 step 已完成**：pos_frac / near_math 差异很小（mean Δ≈0.013）；默认保留 `math_aware`（不误伤单字符数学、不 blacklist Wait/Hmm） | 主设定各跑短训或抽样 dump 复验即可；**不必进主消融表**除非差异变大 |
| E2  | 选点实现         | Scheme A（二次 forward）vs **Scheme B**（主生成 top-k） | smoke：`scheme_b_pos_match` ≈ 88–91%；默认可开 Scheme B 省一次 forward                                            | 主设定开 B 即可；若担心温度/logprobs 不一致再开 `scheme_b_validate` 抽查 |
| E3  | 分支次数 B       | B∈{1,2}                                  | 未系统跑；算力敏感                                                                                                 | M 胜出后再做；B=2 必须严格锁 token 预算          |


**E1 smoke 结论（勿过度外推）**：在 Scheme B + entropy 下，`strip_len` 并未显著抬高 late-fork 或压低 near_math；正式训练仍以 **math_aware** 为默认，strip_len 仅作 negative control。

#### 4.2.3 机制 / 诊断消融（支撑正文 Figure，不挡主表）


| ID  | 内容                                                         | 依赖           | 产出                          |
| --- | ---------------------------------------------------------- | ------------ | --------------------------- |
| D0  | Phase 0 v2：`recover.fork` vs resample/continue；hi vs lo 候选方差 | DAPO 300 题，长度对齐 16384 | Figure 1；旧「高熵 vs 随机位置」不再作主图 |
| D1  | 选中 fork 的 `fork_pos_frac` / near_math / near_wait 分布        | dump + `analyze_fork_pos_frac.py` | 证明分支落在推理中段而非 EOS 附近        |
| D2  | 分支后学生 top-1 是否命中高 P_{\text{succ}} 候选（训练前后对比）               | 需 Phase 0 管线  | “学会在 fork 点选对分支”            |
| D3  | 训练熵曲线 / forking token 占比（相对 B1）                            | 日志           | 防熵塌；对齐 CURE/80/20          |


#### 4.2.4 执行优先级（Phase 2 内）

```
主表 M* 有信号后：
  1) A3 (Fail vs Always) + A2 (k=2 vs 4)     ← 直接改写 Table B 默认行
  2) A0 (entropy vs topk_gap；hybrid 实现后补) ← 选点轴
  3) A5 (Uniform vs Outcome；有余力再 FKL@fork)
  4) A6 / A7（效率与作用域）
  5) E3 (B=2) 仅当预算与实现都稳

并行不挡路：
  D0 Phase 0；E2 默认 Scheme B；E1 默认 math_aware
```

#### 4.2.5 报告指标（每条消融统一）

- **主**：MATH-500 / AIME24 / AIME25，avg@16（与 Table A 同协议）
- **公平**：Tokens/prompt ≈ C；可选 wall-clock
- **过程**：`tb_opd/fork_pos_frac`、branch_rate、fail_rate、actor/distillation/loss、训练熵
- **机制（有则报）**：near_math / near_wait、\Delta P_{\text{succ}}、scheme_b_pos_match

### Phase 3 — 扩展（可选）

- 第二模型对：如 Llama-70B→8B 或 Qwen2.5-14B→1.5B（对齐 TIP 多家族）
- 同数据短跑 GRPO vs TB-OPD（效率叙事）
- 机制：forking token 上高 P_{\text{succ}} 候选的 top-1 命中率（接 D2）

---

## 5. 我认为更好的实验开展方式（执行建议）

### 5.1 直接主设定，诊断并行

不再做 1.7B 短跑。工程上先打通 **Qwen3-4B + DAPO-Math-17K 的 B1**，同时用同模型抽子集跑 Phase 0 诊断图。

### 5.2 公平性：永远先锁预算再比方法

主表只用 **matched generation tokens**。额外报 “matched wall-clock”。  
避免 “分支了所以多采样所以更好”。 

### 5.3 基线三角：独立多轨迹 / 重采样分支 / 选 token 不分支

**B2、B3、B4 与 M 同为第一优先级。**

### 5.4 默认方法：Only-fail + B=1 + k=2

先证明窄树；再放开 Always / 更大 k。

### 5.5 损失先对齐标准 OPD

第一版只用 reverse KL，只改采样分布，增益归因于 tree sampling。

### 5.6 工程落地顺序（进度）

1. ~~verl / OPD 跑通 B1~~ → **进行中**：`iclr_opd_b1_dapo_formal_r16k`（resp=16384）；
2. ~~强制 next-token 续写 + TB-OPD 固定槽位~~ → **已实现**（`tb_opd.enable`，默认关）；
3. ~~熵门控 / Scheme B / math_aware~~ → **已实现**；GSM8K smoke 验证过过滤与 Scheme B；
4. Phase 0 v2（16384 对齐 + fork vs resample）→ **进行中**；
5. 待做：B2 / B3 / B4 / M\* 同预算主表；共享前缀 packing 可后做。

**运行注意**：训练栈用容器 Python + vLLM（与 clawgym 一致）；`conda/opd` 有依赖但镜像/vLLM 版本需对齐。Student / Teacher 分池（如 8+4 或 2×8）。

### 5.7 成功/失败的决策树

```
Phase 0 v2：recover.fork > recover.resample（及 CI）？
  否 → 机制叙事收缩；主表仍可跑，但「多采样」威胁更大
  是 → 支撑 Figure 1 / Only-fail + 高熵展开

主实验（8B→4B）等预算下：
M* > B1？
  否 → 查实现/阈值；仍否 → idea 可能不成立
  是 → M* > B2？
        否 → 只是多采样，改叙事或停
        是 → M* > B4？
              否 → 贡献缩成「CURE-KD」
              是 → 相对 B3 仍有增益或机制优势？
                    否 → 贡献偏「采样侧 TIP」
                    是 → 完整故事 → Phase 2/3
```

---

## 6. 风险与缓解


| 风险               | 来源         | 缓解                             |
| ---------------- | ---------- | ------------------------------ |
| 只是多花算力           | 审稿常识       | 等 token 预算；报准确率–compute 曲线     |
| 被说成 CURE 变体      | CURE       | 显式 B4 对照；强调 top-k 全展开 + KD     |
| 被说成 Unmasking 变体 | Unmasking  | 对方是诊断；我们给训练结果 + 在线环            |
| 高熵处强蒸馏伤探索        | DASD/EGRSD | Fork-aware loss；保熵曲线；Only-fail |
| 漏掉低熵高分歧 token    | TIP        | 附录加 Q3-token 补充监督，不与主方法绑死      |
| 分支爆炸             | 算力         | B=1,k=2；截断续写；每 batch 限树大小      |
| 截断假阴性（Phase 0）  | v1 实测      | 诊断/训练统一 16384；自适应分支预算          |
| 犹豫点 / 噪声高熵       | smoke 观察   | math_aware；topk_gap/hybrid 消融  |
| 选点二次 forward 贵   | 算力         | 默认 Scheme B；抽查 scheme_b_pos_match |


---

## 7. 预期贡献（论文可写三条）

1. **方法**：提出 TB-OPD——在 OPD 中于 forking tokens 做 top-k 轨迹树展开，并用教师 dense 监督整树（默认 Only-fail + 窄树）。
2. **实证**：在等生成预算下，相对独立多轨迹、CURE-style 重采样、TIP 式选损，量化分支策略的收益来源。
3. **分析**：Phase 0 用 recover(fork vs resample) + hi/lo 候选方差，连接 80/20 forking tokens、Unmasking 的 P_{\text{succ}} 与 OPD 训练动态，给出“何时 / 是否该分支”的实证规则。

---

## 8. 参考文献（精简优先级）


| 优先级 | 论文                                   | arXiv        | 用途                             |
| --- | ------------------------------------ | ------------ | ------------------------------ |
| P0  | GKD                                  | 2306.13649   | OPD 底座 / B1                    |
| P0  | Beyond 80/20                         | 2506.01939   | forking token 前提               |
| P0  | CURE                                 | 2508.11016   | 分支训练 / B4                      |
| P0  | Unmasking OPD                        | 2605.10889   | 树与 P_{\text{succ}} / Only-fail |
| P0  | TIP                                  | 2604.14084   | B3；低熵高分歧提醒                     |
| P1  | ReasonMaxxer                         | 2605.06241   | 稀疏决策点                          |
| P1  | SEAD / EGRSD / DASD                  | 2606.28562 等 | fork 处 loss 设计                 |
| P1  | BOND                                 | 2407.14622   | 序列级 BoN 对照叙事                   |
| P2  | ToT / rStar / SpecInfer              | …            | 概念与系统参考                        |
| P2  | Are Full Rollouts Necessary for OPD? | 2605.31490   | 截断消融动机                         |


---

## 9. 执行清单与进度（截至 2026-07-31）

**已完成**

- [x] 文献定位与 novelty 收窄（CURE / Unmasking / TIP / 80/20）；
- [x] `iclr/verl` 实现 TB-OPD（固定槽位、Only-fail、Scheme B、math_aware；默认关）；
- [x] E1/E2 GSM8K smoke（math_aware vs strip_len；Scheme B pos_match）；
- [x] Phase 0 v1 跑完并诊断失败原因（截断）；备份旧结果；
- [x] Phase 0 v2 脚本重写（fork / resample / continue + hi/lo 方差 + 16384）；
- [x] B1 正式训启动：`iclr_opd_b1_dapo_formal_r16k`（修过 val NoneType bug）。

**进行中**

- [ ] B1 训满 675 steps（~36h 量级）；盯 step 50 val/ckpt；
- [ ] Phase 0 v2 跑完 300 prompts；读 `recover` / `PASS_HINTS`（与 B1 并行，master GPU 4–7）。

**待做**

- [ ] M\* 正式训 + B2 / B3 / B4 等预算对照 → Table A；
- [ ] M\* 若胜出：按 §4.2.4 跑 A3→A2→A0→A5（及 A6/A7）→ Table B；
- [ ] 准确率–compute 曲线 → Table C；
- [ ] 等 token 预算严格对照（暂缓，主表后再做）。

---

## 10. 主表模板与文献用法

> 协议已在 §4.0 冻结。此处只保留填表模板与“引用 vs 自跑”边界。

### 10.1 预算 C

B1：每 prompt 1 条，平均长 \bar L（实测 mean ~10k，max 16384），则 C \approx \bar L。  
B2：独立 N 条，截断使总 token \approx C。  
B4 / M\*：主轨迹 + 分支续写总 token \approx C（固定槽位 n=1+k 时，正确轨迹上额外槽位也计入预算）。

### 10.2 Table A — Main results（avg@16, matched tokens）

**设定栏（正文必写）**：DAPO-Math-17K；Teacher Qwen3-8B；Student Qwen3-4B。


| Method                     | Tokens/prompt | MATH-500 | AIME24 | AIME25 | Avg |
| -------------------------- | ------------- | -------- | ------ | ------ | --- |
| Student init               | —             |          |        |        |     |
| B0 SFT                     | —             |          |        |        |     |
| B1 OPD (full tokens)       | C             |          |        |        |     |
| B2 OPD-Indep-N             | C             |          |        |        |     |
| B3 OPD-EntSelect (top 50%) | C†            |          |        |        |     |
| B4 OPD-Resample (k{=}2)    | C             |          |        |        |     |
| **M TB-OPD-Fail**          | C             |          |        |        |     |
| M1 TB-OPD-Always（可选）       | C             |          |        |        |     |


† B3 生成预算 =C；旁注 peak mem（反传 token 更少）。

**判据**：Avg(M) > Avg(B1) ∧ Avg(M) > Avg(B2)；且 Avg(M) > Avg(B4)；相对 B3 有准确率或机制优势。

### 10.3 Table B / C

**Table B — Ablations**（同一 8B→4B 设定，固定 C；对应 §4 Phase 2 的 A\*）


| ID  | Variant                              | MATH-500 | AIME24 | AIME25 | Avg |
| --- | ------------------------------------ | -------- | ------ | ------ | --- |
| —   | Fail, k{=}2, B{=}1, entropy（M\* 默认） |          |        |        |     |
| A3  | Always, k{=}2, B{=}1                 |          |        |        |     |
| A2  | Fail, k{=}4, B{=}1（等预算）             |          |        |        |     |
| A0  | Fail, topk_gap（及 hybrid）            |          |        |        |     |
| A5  | Fail + outcome weight                |          |        |        |     |
| A5  | Fail + FKL@fork only                 |          |        |        |     |
| A6  | Fail + truncated branch              |          |        |        |     |
| A7  | Fail + KD only post-fork             |          |        |        |     |


工程默认（不必占满 Table B 行）：`math_aware` 过滤、Scheme B 选点；E1/E2 仅在附录或 smoke 注记。

**Table C**：准确率 vs 累计生成 tokens / wall-clock（B1 / B2 / B4 / M）。

### 10.4 文献软对照（不进主表）


| 来源   | 设定                             | 原文数字                                            | 用法           |
| ---- | ------------------------------ | ----------------------------------------------- | ------------ |
| TIP  | **同：8B→4B, DAPO, @16**         | OPD MATH 76.7 / A24 21.9；Soft-OR50% 79.1 / 25.7 | 量级参考；B3 仍须自跑 |
| EOPD | 1.7B-Base, Avg@8               | OPD/EOPD 小幅差                                    | 只作 OPD 形态参考  |
| CURE | Math-7B RLVR, DAPO-17K         | Avg 54.3；A24 35.5                               | 分支在 RL 有效    |
| OPSD | Instruct self-KD, OpenThoughts | 8B Avg 52.2                                     | 效率叙事         |
| GKD  | T5/GSM8K                       | —                                               | 方法史          |


### 10.5 自跑 vs 引用


| 必须自跑               | 只引用                                                 |
| ------------------ | --------------------------------------------------- |
| B1–B4、M（同代码/数据/预算） | 完整 CURE RL、完整 TIP Soft-OR、OPSD、Thinking Machines 长训 |
|                    | 可选同数据短跑 GRPO                                        |


### 10.6 一页纸

```
唯一设定：DAPO-Math-17K + Qwen3-8B→4B + avg@16 + matched tokens C
长度：max_response=16384，max_model_len=18433（Phase0/训练一致）
M*：Only-fail + B=1 + k=2 + entropy + math_aware + Scheme B
必跑：B1, B2, B3, B4, M*
claim：M* > B1 ∧ M* > B2；M* > B4；不被 B3 完全解释
Phase0 v2：recover(fork vs resample/continue) + hi/lo 候选方差
         （弃用：高熵位置 vs 随机位置的 P_succ / ΔP）
代码：iclr/verl（tb_opd.enable 默认 False）；脚本 iclr/scripts/
```

---

## 11. Phase 0：hi vs lo 候选方差（机制细节）

对**同一条失败主轨迹**：

1. 逐步熵选 **hi**（≥80 分位）与 **lo**（≤20 分位）；
2. 在各自位置对 top-k 候选强制展开，各续写 m（或 m_lo）次，得候选成功率向量 \((\hat P_1,\ldots,\hat P_k)\)；
3. 算 \(\mathrm{Var}_{\mathrm{hi}}\) 与 \(\mathrm{Var}_{\mathrm{lo}}\)。

**解读**：方差大 ⟹ 换候选下游对错差很多 ⟹ 真 forking token；方差小 ⟹ 分支几乎不增加信息。期望 \(\mathbb{E}[\mathrm{Var}_{\mathrm{hi}}] > \mathbb{E}[\mathrm{Var}_{\mathrm{lo}}]\)，并用配对符号检验 + prompt 聚类 bootstrap 报告。

这替代旧「高熵−随机 ΔP」里「**为什么选高熵点**」那一半；「**要不要分支**」由 `recover.fork − recover.resample` 承担。

