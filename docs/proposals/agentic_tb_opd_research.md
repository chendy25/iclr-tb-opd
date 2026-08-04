# Turn-Branching On-Policy Distillation for Multi-Turn Agents

> 本文件是在 `branching_distillation_research.md`（单轮数学 Token-Branching OPD，下称 **母方法 / TB-OPD-Token**）之上，讨论把「按 turn 熵分支」纳入论文的**新版 proposal**。
> 分支：`agentic-tbopd`。定位：**同一算子（在不确定决策点做树展开 + 教师 dense KD）从 token 粒度提升到 turn 粒度**。
> 结论先行：方向成立，但 2026 的 agentic 分支/蒸馏赛道**已非常拥挤**，缝隙比母方法窄得多。**唯一还站得住的缝隙**是——
> **在高熵 turn 上「扩展新的分支 rollout」并用「教师 dense KD」监督整棵 turn 树**；这既不同于 turn-aware OPD（只重加权/门控/跳过单条轨迹，不展开新分支），也不同于 ARPO/AEPO/AT²PO/Tree-GRPO（展开分支但用 RL outcome/credit，不用教师 KD）。

---

## 0. 一句话定位

在 **多轮 tool-use / agentic On-Policy Distillation** 中，于**教师-学生分歧或学生不确定度最高的 turn**（典型是 tool feedback 之后的决策 turn）**强制展开若干候选子轨迹（top-k / 温度重采样）**，形成共享前缀的 **turn 树**，再由教师对树上所有分支做 **token-level dense KD**——把母方法 TB-OPD 的「树展开 + dense 监督」算子从**单轮 token**迁到**多轮 turn**。

**建议方法名**：**TB-OPD-Turn**（Turn-Branching OPD）或 **AgentForkKD**。母方法记为 **TB-OPD-Token**，两者共享 §3 的统一算子 `ForkUnit ∈ {token, turn}`。

**一句可写进 paper 的定位句（英文）**：

> *We instantiate tree-branching on-policy distillation at two granularities—token forks on single-turn math and turn forks after high-entropy tool feedback in multi-turn agents—under one operator: expand candidate sub-trajectories at uncertain decision points and supervise the resulting tree with teacher dense KD. Unlike agentic-RL tree methods (ARPO, AEPO, AT²PO, Tree-GRPO) that expand branches but learn from sparse outcome rewards, we supervise branches with a teacher; unlike turn-aware OPD (ATOD, TurnOPD, SOD, SAGE-OPD) that reweight or gate a single trajectory, we expand new branches at uncertain turns.*

---

## 1. 核心研究问题

**假设**：在多轮 agentic 蒸馏里，真正值得额外花采样预算的位置是少数「高分歧 / 高不确定」的决策 turn（往往紧跟 tool 反馈）。在这些 turn **展开候选子轨迹并让教师 dense 监督整棵树**，比：

- (a) 单轨迹 agent OPD（B-A0）、
- (b) 等预算的 turn 级重加权 / 门控 OPD（B-A1，≈ ATOD/SOD/TurnOPD 简化）、
- (c) 高熵 turn 展开但只用 outcome RL（B-A2，≈ ARPO/AT²PO 精神）

**更能提升多轮任务的成功率与蒸馏效率**。

**待验证的方法（流程）**：

1. 学生按当前策略跑完整条多轮轨迹（含 tool 调用），记录每个 turn 的不确定度信号；
2. 在满足分叉准则的 turn，从该 turn 的**共享前缀**强制展开 k 个候选续写（top-k 首 token / 温度重采样），各自把**剩余多轮任务跑到结束**（或截断预算）；
3. 教师对 turn 树上所有（或加权后的）分支做 token-level KD；
4. 预算严格对齐（matched generation tokens **且** matched tool calls）。

---

## 2. 与已有工作的关系（这是本 proposal 的重点）

### 2.1 赛道已经很拥挤：两大阵营都各自很密

**阵营 A — Agentic RL 的「熵驱动分支 / 树展开」（展开分支，但用 RL，不用教师 KD）**

| 工作 | arXiv / 会议 | 机制 | 与我们的差距 |
|---|---|---|---|
| **ARPO** | 2507.19849 (ICLR 2026) | tool-call 后监测 token 熵变 ΔH，超阈值就 **partial branch sampling**；hard/soft advantage | **RL outcome**，非教师 dense KD |
| **AEPO** | 2510.14545 | 熵预监测分配 global/branch 预算 + **连续高熵 turn 分支惩罚**（防 over-branching）+ stop-grad 熵裁剪 | RL；主要解决 ARPO 的稳定性 |
| **AT²PO** | 2601.04767 (ACL 2026) | **Entropy-Guided Tree Expansion**（从最不确定 turn 长树）+ turn-wise credit assignment + turn 级 PO | **和我们最像的对手**：也是「高熵 turn 树展开」，但监督是 **RL credit**，非教师 |
| **Tree-GRPO / Tree Search for Agent RL** | 2509.21240 | initialize-then-expand，step 级节点，同 token/tool 预算下拿更多 rollout | RL；未显式优先高熵展开 |

**阵营 B — 多轮 agent 的 On-Policy Distillation（用教师，但只在单条轨迹上重加权/门控/跳过，不展开新分支）**

| 工作 | arXiv / 会议 | 机制 | 与我们的差距 |
|---|---|---|---|
| **ATOD** | 2606.27814 | **T-DUR**：turn 级 disagreement×uncertainty 的 Soft-OR 软门控 OPD + 退火 RL | **不展开分支**；且明确指出「Entropy-RW 单信号」比双信号差 |
| **TurnOPD** | 2607.05804 | turn 级预算：自适应 rollout 深度 + 渐进 turn-normalized loss | 不展开分支；解决的是深/浅 turn 监督失衡 |
| **SOD** | 2605.07725 | step 级 divergence 自适应重加权（TIR 小模型，tool 错误级联） | 不展开分支 |
| **SAGE-OPD** | 2606.19659 | 教师按置信度决定每个 turn **skip / intervene** | 不展开分支 |
| **Guided-OPD** | 2606.15912 | rollout 内混合 teacher/student turn + 递减干预 curriculum | 不展开分支 |
| **SmartAD / SAD / SCoRe / BRTS** | ACL2026 / 2505.13820 / 2509.14257 / 2605.09725 | 轨迹/段级选择与加权、最早错误纠正、Best-of-N 教师选择 | 离线或单轨迹层面，非高熵 turn 树展开 |

### 2.2 真正的缝隙（唯一还成立的 novelty）

把两阵营叠在一起看，**没有人做的恰是它们的交集**：

```
              展开新分支 rollout？
                 否            是
教师       ┌───────────────┬────────────────┐
dense  否  │   标准 OPD     │ ARPO/AEPO/AT²PO │  ← 用 RL outcome/credit
KD?        │               │ Tree-GRPO       │
       ─── ├───────────────┼────────────────┤
       是  │ turn-aware OPD │  ★ TB-OPD-Turn  │  ← 本 proposal
           │ ATOD/TurnOPD/ │   （空白格）     │
           │ SOD/SAGE/Guided│                │
           └───────────────┴────────────────┘
```

**一句话缝隙**：
> **高熵 turn 上「扩展候选子轨迹」× 教师「dense KD 监督整棵树」**。
> = ARPO/AT²PO 的「熵引导 turn 树展开」 + OPD 的「教师 dense 监督」，两者此前从未合并。

这与母方法完全同构：母方法在**单轮 token**上「展开 top-k + 教师整树 KD」；本方法在**多轮 turn**上做同一件事。因此可作为**同一算子的第二个粒度实例**，而不是全新故事。

### 2.3 必须 cite 且必须划清界限的三篇（审稿人一定会问）

1. **AT²PO（2601.04767）** — 最危险的对手。它已经做「entropy-guided tree expansion at turns」。**唯一区别**：AT²PO 用 turn-wise credit assignment（RL，把稀疏 outcome 反传到 turn 节点）；我们用**教师 dense KD**（每个 turn 每个 token 都有监督，不依赖 outcome，也不需要 reward 可验证）。→ 必须有一条 **B-A2「同样 turn 树展开但换成 RL 监督」** 的对照臂来证明「教师 dense 监督 > outcome credit」。
2. **ARPO（2507.19849）** — 分叉时机的来源。它证明了「tool 反馈后熵尖峰」是自然分叉点。我们沿用其 **ΔH 选点**，但用途从 RL 探索改成 KD 监督覆盖。
3. **ATOD（2606.27814）** — 最危险的「被说成同类」风险。它是 turn 级 OPD 重加权，且**实验表明 entropy 单信号重加权比 disagreement+uncertainty 双信号差**（Entropy-RW 掉 2.35 分）。这给我们两个直接启示：
   - **不要只用熵选点**：应把 **teacher–student disagreement** 纳入 turn 选点（见 §3.2），否则容易被 ATOD 的结论反噬；
   - 必须有 **B-A1「turn 重加权但不展开」** 对照，证明「展开」相对「只重加权」有额外收益。

---

## 3. 方法草图（统一算子，turn 实例）

### 3.1 统一接口（与母方法共享）

```text
ForkUnit ∈ {token, turn}          # 母方法=token；本方法=turn
ForkMetric(turn):
  ent      : 该 turn 首 k 个 token 的平均熵（ARPO 式）
  dHtool   : ΔH_post-tool = Normalize(H_after_tool − H_init)   （ARPO/AEPO 式，默认）
  disagree : teacher–student 在该 turn 上的 log-prob 分歧 d_k   （ATOD 式，抗噪）
  hybrid   : Soft-OR(dHtool, disagree)  = 1−(1−a)(1−b)          （推荐主用）
ExpandPolicy(turn):
  forced-topk   : 该 turn 首 token 取 top-k 各强制展开，续写整条剩余多轮任务
  temp-resample : 同前缀温度重采样 k 条子轨迹（≈ CURE/ARPO 对照）
Trigger  : Only-fail（默认，主轨迹任务失败才分支）+ 每轨迹最多 B 个 turn 分叉
Budget   : 同时锁 matched generation tokens 与 matched tool calls
Loss     : 整棵 turn 树的 reverse KL（与母方法一致，先不改）
```

### 3.2 分叉准则（关键设计，吸取 ATOD 教训）

- **默认主用 `hybrid = Soft-OR(ΔH_post-tool, disagreement)`**，而非纯熵。
  - `ΔH_post-tool` 承接 ARPO 的实证：tool 反馈后前 10–50 token 熵尖峰是天然决策点；
  - `disagreement` 承接 ATOD：纯熵会被「student 已漂移导致 gap 被压扁」误导，加分歧更稳。
- 绝对地板 `min_fork_signal` 抑制均匀噪声高熵 turn；
- **借鉴 AEPO 的 branch penalty**：对**连续**高熵 turn 施加分叉惩罚，避免在同一条链上过度 over-branching（否则预算爆炸 + 多样性反降）。

### 3.3 展开方式（相对 turn-aware OPD 的硬差异）

- **TB-OPD-Turn（本方法）**：在选中 turn 从共享前缀**展开 k 条候选子轨迹**，每条把**剩余多轮任务继续跑到 done/预算**（含真实 tool 调用）。这是「新增 rollout」，turn-aware OPD 家族都没有。
- **对照 B-A1（turn 重加权，不展开）**：只在原单条轨迹上按 turn 权重做 KD（≈ ATOD/SOD 简化）。
- **对照 B-A2（展开但 RL 监督）**：同样展开 turn 树，但用 outcome/credit（≈ ARPO/AT²PO 精神），不用教师。
- 三者**等 token 且等 tool-call 预算**对比，才能把收益拆成「展开」「教师 dense 监督」两部分。

### 3.4 损失

令 turn 树轨迹集合 \(\mathcal{T}\)，教师 \(\pi_T\)：

\[
\mathcal{L} = \sum_{\tau\in\mathcal{T}} w(\tau)\,\frac{1}{|\tau|}\sum_t D\big(\pi_T(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t)\big)
\]

- \(D\) 先用 reverse KL（与母方法、GKD 对齐）；
- **教师在 tool 返回 token 上不计 loss**（这些是环境注入、非策略生成，务必 mask，否则监督被污染）；
- 权重 \(w(\tau)\) 消融：Uniform / Outcome(+α) / fork-turn 加权。

### 3.5 预算控制（agentic 特有，务必写死）

多轮 + tool 让预算维度比 math 多一个「tool 调用次数」。主表**同时锁两轴**：

- **matched generation tokens**（学生生成 token 总量对齐）；
- **matched tool calls**（tool 调用次数对齐——ARPO/AEPO 的核心叙事就是「省一半 tool budget」，审稿人必看）。

TB-OPD-Turn 的「免费午餐」只能来自把预算花在更高信息的 turn，而不是多打 tool / 多生成。

---

## 4. 实验设计

### 4.0 环境与设定（为可对比性，靠拢 turn-aware OPD 家族）

| 项 | 取值 | 理由 |
|---|---|---|
| **环境** | 先 **tool-integrated math / code interpreter**（与母方法 DAPO-Math 连续、环境稳）；再补 **ALFWorld / WebShop / Search-QA** | 后三个是 ATOD/TurnOPD/SAGE-OPD/Guided-OPD 的公共 benchmark，能直接软对照 |
| **Teacher→Student** | 与母方法一致的族内蒸馏，如 **Qwen3-8B/30B-A3B → Qwen3-4B/1.7B/0.6B** | 对齐 ATOD/Guided-OPD 的 Qwen3 家族设定 |
| **公平轴** | matched generation tokens **且** matched tool calls | 见 §3.5 |
| **评测** | 各环境 success rate / score（avg 多 seed） | 与 agentic 家族一致，不用 math avg@16 |
| **默认方法 M** | Only-fail + B=1 + k=2 + `hybrid` 选点 + forced-topk 展开 + 整树 reverse KL | 先窄树 |

### 4.1 主表（Table A' — Agentic）

| ID | 方法 | 展开? | 监督 | 目的 |
|---|---|---|---|---|
| B-A0 | Agent OPD（单轨迹全 token KD） | 否 | 教师 | 底座（vanilla agent OPD） |
| B-A1 | Turn-Reweight OPD（≈ ATOD/SOD 简化：hybrid 权重，不展开） | 否 | 教师 | 证「展开」有额外收益 |
| B-A2 | Entropy-Turn-Tree + RL（≈ ARPO/AT²PO 精神：展开但 outcome/credit） | 是 | RL | 证「教师 dense KD > outcome credit」 |
| B-A3 | OPD-Indep-N（等预算独立多轨迹，逐条 KD） | N 条独立 | 教师 | 证「树展开 > 盲目多采样」 |
| **M** | **TB-OPD-Turn**（本方法） | 是 | 教师 | 主方法 |

**主结论三问**：
1. 等 token & tool 预算下，M > B-A0 / B-A3？（分支 & 集中投放是否有用）
2. M > B-A1？（**展开** vs **只重加权**）
3. M > B-A2？（**教师 dense KD** vs **outcome RL credit**，即对 AT²PO 的关键差异）

### 4.2 消融（主表有正向信号后）

| ID | 轴 | 变体 | 要回答 |
|---|---|---|---|
| C0 | 选点信号 | `ent` / `dHtool` / `disagree` / **`hybrid`** | 呼应 ATOD：纯熵是否真的更差？hybrid 是否最稳 |
| C1 | 展开方式 | forced-topk vs temp-resample | 相对 CURE/ARPO 重采样是否有额外价值 |
| C2 | 树宽 k / 分叉数 B | k∈{2,3}，B∈{1,2} + **连续高熵惩罚 on/off** | 呼应 AEPO：over-branching 是否伤收益 |
| C3 | 触发 | Only-fail vs Always | 呼应 Unmasking：是否只在失败轨迹上分支 |
| C4 | 续写长度 | 分叉后跑到 done vs 截断剩余预算 | 多轮截断是否够用 |
| C5 | KD 作用域 | 整树 vs 仅分叉 turn 及之后 | 共享前缀重复监督是否浪费 |
| C6 | tool token mask | mask vs 不 mask 环境返回 | 验证监督污染这个坑 |

### 4.3 机制图（Figure 1' — Agentic）

- 复现 ARPO 观察：tool 反馈后 turn 熵尖峰；
- 叠加我们实际 fork 的 turn 分布（是否集中在 post-tool 决策 turn，而非开场）；
- **Phase 0' 诊断**（对齐母方法的 recover 思路）：在**失败**多轮轨迹上，比较
  - `recover.fork`（高信号 turn 强制展开后，≥1 分支任务成功的比例）
  - vs `recover.resample`（同 turn 不强制、等预算自然重采样）
  - vs `recover.continue`（单轨迹贪心续写）
  用 prompt 聚类 bootstrap CI；证「turn 展开确实救回失败轨迹」。

---

## 5. 与母方法（单轮 math）的关系与论文编排

### 5.1 三种编排方案（建议二选一）

| 方案 | 形态 | 适用 | 风险 |
|---|---|---|---|
| **P1 同一篇：主 math + 一节 agentic transfer**（推荐先按此推进） | math 主表不动；加 §「Turn-granularity transfer」1 表(Table A')+1 图 | 母方法已站稳、想一篇讲完「一个算子两种粒度」 | agentic 只能做轻量，深度不足会被说「加戏」 |
| **P2 拆成第二篇：Agentic TB-OPD 独立成文** | 主基线换成 B-A0/B-A1/B-A2/AT²PO/ATOD | 若 agentic 结果强、想正面刚 AT²PO/ATOD | 需要完整 agentic 工程与多 benchmark，成本高 |
| P3 只写 discussion + future work | related work 一句 + 展望 | 若 math 主线时间紧 | novelty 不计分 |

**当前建议**：**先按 P1 的最小形态实现与验证**（§4.1 的 M vs B-A0/B-A1/B-A2），若 M 在 tool-integrated math 上同时打过 B-A1（vs 重加权）与 B-A2（vs RL），再决定是否升级为 P2 独立成篇。

### 5.2 触发升级为独立论文（P2）的判据

- M 在 ≥2 个环境（如 code-math + ALFWorld）上稳定 > B-A1 且 > B-A2；
- 机制图（Phase 0'）显示 `recover.fork > recover.resample` 且 CI 排除 0；
- 相对 AT²PO 能给出「无需可验证 reward / 稀疏 outcome 也能训」的额外卖点。

---

## 6. 风险与缓解

| 风险 | 来源 | 缓解 |
|---|---|---|
| **被说成 AT²PO 换 KD 外壳** | AT²PO 已做熵引导 turn 树展开 | 必设 B-A2 对照臂；主打「教师 dense 监督覆盖，不需可验证 reward / credit assignment」 |
| **被说成 ATOD/turn-aware OPD 变体** | 它们已做 turn 级 OPD 重加权 | 必设 B-A1 对照臂；强调「展开新分支」是它们都没有的操作 |
| **纯熵选点反而更差** | ATOD 明确报告 Entropy-RW < 双信号 | 默认 `hybrid=Soft-OR(ΔH, disagreement)`，把纯熵降为消融 C0 的一个点 |
| **over-branching / 预算爆炸** | AEPO 指出连续高熵分支塌缩 | B=1 起步 + 连续高熵 turn 惩罚 + 同时锁 token & tool 预算 |
| **教师信号在漂移深 turn 不可靠** | SOD/ATOD/Guided-OPD 共同痛点 | Only-fail + disagreement 门控；必要时借 Guided-OPD 的 teacher-prefix 稳定早期 |
| **tool 返回 token 污染 KD** | 工程坑 | 环境注入 token 一律 mask（消融 C6 验证） |
| **多轮 + tool + 树，工程/算力陡增** | agentic 本身 | 先 code interpreter 单 tool、短 horizon；共享前缀 packing 后做 |
| **成本高但只是「多打 tool」** | 审稿常识 | matched tool calls 主表；报 success–tool budget 曲线 |

---

## 7. 预期贡献（论文可写三条）

1. **方法**：提出 **TB-OPD-Turn**——把「不确定决策点树展开 + 教师 dense KD」这一算子从单轮 token 提升到多轮 turn；在高分歧/高熵 turn 展开候选子轨迹并对整棵 turn 树 dense 蒸馏。填补「分支展开 × 教师监督」的空白格。
2. **实证**：在等 token 且等 tool-call 预算下，量化拆分收益来源——相对 turn 重加权 OPD（ATOD 类）证明「展开」的价值，相对熵引导 turn 树 RL（ARPO/AT²PO 类）证明「教师 dense 监督」的价值。
3. **分析**：给出「哪些 turn 值得展开」的实证规则（ΔH_post-tool + teacher disagreement 的 Soft-OR），并用 Phase 0' 的 `recover.fork vs resample` 连接 ARPO 的熵尖峰观察与 OPD 训练动态。

---

## 8. 参考文献（新增，按用途）

| 优先级 | 论文 | arXiv / 会议 | 用途 |
|---|---|---|---|
| P0 | **ARPO** | 2507.19849 (ICLR 2026) | 分叉时机来源（tool 后熵尖峰、ΔH 选点） |
| P0 | **AT²PO** | 2601.04767 (ACL 2026) | 最近对手：熵引导 turn 树展开（RL）；必设对照 B-A2 |
| P0 | **ATOD** | 2606.27814 | turn 级 OPD 重加权（T-DUR）；必设对照 B-A1；纯熵更差的教训 |
| P0 | **TurnOPD** | 2607.05804 | turn 级 OPD 预算/归一化；对照与 benchmark |
| P1 | **AEPO** | 2510.14545 | over-branching 惩罚（连续高熵 turn）；C2 依据 |
| P1 | **SOD** | 2605.07725 | step 级 divergence 重加权；TIR tool 错误级联 |
| P1 | **SAGE-OPD** | 2606.19659 | turn 级 skip/intervene；对照叙事 |
| P1 | **Guided-OPD** | 2606.15912 | teacher/student turn 混合 + 递减 curriculum（稳早期漂移） |
| P1 | **Tree-GRPO / Tree Search for Agent RL** | 2509.21240 | initialize-then-expand 树、同 tool 预算多 rollout |
| P2 | **SmartAD / SAD / SCoRe / BRTS** | ACL2026 / 2505.13820 / 2509.14257 / 2605.09725 | 轨迹/段级选择与加权对照 |
| P2 | agent-distillation (retrieval+code) | 2505.17612 | 小模型 agent 蒸馏底座 |

> 母方法（TB-OPD-Token）的参考见 `branching_distillation_research.md` §8（GKD / Beyond 80/20 / CURE / Unmasking OPD / TIP 等）。

---

## 9. 执行清单（agentic-tbopd 分支）

**先做（P1 最小验证）**
- [ ] 选定首个环境：tool-integrated math / code interpreter（复用母方法 reward）；
- [ ] 在 `iclr/verl` 上把 `ForkUnit=turn` 接入现有 tb_opd（选点信号 `hybrid`，展开 forced-topk，tool token mask）；
- [ ] 跑 B-A0（agent OPD 底座）+ Phase 0'（recover.fork vs resample）；
- [ ] 跑 M vs B-A1 vs B-A2（等 token & tool 预算）→ Table A'。

**若 M 胜出再做**
- [ ] 补 ALFWorld / WebShop / Search-QA 做跨环境验证与软对照；
- [ ] 消融 C0–C6；success–tool budget 曲线；
- [ ] 决定 P1（并入 math 论文一节）还是 P2（独立成篇正面对比 AT²PO/ATOD）。

**默认不做（除非升级 P2）**
- 完整 deep-search / GAIA / 浏览器 agent（工程与算力过重，留待独立论文）。

---

## 10. 一页纸

```
定位：同一算子（不确定决策点树展开 + 教师 dense KD）从 token→turn
缝隙：高熵/高分歧 turn「展开新分支」× 教师「dense KD 整树」
      —— turn-aware OPD 不展开；ARPO/AEPO/AT²PO/Tree-GRPO 不用教师
选点：hybrid = Soft-OR(ΔH_post-tool, teacher–student disagreement)   # 别只用熵(ATOD教训)
展开：forced-topk 候选子轨迹跑到 done；Only-fail；连续高熵惩罚(AEPO)
监督：整棵 turn 树 reverse KL；tool 返回 token 必须 mask
预算：同时锁 matched generation tokens & matched tool calls
主表：M vs B-A0(底座)/B-A1(只重加权≈ATOD)/B-A2(展开但RL≈AT²PO)/B-A3(独立N)
claim：M>B-A1(展开有用) ∧ M>B-A2(教师KD>outcome credit) ∧ M>B-A0/B-A3
编排：先 P1(并入 math 论文一节)，强则升 P2(独立对刚 AT²PO/ATOD)
代码：iclr/verl，分支 agentic-tbopd（ForkUnit=turn，默认关，不影响 math B1）
```
