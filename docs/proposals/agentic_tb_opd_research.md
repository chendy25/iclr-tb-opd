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

### 4.0 环境与设定（按阶段写死，对齐开源可复用栈）

| 项 | 取值 | 理由 |
|---|---|---|
| **Phase 0'/1' 环境** | **SOD / Open-AgentRL**：tool-integrated math + code interpreter（SandboxFusion） | 有 HF 数据 + **已 release teacher ckpt**；与母方法 math 连续；veRL 底座 |
| **Phase 2' 环境** | **ATOD 三环境栈**：ALFWorld / WebShop / Search-QA | TurnOPD/SAGE/Guided 公共 benchmark；环境安装与 teacher GRPO 脚本最完整 |
| **Teacher→Student（0'/1'）** | Teacher = **SOD-GRPO_teacher-4B**（免训）；Student = Qwen3-0.6B/1.7B/4B base（或 SOD student 作对照） | 开源已成对，立刻可跑 Phase 0' |
| **Teacher→Student（2'）** | 方案 A（快）：GiGPO/SPEAR 7B + Search-R1 7B 代理；方案 B（严）：按 ATOD 脚本自训 Qwen3 GRPO teacher ~150 step | **无**匹配 ATOD 的三环境 Qwen3 teacher/student 官方套件（见 §9.10） |
| **公平轴** | matched generation tokens **且** matched tool calls | 见 §3.5 |
| **评测** | SOD Eval（AIME/GPQA/LCB 等）；跨环境用各 env success rate / score | 与 agentic 家族一致 |
| **默认方法 M** | Only-fail + B=1 + k=2 + `hybrid` 选点 + forced-topk 展开 + 整树 reverse KL | 先窄树 |

**阶段资源一句话**：

```
Phase 0'/1'  →  SOD 栈（数据+teacher ckpt+code tool）
Phase 2'     →  ATOD 栈（三环境）+ ARPO/AT²PO（树展开/B-A2）
```

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

## 8. 参考文献（新增，按用途 + 可复现性）

| 优先级 | 论文 | arXiv / 会议 | 用途 | 代码 | 数据 | ckpt |
|---|---|---|---|---|---|---|
| P0 | **ARPO** | 2507.19849 (ICLR 2026) | ΔH 选点、tool 后熵、partial branch | ✅ [RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO) | ✅ HF | ✅ 多尺度 |
| P0 | **AT²PO** | 2601.04767 (ACL 2026) | 熵引导 turn 树；B-A2 对照 | ✅ [zzfoutofspace/ATPO](https://github.com/zzfoutofspace/ATPO) | ✅ 脚本 | ❌ 需自训 |
| P0 | **ATOD** | 2606.27814 | T-DUR / B-A1；三环境栈 | ✅ [TanQitai/ATOD](https://github.com/TanQitai/ATOD) | ⚠️ 需装 env | ❌ teacher 需自训 |
| P0 | **SOD** | 2605.07725 | Phase 0'/1' TIR 环境与 teacher | ✅ [YoungZ365/SOD](https://github.com/YoungZ365/SOD) | ✅ Open-AgentRL | ✅ teacher+student |
| P0 | **TurnOPD** | 2607.05804 | turn 预算/归一化（读论文） | ❌ | — | — |
| P1 | **AEPO** | 2510.14545 | 连续高熵 turn 惩罚（C2） | ✅ 同 ARPO 仓 | ✅ | ✅ |
| P1 | **SAGE-OPD** | 2606.19659 | skip/intervene 叙事 | ❌ (Meta) | — | — |
| P1 | **Guided-OPD** | 2606.15912 | curriculum turn guidance | ⚠️ 论文链 404 | — | — |
| P1 | **Tree-GRPO** | 2509.21240 | initialize-then-expand、tool 预算 | ✅ [AMAP-ML/Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) | ✅ | ❌ |
| P2 | **SmartAD / SAD / SCoRe / BRTS** | ACL2026 等 | 轨迹/段级选择对照 | 部分 | 部分 | 部分 |
| P2 | agent-distillation | 2505.17612 | 离线 code/retrieval 蒸馏底座 | ✅ | ✅ | ✅ |

> 母方法（TB-OPD-Token）的参考见 `branching_distillation_research.md` §8。  
> **没有任何论文做了「turn 树展开 × 教师 dense KD」**——M 方法本身无可直接 fork 的 repo；可复用的是零件（环境、选点、分支 rollout、对照臂）。详细路径见 §9.10。

---

## 9. 执行路线：四阶段阶梯 + 工程清单 + Kill 判据

> **原则**：先验证「高不确定 turn 展开分支是否真能救回失败轨迹」（Phase 0'），再投断点续跑训练工程；math 主线 `M* vs B1/B2/B4` 仍优先，agentic 不抢训练卡。

### 9.0 现状盘点（决定改造量）

| 模块 | 现状（`iclr/verl`，`agentic-tbopd` 分支） | Agentic 缺口 |
|---|---|---|
| 多轮 rollout | `ToolAgentLoop`（`tool_agent_loop.py`）已有；`AgentData` 带 `messages / response_ids / response_mask / turn_scores / user_turns / assistant_turns` | 需**记录 turn 边界** + 每 turn 的 post-tool top-k logprob（供选点） |
| KD / OPD | teacher logprob、reverse KL、`AgentLoopWorker._run_tb_opd_group` 固定槽位 fan-out | 可直接复用 |
| TB-OPD 选点 | `tb_opd.py`：扁平 `response_ids` 上 token 级 entropy / topk_gap | 需 `ForkUnit=turn` + `hybrid=SoftOR(ΔH, disagreement)` |
| TB-OPD 展开 | `forced_topk`：从 fork 位置**续写 token 到 EOS** | 需**断点续跑**：从选中 turn 前缀重进 tool loop，跑完剩余多轮（**唯一真正难的工程点**） |
| 预算 | token 预算（math） | 需加 **matched tool calls** 轴 |
| over-branching | 无 | 借 AEPO：连续高熵 turn 分支惩罚 |

**结论**：除「从 turn 前缀续跑完剩余多轮 tool loop」外，其余均可复用或薄封装；Phase 0' 诊断可离线实现断点续跑，**不必先改训练主循环**。

### 9.1 核心 go/no-go 问题（Phase 0' 必须回答）

在多轮 agent 里，**在高不确定 turn 展开分支**，是否真能在**等预算**下比「不展开」救回更多失败轨迹？

- 若 **否** → 不进 Phase 1'，agentic 缩为 math 论文 discussion / future work。
- 若 **是** → 再开断点续跑 + 训练。

---

### 9.2 Phase 0' — 诊断（≈1 周，几乎不花训练卡）

**目的**：验证 §9.1 前提；**不改训练代码**。

**环境（写死）**：**SOD / Open-AgentRL**（code interpreter + SandboxFusion），不用先上 ALFWorld。

| 需求 | 推荐来源 | 说明 |
|---|---|---|
| 数据 / Eval | Open-AgentRL-SFT-3K / RL-30K / Eval | HF 可直接拉 |
| Teacher | `youngzhong/SOD-GRPO_teacher-4B` | **免训** |
| Student | Qwen3-0.6B/1.7B base | cold-start；SOD-0.6B/1.7B 可作对照 |
| ΔH_post-tool | ARPO 仓库熵监测逻辑 | 抄实现 + 对照论文 Figure |
| disagreement | ATOD T-DUR `d_k` 或 SOD step divergence | Soft-OR 公式直接抄 |
| 脚本 | 自建 `phase0_agentic_recover.py`（E8） | 离线三臂；可不改训练主循环 |

**流程**：

1. 用 `ToolAgentLoop`（或 SOD 评测栈）跑学生多轮轨迹（G 条/prompt）；
2. 对每条**失败**轨迹，逐 turn 计算：
   - `ΔH_post-tool`：tool 返回后首 k token 熵 − 初始熵（ARPO 式，归一化）；
   - `disagreement`：该 turn 内 teacher–student 平均 \|Δlog p\|（ATOD 式）；
   - `fork_signal = SoftOR(ΔH, disagreement)`；
3. 取 `fork_signal` 最高的 turn（或 hi 分位 turn），在同一前缀上做三臂**等预算**对比：
   - **fork**：强制 top-k 展开，续跑到 done；
   - **resample**：同前缀不强制、等预算自然重采样；
   - **continue**：单轨迹贪心续跑 1 次；
4. 主指标：`recover.fork`、`recover.resample`、`recover.continue`；配对 bootstrap CI（prompt 聚类）。

**工程**：诊断脚本可「拼接 messages 前缀 + 重新走 ToolAgentLoop」离线实现，无需 `_generate_sequences_tb_opd` 改造。

**Kill 判据（不进 Phase 1'）**：

| 条件 | 动作 |
|---|---|
| `recover.fork ≤ recover.resample` 且 CI 含 0 | **停止**；机制不成立 |
| fork 点 90%+ 落在 turn 0–1（开场白） | 改选点/过滤后再诊断；仍失败则停止 |
| 主轨迹 solve rate 异常低（环境/截断问题） | 先修环境/长度，再诊断 |

**通过标准（进 Phase 1'）**：

1. `recover.fork > recover.resample` 且 bootstrap CI 倾向排除 0；
2. fork 分布集中在 **post-tool 决策 turn**（非纯开场噪声）；
3. 可选：`recover.fork > recover.continue`。

---

### 9.3 Phase 1' — 最小训练验证（Phase 0' 通过后，≈2–3 周）

**目的**：证「展开 + 教师 KD」相对底座与 turn 重加权有增益；**仍限 SOD 单环境**。

**默认方法 M**：Only-fail + B=1 + k=2 + `hybrid` 选点 + forced-topk + 整树 reverse KL + tool token mask。

**主表（Table A' 子集，等 token 且等 tool-call）+ 开源对照实现**：

| ID | 必跑 | 证明 | 最接近的开源实现 |
|---|---|---|---|
| B-A0 | ✓ | agent OPD 底座 | 现有 `distillation.yaml`（`tb_opd.enable=false`） |
| B-A1 | ✓ | **展开** vs **只重加权** | **ATOD T-DUR**（去掉 RL 退火）或 SOD step weight |
| **M** | ✓ | TB-OPD-Turn | **自研**；树展开参考 ARPO/AT²PO，KD 用现有 OPD loss |

Teacher 继续用 **SOD-GRPO_teacher-4B**（勿在 1' 阶段自训 ATOD teacher）。

**Kill 判据（不进 Phase 2'）**：

| 条件 | 动作 |
|---|---|
| M ≤ B-A0（同预算） | 查实现/阈值；仍否 → idea 在 agentic 不成立 |
| M ≤ B-A1 | 收益只是「重加权」，非「展开」；缩叙事或停 |
| tool mask 消融显示 KD 被环境 token 严重污染 | 先修 mask 再比 |

**通过标准（进 Phase 2'）**：M > B-A0 且 **M > B-A1**（同预算 success rate）。

---

### 9.4 Phase 2' — 正面对比 + 跨环境（Phase 1' 通过后）

**新增对照**：

| ID | 目的 | 开源参考 |
|---|---|---|
| B-A2 | 同样 turn 树展开但 **RL outcome/credit**（≈AT²PO）→ 证「教师 dense KD > outcome credit」 | **AT²PO** pipeline 或 ARPO branch rollout（需自训） |
| B-A3 | OPD-Indep-N → 证「树展开 > 盲目多采样」 | 现有 veRL multi-rollout + 标准 OPD |

**跨环境**：直接 clone **[TanQitai/ATOD](https://github.com/TanQitai/ATOD)** 作为环境栈起点（ALFWorld / WebShop / Search-QA + GRPO teacher 脚本）。

**Teacher ckpt 现实约束**（调研结论，务必遵守）：

> **没有任何论文 release 过与 ATOD 对齐的三环境 × matched Qwen3 teacher/student 成对 ckpt。**

| 方案 | Teacher | 何时用 |
|---|---|---|
| **A（快）** | ALFWorld/WebShop → GiGPO/SPEAR-7B；Search → Search-R1 Qwen2.5-7B（E5 对齐） | 先跑通 Phase 2' pipeline |
| **B（严）** | 按 ATOD `examples/grpo_teacher_trainer/run_*_grpo_qwen3_*.sh` 自训 ~150 step | 论文级严格对比 ATOD |

Student 仍从 `Qwen/Qwen3-1.7B`（或 0.6B/4B）cold-start；ALFWorld 可选 [OPID-ALFWorld-1.7B](https://huggingface.co/Jinyang23/OPID-ALFWorld-1.7B) 作 distillation baseline。注意方案 A 的 teacher 多为 **Qwen2.5**，与 Qwen3 student 有 tokenizer/chat template 差异，正文需声明。

**消融**：§4.2 的 C0–C6（选点信号、展开方式、k/B、Only-fail、截断、KD 作用域、tool mask）。

**Kill 判据（不升 P2 独立成篇）**：

| 条件 | 动作 |
|---|---|
| M ≤ B-A2 在主要环境 | 相对 AT²PO 无 KD 优势；agentic 仅作 math 论文 transfer 一节 |
| 仅 1 个环境 M 胜出 | 保留 P1 编排，不拆第二篇 |
| over-branching 导致 tool 预算翻倍但 success 不涨 | 加强 AEPO 式惩罚 / 降 B |

**通过标准（升 P2 独立成篇）**：M 在 **≥2 环境**稳定 > B-A1 且 > B-A2；Phase 0' 机制图仍成立。

---

### 9.5 Phase 3' — 论文编排决策

| 结果 | 编排 |
|---|---|
| Phase 0' 失败 | **P3**：discussion + future work only |
| Phase 1' 通过、Phase 2' 一般 | **P1**：math 主文 + § transfer（Table A' 一表 + Figure 1' 一图） |
| Phase 2' 全面通过 | **P2**：Agentic TB-OPD 独立成文，正面 cite/对比 AT²PO、ATOD |

与 §5.1–5.2 一致；此处以 **Phase 0'→1'→2' 实证** 驱动，而非先定 P2 再补实验。

---

### 9.6 工程清单（映射 `iclr/verl` + 开源参考实现）

| # | 任务 | 落点 | 优先级 | 依赖 | **参考实现（勿从零写）** |
|---|---|---|---|---|---|
| E1 | `distillation.tb_opd.fork_unit: token\|turn` | `distillation.yaml` + `_get_tb_opd_cfg` | P0 | — | 本地现有 tb_opd 配置形态 |
| E2 | turn 边界 + post-tool top-k logprob 落盘 | `tool_agent_loop.py` / `AgentData.extra_fields` | P0 | — | ATOD turn 切分；ARPO tool 后熵窗口 |
| E3 | `select_fork_turn()`：`hybrid=SoftOR(ΔH, disagree)` | `tb_opd.py` | P0 | E2 | **ATOD T-DUR** Soft-OR；SOD step divergence |
| E4 | **断点续跑**：从 turn 前缀重进 tool loop，forced top-k × k | `agent_loop.py` | P0 | E2,E3 | **ARPO/AT²PO** branch rollout（只抄采样，不抄 RL loss） |
| E5 | tool 返回 token **loss mask** | trainer / distillation loss | P0 | E4 | ATOD：observation 不计 loss |
| E6 | 连续高熵 turn **branch penalty** | `tb_opd.py` | P1 | E3 | **AEPO** 开关式惩罚 |
| E7 | matched **tool-call** 预算计数与对齐 | rollout 脚本 + metric | P1 | E4 | ARPO/AEPO/Tree-GRPO budget 逻辑 |
| E8 | Phase 0' 诊断脚本（recover 三臂） | `iclr/scripts/phase0_agentic_recover.py` | P0 | E2 | **SOD** eval + Open-AgentRL 数据 |
| E9 | 训练脚本 B-A0 / B-A1 / M | `iclr/scripts/srun_*` | P1 | E4,E5 | B-A1←ATOD；M←自研+ARPO 树 |
| E10 | 共享前缀 packing（算力优化） | 可选，Phase 2' 后 | P2 | E4 | Tree-GRPO initialize-then-expand |

**配置约定**（默认关，不影响 math）：

```yaml
distillation:
  tb_opd:
    enable: false          # math 默认 false
    fork_unit: turn        # agentic 实验显式开
    fork_metric: hybrid    # 非纯 entropy
    scheme_b: true
    only_fail: true
    branch_mode: forced_topk
    k: 2
    max_branches_per_traj: 1
    consecutive_high_entropy_penalty: true  # AEPO 式
```

---

### 9.7 与 math 主线的资源关系

```
math：M* vs B1/B2/B4 未闭合 ──→ 训练卡优先 math
         │
         ├─ 并行：Phase 0'（SOD 栈 + 推理卡，不抢训练）
         │         资源：SOD-GRPO_teacher-4B + Open-AgentRL + ARPO ΔH + ATOD Soft-OR
         │
         ├─ math M* 站稳 ∧ Phase 0' 通过 ──→ Phase 1'（E1–E9，仍限 SOD）
         │
         └─ 任一不成立 ──→ agentic 仅文档/讨论，不硬上训练
```

**现在应做**：Phase 0'（拉 SOD 数据/teacher + E8 诊断）+ 继续 math 主线。  
**现在不应做**：Phase 1' 全量断点续跑；Phase 2' ATOD 三环境自训 teacher（除非 0'+1' 已过）。

---

### 9.8 总决策树

```
Phase 0'（SOD）：recover.fork > recover.resample（CI）？
  否 → P3（discussion）；停止 agentic 训练
  是 → Phase 1'（SOD）：M > B-A0 且 M > B-A1？
        否 → 查实现；仍否 → agentic 不成立
        是 → Phase 2'（ATOD 栈）：M > B-A2？≥2 环境？
              否 → P1（math 一节 transfer）
              是 → P2（独立成篇 vs AT²PO/ATOD）
```

---

### 9.9 执行 checklist（按阶段，含资源动作）

**Phase 0'（当前优先）**

- [ ] 拉 Open-AgentRL 数据 + `SOD-GRPO_teacher-4B`
- [ ] 对照 ARPO 仓库实现 ΔH；对照 ATOD/SOD 实现 Soft-OR disagreement
- [ ] E8：Phase 0' 脚本（recover fork / resample / continue + hi-turn 选点）
- [ ] E2（只读）：dump turn 边界 + ΔH + disagreement
- [ ] 300 prompt 子集跑完；读 `recover.*` + fork turn 分布图

**Phase 1'（0' 通过后）**

- [ ] E1–E5：ForkUnit=turn + 断点续跑（参考 ARPO/AT²PO）+ tool mask
- [ ] B-A1：fork ATOD T-DUR（去掉 RL 退火）或 SOD step weight
- [ ] E9：B-A0 / B-A1 / M 短训（仍限 SOD）；teacher 继续用 SOD-4B
- [ ] 对照 §9.3 kill / pass 标准

**Phase 2'（1' 通过后）**

- [ ] clone ATOD 环境栈；先方案 A 代理 teacher（GiGPO / Search-R1）打通
- [ ] B-A2（AT²PO/ARPO 树+RL）/ B-A3；ALFWorld 或 WebShop 其一
- [ ] 若需严格论文对比：方案 B 自训 ATOD Qwen3 GRPO teacher
- [ ] C0–C6 消融子集；success–tool budget 曲线
- [ ] 决定 P1 vs P2（§9.5）

**默认不做（除非 P2）**

- GAIA / deep-search / 浏览器 agent；E10 共享前缀 packing 可后做。
- 不要在 Phase 0'/1' 上 ALFWorld/WebShop 重安装（WebShop 独立 py3.10、Search 需常驻 E5，成本高）。

---

### 9.10 Resource / Repro 表（开源调研结论，2026-08）

#### 9.10.1 可复现性总览

| 论文 | 代码 | 数据 | 模型权重 | 对 TB-OPD-Turn 价值 |
|---|---|---|---|---|
| **ARPO** | ✅ | ✅ HF | ✅ | **P0**：ΔH、tool 后熵、分支 rollout |
| **AEPO** | ✅ 同 ARPO | ✅ | ✅ | **P1**：连续高熵 turn 惩罚（C2） |
| **AT²PO** | ✅ | ✅ 脚本 | ❌ 需自训 | **P0**：turn 树 + credit（B-A2） |
| **Tree-GRPO** | ✅ | ✅ | ❌ | **P1**：tool 预算、initialize-then-expand |
| **ATOD** | ✅ | ⚠️ 自装 env | ❌ teacher 自训 | **P0**：T-DUR/B-A1、三环境栈 |
| **SOD** | ✅ | ✅ Open-AgentRL | ✅ teacher+student | **P0**：Phase 0'/1' 首选环境 |
| **TurnOPD / SAGE-OPD** | ❌ | — | — | 只读论文 |
| **Guided-OPD** | ⚠️ repo 404 | — | — | 只读论文 |
| **SmartAD / SAD** | ❌ | — | — | 叙事对照 |

#### 9.10.2 按阶段购物单

| 阶段 | 环境栈 | Teacher | Student | 关键抄什么 |
|---|---|---|---|---|
| **0'** | SOD + Open-AgentRL | SOD-GRPO_teacher-4B | Qwen3-0.6B/1.7B base | ARPO ΔH；ATOD/SOD Soft-OR；自建 recover 三臂 |
| **1'** | 同上 | 同上（免训） | 同上 | B-A0=本地 OPD；B-A1=ATOD T-DUR；M=自研+ARPO/AT²PO 树展开 |
| **2'** | ATOD 三环境 | 方案 A：GiGPO/SPEAR/Search-R1；方案 B：自训 Qwen3 GRPO | Qwen3-1.7B；ALFWorld 可选 OPID-1.7B baseline | B-A2=AT²PO；budget=ARPO/Tree-GRPO |

#### 9.10.3 关键链接（落地用）

| 资源 | URL |
|---|---|
| SOD 代码 | https://github.com/YoungZ365/SOD |
| SOD teacher | https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B |
| Open-AgentRL 数据 | https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K |
| ARPO / AEPO | https://github.com/RUC-NLPIR/ARPO |
| AT²PO | https://github.com/zzfoutofspace/ATPO |
| ATOD | https://github.com/TanQitai/ATOD |
| Tree-GRPO | https://github.com/AMAP-ML/Tree-GRPO |
| GiGPO ALFWorld/WebShop | https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554 |
| Search-R1 | https://huggingface.co/collections/PeterJinGo/search-r1-v03 |
| OPID ALFWorld student | https://huggingface.co/Jinyang23/OPID-ALFWorld-1.7B |

#### 9.10.4 不能指望的开源

- TurnOPD / SAGE-OPD / SmartAD / SAD：**无代码**，不能当实现依赖。
- Guided-OPD：论文 GitHub **404**。
- AT²PO / ATOD：**有代码、无官方 matched ckpt**；严格对比需预留自训算力。
- **没有任何一家** release「turn 树展开 × 教师 dense KD」——M 必须自研，零件从上面抄。

---

## 10. 一页纸

```
定位：同一算子（不确定决策点树展开 + 教师 dense KD）从 token→turn
缝隙：高熵/高分歧 turn「展开新分支」× 教师「dense KD 整树」——无现成 repo
阶段：0'/1'=SOD 栈 → 2'=ATOD 三环境 → 3'=定 P1/P2/P3
选点：hybrid=SoftOR(ΔH, disagreement)；抄 ATOD T-DUR / SOD
展开：forced-topk + 断点续跑；抄 ARPO/AT²PO rollout，KD 用本地 OPD
0'：SOD-GRPO_teacher-4B + Open-AgentRL；recover.fork>resample 才进 1'
1'：B-A0=本地OPD；B-A1=ATOD T-DUR；M=自研；仍限 SOD
2'：ATOD 栈；teacher 方案A=GiGPO/Search-R1，方案B=自训 Qwen3 GRPO
预算：matched tokens & tool calls；惩罚抄 AEPO
0' kill：recover.fork≤resample → 停
1' kill：M≤B-A1 → 只是重加权
2' pass：M>B-A2 且 ≥2 环境 → 升 P2
代码：iclr/verl agentic-tbopd；先 E8 Phase0'（SOD），再 E1–E5
```
