---
title: "Learning-Planning-based-Reasoning-via-Trajectories-Collectio"
source: https://aclanthology.org/2024.emnlp-main.20.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:46:53"
field: "大语言模型推理与强化学习对齐"
keywords: ["过程奖励模型", "Direct Preference Optimization", "推理即规划", "离线模拟", "大语言模型推理", "逻辑推理", "数学推理"]
innovations: ["提出离线轨迹收集+部分探索估算中间步骤期望回报的过程奖励合成方法", "构建同时包含结果偏好与合成过程偏好的DPO训练框架（pDPO）", "仅需10%数据训练PRM、40%结果标注即可逼近全量标注效果"]
benchmarks: ["LogiQA-v2", "ReClor", "GSM8K", "MATH"]
---

# 论文速读：Learning-Planning-based-Reasoning-via-Trajectories-Collectio

## 一句话总结
本文提出一种**离线轨迹收集+过程奖励合成**的学习框架，将"推理即规划（RAP）"中的在线搜索决策转化为离线偏好学习：通过部分轨迹探索采样估计中间步骤的期望回报、训练过程奖励模型（PRM），再结合结果监督与过程监督共同构建偏好对，利用 DPO 优化策略模型。7B 模型在逻辑推理基准上超越 GPT-3.5-Turbo，同时显著降低对人工标注的依赖。

## 研究问题与动机
1. **LLM 多步推理存在幻觉与中间步骤错误**：即使最终答案正确，推导过程可能不合理；复杂推理场景尤为严重。
2. **在线搜索（如 MCTS）延迟高、成本高**：RAP 类方法在每个中间状态进行 LLM 或外部验证器评估，搜索空间大，推理延迟和计算开销巨大。
3. **人工过程标注成本极高**：Lightman 等（Step-By-Step Verification）虽证明 PRM 有效，但需专家逐步骤标注，难以扩展至 LLM 训练规模。
4. **仅用最终结果反馈无法区分推理质量**：传统 outcome-only DPO/RFT 等方法忽略了中间推理步骤的可靠性差异，两版正确但过程质量不同的推理无法被区分。

## 核心贡献（创新点）
1. **离线模拟合成过程奖励**：从已生成轨迹中采样中间推理状态，让模型多次续写完成，以成功次数作为该中间状态的期望回报估计，替代在线 MCTS 评估。*本质区别：无需在线搜索，将Planning问题转化为离线学习问题。*
2. **训练过程奖励模型（PRM）降噪**：用估计值构建分类任务训练 PRM，避免直接对每个中间步骤做启发式模拟（成本类似 MCTS）且能缓解初始策略噪声。*与直接采用原始估计值的区别：PRM 提供更鲁棒的稠密奖励信号。*
3. **合成过程监督与结果监督联合的 DPO（pDPO）**：除结果正确/错误对（$\mathcal{D}_o$）外，引入 PRM 打分相差超过阈值 $\sigma$ 的两条正确轨迹构成过程偏好对（$\mathcal{D}_p$），联合优化。*与 vanilla DPO 的本质区别：同时考虑推理路径的质量而不仅是最终答案。*
4. **大幅降低对人工作业标注的依赖**：仅需 10% 数据用于训练 PRM，仅需 40% 结果标注即可让 pDPO 达到甚至超过全量 outcome-only DPO 的效果。

## 方法详解
**形式化定义**：将自然语言推理建模为 MDP，轨迹 $\tau = \{s_0, a_0, \cdots, s_T, a_T\}$，其中 action 和 state 均由策略模型 $\pi_\theta$ 生成；终态奖励 $r_f(\tau, y) = \mathbb{1}[\tau \to y]$。

**离线模拟估计过程回报（Sec 3.2）**：给定中间步骤 $a_t$，从该处采样 $K$ 条续写轨迹 $\tau^{k|\tau_{t,a}}$，以成功到达正确答案的条数作为期望回报估计：
$$r_e(\tau_{t,a}, y) = \sum_{k=1}^{K} r_f(\tau^{k|\tau_{t,a}}, y)$$
即统计 K 次续写中有多少次得到正确答案。

**训练 PRM（Sec 3.3）**：将每步的 $r_e$ 作为类别标签，训练分类器 $f_{\text{prm}}: \mathcal{X} \times \mathcal{T} \to \mathbb{R}^K$，最小化交叉熵损失：
$$\mathcal{L}_{\text{step}} = -\log p_r, \quad p = f_{\text{prm}}(x, \tau)$$
$K$ 为可能的成功次数（0 到 K）。

**轨迹级奖励聚合（Sec 3.4）**：完整轨迹的奖励为各步 PRM 预测概率（≥C 次的类）的连乘：
$$r_p(\tau) = \prod_{t} \prod_{*\in\{a,s\}} \sum_{i \geq C}^{K} f_{\text{prm}}(\tau_{t,*})_i$$
其中超参 $C$ 控制最低成功次数阈值，用于过滤噪声。

**DPO 偏好对构建与优化（Sec 3.5）**：
- 结果对 $\mathcal{D}_o$：正确轨迹 vs 错误轨迹，使用标准 DPO loss（公式 9）。
- 过程对 $\mathcal{D}_p$：两条均正确的轨迹，按 PRM 打分差 $> \sigma$ 排序，高分为 chosen，低分为 rejected。
- 总目标：$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}; \mathcal{D}_o \cup \mathcal{D}_p)$。

**整体流程**：① 收集种子轨迹 → ② 采样中间状态并多次续写 → ③ 统计成功数训练 PRM → ④ 用 PRM 标注全量轨迹 → ⑤ DPO 联合优化。

## 实验与结果
**数据集**：逻辑推理 — LogiQA-v2（12,567 训练题）、ReClor（4,638 训练题）；数学推理 — GSM8K、MATH（训练子集来自 MetaMath）。

**基线**：Llama-2-70B-Chat、Mixtral-8×7B-Instruct、GPT-3.5-Turbo、GPT-4-Turbo、SFT、vanilla DPO、IPO、RFT、ReST-EM、PPO、GRPO。

**逻辑推理（表 1，核心结果）**：
- Llama2-7B-pDPO（LogiQA-v2 训练）：**LogiQA-v2 55.5%，ReClor 61.7%**，超越 GPT-3.5-Turbo（LogiQA-v2 45.4，ReClor 53.7）和 Mixtral-8×7B-Instruct（49.5/56.7）。
- 相对 vanilla DPO（53.1/60.4）提升 **+2.4% / +1.3%**。
- 迭代训练后 Iter-1-pDPO 达 **57.3 / 61.8**，略优于 Iter-1-process GRPO（57.3/61.7），且训练时间仅 16h（H100 ×4）vs PPO/GRPO 超 40h。

**数学推理（表 2）**：
- Gemma-2B-pDPO：GSM8K **52.8**，MATH **15.7**；超越 Gemma-7B-Instruct（46.4/24.3）的 GSM8K 表现。
- DeepSeekMath-7B-pDPO：GSM8K **82.3**，MATH **46.8**，略优于 DPO（82.4/46.3）。

**标注效率（图 4）**：仅用 40% 结果标注时，pDPO 在测试集上达 53.5，与全量 DPO（53.9）接近；PRM 仅需 10% 数据训练。

## 相关工作脉络
1. **RAP（Reasoning as Planning, Hao et al., 2023）**：将推理建模为 MDP 并用 MCTS 在线搜索；本文将其转化为离线学习问题，避免在线评估的高延迟。
2. **Step-By-Step Verification（Lightman et al., 2023a/b）**：用人工标注训练 PRM；本文用离线模拟+结果标注合成效替代人工步骤标注。
3. **DPO（Rafailov et al., 2023）**：基于偏好对直接优化策略；本文扩展至同时使用 outcome 和 synthesized process 两类偏好对。
4. **Math-Shepherd（Wang et al., 2023，并发工作）**：同样采用离线模拟合成过程奖励；但本文聚焦逻辑推理并用 DPO 而非验证排名/PPO 的方式利用 PRM。
5. **Process Supervision / Uesato et al.（2022）**：主张对中间步骤提供反馈；本文在不需额外人工标注的前提下近似实现这一目标。
6. **ReST-EM / RFT（Singh et al., 2023; Yuan et al., 2023）**：基于结果的拒绝采样方法；本文在拒绝采样基础上进一步用 PRM 对正确解排序，实现更细粒度的过程监督。

## 局限性与未来方向
1. **离线模拟仍需大量计算资源**：需对每个中间状态多次采样续写，限制了在更大模型（如 70B）和更长上下文任务（如代码生成）上的实验。
2. **PRM 质量依赖初始策略能力**：若基础模型在该任务上能力不足（如 Gemma-2B 在 MATH 上改进有限），模拟噪声会传入 PRM 训练。
3. **过程奖励估计存在噪声**：即便训练 PRM 降噪，合成奖励仍不如人工标注精确。
4. **未来方向**：探索更多弱监督来源合成过程奖励；实现一致的自我改进闭环；扩展到代码生成等需要长上下文的复杂推理任务。

## 研究启发与可借鉴点
1. **"规划转学习"的思路具有通用性**：将在线搜索/规划问题（MCTS、Tree-of-Thoughts 等）通过离线轨迹收集转化为偏好学习问题，可迁移至其他需要搜索增强的推理场景（如代码生成、多步决策）。
2. **部分轨迹探索（partial trajectory exploration）策略简洁有效**：不需完整 MCTS 树展开，只需从已有轨迹采样中间节点续写，即可低成本估计过程价值，值得在更多任务中尝试。
3. **Outcome + Process 联合 DPO 的偏好对构建方式**：将 PRM 打分引入 DPO 的选择/拒绝对构造（设 margin $\sigma$），是一种不需要额外 RL 训练即可利用过程信号的方法，比 PPO 更稳定高效。
4. **少量标注实现数据高效的思路**：仅用 10% 数据训练 PRM、40% 结果标注即可接近全量效果，对资源受限场景有重要参考价值。
5. **PRM 连乘聚合 vs logits 累加的选择**：论文在 DPO 中使用概率连乘、在 RL 中使用 logits 累加以避免非正奖励导致的极端长轨迹，这一细节设计值得注意。

## 关键术语表
**RAP（Reasoning as Planning）**：将多步推理建模为 Markov 决策过程，用搜索算法（如 MCTS）在推理状态空间中寻找最优路径。
**PRM（Process Reward Model）**：对推理轨迹中每个中间步骤/状态给出奖励评分的模型，用于过程监督。
**DPO（Direct Preference Optimization）**：不训练显式奖励模型，直接从偏好对（chosen/rejected）通过似然比损失直接优化策略模型的算法。
**pDPO（Process-supervised DPO）**：本文方法，在 DPO 中同时利用结果偏好对和 PRM 合成的过程偏好对进行联合优化。
**轨迹（Trajectory）**：从初始状态到终止状态的动作-状态序列，在推理任务中等价于完整的解题步骤链。
**MCTS（Monte Carlo Tree Search）**：基于随机模拟的树搜索算法，用于在巨大搜索空间中寻找高回报路径。
**MetaMath**：通过自我Bootstrap生成大规模数学推理问题的数据集，用于本文数学推理实验的训练数据源。
**ReAct**：结合 Reasoning（Thought）和 Acting（Action/Observation）的推理格式，本文实验采用的输出模板。

## 可复现要素
- **数据集**：LogiQA-v2（公开）、ReClor（公开）、GSM8K（公开）、MATH（公开）、MetaMath（公开）；论文均使用其测试集或训练子集。
- **代码/权重**：论文未明确声明开源代码；基线模型 Llama-2-7B-Chat、DeepSeekMath-7B-Instruct、Gemma-2B-Instruct 权重公开。
- **关键超参**：$\beta = 0.1$（逻辑推理）/ $0.5$（数学推理）；$C = 2$（逻辑）/ $3$（数学）；$\sigma = 0.4$（ReClor）/ $0.5$（LogiQA-v2, Gemma）/ $0.3$（DeepSeekMath）；温度 $0.7$；每题采样 10 条轨迹。
- **硬件**：NVIDIA A100 / H100；DPO 训练约 16 小时（H100 × 4）。
