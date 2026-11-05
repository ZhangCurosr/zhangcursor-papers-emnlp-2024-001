---
title: "Direct-Multi-Turn-Preference-Optimization-for-Language-Agent"
source: https://aclanthology.org/2024.emnlp-main.138.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:16:32"
field: "语言代理的偏好优化"
keywords: ["preference optimization", "language agent", "DPO", "multi-turn RL", "SAOM", "compounding error"]
innovations: ["以SAOM约束替代策略约束使partition function变为全局常数从而在多轮场景中可消去", "引入长度归一化并为其提供理论解释以处理不等长偏好轨迹"]
benchmarks: ["WebShop", "ScienceWorld", "ALFWorld"]
---

# 论文速读：Direct-Multi-Turn-Preference-Optimization-for-Language-Agent

## 一句话总结
本文提出 **DMPO（Direct Multi-turn Preference Optimization）** 损失函数，通过将以策略约束替换为状态-动作占用度量（SAOM）约束，并引入长度归一化，使 DPO 能直接优化多轮语言代理任务的强化学习目标，有效缓解累积误差问题。

## 研究问题与动机
- **多轮场景下 DPO 失效**：标准 DPO 仅适用于单轮偏好对齐，其推导依赖的策略约束导致 partition function $Z(s)$ 依赖于当前状态 $s$，无法在多轮场景中被消去。
- **BC/SFT 方法的累积误差**：行为克隆（Behavioral Cloning）在对专家轨迹做 SFT 后，模型在非专家覆盖的状态上会随机选择动作，导致微小错误沿交互逐步放大。
- **现有 DPO 变体不适配多轮**：ETO（Song et al., 2024）等尝试将 DPO 应用于代理任务，但其仍针对单轮 bandit 设置设计，在多轮场景下性能不佳。
- **长度不一致问题**：多轮偏好数据中，"win"与"lose"轨迹的长度往往不同，导致 Bradley-Terry 模型中出现长度依赖的偏差项。

## 核心贡献（创新点）
- **提出 DMPO 损失函数**：直接优化多轮场景下的 RL 目标，避免了传统 RL 的环境交互和不稳定性，相比直接套用 DPO 在多轮代理任务上更鲁棒。
- **以 SAOM 约束替代策略约束**：将 RL 目标中的策略 KL 散度约束替换为状态-动作占用度量约束，使 partition function 变为与状态无关的全局常数，从而可以在 Bradley-Terry 模型中被消去。
- **为长度归一化提供理论解释**：推导表明长度归一化（length normalization）是消去 partition function 所必需的，也为 SimPO 中经验使用的长度归一化技术提供了理论依据。

## 方法详解
- **SAOM 约束引入**：将 DPO 的 RL 目标从 $\max_{\pi_\theta} \mathbb{E}[\sum \gamma^t r] - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$ 替换为 $\max_{\pi_\theta} \mathbb{E}_{(s,a)\sim d^{\pi_\theta}}[r(s,a)] - \beta \mathbb{D}_{KL}[d^{\pi_\theta}(s,a) \| d^{\pi_{ref}}(s,a)]$，其中 $d^\pi(s,a) = \frac{1-\gamma}{1-\gamma^T}\sum_{t=0}^{T-1} \gamma^t \mathbb{P}(s_t=s, a_t=a|\pi)$。
- **最优解形式**：对应 RL 目标的最优 SAOM 为 $d^{\pi^*}(s,a) = \frac{1}{Z} d^{\pi_{ref}}(s,a) \exp(\frac{1}{\beta} r(s,a))$，其中 $Z$ 是与 $(s,a)$ 无关的全局常数。
- **Bradley-Terry 模型 + 长度归一化**：在多轮 BT 模型中对 win/lose 轨迹的累积奖励分别除以长度归一化系数 $\frac{1-\gamma}{1-\gamma^{T}}$，使得 $Z$ 可以在比值中被消去。
- **最终 DMPO 损失**：
$$\mathcal{L}_{DMPO} = -\mathbb{E} \log \sigma\left[\sum_{t=0}^{T_w-1} \beta \phi(t,T_w) \log\frac{\pi_\theta(a_t^w|s_t^w)}{\pi_{ref}(a_t^w|s_t^w)} - \sum_{t=0}^{T_l-1} \beta \phi(t,T_l) \log\frac{\pi_\theta(a_t^l|s_t^l)}{\pi_{ref}(a_t^l|s_t^l)}\right]$$
其中折扣函数 $\phi(t,T) = \frac{1-\gamma^{T-t}}{1-\gamma^T}$。
- **关键性质**：DMPO 对早期时间步（$t$ 较小）的 state-action 赋予更高权重；当 $\gamma \to 0$ 时退化为单轮 DPO；转移函数在比值中被约去，无需显式建模。

## 实验与结果
- **数据集**：WebShop（1938 train / 200 test-seen）、ScienceWorld（1483 train / 194 test-seen / 241 test-unseen）、ALFWorld（3321 train / 140 test-seen / 134 test-unseen）。
- **基线**：Base、SFT、Best-of-N（N=10）、RFT、PPO、ETO、DPO（直接应用于多轮）。
- **主要结果（Noisy Setting，Table 2）**：DMPO 在 Llama-2-7B-Chat 和 Mistral-7B-Instruct-v0.2 上均优于直接应用 DPO；在全部 Unseen 及多数 Seen 测试集上领先。
- **主要结果（Clean Setting，Table 3，Llama-2-7B-Chat）**：DMPO 在 WebShop 取得 0.701±0.003，ScienceWorld 取得 **0.724±0.005**，均超过 SFT（0.631 / 0.568）、PPO（0.642 / 0.594）、ETO（0.698 / 0.685）等基线，ScienceWorld 相较 SFT 提升约 **27.4%**。
- **消融分析**：更小的 $\gamma$ 在噪声设置下表现更好（抗噪），更大的 $\gamma$ 在干净设置下表现更好（充分利用后期轨迹）；DMPO 对 loss 轨迹长度变化鲁棒，而 DPO 性能随长度增加迅速下降。

## 相关工作脉络
- **ETO / EMMA**：直接使用 DPO 损失于多轮代理任务；本文指出其仅适配单轮 bandit 设置，无法消去多轮 partition function，DMPO 通过 SAOM 约束和长度归一化解决该问题。
- **Pangu / CMAT**：使用传统 RL（PPO 等）微调 LLM；DMPO 提供单阶段、无环境交互的稳定替代方案。
- **IPL / CPL**：同样消除奖励学习阶段，但仅适用于等长轨迹对；DMPO 支持不等长轨迹，适用范围更广。
- **SimPO**：经验性地使用长度归一化；本文从 RL 理论出发为其提供理论解释。
- **BC/SFT（Fire-Act / AgentTuning / Lumos 等）**：依赖专家轨迹做监督微调；DMPO 通过偏好优化直接缓解累积误差，不再完全依赖 expert trajectory coverage。

## 局限性与未来方向
- 当前仅考虑 turn-wise 任务设定，奖励稀疏；未来可探索 token-wise 任务设定以获得更密集的反馈信号。
- 实验仅在 7B 规模模型及模拟数据集上进行；需要在更大规模模型和真实复杂环境中进一步验证。

## 研究启发与可借鉴点
- **SAOM 约束替代策略约束**的思路可迁移至其他需要直接优化 RL 目标且希望避免环境交互的场景。
- **折扣权重函数 $\phi(t,T)$** 的设计为多步偏好优化中的时间步加权提供了可调节机制，可通过调整 $\gamma$ 在抗噪与利用长轨迹之间灵活切换。
- **长度归一化的理论解释**可直接指导其他偏好优化方法（如 IPO、SimPO 的扩展）在多步场景中的正确设计。
- **干净/噪声两套训练设置**的对比评估设计值得借鉴，可用于系统评估 preference-based 方法在真实噪声数据下的鲁棒性。

## 关键术语表
**DMPO**：Direct Multi-turn Preference Optimization，专为多轮语言代理任务设计的偏好优化损失函数。
**SAOM（State-Action Occupancy Measure）**：状态-动作占用度量，描述策略在空间中访问 $(s,a)$ 对的分布，用于替代策略约束以消去 partition function。
**Partition Function**：归一化常数，在 DPO 的单轮推导中被消去；在多轮策略约束下依赖于状态而无法消去，SAOM 约束使其变为全局常数。
**Compounding Error**：累积误差，指 BC/SFT 学习中微小偏差沿多步交互逐步放大导致性能显著下降的现象。
**Bradley-Terry Model**：用于建模二选一偏好概率的统计模型，在 DPO/DMPO 中用于从偏好数据中学习隐含奖励。
**Length Normalization**：长度归一化，通过对轨迹累积奖励除以长度相关系数来消除不等长轨迹带来的偏差。
**Discount Function $\phi(t,T)$**：DMPO 中对不同时间步 state-action 进行加权的函数，$\phi(t,T) = (1-\gamma^{T-t})/(1-\gamma^T)$，早期步权重更高。

## 可复现要素
- **数据集**：WebShop、ScienceWorld、ALFWorld（均为公开数据集）；论文未说明是否提供了处理后的偏好数据。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：$\beta \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$，$\gamma \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99\}$；batch size=32，AdamW 优化器，3% warmup + cosine scheduler；基础模型为 Llama-2-7B-Chat 和 Mistral-7B-Instruct-v0.2，在 8×NVIDIA A100 上训练。
