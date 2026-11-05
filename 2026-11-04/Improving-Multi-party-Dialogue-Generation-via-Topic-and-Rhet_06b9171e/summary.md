---
title: "Improving-Multi-party-Dialogue-Generation-via-Topic-and-Rhet"
source: https://aclanthology.org/2024.emnlp-main.189.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:10:46"
---

# 论文速读：Improving-Multi-party-Dialogue-Generation-via-Topic-and-Rhet

## 一句话总结
本文针对多方对话生成（MDG）中现有方法仅建模 `reply-to` 结构而忽视回复内容与目标话语在话题和修辞层面对齐的问题，提出 RL-TRC 强化学习框架，通过话题连贯性预测任务、修辞连贯性预测任务与三类话语感知奖励的联合优化，显著提升生成回复与目标话语的语义一致性与逻辑连贯性。

## 研究问题与动机
- **核心问题**：多方对话中存在多个参与者与交错的话题流，目标话语（target utterance）的回复位置不固定；传统方法过度依赖 `reply-to` 结构建模，但无法保证生成回复在“说什么（话题词汇）”和“如何说（逻辑修辞关系）”上与目标话语保持连贯。
- **现有方法不足**：
  1. `reply-to` 结构无法解耦多话题纠缠，模型易发生话题漂移（如图1中 EMMDG 的回复偏向了另一话题 Firefox 而非目标话语 Opera）。
  2. 结构建模缺乏对生成回复内部逻辑一致性的约束，可能产生表面相关但语义矛盾的回复（如图1中 MADNet 的回复自相矛盾）。
  3. 既有工作对连贯性的建模多集中于 persona-based 场景，缺乏面向“回复-目标话语”对齐的显式话题与修辞引导机制。

## 核心贡献（创新点）
- **提出 RL-TRC 强化学习生成框架**：将话题与修辞连贯性统一纳入 Actor-Critic 决策流程，区别于传统固定结构解码器，agent 可动态选择连贯语义路径引导生成。
- **构建基于 PMI 的话题连贯矩阵与预测任务**：利用 ChatGPT 提取关键词，基于 reply-to 关联话语对计算点互信息构建矩阵，显式预测回复关键词分布，从源头上缓解多话题流干扰。
- **设计修辞连贯性预测任务**：引入 discourse parser 标注目标话语与金标回复间的修辞关系，通过 MLP 预测该关系，弥补结构建模对逻辑语义对齐的缺失。
- **提出三类话语感知奖励函数**：组合话题分类置信度奖励、修辞关系 KL 散度奖励与回复对象识别 KL 散度奖励，实现从局部词汇到全局结构的多粒度策略优化。

## 方法详解
- **Encoder**：采用 BART-base，输入格式为 `[CLS][SEP]p_1u_1[SEP]...[RT]p_tu_t...[SEP]p_r`，提取话语隐状态 $h_{ui}$ 与整体对话语义 $H$。
- **Topic Coherence Task**：用 ChatGPT 提取目标话语 $u_t$ 与金标回复 $y$ 的关键词；对所有 reply-to 关联话语对计算 PMI 得分，取每词 Top-10 构建话题连贯矩阵 $E_{ck}$；通过注意力融合 $\mathbf{h}_{ut}$ 与 $E_{ck}$ 预测回复关键词概率 $P_a$，以交叉熵损失 $\mathcal{L}_t$ 监督。
- **Rhetorical Coherence Task**：使用 discourse parser 获取目标话语与金标回复的修辞关系 $r$；将 $\mathbf{h}_{ut}$ 经 MLP 预测关系分布 $P_b$，以交叉熵损失 $\mathcal{L}_r$ 监督。
- **Discourse-aware RL Agent**：
  - **State**：$s_1$ 为对话历史；第 $k$ 步状态 $s_k$ 包含历史及已生成的 trigger sequence $\tau_{1:k-1}$，经编码器得到 $\mathbf{s}_k^e = [\mathbf{h}_{s1}: \cdots : \mathbf{h}_{sk}]$。
  - **Action**：$a_k \in \{0, 1\}$，0 选话题连贯语义 $\mathbf{h}_{ut}^a$，1 选修辞连贯语义 $\mathbf{h}_{ut}^b$，作为 context embedding 送入 Decoder 生成 $\tau_k$。
  - **Policy**：Actor $\pi_\theta(a_k|s_k)$ 与 Critic $Q_\phi(s_k)$ 共享 ELU 多层感知机，策略梯度损失为 $\mathcal{L}_{ag} = -\mathbb{E}[\log \pi_\theta(a_k|s_k)(\sum \epsilon^k r_k - Q_\phi(s_k))]$。
- **Reward 设计**：
  - 话题连贯奖励：$R_{tc} = f_{tc}([u_t; u_{t\_kws}], [y; y_{kws}]) \cdot e^{(n/|y_{kws}|-1)}$，$f_{tc}$ 为 RoBERTa-base 训练的二分类器（准确率 84%），$n$ 为可通过矩阵推导的关键词数。
  - 修辞连贯奖励：$R_{rc} = -KL(f_{rc}(u_t, y^*) || f_{rc}(u_t, y))$，$f_{rc}$ 为基于 Molweni 的关系分类器（准确率 65%）。
  - Reply-to 奖励：$R_{rt} = -KL(f_{rt}(C, y^*) || f_{rt}(C, y))$，$f_{rt}$ 为回复对象识别模型（准确率 91%）。
  - 总奖励：$r = 0.5 R_{tc} + 0.5 R_{rc} + 1.0 R_{rt}$。
- **训练目标**：$\mathcal{L} = \mathcal{L}_{ag} + \mathcal{L}_{gen} + \mathcal{L}_t + \mathcal{L}_r$，其中 $\mathcal{L}_{gen} = -\sum \log P(y_t|y_{<t}, \mathbf{s}_k^e)$。

## 实验与结果
- **数据集**：Hu（Train 311,725 / Valid
