---
title: "Aligning-Language-Models-to-Explicitly-Handle-Ambiguity"
source: https://aclanthology.org/2024.emnlp-main.119.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:05:13"
field: "自然语言理解与对齐"
keywords: ["歧义处理", "大语言模型对齐", "感知歧义", "INFOGAIN", "澄清请求", "数据筛选"]
innovations: ["提出利用模型感知歧义（INFOGAIN）进行数据筛选的对齐方法", "构建四阶段流水线使LLM显式识别并请求澄清歧义查询", "在仅用部分训练数据情况下超越全量训练基线"]
benchmarks: ["AmbigQA", "SituatedQA", "AmbigTriviaQA", "AmbigWebQuestions", "AmbigFreebaseQA"]
---

# 论文速读：Aligning-Language-Models-to-Explicitly-Handle-Ambiguity

## 一句话总结
本文提出 APA（Alignment with Perceived Ambiguity）对齐方法，通过模型自身感知到的歧义程度（利用INFOGAIN量化）来选择训练数据，使LLM能够显式识别并请求澄清歧义查询，同时保持对明确查询的问答能力；并在多个QA数据集上验证了该方法的有效性。

## 研究问题与动机
- **核心问题**：用户在对话中常因省略或不精确表达产生歧义，LLM需能显式识别并请求澄清，而非随意猜测回答，否则在医疗、法律等高可靠性场景可能引发严重后果。
- **现有方法不足1**：当前LLM未显式训练处理歧义，即使能识别歧义，也缺乏明确的信号（如表达不确定性或多重解释）来确认其感知。
- **现有方法不足2**：模型感知歧义的程度取决于其内在知识——知识丰富的模型可能识别歧义，而知识匮乏的模型可能认为查询无歧义（图1示例："national championship"）。
- **现有方法不足3**：推理阶段方法（如AMBIG-AWARE提示、SAMPLE REP）存在偏差或检测率低的问题，而训练方法缺乏高质量的歧义数据筛选机制。

## 核心贡献（创新点）
1. **提出APA四阶段对齐流水线**：通过模型自歧义化生成disambiguation，利用INFOGAIN量化感知歧义，构建clarification request训练数据——与既有工作相比，首次将"模型自身感知"作为数据质量估计和样本选择的依据。
2. **设计INFOGAIN作为隐式歧义度量**：计算原始查询与自歧义化结果之间的平均熵差——区别于以往基于输出熵的方法（如SUBSETENT），INFOGAIN捕捉的是模型内在知识被激活的程度。
3. **构建三个新评测数据集AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA**：分别源于TriviaQA、WebQuestions、FreebaseQA，通过GPT-4o生成歧义化问题并经人工校验，弥补了现有评测在多样QA领域覆盖的不足。
4. **实验验证APA在OOD场景下的优越性**：在5个数据集上，APA仅用约32%（LLAMA2）或13%（MISTRAL）的训练样本即超越FULL-SET，证明数据质量优于数量。

## 方法详解
**四阶段流水线（Figure 2）**：

**Stage 1：初始预测评估**
- 用标准推理模板$t(\cdot)$让模型对查询$x$生成预测$\hat{y}$
- 正确样本（明确查询答对+歧义查询正确澄清）归入$D_{correct}$
- 错误样本（明确查询答错+歧义查询未澄清或错误澄清）归入$D_{incorrect}$

**Stage 2：感知歧义检测（核心）**
- 对$D_{incorrect}$中的每个查询$x$，提示模型利用内在知识进行自歧义化，生成$\hat{x}_{disambig}$
- 计算原始查询和歧义化结果的平均熵：
  $$\mathcal{H}_x = \frac{1}{N}\sum_{i=1}^{N}\mathcal{H}_{x,i}, \quad \mathcal{H}_{x,i} = -\sum_{v \in \mathcal{V}}p_{x,i}(v)\log p_{x,i}(v)$$
- 定义**INFOGAIN**为歧义化带来的信息增益：
  $$\text{INFOGAIN}_{x,\hat{x}_{disambig}} = \mathcal{H}_x - \mathcal{H}_{\hat{x}_{disambig}}$$
- INFOGAIN > $\epsilon$（实验中$\epsilon=0.1$）的样本视为感知歧义样本$x_{ambig}$

**Stage 3：澄清请求构建**
- **Fixed Response**：从预定义列表随机选取澄清请求作为$y_{clarify}$
- **Generated Response**：给定$x_{ambig}$和$\hat{x}_{disambig}$，让模型生成针对具体歧义因子的澄清请求

**Stage 4：SFT训练**
- 构建平衡数据集$D = D_{correct} + D_{ambig}$（$n=m$）
- 使用统一推理模板，用标准 causal LM loss 训练：
  $$\min_\theta \sum_{(x,y) \in D}\sum_{i=1}^{|y|} -\log M_\theta(y_i|y_{<i}, t(x))$$
- 得到两个版本：$\text{APA}_{FIXED}$ 和 $\text{APA}_{GEN}$

## 实验与结果
**数据集**：
- In-domain：AmbigQA（训练/验证）
- OOD测试：SituatedQA-Geo、SituatedQA-Temp、AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA

**评估指标**：
- $F1_u$：明确查询预测的F1（考虑误澄清为负样本）
- $F1_a$：歧义查询澄清请求的F1（考虑误澄清为假阳性）

**主要结果（Table 1）**：
- **LLAMA2 7B**：APA$_{GEN}$在SituatedQA-Geo上$F1_u=41.01$、$F1_a=43.10$；在AmbigFreebaseQA上$F1_a=84.90$，均超越FULL-SET（$F1_u=37.67$、$F1_a=80.34$）和SUBSETENT（$F1_u=39.33$、$F1_a=83.89$）
- **MISTRAL 7B**：APA$_{FIXED}$在AmbigFreebaseQA上$F1_a=90.12$，最高；$F1_u$在SituatedQA-Temp上达45.01
- **LLAMA2 13B**：APA$_{FIXED}$在AmbigTriviaQA上$F1_u=70.83$、$F1_a=80.99$
- APA用约3,088（LLAMA2）或1,382（MISTRAL）样本，超越使用10,036样本的FULL-SET

**消融实验（Table 2）**：
- INFOGAIN-based选择（MAX）优于RAND和MIN，验证了感知歧义度量的有效性
- 不同阈值$\epsilon$下APA均稳定超越基线（Figure 5）
- APA的MCR（Misaligned Clarification Request Rate）最低，说明对原有能力的保持最好（Figure 4）

## 相关工作脉络
- **AmbigQA（Min et al., 2020）**：首个开放域歧义QA数据集，关注歧义问题的多答案特性——本文沿用其定义，但聚焦于模型对齐而非数据集构建。
- **Cole et al. (2023) SAMPLE REP**：通过多次采样一致性衡量不确定性——本文将其作为推理-only基线，指出其存在澄清请求偏好偏差。
- **Kim et al. (2023a) Tree-of-Clarification**：生成澄清树来细化歧义输入——本文与之共享目标，但采用直接对齐方式而非检索增强。
- **Zhou et al. (2024) LIMA**：证明高质量少量数据优于大规模数据——本文呼应此发现，证明利用感知歧义筛选的少量数据可超越全量训练。
- **Chen et al. (2024) Alpagasus**：用ChatGPT筛选小样本实现有效对齐——本文扩展数据质量视角，提出用模型感知歧义替代人工/外部模型筛选。
- **Dong et al. (2023) RAFT / RLHF**：基于人类偏好的对齐方法——本文扩展对齐目标至歧义管理能力，弥补安全、有用性之外的可靠性维度。

## 局限性与未来方向
- 研究范围限于短形式QA任务，可扩展至长文本生成和复杂推理。
- 仅考虑单轮查询场景，未涵盖对话QA中的上下文依赖歧义。
- 仅在LLAMA2和MISTRAL上进行实验，未来需评估更多架构和大模型规模。
- 仅使用SFT作为对齐方法，未探索RLHF或DPO等替代方案。

## 研究启发与可借鉴点
1. **INFOGAIN作为数据质量度量**：将模型内在知识与查询交互产生的信息增益作为隐式样本筛选标准，可迁移至其他需要模型"自知"的任务（如事实性检查、不确定性量化）。
2. **Perceived Ambiguity驱动的对齐策略**：不以ground-truth标签为唯一依据，而是利用模型自身判断来选择训练数据，这一思路可扩展至out-of-distribution泛化和主动学习。
3. **澄清请求作为对齐目标**：将"请求澄清"而非"猜测回答"作为歧义场景的期望行为，为agent系统的人机协作交互设计提供了新的训练范式。
4. **实验设计借鉴**：同时评估$F1_u$和$F1_a$两个指标，显式建模能力保持与能力提升的权衡，可被类似对齐研究复用。

## 关键术语表
- **Perceived Ambiguity**：模型基于自身知识对查询歧义程度的主观感知，可能与ground-truth歧义不一致。
- **INFOGAIN**：原始查询与自歧义化结果之间的平均熵差，用于量化模型感知到的歧义程度。
- **Clarification Request**：模型对歧义查询生成的请求用户进一步澄清的响应。
- **Misaligned Clarification Request Rate (MCR)**：对齐后模型对原本正确的明确查询错误生成澄清请求的比例，衡量能力保持程度。
- **$F1_u$ / $F1_a$**：分别衡量明确查询预测质量和歧义查询澄清请求质量的F1分数。
- **Self-Disambiguation**：模型利用内在知识对模糊查询进行补全和具体化的过程。
- **AmbigQA**：Min等（2020）提出的开放域歧义QA数据集，包含明确和歧义两类查询。
- **QLoRA**：Dettmers等（2023）提出的高效微调方法，本文用于参数高效对齐训练。

## 可复现要素
- **数据集**：AmbigQA公开可用；三个新数据集（AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA）未声明公开，但论文提供了构建流程
- **代码**：GitHub链接 https://github.com/heyjoonkim/APA 已开源
- **关键超参**：QLoRA r=4, alpha=16；threshold $\epsilon=0.1$；batch size=32；learning rate从{1e-3, 5e-4, 1e-4}选择；epochs从{1, 2, 3}选择
- **训练设备**：单卡Tesla V100 GPU，约半小时
- **模型**：LLAMA2 7B/13B、MISTRAL 7B
