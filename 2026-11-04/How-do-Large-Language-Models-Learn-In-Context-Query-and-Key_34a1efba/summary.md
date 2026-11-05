---
title: "How-do-Large-Language-Models-Learn-In-Context-Query-and-Key"
source: https://aclanthology.org/2024.emnlp-main.192.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:09:34"
field: "大模型可解释性与机制分析"
keywords: ["in-context learning", "mechanistic interpretability", "attention mechanism", "bias mitigation", "query-key matrices", "metric learning"]
innovations: ["提出ICL双塔度量学习假设：Value-Output提取标签特征、Query-Key学习末位置与演示位置的相似度", "识别仅1%的in-context heads即可将ICL准确率从87.6%降至24.4%", "从机制层面解释并量化缓解多数标签偏差（22%）和近因偏差（17%）"]
benchmarks: ["Financ", "AG News", "Amazon Reviews", "ETHOS", "SST2"]
---

# 论文速读：How-do-Large-Language-Models-Learn-In-Context-Query-and-Key

## 一句话总结
论文通过语义无关标签（"foo"/"bar"）的句分类任务，识别出仅占1%的关键"in-context heads"，并基于对Value-Output与Query-Key矩阵的分析，提出ICL机制假设：这些头的Value-Output矩阵负责提取标签特征，而Query-Key矩阵作为"双塔"学习末位置与演示位置特征间的相似度度量。

## 研究问题与动机
- **ICL机制尚不明确**：现有研究（如Wang et al., 2023）仅在注意力头上做平均分析，无法揭示每个head、每个矩阵的具体角色。
- **缺乏逐头重要度定位**：需要精确识别哪些head对ICL accuracy贡献最大，以及其内部各矩阵（Q/K/V/O）的功能分工。
- **ICL已知偏差缺乏解释**：多数标签偏差（majority label bias）和近因偏差（recency bias）尚无基于机制层面的系统性解释。
- **已有工作局限**：潘等（Pan et al., 2023）将ICL拆分为任务识别（TR）和任务学习（TL），但未深入至矩阵层面解释信息流转的具体路径。

## 核心贡献（创新点）
1. **识别出仅1%的in-context heads即可显著影响ICL accuracy（从87.6%降至24.4%）**：使用因果追踪（causal tracing）和log概率增量两种方法定位出6个"fooheads"和6个"barheads"，在Llama-7B与GPT-J上均验证有效。
2. **提出ICL双塔度量学习假设**：In-context heads中，Value-Output矩阵提取标签特征（如"foo"位置→"foo"概念），Query-Key矩阵充当双塔计算末位置与演示位置特征间的相似度，决定标签信息流向。
3. **从机制层面解释多数标签偏差与近因偏差**：多数偏差源于softmax注意力权重分配中，多演示标签的注意力之和更大；近因偏差源于位置编码在浅层和深层层中对注意力计算的持续影响。
4. **提出两种去偏方法并实验验证**：放大不平衡标签的注意力权重，使多数标签偏差降低22%；移除in-context heads中的位置编码影响，使近因偏差降低17%。

## 方法详解
**In-context heads识别方法**：
- **因果追踪法**：将每个深层层头的参数置零，计算ICL accuracy下降幅度。
- **Log概率增量法**：$S_l^h = \log p(b | o_l^h + \text{Lin}_l) - \log p(b | \text{Lin}_l)$，衡量head输出对预测标签$b$的log概率贡献。
- 取两种方法排名Top10的head交集，得到每模型各6个fooheads/barheads。

**Hypothesis核心逻辑**：
- 浅层：演示文本的特征被提取到标签位置（如"foo"、"bar"词位），末位置编码输入文本及上下文。
- 深层in-context heads中：
  - **Value-Output矩阵**（$vo^p = W_{VO} \cdot h^p$）：将标签特征投影到value-output向量中（如"foo→foo"、"bar→bar"），Table 4显示标签位置向量logit差值占比99.1%。
  - **Query-Key矩阵**：计算末位置query向量与标签位置key向量的内积，经softmax得注意力权重$\alpha^p$，控制标签信息流向末位置的程度。预测从"foo"切换到"bar"主要由attention score变化驱动（Table 6中attn变化达-44.4%/+91.4%）。
- 整体框架视为**双塔度量学习**：Q塔和K塔共享参数学习相似度度量函数，相似度高则对应标签概率增大。

**输出公式**：$o = \sum_p \alpha^p \cdot vo^p$（Eq.3），其中$\alpha^p = \text{softmax}(q_T^\top k_p)$。

## 实验与结果
**数据集**：5个句分类数据集——Financial PhraseBank（Financ）、AG News（AGnews）、Amazon Reviews（Amazon）、ETHOS仇恨言论检测、SST2；语义无关标签"foo"/"bar"。

**模型**：Llama-7B（32层×32头）与GPT-J（28层×16头）。

**基线ICL准确率**（Table 2）：Llama上foo/ bar分别达90.6%/99.8%（Financ）、96.6%/100%（AGnews）等；GPT-J上类似。

**核心发现数字**：
- 干预6个fooheads或barheads（各占约1%总头数）后，平均accuracy从87.6%降至24.4%（Table 3）。
- Table 4：fooheads的"foo"位置logit差值均值0.26~0.32，far相关"bar"位置接近0；反之barheads亦然，占比99.1%。
- Table 6：预测翻转时，fooheads中"foo"位置attention score下降约-21.8%~-43.0%，barheads中"bar"位置上升约+86.0%~+237.8%。
- **去偏效果**：多数标签偏差降低**22%**（Table 7，accuracy变化量减少29.1%/14.9%）；近因偏差降低**17%**（Table 8，accuracy标准差减少23.4%/10.6%，attention score标准差减少40.1%/37.7%）。

## 相关工作脉络
1. **Pan et al. (2023)**：将ICL拆分为任务识别（TR）与任务学习（TL）；本文在此基础上进一步将TL机制定位到具体head和矩阵级别。
2. **Wang et al. (2023) "Label words are anchors"**：发现标签词在浅层合并语义信息、在深层提取到末位置；本文将其作为证据(a)支持浅层特征提取假设，并深化至Value-Output向量层面。
3. **Olsson et al. (2022) Induction heads**：发现induction head帮助复制词；本文的in-context heads功能不同，负责标签特征提取与相似度度量而非简单复制。
4. **Wei et al. (2023)**：用"foo"/"bar"证明ICL依赖演示-标签映射（任务学习），本文沿用此实验范式但深入到机制解释。
5. **Xie et al. (2021)、Von Oswald et al. (2023)**：将ICL解释为隐式贝叶斯推断/梯度下降；本文从表征空间角度提供另一种互补解释。
6. **Zhao et al. (2021)多数标签偏差、Lu et al. (2021)近因偏差**：前人经验性观察到偏差，本文首次在机制层面给出解释并提出量化缓解方法。

## 局限性与未来方向
- **聚焦深层in-context heads**，浅层如何将特征传递到标签位置和末位置尚未深入研究（论文自述）。
- **仅验证于分类任务**，对其他ICL任务（如chain-of-thought推理）的泛化性待检验。
- **head重要度归因方法缺乏统一标准**：因果追踪和saliency score均非无争议，未来需探索更优的归因方法（论文自述）。
- **近因偏差缓解未涉及浅层位置编码**：仅移除深层in-context heads中的位置影响，浅层特征提取阶段的影响留作未来工作。

## 研究启发与可借鉴点
1. **"foo"/"bar"语义无关标签范式值得复用于其他研究**：本文证明该设置能有效剥离预训练先验，纯粹测量ICL的任务学习能力，是消融预训练知识影响的干净实验设计。
2. **双塔度量学习视角可迁移**：将Q/K矩阵视为双塔学习相似度度量，为理解注意力机制提供了新的理论框架，可延伸至少样本微调、prompt设计等方向。
3. **因果追踪+log概率增量联合定位关键组件的方法**：两种方法交叉验证提高了head识别的可靠性，可作为后续interpreability研究的标准化流程参考。
4. **Attention score放大/位置编码移除的去偏技术**：本文提出的两种简单干预方法（乘性放大$\alpha$、移除positional embedding项）可直接迁移到其他注意力密集型任务，改进模型鲁棒性。
5. **In-context heads的数量与能力关系**：本文暗示每个head相当于一个独立投票/ensemble单元，head数量与单个head参数规模对ICL能力的控制规律值得系统研究。

## 关键术语表
**In-context Learning (ICL)**：不更新模型参数，仅通过prompt中演示样本进行任务学习的能力。
**In-context Heads**：本文识别出的对ICL accuracy影响最大的少数attention heads（约占总头数的1%）。
**Fooheads / Barheads**：分别专门增强"foo"和"bar"标签预测概率的in-context heads子集。
**Value-Output 矩阵 (VO)**：将层输入线性投影为value-output向量，在in-context heads中负责提取标签语义特征。
**Query-Key 双塔**：Query和Key矩阵各自作为独立塔，学习末位置特征与演示标签位置特征之间的相似度度量函数。
**Majority Label Bias（多数标签偏差）**：ICL倾向于预测出现次数更多的标签。
**Recency Bias（近因偏差）**：ICL预测受演示顺序影响，靠近输入的演示权重更高。
**Causal Tracing（因果追踪）**：通过置零某模块参数并观察输出变化，量化其对最终预测的贡献度。

## 可复现要素
- **数据集**：5个公开数据集（Financial PhraseBank、AG News、Amazon Reviews、ETHOS、SST2），均为公开可用。
- **代码**：论文声明将开源至 https://github.com/zepingyu0512/in-context-mechanism（论文发表时为预开源状态）。
- **模型**：Llama-7B 与 GPT-J-6B，均为公开权重。
- **关键超参**：放大常数 $a_c = 0.03$；平衡数据集演示数量为每类2-4个；随机采样1,000句/数据集。
