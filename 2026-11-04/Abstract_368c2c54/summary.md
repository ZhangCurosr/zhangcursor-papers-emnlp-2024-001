---
title: "Abstract"
source: https://aclanthology.org/2024.emnlp-main.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:13:10"
field: "假新闻检测与图表示学习"
keywords: ["Fake News Detection", "Large Language Models", "Heterogeneous Graph", "Generalized PageRank", "Consistency Regularization", "Semantics Mining", "Graph Representation Learning"]
innovations: ["揭示纯LLM新闻Embedding的局限性，转向实体-主题高层语义挖掘", "提出双尺度广义PageRank传播机制，统一捕获局部细节与全局叙事一致性", "设计基于原型的一致性正则化，实现标注与无标注新闻的联合半监督优化"]
benchmarks: ["MM COVID", "ReCOVery", "MC Fake", "LIAR", "PAN2020"]
---

# 论文速读：Abstract

## 一句话总结
提出 **LESS4FD** 框架，通过 LLM 抽取实体与主题构建异构图，并结合广义 PageRank 的双重尺度特征传播与一致性正则化，有效挖掘新闻的局部细节与全局叙事语义，显著提升假新闻检测性能。

## 研究问题与动机
1. **纯 LLM Embedding 的局限性**：初步实验表明，直接将 GPT-3.5 / Llama2 输出的新闻向量输入 MLP 分类器效果不佳，因其仅编码词级别词汇语义，难以区分刻意模仿真实语言风格的假新闻。
2. **高层语义缺失**：假新闻的本质常表现为实体间关系与主题叙事的结构性矛盾（如将“转基因作物”错误关联至“COVID-19 传播”），现有文本方法缺乏跨新闻的全局知识整合能力。
3. **两个子问题**：P1 如何利用 LLM 探索高层语义（实体/主题抽取与建模）；P2 如何同时捕获单篇新闻的局部语义与跨文档的全局语义，以识别叙事不一致性。

## 核心贡献（创新点）
1. **实证揭示 LLM 直接嵌入的瓶颈**：明确指出 token 级词汇语义不足以支撑假新闻检测，推动研究重心向实体-主题高层关系迁移。
2. **LLM 增强的异构图语义建模**：设计专用 Prompt 引导 LLM 抽取多维度实体，结合 Bertopic 生成主题 embedding，构建 News-Entity-Topic 异构图，突破传统词袋/序列建模范式。
3. **双尺度广义 PageRank 传播**：提出可学习步长权重的特征传播机制，统一捕获局部（短步长）与全局（长步长）语义，避免随机游走带来的信息损失。
4. **一致性正则化半监督优化**：引入基于局部/全局双视图预测原型的 KL 散度损失，在不依赖额外标签的前提下提升未标注新闻的表示质量，最终实现端到端联合优化。

## 方法详解
- **LLM 增强语义建模**：使用 Table 2 中的结构化 Prompt 从所有新闻中抽取人名、日期、地点、组织等实体；通过 LLM API 获取新闻与实体 embedding；利用 Bertopic 提取主题词及其权重，主题 embedding 由词 embedding 加权求和得到（式1）。据此构建异构图 $\mathcal{HG}$，边关系为 `<新闻, contains, 实体>` 与 `<新闻, focuses on, 主题>`。
- **广义特征传播（GPR）**：将三类节点特征经两层 MLP 投影至统一空间得 $\mathbf{H}$，将异构图转为同构邻接矩阵 $\mathbf{A}$。特征按 $\mathbf{H}^s = \mathbf{P}\mathbf{H}^{s-1}$ 迭代传播，最终新闻表示为多步加权求和：$\mathbf{Z} = \sum_{s=0}^{S} w_s \mathbf{H}^s$，其中 $w_s$ 为可学习标量（可正可负），直接反映不同传播深度对预测的贡献。
- **局部与全局语义挖掘**：分别设置小步长 $s_l$（如2）和大步长 $s_g$（如20）生成 $\mathbf{Z}^l$ 与 $\mathbf{Z}^g$，对已标注新闻联合计算交叉熵：$\mathcal{L}_{sup} = \frac{1}{|\mathbb{N}^L|} \sum [\mathcal{L}_{ce}(p_i^l, y_i) + \lambda_g \mathcal{L}_{ce}(p_i^g, y_i)]$（式4）。
- **一致性正则化**：构建原型 $\bar{p}_i = (p_i^l + \lambda_g p_i^g)/2$，对无标签新闻计算原型与双视图预测的 KL 散度损失 $\mathcal{L}_{con}$（式6）。总目标为 $\lambda_{ce}\mathcal{L}_{sup} + (1-\lambda_{ce})\mathcal{L}_{con}$（式7），实现传播权重 $w$ 与 MLP 参数 $\theta$ 的端到端联合训练。

## 实验与结果
- **数据集**：MM COVID、ReCOVery、MC Fake、LIAR、PAN2020（覆盖健康、政治、多领域）。
- **基线**：TextCNN、TextGCN、HAN、BERT、SentenceBERT、HGNNR4FD、HeteroSGT。
- **主要结果**：LESS4FD 在全部 5 个数据集的 Acc、Pre、Rec、F1、AUC-ROC 上均优于所有基线。MM COVID 上 Acc 达 **0.973±0.010**，较最强基线 HeteroSGT（0.924）提升约 **5%**；其余数据集提升约 **2~4%**，且均通过 95% 置信度配对 t 检验。
- **关键结论**：高层语义建模是关键；双尺度 GPR 传播优于单一尺度或随机游走；一致性正则化平均带来约 **2%** 的性能增益；计算效率上 LESS4FD 单轮训练时间显著低于 HAN/BERT 等基线（Table 6）。

## 相关工作脉络
1. **文本分类基线（TextCNN/TextGCN/HAN/BERT）**：仅挖掘单篇新闻的词级或句级特征，无法建模新闻-实体-主题的高层关联，面对风格伪装型假新闻时上限明显。
2. **异构图基线（HGNNR4FD/HeteroSGT）**：同样构建异构图，但 HGNNR4FD 仅侧重局部语义，HeteroSGT 依赖随机游走易造成语义稀释；本文通过可学习权重的 GPR 与双尺度设计实现更精细的上下文聚合。
3. **依赖社交/传播上下文的检测方法**：需用户评论、转发链或互动数据，受隐私与网络演化限制适用性差；本文仅使用新闻文本，鲁棒性与泛化范围更广。
4. **LLM 特征增强工作**：现有研究多将 LLM 输出直接作为文本表征；本文转向 Prompt 驱动的实体/主题结构化抽取，并结合图传播挖掘关系，为 LLM+GNN 融合提供了新范式。

## 局限性与未来方向
- **局限性**：仅采用 GPT-3.5 与 Llama2 作为语义增强器，未探索对 LLM 本身进行下游任务微调或领域适配；话题数量需手动搜索 Coherence/Diversity/Sil 指标的综合平衡点，缺乏完全自动化选取机制。
- **未来方向**：扩展至更多 LLM 架构与开源模型；探索面向假新闻检测的 LLM 轻量微调或动态 Prompt 优化；研究自适应话题数选择与跨语言/跨模态假新闻检测。

## 研究启发与可借鉴点
1. **LLM 角色转换**：不将 LLM 仅视为黑盒 Embedding 器，而是通过 Prompt 工程将其转化为结构化知识抽取工具，再与图模型结合挖掘深层关系，该思路可迁移至医疗、法律等知识密集型文本分析任务。
2. **双尺度图传播的统一设计**：用同一套 GPR 框架通过调节步长同时捕获局部细节与全局上下文，避免多分支网络带来的复杂度，计算开销低且易于复现，适用于各类图分类/节点分类任务。
3. **原型一致性正则化**：利用不同视图预测的加权平均作为锚点，对无标签数据施加一致性约束，可有效缓解标签稀缺问题，值得借鉴至半监督学习与噪声标签学习场景。
4. **严谨的消融与统计验证**：逐模块消融（∅HG、∅E、∅T、∅CR）结合 95% 置信度 t 检验，清晰剥离各组件贡献，实验设计可作为后续工作的标准对比范式。

## 关键术语表
- **LESS4FD**：LLM Enhanced SemanticS mining for fake news detection，本文提出的假新闻检测框架。
- **Generalized PageRank (GPR)**：广义 PageRank，通过可学习标量权重对多步图传播特征进行加权融合，替代传统固定衰减的扩散机制。
- **异构图 (Heterogeneous Graph)**：包含多种类型节点（新闻、实体、主题）及显式关系边的图结构，用于建模高层语义网络。
- **一致性正则化 (Consistency Regularization)**：利用未标注数据，约束模型在不同语义视图下的预测分布保持一致，提升半监督泛化能力。
- **Bertopic**：基于 Transformer 的神经主题模型，用于自动提取新闻主题词及其权重分布。
- **双尺度语义挖掘**：通过短步长（局部）与长步长（全局）图传播分别捕获单篇新闻细节与跨文档宏观叙事模式。

## 可复现要素
- **数据集**：MM COVID、ReCOVery、MC Fake、LIAR、PAN2020（均为公开学术数据集，统计见 Table 7）。
- **代码/权重**：论文未提及开源代码仓库或预训练权重；提示词（Prompt）声明公开，但未提供具体链接。
- **关键超参**：局部步长 $s_l \in [2, 12]$，全局步长 $s_g \in [15, 25]$，$\lambda_g \in [0.1, 0.9]$，$\lambda_{ce} \in [0.1, 0.9]$；采用 10 折交叉验证（8:1:1 划分）。
- **硬件环境**：Rocky Linux 8.6，12-core CPU，1× NVIDIA Volta GPU（30G RAM）。
