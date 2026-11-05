---
title: "Integrating-Structural-Semantic-Knowledge-for-Enhanced-Infor"
source: https://aclanthology.org/2024.emnlp-main.129.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:11:14"
field: "信息抽取预训练"
keywords: ["信息抽取", "预训练", "AMR", "对比学习", "图神经网络", "自监督学习", "结构化语义"]
innovations: ["提出SKIE预训练框架，利用AMR图谱结构语义知识自监督学习IE模型", "设计确定性+概率性双路径拓扑增强策略提取多粒度凝聚子图", "提出T-GSN拓扑感知图编码器，保留边关系与子结构信息"]
benchmarks: ["ACE04", "ACE05", "CoNLL03", "CoNLL04", "SciERC", "CASIE", "CrossNER"]
---

# 论文速读：Integrating-Structural-Semantic-Knowledge-for-Enhanced-Infor

## 一句话总结
本文提出 SKIE，一种面向信息抽取（IE）的新型预训练框架，通过将抽象意义表示（AMR）图谱中的结构语义知识引入对比学习，以低成本无标注数据大幅提升模型在 NER、RE、EE 等任务上的表现。

## 研究问题与动机
- **标注成本高昂**：现有 IE 预训练方法严重依赖大规模标注数据，而手工标注成本高、可扩展性差，限制了模型的泛化能力。
- **单一文本知识局限**：既有方法仅利用标注文本知识，忽视了文本中内在的结构语义知识，难以捕捉复杂语义结构。
- **跨任务迁移受限**：由于预训练数据类别有限、规模不足，模型在少样本和零样本场景下适应能力弱。
- **AMR 作为自监督信号潜力未被充分挖掘**：AMR 能以低成本自动捕获文本的结构语义表示，但尚未被广泛用于 IE 预训练。

## 核心贡献（创新点）
- **提出 SKIE 预训练框架**：将 AMR 图谱中的结构语义知识引入 IE 预训练，无需人工标注即可自监督学习，区别于 UIE/USM/Mirror 等依赖标注数据的预训练方法。
- **设计拓扑增强模块提取凝聚子图**：结合确定性（DTE）与概率性（PTE）两种拓扑增强策略生成多粒度 k-core 凝聚子图，为对比学习提供多层次结构语义样本，区别于已有工作仅利用完整图的表达方式。
- **提出拓扑感知图子结构网络（T-GSN）**：在 GSN 基础上引入边关系感知机制，保留图中边关系信息与凝聚结构信息，区别于传统 GCN 只聚合邻居全局特征、丢失子结构细节的不足。
- **构建文本-图对比学习范式**：利用 triplet loss 学习文本与 AMR 图/凝聚子图之间的语义对应关系，增强模型对复杂文本语义的理解能力，区别于纯文本预训练方法。

## 方法详解
- **AMR 图谱构建**：使用基于 Transformer 的 AMR 解析器将文本 s 解析为有向图 G=(V,E)，节点表示实体/谓词等语义单元，边表示语义关系，每条边带有关系类型 r∈R。
- **凝聚子图提取（拓扑增强模块）**：
  - **确定性拓扑增强（DTE）**：基于 k-core 算法提取多级凝聚子图集合 G={G^k|k=k_min,...,k_max}；通过节点出现频次计算重要性权重 w_v(v_i)，并沿边扩散更新边权重 w_e'(e_ij)；利用 PPR 扩散矩阵增强图连通性。
  - **概率性拓扑增强（PTE）**：引入衰减因子 ε 控制节点/边的丢弃概率 P'(v_i)=(1-w_v'(v_i)·ε)·P，弥补 DTE 可能丢失的知识，保证生成子图的多样性与凝聚性。
- **文本编码器**：采用 RoBERTa-Large 作为文本编码器，提取包含词汇、短语及上下文细微差异的向量表示。
- **图编码器（T-GSN）**：
  - 改进 GSN，引入关系特异性变换：h_i^(l+1)=σ(Σ_r Σ_{j∈N_i^r} (1/n_{i,r}) W_r^(l) h_j^(l) + W_o^(l) h_i^(l))
  - 将更新特征与编码特征输入 T-GSN：T-GSN(v_i)=AGG(h_i^(l+1), h_j^(l+1), x_i, x_j)，保留边关系与子结构细节。
- **对比学习模块**：采用 triplet loss：L=max(0, |s-g_+|^2-|s-g_-|^2+m)，其中 m 为 margin，促使同语义文本与图在嵌入空间中更近，不同语义样本更远。
- **下游微调**：采用统一指令格式 [I]+schema label，通过 biaffine attention 生成多 span 循环图邻接矩阵，使用 Circle Loss 作为微调损失。

## 实验与结果
- **数据集**：预训练使用约 134 万条未标注语料（NER: 26.2 万、RE: 91.7 万、EE: 6.6 万）；评测使用 8 个 IE 基准（ACE04/ACE05/CoNLL03/CoNLL04/SciERC/CASIE）及 5 个 CrossNER 零样本数据集。
- **基线方法**：TANL、UIE、UniEX、USM、UTC-IE、Mirror、MetaRetriever。
- **主要结果**：
  - NER：CoNLL03 上 F1=93.62（+0.47 over USM），ACE05 上 88.52（+1.38 over USM）
  - RE：SciERC 上 46.90（+8.13 over USM），CoNLL04 上 78.91（+0.07 over USM）
  - EE：ACE05-Arg 上 61.77（+5.94 over USM），ACE05-Tgg 上 75.15（+0.71 over USM）
  - 整体平均提升：NER +1.49、RE +6.75、EE +3.42
- **Few-shot 结果**：CoNLL03 10-shot F1=85.46（最优），ACE05-Arg 10-shot F1=43.84（最优）
- **Zero-shot 结果**：CrossNER 平均 F1=58.03，优于 USM(41.98) 和 Mirror(54.46)
- **消融实验**：去除 T-GSN 导致平均 F1 下降 7.69；去除凝聚子图导致下降 6.97；最优超参：decay ε=0.2，图编码器 3 层，margin=0.1，k=5

## 相关工作脉络
- **UIE（Lu et al., 2022）**：基于 schema prompt 的文本到结构生成预训练方法，依赖标注数据；SKIE 无需标注，利用 AMR 自监督信号。
- **USM（Lou et al., 2023）**：统一语义匹配框架，使用三种监督数据集预训练；SKIE 在无标注语料上预训练，降低标注成本。
- **Mirror（Zhu et al., 2023）**：多槽元组数据接口预训练方法；SKIE 通过结构语义知识增强而非仅依赖数据接口重组。
- **MetaRetriever（Cong et al., 2023）**：元预训练检索算法；SKIE 直接学习文本-图对比表示，不依赖检索机制。
- **TANL（Paolini et al., 2021）**：将结构化预测翻译为增广自然语言任务；SKIE 引入图结构知识而非仅文本翻译视角。
- **GLiNER（Zaratiana et al., 2024）**：通用 NER 模型；SKIE 在零样本对比中部分数据集优于 GLiNER，但整体略逊于 UniNER/GoLLIE 等大模型。

## 局限性与未来方向
- **计算成本高**：预训练使用近百万级语料，文本与图谱分别编码导致训练时间长。
- **任务数据不平衡**：EE 预训练数据量（6.6 万）远少于 NER（26.2 万）和 RE（91.7 万），限制 EE 任务性能上限。
- **输入长度限制**：RoBERTa-Large 最大序列长度为 512 tokens，难以处理长文本 IE 任务。
- **AMR 解析噪声**：AMR 解析可能引入错误，影响预训练质量，未来需降噪处理。
- **领域偏差**：互联网语料存在新闻等领域偏差，可能影响领域特定 IE 任务准确性。
- **未来方向**：探索滑动窗口或 Longformer/BigBird 等长文本模型；扩展至更广泛 NLP 任务。

## 研究启发与可借鉴点
- **AMR 作为自监督信号的创新思路**：将 AMR 图作为低成本结构化语义来源，为其他 NLP 任务（如语义角色标注、知识图谱构建）的预训练提供了可迁移范式。
- **确定性+概率性双路径拓扑增强策略**：DTE 保证凝聚结构质量，PTE 引入随机性增强多样性，这种组合策略可应用于图神经网络预训练的其他场景。
- **边关系感知的图编码器设计**：T-GSN 的关系特异性变换机制有效保留子结构细节，对关系抽取、事件抽取等需精细结构建模的任务具有借鉴价值。
- **文本-图对比学习范式**：triplet loss 桥接文本与 AMR 图的思路可扩展至其他结构化表示（如句法树、知识图谱）的预训练。
- **统一指令微调接口**：采用 [I]+schema label 的统一数据格式，便于跨任务迁移，可与本团队的多任务学习方向结合。

## 关键术语表
**AMR（Abstract Meaning Representation）**：一种以有向图形式表示句子语义的结构化表示，节点为语义单元，边为语义关系。
**Cohesive Subgraph（凝聚子图）**：图中连接紧密的子图结构，本文通过 k-core 算法提取，捕捉文本的多层次语义核心。
**k-core**：图中每个节点度数至少为 k 的最大连通子图，用于识别图的核心结构。
**T-GSN（Topology-aware Graph Substructure Network）**：拓扑感知图子结构网络，在 GSN 基础上引入关系特异性变换，保留边关系与子结构信息。
**DTE（Deterministic Topological Enhancement）**：确定性拓扑增强策略，通过固定规则生成凝聚子图并扩散边权重。
**PTE（Probabilistic Topological Enhancement）**：概率性拓扑增强策略，基于节点权重引入随机丢弃机制，生成多样化子图。
**Contrastive Learning（对比学习）**：通过拉近正样本对、推远负样本对的学习范式，本文用于对齐文本与 AMR 图表示。
**Triplet Loss**：三元组损失函数，形式为 max(0, |anchor-positive|²-|anchor-negative|²+m)，用于约束嵌入空间中的样本间距。

## 可复现要素
- **预训练数据**：约 134 万条未标注语料（NER/RE/EE 三类），部分来自公开数据集（AnatEM、GENIA、OntoNotes5、FewRel、NYT10、RAMS 等），论文未说明完整开源状态，代码仓库：https://anonymous.4open.science/r/SKIE
- **模型代码**：已匿名开源
- **关键超参数**：decay ε=0.2，图编码器层数=3，k=5，margin=0.1，text encoder lr=1e-5，graph encoder lr=1e-3，epochs=50，batch size=64，d_h=128
- **评测指标**：Span-based offset Micro-F1
