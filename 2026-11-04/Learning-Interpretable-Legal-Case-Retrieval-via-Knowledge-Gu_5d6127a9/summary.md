---
title: "Learning-Interpretable-Legal-Case-Retrieval-via-Knowledge-Gu"
source: https://aclanthology.org/2024.emnlp-main.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:46:43"
field: "法律信息检索与可解释检索"
keywords: ["legal case retrieval", "knowledge-guided reformulation", "large language model", "interpretable retrieval", "contrastive learning", "MaxSim"]
innovations: ["法律知识引导的两步案例重构", "子事实级启发式对比学习", "MaxSim+Sum 聚合在法律跨匹配中的适用与可解释性分析"]
benchmarks: ["LeCaRD", "LeCaRDv2"]
---

# 论文速读：Learning-Interpretable-Legal-Case-Retrieval-via-Knowledge-Gu

## 一句话总结
本文提出 KELLER，一种基于大语言模型的法律知识引导案例重构方法，通过将复杂法律案例解构为结构化的罪名-事实"子事实"，实现高效且可解释的法律案例检索，在 LeCaRD 和 LeCaRDv2 两个基准上均达到 SOTA。

## 研究问题与动机
- 法律案例检索中，query 和 document 均为长篇结构化法律文书，平均长度常超过主流检索器（如 BERT）的 512 token 输入限制，传统长文本建模策略难以有效覆盖全部犯罪行为的描述。
- 既有方法多依赖启发式规则或 LLM 自身知识进行直接文本摘要，但在高度专业化的法律领域易出现不准确的关键信息压缩，丢失多罪名情形下的分散事实。
- 现有长文本处理方案（如分段交互、扩展上下文窗口）难以在"长文本理解"和"复杂语义匹配"两个任务上同时取得突破。

## 核心贡献（创新点）
- **法律知识引导的两步重构**：先由 LLM 提取罪名与法条并建立映射，再借助法条作为高层抽象引导生成各罪名的关键子事实；区别于仅做朴素摘要（如 PromptCase）的前作。
- **子事实级跨匹配与 MaxSim 聚合**：将案例重构为若干子事实后，采用 MaxSim+Sum 进行细粒度交叉匹配；相比 kernel pooling 等软聚合更契合"一问一答最多对应一个关键事实"的法律匹配场景。
- **双层级对比学习**：在案例级 ranking loss 基础上，引入基于启发式标签的子事实级对比学习，同时捕获粗粒度和细粒度匹配信号；避免仅靠 case-level 监督导致的细粒度信号缺失。
- **强零样本与微调双重表现**：在无微调 zero-shot 设定下同样显著优于 SAILER、PromptCase 等基线，体现重构策略对领域适配的鲁棒性。

## 方法详解
- **罪名与法条抽取 + 后处理**：用 LLM 提取案例中的罪名列表与法条标题；将法条标题扩展为完整条文内容，并参照法律专家数据库建立"罪名—法条"映射，得到结构化 crime-article 对。
- **法律知识引导的事实总结**：以 crime-article 对为提示，让 LLM 从冗长案例中概括每个罪名的起因、程序与结果（每部分不超过 100 词），形成"标题为罪名、正文为事实"的子事实片段。
- **编码与相似度矩阵**：对 query 子事实集合 $\{q_i\}$ 和 document 子事实集合 $\{d_j\}$ 分别经预训练 encoder 得到 $[CLS]$ 向量，计算归一化点积得相似度矩阵 $M_{i,j}$。
- **MaxSim+Sum 聚合**：$s_{q,d}=\sum_i \max_j M_{i,j}$，兼顾效果、效率与可解释性；每个 query 子事实只需在文档中找最高匹配子事实。
- **双级对比学习损失**：
  - 案例级：$\mathcal{L}_{\mathrm{R}}=-\log \frac{\exp(s_{q,d^+}/\tau)}{\exp(s_{q,d^+}/\tau)+\sum_{d^-}\exp(s_{q,d^-}/\tau)}$
  - 子事实级：对正/负例子事实对构造 ranking loss $\mathcal{L}_{\mathrm{S}}$
  - 总损失：$\mathcal{L}=\mathcal{L}_{\mathrm{R}}+\alpha \mathcal{L}_{\mathrm{S}}$（文中 $\tau=0.01,\alpha=0.9$）
- **子事实标签启发式构造**：对正例文档，优先取与 query 子事实同罪名的子事实为正；无同罪名时取相似度最高对为正；对负例文档仅在存在正样本时选取同罪名子事实作为负样本，其余丢弃以减少噪声。

## 实验与结果
- **数据集**：LeCaRD（107 query / 9,195 doc）、LeCaRDv2（800 query / 55,192 doc）；指标 MAP、P@3、NDCG@k（k=3,5,10）。
- **主要结果（fine-tune，LeCaRDv2）**：KELLER MAP=68.29、P@3=63.13、NDCG@3=84.97、NDCG@5=85.63、NDCG@10=87.61，显著优于所有基线。
- **相对最强基线提升**：较 PromptCase（MAP 62.25、NDCG@3 78.51）提升约 +6.04（MAP）、+6.46（NDCG@3）；较 SAILER（MAP 60.62、NDCG@3 78.67）提升约 +7.67、+6.30。
- **Zero-shot（LeCaRDv2）**：KELLER MAP=65.87、NDCG@3=83.33，仍显著优于 SAILER（62.80 / 79.38）等。
- **查询类型**：在复杂争议型查询上提升更大，LeCaRD 与 LeCaRDv2 分别提升 24.04%、13.41%（相对基线）。
- **消融**：去除 KGCR 模块降幅最大；MS→Mean 影响最小（数据以单罪名案例为主）；MS→NC 降幅最大（破坏跨匹配结构）；双级对比学习各自均有正向贡献。

## 相关工作脉络
- BM25/TF-IDF 等传统统计检索与LEGAL-BERT、SAILER 等法律专用预训练模型：前者依赖词频与结构特征，后者通过领域预训练提升语义表示；KELLER 在此基础上引入法律专家知识引导的结构化重构，绕过单纯扩大上下文窗口的路径。
- BERT-PLI、Lawformer 等长文本建模方法：在输入表示层面扩展容量；KELLER 将长文本理解"卸载"到 LLM 重构阶段，编码器只需处理简短子事实。
- PromptCase 等 LLM 摘要方案：仅做朴素总结，难以覆盖多罪名分散事实；KELLER 以罪名-法条对为锚点分罪名生成子事实。
- HyDE、Query2Doc 等查询重写方法：面向通用问答/短文档检索；KELLER 处理 query 与 doc 同为长篇案例的独特设定。
- Kernel pooling、MaxSim 等粗排/精排聚合策略：KELLER 采用后者并在法律案例结构下给出针对性设计与可解释性分析。

## 局限性与未来方向
- 需要构建法律专业知识库（罪名-法条映射）作为后处理环节，增加工程开销。
- 在线推理需调用大模型处理 query，带来额外计算成本（虽可通过 vLLM 等手段优化）。
- 面对复杂真实 query 时推荐结果仍可能不够精确，现有案例库也不一定能完全覆盖用户需求。
- 未来可扩展更多专业知识源，并结合生成式模型输出自然语言解释。

## 研究启发与可借鉴点
- **"理解-匹配"解耦范式**：将复杂长文档理解交给 LLM 的知识引导重构，将匹配任务留给轻量编码器，值得迁移至其他专业领域的长文档检索。
- **子事实级对比学习标签构造**：基于业务规则（如同罪名匹配）在仅有 case-level 标签下构造高质量细粒度监督信号，避免 naive 正负样本引入的噪声。
- **MaxSim 在法律匹配中的适用性**：对"一对多但主要单点强匹配"的场景，MaxSim+Sum 兼具效率与可解释性，可作为强基线保留。
- **领域知识库+LLM 提示**的串联设计：先抽取结构化元素、再利用知识库对齐，再指导生成，为其他垂直领域（医疗、专利）的可解释检索提供模板。

## 关键术语表
- **Legal case retrieval**：以相似法律案例为查询与候选文档的信息检索任务，用于辅助司法决策。
- **Knowledge-guided case reformulation**：利用罪名与法条等法律专家知识，引导 LLM 将长案卷重构为结构化子事实的预处理过程。
- **Sub-fact**：以罪名为标题、关键案件事实为主体的精简片段，是检索匹配的基本单元。
- **MaxSim**：对相似度矩阵沿文档维度取最大后再沿 query 维度求和的聚合算子，强调最强局部匹配。
- **Dual-level contrastive learning**：同时在案例级与子事实级施加对比/ranking 损失，捕获多粒度匹配信号。
- **LeCaRD / LeCaRDv2**：面向中文法律体系的案例检索基准数据集，后者规模更大且包含争议型查询标注。
- **PromptCase**：基于 LLM 将案例压缩至 50 词的朴素摘要方法，作为本文对照的 LLM 重构基线。

## 可复现要素
- 数据集：LeCaRD、LeCaRDv2（论文附录提供统计，通常公开可得，但需遵循原数据集使用协议）。
- 代码/权重：论文未明确说明开源仓库地址；使用了 Qwen-72B-Chat 进行重构、SAILER 作为文本编码器，相关权重可在对应项目开源地址获取。
- 关键超参：温度 $\tau=0.01$；子事实损失权重 $\alpha=0.9$；每案例最多保留 4 个罪名；学习率 $1e{-5}$；batch size 128；encoder 使用 [CLS] pooling；输入上限依 SAILER 默认设置。
- 硬件：4× Nvidia Tesla A100-40G。
