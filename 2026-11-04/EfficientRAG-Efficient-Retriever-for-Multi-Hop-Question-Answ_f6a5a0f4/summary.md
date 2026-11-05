---
title: "EfficientRAG-Efficient-Retriever-for-Multi-Hop-Question-Answ"
source: https://aclanthology.org/2024.emnlp-main.199.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:17:29"
field: "检索增强生成与多跳问答"
keywords: ["EfficientRAG", "多跳问答", "检索增强生成", "迭代检索", "小型模型", "query decomposition"]
innovations: ["纯小模型Labeler+Filter迭代检索，替代LLM多轮调用生成下一跳查询", "多跳问题分解+Token标注+负采样的合成数据构建流水线，规模化生产训练数据", "跨数据集迁移验证小模型学到的关系推理泛化能力"]
benchmarks: ["HotpotQA", "MuSiQue", "2WikiMQA"]
---

# 论文速读：EfficientRAG-Efficient-Retriever-for-Multi-Hop-Question-Answ

## 一句话总结
EfficientRAG提出了一种仅用小模型（DeBERTa-v3-large）完成的迭代多跳检索框架，通过Labeler和Filter两个轻量级组件替代大模型迭代生成查询并过滤噪声，在HotpotQA、MuSiQue、2WikiMQA三个多跳QA数据集上以极少检索chunk数量实现了可与LLM基线媲美的端到端性能，同时速度比LLM迭代方法快3倍以上。

## 研究问题与动机
- **单轮RAG在多跳问题上失效**：传统RAG（如RAG、RETRO）仅用原始查询检索一次，面对需要多步推理的问题无法获取全部所需信息。
- **现有迭代检索高度依赖LLM调用**：Iter-RetGen、Self-Ask、Decompose等方法每轮都需调用LLM进行查询改写/分解，导致延迟高、成本高，且需要维护多样化的few-shot prompt。
- **小模型具备关系推理潜力**：多跳问题中的关系类型有限且远少于实体数量（Zhu et al., 2023），小模型足以完成"识别关系→抽取关键token→生成下一跳查询"的任务。
- **噪声检索影响LLM生成质量**：实验表明，即使引入大量irrelevant chunks，LLM生成器的准确率也显著下降，说明"精准检索+去噪"比"多轮LLM改写"更重要。

## 核心贡献（创新点）
- **纯小模型迭代检索框架**：Labeler与Filter均为基于DeBERTa-v3-large的轻量token级分类器，每轮迭代不再调用LLM，与Iter-RetGen/Self-Ask等需要3–7次LLM调用的方法形成本质差异。
- **端到端数据合成流水线**：利用Llama-3-70B-Instruct自动合成含multi-hop decomposition、token-level标注、next-hop过滤和hard negative的多跳训练数据（合计约538k条），绕开了人工标注瓶颈。
- **标签驱动的检索停止机制**：引入`<Continue>`/`<Terminate>` chunk标签，让检索过程动态决定何时停止，避免无效chunk浪费检索预算。
- **高效的Next-hop查询构建**：Filter通过提取已标注有用token替换查询中未知部分来构造下一跳查询，避免了LLM自由生成导致的格式/领域漂移。
- **跨数据集强迁移能力**：在HotpotQA上训练的EfficientRAG在2WikiMQA上测试精度反超在该数据集上单独训练的模型，表明其学习的是通用关系推理模式而非数据集特有知识。

## 方法详解
**整体架构**：EfficientRAG由Labeler（含Token Label + Chunk Tag两个head）和Filter两个子模块构成，均基于DeBERTa-v3-large（304M参数，24层）。

**Labeler**：
- 输入：`query + chunk`序列，经编码器得到每个token的嵌入表示。
- Token Label头：通过全连接层将每个token嵌入映射到二维空间（useful / useless），用SpaCy分词后进行逐token二分类。
- Chunk Tag头：对序列平均池化后的表示映射到二维空间，输出`<Continue>`（需继续检索）或`<Terminate>`（检索终止）。

**Filter**：
- 输入：`query + labeled tokens`的拼接序列，结构同Labeler。
- 功能：识别出可作为下一跳查询关键词的token，拼出新的检索查询，替代LLM改写。

**合成数据构建（§3.2）**：
1. **Multi-hop分解**：用Llama-3-70B-Instruct将多跳问题拆解为多个single-hop子问题并解析依赖关系。
2. **Token Labeling**：对每个子问题及其对应chunk，标注哪些token对回答问题有帮助（保留）/ 无用（丢弃）。
3. **Next-hop过滤**：将子问题与其依赖chunk的已标注token组合，生成理想的下一跳查询，并提取对应token。
4. **Negative Sampling**：检索与过滤后查询最相似但不相关的chunk作为hard negative，标记为`<Terminate>`。

**训练细节**：AdamW优化器，学习率5e-6，在4×A100 GPU上分别训练Labeler和Filter约10 GPU小时。检索器固定为Contriever-MSMARCO。

## 实验与结果
**数据集**：HotpotQA、MuSiQue、2WikiMQA（均为开放域多跳QA基准）。

**检索性能（Table 2）**：
- HotpotQA：Recall@6.41 = **81.84**（远超Direct-R@10的70.52）。
- 2WikiMQA：Recall@3.69 = **84.08**（远超Direct-R@10的61.58）。
- MuSiQue：Recall@6.09 = 49.51（较低，与该数据集复杂度较高有关）。
- 在相近召回率下，EfficientRAG只需检索约1/10的chunk数，体现极高检索效率。

**端到端QA性能（Table 3，Llama-3-8B-Instruct生成器）**：
- HotpotQA Acc = **57.86**（仅次于Iter-RetGen iter3的59.47）。
- MuSiQue Acc = 20.00（优于Self-Ask的19.53）。
- 2WikiMQA Acc = **53.41**（次优）。

**换用GPT-3.5生成器（Table 5，2WikiMQA）**：
- EfficientRAG Acc = **61.88**，超越Iter-RetGen iter3的60.60，达到SOTA。

**推理效率（Table 4，MuSiQue 200样本）**：
- LLM调用次数：EfficientRAG仅1次 vs Self-Ask的7.18次。
- Latency：3.62s vs Self-Ask的27.47s，**提速约7.6倍**；比Iter-RetGen iter3（9.68s）快约2.7倍。
- GPU利用率与LLM方法相近（~65%），说明额外开销主要来自模型推理本身。

**跨数据集迁移（Table 6）**：
- HotpotQA训练→2WikiMQA测试：Acc=56.59，优于2WikiMQA自训的53.41。
- 2WikiMQA训练→HotpotQA测试：Acc=53.38，略低于自训的57.86，但仍表现良好。

## 相关工作脉络
- **RAG / RETRO（Lewis et al., 2020; Borgeaud et al., 2022）**：单轮检索范式，仅用原始query检索，无法处理多跳问题；EfficientRAG在其基础上引入迭代检索与噪声过滤。
- **Iter-RetGen（Shao et al., 2023）**：每轮用LLM迭代检索+生成，性能较强但需3次LLM调用；EfficientRAG用小型模型替代LLM达成相似/更好的效果，且仅1次LLM调用。
- **Self-Ask（Press et al., 2023）**：通过LLM自问自答式分解问题，共需~7次LLM调用；EfficientRAG用小模型完成等价分解任务，大幅降低延迟与成本。
- **Active RAG（Jiang et al., 2023b）/ DR3（Gao et al., 2024）**：前者通过LLM决定是否继续检索，后者约束LLM不偏离主题；均依赖LLM，EfficientRAG则用轻量分类器完成类似判断。
- **Contriever-MSMARCO（Izacard et al., 2022）**：无监督对比学习密集检索器，被本文作为固定检索器使用，提供了稳定可靠的底层召回能力。
- **LLMLingua-2（Pan et al., 2024）**：利用LLM进行prompt压缩，与本文Token标注思路相似——都是通过小模型/结构化信号筛除噪声信息，可作为融合方向。

## 局限性与未来方向
- **仅开放域评测**：论文主要在开放域数据集上验证，未探讨闭域/垂直领域多跳QA，泛化性有待进一步验证。
- **生成器规模受限**：作者因时间和资源限制未测试更大LLM作为最终生成器，可能低估了EfficientRAG与更大模型的协同潜力。
- **MuSiQue表现偏弱**：该数据集 hop数更多、结构更复杂，当前Filter生成下一跳查询的能力在此类问题上有瓶颈。
- **合成数据质量依赖LLM**：训练数据由Llama-3-70B-Instruct生成，若生成过程中出现分解错误或标注偏差，可能传导至小模型训练。
- **可扩展方向**：将Labeler/Filter扩展到其他知识密集型任务（如Fact Verification、Open-Domain NLG）、引入更大Generator联合微调、探索多模态多跳检索。

## 研究启发与可借鉴点
- **小模型替代LLM做迭代决策**：在检索控制、查询改写、信息过滤等环节可用小型分类器替代LLM调用，显著降低延迟与成本，可迁移至文档路由、知识图谱query扩展等场景。
- **Token-level二分类标注范式**：Labeler的"保留有用token/丢弃无关token"思路可直接用于信息抽取中的span选择、摘要中的句子筛选等任务。
- **合成数据构建流水线可复用**：分解→标注→负采样三步法无需人工标注即可规模化生产高质量多跳数据，对缺乏领域标注的团队极具参考价值。
- **Chunk级`<Continue>`/`<Terminate>`信号**：以轻量分类器判断"还需不需要检索"可替代LLM-based early stopping，提升整体系统效率。
- **跨数据集迁移验证**：本文在HotpotQA/2WikiMQA间做迁移实验的设计值得借鉴，可作为评估模型"是否学到了泛化关系推理"的标准指标。

## 关键术语表
**EfficientRAG**：一种基于小型模型的多跳QA迭代检索框架，通过Labeler和Filter组件替代LLM进行查询生成与信息过滤。
**Labeler**：EfficientRAG的核心组件之一，对检索到的chunk进行逐token有用性标注并判断chunk整体是否应继续检索（`<Continue>`/`<Terminate>`）。
**Filter**：EfficientRAG的另一核心组件，基于query和已标注token生成下一跳检索查询，无需LLM调用。
**Multi-hop Question Answering**：需要整合多个信息源/推理步骤才能回答的问答任务，典型数据集包括HotpotQA、MuSiQue、2WikiMQA。
**Token Labeling**：对文本中每个token进行二元分类（useful/useless），以识别对回答子问题有帮助的关键片段。
**Contriever-MSMARCO**：基于无监督对比学习的密集信息检索模型，被EfficientRAG用作底层检索器。
**Recall@K**：在Top-K个检索结果中包含至少一个相关chunk的比例，用于评估检索质量。
**Iter-RetGen / Self-Ask**：基于LLM的多轮迭代检索基线方法，分别通过LLM迭代检索生成和自问式子问题分解实现多跳推理。

## 可复现要素
- **数据集**：HotpotQA、MuSiQue、2WikiMQA（均为公开数据集）。
- **代码开源**：https://aka.ms/efficientrag（论文声明）。
- **权重开源**：代码链接已提供，权重可通过代码仓库获取。
- **关键超参**：DeBERTa-v3-large（304M参数）；AdamW优化器，学习率5e-6；batch size论文未明确提及；训练时长约10 GPU小时（4×A100）。
- **检索器**：Contriever-MSMARCO。
- **生成器**：主实验使用Llama-3-8B-Instruct，表5使用GPT-3.5-turbo-1106。
- **数据合成LLM**：Llama-3-70B-Instruct。
