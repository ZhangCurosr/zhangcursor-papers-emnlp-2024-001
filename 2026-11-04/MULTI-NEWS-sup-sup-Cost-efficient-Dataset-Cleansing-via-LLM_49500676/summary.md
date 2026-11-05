---
title: "MULTI-NEWS-sup-sup-Cost-efficient-Dataset-Cleansing-via-LLM"
source: https://aclanthology.org/2024.emnlp-main.2.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:09:48"
field: "多文档摘要与数据集质量"
keywords: ["dataset cleansing", "multi-document summarization", "LLM-based annotation", "chain-of-thought", "self-consistency", "data quality"]
innovations: ["首次利用LLM+CoT+多数投票对已有真实数据集进行低成本去噪清洗", "构建MULTI-NEWS+数据集并证明下游任务性能提升", "提供可解释的推理链供人工复核"]
benchmarks: ["Multi-News", "ROUGE", "BERTScore", "BARTScore"]
---

# 论文速读：MULTI-NEWS+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation

## 一句话总结
论文提出了一种基于大语言模型的低成本数据集清洗方法，通过链式思维（CoT）与多数投票机制，自动识别并移除 Multi-News 数据集中与摘要无关的噪声文档，构建了质量更高的 MULTI-NEWS+ 数据集。

## 研究问题与动机
1. **数据集噪声严重影响下游模型性能**：自动爬取的网页数据常包含与目标摘要无关的噪声文档（如平台系统消息、爬取错误内容），导致模型训练效果下降。
2. **人工标注成本高昂且耗时**：传统数据集清洗依赖人工标注员，需要大量预算（如多人投票或专家验证），且标注质量受主观偏差影响。
3. **现有过滤策略不适用于多文档摘要任务**：前序研究（如 Urlana et al., 2022）采用的粗粒度规则过滤方法主要针对单文档摘要任务，在 Multi-News 等多文档场景下效果有限。
4. **LLM 数据标注在已有数据集清洗中的应用尚未充分探索**：尽管 LLM 已被用于新数据标注，但将其应用于已有真实世界数据集的质量提升仍属空白。

## 核心贡献（创新点）
1. **首次利用 LLM 对已有真实世界数据集进行去噪清洗**：提出了一套可扩展的 LLM 驱动数据清洗框架，区别于传统人工标注或基于规则的方法。
2. **结合 CoT 与自一致性投票的多代理标注机制**：通过 5 个 LLM 代理独立标注并结合推理链，模拟人工多数投票过程，提升分类可靠性。
3. **构建并公开 MULTI-NEWS+ 高质量数据集**：移除 27,052 篇噪声文档（约占 15%+），保留原始 train/valid/test 划分，下游模型在 ROUGE 等指标上显著提升。
4. **低成本高效清洗策略**：整个 Multi-News 数据集（56,216 组样本）的清洗成本仅约 550 美元（每样本约 0.01 美元），远低于人工标注。
5. **提供可解释的推理过程**：CoT 机制为每条标注决策提供 rationale，便于人工复核与错误修正，这是 LLM 标注区别于传统众包标注的核心优势。

## 方法详解
1. **问题定义**：将文档相关性判断形式化为二分类任务——判断每个来源文档是否与摘要相关，若无关则标记为噪声文档并移除。
2. **提示设计（Prompt）**：采用 3-shot CoT 提示模板（详见 Appendix J），要求模型输出推理过程（rationale）和标注结果（文档编号或 'None'）。
3. **多代理自一致性投票**：部署 5 个独立的 GPT-3.5-turbo-0125 代理，每个代理独立读取摘要与文档集后进行标注，采用 majority voting 聚合结果。
4. **推理流程**：
   - 输入：摘要 + N 个来源文档
   - 每个代理执行 CoT 推理，分析每篇文档与摘要的关键信息关联度
   - 输出：不相关文档的编号列表（如 'Document 2|Document 4'）
   - 聚合：若超过半数代理标记某文档为无关，则该文档被移除
5. **规模与成本**：共处理 56,216 组（摘要，文档集），每组约消耗 3,500 tokens（输入）+ 100 tokens（输出），单样本成本约 0.01 美元，总计约 550 美元。
6. **输出**：MULTI-NEWS+，保留原始划分比例（train 80%, valid 10%, test 10%），从原 156,289 篇文档中移除了 27,052 篇噪声文档，另有 379 组无任何相关文档的样本被单独标注。

## 实验与结果
1. **实验设置**：
   - 模型：BART-large-cnn（406M 参数）、T5-base（220M 参数）
   - 数据集：Multi-News vs. MULTI-NEWS+（相同 test split）
   - 指标：ROUGE-1/2/L、BERTScore-F1、BARTScore
   - 训练细节：Adam, lr=2e-5, 3 epochs, batch size=16, 单张 A100 GPU
   
2. **主要结果（Table 2）**：
   | 模型 | 指标 | Multi-News | MULTI-NEWS+ | 提升 |
   |---|---|---|---|---|
   | BART-large-cnn | ROUGE-1 | 48.64 | **49.17** | +0.53 |
   | BART-large-cnn | ROUGE-2 | 18.86 | **19.04** | +0.18 |
   | BART-large-cnn | ROUGE-L | 24.11 | **24.36** | +0.25 |
   | BART-large-cnn | BERTScore | 0.6401 | **0.6418** | +0.0017 |
   | T5-base | ROUGE-1 | 40.11 | **40.45** | +0.34 |
   | T5-base | ROUGE-2 | 13.90 | **14.17** | +0.27 |
   | T5-base | ROUGE-L | 21.58 | **21.84** | +0.26 |
   | T5-base | BERTScore | 0.6003 | **0.6027** | +0.0024 |

3. **最强结果**：BART-large-cnn + MULTI-NEWS+ 在所有指标上均优于原始 Multi-News，ROUGE-1 提升最大（+0.53）。

4. **对比基线**：Urlana et al. (2022) 的基于规则的过滤方法在 Multi-News 上效果不佳，各指标反而低于原始数据集（如 BART ROUGE-1 仅 47.48），证明规则方法不适用于多文档摘要场景。

5. **人工评估**：对 379 组"无任何相关文档"的极端样本进行人工校验，人机一致率达 93.9%（356/379）。

6. **误差分析**：少量 FN（漏检）案例源于模型对"部分相关"文档的严格判断，或 CoT 推理中被次要信息干扰。

## 相关工作脉络
1. **Urlana et al. (2022) - TeSum**：提出了基于规则的摘要数据集过滤方法（如空摘要检测、压缩率阈值），但本文证明其在 Multi-News 多文档场景下效果有限，凸显了需要新的清洗策略。
2. **Wang et al. (2021) - "Want to reduce labeling cost? GPT-3 can help"**：开创性地将 GPT-3 用于数据标注，证明 LLM 标注可媲美甚至超越人工，为本工作的方法基础。
3. **Ding et al. (2023) - "Is GPT-3 a good data annotator?"**：系统评估 GPT-3 在不同任务和领域的标注能力，支持了 LLM 作为低成本替代标注者的可行性。
4. **He et al. (2024) - ANNOLLМ**：利用 CoT 和 LLM 改进众包标注质量，本文借鉴了其 CoT 设计思路，但应用场景不同（数据集清洗而非标注质量提升）。
5. **Choi et al. (2024) - "GPTs are multilingual annotators"**：探索 LLM 在多语言序列生成任务中的标注能力，本文在方法与框架上与之呼应，但聚焦于已有数据集的去噪而非从零标注。
6. **Swayamdipta et al. (2020) - Dataset Cartography**：提出通过数据集地图学预筛出需清洗的样本，本文在 Future Work 中提及可作为预筛选策略以降低清洗成本。

## 局限性与未来方向
1. **分类错误仍可能存在**：即便使用多数投票和 CoT，个别文档仍可能被误判（如将部分相关但贡献微弱的文档标记为噪声），未来可采用加权投票或引入更强大的模型作为专家标注者。
2. **数据集特性的现实反映**：原始 Multi-News 中的噪声反映了真实网络爬取场景，若目标是训练能处理噪声的实时系统，则 MULTI-NEWS+ 可能不适用——作者建议两种数据集互补使用。
3. **模型依赖**：当前使用 GPT-3.5-turbo-0125，可探索成本更低或性能更强的模型（如 GPT-4o-mini、LLaMA-3、Gemma-2 等开源替代）。
4. **伦理偏见风险**：CoT 推理可能引入伦理偏见（Shaikh et al., 2023），未来需系统性评估。
5. **未扩展到其他数据集**：本文仅为 Multi-News 的案例研究，未来计划将此方法推广至其他领域的数据集。

## 研究启发与可借鉴点
1. **LLM 驱动的数据清洗范式可迁移**：将 CoT + 自一致性投票应用于其他含噪声的大规模数据集（如 DocRED、MultiWOZ），验证其通用性。
2. **成本效益分析框架**：以明确的经济指标（0.01 美元/样本）论证 LLM 清洗的成本优势，可作为后续研究的基准参考。
3. **可解释标注的实用价值**：CoT 提供的推理链便于人工复核与错误追溯，这一设计可复用于需要高可信度的标注任务。
4. **预筛选降本策略**：结合 dataset cartography 等预筛技术，仅对"可疑"样本执行完整 CoT 标注，可进一步降低大规模数据集的清洗成本。
5. **专家-众包混合验证**：Future Work 中提出的"加权多数投票"思路（强模型作为专家，普通模型作为众包）可探索为分层标注架构。

## 关键术语表
**MULTI-NEWS+**：本文构建的高质量多文档摘要数据集，通过 LLM 清洗去除了原始 Multi-News 中的噪声文档。
**Chain-of-Thought (CoT)**：链式思维推理，要求模型在输出最终答案前展示逐步推理过程，提升标注可解释性。
**Self-Consistency**：自一致性，通过多次独立采样并取多数投票结果来提高 LLM 推理的可靠性。
**Multi-Document Summarization**：多文档摘要，从多个相关源文档中提取信息并生成单一摘要的任务。
**Dataset Cleansing**：数据集清洗，识别并移除数据集中的噪声、错误或无关样本以提升数据质量的过程。
**Annotation Rationale**：标注理由，模型对每项标注决策给出的解释性推理文本。
**Dataset Cartography**：数据集地图学，通过分析训练动态来绘制和诊断数据集特性的方法，可用于预筛需清洗的样本。

## 可复现要素
- **数据集**：MULTI-NEWS+ 已公开可用（论文声明 "We make MULTI-NEWS+ and our source code publicly available"）
- **代码**：源代码已开源（论文链接指向 GitHub，见 Appendix C footnote 6）
- **关键超参**：
  - LLM: GPT-3.5-turbo-0125，5 个并行代理
  - Prompt: 3-shot CoT，约 3,500 tokens 输入 + 100 tokens 输出
  - 训练: Adam, lr=2e-5, 3 epochs, batch size=16, 单张 NVIDIA A100 40GB
  - 模型权重: facebook/bart-large-cnn, googlet5/t5-base（HuggingFace 开源权重）
- **评估指标实现**：rouge-score 库（F1 分数），bert-base-uncased 用于 BERTScore，facebook/bart-large-cnn 用于 BARTScore
