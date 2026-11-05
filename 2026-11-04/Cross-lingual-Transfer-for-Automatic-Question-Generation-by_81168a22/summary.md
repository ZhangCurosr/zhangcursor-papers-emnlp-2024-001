---
title: "Cross-lingual-Transfer-for-Automatic-Question-Generation-by"
source: https://aclanthology.org/2024.emnlp-main.186.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:15:38"
field: "跨语言自动问答与生成"
keywords: ["跨语言问题生成", "XLT-QG", "QuIST", "疑问结构", "零样本迁移", "多语言生成", "代码切换"]
innovations: ["两阶段QTC+QG框架，推理时引入目标语言问题范例学习疑问结构", "仅用英文数据训练的小参数模型实现多语言零额外训练部署", "合成问题数据显著提升多语言QA模型性能"]
benchmarks: ["SQuAD1.1", "TyDiQA", "XQuAD", "MLQA"]
---

# 论文速读：Cross-lingual Transfer for Automatic Question Generation by Learning Interrogative Structures in Target Languages

## 一句话总结
本文提出 QuIST 方法，通过仅用英文 QA 数据训练、在推理时引入少量目标语言问题范例，使小参数多语言模型（mBERT + mT5）能在零目标语言训练数据下生成符合目标语言疑问句法结构的自动问题，在9种语言上超越多个 XLT-QG 基线并与 GPT-3.5-turbo 性能相当。

## 研究问题与动机
- **低资源语言 QG 数据稀缺**：现有 QA 数据集（SQuAD、HotpotQA、QuAC 等）以英语为主，其他语言标注成本和时间高昂。
- **跨语言迁移在 NLG 任务中易发生疑问句代码切换（interrogative code-switching）**：仅对 mPLM 微调编码器/解码器的 zero-shot XLT 方法无法完全避免生成问题时混入英语疑问表达（如"How long""When did"）。
- **既有 XLT-QG 方法依赖目标语言数据训练**：需单语语料、源-目标平行语料或有限 QA 范例，导致模型扩展新语言时需额外训练，语言可扩展性差。
- **商业大模型成本高、开源自训练 LLM 算力负担重**：需要一种参数量小、无需目标语言额外训练即可部署的高效方案。

## 核心贡献（创新点）
- **提出 QuIST 两阶段框架**：QTC（问题类型分类）+ QG（基于问题范例的问题生成），区别于以往直接微调 mPLM 端到端生成问题的做法。
- **推理时引入目标语言问题范例以学习疑问结构**：模型仅在英文数据上训练，在目标语言推理时借助该语言少数问题范例激活目标语言的疑问句法模式，从而缓解疑问句代码切换。
- **实现高语言可扩展性**：面向新语言无需额外参数更新，只需提供少量目标语言问题范例即可部署。
- **小参数模型达到 GPT-3.5-turbo 级性能**：使用 110M 参数的 mBERT（QTC）和 1.2B 参数的 mT5（QG），平均 ROUGE-L 达 30.53，优于多个 XLT-QG 基线并与 GPT-3.5-turbo 相当。
- **合成数据有效提升多语言 QA 训练**：QuIST 生成的合成问题用于训练多语言 QA 模型时，在多个语言上的 Exact Match 高于 GPT-3.5-turbo 和 prompt-tuned PaLM-540B。

## 方法详解
- **两阶段流程**：
  - **QTC 阶段**：将答案与上下文拼接为 `[CLS] answer [SEP] context [SEP]`，输入 mBERT（110M 参数），用 [CLS] 隐向量经前馈层 + softmax 预测 8 类问题类型（When/Where/What/Which/Who/Why/How_way/How_number），采用交叉熵损失训练。
  - **QG 阶段**：在 QTC 预测的类型下，选取对应类型的问题范例（question exemplars）作为输入的一部分，结合上下文与答案，由 mT5（1.2B 参数，1100M-1.2B 规模）以序列到序列方式生成问题，采用 teacher-forcing + 交叉熵训练。
- **问题类型定义**：基于英语 Wh- 疑问词划分为 8 类；`How` 开头若后接助动词归为 `How_way`，后接形容词/副词归为 `How_number`。
- **问题范例构建**：英语范例从 SQuAD1.1 训练集按类型抽样；目标语言范例从多语言 QA 数据集（TyDiQA/XQuAD/MLQA）取题，经 Google Translation API 译为英文后按同法构建。实验对比了范例数量 {1, 5, 10, 15}，每种 5 次随机采样，报告 25 次平均。
- **静态 vs 动态范例训练**：静态范例（训练全程使用同一组英文范例）在多数语言上优于动态范例（每样本从训练集采样），模型更聚焦范例句法。
- **推理无需目标语言训练**：模型参数固定，仅通过 QTC 预测类型 + 目标语言范例引导生成，实现零额外训练跨语言部署。

## 实验与结果
- **数据集**：训练使用 SQuAD1.1（79,321 训/8,283 验/1,190 测），评估使用9种语言（bn/zh/de/fi/hi/id/ko/SW/te）来自 TyDiQA、XQuAD、MLQA 的 QA 样例。
- **评估指标**：BLEU-4、METEOR、ROUGE-L（英文及多数语言）、SP-ROUGE（中文）。
- **主要自动评测结果（ROUGE-L 均值）**：
  - QuIST15 平均 **30.53**，优于 Baseline Adapter（24.53，+6.00）、Baseline Enc（20.82）等。
  - 在 fi（38.79）、te（30.74）、zh（47.71）、id（28.36）等语言上表现突出。
  - 与 GPT-3.5-turbo_10（平均 29.95）相当，部分语言（hi/ko/SW/te/zh）QuIST 更高。
- **人类评测**（6 种语言，3 位母语者评分）：在 de/fi/id 等语言上 QuIST 在 interrogative sentence、grammatical correctness、clarity、answerability、answer-match 五项指标上与 GPT-3.5-turbo 接近，优于 Baseline Adapter；在 bn/hi/SW 上略有差距但仍具可用性。
- **代码切换分析**：QuIST 在低资源语言（bn、SW）上有效抑制疑问句代码切换，与 Adapter 相当，显著优于 Baseline EncDec。
- **数据增强效果**：QuIST 合成数据训练的 multilingual QA 模型在6种语言平均 Exact Match 达 **59.65**，优于 GPT-3.5-turbo_10（57.79）和 prompt-tuned PaLM-540B（59.54）。

## 相关工作脉络
- **Sherborne & Lapata (2022), Wu et al. (2022a), Vu et al. (2022), Deb et al. (2023), Pfeiffer et al. (2023)**：跨语言迁移（XLT）在分类/生成任务中的工作，本文聚焦 QG 的疑问结构迁移。
- **Kumar et al. (2019), Chi et al. (2020), Shakeri et al. (2021), Wang et al. (2021), Agrawal et al. (2023)**：既有 XLT-QG 方法不同程度依赖目标语言数据（单语/平行语料/QA 范例），本文仅在推理时引入范例而不参与训练。
- **Maurya et al. (2021)**：通过冻结嵌入/解码器仅微调编码器缓解 catastrophic forgetting，本文在 preliminary 中发现该方法仍出现疑问句代码切换。
- **Pfeiffer et al. (2020, 2023), Deb et al. (2023)**：Adapter 等模块化跨语言迁移方法，需目标语言大规模单语语料训练语言适配器；本文方法无需此类训练。
- **Mallinson et al. (2020), Wu et al. (2022b)**：利用平行语料促进源-目标语言对齐的方法；本文不依赖平行数据。
- ** Agrawal et al. (2023) Qameleon**：用 PaLM-540B 仅 5 个目标语言示例 prompt-tune 生成多语言 QA；本文用更小模型、无需目标语言微调即达可比甚至更优的 QA 增强效果。

## 局限性与未来方向
- **适用语言受限于 mPLM 预训练覆盖**：mT5 覆盖 101 种语言，但未覆盖或资源极低的语言性能可能受限。
- **范例配置导致的性能不稳定性**：不同范例组合可能引起输出波动；仍有部分问题出现代码切换，需规则过滤进一步优化。
- **机器翻译范例效果弱于人工撰写范例**：推理时若目标语言范例由翻译获得，生成质量下降。
- **仅关注 Wh- 疑问句的 8 类划分**：其他语言可能具有更丰富的疑问结构，扩展性有待验证。
- **未来方向**：探索自适应范例选择、结合更多疑问句类型、提升极低资源语言泛化、优化代码切换过滤机制。

## 研究启发与可借鉴点
- **推理时注入范例而非训练时吸收**：为低资源语言生成任务提供了一种"零训练扩语言"的范式，可在问答、摘要等 NLG 迁移任务中复现。
- **静态范例优于动态范例的启示**：在 seq2seq 生成中，固定语法规例有助于模型聚焦句法而非被多样语义噪声干扰，值得在其他结构化生成任务中验证。
- **QTC 类型预测的误差传播可控**：relaxed 标注下 QTC F1 > 91%，说明两阶段设计中阶段间误差累积风险低，可借鉴到类似 pipeline 方法。
- **合成数据对下游 QA 的有效性强于自动指标预期**：Baseline Enc 在自动 ROUGE 上不如 Adapter，但在 QA 训练增强上更优，提示生成质量评估应结合下游任务验证。
- **可与本团队方向结合**：将 QuIST 的范例引导思路迁移到多语言对话系统、教育题库生成、跨语言信息抽取等场景，降低新语言接入成本。

## 关键术语表
- **Cross-lingual Transfer for QG (XLT-QG)**：将在英语上训练的问答生成模型迁移到低资源语言生成问题的任务设定。
- **QuIST (Questions by learning Interrogative Structures in Target languages)**：本文提出的两阶段跨语言自动问题生成方法。
- **Interrogative code-switching**：多语言生成中混入源语言（英语）疑问表达的失败现象，如用英语"How long"开头生成非英语问题。
- **Question exemplar**：按问题类型预置的少量目标语言问题样本，用于在推理时引导生成句法。
- **Question Type Classification (QTC)**：第一阶段，基于答案与上下文预测问题所属的 8 类疑问类型。
- **Zero-shot XLT**：仅在源语言数据上训练，直接在目标语言推理而不使用该语言标注数据的方法设定。
- **Catastrophic forgetting**：多语言模型在源语言上微调后对目标语言能力的退化现象。
- **SP-ROUGE**：基于 SentencePiece 子词切分的 ROUGE 变体，适用于中文等无空格分隔语言。

## 可复现要素
- **数据集**：SQuAD1.1 公开；评估用 TyDiQA、XQuAD、MLQA 公开。
- **代码/权重**：模型基于 HuggingFace 的 bert-base-multilingual-cased（110M）和 google/mt5-large（1.2B）；论文未明确开源代码与训练权重（"论文未提及" 开源仓库）。
- **关键超参**：QTC batch=8, lr=1e-5, warmup=1000；QG batch=16, lr=5e-5, warmup=1000；beam size=4；训练至验证集损失优化停止。
- **硬件**：单卡 NVIDIA Tesla A100-80GB。
