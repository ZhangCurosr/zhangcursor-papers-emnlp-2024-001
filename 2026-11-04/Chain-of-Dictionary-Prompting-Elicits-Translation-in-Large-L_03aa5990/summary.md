---
title: "Chain-of-Dictionary-Prompting-Elicits-Translation-in-Large-L"
source: https://aclanthology.org/2024.emnlp-main.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:14:41"
field: "多语言自然语言处理"
keywords: ["多语言机器翻译", "提示工程", "链式词典", "低资源语言", "大语言模型"]
innovations: ["提出链式多语言词典提示框架COD提升LLM翻译能力", "首次在200+语言规模验证词典增强效果并超越NLLB", "证明链式词典优于few-shot ICL和分解词典"]
benchmarks: ["FLORES-200", "chrF++", "BLEU", "COMET"]
---

# 论文速读：Chain-of-Dictionary-Prompting-Elicits-Translation-in-Large-L

## 一句话总结
本文提出 COD (Chain-of-Dictionary Prompting) 框架，通过将链式多语言词典作为先验知识注入提示中，显著提升了 LLM 在多语言机器翻译（MNMT）中的表现，特别是在低资源语言翻译任务上取得了大幅提升，甚至超越 SOTA 翻译器 NLLB 3.3B。

## 研究问题与动机
- **核心问题**：LLM 虽然具备无监督多语言翻译能力，但在翻译低资源语言时仍存在显著困难，尤其是稀有词汇和词频较低的语言对。
- **现有方法不足**：
  1. 少数样本上下文学习（few-shot ICL）难以检索到与目标翻译相关的示范，特别是在低资源语言场景下效果有限。
  2. 双语词典方法仅利用源语言和目标语言的映射，缺乏跨语言线索。
  3. 现有的词典增强方法未充分利用多语言训练对跨语言性能的促进作用。

## 核心贡献（创新点）
- **提出 COD 框架**：通过将链式多语言词典注入提示中，为 LLM 提供先验知识来增强翻译能力；与双语词典方法的本质区别在于引入了辅助语言链条，提供了更强的跨语言线索。
- **系统性评估 200+ 语言**：在 FLORES-200 基准上对 ChatGPT、InstructGPT、BLOOM 等模型进行全方向评估，揭示现有 LLM 在大量语言对上仍有改进空间；与先前工作的区别在于首次将词典方法规模化应用到 200 多种语言。
- **验证 COD 优于 few-shot ICL**：证明链式词典方法在低资源语言翻译上优于传统 few-shot 演示；关键差异在于词典比示范更容易获取且不受相关示例限制。
- **超越 SOTA 翻译器**：COD 辅助的 ChatGPT 在 X→En 方向上超越了 NLLB 3.3B；这是首次证明无微调的 LLM 可通过提示工程超越专用翻译模型。

## 方法详解
- **提示结构设计**：每个句子提示包含两部分：(1) 标准翻译提示"Translate the following text from <source-language> into <target-language>: <source-sentence>"；(2) 链式多语言词典部分"<word X in source-language> means <word X in target-language> means <word X in auxiliary-language 1> means <word X in auxiliary-language 2>."
- **词典构建流程**：首先使用 LLM（如 ChatGPT）通过提示"Extract the words from the following texts: <input-sentence>"提取关键词；然后使用 NLLB 等现成翻译模型将提取的单词翻译成不同语言构建词典。
- **辅助语言选择**：选用法语 (fra_Latn)、德语 (deu_Latn) 和葡萄牙语 (por_Latn) 三个高资源语言作为辅助语言，形成包含源语言、目标语言和三个辅助语言的 5 语言链条。
- **去停用词优化**：可通过截断停用词来减少计算开销，约减少 1/3 的词典提示，同时保持满意翻译性能。
- **词典质量控制**：使用 NLLB 3.3B 翻译罕见词汇后回译，通过 ChatGPT 验证回译是否与原文等价，成功率约 71%。

## 实验与结果
- **数据集**：FLORES-200 full devtest（1,012 个句子，覆盖 200 种语言），使用 chrF++、BLEU 和 COMET 作为评估指标。
- **主要结果**：
  1. ChatGPT (GPT-3.5-TURBO) 在 En→X 方向上，135/200 种语言（67.5%）得到提升，71/135（52.6%）提升超过 5 分 chrF++。
  2. 最佳提升：英语→塞尔维亚语（西里尔字母）从 3.08 提升到 42.63，提升 13 倍。
  3. 在 335/400（83.75%）的方向上都得到改善。
  4. X→En 方向：所有 200 种语言都得到提升。
  5. COMET 得分：COD 平均 0.325 vs 基线 0.277。
  6. **超越 NLLB 3.3B**：在 X→En 方向上 COD (66.12) 超越 NLLB (54.77)。
- **消融实验**：链式词典优于分解词典和多语言词典；辅助语言选择高资源语言效果更好；few-shot ICL 效果不如 COD。

## 相关工作脉络
- **基于词典的神经机器翻译**：Zhang and Zong (2016) 使用双语词典增强 NMT；Arthur et al. (2016) 结合离散翻译词汇表；Ghazvininejad et al. (2023) 使用短语级词典提示——本文首次将词典方法扩展到链式多语言词典用于 LLM 翻译。
- **LLM 翻译提示**：Brown et al. (2020) 开创 few-shot 翻译；Lin et al. (2022) 和 Wang et al. (2023) 评估 LLM 翻译能力——本文在此基础上引入词典先验知识。
- **词汇约束 NMT**：Hokamp and Liu (2017)、Post and Vilar (2018) 提出硬约束解码；本文采用软性提示方式而非约束解码。
- **多语言预训练**：Liu et al. (2020) 的 mBART 展示多语言训练对跨语言性能的促进——本文受此启发使用多语言词典。
- **少样本上下文学习**：Agrawal et al. (2022) 研究示范选择——本文证明词典方法优于基于 BERTScore 检索的 few-shot ICL。

## 局限性与未来方向
- **语言覆盖有限**：仅评估 200 种语言，而全球有数千种语言。
- **少量语言性能下降**：虽然影响通常不显著，但 COD 对极少数语言有轻微降级。
- **推理时间增加**：COD 最多增加 1.8 倍推理时间，提示长度最多增加 3 倍。
- **辅助语言固定**：当前固定使用 3 种辅助语言，未探索更多语言链的组合。
- **未与微调方法对比**：未直接比较需要微调的方法（如 Jiao et al. 2023 的 ParroT）。
- **未来方向**：探索更长链条、更多辅助语言选择、自动选择最优辅助语言。

## 研究启发与可借鉴点
- **词典作为通用先验知识**：将外部知识（词典）以链式形式注入提示的思路可扩展到其他 NLP 任务，如文本生成、问答等需要词汇知识的任务。
- **多语言辅助链路设计**：通过高资源语言作为桥梁辅助低资源语言翻译的策略，可迁移到其他跨语言任务如跨语言信息检索。
- **提示工程与效率平衡**：停用词截断策略展示了如何在保持性能的同时控制提示长度和计算开销，对实际部署有参考价值。
- **无需微调的性能提升**：证明仅通过提示工程即可超越专用模型，为资源受限场景下的模型利用提供了新思路。

## 关键术语表
- **COD (Chain-of-Dictionary Prompting)**：一种将链式多语言词典注入提示的框架，用于增强 LLM 的多语言机器翻译能力。
- **FLORES-200**：Facebook 发布的包含 200 种语言的多语言翻译基准数据集。
- **chrF++**：基于字符 n-gram F-score 的机器翻译评估指标，对形态丰富语言更敏感。
- **NLLB (No Language Left Behind)**：Meta AI 发布的开源多语言神经机器翻译系统。
- **Few-shot In-Context Learning (ICL)**：通过在提示中提供少量示例来引导模型完成任务的推理方式。
- **多语言词典链**：将同一词汇在不同语言中的翻译以链式结构组织（如"A means B means C means D"）。

## 可复现要素
- **数据集**：FLORES-200 full devtest（公开可用）
- **代码/权重**：论文未提及开源代码，使用 GPT-3.5-TURBO API 和 NLLB 3.3B
- **关键超参**：辅助语言为法语、德语、葡萄牙语；词典构建使用 NLLB 3.3B；停用词截断可减少约 1/3 词典条目
