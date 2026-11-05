---
title: "Hateful-Word-in-Context-Classification"
source: https://aclanthology.org/2024.emnlp-main.10.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:08:58"
field: "仇恨言论检测与词汇语义计算"
keywords: ["hate speech detection", "word sense disambiguation", "subjective NLP", "computational lexicon semantics", "contextual embeddings"]
innovations: ["提出HateWiC任务，首次系统性研究语境中仇恨词的主观性分类", "构建4000实例多标注者数据集并揭示hate-heterogeneous sense现象", "验证定义嵌入与标注者人口统计信息对主观仇恨分类的互补价值"]
benchmarks: ["HateWiC", "DINU1", "DINU2"]
---

# 论文速读：Hateful-Word-in-Context-Classification

## 一句话总结
论文提出了仇恨词在语境分类任务（HateWiC），构建了包含4000个WiC实例的多标注者数据集，探究仇恨词意义的描述性（定义）与主观性（语境依赖）之间的交互关系，并验证了引入词义定义与标注者人口统计信息可有效提升仇恨词识别性能。

## 研究问题与动机
- **现有HSD研究多聚焦句级别分类，缺乏对词级别仇恨语义的计算建模**：当前仇恨言论检测（HSD）研究主要集中在社交帖子等完整话语的分类，而针对特定语境中仇恨词含义的细粒度建模仍严重不足。
- **通用LM对领域特异/新型仇恨用法敏感度不足**：大型语言模型（如BERT）难以捕捉随社会事件演变的非标准或新兴仇恨表达方式（如"Oreo"从饼干名变为种族贬义词）。
- **仇恨词义兼具描述性与表达性，不能仅依赖字典定义**：仇恨词的含义存在于从描述性到表达性的连续体上，其仇恨内涵不仅取决于词典定义，还高度依赖语境与读者身份，这种现象被称为"hate-heterogeneous senses"。
- **现有标注多采用单一视角，忽视了仇恨判断的主观性差异**：传统HSD数据集与系统通常基于单一（多数）视角标注，忽略了不同社会文化背景下的解读差异。

## 核心贡献（创新点）
- **提出HateWiC任务与数据集**：构建了首个专注于语境中仇恨词分类的benchmark，包含约4000个WiC实例，每个实例由3位不同背景标注者进行仇恨评级（Not hateful/Weakly hateful/Strongly hateful）。
- **揭示hate-heterogeneous sense现象并量化其影响**：发现319个词义定义下同时存在仇恨与非仇恨标注实例（如"carrot cruncher"），证明词义的仇恨性并非仅由定义决定，而是高度语境依赖。
- **探索多维度输入对主观仇恨分类的影响**：系统研究了词义定义（Wiktionary定义与T5生成定义）、上下文嵌入、标注者人口统计信息三种输入类型对多数标签预测与个体标注预测的差异化作用。
- **验证定义嵌入与标注者信息在不同场景下的互补价值**：定义嵌入在out-of-vocabulary场景与仇恨同质词义上效果显著，而标注者信息在主观分歧场景下可缓解性能下降达11个百分点。

## 方法详解
- **Sense表示构建**：
  - **WiC embeddings**：将示例句子输入预训练编码器，提取目标词子词位置的最后一层隐藏状态（多子词取平均）。
  - **Def embeddings**：使用Wiktionary提供的词义定义句子作为输入，采用与WiC相同的平均池化策略生成定义嵌入。
  - **T5Def embeddings**：使用FLAN-T5 Base模型生成上下文特定的定义，prompt模板为"[SENTENCE] What is the definition of [TERM]?"，再对生成定义进行嵌入。
- **标注者信息建模**：将标注者人口统计信息（年龄、性别、族裔）以模板形式拼接："Reader is [AGE], [GENDER] and [ETHNICITY]."，通过相同编码器生成Ann embedding，与sense embedding拼接后输入分类器。
- **分类架构**：采用四层MLP分类器（隐藏层维度300→200→100→50），学习率0.0005，最大迭代次数10，经网格搜索调优。编码器对比实验包括BERT-base、HateBERT（仇恨言论再训练）、WSD Biencoder（在WordNet上微调的词义消歧双编码器）。
- **评估任务**：
  - **多数标签预测**：gold label为3位标注者多数投票的二进制结果。
  - **个体标注预测**：gold label为每位标注者的单独评级（转为二进制）。
- **数据划分策略**：采用10折交叉验证，分为Random（按句子随机划分）与OoV（按词汇划分，测试未见词汇的零样本能力）两种设置。

## 实验与结果
- **数据集规模**：4029个实例（排除"Cannot decide"后），覆盖1888个独特词义、826个独特词汇，平均每个词义4.88个示例。总体仇恨/非仇恨比例接近平衡（47.2%/52.8%）。
- **多数标签预测**（Table 2）：
  - **最佳结果**：HateBERT + WiC+Def embeddings，Random split准确率达0.80，OoV split达0.77。
  - Def embeddings单独使用即优于WiC embeddings（HateBERT: 0.78 vs 0.75），二者组合效果最佳。
  - T5Def单独表现最差（0.67-0.70），但与WiC拼接后可达0.73-0.76。
  - LLaMA 2 zero-shot准确率仅0.68，显著低于专门设计的编码器方法。
- **个体标注预测**（Table 4）：
  - 整体准确率较多数标签预测下降2-5个百分点。
  - **Annotator信息缓解主观分歧**：在标注者与多数标签不一致的实例上，引入Ann embedding可缓解准确率下降最高达11个百分点。
  - **Hate-heterogeneous场景的性能差异**（Table 3）：对于仇恨同质词义，HateBERT+WiC+Def准确率达0.91；对于hate-heterogeneous词义，准确率骤降至0.49，下降47%。T5Def+WiC组合在此类场景下降幅更小（0.84→0.55，降幅29%）。
- **DINU基线对比**（Table 6）：在DINU1和DINU2数据集上，除纯T5Def外所有方法均优于Dinu et al. (2021)的最优结果，DINU1上提升幅度达8个百分点（0.90 vs 0.82）。

## 相关工作脉络
- **Dinu et al. (2021)**：首次系统探讨pejorative word disambiguation任务，使用MLP分类BERT嵌入，但数据集规模小且仅单视角标注；本文扩展至更大规模的多视角主观仇恨词分类。
- **Hoeken et al. (2023b)**：提出slur检测的dimension projection方法，需要预定义minimal pairs；本文指出该方法不适用于具有hate-heterogeneous特性的广泛仇恨词集合。
- **Fleisig et al. (2023)**：将标注者人口统计、偏好与经历作为输入预测个体标注；本文沿用此思路但聚焦于仇恨词级别的细粒度建模。
- **Qian et al. (2019)**：针对hate symbols（仇恨符号）的定义预测，但未解决仇恨与非仇恨的消歧问题；本文聚焦于具有双重含义的普通词汇。
- **Mendelsohn et al. (2023)**：研究dog whistles（暗语）的识别，强调其故意模糊性；本文指出dog whistles与hateful words属不同类别。
- **Giulianelli et al. (2023)**：提出definition-as-sense范式，用T5生成上下文特定定义进行词汇语义相似度计算；本文借鉴其定义生成方法并探索其在主观仇恨分类中的应用。

## 局限性与未来方向
- **词典数据质量限制**：依赖Wiktionary数据，其用户贡献性质可能导致定义不准确或不一致，质量低于专家编纂词典。
- **二元标签过于简化**：当前评估仅使用二值化标签（仇恨/非仇恨），忽略了原有三级标注的细粒度信息，可能损失主观性 nuance。
- **定义生成的准确性不足**：T5生成定义与真实定义的SacreBLEU仅3.822，说明自动生成定义质量有限，限制了其有效性。
- **未来方向**：探索更先进的定义生成技术（如更大模型或针对性微调）、研究其他标注者信息维度（如 gaze features）、分析语义关系（隐喻/转喻）与词汇类别（食物/动物）对仇恨判定的影响、跨语言对比研究。

## 研究启发与可借鉴点
- **主观NLP任务中引入标注者特征的有效性验证**：本文系统证明了在主观性强的分类任务中，将标注者人口统计信息作为模型输入可有效缓解性能下降，这一思路可迁移至情感分析、立场检测等主观NLP任务。
- **Hate-heterogeneous sense的量化分析方法**：通过对比同一词义定义下不同示例的标注分布，识别"hate-homogeneous"与"hate-heterogeneous"实例，这种基于标注变异性的细粒度分析框架可用于其他具有多义/主观特征的词汇现象研究。
- **定义嵌入与上下文嵌入的互补性设计**：本文证明了显式定义信息与隐式上下文信息的组合优于单一信息源，尤其对OOV词汇有益；这一设计可推广至其他需要结合外部知识的词义理解任务。
- **零样本LLM在主观词级任务上的局限性**：LLaMA 2 zero-shot仅达0.68准确率，远低于专用编码器方法，提示在高度主观、细粒度的词级任务上，通用指令微调模型仍存在明显不足，需探索领域适配或提示工程优化。

## 关键术语表
- **HateWiC（Hateful Word in Context Classification）**：本文提出的新任务，旨在判断特定语境中目标词的仇恨性。
- **Hate-heterogeneous senses**：指同一词义定义下，部分示例被标注为仇恨、另一部分被标注为非仇恨的现象，体现词义主观性。
- **WiC embeddings（Word in Context embeddings）**：从预训练编码器中提取的目标词在特定句子语境中的上下文向量表示。
- **Definition-as-sense paradigm**：将词义定义本身作为sense表示输入模型的范式，本文扩展至仇恨词分类任务。
- **Dimension projection**：通过计算最小对立对（minimal pairs）的差向量构建语义维度（如"hate dimension"）并进行分类的方法。
- **Krippendorff's alpha**：衡量标注者间一致性的统计量，本文报告中二分类alpha=0.45，三分类alpha=0.33。
- **OoV（Out-of-Vocabulary）split**：按词汇划分测试集的数据分割策略，用于评估模型对未见词汇的零样本泛化能力。

## 可复现要素
- **数据集**：HateWiC数据集约4000个实例，论文声明"available only upon request"（CC BY-NC 4.0许可证），需联系作者获取。
- **代码**：论文未明确提及代码开源状态，但提到使用Argilla平台进行标注、WiktionaryParser库解析数据。
- **关键超参**：MLP隐藏层维度(300, 200, 100, 50)，初始学习率0.0005，最大迭代次数10；T5定义生成prompt模板见Appendix C。
- **编码器模型**：BERT-base、HateBERT、WSD Biencoder（WordNet上训练7个epoch，batch size=8）。
- **LLaMA 2配置**：7B参数模型，zero-shot推理，max new tokens=10。
