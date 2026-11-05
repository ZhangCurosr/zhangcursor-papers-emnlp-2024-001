---
title: "On-the-Influence-of-Gender-and-Race-in-Romantic-Relationship"
source: https://aclanthology.org/2024.emnlp-main.29.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:48:34"
field: "NLP公平性与偏见评估"
keywords: ["大型语言模型", "偏见评估", "名字替换实验", "异性恋规范偏见", "关系预测", "公平性", "DDRel"]
innovations: ["通过受控名字替换实验首次在关系预测任务中揭示LLM的异性恋规范偏见与亚裔名字系统性劣势", "提出embedding可分性分析作为偏见归因方法，证明亚裔名字嵌入中性别信息编码较弱导致模型难以利用性别线索", "构建兼顾gender-neutral名字与非线性bin的精细化名字分类体系，支持对性别信息连续谱的因果分析"]
benchmarks: ["DDRel"]
---

# 论文速读：On-the-Influence-of-Gender-and-Race-in-Romantic-Relationship

## 一句话总结
本文通过受控的名字替换实验，系统揭示了大型语言模型（LLMs）在电影剧本对话的浪漫关系预测任务中存在**异性恋规范偏见（heteronormative bias）**和**对亚裔名字的跨种族关系偏见**：模型对同性伴侣和涉及亚裔名字的配对预测浪漫关系的召回率显著低于不同性别/白人或黑人伴侣。

## 研究问题与动机
- **核心问题**：LLM在基于对话预测两人浪漫关系时，是否会因角色名字所关联的性别与种族身份而产生偏见性预测，从而忽视非主流关系类型？
- **现有不足**：先前工作（Cao et al., 2022; Cheng et al., 2023; An et al., 2023）多通过识别个体 stereotype 属性来研究性别/种族偏见，未直接从"两人关系推断"角度检验异性恋规范偏见与跨种族偏见。
- **社会背景**：美国跨种族婚姻与同性婚姻长期面临法律与社会歧视，若LLM延续此类偏见，将加剧代表性伤害（representational harms）与资源配置不公（allocational harms）。
- **动机**：探索LLM是否会在关系预测中反映人类社会的异性恋规范与种族偏见，并为构建包容性技术提供证据支撑。

## 核心贡献（创新点）
1. **首次系统研究LLM在关系预测任务中的异性恋规范偏见**：通过受控名字替换实验证明，模型对同性伴侣（男女均覆盖）的浪漫关系预测recall显著低于不同性别伴侣，且男-男配对的偏见强度高于女-女配对。
2. **发现亚裔名字在关系预测中存在系统性劣势**：涉及至少一个亚裔名字的跨种族/同种族配对，模型预测浪漫关系的recall显著低于黑人、西班牙裔和白人配对，但当角色双方均为亚裔时recall最低（0.68）。
3. **揭示亚裔名字嵌入中性别信息编码较弱是偏见成因之一**：通过分析contextualized embeddings并用logistic regression分类，发现亚裔名字嵌入区分性别的准确率（53.3%）接近多数类基线（54.2%），远低于其他种族（Black: 96.4%、Hispanic: 80.5%、White: 99.9%），说明模型难以从亚裔名字中提取性别信息，进而影响其关系推理。
4. **与已有工作的本质区别**：不同于仅考察个体属性刻板印象的研究，本文聚焦"二元关系预测"这一下游任务，直接将姓名 demographics 与关系类型预测关联，揭示了偏见在关系推断层面的具体表现。

## 方法详解
- **任务定义**：给定对话 $C = ((S_1, u_1), (S_2, u_2), \ldots, (S_n, u_n))$，预测说话者A与B之间的关系类型（13类之一），将 Lovers/Spouse/Courtship 定义为浪漫关系，其余为非浪漫关系。
- **模型**：Llama2-7B-chat、Llama2-13B-chat（官方实现）、Mistral-7B-Instruct（HuggingFace实现），zero-shot提示，temperature=0，nucleus sampling默认参数，最大生成长度512。
- **数据集**：DDRel（Jia et al., 2021）测试集的子集，选取327条原始角色为不同性别的对话（训练/测试集中无同性浪漫关系样本）。其中271条经人工筛选去除了显式性别提示词（如"sir""ma'am"）。
- **名字收集**：
  - 性别研究：从Rosenman et al. (2023)中选取四种族（Asian/Black/Hispanic/White）各30个名字，按SSA数据按女性占比分为10个非线性bin（覆盖从0%到100%女性比例的名字），确保包含gender-neutral名字。
  - 种族研究：收集80个强种族+性别指示性名字（每种族×每性别各10个），Asian/Black/Hispanic要求≥90%人口归某种族，White要求≥70%。
- **名字替换实验**：将原对话中角色名替换为所有可能的名字组合对（per race 或 per race-pair），保持对话内容不变，测试模型预测变化。
- **评估指标**：以**recall**为主要指标（假设偏见表现为特定群体的浪漫recall偏低），同时报告precision、F1和accuracy。
- **匿名对照实验**（§B）：将角色名替换为占位符"X"/"Y"，获得纯上下文基线（Table 2），用于衡量名字引入的demographic信息对预测的影响幅度。
- **嵌入分析**（§4）：提取Llama2-7B对每个名字在15条浪漫+15条非浪漫随机对话中出现的contextualized embedding，共209,800个embedding，训练logistic regression分类器预测名字关联的性别或种族，对比多数类基线。
- **Prompt选择**：在原始数据上测试多个prompt formulation，选择accuracy最高且最接近文献报告的prompt（Figure 4），在名字替换实验前确定。

## 实验与结果
- **数据集**：DDRel测试集，327条不同性别角色对话，13类关系标签（表3），浪漫类共254条（Lovers 182 + Courtship 15 + Spouse 57），非浪漫356条。
- **性别配对结果**（Llama2-7B，Figure 2）：
  - 不同性别配对recall最高；同性别配对recall显著降低，男-女 > 女-女 > 男-男。
  - 所有种族下，含女性名字的配对recall均高于含男性名字的配对（Table 4），说明模型将女性名字与浪漫关系更强关联。
  - Asian/Hispanic名字的男女配对recall差距小于White/Black，可能源于模型难以区分这些名字的性别。
- **种族配对结果**（Llama2-7B，Figure 3）：
  - 涉及亚裔名字的配对recall显著偏低；双方均为亚裔时recall最低（**0.68**）。
  - 其他种族内/跨种族配对间无显著差异，未观察到对非亚裔跨种族配对的强烈偏见。
  - Precision across all racial pairs稳定在0.80–0.82（Figure 8），说明低recall非precision下降所致。
- **匿名基线**（Table 2）：完全匿名时Llama2-7B recall为0.6887，与各名字替换条件下的recall偏差显著，证实名字demographic信息确实影响了预测。
- **嵌入分类结果**（Table 1）：
  | 目标变量 | 种族 | Logistic回归准确率 | 多数类基线 |
  |---|---|---|---|
  | 性别 | Asian | 53.3±12.7% | 54.2±0.0% |
  | 性别 | Black | 96.4±2.9% | 54.2±0.0% |
  | 性别 | Hispanic | 80.5±13.0% | 54.2±0.0% |
  | 性别 | White | 99.9±0.2% | 53.9±0.3% |
  | 种族（One-vs-All） | Asian | 97.6±1.9% | 50.6±0.2% |
  | 种族（One-vs-All） | Black | 70.5±6.3% | 50.6±0.4% |
- **最强结果**：Llama2-7B在匿名条件下recall为0.6887；涉及亚洲名字的配对recall降至0.68；女-女配对recall高于男-男配对，暗示更强的对男-男关系的异性恋规范偏见。
- **Llama2-13B与Mistral-7B**呈现相同趋势（Appendix Figures 6-10）。

## 相关工作脉络
1. **姓名偏见研究**（Wang et al., 2022; Maudslay et al., 2019; Shwartz et al., 2020; An & Rudinger, 2023）：证明LLM对名字存在系统性差异处理，本文延续此脉络但聚焦于关系预测任务而非个体属性推断。
2. **社会常识推理中的偏见**（Cao et al., 2022; Cheng et al., 2023; An et al., 2023）：多采用 stereotype 属性匹配方式，本文改用受控名字替换的因果实验范式，直接测量下游任务性能变化。
3. **关系预测与社交关系建模**（Jia et al., 2021; Tigunova et al., 2021; Seraj et al., 2021）：本文使用DDRel数据集，将其扩展至公平性评估维度。
4. **机器翻译中的同性关系偏见**（Stewart & Mihalcea, 2024）：与本文发现相呼应，不同NLP任务中均存在对同性关系的系统性忽视。
5. **算法公平性评估框架**（Blodgett et al., 2020）：本文引用其 representational harms 与 allocational harms 概念，分析偏见对LGBTQIA+群体的社会影响。
6. **性别中立名字与包容性评估**（Nozza et al., 2022; Felkner et al., 2023）：本文在名字收集中纳入gender-neutral bin，借鉴了此类工作的方法设计。

## 局限性与未来方向
- **数据源覆盖不足**：DDRel测试集中无同性浪漫关系对话，无法直接评估模型在真实同性关系语境下的表现；名字来源亦有限，难以覆盖更广泛的种族与性别认同。
- **隐式性别提示词难以完全消除**：尽管筛选了无显式性别词的对话，但对话中仍可能存在隐式性别线索（如代词指向第三方、性取向暗示等），可能混淆分析。
- **Prompt敏感性**：不同prompt formulation可能导致准确率差异，未来需系统研究prompt对偏见程度的影响。
- **In-context learning潜力未探索**：未来可研究通过ICL示例是否有助于缓解偏见。
- **名字统计归类与自我认同不一致**：基于统计关联的名字分类无法涵盖个体的自我认同多样性，需在伦理框架下审慎对待。
- **未来方向**：扩展至更多种族/性别认同类别；开发偏见缓解方法（如debiasing names embeddings、prompt干预）；探索跨语言/跨文化场景下的偏见泛化。

## 研究启发与可借鉴点
1. **受控名字替换实验范式**：保持对话内容不变、仅替换角色名字，可有效隔离demographic属性对下游任务的影响，适用于多种关系/社会推理任务的公平性评估。
2. **Embedding可解释性分析作为偏见归因工具**：通过训练线性分类器检测contextualized embeddings中是否编码了特定demographic信息，可为偏见来源提供可量化的因果证据。
3. **Gender-neutral名字bin的设计**：将名字按女性占比分为非线性bin（而非简单二元男女）并纳入neutral名字，使实验能更精细地捕捉性别信息连续谱对模型行为的影响。
4. **Recall作为主要公平性指标的选择依据充分**：针对"系统性忽视特定群体"的偏见类型，recall比accuracy或F1更能直接反映偏见的社会危害，此指标选择值得借鉴。
5. **可与本团队方向的结合点**：将本方法迁移至其他社交关系任务（如友谊检测、权力动态分析、家庭关系推断），或扩展至多语言场景研究跨文化偏见，均具有较高可行性。

## 关键术语表
**Heteronormative bias（异性恋规范偏见）**：假设并偏好传统异性恋关系与性别角色，将非异性恋/非传统关系边缘化的社会偏见，本文发现LLM在关系预测中复现了此类偏见。

**Contextualized embeddings（上下文嵌入）**：模型根据输入语境动态生成的词向量，本文通过分析名字embedding中性别/种族信息的可分性来解释偏见的产生机制。

**Recall（召回率）**：在所有真实浪漫关系样本中被正确预测的比例，本文选用recall作为主要指标以捕捉模型对非主流关系类型的系统性忽视。

**Name-replacement experiment（名字替换实验）**：控制对话内容不变、仅替换角色名字的研究方法，用于隔离demographic属性对模型预测的因果影响。

**DDRel（Dialog-based Relationship dataset）**：Jia et al. (2021)构建的电影剧本对话关系预测数据集，包含13类关系标签，本文使用的测试集子集为327条不同性别角色对话。

**Representational harms（代表性伤害）**：Blodgett et al. (2020)提出的概念，指技术系统对特定群体的歪曲、缺失或刻板化表征所造成的社会伤害。

**Allocational harms（资源配置伤害）**：指技术系统因偏见导致资源、机会或服务的分配不公，如本文所述广告系统可能将跨性别夫妻排除在优惠之外。

**One-vs-All classification（一对多分类）**：用于评估种族信息编码的二分类策略，每次将某一特定种族作为正类、其余所有种族合并为负类进行分类。

## 可复现要素
- **数据集**：DDRel（Jia et al., 2021），可从ACL Anthology获取；名字数据来源为Rosenman et al. (2023)与Social Security Administration (SSA)公开数据。
- **代码/权重**：模型权重为开源（Llama2与Mistral-7B-Instruct均可从官方/HuggingFace获取）；论文附录提供了详细实验配置，但未公开实验代码（论文未提及代码仓库链接）。
- **关键超参**：temperature=0，nucleus sampling默认参数，max generation length=512，logistic regression最大迭代1000，random state=0，5次随机划分取平均。
- **实验规模**：每个种族870次实验（性别研究），1600次实验（种族研究，每种族对400次）；匿名实验共327次。
