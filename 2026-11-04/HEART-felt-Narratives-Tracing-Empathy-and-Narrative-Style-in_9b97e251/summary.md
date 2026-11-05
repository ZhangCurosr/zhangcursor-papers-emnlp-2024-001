---
title: "HEART-felt-Narratives-Tracing-Empathy-and-Narrative-Style-in"
source: https://aclanthology.org/2024.emnlp-main.59.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:09:20"
---

# 论文速读：HEART-felt-Narratives-Tracing-Empathy-and-Narrative-Style-in

## 一句话总结
本文提出理论驱动的叙事风格分类体系 HEART，利用 GPT-4 / Llama 3 自动提取个人故事中的叙事元素，并通过 N = 2,624 的大规模众包实验实证揭示了“情感生动性、人物发展、情节体量”驱动共情的心理路径，同时证明共情响应具有显著的个性化差异。

## 研究问题与动机
- **核心问题**：叙事风格（即故事“如何被讲述”）究竟通过哪些具体维度影响读者对个人故事的共情？现有研究多聚焦故事内容或读者自身特征，对“讲述方式”的计算化量化与机制拆解仍不充分。
- **现有方法不足**：传统 NLP 工作依赖 LIWC 等词典或手工特征，难以捕捉情节转折、情感生动性、人物弧光等高阶抽象叙事手法；少量探索 LLM 叙事理解的工作集中于虚构文本或事件分割，缺乏面向真实个人故事、并与社会行为洞察挂钩的系统验证。
- **科学缺口**：如何将叙事学/心理学理论转化为可计算的标注体系，并用 LLM 替代/辅助人工标注，进而通过大规模人类行为数据反哺可解释的共情机制模型，尚属空白。

## 核心贡献（创新点）
1. **提出 HEART（Human Empathy and Narrative Taxonomy）分类体系**：首个融合 Keen (2006)、van Krieken 等心理学与叙事学理论的结构化叙事风格维度，覆盖人物认同、情节、视角与环境四大类共 12+ 细项，为计算叙事分析提供统一度量框架。与 prior work 零散的理论假设相比，本文首次将其系统化并支持自动提取。
2. **系统评估 LLM 在复杂叙事元素提取上的人机对齐程度**：对比 GPT-4 与 Llama 3 8B Instruct 在 50 份专家标注故事上的表现，证明在乐观语调、情感生动性、人物脆弱性、情节体量等维度上 LLM 显著优于 LIWC-22 词典对应指标，但提示了认知/评价类特征的系统性高估误差。与仅追求预测准确率的工作相比，本文以“可解释特征提取质量”为核心评估目标。
3. **发布 HEART-felt Stories Dataset 并完成大规模共情行为实验**：从 EMPATHICSTORIES / EMPATHICSTORIES++ 筛选 874 个高质量个人故事，配合 N = 2,624 参与者的 State Empathy 评分、叙事运输、特质共情与人口统计学问卷，构建开放数据集。填补了“带细粒度风格标注的大规模真实故事共情语料”空白。
4. **揭示叙事风格→共情的中介路径与个性化调节机制**：通过结构方程模型（SEM）与混合效应模型证明：情感生动性通过叙事运输（transportation）正向驱动共情；同故事内共情评分标准差显著大于零，且特质共情与情感生动性存在显著交互（est = 0.252, p < 0.001），说明“共情并非一刀切”。相比以往仅建模均值响应的 NLP 共情预测任务，本文强调机制可解释性与个体差异。

## 方法详解
- **HEART 分类体系设计**：基于心理学与叙事学文献，划分为四类：
  - *Character Identification*：人物平坦/圆形度（人物发展、脆弱性）、情感主体（情感生动性）、认知主体、道德主体（评价）、行动主体、感知主体（身体感知）、时间指涉（怀旧/展望）。
  - *Plot*：情节体量（事件频率与显著性）、情感转折（效价起伏）、结局（冲突释放程度）。
  - *Point of view*：叙述视角（第一人称等）。
  - *Setting*：环境生动性（世界构建/沉浸感）。
- **数据筛选**：原始 EMPATHICSTORIES（1,500 条）+ EMPATHICSTORIES++（500 条对话转录）→ 过滤有害话题与 <200 词短文 → 最终 874 条个人叙事。
- **专家标注与一致性检验**：随机抽取 50 条，由 2 名具备文本分析背景的专家按 Likert 量表标注 12 个非平凡元素；用 Krippendorf’s alpha（KA）、percent pairwise agreement（PPA）、Spearman’s ρ 评估，排除 KA < 0.2 的低一致特征（如 bodily sensations, temporal references）。
- **LLM 提示与提取**：向 GPT-4 与 Llama 3 8B Instruct 输入与专家相同的 codebook 与示例，要求输出单一整数评分；GPT-4 整体与人类一致性更高，故用其对全量 874 条故事进行批量标注。
- **共情众包实验**：Prolific 招募 2,624 人（男女均衡、白人为主、平均特质共情偏高），每人阅读 1 条故事后完成：① State Empathy Scale（10 项李克特）；② 勾选 contributing narrative elements；③ 开放式文本；④ 叙事运输（TS-SF）；⑤ 相似经历与 perceived similarity；⑥ 读者特征问卷（情绪状态、人口学、阅读习惯、SITES/TEQ）。每人支付 $1，平均耗时 7 分钟。
- **统计分析 pipeline**：
  - 故事级聚合：对每条故事取 3 位读者均值，Mann-Whitney u-test 比较高/低特征组的平均共情（Benjamini-Hochberg 校正，9 次比较）。
  - 个体差异建模：线性混合效应模型（以年龄/性别/特质共情/族裔分组为固定效应，故事 ID 为随机效应），似然比检验显著（p = 0.002）。
  - 路径分析：semopy 库构建 SEM，检验 *narrative style → transportation → empathy* 的中介路径，并引入 trait empathy 与 similar experience 作为调节/直接路径。

## 实验与结果
- **数据集**：HEART-felt Stories Dataset（874 stories）；专家标注子集 50 条；众包共情评分 2,624 条（每条至少 3 位独立读者）。
- **LLM vs Lexica 对比（Table 3）**：GPT-4 在 optimistic tone（ρ = 68.06***）、vivid emotions（59.21***）、character vulnerability（80.15***）上均优于 LIWC-22；仅在 cognition 维度 LIWC（41.29**）略高于 GPT-4（34.91*），但 Llama 3 在 cognition 上（52.89***）反超两者。GPT-4 整体与人类一致性最高。
- **故事级共情差异检验**：高 *character development*（p = 0.03）与高 *plot volume*（p = 0.03）的故事平均共情显著更高；前者为本文首次实证。
- **SEM 路径发现**：*vividness of emotions* 显著正向预测 *narrative transportation*，transportation 进而显著提升 *state empathy*；*similar experience* 与 *trait empathy* 亦为显著预测因子，但效应弱于 transportation。
- **个性化与共情方差**：同一故事的共情评分标准差显著大于 0（p < 0.001）；加入 demographic group 的混合模型拟合显著更优（p = 0.002）；*trait empathy × vivid emotions* 交互项显著（est = 0.252, p < 0.001），表明高特质共情读者对生动情感叙事更敏感，偏好呈现个性化。
- **最强结果**：GPT-4 在 character vulnerability 上达到 ρ = 80.15***（KA = 62.89, PPA = 86.50%），大幅领先 LIWC（ρ = −6.95, 不显著）；情感生动性经叙事运输中介驱动共情的路径在 SEM 中最为稳健。

## 相关工作脉络
- **词典/心理语言学方法**（Roshanaei 2019; Zhou 2021; Yaden 2024）：依赖 LIWC 词库或 LDA 主题，擅长情感词频与简单认知词统计，但无法刻画情节体量、人物弧光等高阶结构；本文 LLM 提取在多数维度上显著超越此类对应指标。
- **叙事学与读者反应研究**（Keen 2006; van Krieken 2017; Kuzmicová 2017; Fernandez-Quintanilla 2020）：提出视角、情感转折、 foregrounding 等理论，但多为小样本定性/实验室操控；本文将其计算化并提供大规模真实社区故事验证。
- **LLM 叙事理解**（Michelmann 2023; Zhu 2023; Sap 2022; Subbiah 2024）：聚焦事件分割、虚构角色建模或叙事流量化，本文转向“真实个人故事”的共情导向风格提取，并强调与社会行为洞察的闭环。
- **共情预测 NLP**（Shen 2023; Buechel 2018; Sharma 2020; Bao 2021）：多以端到端分类/回归为目标优化 F1/Acc，缺乏对“为何共情”的路径解释；本文明确以可解释机制挖掘优先，而非单纯刷榜。
- **情感计算与个性化建模**（Konrath 2018; Christov-Moore 2014; Koopman 2015）：揭示性别、特质共情、阅读习惯对共情的调节；本文进一步将读者特征与叙事风格交互纳入 SEM，弥补以往“单一特征变化”设计的不足。

## 局限性与未来方向
- **低一致性特征标注困难**：bodily sensations、evaluations、temporal references 等维度人机一致度偏低，Likert 全篇打分可能掩盖局部频次变化，未来可尝试细粒度定位或频次计数。
- **样本与文化推广性**：Prolific 众包以白人为主，且每人仅评 1 条故事，跨文化、跨语言及多故事重复测量情境下的结论仍需验证。
- **统计模型 vs 预测性能**：研究优先追求可解释路径而非预测 SOTA，未来可结合 Transformer 做特征消融或端到端个性化共情预测。
- **伦理双刃剑**：共情叙事建模可能被用于说服、营销或情感操纵，作者呼吁社区聚焦社会公益导向（如辅助用户更共情地表达、规模化理解线上叙事驱动的共情迁移）。

## 研究启发与可借鉴点
- **“理论 taxonomy → LLM 自动化标注 → 大规模人类行为验证”** 的三段式范式高度可复用，适合计算社会科学与 NLP 交叉团队快速落地机制型研究。
- 将 *narrative transportation* 作为中介变量串联风格特征与共情输出，符合社会科学因果推断规范；该思路可迁移至态度改变、健康传播、教育内容设计等下游任务。
- 混合效应模型 + SEM 的组合兼顾了故事级固定特征与读者级随机差异，为需要同时建模“内容特征”与“个体异质性”的推荐/反馈系统提供了方法论参考。
- 明确区分“共情预测性能”与“共情路径解释”，提示团队在构建下游应用时应避免过度追求端对端准确率而牺牲可解释性，尤其涉及敏感用户情感时。
- GPT-4 在乐观语调、情感生动性、人物脆弱性上显著优于 LIWC，但在 cognition/evaluations 上存在“情绪反应≠评价/认知过程”的系统性混淆，后续工作可在提示工程中引入反事实对照或思维链约束以缓解该偏差。

## 关键术语表
- **HEART (Human Empathy and Narrative Taxonomy)**：面向叙事共情的理论驱动风格分类体系，涵盖人物认同、情节、视角与环境四大类维度。
- **Narrative Empathy**：读者通过阅读个人故事而产生的对叙述者处境与情感的理解、共鸣与关怀反应。
- **Narrative Transportation（叙事运输）**：读者在阅读过程中沉浸于故事世界、脱离现实感知的心理状态，是连接叙事风格与 downstream 共情的重要中介变量。
- **State Empathy Scale**：测量阅读瞬间共情水平的 10 项自陈量表，强调情境性而非人格特质。
- **Trait Empathy（特质共情
