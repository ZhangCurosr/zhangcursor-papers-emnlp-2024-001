---
title: "Eyes-Don-t-Lie-Subjective-Hate-Annotation-and-Detection-with"
source: https://aclanthology.org/2024.emnlp-main.11.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:46:26"
---

# 论文速读：Eyes Don’t Lie: Subjective Hate Annotation and Detection with Gaze

## 一句话总结
本文构建了首个包含连续眼动数据的仇恨言论主观标注数据集 GAZE4HATE，验证了注视时长、注视次数等眼动特征能稳健预测 annotator 的主观仇恨评分，并首次提出融合眼动信号的仇恨言论检测模型 MEANION，证明眼动行为可为文本 (L)LM 提供正交互补信息，从而提升主观仇恨检测性能。

## 研究问题与动机
- 仇恨言论的定义与标注高度主观，受话题领域、社会语境及 annotator 个人背景与偏见共同影响，现有 HSD 黑盒模型常依赖数据集表面统计模式，难以刻画并适应真实的人类主观变异。
- NLP 社区近年开始利用人眼动数据评估 Transformer 认知合理性，但针对仇恨言论阅读的眼动语料仍属空白，且尚未系统比较“连续 gaze 信号”与“显式 rationale 标注”在解释模型时的对齐程度。
- 现有可解释性工作多依赖人工勾选的 token-level rationale 评估模型注意力，但 plain reading 过程中的生理与认知眼动是否携带模型未捕获的语义/情绪线索，缺乏实证。
- 核心待解问题聚焦为三问：眼动特征能否作为主观仇恨评分的稳健预测因子？其与人类/模型 rationale 的相关性结构如何？能否有效增强现有 (L)LM 用于 HSD？

## 核心贡献（创新点）
- 发布 GAZE4HATE 数据集：包含 90 个严格控制的最小配对句、43 名德语母语者的连续眼动轨迹、主观仇恨/置信度评分及 token 级 rationale 标注，填补了仇恨言论认知眼动研究的资源空白。（与既有眼动数据集仅收录少数高层特征或纯任务标注的本质区别：首次在同一协议下同步采集阅读过程眼动、主观等级评分与显式理由标注。）
- 系统量化 13 种眼动特征与主观仇恨类别的统计关系，筛选出 6 个高区分度特征，并揭示瞳孔直径变化对情绪效价不敏感、仅对唤醒强度敏感的特性。（与以往仅用 Fixation/Dwell 等少数特征的研究不同：全面对比了注视行为与自主神经信号在主观仇恨分类中的独立预测力。）
- 首次在 HSD 任务上对齐多种模型解释方法与人类认知信号，证明梯度类 rationale（InputXGradient/Saliency）与人类注视及显式 rationale 对齐度最高，而低相关的 pupil 特征恰是提升分类性能的互补盲区。（与 Atanasova et al. 在情感分类上的工作不同：将 explainability 评估拓展至连续 gaze 维度，并揭示“低对齐即高互补”的现象。）
- 提出 MEANION：首个将 CLS embedding、token 级眼动序列与 rationale BoW 向量拼接后输入 MLP 的融合基线，证明眼动信号在不同规模 (L)LM 上均可稳定带来增益。（与单纯依赖 fine-tuning 或注入人口统计学特征的工作不同：直接从内在认知行为流中提取特征，无需额外外部属性。）

## 方法详解
- **实验设计与数据采集**：采用被试内设计，43 名德语母语者（32 女/10 男/1 非二元，均值 23.5 岁）在 SR Eyelink 1000 Plus 眼动仪前阅读 90 个句子（每句限时 20s）。每轮先记录自然阅读眼动，再进行三步主观标注：1-7 Likert 仇恨评分、1-5 置信度评分、点击构成 rationale 的词汇。
- **眼动特征工程与归一化**：采集 13 种特征（FIXATION-COUNT, DWELL-TIME, PUPIL-SIZE 系列, RUN-COUNT, REGRESSION 系列, FIRST-FIX-DURATION, FIRST-RUN-FIXATION-COUNT, SKIP 等）。为消除个体基线差异，所有特征按 participant 单独进行 min-max 缩放，缺失值按跳过置零、眨眼插值均值的策略补齐。
- **统计检验**：对 multiclass（hate/neutral/positive）与 binary（hate/no-hate）条件分别执行 Anova 与 Chi-square 检验；Tukey 事后检验进一步定位差异来源，确认 fixation/dwell 差异主要来自 hate vs neutral/positive，pupil 差异来自 neutral vs hate/positive。
- **模型 Rationale 提取与相关性分析**：在性能最优的 rott-hc 模型上使用 Captum 提取 InputXGradient、Saliency 与 Shapley value rationale，通过子词均值聚合对齐至词级别。以 Pearson’s r 计算各人类 gaze 特征、人类 rationale 与模型两类 rationale 的平均相关性。
- **MEANION 融合架构**：将 (L)LM 的 CLS token 嵌入与每个 token 对应的 6 个眼动特征序列（填充至最大长度 t=14）及 rationale 的 BoW 计数向量（N=248）拼接，输入 MLP 分类器。通过 sklearn 网格搜索优化 hidden_layer_sizes、activation、solver 与 alpha，最终输出 hate vs no-hate 二元预测。

## 实验与结果
- **数据集与基线**：GAZE4HATE 共 3870 条主观标注实例。基线选取 5 个 HuggingFace 德语/多语言 BERT 模型（deepset, ortiz, aluru, rott, ml6），其中 rott 因显式针对性别仇恨微调而表现最佳（macro F1 0.60），其在 German HateCheck 上进一步微调得 rott-hc（macro F1 0.68）。
- **眼动统计结果**：FIXATION-COUNT、DWELL-TIME、FIRST-RUN-FIXATION-COUNT 在 binary 条件下 F-score 最高（49.98 / 44.25 / 54.19）；瞳孔三参数（MAX/MIN/AVERAGE-FIX-PUPIL-SIZE）同样显著，证实两者对主观仇恨具有稳健区分力。
- **Rationale 对齐结果**：Human rationale 与 InputXGradient 相关性最高（r=0.335），其次为 Saliency；pupil 相关特征与模型 rationale 基本不相关（r ∈ [-0.1, 0.1]），表明模型未显式编码该情绪唤醒信号。
- **MEANION 性能**：逐次注入单眼动特征或全部
