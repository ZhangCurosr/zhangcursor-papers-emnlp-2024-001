---
title: "Fine-Grained-Prediction-of-Reading-Comprehension-from-Eye-Mo"
source: https://aclanthology.org/2024.emnlp-main.198.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:46:04"
field: "眼动与阅读理解预测"
keywords: ["eye-movement prediction", "reading comprehension", "multimodal transformer", "item-level assessment", "OneStop dataset"]
innovations: ["提出RoBERTa-QEye/MAG-QEye/PostFusion-QEye三种融合眼动与文本的Transformer模型用于item-level阅读理解预测", "建立New Participant/New Item/New Both三类严格泛化评估协议并引入强文本基线", "系统比较普通阅读与信息搜索两种情境下的眼动预测表现"]
benchmarks: ["OneStop Eye Movements", "Text-only RoBERTa baseline", "SB-SAT"]
---

# 论文速读：Fine-Grained-Prediction-of-Reading-Comprehension-from-Eye-Mo

## 一句话总结
本文系统评估了从眼动数据预测单个读者对单个阅读理解题回答正确性的可行性，提出了三种融合文本与眼动的Transformer模型（RoBERTa-QEye、MAG-QEye、PostFusion-QEye），发现在最严格的跨参与者+跨项目泛化设置下，眼动信号对阅读理解预测的提升幅度有限，强文本基线（Text-only RoBERTa）具有重要参考价值。

## 研究问题与动机
- **核心问题**：能否从读者在单个段落上的眼动轨迹，预测其对单个阅读理解题的回答是否正确（item-level预测）？
- **现有方法的不足**：
  1. 数据规模小：此前研究常用数据集仅含几十名参与者和十几道题（如SB-SAT仅20题/95参与者），难以充分训练NLP/ML模型。
  2. 预测粒度粗：已有工作多预测多个题目聚合后的综合得分，而非单个题目的理解程度。
  3. 阅读情境单一：多数研究仅关注普通阅读（ordinary reading），忽略了日常常见的信息搜索（information seeking）场景。
  4. 缺乏强文本基线对比：先前研究未与"仅用文本"的强基线比较，难以评估眼动信息的实际增量贡献。

## 核心贡献（创新点）
- **引入新任务与新数据**：提出"单读者-单题目-单段落"粒度的阅读理解预测任务，利用OneStop数据集（486题/19,440响应/360参与者）实现item-level预测。
- **设计三种多模态Transformer模型**：提出RoBERTa-QEye（将眼动特征嵌入作为额外输入）、MAG-QEye（用眼动信息修改Transformer中间层词表示）、PostFusion-QEye（分离处理文本与眼动后通过交叉注意力融合）。
- **系统评估阅读情境差异**：首次在普通阅读与信息搜索两种情境下分别训练/评估模型，揭示眼动信息在不同任务设置下的表现差异。
- **建立严格的泛化评估协议**：提出三类评估设置（新参与者、新项目、两者皆新），并指出先前的"test参与者或test项目已在训练集中出现"的评估方式过于宽松。
- **引入Text-only RoBERTa基线**：强调该基线能捕捉题目难度等文本统计信息，证明多数眼动模型在New Participant设置下难以超越此基线。

## 方法详解
- **眼动特征表示**：采用词级别（Words）和注视点级别（Fixations）两种粒度。词级别特征包括IA_DWELL_TIME、IA_FIXATION_COUNT、IA_SKIP等约20项；注视点级别特征包括CURRENT_FIX_DURATION、CURRENT_FIX_PUPIL、NEXT_SAC_AMPLITUDE等。同时融入语言学属性（surprisal、frequency、length、 syntactic properties等）。
- **RoBERTa-QEye**：将眼动特征通过全连接层投影到词嵌入空间，加入位置嵌入和眼动类型嵌入后与文本序列拼接，统一输入RoBERTa_LARGE编码器。
- **MAG-QEye**：借鉴多模态情感分析中的MAG架构，用眼动特征$H_{W_i}$对Transformer中间层的隐藏表示进行偏移：$\bar{Z}_{W_i}^k = Z_{W_i}^k + \alpha H_{W_i}$，其中$\alpha$控制偏移幅度上限。
- **PostFusion-QEye**：文本经语言模型获取上下文嵌入$Z_P$；眼动特征经两个1D卷积层得到$Z_{E_P}$；先以眼动为query、文本为key/value做交叉注意力，再通过全连接层投影到共享空间；再以问题嵌入$Z_Q$为query做第二次交叉注意力。
- **多分类变体**：针对具体选项预测任务，将四个选项作为输入。
- **评估指标**：使用平衡准确率（balanced accuracy）处理类别不均衡（正确答案约占81%）。
- **训练细节**：AdamW优化器，batch size=16，dropout∈{0.1, 0.3, 0.5}，学习率∈{1e-5, 3e-5, 1e-4}，10折交叉验证。

## 实验与结果
- **数据集**：OneStop Eye Movements（Eyelink 1000+，1000Hz采样率），360名英语母语成人参与者，162个段落，486道多选题，共19,440个响应（普通阅读与信息搜索各9,720个）。
- **基线模型**：Majority baseline、Text-only RoBERTa、Logistic Regression（Mézière et al., 2023b）、CNN（Ahn et al., 2020）、BEyeLSTM（Reich et al., 2022）、Eyettention（Deng et al., 2023）。
- **二元任务（正确vs错误）最强结果**：
  - Ordinary Reading：MAG-QEye（Words）在"New Participant"设置下达64.1%（vs Text-only RoBERTa的63.1%，$p<0.05$）；总体"A ll"达59.2%。
  - Information Seeking：PostFusion-QEye在"New Item"设置下达53.8%（$p<0.05$）；总体58.0%。
  - 在New Item & Participant严格设置下，所有模型与Text-only RoBERTa差距极小甚至更低。
- **多项选择任务最强结果**：RoBERTa-QEye-Fixations在Ordinary Reading的"New Item & Participant"设置下达37.9%（$p<0.001$）。
- **关键结论**：眼动信息带来的提升幅度较小（通常1-3个百分点），且仅在部分设置下显著；New Participant设置下的性能主要来自文本/题目统计信息而非眼动本身。

## 相关工作脉络
- **Mézière et al. (2023b)**：使用线性模型从眼动特征预测阅读理解分数；本文将其适配到item-level二元分类任务并与新模型对比。
- **Ahn et al. (2020)**：使用CNN预测SB-SAT数据上的低/高理解二分；本文指出其依赖手动分段的局限性，改用完整注视序列。
- **Reich et al. (2022)**：BEyeLSTM结合注视序列与文本特征；本文对其进行了语言特征扩展和更细粒度的tokenization适配。
- **Deng et al. (2023)**：Eyettention原用于scanpath预测；本文将其调整为全局交叉注意力用于理解预测。
- **Rahman et al. (2020)**：MAG多模态Transformer架构（原文用于情感分析）；本文首次将其适配到眼动-文本融合的理解预测。
- **Malmaud et al. (2020)**：OneStop数据集构建者；本文在此基础上进行系统性的预测建模评估。

## 局限性与未来方向
- **任务粒度限制**：当前研究仅涉及单个段落的理解，未涵盖需要跨段落整合信息的更长文本场景。
- **实验环境的人为性**：被试无法回看段落，与实际阅读习惯存在偏差；实验室环境也可能影响阅读模式。
- **模型性能距离实用尚有距离**：最大准确率约59%（binary），远未达到可部署的可靠水平。
- **设备依赖**：仅在Eyelink 1000+（1000Hz）高精度设备上验证，低分辨率设备（如消费级眼动仪）的可行性待探索。
- **人群局限性**：数据仅来自健康英语母语成人，儿童、L2学习者、认知障碍人群的泛化能力未知。
- **语言限制**：目前仅适用于有高质量眼动数据和预训练语言模型的英语，其他语言缺乏对应资源。

## 研究启发与可借鉴点
- **强文本基线的重要性**：Text-only RoBERTa能有效捕捉题目难度和项目统计信息，为任何多模态预测任务提供了必要参考基线，后续研究应沿用此做法。
- **三类泛化评估设置的系统性**：New Participant / New Item / 两者皆新的分层评估框架值得迁移到其他生物信号预测任务（如脑电识别、语音情感）。
- **眼动-语言特征联合建模思路**：将linguistic word properties（surprisal、frequency等）与眼动特征联合输入，使模型能学习"眼动-语言交互"模式，这一设计可推广到其他多模态融合场景。
- **MAG风格的中间层注入机制**：通过可学习门控机制在Transformer中间层注入外部模态信号，相比末段融合能更早影响语义表示，这一架构可探索于其他时序-文本融合任务。
- **信息搜索与普通阅读的对比分析**：将不同阅读策略（Gathering vs Hunting）分开建模，发现前者准确率更高，提示任务目标对眼动-理解关系的调节作用，值得在更多场景中探索。

## 关键术语表
- **OneStop Eye Movements**：目前最大的英语母语阅读眼动数据集，包含360名参与者、486道阅读理解题、19,440个响应，支持item-level预测研究。
- **STARC annotation framework**：阅读理解答案标注框架，将答案分为A（完全正确）、B（识别了相关但未正确理解的信息）、C（文本其他不相关部分）、D（无文本支持）四类，对应不同理解程度。
- **Balanced Accuracy**：平衡准确率，对各类别单独计算准确率后取平均，用于缓解类别不均衡问题。
- **Ordinary Reading (Gathering)**：普通阅读情境，被试先阅读段落再回答问题，无题目预览。
- **Information Seeking (Hunting)**：信息搜索情境，被试先看到问题（无选项）再阅读段落，模拟有明确目标的搜索式阅读。
- **Surprisal**：信息量度量，定义为$-log_2 P(word|context)$，衡量给定上下文预测某个词的不确定性，是心理学中重要的语言加工指标。
- **IA_DWELL_TIME**：兴趣区 dwell time，指在某个词/短语上所有注视时长的总和，是测量阅读加工难度的核心眼动指标。
- **Cross-Attention**：交叉注意力机制，允许一类序列（如眼动特征）查询另一类序列（如文本表示）的信息，是多模态融合的关键技术。

## 可复现要素
- **数据集**：OneStop Eye Movements，由Malmaud et al. (2020)收集，论文声明数据为匿名化IRB批准数据，可用于学术研究（来源见ACL Anthology）。
- **代码/权重**：论文未明确提及代码开源情况，模型基于PyTorch Lightning和HuggingFace Transformers实现。
- **关键超参**：AdamW优化器，batch size=16，warmup ratio=0.1，weight decay=0.1，dropout∈{0.1,0.3,0.5}，learning rate∈{1e-5, 3e-5, 1e-4}，MAG注入层索引∈{0,11,23}，最大训练10 epoch（早停patience=3）。
- **硬件**：NVIDIA A100-40GB / A40-48GB GPU。
