---
title: "EFUF-Efficient-Fine-Grained-Unlearning-Framework-for-Mitigat"
source: https://aclanthology.org/2024.emnlp-main.67.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:17:01"
field: "多模态大语言模型可靠性"
keywords: ["多模态大语言模型", "幻觉缓解", "遗忘学习", "细粒度对齐", "CLIP", "梯度上升"]
innovations: ["首次将细粒度遗忘学习引入多模态幻觉缓解", "利用CLIP相似度自动筛选无标注样本构建微调数据集", "设计正负损失与句子损失协同的三损失遗忘框架"]
benchmarks: ["CHAIR", "MHumanEval", "POPE", "BLEU", "MME", "GQA", "ScienceQA"]
---

# 论文速读：EFUF-Efficient-Fine-Grained-Unlearning-Framework-for-Mitigat

## 一句话总结
本文提出了一种高效细粒度遗忘框架（EFUF），通过CLIP模型评估文本-图像相似度来自动筛选幻觉样本，并利用梯度上升策略在无需人工标注配对数据的情况下，有效减轻多模态大语言模型（MLLMs）中的对象幻觉问题。

## 研究问题与动机
- **核心问题**：多模态大语言模型（MLLMs）存在严重的对象幻觉现象，即生成描述中包含图像中不存在的对象（如图1中生成"landing gear"但图中并无此物）。
- **现有方法不足**：
  - 推理阶段方法需要额外计算步骤，增加延迟和成本；
  - 微调方法依赖大量人工标注的配对数据（幻觉响应vs非幻觉响应），标注成本高且耗时；
  - 偏好对齐算法（如RLHF、DPO）需同时运行多个模型，计算资源消耗大。

## 核心贡献（创新点）
1. **首次将遗忘学习引入多模态幻觉缓解**：从全新视角利用梯度上升策略消除幻觉，区别于传统基于配对数据的偏好对齐方法。
2. **提出高效细粒度遗忘框架EFUF**：利用CLIP相似度分数自动筛选正负样本，无需昂贵人工标注即可构建微调数据集。
3. **设计三种定制损失函数**：负样本损失（梯度上升消除幻觉）、正样本损失（强化正确对象）、句子损失（保持文本连贯性），实现细粒度遗忘。
4. **实验验证广泛适用性**：在MiniGPT4、LLaVA、mPLUG-owl、ShareGPT4V四种MLLM上均显著降低幻觉率，同时保持生成质量。
5. **计算效率显著提升**：仅需3 A100 GPU小时即可完成微调，远低于RLHF（20小时）、DPO（8-16小时）等方法。

## 方法详解
### 整体流程
EFUF分为两个阶段：数据集构建 → 遗忘训练。

### 1. 预实验验证CLIP判别能力
- 对MiniGPT4和LLaVA生成的200条图像描述进行人工标注，区分幻觉对象和非幻觉对象。
- 计算每个对象与对应图像片段的细粒度CLIP相似度（滑动窗口取最大值）：
  $$S(o_i^j) = \max_{w_i \in W_i} \text{CLIP}(o_i^j, w_i)$$
- 统计结果显示：幻觉对象的CLIP分数均值显著低于非幻觉对象（p值极小），证明可用相似度阈值区分正负样本。

### 2. 数据集构建（无手动标注）
- **正子句数据集** $D^+$：CLIP分数 $S(o) > T_0$（阈值设为32）的对象所在子句
- **负子句数据集** $D^-$：CLIP分数 $S(o) < T_1$（阈值设为23）的对象所在子句
- **完整句子数据集** $D^s$：句子级平均CLIP分数 $S(t_i) > T_2$（阈值设为27.5）的完整响应
- 使用LLaMA-2-70b提取对象集合，按标点分隔定位子句

### 3. 三种损失函数
- **负损失**（梯度上升消除幻觉）：
  $$L_{neg} = -L_{ft}(v, x, y), \quad (v, x, y) \sim D^-$$
- **正损失**（强化正确对象）：
  $$L_{pos} = L_{ft}(v, x, y), \quad (v, x, y) \sim D^+$$
- **句子损失**（保持文本连贯性）：
  $$L_{sent} = L_{ft}(v, x, y), \quad (v, x, y) \sim D^s$$
- **总体损失**：
  $$L = L_{pos} + \lambda_1 L_{neg} + \lambda_2 L_{sent}$$
  其中 $\lambda_1=0.3$, $\lambda_2=0.2$

### 4. 训练细节
- 仅微调多模态映射层（image feature → text token embedding）
- 1 epoch, AdamW优化器, learning rate $1e^{-5}$, weight decay 0.05

## 实验与结果
### 数据集与基线
- **数据集**：MSCOCO（仅用图像，不用标注）
- **基线模型**：MiniGPT4, LLaVA, mPLUG-owl, ShareGPT4V
- **对比方法**：RLHF-V, HADPO, POVID

### 评估指标
- **幻觉率**：CHAIR_S, CHAIR_I, MHumanEval_S, MHumanEval_I, POPE
- **生成质量**：BLEU-1/2/4, Informativeness (GPT-4), Perplexity (GPT-2)

### 主要结果
| 模型 | CHAIR_S↓ | CHAIR_I↓ | Human_S↓ | Human_I↓ | POPE↑ |
|------|----------|----------|----------|----------|-------|
| MiniGPT4→+EFUF | 45.9→38.9 | 23.2→21.1 | 69.0→45.0 | 27.3→12.7 | 81.0→82.3 |
| LLaVA→+EFUF | 52.8→41.9 | 22.8→18.7 | 42.0→24.0 | 14.7→7.7 | 85.3→85.9 |
| mPLUG-owl→+EFUF | 71.1→40.5 | 33.5→23.2 | 60.0→46.0 | 24.1→17.7 | 88.5→90.7 |
| ShareGPT4V→+EFUF | 46.8→36.9 | 22.3→18.4 | 31.0→14.0 | 9.9→5.4 | 87.8→88.1 |

- **平均提升**：CHAIR_S平均下降约15%，CHAIR_I平均下降约5%
- **生成质量**：BLEU-1提升4%，BLEU-2提升3%，BLEU-4提升2%

### 最强结果
mPLUG-owl经EFUF处理后，CHAIR_S从71.1降至40.5（降幅43%），达到最优幻觉抑制效果。

## 相关工作脉络
1. **RLHF-V (Yu et al., 2023)**：通过人工校正构建配对数据进行偏好对齐，需昂贵标注；EFUF无需人工标注，仅用CLIP分数筛选。
2. **HADPO (Zhao et al., 2023)**：基于DPO的幻觉感知偏好优化，需构造偏好对；EFUF仅用单样本梯度上升，避免配对数据需求。
3. **POVID (Zhou et al., 2024)**：通过偏好微调对齐模态，计算成本高；EFUF训练仅需3 GPU小时，远低于其所需资源。
4. **VCD (Sicong Leng, 2023)**：推理阶段视觉对比解码；EFUF为微调方法，无额外推理开销。
5. **Unlearning for LLM (Yao et al., 2023)**：首次在LLM上应用遗忘学习；本文扩展至多模态场景，并创新性地聚焦细粒度对象级遗忘而非全文本遗忘。

## 局限性与未来方向
- **局限性**：
  - 仅针对对象级幻觉，未覆盖属性幻觉、位置幻觉等类型
  - CLIP相似度作为判别依据可能不够精确，存在误判风险
- **未来方向**：
  - 探索更可靠的文本-图像相似度评估方法
  - 扩展至属性、空间关系等其他幻觉类型

## 研究启发与可借鉴点
1. **CLIP分数用于自动样本筛选**：利用预训练视觉语言模型的相似度分数替代人工标注，可作为低成本数据构建的通用范式。
2. **细粒度遗忘策略**：仅对包含幻觉对象的子句进行梯度上升，而非整句遗忘，保护模型其他能力——可迁移至其他需局部修正的场景。
3. **多损失协同设计**：正损失+负损失+约束损失的三角平衡机制，在消除错误行为同时保持基础能力，适用于多种对齐任务。
4. **仅微调映射层**：冻结LLM主体，仅调整多模态映射层，大幅降低训练成本和灾难性遗忘风险。

## 关键术语表
- **Object Hallucination**：多模态大模型生成描述中包含图像中不存在的对象的现象
- **Fine-grained Unlearning**：针对特定对象或子句的梯度上升遗忘策略，而非整句遗忘
- **CLIP Image-Relevance Score**：基于CLIP模型计算的文本对象与图像片段的细粒度相似度分数
- **Negative Loss**：对低相似度对象子句施加的负损失（梯度上升），用于消除幻觉
- **Sentence Loss**：对高相似度完整句子施加的正损失，用于保持文本连贯性
- **CHAIR (Caption Hallucination Assessment with Image Relevance)**：通过计算生成描述中不存在对象的比例来评估幻觉的指标
- **MLLM (Multimodal Large Language Model)**：能够理解和生成文本与图像的多模态大语言模型

## 可复现要素
- **数据集**：MSCOCO（公开），代码和数据集已开源：https://github.com/starreeze/efuf
- **训练硬件**：NVIDIA A100 GPU
- **关键超参**：
  - 阈值：$T_0 = 32$, $T_1 = 23$, $T_2 = 27.5$
  - 权重：$\lambda_1 = 0.3$, $\lambda_2 = 0.2$
  - Learning rate: $1e^{-5}$, weight decay: 0.05
  - 训练轮次：1 epoch
  - 优化器：AdamW
- **微调范围**：仅多模态映射层（image feature → text token embedding）
