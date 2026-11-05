---
title: "Predicate-Debiasing-in-Vision-Language-Models-Integration-fo"
source: https://aclanthology.org/2024.emnlp-main.97.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:49:29"
field: "视觉-语言建模与场景理解"
keywords: ["Scene Graph Generation", "Predicate Debiasing", "Vision-Language Models", "Logits Adjustment", "Zero-shot VLM", "Label Shift", "Visual Genome"]
innovations: ["提出LM Estimation估计不可达预训练谓词分布实现零样本VLM去偏", "设计置信度感知动态集成策略融合预训练VLM与SGG模型", "免训练的即插即用方法显著提升欠表征三元组预测性能"]
benchmarks: ["Visual Genome", "Predicate Classification", "Scene Graph Classification"]
---

# 论文速读：Predicate-Debiasing-in-Vision-Language-Models-Integration-fo

## 一句话总结
本文提出在场景图生成（SGG）任务中整合预训练视觉语言模型（VLMs），通过创新的拉格朗日乘子估计（LM Estimation）近似不可达的预训练谓词分布，并设计置信度感知动态集成策略，实现免训练的去偏与表示增强，显著提升了SGG性能。

## 研究问题与动机
- **欠表征问题严重**：SGG三元组（subject-object-predicate）具有指数级组合多样性，训练集难以覆盖，导致测试集部分标签在训练中完全未见，预测质量差。
- **直接零样本VLM推理存在严重谓词偏差**：预训练语言数据集中谓词分布高度不平衡，导致VLM偏向常见谓词，与SGG任务分布错位。
- **现有去偏方法依赖显式训练分布，不适用于预训练VLM**：预训练数据通常保密，且预训练目标与SGG无直接标签对应关系，已有方法不可用。
- **现有研究主要聚焦架构优化，忽视预训练知识利用**：已有SGG去偏方法多在训练阶段干预，无法迁移至已训练的预训练VLM。

## 核心贡献（创新点）
1. **首个利用预训练VLMs缓解SGG欠表征问题的免训练方法**：与仅优化架构的工作不同，本文从知识增强角度解决欠表征问题。
2. **提出LM Estimation估计不可达预训练谓词分布**：通过约束优化从SGG验证集推断预训练语言分布π_pt，本质区别于依赖显式分布的现有去偏方法。
3. **设计置信度感知动态集成策略**：基于certainty-aware indicator为每个样本动态分配两模型权重，而非固定融合。
4. **免训练的即插即用框架**：无需额外训练，仅需前向推理，最小化计算与内存开销。
5. **在两种评估目标（Recall与mean Recall）下均达到最优**：可无缝适配不同目标分布，无额外成本。

## 方法详解
- **双分支框架**：固定零样本VLM $f_{zs}$（使用MLM提示"{z_i} is [MASK] {z_j}"）和可微调SGG模型$f_{sg}$（使用VQA提示"what is the relationship between the {z_i} and the {z_j}?"）。背景类仅由$f_{sg}$预测。
- **谓词偏差形式化**：基于标签漂移（label shift）假设，训练分布$P_{tr}(r)$与测试分布$P_{ta}(r)$不一致导致偏差，通过贝叶斯推导得到logits调整公式：$\hat{\mathbf{o}}^k(r) = \mathbf{o}^k(r) - \log P_{tr}(r) + \log P_{ta}(r)$。
- **LM Estimation**：以SGG训练/验证集样本为约束，求解最优$\pi_{pt}$使调整后logits的交叉熵最小：$\pi_{pt} = \arg\min R_{ce}(\mathbf{o}^k - \log\pi_{pt} + \log\pi_{sg}, r)$，约束$\sum\pi_{pt}(r)=1, \pi_{pt}(r)\geq0$，用Lagrange乘子法求解。
- **τ-calibration**：对去偏后logits除以温度系数τ避免过置信。
- **置信度感知集成**：$\text{conf} = \max_{r} P(r|...), W_{cer} \propto \text{sigmoid}(\text{conf}_{sg} - \text{conf}_{zs})$，加权融合两个模型的预测概率。

## 实验与结果
- **数据集**：Visual Genome（VG），108,077张图，70%训练/30%测试，150个object类别、50个predicate类别。
- **基线模型**：fine-tuned ViLT、fine-tuned Oscar、PENET-Rwt。
- **主要结果（mean Recall, PredCls）**：ViLT-ft-la + Ours达到mR@100 = 46.5（+2.0 vs baseline）；Oscar-ft-la + Ours达到42.7（+1.4）；PENET + Ours达到42.3（+1.6）。均超越所有SOTA去偏方法。
- **主要结果（Recall, PredCls）**：PENET + Ours达到R@100 = 71.1（+1.0）；ViLT-ft + Ours达到69.8（+1.4）。
- **未见过三元组表现**：去偏后零样本Oscar在unseen triplet上mAcc=16.01/Acc=20.05，超过微调Oscar（13.85/18.01）。
- **稀有类别增益最大**：rare类别mR@100提升+4.13（ViLT）、+3.49（PENET），证实对欠表征样本改善显著。

## 相关工作脉络
1. **SGG架构优化方法**（Tang et al., 2019; Gu et al., 2019; Zheng et al., 2023）：聚焦网络设计，本文从知识增强角度补充。
2. **无偏SGG方法**（Tang et al., 2020; Dong et al., 2022; Li et al., 2021）：训练时干预去偏，无法用于预训练VLM。
3. **VLMs用于关系识别**（He et al., 2022; Zhang et al., 2023; Yu et al., 2023）：需要微调或prompt tuning，本文是完全免训练的零样本集成。
4. **Logits Adjustment**（Menon et al., 2020）：适用于已知训练分布场景，本文扩展至不可达的预训练分布估计。
5. **标签漂移校正**：本文首次将post-hoc logit adjustment应用于预训练VLM的谓词去偏。

## 局限性与未来方向
- **计算成本**：ensemble框架增加了零样本VLM的前向推理开销，尤其在高proposal数量场景（如SGDet）下。
- **依赖预训练质量**：最终性能高度依赖$f_{zs}$的预训练数据全面性。
- **成对推理效率**：当前采用pair-wise转发模式，每个object pair需完整前向传播，对SGDet任务效率较低。
- **未来方向**：探索高效VLM推理策略；将LM Estimation迁移至其他预训练语言模型的去偏任务；结合更高效的detector减少proposal数量。

## 研究启发与可借鉴点
1. **LM Estimation可用于其他预训练模型分布估计**：当预训练数据不可得时，通过约束优化估计隐式分布的方法具有通用价值。
2. **免训练集成策略可迁移至其他长尾/分布偏移任务**：置信度感知的动态权重分配机制可直接应用于多模型集成场景。
3. **零样本VLM用于SGG的潜力被低估**：去偏后零样本模型在未见过三元组上超越微调模型，启示应更重视预训练知识的合理利用。
4. **温度校准（τ-calibration）在集成中的重要性**：避免过置信对去偏后概率融合效果至关重要。
5. **可结合团队在prompt tuning或知识蒸馏方向**：将LM Estimation与高效微调策略结合可能进一步降低计算开销。

## 关键术语表
**Scene Graph Generation (SGG)**：从图像中生成包含对象及其关系的结构化场景图表示的视觉-语言任务。
**Predicate Debiasing**：消除模型因训练数据中谓词分布不平衡而产生的偏向常见谓词的预测偏差。
**LM Estimation (Lagrange-Multiplier Estimation)**：通过约束优化从SGG数据推断预训练语言模型中不可达谓词分布的方法。
**Post-hoc Logits Adjustment**：基于标签漂移假设，在推理阶段对logits进行离线调整以校正分布偏差的技术。
**Certainty-aware Ensemble**：基于样本在两模型中置信度差异动态分配集成权重的策略。
**Label Shift**：训练分布与测试分布中类别先验概率不同的分布偏移情形。
**Visual Genome (VG)**：大规模场景图数据集，包含108K图像及密集的object和relation标注。
**ViLT / Oscar**：两种代表性预训练视觉语言模型，分别采用without-convolution和object-aligned预训练策略。

## 可复现要素
- **数据集**：Visual Genome（公开可用），采用70/30划分，150 object + 50 predicate类别。
- **代码/权重**：论文未明确提及开源声明；ViLT和Oscar为开源预训练模型（需自行下载权重）。
- **关键超参**：τ-calibration温度系数（论文未给出具体值）；ViLT冻结backbone训练MLP头50 epochs；Oscar端到端fine-tune 70k steps。
- **评估协议**：PredCls和SGCls子任务，指标为Recall@K和mRecall@K。
