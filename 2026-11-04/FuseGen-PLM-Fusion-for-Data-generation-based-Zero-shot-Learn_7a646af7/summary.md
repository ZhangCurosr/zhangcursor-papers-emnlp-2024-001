---
title: "FuseGen-PLM-Fusion-for-Data-generation-based-Zero-shot-Learn"
source: https://aclanthology.org/2024.emnlp-main.130.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:08:16"
---

# 论文速读：FuseGen-PLM-Fusion-for-Data-generation-based-Zero-shot-Learn

## 一句话总结
本文提出 FuseGen，一种基于多预训练语言模型（PLM）协作的零样本数据生成框架，通过跨模型样本筛选构建迭代反馈提示，并结合自提升权重调整（SWA）动态重加权合成样本，有效缓解单PLM固有的分布偏差，显著提升小型任务模型（STM）的零样本训练性能。

## 研究问题与动机
- **核心问题**：基于数据生成的零样本学习（如ZeroGen）依赖单个PLM生成合成数据训练轻量级STM，但单PLM生成的合成数据集存在严重的分布偏差（如易学习样本过剩或难学习样本混杂），限制STM性能上限。
- **现有方法不足**：
  1. 单PLM设置无法突破内在分布偏差，即使ProGen引入单模型内部上下文反馈，也仅能轻微改善易学/模糊样本比例，部分情况下反而增加难学习样本。
  2. 简单混合多个PLM生成的数据（不加智能筛选）仍不及最优单PLM效果，说明粗暴堆叠无法发挥多模型多样性优势。
  3. 现有质量优化方法（如SunGen）依赖昂贵的双级SGD优化进行样本重加权，计算开销大，难以大规模应用。
- **动机**：利用多个PLM的多样性互补特性，设计高效的跨模型评估与反馈机制，在无需额外查询或微调PLM参数的前提下，系统性提升合成数据质量与STM泛化能力。

## 核心贡献（创新点）
1. 提出FuseGen多PLM协作的零样本数据生成框架，在不增加PLM查询次数、不访问/微调PLM参数的前提下协同生成高质量合成数据；与ZeroGen/ProGen/SunGen等单模型工作的本质区别在于以跨模型多样性主动抵消分布偏差，而非在单一模型内部做局部优化。
2. 设计跨模型数据集生成（CDG）中的两级样本筛选标准：先基于跨模型预测变异性（cross-model variability）粗筛候选集，再用噪声鲁棒的影响函数（influence function）精筛Top-S样本作为上下文反馈；与已有多样性增强或冗余消除方法的本质区别在于以“易学+模糊”组合为目标，构建跨PLM反馈闭环。
3. 提出自提升权重调整（SWA）策略用于STM训练阶段，受TrAdaBoost启发动态降低低质量样本权重；与SunGen双级SGD优化的本质区别在于无需昂贵梯度反向传播，计算成本降低两个数量级且性能持平。
4. 在9个NLI/NLU/NLG任务与8个数据集上进行系统实验，验证FuseGen的PLM无关性（PLM-agnostic），平均较最优单PLM基线提升约1.2%，并在未见任务（MarkedNews）与闭源模型（GPT-3.5/GPT-4）上均保持稳定优势。

## 方法详解
- **整体架构**：FuseGen包含跨模型数据集生成（CDG）与跨模型数据质量提升（CDI）两阶段（Algorithm 1）。共$J$轮迭代，每轮各PLM并行生成$N/(J+1)$条样本并训练对应STM $m_k$，最终合并所有数据训练目标STM $\tilde{m}$。
- **跨模型变异性计算**：对每个合成样本$(x_{k,i}, y_{k,i})$，利用其他$K-1$个PLM训练的STM预测其标签概率$p_{k',k,i}[y_{k,i}]$，计算标准差$d_{k,i}=\text{STD}(\{p_{k',k,i}[y_{k,i}]\}_{k'=1}^K)$。低$d_{k,i}$对应易学样本，高$d_{k,i}$对应模糊样本。
- **两级样本筛选**：从全部候选中按$d_{k,i}$选取Top-$\alpha R$高变异性与Top-$(1-\alpha)R$低变异性样本构成候选集$\hat{D}$；再用全量数据训练的STM $\tilde{m}$ 结合ProGen的影响函数公式计算每个候选的训练价值，选取Top-$S$样本作为下一轮反馈。
- **跨PLM上下文学习**：将选出的$S$个样本拼接到原始任务提示$\tau(\cdot)$中形成few-shot prompt，输入各PLM继续生成下一批数据，实现知识共享与迭代优化。
- **自提升权重调整（SWA）**：为每个样本初始化权重$w_{k,i}=0.5$，在$E_1$轮调整中基于当前STM $\tilde{m}$ 的预测误差$err_{k,i}=1-p_{k,i}[y_{k,i}]$与正确标记$correct_{k,i}$更新：$w_{k,i}^{(e_1+1)} = w_{k,i}^{(e_1)} \beta^{-err_{k,i}(1-correct_{k,i})}$，其中$\beta=1/(1+\sqrt{2\
