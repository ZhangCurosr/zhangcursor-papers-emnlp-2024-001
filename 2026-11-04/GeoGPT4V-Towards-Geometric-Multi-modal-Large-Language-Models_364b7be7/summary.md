---
title: "GeoGPT4V-Towards-Geometric-Multi-modal-Large-Language-Models"
source: https://aclanthology.org/2024.emnlp-main.44.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:09:27"
---

# 论文速读：GeoGPT4V-Towards-Geometric-Multi-modal-Large-Language-Models

## 一句话总结
本文提出了一套自动化生成几何图文数据的新流水线：利用 GPT-4V 简化复杂几何题目，再通过 GPT-4 生成可执行的 Wolfram 代码绘制对应图像，最后由 GPT-4V 进行图文对齐评分与筛选，构建了 4.9K 高质量合成数据并与 19K 开源数据混合成 GeoGPT4V 数据集，显著提升了多种多模态大模型的几何推理能力。

## 研究问题与动机
- 即使最先进的 MLLMs 在处理需视觉辅助的几何问题时，性能仍显著落后于人类水平。
- 现有开源几何数据集难度偏高，模型缺乏基础几何概念支撑，难以直接高效学习。
- 现有数据增强方法（如 G-LLaVA）仅通过 ChatGPT-3.5 修改文本数值，未同步更新图像，导致严重的图文错位，干扰模型训练。
- 缺乏既保证题目难度适宜、又确保文本与几何图像严格对齐的大规模高质量数据。

## 核心贡献（创新点）
- **首创几何图文对齐的自动化数据生成流水线**：首次将题目简化、代码绘图与视觉评分筛选串联，系统性解决几何数据中文图错位问题。
- **引入 Wolfram 代码生成与执行机制**：利用 GPT-4 将自然语言几何题转化为可执行脚本并渲染精准图像，避免人工绘图的昂贵成本。
- **提出基于 CoT 评分的图文一致性筛选策略**：让 GPT-4V 对 K 个候选图像输出打分理由并选取高分（>0.9）样本，以低成本保障数据集质量。
- **开源 GeoGPT4V 数据集与多模型微调权重**：公开 4.9K 合成几何数据及混合后数据集，同时开源 LLaVA、ShareGPT4V、InternVL 的 checkpoint。
- **验证生成数据对多架构 MLLM 的普适增益**：在 MathVista 与 MathVision 上，7B 至 40B 的多种主流开源模型几何能力均显著提升。

## 方法详解
- **问答对生成与简化**：输入开源数据集 $D = \{Q, A, I\}$，指令 GPT-4V 从三个视角（生成铺垫问题、生成子问题、将答案结论反哺为题目条件）生成简化版 $\tilde{Q}, \tilde{A}$ 及图像描述 $\tilde{Des}$。为平衡效率与多样性，每个原始样本生成 N=3 个新样本，得到 $\tilde{D}_1 = \{\tilde{Q}, \tilde{A}, \tilde{Des}\}$。
- **几何图像生成**：将 $\tilde{Q}, \tilde{A}, \tilde{Des}$ 输入 GPT-4 生成 Wolfram 代码，要求显式命名所有变量以提升代码可读性与逻辑清晰度。由于代码生成存在不稳定性，对同一样本生成 K=3 段不同代码并执行，得到 K 个候选图像 $\tilde{I}^{(i)}$，构成 $\tilde{D}_2$。
- **评分与过滤**：采用 GPT-4V + Chain-of-Thought 提示，让模型先输出评估理由再给出 0~1 分的图文对齐得分。选取每道题得分最高的图像 $\tilde{I}$，若最高分 < 0.9 则丢弃该样本，最终得到 $\tilde{D} = \{\tilde{Q}, \tilde{A}, \tilde{I}\}$。
- **数据集构建**：将生成的 4.9K 样本与 ChartQA、UniGEO-Calculation、Geometry3K、GeoQA+ 等共 19K 开源数据混合，形成难度分层的 GeoGPT4V。
- **训练配置**：严格遵循各模型原始作者的超参数，Epoch=1，Global Batch Size=128，学习率 $1e^{-5} \sim 2e^{-5}$，cosine decay，冻结视觉编码器，仅微调 MLP 适配器与 LLM。

## 实验与结果
- **基准与设置**：在 MathVista testmini 与 MathVision test 的几何相关子任务上评估；基线模型为 LLaVA-1.5 (7B/13B)、ShareGPT4V (7B/13B)、InternVL-Chat-V1.2-Plus (40B)。
- **主实验结果**：GeoGPT4V 普遍提升各模型几何能力。LLaVA-1.5-7B 在 MathVista GPS minitest 上从 20.67 提升至 32.69（相对提升 58.2%）；ShareGPT4V-7B 从 21.63 提升至 32.69（相对提升 33.8%）。InternVL-Chat-V1.2-Plus-G 在 MathVista GPS/GEO 及 MathVision 多项任务上均达开源模型最高分（MathVista GPS 64.42，MathVision 几何平均分 17.84）。
- **消融实验**：① 用原图替换生成图训练，性能下降，验证新生成图像的必要性；② 随机选图 vs GPT-4V 评分选图，评分筛选显著更优；③ 纯开源 Base 数据集 vs 替换/混合生成数据（Replace/Mix），生成数据带来额外增益，证明性能提升并非仅源于开源数据。
- **数据质量分析**：GPT-4V 评估显示超 80% 生成题目难度≤原题；人工评估（n=200，互评一致率 0.74~0.78）同样确认生成图像与简化题目的图文匹配度显著优于原图匹配度（生成数据得分 0.9636，原图替换仅 0.7276，G-LLaVA 仅 0.6754）。

## 相关工作脉络
- **LLaVA / ShareGPT4V / InternVL**：主流开源 MLLM 基线；本文在其上验证 GeoGPT4V 的通用性，证明该数据对多种架构均有显著提升。
- **G-LLaVA (Gao et al., 2023)**：早期利用 ChatGPT-3.5 改写几何题目文本的方法，仅做数值替换未同步更新图像，导致图文错位；本文明确针对此缺陷提出代码绘图+视觉评分的对齐方案。
- **UniGeo / GeoQA / Geometry3K**：经典几何问答与证明数据集；本文以其训练集为原始素材进行难度简化与图像重生成，弥补了原数据难度过高且图文难以严格同步的问题。
- **MathVista / MathVision**：当前多模态数学推理主流评测基准；本文在此两个基准的几何子任务上全面对标，确立了评估坐标系。
- **WizardMath / Meta-Math**：纯文本数学推理的数据增强工作；本文将其思路延伸至多模态几何场景，强调视觉信息
