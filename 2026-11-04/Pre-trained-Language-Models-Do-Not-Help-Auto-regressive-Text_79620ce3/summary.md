---
title: "Pre-trained-Language-Models-Do-Not-Help-Auto-regressive-Text"
source: https://aclanthology.org/2024.emnlp-main.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:49:50"
---

# 论文速读：Pre-trained-Language-Models-Do-Not-Help-Auto-regressive-Text

## 一句话总结
本文系统验证了将预训练语言模型（LLM）直接适配到自回归文生图任务的可行性，发现其在图像生成质量与训练损失上与从零随机初始化模型完全一致，无任何增益；同时揭示图文联合微调会导致LLM在仅需5B token的极短训练后发生灾难性的语言先验遗忘。

## 研究问题与动机
- 扩散模型（如Imagen、DALL-E 2）普遍依赖强预训练文本编码器（T5/CLIP），而自回归模型（如Parti、CM3leon）通常放弃使用预训练文本模块，两者生成质量已趋同，LLM能否直接驱动自回归文生图尚属空白。
- 直接将LLM适配多模态需扩展词嵌入层与输出层以兼容图像token，但未经充分研究的“粗暴扩词表+联合微调”可能引发未知的表征冲突与能力退化。
- 现有研究缺乏对图文混合训练中“图像/文本损失分解”与“语言遗忘机制”的系统性归因，难以解释为何预训练先验在视觉任务中失效。
- 探究离散图像tokenizer输出token与文本token在LLM隐空间中的语义对齐程度，为多模态预训练架构选型提供理论依据。

## 核心贡献（创新点）
- 提出了首个系统验证预训练LLM对自回归文生图有效性的实验框架，证明其在图像生成质量与训练损失上与随机初始化模型表现一致；与已有工作聚焦扩散模型或Encoder-Decoder架构不同，本文首次针对Decoder-only LLM的直驱适配给出了明确的负向结论。
- 揭示了图文联合微调导致的灾难性语言遗忘现象，证明仅需5B token的联合训练即可使LLM的世界知识与in-context学习完全退化；区别于以往多模态研究侧重性能提升，本文从负向视角量化了简单图文数据对强语言先验的破坏性影响。
- 通过无条件图像生成与参数冻结消融，证实图像token与文本token的最优权重空间存在根本性分歧；与CLIP等跨模态对齐工作假设语义可共享不同，本文表明在离散VQ-GAN空间下任何部分的预训练权重均难以直接迁移至图像建模任务。

## 方法详解
- **模型适配与初始化**：基于开源`open_lm-1b`（1B参数，1.6T token预训练），将embedding层与output层的词表维度扩展，纳入图像tokenizer的词表（vocab size 16,384）。新增权重分别采用随机初始化与对比对齐初始化，其余预训练参数保持冻结或全量微调。
- **序列构建与数据处理**：使用SBER-MoVQGAN将256×256图像转换为1,024个图像token；caption使用NeoX tokenizer处理，平均约30个文本token。加入`<doc>`, `<text>`, `</text>`, `<image>`, `</image>`, `<pad>`等6个特殊token，构建固定长度1,152的混合序列。
- **训练配置**：在HQITP-134M数据集上训练100B token，使用64块A100 80GB GPU，batch size 1M token，AdamW优化器，余弦学习率调度（peak 0.0003，2K warmup）。
- **损失拆解分析**：将标准交叉熵损失按token模态独立计算，绘制`loss_image_tokens`与`loss_text_tokens`两条独立曲线，以区分预训练带来的文本loss优势与图像loss的无效性。
- **对比对齐实验（Contrastive Alignment）**：采用bag-of-words方式将1024个图像token与对应文本token编码为嵌入向量，使用类CLIP对比损失：最大化匹配图文对的cosine相似度，最小化非匹配对相似度；仅更新图像嵌入，验证两类token在低维语义空间的直接对齐可行性。

## 实验与结果
- **训练损失**：全程100B token训练中，预训练模型与随机初始化模型的图像token perplexity曲线几乎完全重合；文本token loss虽在初始阶段预训练模型更低，但因图文token比例高达30:1，该优势迅速消失。
- **生成质量**：在MS-COCO上评估zero-shot FID，预训练模型得分为12.21，随机初始化模型为12.27，**提升幅度为0（无显著差异）**；略优于DALL-E 2报告值，但受限于仅100B token训练量。
- **语言能力退化**：微调仅5B token后，模型的世界知识（如相对论常识填空）与in-context翻译能力彻底崩溃（Table 1示例），证明极端图文比例会快速覆盖语言先验。
- **无条件图像生成**：仅用图像token训练时，预训练模型损失同样等于随机初始化模型；冻结任何子模块（仅调embedding/output、仅调前N层等）均导致loss上升，证明预训练权重与图像建模的最优权重不兼容。
- **对比对齐**：bag-of-words层面的图文对比损失快速 plateau，且对齐后的图像token查询到的最邻近文本token多为噪声/无语义内容，无法带来生成性能提升。

## 相关工作脉络
- **DALL-E 2 / Imagen (Diffusion)**：依赖强预训练文本编码器引导生成；本文指出自回归路线长期刻意回避此设计，填补了“LLM直驱自回归文生图”的系统验证空白，并证明该回避策略在Decoder-only设定下是合理的。
- **Parti / CM3leon (Auto-regressive)**：采用encoder-decoder或BERT初始化编码器，未充分测试现代LLM的适配潜力；本文证明即使使用经过大规模文本预训练的Decoder-only LLM，直接适配仍无效，明确了架构选择的关键边界。
- **VQ-VAE / VQ-GAN / MoVQGAN**：主流离散图像tokenizer；本文实验证实其输出token与文本语义空间割裂，提示未来多模态生成需向SEED、SPAE等内生语义对齐型tokenizer演进。
- **CLIP**：在跨模态检索上成功，但本文的token级对比对齐实验表明，仅
