---
title: "Autoregressive-Pre-Training-on-Pixels-and-Texts"
source: https://aclanthology.org/2024.emnlp-main.182.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:13:48"
field: "多模态语言模型"
keywords: ["pixel-based language modeling", "autoregressive pre-training", "multimodal pre-training", "cross-lingual generalization", "vision-language integration"]
innovations: ["首次系统验证自回归像素预训练在语言理解任务上的可行性", "提出双模态配对预训练框架揭示像素-文本协同效应"]
benchmarks: ["GLUE", "XNLI", "HatemojiBuild"]
---

# 论文速读：Autoregressive-Pre-Training-on-Pixels-and-Texts

## 一句话总结
本文提出了 PixelGPT 和 DualGPT，探索了在 RGB 像素图像上直接进行自回归语言预训练的可行性，并系统研究了像素模态与文本模态联合预训练的协同效应，证明纯像素自回归模型在语言理解任务上可达到与双向 PIXEL 模型相当的性能。

## 研究问题与动机
- **词汇瓶颈问题**：传统 tokenization 方法在跨语言场景下受限于预定义词表，难以处理未见语言；像素表示天然摆脱词汇依赖。
- **自回归像素建模的空白**：现有像素语言模型多为编码器架构（如 PIXEL 的双向 MAE），解码器-centric 的自回归像素建模尚未充分探索。
- **视觉信息利用不充分**：现有工作多用 8-bit 灰度或 2-bit 二值图像，丢失了颜色信息（如 emoji、高亮文本）；RGB 原始像素蕴含更丰富的正字法特征。
- **多模态交互机制不明**：像素与文本两种模态联合预训练时，是独立训练还是配对训练，其协同效应尚不清楚。

## 核心贡献（创新点）
- **首次系统验证自回归像素预训练的可行性**：PixelGPT 仅在 RGB 像素上预训练，在 GLUE 基准上达到与双向 PIXEL 模型相当甚至更优的性能，证明 causal decoder-only 架构同样适用于像素语言建模。
- **提出双模态联合预训练框架 DualGPT**：将像素（next patch prediction）与文本（next token prediction）预训练统一在同一架构中，通过配对图像-文本数据有效缓解模态冲突，实现语言理解性能提升。
- **揭示像素模型的跨语言优势**：PixelGPT 在 XNLI 上超越 BERT 和 PIXEL，尤其在泰语（+11.3）、中文（+4.3）等与英语差异较大的语言上表现突出，验证了无 tokenization 设计对多语言泛化的帮助。
- **开源训练数据与模型检查点**：发布用于语言理解和多语言评估的微调数据集及模型权重，推动该新兴方向的发展。

## 方法详解
- **文本渲染为图像**：使用文本渲染器将文本转换为 RGB 图像（H=16, W=16384, C=3），划分为 1024 个 16×16 像素的 patch；黑色 patch 作为 EOS 标记，不参与注意力与损失计算。
- **输入表示**：
  - 图像输入：借鉴 ViT 思路，将图像划分为 patch 序列后通过线性投影映射到 D 维空间，按文本顺序排列送入 transformer decoder。
  - 文本输入：使用 Llama 2 的 tokenizer，词表大小 32k。
- **预训练目标**：
  - Next Patch Prediction：对像素输入使用归一化 MSE 损失，逐 patch 预测下一个 patch 的像素值。
  - Next Token Prediction：对文本输入使用标准交叉熵损失，teacher-forcing 预测下一个 token。
- **模型配置**：基于 Llama 2 的 24 层 transformer decoder，使用 RMSNorm、SwiGLU、RoPE、GQA；共四种变体：TextGPT（纯文本）、PixelGPT（纯像素）、MonoGPT（独立双流）、DualGPT（配对双流）。
- **配对训练策略**：DualGPT 将图像序列拼接在文本序列前，联合送入模型，实现模态间交互。

## 实验与结果
- **数据集**：GLUE（NLU）、XNLI（跨语言 NLI，15 种语言）、HatemojiBuild（emoji 仇恨检测）。
- **预训练数据**：peS2o、English Wikipedia、C4（像素）；Dolma、Common Crawl、The Stack v1、Project Gutenberg、Wikipedia（文本）。
- **GLUE 结果**：PixelGPT（317M 参数）平均 74.2，优于 PIXEL（74.1）和 CLIPPO；在 STS-B（+5.4）、MRPC（+13.1）、RTE（+11.9）、WNLI（+5.4）上显著超越 GPT-2。
- **XNLI 结果**：PixelGPT 平均 63.2，超越 PIXEL（62.8）和 BERT（63.9 的 translate-train-all 设置下持平）；泰语 +11.3、中文 +4.3。
- **双模态消融**：DualGPT（76.9）> TextGPT（76.3）> MonoGPT（75.4），配对数据带来明显增益。
- **Scaling 趋势**：PixelGPT 在约 200B tokens/patches 时超越 PIXEL，并仍在上升，有潜力超越文本模型。

## 相关工作脉络
- **PIXEL（Rust et al., 2023）**：双向编码器架构，灰度图像，MSE 重构；本文是其自回归解码器版本，改用 RGB 并探索下游理解任务。
- **CLIPPO（Tschannen et al., 2023）**：统一 encoder 处理图像与文本的对比学习；本文用 causal decoder 分别建模两种模态，强调生成式预训练。
- **PIXAR（Tai et al., 2024）**：二值图像 + 二元交叉熵；本文在连续像素空间使用 MSE，保留更细粒度信息。
- **DONUT（Kim et al., 2022）**：OCR-free 文档理解模型，聚焦结构化信息抽取；本文关注通用语言理解与跨语言泛化。
- **VQ-VAE/VQGAN + next token**：将连续图像离散化为 token 再预测；本文直接预测原始像素 patch，避免量化信息损失。
- **Saurel et al. (2024)**：大规模自回归图像模型；本文聚焦于视觉文本而非自然图像，探索语言建模的像素化路径。

## 局限性与未来方向
- **模型规模有限**：当前仅 317M 参数，未扩展到 7B/70B/100B 级别，Scaling 潜力待验证。
- **训练算力受限**：预训练仅 100–200B tokens/patches，无法充分挖掘数据 Scaling 效果。
- **生成任务受限**：输出为像素 patch，需额外 OCR 后处理才能得到文本，缺乏端到端生成能力。
- **初步探索性质**：作为像素自回归预训练的开创性工作，鲁棒性与泛化能力需在更多场景下验证。

## 研究启发与可借鉴点
- **无 tokenization 的多语言策略**：像素表示天然摆脱词表依赖，可迁移至低资源语言建模，避免 BPE/WordPiece 的OOV问题。
- **大 batch size 对像素微调的必要性**：像素模态方差更大，使用 512 等大 batch 可显著提升训练稳定性和性能，值得在类似任务中借鉴。
- **RGB 渲染保留颜色信息的价值**：在含 emoji/高亮的文本理解任务中，RGB 相比灰度/二值有明显增益，可推广至多模态情感分析等场景。
- **配对双模态预训练的协同效应**：单纯双流独立训练（MonoGPT）不如配对训练（DualGPT），提示多模态对齐对模态交互至关重要。
- **字体鲁棒性验证**：PixelGPT 在不同字体（包括符号字体）下保持性能稳定，说明像素预训练具有强视觉泛化能力，可用于 OCR 增强场景。

## 关键术语表
- **PixelGPT**：仅在 RGB 视觉文本图像上进行自回归预训练的解码器模型，采用 next patch prediction 目标。
- **DualGPT**：联合像素与文本双模态预训练的模型，通过配对图像-文本数据实现模态交互。
- **Next Patch Prediction**：自回归像素预训练的目标函数，逐 patch 预测下一个视觉 patch 的像素值，使用归一化 MSE 损失。
- **Vocabulary Bottleneck**：传统 tokenization 方法因预定义词表导致低资源语言或新词无法表示的限制，像素方法可规避此问题。
- **Translate-Train-All**：XNLI 跨语言评估设置，使用 15 种语言的微调数据联合训练模型。
- **MonoGPT**：在独立流上分别用文本和像素数据预训练的模型，不使用配对数据。
- **RGB Rendering**：将文本渲染为 24-bit 三通道彩色图像（而非灰度或二值），保留颜色等视觉细节。
- **Paired Image-Text Data**：图像与其对应文本配对的数据，用于 DualGPT 的双模态联合预训练。

## 可复现要素
- **数据集**：peS2o、English Wikipedia、C4（像素预训练）；Dolma、Common Crawl、The Stack v1、Project Gutenberg、Wikipedia（文本预训练）；GLUE、XNLI、HatemojiBuild（下游评估）。论文声明开源微调数据集。
- **代码/权重**：已开源，见 https://github.com/ernie-research/pixelgpt。
- **关键超参**：patch size=16，max seq length=1024，hidden size=1024，num layers=24，vocab size=32k，batch size=4M（单模态）/8M（MonoGPT）/10M（DualGPT），lr=5e-4，warmup=200，AdamW，bfloat16。
