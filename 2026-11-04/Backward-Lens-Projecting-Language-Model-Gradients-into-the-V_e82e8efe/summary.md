---
title: "Backward-Lens-Projecting-Language-Model-Gradients-into-the-V"
source: https://aclanthology.org/2024.emnlp-main.142.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:14:05"
field: "语言模型可解释性"
keywords: ["可解释性", "语言模型", "梯度分析", "知识编辑", "Logit Lens", "反向传播"]
innovations: ["首次将 LM 梯度矩阵投影到词表空间，揭示低秩结构", "提出印记-偏移（imprint and shift）机制解释 MLP 知识存储", "提出仅需单次前向传播的 Forward Pass Shifting 编辑方法"]
benchmarks: ["CounterFact", "WikiText"]
---

# 论文速读：Backward-Lens-Projecting-Language-Model-Gradients-into-the-V

## 一句话总结
本文首次将语言模型（LM）的梯度矩阵投影到词表空间，通过分析反向传播中的梯度低秩特性，揭示了 Transformer MLP 层存储知识的"印记-偏移（imprint and shift）"机制，并据此提出仅需单次前向传播的高效知识编辑方法。

## 研究问题与动机
- 现有 LM 可解释性工作（如 Logit Lens）仅将前向传播的权重和隐状态投影到词表空间以观察信息流，**从未有人将梯度本身投影到词表空间**。
- LM 每层含数千个神经元，特征分布跨多个神经元，直接分析梯度矩阵维度过高（如 GPT2-medium 每 MLP 层 4096 个神经元），需找到降维/解释路径。
- 尽管已知梯度是低秩矩阵（Mitchell et al., 2021; LoRA 工作），但**利用该性质进行可解释性研究和预测编辑 prompt 长度对秩的影响**尚未探索。
- 理解 LM 如何在训练/编辑过程中将新知识存储于 MLP 层内、以及具体的更新机制尚不明确。

## 核心贡献（创新点）
1. **首次将 LM 梯度投影至词表空间**：证明梯度矩阵可由前向输入 $x_i$ 和反向 VJP $\delta_i$ 张成，利用低秩特性将分析从全矩阵降至 $n$ 个向量（$n$ = prompt 长度），本质区别在于前作聚焦静态权重或前向隐状态。
2. **提出"印记-偏移（imprint and shift）"机制**：发现 MLP 层的知识存储分为两阶段——$FF_1$ 通过加减前向输入 $x_i$ 实现"印记"，$FF_2$ 通过减去 VJP $\delta_i$ 实现"偏移"，揭示了反向传播中信息写入权重的具体力学机制。
3. **基于偏移机制提出单次前向传播编辑方法（Forward Pass Shifting）**：用目标 token 嵌入近似 VJP，仅需单次前向传播即可编辑模型内部知识，在编辑成功率上与 SOTA 方法 ROME 持平（99.4%），且运行复杂度大幅降低。

## 方法详解
- **梯度秩的理论证明（Lemma 4.1）**：对于长度为 $n$ 的 prompt，梯度矩阵 $\frac{\partial L}{\partial W} = \sum_{i=1}^{n} x_i^\top \cdot \delta_i$ 的秩不超过 $n$；当所有 $x_i$ 或 $\delta_i$ 线性无关时，秩恰好为 $n$。最后一层 MLP 梯度秩恒为 1（仅最后一个 token 贡献）。
- **梯度矩阵作为向量张成空间**：利用梯度矩阵的张量分解形式，将每个梯度矩阵解释为 $n$ 对向量 $(x_i, \delta_i)$ 的线性组合。对于 $FF_1$，选择前向输入 $x_i$ 作为生成集（因 $\delta_i$ 维度不是 $d$）；对于 $FF_2$，选择 VJP $\delta_i$ 作为生成集。
- **Logit Lens（LL）投影**：将生成集中的每个 $d$ 维向量通过解码矩阵 $D$ 投影到词表空间：$\mathrm{LL}(x) = \mathrm{Softmax}(ln_f(x) D)$。对 $FF_1$ 看最高概率 token，对 $FF_2$ 因梯度更新方向相反，看最低概率 token。
- **"印记-偏移"机制（Lemma 5.2）**：更新 $FF_1$ 时，前向输入 $x_i$ 被加/减到神经元权重中，放大后续激活；更新 $FF_2$ 时，VJP $\delta_i$ 从神经元权重中减去，将目标 token 嵌入注入输出。
- **前向传播偏移编辑方法**：跳过反向传播，直接取目标 token 嵌入 $D^\top[t]$ 作为 VJP 的近似，更新公式为 $\hat{FF}_2 \leftarrow \hat{FF}_2 + \eta \cdot x_n^\top \cdot D^\top[t]$，其中 $x_n$ 为最后一 token 的前向输入。

## 实验与结果
- **数据集**：CounterFact（100 条 prompt-target 对用于主要实验，1000 条用于编辑基准测试）。
- **模型**：GPT2（small/medium/xl）和 Llama2-7B。
- **梯度秩验证**：除最后一层外，所有层的梯度矩阵秩几乎总是恰好等于 prompt 长度 $n$（>98.5% 的情况），验证了 Lemma 4.1。
- **LL 投影可视化**：编辑"Obama grew up in → Paris"时，$FF_2$ 的 VJP 在大部分层中都投影出目标 token "Paris" 及其相关词（如 "Macron"）；编辑"Jack Dorsey founded → IBM"时，梯度一致投影出 "IBM"。
- **Prompt 各段影响分析**（GPT2-xl）：编辑更新主要集中在两个区域——初始层中主语 token、第二四分之一层中最后一个 prompt token；其余 token 的 VJP norm 接近零，对权重更新贡献极小。
- **编辑性能（CounterFact，GPT2-xl）**：

| 方法 | Efficacy ↑ | Paraphrase ↑ | Neighborhood ↑ | N-gram ↑ |
|------|-----------|-------------|---------------|---------|
| ROME | 99.4 | 71.9 | 10.91 | 622.78 |
| Forward Pass Shift | **99.4** | 41.6 | 6.02 | 622.45 |
| MEMIT | 79.4 | 40.7 | 10.98 | 627.18 |
| MEND | 71.4 | 17.6 | 7.73 | 623.94 |

- 最强结果：**Forward Pass Shifting 在 Efficacy 上与 ROME 持平（99.4%）**，且在 WikiText Perplexity 上几乎无退化（93.66 vs. 原始 93.56），显著优于 ROME（137.38）和 Fine-tuning（199.84）。

## 相关工作脉络
- **Logit Lens（nostalgebraist, 2020; Geva et al., 2022a）**：将前向隐状态投影到词表空间；本文将其扩展到反向传播的梯度，首次实现"梯度-词表"映射。
- **Saliency Maps（Simonyan et al., 2014）**：分析梯度矩阵与输入的关系；本文不同在于聚焦 VJP 而非整个梯度矩阵，并投影到 token 而非分析预测影响。
- **知识神经元（Dai et al., 2022）**：在编码器 LM 中识别激活神经元并注入目标嵌入；本文发现梯度编辑机制与其高度相似（均为"偏移"思想）。
- **模型编辑 SOTA（MEND, ROME, MEMIT）**：多步迭代或复杂优化；本文方法仅需单次前向传播，在 Efficacy 上持平 ROME，但通用性和特异性略逊。
- **LoRA（Hu et al., 2022）**：利用梯度低秩特性加速微调；本文则利用低秩进行可解释性分析，并提出秩可被 prompt 长度预测的新发现。
- **Tuned Lens（Belrose et al., 2023; Din et al., 2023）**：引入学习变换提升早期层 LL 可解释性；本文未采用，因关注核心机制发现，但指出可作为未来方向。

## 局限性与未来方向
- **Logit Lens 在早期层可解释性差**：第一层和最后几层的 LL 投影效果下降，尚未明确梯度是否在所有层共享同一嵌入空间。
- **忽略优化器缩放效应**：实验使用 SGD，未考虑 Adam 等优化器对梯度秩和权重的额外缩放影响。
- **仅探索单条 prompt 的单步编辑**：实际微调涉及多条 prompt 和多步迭代，低秩特性可能消失；但发现少数 VJP 主导了更新，可进一步探索。
- **未探索不同 prompt 编辑间梯度的相似性**：跨任务/跨编辑的子空间共享机制（与 superposition 相关）仍是开放问题。
- **仅关注 MLP 层，未涉及 Attention 层**：同时编辑两者的效果尚不清楚。
- **Forward Pass Shifting 在 Neighborhood 指标上表现较差**：可能因过度编辑了与其他 prompt 共享的激活模式。

## 研究启发与可借鉴点
1. **梯度分解为生成集的思路可迁移**：任何满足 $\frac{\partial L}{\partial W} = \sum x_i^\top \cdot \delta_i$ 的层均可类似分析，可用于 Vision Transformer、多模态模型等的梯度可解释性研究。
2. **"印记-偏移"机制启发了极简编辑设计**：无需反向传播即可近似梯度更新，为高效在线学习、持续学习中的参数高效修改提供新思路。
3. **Normalized Logit Lens（附录 H）**：对低 norm 向量归一化后再做 LL 投影，可显著提升早期层和低幅值 VJP 的可读性，可直接复用于其他 LL 相关研究。
4. **跨层 VJP 余弦相似度分析（附录 G）**：展示了梯度信息如何从最后一层逐步传播到前层，这种"梯度流追踪"方法可用于理解深层网络的误差传播路径。
5. **与团队方向结合机会**：可将梯度词表投影与注意力机制分析结合，探索"Attention 层是否在编辑中也遵循类似机制"；或将 Forward Pass Shifting 与 LoRA 适配结合，探索低秩编辑的统一框架。

## 关键术语表
**Backward Lens**：将语言模型梯度矩阵通过生成集（前向输入 $x_i$ 或 VJP $\delta_i$）投影到词表空间的新型可解释性方法。
**Vector-Jacobian Product (VJP)**：反向传播中的中间误差信号 $\delta = \frac{\partial L}{\partial z}$，可视为反向传播的"隐状态"，决定梯度更新的权重和方向。
**Imprint and Shift**：MLP 层知识存储的双阶段机制——$FF_1$ 通过加减前向输入实现"印记"，$FF_2$ 通过加减 VJP 实现"偏移"。
**Logit Lens (LL)**：通过将隐藏状态乘以解码矩阵 $D$ 并取 Softmax，将任意 $d$ 维向量投影为词表概率分布的方法。
**Spanning Set（生成集）**：梯度矩阵可表示为其向量的线性组合，这些向量（$x_i$ 或 $\delta_i$）构成梯度矩阵的生成集，数量仅为 prompt 长度 $n$，远小于模型维度 $d$。
**Forward Pass Shifting**：仅用单次前向传播和目标 token 嵌入近似 VJP，直接更新 $FF_2$ 的知识编辑方法，无需反向传播。
**CounterFact**：常用于 LM 知识编辑评测的数据集，包含事实性错误的 prompt-target 对（如"巴黎属于法国"改为"巴黎属于德国"）。

## 可复现要素
- **数据集**：CounterFact（Meng et al., 2022），论文使用其中 100/1000 条样本，公开可用。
- **代码**：已开源，https://github.com/shacharKZ/BackwardLens
- **模型**：GPT2（small/medium/xl）、Llama2-7B，均为公开模型。
- **关键超参**：编辑层选 30–40（实验中最佳为 layer 35），学习率 $\eta = 0.24$，无批量处理，SGD 优化器，不进行 Adam 缩放。
- **评测指标**：Efficacy（编辑成功率）、Paraphrase（泛化能力）、Neighborhood（特异性）、N-gram entropy（生成流畅度）、WikiText Perplexity（通用语言能力提升退）。
