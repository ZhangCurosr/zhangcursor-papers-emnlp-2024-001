---
title: "On-the-Role-of-Context-in-Reading-Time-Prediction"
source: https://aclanthology.org/2024.emnlp-main.179.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:48:36"
field: "计算心理语言学"
keywords: ["surprisal theory", "reading time prediction", "eye-tracking", "multilingual corpus", "orthogonalization", "predictor disentanglement"]
innovations: ["提出正交化surprisal以解耦上下文与词频的共线性效应", "形式化证明surprisal、PMI与frequency在三变量线性模型中等价", "发现正交化后上下文对阅读时间的独立解释力远低于先前估计"]
benchmarks: ["MECO (Multilingual Eye-movement Corpus)", "mGPT multilingual language model"]
---

# 论文速读：On-the-Role-of-Context-in-Reading-Time-Prediction

## 一句话总结
本文通过数学上解耦surprisal与词频（frequency）的共线性关系，提出正交化surprisal作为纯上下文预测因子，发现其在预测阅读时间时解释的方差远小于传统surprisal，从而揭示先前研究可能高估了上下文在阅读理解中的作用。

## 研究问题与动机
- **核心问题**：现有surprisal理论如何将"上下文效应"与"词频效应"区分开来？两者在统计上高度共线，导致对上下文作用的量化存在模糊性。
- **现有方法不足**：surprisal（信息论视角）和PMI（关联视角）在控制词频后具有等价的预测力，三者（surprisal、PMI、frequency）在希尔伯特空间中线性相关，无法通过简单回归剥离各自独立贡献。
- **动机**：词频已被广泛证实对加工难度有重要影响，真正有意义的科学问题是：在排除词频影响后，上下文到底额外贡献了多少解释力？

## 核心贡献（创新点）
- **提出正交化surprisal预测器**：通过将surprisal投影到词频的正交补空间，构造出与词频零相关的纯上下文预测因子，区别于传统仅用raw surprisal的做法。
- **揭示surprisal理论的共线性陷阱**：形式化证明surprisal、PMI与frequency三者在希尔伯特空间中线性相关，任何包含其中两个变量的线性模型在预测力上等价。
- **重新评估上下文的作用规模**：发现正交化surprisal解释的方差远小于raw surprisal，多数语言中词频的解释力仍高于上下文，表明先前研究可能高估了上下文效应。
- **提供可迁移的去相关方法论**：将残差化（residualization）推广为函数空间中的正交投影，为认知建模中控制混淆变量提供了更严格的数学框架。

## 方法详解
- **形式化框架**：定义概率空间 $(\Sigma^* \times \overline{\Sigma}, \mathcal{P}, \pi_{\mathrm{H}} \cdot p_{\mathrm{H}})$，surprisal $\mathbf{I}_{\mathrm{H}}$、PMI $\mathbf{M}_{\mathrm{H}}$、频率 $\mathbf{Y}_{\mathrm{H}}$ 作为该空间上的随机变量（希尔伯特空间元素）。
- **正交化公式**：对mean-centered的surprisal $\mathbf{I}_{\mathrm{H}}$ 进行投影：
  $$\text{proj}_{\mathbf{Y}_{\mathrm{H}}^{\perp}}(\mathbf{I}_{\mathrm{H}}) = \mathbf{I}_{\mathrm{H}} - \frac{\langle \mathbf{I}_{\mathrm{H}}, \mathbf{Y}_{\mathrm{H}} \rangle}{\langle \mathbf{Y}_{\mathrm{H}}, \mathbf{Y}_{\mathrm{H}} \rangle} \mathbf{Y}_{\mathrm{H}}$$
  其中内积 $\langle \mathbf{X}, \mathbf{Y} \rangle = \mathbb{E}[\mathbf{X}\mathbf{Y}]$，由训练语料中的样本协方差近似。
- **线性回归模型**：三个对比模型——(i) surprisal + frequency + length；(ii) PMI + frequency + length；(iii) orthogonalized surprisal + frequency + orthogonalized length。
- **重要性度量**：使用LMG（Lindeman–Merenda–Gold）方法，对所有预测器加入顺序取平均，得到各预测器的相对重要性比例，替代传统系数解读。
- **与残差化的关系**：数学上等价于将surprisal对frequency回归后的残差，但作者强调其基于函数空间的内积结构而非有限样本估计。

## 实验与结果
- **数据集**：Multilingual Eye-movement Corpus (MECO)，包含11种语言（Dutch, English, Finnish, German, Greek, Hebrew, Italian, Korean, Russian, Spanish, Turkish）的眼动注视时长（gaze duration）。
- **预测器来源**：surprisal来自mGPT（多语言GPT-2）；frequency来自Speer (2022)；PMI由surprisal与frequency按公式(10b)分解计算；word length亦做正交化处理。
- **主要结果**：
  - 三种线性模型的 $R^2$ 相同（约0.6–0.8），验证了三者共线性。
  - 正交化surprisal解释的方差**显著低于**raw surprisal，跨越所有测试语言一致。
  - 多数语言中，正交化后第三重要预测器从"context"变为"前一个词的frequency"。
  - PMI的解释力介于surprisal与正交化surprisal之间，高估程度较小。
- **最强结果**：原始surprisal模型解释方差最高，但其与frequency共享了大量方差；正交化后context的独立贡献大幅缩减。

## 相关工作脉络
- **Surprisal理论**（Hale 2001; Levy 2008; Smith & Levy 2013）：奠定" surprisal与阅读时间呈仿射关系"的实证基础，本文在此基础上质疑其对上下文作用的解释。
- **词频效应研究**（Broadbent 1967; Bybee 2006; Shain 2019, 2024）：确立frequency作为加工难度的核心预测因子，本文将其从"控制变量"提升为"主要解释变量"进行重新评估。
- **PMI与语言处理**（Church & Hanks 1990; Levy & Goldberg 2014; Tsipidi et al. 2024）：PMI作为上下文关联度量近年被引入 psycholinguistics，本文形式化揭示其与surprisal在线性模型下的等价性。
- **残差化方法争议**（Wurm & Fisicaro 2014; York 2012; García et al. 2019）：心理学中残差化常被用于去相关，但存在系数解释混乱的批评，本文通过希尔伯特框架澄清其适用条件。
- **大模型surprisal拟合下降现象**（Oh & Schuler 2023a,b; Oh et al. 2024）：大语言模型的surprisal对阅读时间拟合反而变差，本文提出"大模型更彻底地解耦surprisal与frequency"可能正是原因。
- **Lossy-context surprisal**（Futrell et al. 2020; Kuribayashi et al. 2022）：限制上下文长度使surprisal更接近unigram频率，从而提升对阅读时间的预测力，与本文结论呼应。

## 局限性与未来方向
- **固定效应模型局限**：使用普通最小二乘回归而非混合效应模型，而后者是眼动数据建模的标准做法；混合模型的 $R^2$ 估计存在方法论争议。
- **近似计算的误差**：正交化依赖训练语料上的样本协方差估计，无法精确计算希尔伯特空间中的内积，正交化仅为"近似正交"。
- **语言偏向**：11种语言中7种为印欧语系，仅每种语系1种代表（Fino-Uralic、Semitic、Turkic、Koreanic），跨语言泛化性受限；韩语结果呈现独特模式需进一步验证。
- **单词长度的影响**：正交化后word length仍解释10%–21%方差，可能与多次眼跳(saccade)规划有关，建议未来引入"saccade数量"作为额外预测器。
- **替代角度未完全探索**：反向正交化（将frequency投影到surprisal正交补）结果方向一致但未系统比较，可进一步验证结论稳健性。

## 研究启发与可借鉴点
- **共线性诊断的前置价值**：在构建认知预测模型前，应显式检验关键预测器之间的相关性结构，避免将共享方差错误归因于某一理论变量。
- **正交化作为可复用的去混淆工具**：希尔伯特投影框架可推广到其他预测器对的解耦场景（如句法复杂度与语义预测性），具有跨领域迁移潜力。
- **LMG重要性分解优于系数解读**：在多重共线性不可避免的认知建模中，采用方差分解法评估预测器相对贡献，比关注回归系数更具解释力。
- **大模型surprisal退化的新解释**：本文结果为大语言模型规模增大反而降低阅读时间预测力这一现象提供了机制性解释，启发未来研究可主动引入频率正则化或上下文截断策略。
- **非线性扩展的实验设计**：Appendix F.3使用GAM验证非线性场景下surprisal/PMI/orthogonalized surprisal预测力等价，为后续复杂模型评估提供了参考范式。

## 关键术语表
- **Surprisal**：语言单位在给定上下文下的信息量，定义为 $-\log p(\text{unit}|\text{context})$，假设与认知加工 effort 呈仿射关系。
- **PMI (Pointwise Mutual Information)**：上下文与目标单位之间的联合关联度量，$\mu(c, u) = \log \frac{p(u|c)}{p(u)}$，本文揭示其与surprisal在线性模型中等价。
- **Frequency（词频）**：操作化为unigram surprisal $v_H(\bar{u}) = -\log q_H(\bar{u})$，反映单词的先验出现概率，是加工难度的核心预测因子。
- **Orthogonalized Surprisal**：将surprisal在希尔伯特空间中投影到frequency的正交补空间后得到的预测器，与frequency零相关，代表"纯上下文效应"。
- **Normalized Prefix Probability**：语言模型下前缀$c$的概率分布 $\pi_H(c) = \frac{1}{Z} \sum_{u} p_H(cu)$，用于定义条件概率 $p_H(\bar{u}|c)$。
- **LMG (Lindeman–Merenda–Gold) Method**：通过枚举所有预测器加入顺序并取平均，计算各预测器对解释方差的相对贡献度，适用于共线性场景。
- **Gaze Duration**：眼动实验中读者对目标词第一次凝视的总时长（首次离开该词前的所有注视之和），作为认知加工难度的核心指标。
- **Hilbert Space Projection**：在平方可积函数空间中，将向量投影到子空间的正交补上以实现去相关的数学操作，本文的核心技术工具。

## 可复现要素
- **数据集**：Multilingual Eye-movement Corpus (MECO, Siegelman et al. 2022)，公开可用（需申请许可）。
- **代码与权重**：代码已开源，见 https://github.com/rycolab/context-reading-time；mGPT模型为开源多语言语言模型。
- **关键超参**：GAM模型使用mgcv包，basis functions数量设为6；10折交叉验证；预测器标准化（mean=0, std=1）。
- **注意事项**：论文指出MECO存在off-by-one的ID错误（已在代码中修正）、句子内重复词（如"als als"）被保留；建议使用Version 1.3+数据。
