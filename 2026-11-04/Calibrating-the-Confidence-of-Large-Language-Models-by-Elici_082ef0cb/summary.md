---
title: "Calibrating-the-Confidence-of-Large-Language-Models-by-Elici"
source: https://aclanthology.org/2024.emnlp-main.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:45:36"
---

# 论文速读：Calibrating-the-Confidence-of-Large-Language-Models-by-Elici

## 一句话总结
本文提出即插即用的 UF Calibration 方法，将大语言模型的置信度解耦为“题目不确定性”与“答案忠实度”两个正交维度并通过交互探测联合估计，在不依赖内部 logit、无需提示词引导的条件下，显著提升了 RLHF-LM 在多项选择题任务上的校准表现。

## 研究问题与动机
1. RLHF 对齐后的大模型普遍出现过度自信现象，输出置信度与实际准确率严重失配。
2. 基于 logit 的方法需访问单 token 概率或采用高温温标（t>2.0），会破坏生成质量且无法用于黑盒商业模型。
3. 基于语言表述（Verb/Ling）的方法依赖复杂提示词与强自我感知能力，不同模型倾向固定输出高置信短语（如 0.8/0.9）。
4. 现有 ECE 指标无法捕捉“置信度过度集中”的假象，缺乏兼顾单调性与分布均匀性的多维评估体系。

## 核心贡献（创新点）
1. 提出 UF Calibration 框架，将置信度形式化为 Uncertainty(Q) 与 Fidelity(a_i) 的乘积。与仅依赖单次采样频率或温度缩放的方法相比，本方法通过贪心解码构建层级忠实度链，从偏好强度而非单纯出现频次刻画模型置信。
2. 设计基于 “All other options are wrong.” 替换的交互式忠实度探测机制。与直接询问模型置信度（Verb/Ling）需强自我意识不同，该方法通过逐步剔除选项的观察行为间接量化忠实度，完全兼容黑盒 API。
3. 提出 IPR 与 CE 两个新评估指标。与传统仅优化 ECE 的做法相比，IPR 衡量可靠性图单调性、CE 衡量置信度分布均匀性，有效暴露单指标优化下的置信度集中假阳性。
4. 系统论述“何种置信度才是真正良好校准的”。与以往追求最低 ECE 的文献不同，本文指出真实校准需在 ECE、IPR、CE 三者间取得平衡，为后续研究提供了理论参照。

## 方法详解
- **采样阶段**：对题目 Q 以温度 T=1.0、top_p=1.0 进行 K=10 次采样，统计候选答案频率分布 P_sampled，取众数作为最终答案。
- **忠实度估计（Eliciting Fidelity）**：对每个候选答案 (a_i, o_i)，将选项内容替换为 “All other options are wrong.” 后重新输入模型，采用贪心解码（Greedy Decoding）依次剔除模型不选的选项，直至模型选择该替换项或仅剩该选项，形成层级链 C（如 "A→C→D"）。链中从右向左第 i 个元素权重设为 τ^i（默认 τ=2），归一化忠实度为 Fidelity_C(a_i) = τ^i / Στ^j；最终答案为跨所有链的加权平均 F(a_i) = Σ P_sampled(C_j)·Fidelity_{C_j}(a_i)。
- **不确定性估计（Uncertainty）**：计算采样频率分布的信息熵并按选项数 M 归一化：Uncertainty(Q) = -Σ(p_i·log p_i) / log M，值越大表示模型对该题目越不确定。
- **置信度合成**：Conf(Q, a_i) = (1 - Uncertainty(Q)) × F(a_i)，实现“整体把握程度 × 具体答案偏好强度”的解耦计算。

## 实验与结果
- 数据集：ARC-Challenge、MMLU、CommonSenseQA、TruthfulQA（均为 0-shot 评测）。
- 模型：GPT-3.5-Turbo、GPT-4-Turbo、Baichuan2-13B-Chat、LLaMA2-7B/13B/70B-Chat（共 6 个 RLHF-LM）。
- 基线：Verb、Ling、Sampled（K=10）、Token（logit softmax）、Conformal Prediction、CAPE-ENUM。
- 评估指标：ECE_10（↓）、IPR_10（↓）、CE_10（↑）、Accuracy（↑）。
- 主要结果：UF Calibration 在所有模型与数据集上的 ECE_10 与 IPR_10 均显著优于全部基线，CE_10 保持高位，且准确率与基线持平。
- 最强表现：GPT-4-Turbo 在 MMLU 上 ECE_10 降至 0.089（Sampled 为 0.153），IPR_10 降至 0.083（Sampled 为 0.311）；LLaMA2-13B-Chat 在 ARC 上 ECE_10 达 0.069，IPR_10 仅为 0.028，显著优于 Token 基线的 0.064/0.067。
- 鲁棒性：方法对不同采样温度（0.1~1.0）与模型参数量（7B~70B）均保持稳定，验证了 plug-and-play 的通用性。

## 相关工作脉络
1. Logit-based / Temperature-Tuning（Guo et al., 2017; Kadavath et al., 2022）：依赖单 token 概率与高温平滑，需访问模型内部参数；本文方法完全不依赖 logit，可直接作用于黑盒 API。
2. Verbalization-based（Lin et al., 2022a; Tian et al., 2023）：通过 Prompt 直接询问置信度，易受自我感知能力与语言主观性影响；本文以选项替换交互间接推断忠实度，规避了提示词偏差。
3. Conformal Prediction for LLMs（Kumar et al., 2023）：需划分校准集并预设误差率 α 计算分位数阈值；本文无需额外校准数据，为纯推理时方法。
4. CAPE（Jiang et al., 2023）：通过排列选项顺序平均概率分布，需完整 logit 或大量采样近似；本文利用贪心链直接刻画偏好层级，调用开销更低且兼容黑盒。
5. 本文定位：填补了“无需 logit、无需提示词引导、无需校准集”的通用黑盒置信度校准空白，并以“不确定性×忠实度”分解提供了可解释的分析视角。

## 局限性与未来方向
1. 主要适用于答案集合已知的任务（MCQA、文本分类、RLHF 偏好标注），开放生成场景下的忠实度估计尚未探索。
2. 需多次调用模型（采样 + 贪心链构建），调用成本较高，如何以极少请求次数估计分布仍需研究。
3. 忠实度链依赖贪心解码，未系统评估不同 decoding 策略（如 top-k/核采样）对链稳定性的影响。
4. 未来可将该方法迁移至开放式问答/生成任务的置信度估计，或结合缓存机制与并行请求降低多轮调用开销。

## 研究启发与可借鉴点
1. **置信度解耦范式**：将“整体把握”与“特定答案偏好”
