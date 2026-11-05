---
title: "AdaZeta-Adaptive-Zeroth-Order-Tensor-Train-Adaption-for-Memo"
source: https://aclanthology.org/2024.emnlp-main.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:12:50"
field: "大模型高效微调"
keywords: ["zeroth-order optimization", "parameter-efficient fine-tuning", "tensor-train decomposition", "large language models", "memory-efficient training", "gradient estimation"]
innovations: ["提出自适应查询数量调度策略亚线性递增query数以消除ZO微调发散问题", "将Tensor-Train适配器与快速并行收缩引入ZO场景以降低可训练参数维度并加速前向", "在Llama-2-7B上以<14GB显存超越多数ZO基线并在部分任务超过一阶全参数微调"]
benchmarks: ["GLUE", "SuperGLUE", "SQuAD", "DROP"]
---

# 论文速读：AdaZeta-Adaptive-Zeroth-Order-Tensor-Train-Adaption-for-Memo

## 一句话总结
本文提出 AdaZeta 框架，结合张量训练（Tensor-Train）适配器与自适应查询数量调度策略，解决零阶（ZO）优化在大型语言模型微调中的性能下降和发散问题，在 Roberta-Large 和 Llama-2-7B 上显著优于 MeZO、MeZO-LoRA 等基线方法，同时保持 ≤14GB 内存开销。

## 研究问题与动机
- **LLM 微调内存瓶颈**：随着模型规模增长，传统一阶（FO）微调依赖反向传播图，显存开销巨大。
- **ZO 方法性能与稳定性不足**：MeZO 等纯前向方法虽可节省 >8× 内存，但存在与 FO 方法的性能鸿沟，且大规模微调时频繁发散（如 Fig.1 所示）。
- **现有 PEFT-ZO 方案表达能力受限**：MeZO-LoRA 因 LoRA 极低秩导致表征能力不足；Sparse-MeZO 跨任务/超参表现不稳定。
- **增大 Batch Size 的代价**：既有工作通过扩大 batch size 降低 ZO 梯度方差，但引入显著运行时开销且无法根本改善发散。

## 核心贡献（创新点）
1. **提出 AdaZeta 框架**：融合快速前向张量适配器与自适应查询调度，在多项任务上超越 MeZO/MeZO-LoRA/Sparse-MeZO，收敛更快。与已有 ZO 方法的区别在于同时从"参数空间降维"和"梯度噪声控制"两方面协同优化。
2. **设计自适应查询数量调度**：按 `Q_k = min(α·e_k^β, Q_max)` 亚线性递增 query 数，理论保证收敛性；相比固定 query 或增大 batch size 的方案，在更低内存下消除发散。
3. **实现快速前向张量适配器**：将 TT 因子的顺序收缩改为并行分组收缩（二分/三分/四分结构），显著加速 ZO 每步所需的两次前向传播。
4. **提供非凸设置下的收敛率理论分析**：证明梯度范数上界与可训练参数维度 d 及 query 调度相关，揭示 PEFT 降维对 ZO 收敛的必要性。

## 方法详解
**零阶梯度估计（RGE）**：每步对可训练参数 w ∈ ℝ^d 施加 Q_k 组随机扰动 z_q ~ N(0, I_d)，通过前后两次前向损失差估计梯度：
`∇̂ℓ(w_k) = Σ_{q=1}^{Q_k} [(ℓ(w_k + εz_q) - ℓ(w_k - εz_q)) / (2ε)] · z_q`

**张量训练适配器（Tensorized Adapter）**：
- 将 adapter 权重矩阵 W ∈ ℝ^{m×n} 重塑为 2o 阶张量，用 TT 格式分解为因子序列 [G_1, …, G_{2o}]，每因子 G_i ∈ ℝ^{r_{i-1}×k_i×r_i}，首尾秩为 1。
- 可训练参数较全参数减少 >80%；冻结 layer norm 以避免放大噪声。
- **快速前向**：将因子序列分组并行收缩，例如二分结构：`W = R(∏_{i=1}^o G_i · ∏_{j=o+1}^{2o} G_j)`，避免高阶张量中间形态，加速 ZO 必需的双前向。

**自适应查询调度**：
- 每 epoch 初始计算 `Q_k = min(α·e_k^β, Q_max)`，其中 e_k = ⌊k / ⌈D/B⌉⌋ 为当前 epoch 编号，α∈(0,1)、β∈(0,1)、Q_max 为阈值。
- 中规模模型（Roberta-Large）固定 Q=1；大规模模型使用递增 schedule（本文取 α=0.85, β=0.45, Q_max=20）。
- 梯度更新：`w ← w - η · (1/Q_k) Σ_q [(ℓ_+^q - ℓ_-^q)/(2ε)] z_q`。

**收敛定理（Theorem 1）**：在 L-Lipschitz 梯度假设下，随机选取迭代点 T 满足：
`E[‖∇ℓ(w_T)‖²] ≤ O((R + ε²L + C(d,ε) Σ_k 1/Q_k) / (Kε))`
当 Q_k 亚线性增长时 Σ_k 1/Q_k 有界，保证梯度趋于零；若 Q=1 常数，上界含 C(d,ε)∝d 项难以最小化。

## 实验与结果
**数据集与模型**：
- Roberta-Large：SST-2/5、QNLI、MNLI、SNLI、RTE、MR（16-shot，BS=16/64）。
- Llama-2-7B：SuperGLUE（CB、BoolQ、WSC、COPA、ReCoRD）+ SQuAD、DROP（低资源设置，1K train/500 val/1K test）。

**主要结果（Llama-2-7B，Table 2）**：
- AdaZeta 在 10 项任务中 8 项优于所有 ZO 基线；RTE 74.0（MeZO-LoRA 59.6）、CB 75.0、COPA 94.0、SST2 91.0。
- AdaZeta 在 RTE、CB、COPA 上甚至超过 FO-AdamW 全参数微调，而后者需 >8× 显存。
- MeZO-LoRA 在 SST2/RTE/BoolQ 等任务出现发散（SST2 86.8 vs AdaZeta 91.0）。

**收敛速度（Fig.1 & Table 3）**：
- AdaZeta 在 <1K 步达到 loss 0.4，MeZO-LoRA 需 ~6K 步（6× 加速），学习率相同 1e-4。
- GPU 小时对比（Llama-2-7B）：SST2 任务 AdaZeta 1.1h vs MeZO-LoRA(BS=16) 0.6h / Sparse-MeZO 4.1h；MultiRC 12.1h vs MeZO-LoRA(BS=64) 30.0h。

**显存效率（Table 4 & Fig.3）**：
- Llama-2-7B SST2 仅 14GB（FT 需 118.65GB），减少 >8×；比 MeZO/MeZO-LoRA/Sparse-MeZO 省内存或持平。
- 即使与 LoRA(r=1, BS=1) 对比，AdaZeta(r=8, BS=16) 仍节省 ~50% 显存（23.01GB vs 35.60GB）。

## 相关工作脉络
- **MeZO (Malladi et al., 2023)**：纯前向 ZO 微调开山工作，消除反向传播图；本文在其基础上引入 PEFT + 自适应 query。
- **MeZO-LoRA (Malladi et al., 2023)**：将 LoRA 适配于 ZO；本文指出极低秩表征能力不足，改用 TT 适配器获得更强表达能力。
- **Sparse-MeZO (Liu et al., 2024)**：按权重大小生成剪枝 mask 降维；本文认为其跨任务/超参不一致，转而用 PEFT 稳定降维。
- **ZO-AdaMU (Jiang et al., 2024)**：一阶优化技巧迁移至 ZO；本文认为其适配 ZO 特性不足且引入优化器状态内存。
- **MeZO-SVRG (Gautam et al., 2024)**：引入 SVRG 方差缩减，但需额外参数副本使显存翻倍；本文用 query 调度以更低显存实现稳定收敛。
- **LoRA / Adapters / Prefix-tuning**：FO 场景成熟 PEFT；本文首次系统探索 TT 格式 + ZO 的组合并解决其前向效率问题。

## 局限性与未来方向
- **串行 query 执行**：当前多 query 在 for-loop 中顺序前向，限制进一步加速；未来可探索 GPU 并行或分布式多卡并发。
- **仅验证两种模型尺度**：Roberta-Large 与 Llama-2-7B，尚未覆盖更大规模（如 13B/70B）或更长的生成任务。
- **超参依赖手动调优**：TT 因子形状、rank、α/β/Q_max 等需实验试探，缺少自动化搜索机制。
- **论文自述方向**：将自适应 query 调度推广至其他 PEFT 方法（如 LoRA），以及结合 rank-adaptive TT 进一步优化。

## 研究启发与可借鉴点
- **ZO 优化中的"降维优先"思路**：ZO 梯度方差与参数维度 d 直接相关，通过 PEFT 显式压缩可训练参数是提升估计质量的根本途径；可迁移到其它 ZO 场景（如强化学习、超参优化）。
- **亚线性递增查询调度**：以 epoch 为单位的缓慢增 query 策略兼顾早期效率与后期稳定，思想可复用于任意基于有限差分/随机估计的优化框架。
- **TT 并行收缩加速前向**：将顺序收缩改写为分组并行，对任何依赖张量网络前向的方法（神经架构搜索、压缩推理）均有参考价值。
- **冻结层归一化避免噪声放大**：ZO 对 scaling factor 的噪声极度敏感，冻结 LN 是一种实用的工程技巧，可在其它 ZO-PEFT 设计中复用。
- **与本团队的结合机会**：若团队关注低资源微调或绿色 AI，可将自适应 query 与可微分 TT rank 搜索结合，探索全自动 "高效 ZO-PEFT 控制器"。

## 关键术语表
**Zeroth-Order (ZO) Optimization**：不依赖梯度，仅通过函数值差分估计梯度的优化方法，适用于黑盒或反向传播不可用场景。
**Randomized Gradient-Free Estimation (RGE)**：通过对参数加高斯随机扰动并比较前后损失来无偏估计梯度。
**Tensor-Train (TT) Decomposition**：将高维张量分解为低秩因子序列的压缩表示，参数复杂度随维度线性增长而非指数增长。
**Parameter-Efficient Fine-tuning (PEFT)**：冻结预训练主体，仅微调少量附加参数的 LLM 适配技术。
**Adaptive Query Schedule**：按亚线性函数随 epoch 递增每次迭代的 query 次数，平衡梯度估计方差与训练效率。
**Divergence in ZO Fine-tuning**：ZO 梯度估计噪声导致训练损失振荡甚至发散失控的现象，在大模型中尤为突出。
**Forward-pass Only**：仅需前向计算无需反向传播图的训练范式，是 ZO 方法节省显存的核心机制。

## 可复现要素
- **数据集**：GLUE/SuperGLUE、SQuAD、DROP；均公开可用。
- **代码/权重**：论文未明确声明开源（仅提及 Amazon/NERSC 资源），未附 GitHub 链接。
- **关键超参**：
  - Roberta-Large：BS∈{16,64}，lr∈{1e-4,5e-5}，bottleneck=64，TT rank=5，ε=1e-3，固定 Q=1。
  - Llama-2-7B：BS∈{16,64}，lr∈{1e-4,5e-5}，bottleneck∈{8,64}，rank∈{5,8,16}，α=0.85，β=0.45，Q_max=20，ε=1e-3。
- **硬件**：NVIDIA Tesla A100-40GB。
- **复现难点**：TT 因子 shape 需按 Table 9 手工配置；Sparse-MeZO 官方未发布代码，作者自行复现。
