---
title: "Can-Large-Language-Models-Always-Solve-Easy-Problems-if-They"
source: https://aclanthology.org/2024.emnlp-main.92.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:45:11"
field: "大语言模型可信性与一致性评测"
keywords: ["LLM consistency", "hard-to-easy inconsistency", "ConsisEval", "consistency score", "relative consistency score", "in-context learning", "data contamination"]
innovations: ["首次系统研究 LLM 难→易不一致现象并提出成对难度序基准", "提出基于条件概率的一致性分数 CS 与相对一致性分数 RCS 定量指标"]
benchmarks: ["GSM8K", "HumanEval", "IFEval"]
---

# 论文速读：Can-Large-Language-Models-Always-Solve-Easy-Problems-if-They

## 一句话总结
本文首次系统研究大语言模型（LLM）的"难→易不一致"现象——模型能解难题却反常答错简单题，并提出 ConsisEval 基准与一致性分数（CS）/相对一致性分数（RCS）指标来量化评估。

## 研究问题与动机
1. **现有不一致研究的盲区**：前作多关注语义等价重述、顺序扰动、逻辑变换等一致性问题，但未考察"能力更强反而在简单题上出错"这一悖论。
2. **可信应用的需求**：LLM 面向真实世界部署需要更高的可信度与一致性，而人类天然是一致的推理者，LLM 为何出现这种差异仍未知。
3. **缺乏系统化评测**：尚无现成基准能严格保证难度序（easy < hard）并对成对数据进行条件概率评估。
4. **强模型≠高一致性的反例**：GPT-4 以 92.2% CS 领跑，但仍有冗余信息分心、题目误读等不一致案例，且 Claude-3 Opus 数学能力更强却一致性略低，说明能力与一致性并非单调对应。

## 核心贡献（创新点）
1. **首次建立 LLM 难→易不一致的系统化研究**：提出 ConsisEval 基准，通过自动生成分割+人工校验构建严格难度序的成对数据（代码/数学/指令跟随三领域）。
2. **提出一致性分数（CS）**：定义为 $CS = P(a|b) = \frac{\sum P(a_i)P(b_i)}{\sum P(b_i)}$，从概率角度量化"答对难题的前提下答对简单题的条件概率"。
3. **提出相对一致性分数（RCS）**：通过 CS 的上/下界分析当前能力不变时一致性的提升潜力，低 RCS 对应高改进空间。
4. **设计两种概率估计方法**：多采样估计（MLE on Bernoulli/Binomial）与早停估计（针对 API 调用成本优化），前者适合开源本地部署、后者适合闭源模型。
5. **发现训练数据与 ICL 难度的影响**：硬数据微调与硬示例上下文学习均能提升一致性，且 80% 硬 + 20% 易的组合优于纯硬数据。

## 方法详解
1. **ConsisEval 构建流程**：
   - 易题取自公开数据集：GSM8K（3 步算术题，298 条）、HumanEval（全量，164 条）、IFEval（单约束题，270 条）。
   - 难题由 GPT-4 基于好题模板自动生成候选，再由人工校验两条标准：①严格难度子集关系（易题所需步骤/知识为难题的真子集）；②答案或检查函数正确。
   - 提示模板包含五部分：修改任务指令、demo 对、待改题目/答案/检查函数分解。
2. **一致性分数（CS）**：将模型对每道题的回答建模为伯努利过程，单次采样的正确概率 $P(a_i)$ 与 $P(b_i)$ 独立（因题目成对生成但各自采样独立），得联合概率 $P(a_i,b_i)=P(a_i)P(b_i)$，代入条件概率公式得式(1)。
3. **相对一致性分数（RCS）**：定义能力相同的假设模型集合 $\Omega$（每道题对 B 的正确概率相同、对 A 的正确概率多重集相同），推导出 CS 的严格数学上下界；但因过松，改用更紧的启发式界：$CS_{low} = \frac{\sum P(a_i)}{N}$、$CS_{upp} = \frac{\sum (P(b_i)+\hat{\mu})P(b_i)}{\sum P(b_i)}$，其中 $\hat{\mu}$ 为 $P(a_i)-P(b_i)$ 的均值。
4. **多采样估计**：对每道题独立采样 $m$ 次，正确数服从 $B(m, P)$，MLE 给出 $\hat{P} = \frac{\sum a_i^j}{m}$。
5. **早停估计**：先采样 $k_{min}$ 次，若至少有一次正确则停止；否则继续到首次正确或达到 $k_{max}$，MLE 给出 $\hat{P} = k_c/k$。

## 实验与结果
- **评测模型**：闭源（GPT-4 Turbo、GPT-3.5 Turbo、Claude-3 Opus、Qwen Max）和开源（Llama2/3 系列、Qwen1.5 系列、ChatGLM3、DeepseekLLM、Mistral、Baichuan2、Yi，共 17 个 chat 版本）。
- **最强结果**：GPT-4 Turbo 平均 CS 最高达 **92.2%**，代码 88.1%、指令跟随 91.8%、数学 96.8%；Claude-3 Opus 以 91.8% 紧随其后。
- **能力-一致性相关性**：代码域 Hard 准确率与 CS 的 Kendall 相关系数为 **0.801**，但存在例外（Claude-3 Opus 数学能力更强却一致性略低于 GPT-4 Turbo）。
- **RCS 洞察**：GPT-4 Turbo 代码 RCS 仅 34.8%（潜力提升 65.2%），而 Llama2-70B-Chat RCS 达 81.5%（在当前能力下已较一致）。
- **训练数据实验**：固定总量 2500 条，调整 GSM8K 难易比例微调 Llama3-8B，CS 与 RCS 均随硬数据比例上升；80% 硬 + 20% 易 > 100% 硬。
- **ICL 实验**：1-4 shot 下硬示例的一致性优于易示例，且随 shot 数增加而上升。
- **失败归因**（GPT-4 案例分析）：①冗余信息分心（Table 6）；②数据错用（Tuesday→Thursday，Table 7）；③题意误读（cost of travel→cost of tickets，Table 8）；④off-by-one 逻辑错误（Table 9）；⑤计算错误（Table 10）。

## 相关工作脉络
1. **LLM 一致性**（Elazar et al., 2021; Raj et al., 2023; Jang & Lukasiewicz, 2023; Li et al., 2023b）：这些工作聚焦语义等价、顺序扰动、逻辑变换、生成-验证一致性；本文定位为"难度层级"维度的一致性。
2. **Easy-to-Hard 泛化**（Hupkes et al., 2020; Hase et al., 2024; Sun et al., 2024b）：研究模型从简单元素泛化到复杂组合；本文反向——考察"会做难的却做错简单的"悖论。
3. **Leveled Evaluation**（Liu et al., 2024; Xu et al., 2024a）：按推理步数分层评测能力；本文成对配对+严格难度序，粒度更细。
4. **数据污染研究**（Magar & Schwartz, 2022; Xu et al., 2024b）：本文附录 F 分析易/难题泄露对 CS 的相反方向影响，直接关联评估可信度。
5. **LoRA 微调**（Hu et al., 2021）：本文微调实验采用此技术，作为可复用的参数高效训练方法参考。

## 局限性与未来方向
1. **计算成本**：重复采样估计概率比传统单次判定评估昂贵得多。
2. **CS 粒度有限**：只能反映模型整体一致性，无法定位对哪类问题更不稳定；不同模型的不一致题目也各不相同，仍需人工 case study。
3. **数据污染风险**：易题来自公开数据集，HumanEval 已被报道 25% 污染，可能导致 CS 被高估；难题泄露则压低 CS。
4. **缺少人类基线**：理论上人类 CS 应为 100%，但因粗心和耗时难以实测，缺乏参照系。
5. **未揭示根本原因**：预训练+微调范式是否必然导致不一致、人类与 LLM 知识习得机制差异如何引发该现象，仍需深入研究。

## 研究启发与可借鉴点
1. **成对难度序构建方法**可迁移：自动生成+人工校验的两阶段流水线适合其他需要严格难度控制的评测场景。
2. **概率估计设计**（早停策略）可直接复用于需要 MLE 估计的模型评估任务，在 API 成本与估计精度间取 trade-off。
3. **RCS 视角**——在能力不变下评估一致性提升潜力，为模型迭代提供"瓶颈分析"工具，不只看绝对能力。
4. **训练/ICL 数据难度配比发现**：后续可探索在不同任务（如代码生成、知识问答）中引入更多硬样本以提升系统稳定性。
5. **失败归因五类型**（冗余分心/数据错用/题意误读/off-by-one/计算错误）可作为对齐训练的正样本目标，指导 RLHF/DPO 的数据构造。

## 关键术语表
- **ConsisEval**：面向 LLM 难→易一致性的评测基准，成对数据覆盖代码、数学、指令跟随三领域。
- **Consistency Score (CS)**：条件概率 $P(\text{答对易题}|\text{答对难题})$，衡量模型的一致性水平。
- **Relative Consistency Score (RCS)**：$(CS-CS_{low})/(CS_{upp}-CS_{low})$，在能力约束下的一致性相对位置，低值暗示高改进潜力。
- **Hard-to-easy inconsistency**：模型能解难题却反常答错简单题的不一致现象。
- **Multiple Sampling Estimation**：通过对同题独立采样 $m$ 次求 MLE 估计 $P(a_i)$。
- **Early Stopping Estimation**：设定 $[k_{min}, k_{max}]$ 区间，首次出现正确答案或达到上限即停止，减少 API 调用。
- **Data contamination / leakage**：训练数据与测试数据重叠，易题泄露抬高 CS，难题泄露压低 CS。

## 可复现要素
- **数据集**：ConsisEval 代码/数学/指令跟随成对数据；易题来源 GSM8K、HumanEval、IFEval 均为公开。论文声明"数据与代码将在 GitHub 公开"，但截至写作时以论文声明为准。
- **模型权重**：开源模型权重均来自 Huggingface；闭源模型通过 API 调用。
- **关键超参**：小模型（7B/13B）多采样 $m=20$；大模型（70B）/闭源早停 $k_{min}=3, k_{max}=20$；微调用 LoRA；训练数据实验固定总量 2500 条，调整难易比例。
- **硬件**：7B/13B 单卡 Nvidia A100 80G；70B 三卡 A100 80G。
- **Prompt 模板**：附录 J 给出了代码/数学/指令跟随三领域的数据生成 prompt。
