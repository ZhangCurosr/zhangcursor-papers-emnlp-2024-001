---
title: "Collaborative-Performance-Prediction-for-Large-Language-Mode"
source: https://aclanthology.org/2024.emnlp-main.150.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:15:01"
field: "LLM评估与性能预测"
keywords: ["Large Language Models", "Performance Prediction", "Collaborative Filtering", "Scaling Laws", "Matrix Factorization", "Factor Importance Analysis", "SHAPLEY Value", "Benchmark"]
innovations: ["首次将协同过滤范式引入LLM下游任务性能预测，实现跨模型族的高精度预测", "构建72模型×29任务的大规模协作数据集并附设计因素，支持小模型→大模型外推", "利用SHAPLEY-VALUE分析模型/任务设计因素重要性，揭示缩放定律忽视的关键因素"]
benchmarks: ["HELM Lite Leaderboard", "OpenCompass", "Open-LLM Leaderboard"]
---

# 论文速读：Collaborative Performance Prediction for Large Language Models

## 一句话总结
本文提出了**协作性能预测（Collaborative Performance Prediction, CPP）**框架，借鉴推荐系统中的协同过滤思想，利用72个LLM在29个下游任务上的历史评分矩阵及模型/任务设计因素，实现对大规模语言模型性能的准确预测，突破了传统缩放定律对单一模型族和显式计算指标的依赖。

---

## 研究问题与动机

1. **评估成本高昂**：在基准上测试单个LLM往往需要 $10K+ 和 4K+ GPU小时，亟需低成本的性能预测工具。
2. **缩放定律局限**：Hu et al. (2024) 和 Isik et al. (2024) 等方法仅依赖模型族内的相似性，需要透明的设计因素（如FLOPs），且须对每个任务-模型组合单独拟合曲线，资源消耗大。
3. **跨模型族/跨任务关联被忽视**：现有方法未充分利用不同模型族之间以及不同任务之间的共性，难以泛化到新模型或新任务。
4. **专有模型无法适用**：Ruan et al. (2024) 等方法依赖已知计算度量，无法预测GPT-4等专有模型的性能。

---

## 核心贡献（创新点）

1. **提出CPP框架**：首次将协同过滤引入LLM性能预测，通过学习模型与任务的隐式表征实现跨任务、跨模型族的性能预测。与缩放定律的本质区别在于不依赖FLOPs等显式训练指标。
2. **构建大规模协作数据集**：从学术论文、技术报告及开放排行榜（HELM、Open-LLM、OpenCompass）收集了72个模型×29个任务的评分矩阵（密度56%，稀疏度44%），并附带12个模型设计因素和4个任务设计因素，远超以往方法的数据规模。
3. **支持小模型→大模型的预测**：利用跨模型族信息，可在仅有小模型性能数据的情况下准确预测大型模型（包括涌现能力）在下游任务上的表现，超越了传统缩放定律的外推能力。
4. **超越缩放定律的因子级可解释性**：首次利用SHAPLEY-VALUE对模型/任务设计因素进行系统性重要性分析，揭示了模型族、上下文窗口大小、batch size等被缩放定律忽视的重要因素。

---

## 方法详解

**框架组成**：由两部分构成——(1) 协作数据（评分矩阵 + 模型/任务设计因素）；(2) 协作预测方法（协同过滤算法）。

**1. 协作数据定义**
- 定义评分矩阵 $\mathbf{S} \in \mathbb{R}^{n \times m}$，其中 $s_{ij}$ 为模型 $M_i$ 在任务 $T_j$ 上的性能分数，未知项用"unknown"表示。
- 模型设计因素 $\mathcal{V}_m$（12维）：模型族、预训练数据量、参数量、GPU小时、FLOPs、上下文窗口、batch size、层数、注意力头数等。
- 任务设计因素 $\mathcal{V}_t$（4维）：目标能力、任务族、输出格式、few-shot设置。

**2. 预测方法**
- **Matrix Factorization (MF)**：
$$\mathbf{R} \approx \mathbf{P}^\top \cdot \mathbf{Q}, \quad \min_{\mathbf{P},\mathbf{Q}} \sum_{(u,i)\in\Omega}(r_{ui} - \mathbf{p}_u^\top \cdot \mathbf{q}_i)^2$$
通过隐因子捕捉模型与任务的共性。

- **Neural Collaborative Filtering (NCF)**：
$$\widehat{s}_{ij} = \text{MLP}(p_i, q_j, [e_{vi}, e_{vj}])$$
用MLP学习非线性交互，并可选择性接入设计因素嵌入。
仅用因素时：$\widehat{s}_{ij} = \text{MLP}(e_{vi}, e_{vj})$。

- **损失函数**：$L(\theta) = \frac{1}{N}\sum_{(i,j)\in\mathcal{D}}(\widehat{s}_{ij} - s_{ij})^2$

**3. 实验设置**
- 隐因子维度=10，学习率=0.01，迭代250,000次，5次随机分割取平均。

---

## 实验与结果

**数据集**：
- HELM Lite Leaderboard：68模型×16任务，密度82.5%
- 自建协作矩阵：72模型×29任务，密度56%（稀疏度44%）

**评估指标**：Score-Loss（MSE、L1）、Rank-Accuracy（Accuracy、MAE@2）

**主要结果**：
| 方法 | MSE Loss | MAE@2 |
|------|----------|-------|
| MF | 2.16e-2 | 83.16% |
| NCF | 1.58e-2 | 84.98% |
| **+ Factor Enhanced NCF** | **1.25e-2** | 84.54% |
| Only Factor | 1.75e-2 | 84.08% |

- 在HELM（50%训练/50%验证）上：Accuracy≈10%，MAE@2≈39%
- 在自建矩阵上：Accuracy=45%，MAE@2=84%
- **CPP-2**（提供2个 prior task 性能）相比传统缩放定律（SL）在LLaMA-3-70B上MAE@2提升约**18.6个百分点**（71.4%→90.0%）
- 加入设计因素后预测精度进一步提升，证明了显式因素的补充价值

**关键结论**：协同过滤能捕获跨模型族的相似性，即使无显式缩放因子也能实现高精度预测；基于少量prior task性能的预测可动态提升大规模模型预测精度。

---

## 相关工作脉络

1. **Scaling Laws (Kaplan et al., 2020; Hoffmann et al., 2022)**：建立计算量-性能幂律关系，关注预训练loss；本文从预训练扩展到下游任务预测。
2. **Downstream Scaling Law (Hu et al., 2024; Isik et al., 2024)**：对单一模型族内的特定任务做曲线拟合；本文突破单族限制，跨模型族联合建模。
3. **Observational Scaling Laws (Ruan et al., 2024)**：引入潜变量捕捉跨模型族模式，但仍依赖特定任务metric；本文用协同过滤统一建模多任务-多模型。
4. **Collaborative Filtering for ML Selection (Yang et al., 2019)**：将CF用于AutoML模型选择；本文将其拓展至LLM下游任务性能预测，引入更多设计因素。
5. **Efficient Benchmarking (Perlitz et al., 2024; Polo et al., 2024)**：减少评测开销；本文方法可与高效评测结合，先测少量模型/任务再快速预测其余。

---

## 局限性与未来方向

1. **单一真相假设**：不同来源的评测设置（prompt、样本量）会导致分数差异，当前仅保留一个分数，未来可纳入评测设置作为额外维度。
2. **数据质量依赖**：预测精度高度依赖所收集的协作数据质量；被动收集可能导致噪声；未来可结合主动收集（用户评测部分模型）。
3. **稀疏性限制**：当矩阵稀疏度超过60%时预测精度显著下降，需保证足够多的已知评分。
4. **未考虑评测设置变量**：few-shot设置、prompt工程等可能影响性能的变量尚未充分纳入模型。

---

## 研究启发与可借鉴点

1. **方法论迁移**：将推荐系统的协同过滤范式引入LLM性能预测是一个新颖且有潜力的方向，可探索Graph Neural Network、Transformer等更复杂的交互建模方式。
2. **因子重要性分析的借鉴**：SHAPLEY-VALUE用于因素重要性分析揭示了传统缩放定律忽视的关键因素（模型族、上下文窗口），这一分析框架可直接迁移到其他可解释性研究。
3. **从小到大的预测策略**：利用2个少量task的评测结果来预测大规模模型在更多task上的表现，为高效评测提供实践路径。
4. **协作数据构建方法论**：从论文、技术报告、公开排行榜多源收集数据的策略，以及处理缺失值和分数归一化的经验，可为后续数据驱动研究提供模板。

---

## 关键术语表

- **Collaborative Performance Prediction (CPP)**：借鉴协同过滤思想的LLM性能预测框架，通过建模模型-任务交互来预测未知性能分数。
- **Matrix Factorization (MF)**：将模型-任务评分矩阵分解为低秩因子的乘积，学习隐式表征以预测缺失值。
- **Neural Collaborative Filtering (NCF)**：用多层感知机替代MF的内积操作，捕捉模型与任务间的非线性交互。
- **Scaling Law**：描述模型性能与计算资源（如FLOPs、参数量）之间幂律关系的经验规律。
- **Emergent Abilities**：大模型在规模超过某阈值后突然展现出的、小模型不具备的能力跃迁现象。
- **Shapley Value**：源于博弈论的值，用于量化每个特征对模型预测结果的边际贡献。
- **HELM (Holistic Evaluation of Language Models)**：斯坦福发布的综合性LLM评测基准，涵盖多任务多维度评估。
- **MAE@2**：预测排名与实际排名之差的绝对值≤2的实例占比，衡量排名预测的容错精度。

---

## 可复现要素

- **数据集**：HELM Lite Leaderboard（公开）；自建协作矩阵（72模型×29任务，密度56%，论文提供了详细数据来源附录）
- **代码/权重**：代码已开源（论文标注"code is available here"，指向 GitHub 链接）
- **关键超参**：隐因子维度=10，学习率=0.01，迭代次数=250,000，随机分割5次取平均
- **训练/验证划分**：5%已知数据作验证集，其余作训练集（Appendix C.2 有详细说明）

---
