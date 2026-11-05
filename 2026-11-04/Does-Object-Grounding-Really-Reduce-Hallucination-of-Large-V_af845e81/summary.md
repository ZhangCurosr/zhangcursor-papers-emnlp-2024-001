---
title: "Does-Object-Grounding-Really-Reduce-Hallucination-of-Large-V"
source: https://aclanthology.org/2024.emnlp-main.159.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:16:56"
field: "多模态大模型幻觉评估与缓解"
keywords: ["object hallucination", "vision-language model", "grounding", "image captioning", "evaluation benchmark"]
innovations: ["首次系统控制变量分析定位目标对LVLM开放生成幻觉的影响", "提出CHAIR-MEN语义扩展以支持OOV大词汇表幻觉评估", "揭示推理时强制grounded captioning存在faithfulness-informativeness权衡"]
benchmarks: ["POPE", "CHAIR", "FaithScore", "Objects365"]
---

# 论文速读：Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models?

## 一句话总结
本文首次系统分析了细粒度物体定位（grounding）目标对大型视觉语言模型（LVLM）开放图像描述幻觉的实际影响，发现即使模型具备更强的定位能力，也几乎不能减少开放生成中的物体幻觉，质疑了"定位训练能治幻觉"的直觉假设。

## 研究问题与动机
- **核心问题**：LVLM在开放图像描述中频繁产生"幻觉"（hallucination），即描述图像中不存在的物体或属性；先前研究声称加入定位目标（如referring expressions、grounded captioning）可减少幻觉，但该主张缺乏严谨实证。
- **现有评估缺陷一**：既有研究主要依赖POPE等**基于QA**的评测协议（是/否问题），而真实应用更多是自由格式开放生成（captioning），QA评估并不能直接反映开放生成中的幻觉水平。
- **现有评估缺陷二**：常用评测基准（MSCOCO）已广泛存在于LVLM训练数据中，导致指标**低估**了真实幻觉（数据泄露问题）。
- **直觉与实证的gap**：加入定位目标看似能提升细粒度视觉-语言理解，但是否真正能"因果性"地抑制幻觉，缺乏系统控制变量实验。

## 核心贡献（创新点）
1. **首次系统性消融分析**：控制所有其他变量，仅改变训练mix中是否包含两种主流定位目标（RE与GC），在三种不同LLM backbone上全面对比，得到稳健结论。
2. **提出更贴合实际的评估协议**：引入out-of-distribution数据集Objects365，并扩展CHAIR为语义匹配的CHAIR-MEN版本，同时结合FaithScore与POPE，形成互证体系。
3. **揭示定位能力与幻觉减少之间无因果联系**：即使+GC显著提升了expression grounding精度（Precision@50大幅改善），也不转化为开放生成中幻觉的下降。
4. **推理时强制生成grounded captions的边际收益量化**：发现推理阶段要求模型输出bbox虽然能轻微降低CHAIRi/FaithScore，但以牺牲信息量（Objects/Coverage/Facts下降）为代价。
5. **提供可复现的训练基线与评测代码**：开放训练代码与所有实验细节，便于后续工作在同一protocol下比较。

## 方法详解
- **模型架构**：采用标准LVLM范式——OpenAI CLIP ViT-L/14图像编码器 + MLP alignment module（或perceiver-resampler）投影至LLM嵌入空间，LLM参数冻结并以4-bit量化+LoRA（r=64, α=128）微调；实验三种backbone：Vicuna 1.5 7B、Llama-3 8B (instruct)、Phi-3-mini。
- **定位目标定义**：
  - **Referring Expressions (RE)**：双向任务——给定文本描述生成bbox，反之给定bbox生成描述；使用RefCOCO + Visual Genome采样各320k样本。
  - **Grounded Captioning (GC)**：生成包含bbox坐标的交错描述（如"[0.10, 0.05, 0.64, 1.00]"以2位小数文本形式作为token输入）；使用Flickr30k-Entities 150k样本。
- **训练mix**：Base模型仅含标准captioning（MSCOCO 400k）+ 长caption（LLaVA-DETAILED 23k）+ VQA（VQAv2 170k yes/no）；+RE/+GC/+RE+GC依次加入定位任务数据。
- **评估协议**：
  - QA评估：POPE（random/popular/adversarial三类否定问题），测试集包括MSCOCO及从O365构造的O365/COCO（80类）与O365/non-COCO（285类）。
  - 开放生成评估：CHAIRi（扩展为语义匹配版本CHAIR-MEN，使用Sentence-BERT嵌入+余弦相似度阈值t₁=0.73/t₂=0.78校准）与FaithScore（LLM抽取原子事实→OFA VQA验证）。
  - 质量指标：CIDEr、CLIPScore、#Words、Coverage、Objects数、Facts数。

## 实验与结果
- **数据集**：MSCOCO（5000测试图）+ Objects365（5386验证图）；O365额外构造POPE测试集（各1500样本）。
- **Expression Grounding能力**（Table 2）：添加GC后，所有LLM在RefCOCO系列Precision@50显著提升（如Vicuna+RE: 61.89 → Vicuna+RE+GC: 73.66），证明RE与GC目标互不冲突且GC对RE有正向迁移。
- **QA幻觉（POPE）**（Table 1）：三种LLM上，+RE/+GC/+RE+GC均未一致性地、显著地提升POPE准确率；O365/non-COCO（OOV）场景下差异更小，说明定位目标与QA幻觉减少无因果关系。
- **开放生成幻觉（Table 3）**：
  - MSCOCO上：Base vs +RE/+GC在CHAIRi/FaithScore上差异均<0.3%，CIDEr/CLIPScore也无显著变化；Objects365上趋势相同。
  - 最长提升：Llama-3 Base CHAIRi=3.84 → +GC CHAIRi=3.61（-0.23绝对值），几乎可忽略。
- **推理时强制Grounded Captioning（Table 4）**：
  - MSCOCO上：Llama-3 +GC CHAIRi下降0.17，但Objects减少0.21、Facts减少0.18，存在faithfulness-informativeness tradeoff。
  - O365上：下降幅度略大（Llama-3 +GC CHAIRi下降1.07），但信息量仍同步下降。
- **核心结论**：训练中加入RE/GC目标对开放生成幻觉**几乎没有因果效应**；推理时强制输出bbox可轻微降低幻觉，但以牺牲信息密度为代价。

## 相关工作脉络
1. **Shikra (Chen et al., 2023b)**：提出将RE/GC纳入训练mix的LVLM，声称能减少幻觉；本文指出其评估过度依赖QA协议与MSCOCO，结论不可推广到开放生成。
2. **POPE (Li et al., 2023b)**：当前最主流的LVLM幻觉QA基准；本文确认其在QA场景有用，但强调其与open captioning幻觉的相关性未被严格验证。
3. **FaithScore (Jing et al., 2023) / CHAIR (Rohrbach et al., 2018)**：开放生成幻觉自动度量；本文在两者基础上提出语义扩展版CHAIR-MEN，解决原CHAIR只能用于MSCOCO 80类的限制。
4. **Blip-2 / LLava系列 (Liu et al., 2023b,c)**：主流LVLM训练pipeline；本文复现其对齐模块设计，作为控制实验基线。
5. **REINVENT / RLHF-V (Sun et al., 2023; Yu et al., 2023)**：用RL从人类反馈中减少幻觉；本文表明单纯增加定位监督信号并非充分条件，需探索其他方向。
6. **OFA (Wang et al., 2022)**：被本文用作FaithScore中的VQA验证模型，因速度与精度均衡。

## 局限性与未来方向
- **计算预算限制**：训练数据量与步数少于主流工作（如Shikra、LLaVA），无法排除"在更大scale下定位目标才显现效果"的可能性。
- **架构选择固定**：仅测试MLP aligner与perceiver-resampler两种，未探索其他对齐结构或不同规模LLM。
- **度量依赖自动指标**：虽使用CHAIR-MEN与FaithScore双指标互证，但仍缺乏大规模人工评估；定性分析仅覆盖有限样本。
- **未来方向**：
  - 探索定位目标之外的幻觉抑制机制（如解码策略、RLHF、事实一致性训练）。
  - 在更大scale（更多训练数据、更强LLM backbone）上重现实验以验证结论的外推性。
  - 建立更贴合真实应用的开放生成幻觉评估协议（如跨域分布、长caption、多轮对话场景）。

## 研究启发与可借鉴点
1. **实验设计范式**：严格控制单一变量（仅增减定位目标），在三种异构LLM backbone上重复验证，所得结论具有高度可迁移性；后续研究可借鉴此"控制变量+多backbone"范式评估任意训练信号的有效性。
2. **CHAIR-MEN语义扩展**：将字符串匹配替换为句法短语嵌入余弦相似度，使CHAIR可扩展至非MSCOCO的大词汇表数据集（如O365）；该方法可复用于任何需要跨域幻觉评估的工作。
3. **faithfulness-informativeness权衡的量化**：本文同时报告CHAIRi/FaithScore与Coverage/Objects/Facts，避免"只优化单一指标"的陷阱；建议后续幻觉工作也同步报告两者。
4. **推理时约束的策略边界**：发现强制grounded captioning仅在推理阶段带来边际收益，提示未来工作应聚焦于训练阶段或解码阶段的根本性改进，而非依赖输出格式约束。
5. **OOV数据集的必要性**： Objects365作为MSCOCO的超集，能有效暴露数据泄露问题；建议所有幻觉评测报告至少一个O365结果。

## 关键术语表
**Object Hallucination**：LVLM在生成描述中提及图像中不存在的物体或属性的现象。
**Referring Expressions (RE)**：给定文本描述预测对应bbox，或给定bbox生成描述的互逆定位任务。
**Grounded Captioning (GC)**：生成包含bbox坐标交错的图像描述，使每个提及物体都对应明确空间位置。
**POPE**：基于是/否问答的LVLM幻觉评测基准，按random/popular/adversarial三类否定问题统计准确率。
**CHAIR / CHAIRi**：通过匹配caption中的物体词与标注物体集合来计算幻觉比例，CHAIRi为物体级别均值，CHAIRs为图像级别比例。
**FaithScore**：先由LLM将caption抽取为原子事实语句，再由VQA模型验证每条事实正确性的比例。
**CHAIR-MEN**：本文提出的CHAIR语义扩展版，用Sentence-BERT嵌入+余弦相似度替代字符串匹配，支持大词汇表OOV评估。
**LVLM**：Large Vision-Language Model，将预训练LLM与图像编码器对齐后支持多模态理解的模型。

## 可复现要素
- **数据集**：MSCOCO（公开）、Objects365（公开）、RefCOCO（公开）、Visual Genome（公开）、Flickr30k-Entities（公开）、VQAv2（公开）、LLaVA-DETAILED（公开）。
- **代码**：论文未声明开源仓库，但附录提供完整训练/inference细节；通常此类工作在github上会有补充材料，需查阅作者主页。
- **权重**：使用开源LLM（Vicuna 1.5 7B、Llama-3 8B、Phi-3-mini）与CLIP ViT-L/14，**无额外发布训练好的LVLM权重**。
- **关键超参**：LoRA r=64, α=128；对齐模块learning rate=2e-4（训练mix阶段）/1e-3（预训练阶段）；batch size=16/32/64（按LLM）；repetition penalty=1.15；bounding box 2位小数精度；CHAIR-MEN阈值t₁=0.73, t₂=0.78。
