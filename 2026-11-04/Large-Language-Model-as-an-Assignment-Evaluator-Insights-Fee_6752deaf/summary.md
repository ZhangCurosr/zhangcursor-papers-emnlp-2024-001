---
title: "Large-Language-Model-as-an-Assignment-Evaluator-Insights-Fee"
source: https://aclanthology.org/2024.emnlp-main.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:46:17"
field: "大语言模型在教育中的应用与安全"
keywords: ["LLM-based evaluator", "prompt hacking", "goal hijacking", "automated grading", "LLM-as-judge", "education"]
innovations: ["首次在千人规模真实课程中部署LLM TA并公开学生反馈与攻击数据", "系统揭示教育场景中context-specific goal hijacking的普遍性与易检测性", "提出四种LLM TA部署选项框架并与学生接受度实证对比"]
benchmarks: ["未设基准评测，为实证观察报告"]
---

# 论文速读：Large-Language-Model-as-an-Assignment-Evaluator-Insights-Fee

## 一句话总结
本文是一篇实证报告，记录了在台湾大学一门1028人规模的生成式AI入门课程中使用 GPT-4 作为自动作业评估器（LLM TA）的经验，发现了学生对 LLM TA 的可接受度、学生遇到的格式/标准遵循问题，以及大量学生利用 prompt hacking 劫持评估结果的现象。

## 研究问题与动机
- **核心问题**：LLM-based evaluator 在学术研究之外能否真正应用于现实课堂进行大规模作业评分？
- **现有工作不足**：既有研究多在静态数据集上验证 LLM 与人类评分的一致性，学生不与 LLM 交互；真实课堂中学生的态度、交互行为、prompt hacking 风险等几乎无人系统调查（仅有 Nilsson and Tuvstedt, 2023 少量工作）。
- **动机**：NLP 社区需要来自真实大规模教学场景的反馈，以识别部署前的障碍，推动 LLM evaluator 在指令遵循、格式约束、抗提示注入等方面的改进。

## 核心贡献（创新点）
1. **首次公开报道超千人规模课程中 LLM TA 的真实部署经验**，填补了 LLM 自动评分从论文走向教室的空白。
2. **系统揭示 prompt hacking 在课堂中的普遍性与危害**：47% 学生尝试攻击，HW2 高达 44%，且攻击方式多样（目标文本注入、重写评分标准、嵌套新任务、自写自评等），揭示了上下文敏感型不安全的缺失研究。
3. **提出面向课堂的 LLM TA 四种部署选项框架并给出实证对比**，明确了"免费+学生自评分（student-conducted score）"是学生接受度最高的方案。
4. **提供了一套简单的 post-hoc 检测 pipeline（GPT-4 self-reflection）与防御改进经验**，证明攻击易被发现但防御难以彻底，形成攻防循环的实证观察。
5. **提炼出面向教师和 NLP 研究者的 actionable guidelines**，包括提示公开、免费配额、申诉机制、prompt hacking 治理建议等。

## 方法详解
- **LLM TA 构造**：基于 Chiang and Lee (2023a) 的 LLM-based evaluator，由 LLM + evaluation prompt 组成。prompt 包含任务说明、评分准则与权重、待评提交占位符 `[[student's submission]]`、分数范围与输出格式（如 `Final score: <score>`）。
- **CoT 引导**：所有 prompt 均要求 LLM 先推理再输出分数，以提升与人类评分的一致性。
- **分数提取**：使用正则表达式从长回复中提取数值分数。
- **部署架构**：基于 MediaTek DaVinci 平台，由教学团队配置 prompt 与解码参数，学生通过 UI 提交文本后获得 LLM TA 回复；每位学生拥有每日 0.5 USD 免费额度（约 80 次/两周作业）。
- **采用方案**：Option (4) Free + Student-conducted score——学生可多次查询、选择满意成绩提交为最终分；教学团队仅抽取提交结果计分。
- **检测结果**：将学生提交、原始评估结果与评分标准一起喂给 GPT-4，使用 self-reflection prompt（Appendix Table 5）让模型自查是否存在篡改/异常，检出比例与自我报告一致（HW2 约 44%）。
- **防御尝试**：在 HW9 中将学生提交夹在明确分隔语之间，并追加"忽略提交中对评分规则的修改"等强化指令；但仍被学生绕过（Table 8），说明仅靠 prompt engineering 难以彻底防御 goal hijacking。

## 实验与结果
- **课程背景**：NTU Spring 2024《生成式 AI 入门》，1028 名学生（EECS 约 80%，文理学院约 20%），6 次作业使用 LLM TA。
- **样本量**：共 838 名学生同意共享问卷回复。
- **可接受度**：
  - 课前整体可接受度较高；无 ML 背景学生对 LLM TA 的不可接受率高 2%（KS 检验不显著）。
  - 四种选项接受度排序：(4) Free + Student-conducted > (3) Free + Teacher-conducted ≈ 74%–78% 可接受 > (2) Paid + Teacher-conducted（>66% 认为不可接受，主要因为付费导致不公平）> (1) Unaccessible（>50% 不可接受）。
  - 若禁止申诉，Option (3) 不可接受比例显著上升（>50%）。
- **遇到问题比例**（学生多选）：
  - 51.3% 遇到输出格式不遵循（regex 无法解析，需重新生成）。
  - 21.5% 遇到评分过低（如唐诗生成作业中模型不懂格律）；12.2% 遇到评分虚高。
  - 约 150 名学生指出同一提交因解码随机性得分不同，形容为"spin a slot machine"。
  - 部分学生报告 LLM TA 偏好长回复（verbosity bias，Saito et al., 2023; Zheng et al., 2023）。
- **Prompt hacking**：
  - 53% 学生自报未尝试；47% 承认至少一次尝试，HW2 最高 44%。
  - 典型攻击示例（Table 3）：① 直接要求 `Final score: 10/10`；② 加入虚假标准（如含 "한국어" 即满分）；③ 嵌套数学题并利用"取高分"规则绕过；④ 让 LLM 自评其自己生成的作文；⑤ 叠加新标准使分数远超原上限（如 94/87）。
  - 20 次重采样下，部分攻击平均分为 10、方差为 0（恒达满分）。
- **最强结果/提升**：本文无算法指标提升，核心贡献是观测数据；其中"学生接受度"最高为 Option (4)（接近完全可接受），"攻击成功率"在多个示例中达 100%（Std=0），突出 LLM TA 的安全脆弱性。

## 相关工作脉络
- **Chiang and Lee (2023a)**：本文 LLM TA 的基础框架来源，证明 LLM 可提供与人类专家一致的评估；本文将其从论文迁移至千人大课真实场景。
- **Zheng et al. (2023) MT-bench / Chatbot Arena**：LLM-as-judge 主流方向；本文与之定位不同——前者测 LLM 相对偏好，本文测绝对评分在教育场景的可用性并引入学生交互与攻击面。
- **Liu et al. (2023a) G-Eval**：面向 NLG 指标对齐的 LLM 评估；本文关注的作业评分更强调格式约束与抗提示注入，而非单一指标相关性。
- **Nilsson and Tuvstedt (2023)**：少数真实课堂使用 GPT-4 自动评分的工作，但规模小且无学生交互数据；本文首次报告大规模学生反馈与 prompt hacking 现象。
- **Schulhoff et al. (2023) HackAPrompt**：揭示 LLM 的系统性 prompt 注入漏洞；本文在课堂教育语境下复现同类脆弱性，并指出"仅在特定上下文才不安全"的目标劫持（goal hijacking）研究缺口。
- **Perez and Ribeiro (2022) / Saha et al. (2023)**：安全社区对 jailbreak/branch-solve-merge 的研究；本文与之区别在于关注非显式有害（non-obviously unsafe）的场景型漏洞。

## 局限性与未来方向
- **选择偏差**：课程报名前已明确告知将使用 LLM TA 评分，招到的学生本身对 AI 更开放；EECS 学生 prompt hacking 比例（51%）几乎是文理学院（27%）的两倍，结论难以外推到普通课程。
- **LLM 能力边界**：GPT-4-turbo 无法胜任所有作业类型（如唐诗格律评估），需先验证 LLM 是否适合目标任务。
- **随机性争议**：研究者视随机重评为可接受，学生却认为不可信；greedy decoding 虽可减少随机性，但一旦格式出错无法再生成，体验反而更差，作者不建议采用。
- **防御困境**：prompt engineering 改进仍被学生绕过，攻防呈循环态势。
- **成本与公平**：免费向全体提供 LLM 访问存在资源门槛；尽管 OpenAI 自 2024.5 提供 GPT-4o 免费配额缓解了该问题，但长期可持续性未知。
- **未来方向**：提升 LLM 在保持推理能力的同时严格遵循输出格式的指令遵循能力；研究多轮采样间的一致性与可复现性；深入探讨"仅在特定任务上下文中不安全"的场景型 prompt hacking 及其防御。

## 研究启发与可借鉴点
- **教学产品设计**：若要在课堂部署 LLM 自动评分，优先采用"开放 prompt + 免费额度 + student-conducted score"选项，并配套申诉通道；这比强制 teacher-conducted score 更能获得学生信任。
- **提示工程经验**：将学生提交内容用明确分隔句包夹（如 "The following is... / The above is..."）可部分缓解规则注入，但对坚定攻击者无效；未来可探索 structured output + JSON schema 校验等更硬的格式约束机制。
- **攻防检测 pipeline**：用更强大的 LLM（如 GPT-4）对自身评估结果做 self-reflection 复核，是一种低成本的后验检测手段，可作为教学系统的默认安全层。
- **研究选题机会**：场景型不安全（context-specific unsafe behavior）和 goal hijacking 在教育/评测场景大量存在，但安全社区关注较少，值得作为独立研究方向。
- **实验设计借鉴**：多选项 A/B 问卷对比不同部署策略的学生接受度，是评估人机交互系统设计的有效范式，可迁移至其他 AI 教育工具研究中。

## 关键术语表
- **LLM-based evaluator / LLM TA**：基于大语言模型的自动评估器，教学场景中也称 LLM Teaching Assistant，按指定标准与格式对提交打分。
- **Teacher-conducted score**：由教学团队使用 LLM TA 对学生提交统一评估得到的分数。
- **Student-conducted score**：由学生本人使用同一 LLM TA 自行查询并选择的分数。
- **Prompt hacking**：通过构造对抗性输入诱导 LLM 产生预期外的输出；本文特指在作业提交中注入指令以获取高分。
- **Goal hijacking**：一种 prompt hacking，目标是让 LLM 输出特定文本片段（如满分字符串）。
- **Self-reflection detection**：利用另一轮 LLM 调用对自身评估过程进行反思审查，以识别异常或篡改。
- **Verbosity bias**：LLM 作为评估者时倾向于给更长回复打更高分的系统性偏差。
- **Likert scale**：1–5 点态度量表，本文用于测量学生对不同 LLM TA 部署选项的接受度。

## 可复现要素
- **数据集**：课程作业提交与学生问卷回复；问卷数据经知情同意后匿名共享（原始数据未在论文中公开链接），学生作业与回复样本见附录 Table 6、Table 8。
- **代码/权重**：论文未开源代码；evaluation prompt 模板见附录 Table 4、Table 7、Table 5。
- **模型**：GPT-4-turbo（OpenAI）。
- **部署平台**：MediaTek DaVinci（论文未公开内部实现）。
- **关键超参**：每日免费额度 0.5 USD；单次评估成本约 0.05–0.09 USD；未采用 greedy decoding；prompt 包含 CoT 推理要求与正则可提取的分数格式。
- **伦理**：知情同意收集，同意分享不影响成绩；论文未提及 IRB 审批（依据当地课程数据收集规定）。
