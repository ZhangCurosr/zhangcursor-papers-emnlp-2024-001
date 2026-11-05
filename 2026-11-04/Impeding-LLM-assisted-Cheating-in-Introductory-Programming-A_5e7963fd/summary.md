---
title: "Impeding-LLM-assisted-Cheating-in-Introductory-Programming-A"
source: https://aclanthology.org/2024.emnlp-main.27.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:10:59"
---

# 论文速读：Impeding-LLM-assisted-Cheating-in-Introductory-Programming-A

## 一句话总结
本文首次系统研究在黑盒学术场景下，如何通过对抗性提示词扰动（Adversarial Perturbation）削弱主流 LLM 对计算机入门编程作业的代码生成能力；结合自动化评估与 30 人用户田野实验证实，组合扰动可使 LLM 生成的正确率平均下降约 77%，且未被察觉的细微扰动能有效阻断学生直接套用 ChatGPT 答案的行为。

## 研究问题与动机
- **LLM 作弊威胁迫在眉睫**：ChatGPT、GitHub Copilot 等工业级 LLM 能快速生成高质量代码，对高校 CS1/CS2 入门编程课程的学术诚信构成直接冲击。
- **现有反作弊路径的瓶颈**：主流方案依赖检测器识别 LLM 生成内容，但假阳性/假阴性难以避免，且 educators 无法直接控制或限制工业 LLM 的能力上限。
- **提示词扰动策略尚属空白**：尽管 NLP 领域已有对抗扰动研究，但针对“修改作业题目以降低 LLM 可用性、同时保持人类学生可理解性”的黑盒学术场景，缺乏系统性探索。
- **需要验证实际教学场景中的有效性**：扰动是否能在学生真实使用 ChatGPT 时保持降分效果？学生能否察觉并逆向还原？这些问题尚无实证回答。

## 核心贡献（创新点）
1. **首次将对抗提示词扰动系统化引入学术反作弊场景**：不同于以往仅关注模型鲁棒性 benchmark 的研究，本文直接面向教育痛点，提出以最小文本改动抑制 LLM 代码生成可用性的黑盒干预框架。
2. **设计 10 种细粒度扰动策略并引入 SHAP 引导的关键单元选择**：覆盖字符、token、句子、同义词替换、Unicode 伪装及随机字符串注入等；利用 CodeRL 作为代理模型计算 Shapley 值，以数据驱动方式定位最具破坏性的扰动位置，兼顾降分效果与修改代价。
3. **提出面向教育场景的 Efficacy 度量与三分层评估流水线**：定义基于正确率相对下降的 efficacy 公式（优先惩罚低基线题目的大幅衰减），并通过“基线性能评估 → 批量自动化扰动测试 → 30 人用户田野实验”三步验证扰动对真实作弊行为的遏制潜力。
4. **揭示“可检测性”与“实际降分效果”的解耦现象**：用户研究表明，高变动扰动即使被学生察觉，也难以被有效逆转；而细微扰动（如 token 替换、字符移除）在未察觉状态下能保留更高的实际 efficacy，为“隐形反作弊”提供了实证依据。

## 方法详解
- **研究架构（三步法）**：
  - Step 1：在 Arizona 大学 CS1/CS2 题库上测量 5 个 LLM 的基线正确率。
  - Step 2：设计并批量应用 10 种黑盒扰动，计算每个扰动的 Efficacy 与 Edit Distance。
  - Step 3：招募已修读课程的学生，使用 ChatGPT 完成扰动后的作业并提交聊天记录，分析可检测性、逆向策略与最终 efficacy。
- **SHAP 引导的扰动单元选择**：以 CodeRL 为 surrogate model，计算 prompt 中各 token 的 Shapley 值并排序；对 Character/Token/Sentence 级扰动分别选取 Top-K 单元执行修改，确保在最少文本变动下实现最大性能下降。
- **扰动策略分类**：
  - *核心扰动（7 种）*：Character (remove)、Token (remove/synonym)、Sentences (rephrase/remove)、Prompt (unicode)、Random (insert)。
  - *探索扰动（3 种）*：Tokens (synonym, 仅替换 Top-ranked 出现位置)、Prompt (unicode, 全量替换)、Random (replace, 手动定位文件名/函数名/类名并替换为随机字符串，后端自动还原以兼容 autograder)。
- **Efficacy 公式与优化倾向**：
  $$Effi cacy = max \Bigg \{ 0 , 100 \times \frac { S_{no\_prtrb} - S_{prtrb} }{ S_{no\_prtrb} } \Bigg \}$$
  公式设计倾向于在相同绝对降幅下，优先奖励对低基线题目的降分（例如从
