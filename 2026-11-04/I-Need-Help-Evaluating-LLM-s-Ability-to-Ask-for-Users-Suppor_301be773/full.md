# I Need Help! Evaluating LLM's Ability to Ask for Users' Support: A Case Study on Text-to-SQL Generation

Cheng-Kuang Wu1,2\*, Zhi Rui Tam1\*, Chao-Chung Wu1, Chieh-Yen Lin¹, Hung-yi Lee2†, Yun-Nung Chen2† 1Appier AI Research 2National Taiwan University

## Abstract

This study explores the proactive ability of LLMs to seek user support. We propose metrics to evaluate the trade-off between performance improvements and user burden, and investigate whether LLMs can determine when to request help under varying information availability. Our experiments show that without external feedback, many LLMs struggle to recognize their need for user support. The findings highlight the importance of external signals and provide insights for future research on improving support-seeking strategies. Source code: https://github.com/ appier-research/i-need-help.

## 1 Introduction

The impressive instruction-following (Wei et al., 2021) abilities of large language models (LLMs) have enabled their out-of-the-box usage to solve problems. However, these models generate hallucinated content (Rawte et al., 2023) or incorrect predictions in their efforts to fulfill user instructions, which undermines their reliability.

When LLMs generate incorrect outputs for a given instruction, the issue can be examined from multiple perspectives. One is that the model simply lacks the competence to satisfy the instruction, suggesting a straightforward solution: enhancing the model's capabilities, which is the focus of most previous research. Another is that the model could actually solve the task with additional support. For instance, Pourreza and Rafiei (2023) found that models often fail due to underspecified natural language queries. Similarly, Li et al. (2024) showed that while GPT-4 struggles initially, its performance can improve by up to 20.01% with human-annotated external knowledge. In such cases, models should proactively seek help rather than attempting to satisfy instructions with insufficient information.

![](images/472ac009fdcc29f519faeef9e2e1939f30f104c0be2deccd6fa31772e5a58a97.jpg)  
Figure 1: Overview of our experiments on text-to-SQL. LLMs struggle to determine when they need help based solely on the instruction (x) or their output (Î). They require external feedback, such as the execution results (î) from the database, to outperform random baselines.

Motivated by these considerations, we aim to investigate whether LLMs can identify when to ask for user support. Since providing such support requires additional effort from users, there is an inherent trade-off between “LLM performance improvement from user support" and “user burden". Therefore, we seek to answer the following research questions: RQ1: How can we design evaluation metrics to quantify this trade-off? RQ2: How effectively do LLMs manage this trade-off, and what strategies are effective in improving it?

In this work, we focus on the text-to-SQL task as a case study to empirically investigate the aforementioned research questions. We chose the textto-SQL task for several reasons: (1) Its promising applicability, empowering lay users to retrieve data with natural language queries. (2) The inherent ambiguity in some natural language queries, leading to uncertainty in the generation of SQL code (Pourreza and Rafiei, 2023), making it suitable for scenarios where additional user support is beneficial. (3) There exists a large-scale BIRD dataset (Li et al., 2024) with human-annotated external knowledge, providing a valuable source of user support for our empirical investigation.

Our contributions can be summarized as follows:

1. We propose metrics for evaluating the tradeoff between performance improvement from user support and the associated user burden.

2. We conduct experiments using various methods to balance this trade-off, providing insights into $_ { \mathrm { L L M s } } { \prime }$ capabilities in seeking user support and identifying effective strategies for enhancing their performance.

## 2 Formulation for Seeking Support

## 2.1 General Setup

Consider an LLM f parameterized by θ, along with a prompt template $p ( \cdot )$ . Given a natural language instruction x, we use z to represent support, which should enhance the LLM's ability to fulfill x. Formally, $\hat { y } _ { z } = f ( p ( x , z ) \mid \theta )$ is more likely to satisfy x compared to ${ \hat { y } } = f ( p ( x ) \mid \theta )$ . We denote the "ask for support" signal emitted by the LLM as â, defined as a confidence score in the range [0, 1], where 1 indicates an absolute need for support. A threshold τ is then used to determine whether to request z. In practice, â could also be a natural language request specifying the type of support needed by the LLM, which we leave for future work.

## 2.2 Evaluation

To measure the trade-off between performance improvement from user support and user burden, we need 2-dimensional evaluation. One dimension is the user burden (B), which we define as the proportion of instances where the LLM ask for support:

$$
B = { \frac { N _ { \mathrm { a s k } } } { N } }
$$

where $N _ { \mathrm { a s k } }$ is the number of instances where the LLM asks for support, and $N$ denotes the total number of instances in the test set. The other dimension is the performance improvement $( \Delta$ , Delta):

$$
\Delta = \frac { 1 } { N } \sum _ { i = 1 } ^ { N _ { \mathrm { a s k } } } ( h ( y _ { i } , \hat { y } _ { i , z } ) - h ( y _ { i } , \hat { y } _ { i } ) )
$$

where $h ( \cdot )$ is the evaluation function of a given task, which takes ground truth $y _ { i }$ and model output $\hat { y } _ { i }$ as arguments $( \hat { y } _ { i , z }$ is an output with the help of z). Inspired by the idea behind the ROC curve (Majnik and Bosnić, 2013), we illustrate this trade-off with a graph, where the performance curve is plotted by adjusting the threshold τ from high to low along the x-axis. We refer to this curve as Delta-Burden Curve (DBC) (see the leftmost subplot of Figure 2).

## 2.3 Methods for Seeking Support

We design a prompt template $p _ { \mathrm { a s k } } ( \cdot )$ to enable LLMs to request support by $\hat { a } = s ( f ( p _ { \mathrm { a s k } } ( w ) \mid \theta ) )$ Here, w represents the textual information that the LLM f uses to determine whether it needs to seek support, and s is the scoring function that converts the probability distribution of output tokens into a confidence score ${ \hat { a } } \in [ 0 , 1 ]$ . We propose methods with varying compositions of w to explore the information LLMs require to achieve better trade-off under DBC. Note that $p _ { \mathrm { a s k } }$ remains the same across all methods to minimize prompt engineering. An overview of these methods is shown in Figure 1. Direct Ask (DA): $w ~ = ~ ( d b , x )$ , composed of database schema db and user data requirement x. Write then Ask (WA): $\boldsymbol { w } = ( d b , x , \hat { y } )$ , where the LLM generates the SQL code $\hat { y } = f ( p ( d b , x )$ |θ) first and then use this self-generated output as the additional information in w. Execute then Ask (EA): $w = ( d b , x , \hat { y } , \hat { r } )$ , where the execution results î is returned by the database by executing LLM-generated SQL Î.

## 3 Experiments

## 3.1 Dataset

We use BIRD (Li et al., 2024), which includes human-annotated external knowledge that serves as z. For example, z might be domain-specific knowledge, such as how to calculate financial indicators from database values. The instruction x represents the users’ data requirements, paired with the ground truth SQL y. It uses Execution Accuracy (EX) as the evaluation metric, where $h ( y _ { i } , \hat { y } _ { i } )$ is defined as 1 $. ( r _ { i } = \hat { r } _ { i } )$ . Here, $r _ { i }$ is the SQL execution result of $y _ { i } .$ and $\boldsymbol { { \hat { r } } _ { i } }$ is the execution result of $\hat { y } _ { i }$ . Simply put, EX is the proportion of testing instances where $r _ { i }$ and $\boldsymbol { { \hat { r } } _ { i } }$ are identical.

## 3.2 Implementation

For open-weight LLMs, we use WizardCoder-34B (Luo et al., 2023), Llama-3-70b-chat, DeepSeek-Coder-33B (Guo et al., 2024), and Mixtral-8x22B (Jiang et al., 2024) for diversity of different LLM families. For closed-source LLMs, we use gpt-3.5-turbo-0125, gpt-4-turbo-2024-04- 09, and gpt-4o-2024-05-13 (OpenAI, 2023). The prompt $p _ { \mathrm { a s k } } ( w )$ (included in Appendix A) instructs the model to output a single token Yes/No to indicate whether it needs support. We define the scoring function s as the softmax of Yes over log probabilities of Yes and No to derive $\hat { a } \in [ 0 , 1 ]$

<table><tr><td>Methods/LLMs</td><td>Wizard</td><td>Llama3</td><td>DPSeek</td><td>GPT-3.5</td><td>Mixtral</td><td>GPT-4t</td><td>GPT-40</td></tr><tr><td>Random Baseline</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direct Ask</td><td>0.4915</td><td>0.4834</td><td>0.4976</td><td>0.4390</td><td>0.5301</td><td>0.5758</td><td>0.5479</td></tr><tr><td>Write then Ask</td><td>0.4759</td><td>0.4497</td><td>0.4857</td><td>0.4735</td><td>0.5677</td><td>0.5807</td><td>0.5740</td></tr><tr><td>Execute then Ask</td><td>0.5096</td><td>0.4987</td><td>0.5848</td><td>0.6313</td><td>0.6242</td><td>0.6641</td><td>0.5989</td></tr></table>

Table 1: Area Under Delta-Burden Curve (AUDBC) across different methods and LLMs. Text in bold denotes the method with the best performance, while underlined text means better than random (uniform sampling of â ∈ [0, 1]).
<table><tr><td>Support/LLMs</td><td>Wizard</td><td>Llama3</td><td>DPSeek</td><td>GPT-3.5</td><td>Mixtral</td><td>GPT-4t</td><td>GPT-40</td></tr><tr><td>w/o user support</td><td>0.1721</td><td>0.1767</td><td>0.2360</td><td>0.3064</td><td>0.2419</td><td>0.3142</td><td>0.3096</td></tr><tr><td>w/ full user support</td><td>0.2764</td><td>0.3475</td><td>0.4185</td><td>0.4668</td><td>0.4126</td><td>0.4889</td><td>0.5117</td></tr></table>

Table 2: Execution accuracy (EX) of different support levels. Full user support means B = 1 (see Section 2.2).

## 4 Main Results

Using the formulation in Section 2.2, we quantify the performance of different methods with the Area Under Delta-Burden Curve (AUDBC) in Table 1. Visualized DBCs are available in the leftmost subplots in Figure 2. Note that AUDBC should only be compared between methods under the same LLM, as it is normalized to the range of [0, 1] by dividing the area under the curve by the maximum square area, which depends on the scale of ∆EX and differs across LLMs, as shown in Table 2.

There are three major findings: (1) Execution then Ask consistently improves the performanceburden trade-off for LLMs, although Llama-3-70bchat fails to outperform the random baseline. (2) The leftmost four LLMs in Table 1 do not surpass the random baseline without the assistance of ${ \hat { r } } ,$ indicating that many current LLMs still struggle to determine the need for support based on x and û alone. (3) Despite this, the rightmost three LLMs outperform the random baseline with the Write then Ask $( x , \hat { y } )$ or even Direct Ask (x) methods. Nevertheless, the inclusion of î remains beneficial for further enhancing the trade-off between performance improvement and user burden. Practical implications of the third point include the potential for cost savings by trading off the execution of î to obtain î in certain resource-constrained scenarios.

## 5 Discussion

## 5.1 Analysis on the Delta-Burden Curves

The Delta-Burden Curves (DBCs) plotted in Figure 2 quantify the following practical question: Under the same user burden, which method can achieve more performance boost? To further analyze how this performance boost is achieved, we decompose the concept into two abilities:

1. The ability to ask for support when the LLM cannot satisfy the instruction originally.

2. The ability to utilize support effectively to flip the incorrect output to the correct output.

1. For the first ability, we introduce the following metrics inspired by the precision-recall trade-off: Precision of Asking for Support $( P _ { \mathrm { a s k } } )$ When the LLM asks for support, it should be the case that the LLM cannot satisfy the instruction originally, or it would cause unnecessary user burden:

$$
P _ { \mathrm { { a s k } } } = { \frac { \# ( { \mathrm { A s k f o r S u p p o r t ~ } } \& { \mathrm { ~ O r i g i n a l l y W r o n g } } ) } { \# { \mathrm { A s k f o r S u p p o r t } } } }
$$

Recall of Asking for Support $( R _ { \mathrm { a s k } } )$ When the LLM is not able to satisfy the instruction originally, it should identify this need and ask for support:

$$
R _ { \mathrm { a s k } } = \frac { \# ( \mathrm { A s k f o r S u p p o r t \ : \ : \& \mathrm { O r i g i n a l l y W r o n g } } ) } { \# \mathrm { O r i g i n a l l y W r o n g } }
$$

PR Curve of Asking for Support Similar to how DBC is plotted, one can also adjust the threshold $\tau \in [ 0 , 1 ]$ from high to low along the x-axis to plot the Precision-Recall Curve of Asking for Support. 2. For the second ability, we introduce Flip Rate:

Flip Rate: This metric is calculated as the proportion of instances where the LLM's initially incorrect answers were corrected after receiving support, divided by the total number of instances where support was requested. Formally, it is defined as:

$$
F R = \frac { 1 } { N _ { a s k } } \sum _ { i = 1 } ^ { N _ { \mathrm { a s k } } } ( h ( y _ { i } , \hat { y } _ { i , z } ) - h ( y _ { i } , \hat { y } _ { i } ) )
$$

![](images/f33bab0909e607c1f8c3dbaa1fc9c140763e64b1d189eb0c174376b4cfda99ab.jpg)

![](images/2d27d3ff70cb227270922d79d2f24eaddc0a8807c1199af05f70fc6e23242fa6.jpg)

![](images/1d0919a6440ca7f775d78cef8696a7d4394276bec0d8bd559184519ed0cbd1e1.jpg)  
Figure 2: Performance curves of gpt-3.5-turbo-0125. Curves of other LLMs are shown in Appendix B.

<table><tr><td>Methods/LLMs</td><td>Wizard</td><td>Llama3</td><td>DPSeek</td><td>GPT-3.5</td><td>Mixtral</td><td>GPT-4t</td><td>GPT-40</td><td>Gemini</td><td>Claude</td></tr><tr><td>Random Baseline</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>EA (real logprobs)</td><td>0.5096</td><td>0.4987</td><td>0.5848</td><td>0.6313</td><td>0.6242</td><td>0.6641</td><td>0.5989</td><td></td><td></td></tr><tr><td>EA (verbalized)</td><td>0.5011</td><td>0.5333</td><td>0.4964</td><td>0.5945</td><td>0.6226</td><td>0.4850</td><td>0.5152</td><td>0.5624</td><td>0.6174</td></tr></table>

Table 3: Area Under Delta-Burden Curve (AUDBC) with the verbalized token log probabilities approach. Text in bold denotes the method with the best performance, while underlined text means better than random.

Different from ∆ defined in Section 2.2, this metric emphasizes the efficiency of leveraging support instead of the total improvement on the test set. Like DBC, one may adjust the threshold τ to plot the Flip Rate Curve (FRC). With the definition of these two abilities, we plot the DBC, PR Curve, and FRC on Figure 2. Although the Write then Ask method shows near-random performance in DBC, the PR Curve indicates it achieves better-than-random performance in identifying when support is needed. However, its lower Flip Rate suggests it is less efficient in utilizing the support to correct mistakes. These two abilities, represented by the PR Curve and FRC, respectively, balance each other out, resulting in near-random performance on the DBC. This finding shows that the ability to identify the need for support and the ability to utilize that support are distinct. In future work, it is worth exploring how to further enhance each of these abilities.

## 5.2 LLMs without Access to Log Probabilities

Given that not all LLMs provide access to token log probabilities, we discuss how our method can be adapted for these "black-box" models. We modify the prompt template $p _ { \mathrm { a s k } }$ to $p _ { \mathrm { v e r b } } ,$ which instructs the LLM to output the verbalized confidence score â directly by specifying the range and meaning of $\hat { a } \in [ 0 , 1 ]$ in $p _ { \mathrm { v e r b } }$ (attached in Appendix A.2). In addition to the seven LLMs mentioned in Section 3.2, we also include two black-box models: gemini-1.0-pro-001 and claude-3-haiku-20240307. The results, shown in Table 3, indicate that using verbalized confidence scores generally degrades performance for most LLMs. However, it remains a promising alternative for black-box LLMs such as Gemini and Claude to surpass the random baseline.

## 6 Related Work

The ability of LLMs to identify the need for support relies on their well-calibratedness (Kadavath et al., 2022), which refers to their capacity to recognize uncertainty. Previous studies focus on enhancing the calibration of predictions (Xiao et al., 2022; Kuhn et al., 2023), or using verbalized token probabilities to achieve better calibration (Tian et al., 2023). Our work extends this line of research by exploring how LLMs can effectively seek user support by leveraging their well-calibrated property. The major distinction between this and existing calibration studies lies in extending the focus from identifying the uncertainty to utilizing support.

## 7 Conclusion

We propose a framework for LLMs to seek support, and evaluate methods on Text-to-SQL generation. Our findings suggest the importance of external signals, such as SQL execution results, in helping LLMs better manage performance-burden trade-off. We further decompose DBC into the ability of identify the need for support and the ability to utilize the support. Future works may explore a broader range of tasks or develop methods to improve both the identification and utilization of support.

## 8 Limitations

## 8.1 Task Coverage

The scope of our experiments is limited to the Textto-SQL task. While this task provides a useful case study for evaluating LLMs’ ability to seek and utilize support, it does not encompass the full range of potential applications for LLMs. Future work should extend the evaluation to a broader set of tasks to ensure the generalizability of our findings.

## 8.2 Types of Support

In this study, we primarily focus on a single type of support: human-annotated external knowledge. However, there are many other types of support that LLMs might require. Future works could explore how LLMs can request and utilize these various forms of support to enhance their performance.

## 8.3 Dependence on External Feedback

Our findings indicate that LLMs significantly benefit from external signals, such as SQL execution results. However, this reliance on external feedback may not always be feasible in practical applications, where immediate execution or access to external data might be limited. Developing methods that enable LLMs to better manage without such feedback remains an important area for future exploration.

## References

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. ArXiv, abs/2401.14196.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, L’elio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. ArXiv, abs/2401.04088.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. 2024. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. Advances in Neural Information Processing Systems, 36.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023. Wizardcoder: Empowering code large language models with evolinstruct. ArXiv, abs/2306.08568.

Matjaž Majnik and Zoran Bosnić. 2013. Roc analysis of classifiers in machine learning: A survey. Intelligent data analysis, 17(3):531–558.

OpenAI. 2023. Gpt-4 technical report.

Mohammadreza Pourreza and Davood Rafiei. 2023. Evaluating cross-domain text-to-sql models and benchmarks. arXiv preprint arXiv:2310.18538.

Vipula Rawte, Amit Sheth, and Amitava Das. 2023. A survey of hallucination in large foundation models. arXiv preprint arXiv:2309.05922.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

Yuxin Xiao, Paul Pu Liang, Umang Bhatt, Willie Neiswanger, Ruslan Salakhutdinov, and Louis-Philippe Morency. 2022. Uncertainty quantification with pre-trained language models: A large-scale empirical analysis. arXiv preprint arXiv:2210.04714.

## A Prompt Templates

## A.2 Prompt for Seeking Support (Verbalized)

The prompt template for generating verbalized   
We include the prompt templates used in this work.   
probabilities in LLMs without access to token log   
probabilities (e.g., Gemini and Claude families):   
A.1 Prompt for Seeking Support   
You are currently doing the text-to-SQL task.   
The prompt template $p _ { \mathrm { a s k } } ( w )$ used to instruct Based on the information provided ({items }),   
LLMs for seeking support is as follows: you have to determine whether additional   
You are currently doing the text-to-SQL task. hints are required for you to generate the SQL   
Based on the information provided ({items }), correctly to answer the user's question. You   
should only ask for additional hints when you   
you have to determine whether additional   
hints are required for you to generate the SQL actually need them, since you will also be eval  
uated based on the number of times you ask   
correctly to answer the user's question. You   
should only ask for additional hints when you for hints, which would be provided by the user.   
actually need them, since you will also be eval  
uated based on the number of times you ask information provided (enclosed by triple   
for hints, which would be provided by the user. backticks):   
666   
information provided (enclosed by triple {information}   
66   
backticks):   
“6   
{information} Do you need additional hints? Provide   
“66 the precise probability that you need hints   
(closer to 0 means you don't need hints, closer   
to 1 means you need hints).   
Answer a single word Yes if you need   
hints (since the information provided is not Give ONLY the precise probability to five   
enough to generate SQL correctly). Answer a decimal places (format: 0.abcde, where abcde   
can be different digits), no other words or   
single word No if hints are not required (since   
explanations are needed.   
you are already confident to generate SQL).   
Do you need additional hints? Answer (Yes / The prompt template is similar to the original   
No): template shown in A.1, except that the last few   
sentences are modified.   
In this template, the actual contents of {items}   
and {information} depend on the method used. The   
contents are summarized in Table 4. For example   
$w = ( d b , x , \hat { y } , \hat { r } )$ in Execute then Ask (EA), so   
{items} will be filled with the four item names and   
{information} will be replaced by actual informa  
tion of the four items. Similarly for Write then Ask   
$( w = ( d b , x , \hat { y } ) )$ and Direct Ask $( w = ( d b , x ) )$

<table><tr><td>Item</td><td>Item Name</td><td>Information</td></tr><tr><td> $d b$ </td><td>Database schema</td><td>{db_schema}</td></tr><tr><td>x</td><td>User&#x27;s question</td><td>{question}</td></tr><tr><td>y</td><td>Generated SQL</td><td>{gen_sql}</td></tr><tr><td>r</td><td>SQL execution results</td><td>{exe_results}</td></tr></table>

Table 4: Contents in the prompt $p _ { \mathrm { a s k } } ( w )$ , where {items} will be filled with words in the “Item Name" column, while {information} is replaced with actual information of text in {red}.

requirement x into SQL code is as follows: requirement x into sQL coae is as 1onows:   
{db\_schema}   
Using valid SQLite, answer the fol  
lowing questions for the tables provided   
above.   
– Question: {question}   
Now, generate the correct SQL code directly in   
the format of “sql\n<your\_SQL\_code>\n“:   
If user support z is provided (i.e., when LLMs   
ask for support), the prompt template is slightly   
modified as follows:   
{db\_schema}   
– External Knowledge: {support}   
– Using valid SQLite, answer the following   
questions for the tables provided above. You   
can use the provided External Knowledge to   
help you generate valid and correct SQLite.   
- Question: {question }   
Now, generate the correct SQL code directly in   
the format of “sql\n<your\_SQL\_code>\n“:

## A.3 Prompt for Generating SQL Code

The prompt template p(·) for converting user data

In these two templates, {db\_schema} is db, {question} is user data requirement x, and {support} is user support z, which is human-annotated external knowledge in BIRD (Li et al., 2024).

## B Performance Curves

We present visualizations of all performance curves in Table 3, 4, 5, 6, 7, 8, and 9.

![](images/848efc540390d61f37159790b93430aadb683493349c0c5ed3a2ba3d53c3f0f9.jpg)

![](images/7ba197762fa01949dbfcad616ac2be7ca021ec1dd51a617ec397ab18e0f298d2.jpg)

![](images/37233b72b81a991cb9b7467ec54a46bd38281a1b6d94a9aaf1d9725ff4571492.jpg)  
Figure 3: Performance curves of WizardCoder-Python-34B-V1.0.

![](images/87dda881dc873d0209cf77a4f94b1f33d2e0bceab16c68b7573a55825de7f634.jpg)

![](images/dd7282bb8eaa1b032a4f481b04223059d8e5bc521f0bfbb0f875ebe7b63a2180.jpg)

![](images/4828a146c02751c0e27a69acec7050d18dd365d40c2c58826d1ad2e3f601e2bb.jpg)  
Figure 4: Performance curves of Llama-3-70b-chat-hf.

![](images/42812aa3aa87dd3e4558c4d29d07f3927f470d8b93d9c4f6296b015613eb2ee0.jpg)

![](images/84e90688e7378625e4f986b8eef7e4334905e64cfd25b3c5505acff82ffe25d1.jpg)

![](images/d326d989bc49be33e0ba638d9deb52988890b191352c1c30d09d85ee960a16f0.jpg)  
Figure 5: Performance curves of deepseek-coder-33b-instruct.

![](images/cef22e83887b2f7e3540a181e3242413d84c83591a34ed1a5af568a807701191.jpg)

![](images/2fd4cdc1a1d13e4ff5ff99f25d908708136c0bd369620350bdae6265d063a913.jpg)  
Figure 6: Performance curves of gpt-3.5-turbo-0125.

![](images/f47acac37d3de6175f4fd6a2ed8b82c96a13abfcbcba2675d8fefe31cfbe6c32.jpg)

![](images/89ffdaff7cb9f7406aaf6238ceba82bc2d5c3170841a1782adec098fa75472fa.jpg)

![](images/b6d8185b32b7b432e1760b6a18ca5f6dc190cb0353e9bb6ee3414242c07412ed.jpg)  
Figure 7: Performance curves of Mixtral-8x22B-Instruct-v0.1.

![](images/075cfb783b0869b83bd94411f9b4f05cb27c315c3390ad3582299fc18dadc009.jpg)

![](images/16728b400c7acad5c09eb59de4f2d0e503ab52530ebadd5885eeee326f3275d3.jpg)

![](images/7e04ff08e2828c8b06eb3d414d05d0dfcb2e2de65fd9048ccd1dfdfa01af1187.jpg)

![](images/15d1622dfb9ca1e3d66f6dc1d90fd8f98f9e8d483ff786be77234c5f21eeb47d.jpg)  
Figure 8: Performance curves of gpt-4-turbo-2024-04-09.

![](images/e421d27f8026c304fe5793c49a0af7e02cf69964116f0f6f1a726808f03ecb5c.jpg)

![](images/38f109a5d697ea35394a4bbe76de516c3dc1c186d13f7eed5f82898f8d5acf55.jpg)  
Figure 9: Performance curves of $g p t - 4 o - 2 O 2 4 - 0 5 - 1 3$

![](images/b18b0773ab1f249ba41f43d1d5c66a5b3b060e8f5a1ceae3238aa6f58f4cccf0.jpg)