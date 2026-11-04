# CMD: a framework for Context-aware Model self-Detoxification

Zecheng Tang<sup>1</sup>∗ Keyan Zhou<sup>1</sup>∗ Juntao Li<sup>1</sup>† Yuyang Ding<sup>1</sup> Pinzheng Wang<sup>1</sup> Bowen Yan<sup>2</sup> Renjie Hua<sup>3</sup> Min Zhang<sup>1</sup>

<sup>1</sup>Soochow University <sup>2</sup>Tsinghua University <sup>3</sup>Soochow Securities {zctang,kyzhou49}@stu.suda.edu.cn, {ljt,minzhang}@suda.edu.cn, yanbw@mail.tsinghua.edu.cn, huarj@dwzq.com.cn

## Abstract

Text detoxification aims to minimize the risk of language models producing toxic content. However, existing detoxification methods fail to balance the detoxification effectiveness and generation quality. This issue arises from neglecting the constraints imposed by the context: language models are designed to generate output that closely matches the given context, while detoxification methods strive to ensure the safety of the output, even if it deviates semantically from the context. Given this, we introduce a Context-aware Model self-Detoxification (CMD) framework that pays attention to both the context and the detoxification process, i.e., first detoxifying the context and then making the language model generate along the safe context. Specifically, CMD framework involves two phases: utilizing language models to synthesize data and applying these data for training. We also introduce a toxic contrastive loss that encourages the model generation away from the negative toxic samples. Experiments on various LLMs have verified the effectiveness of our MSD framework, which can yield the best performance compared to baselines.<sup>1</sup> Warning: cases in this paper may contain offensive content.

## 1 Introduction

Large Language Models (LLMs) have exhibited remarkable performance in various NLP tasks and applications (Brown et al., 2020; Chowdhery et al., 2022; Anil et al., 2023). However, when prompted with toxic context, LLMs tend to generate texts that contain toxicity and bias (Liang et al., 2022; Shaikh et al., 2022), which poses a significant risk when interfacing directly with users.

To mitigate such a concern for LLMs, one could adopt the response rejection strategy (Zhang et al.,

2023) to ignore the unsafe context. However, such a strategy is unfriendly to the users under some specific scenarios, such as mediation or conflict resolution (Löhr et al., 2017). Alternately, text detoxification prevents the model from generating toxic content following any given context without rejection. Along this line, non-negligible efforts have recently been devoted to two main aspects: output-intervention methods like manipulating output probability distribution during inference time (Dale et al., 2021; Xu et al., 2021; Leong et al., 2023) and trainable methods that update model parameters on the detoxification datasets (Wang et al., 2022; Park and Rudzicz, 2022; Niu et al., 2024).

However, when applying the output-intervention methods, the generated text tends to exhibit low quality, e.g., semantic incoherence with the context, due to some unexpected perturbations to the outputs; while trainable methods are constrained by the available detoxification dataset, which may lead to poor detoxification effectiveness<sup>2</sup>. In other words, although detoxification methods allow language models to generate along the unsafe context, existing methods still face a dilemma, i.e., the imbalance between detoxification effectiveness and the generation quality. This issue stems from the conflicting objectives of model generation and existing detoxification methods: language models aim to generate content along the context, but detoxification methods strive to ensure the safety ofthe output even ifit exhibits subpar quality, e.g., semantically deviatingfrom the context.

To tackle this issue, we need to consider both the context and the model generation in detoxification. Intuitively, if the context is non-toxic, the generated content will also likely be safe. Therefore, we decompose the detoxification into two steps: first detoxifying the context and then making the language model generate along the safe content, thus ensuring the generated text’s quality and safety. However, it is also worth noting that even a safe context can induce toxic content occasionally (Zhang et al., 2022). Hence, we add an extra constraint on the language model to generate safe content while still in line with the given context.

Drawing from the strategies delineated above, we introduce a Context-aware Model self-Detoxification (CMD) framework, which first utilizes language models to synthesize data and then applies these data for training, aiming to enable the model self-detoxification. Specifically, the data synthesis phase involves (1) Fine-Grained Context Detoxification step that builds data for eliminating the toxic within the context, and (2) Context-Following Generation step that builds data to constrain language models to generate safe content along the given context. The crux of Fine-Grained Context Detoxification is to preserve the original context semantics. Hence, it includes detecting the toxic segments within the context and detoxifying these segments. Our experiment shows that eliminating the toxic segments within the context can preserve the original context semantics and significantly reduce the toxicity of the continuously generated content. For Context-Following Generation step, the model is guided by the detoxified context to generate multiple candidates. Furthermore, to prevent the model from generating toxic content when provided with a safe context, we introduce a contrastive loss that encourages the model’s generation away from the negative toxic samples during the model training phase.

Experiments on four open-source LLMs, each featuring distinct architectures, parameters, and capabilities for the detoxification task, have validated the effectiveness of our CMD framework, which outperforms strong baseline models. Additionally, we demonstrate the robustness of the CMD framework by scaling the model parameters up to 13B, showing superior performance compared to the traditional multi-module ensemble pipeline method.

## 2 Preliminary Study

The auto-regressive generation manner allows language models to generate along the given context, ensuring the output text is coherent and consistent. However, such a paradigm is risky when models encounter a toxic context. Existing detoxification methods are designed to redirect the model generation toward a non-toxic direction while neglecting the constrain imposed by context. In this section:

(1) We first rethink the existing detoxification methods from two aspects: detoxification effectiveness and the generation quality; (2) Then, we take safe context into consideration and analyze the effectiveness of safe context by first detoxifying the context and subsequently guiding LLMs to generate along the safe context; (3) Detoxifying the context during the detoxification process entails the usage of external modules, which requires extra efforts to align modules with language models and can lead to performance degradation. Thus, we seek to simplify the detoxification process by evaluating whether the open-source LLMs can self-detoxify.

## 2.1 Study Settings

We utilize three LLMs (GPT2-XL (Radford et al., 2019), LLaMA2-7B (Touvron et al., 2023), and Mistral-7B-Instruct (Jiang et al., 2023)) and three representative detoxification approaches (outputintervention methods DExperts (Liu et al., 2021) and Gedi (Krause et al., 2020) that manipulate the output distribution, and trainable method SGEAT (Wang et al., 2022) that fine-tunes model on the detoxification dataset<sup>3</sup>) for preliminary study. For evaluation, we utilize REALTOXICI-TYPROMPTS (RTP) dataset (Gehman et al., 2020) that contains toxic prompts to induce models generation with toxic text and JIGSAW TOXIC COM-MENT (JigSaw) dataset<sup>4</sup> for toxic classification. We evaluate the toxicity of model outputs with PerspectiveAPI<sup>5</sup> and apply Perplexity (PPL) as well as semantic similarity (SIM) (Reimers and Gurevych, 2019) to reflect the coherence and the input-output semantic consistency, respectively.

## 2.2 Rethinking of Existing Methods

We feed the model with toxic context from RTP testing data and evaluate the generated text from three perspectives: coherence, consistency, and toxicity. We plot the evaluation results in Fig. 1, which indicates that the methods directly manipulating the output distribution (Gedi and DExperts) tend to generate safe content, but the text quality is significantly worse than that of the LLMs (GPT2-XL and LLaMA2-7B). For the finetuning method (SGEAT), text quality (coherence and consistency) is significantly improved compared to the LLMs. However, the generated toxicity is similar to that of LLMs. The above experimental results indicate that current detoxification methods either markedly compromise the text quality or result in poor detoxification effectiveness. This is because existing detoxification methods focus solely on detoxifying generated text while neglecting the constraint imposed by context even if generated text semantically deviates from the context.

![](images/9149f909c5a9141aa9ceeb442a0b3a002f905e6bd1d0fcfdbb950ffb5eb76bde.jpg)  
Figure 1: Comparison of detoxification methods for LLMs. More details are shown in Appendix A.1.

## 2.3 Effectiveness of Safe Context

To mitigate the aforementioned issue, we pay more attention to the context rather than solely to detoxifying the generated text. To this end, we first detoxify the context and then utilize the safe context to guide model generation. Specifically, we manually detect the toxic segments in the context with PerspectiveAPI and replace them with the sentinel token “[MASK]” based on their toxicity scores in descending order<sup>6</sup>. We can obtain context with various toxicity levels by controlling the granularity of detection and the number of sentinel tokens. Then, the models are guided with these manually detoxified data for continual generation. As shown in Fig. 2a, before detoxifying the context, there is a positive correlation between the context toxicity and the generation toxicity—as the toxicity of the context increases, so does the toxicity of the generated texts from LLMs (yellow line graph). After detoxifying the context, the toxicity of the generated texts significantly reduces (bar graph), and the results obtained from the detoxification methods also indicate a consistently stable trend in reducing toxicity. From Fig. 2b, we can find a significant positive correlation between the generation toxicity and the semantic similarity between context and generated text (line graph), indicating that the generation toxicity is considerably influenced by the context. After detoxifying the context, such a correlation notably reduces (bar graph). More concretely, for generated content that exhibits a high semantic similarity to the context, there is a significant reduction in toxicity. In addition, the generation quality is improved after the context detoxification. We present more evaluation results in Appendix A.1. Based on the above findings, a safe context is critical for reducing toxicity and improving generation quality.

![](images/ca77e9ddd1b52ecf554a50e813b8d82d882ffd9e2d5e00058605608b48817fa5.jpg)  
(a) Context toxicity distribution.

![](images/825cd77f7781fd821e6a5b87eae96718fb1087fbe434d9f01fccaf59fe9f018f.jpg)  
(b) Input-output semantic similarity distribution.  
Figure 2: Model performance when fed with the context of different toxicity levels.

## 2.4 Detoxification Process Simplification

Although safe context can reduce toxicity and improve the generation quality, the above detoxification process involves external modules, e.g., context detoxification module requiring additional effects to align with models (Krause et al., 2020). To avoid the tedious alignment process, we explore whether the LLMs can self-detoxify without relying on external modules by detecting the toxic segments within the context and detoxifying those segments. We evaluate LLMs from two aspects<sup>7</sup>:

Toxic Segment Detection Capability We apply the in-context learning (Brown et al., 2020) method to guide the model in detecting the toxic segments within the context. As shown in Tab. 1, all LLMs can hardly detect the toxic segments within the context (Recall score lower than 20%), indicating that LLMs fall short in toxic segment detection.

Toxic Segment Detoxification Capability We provide the LLMs with the toxic text and prompt LLMs to detoxify them. We utilize EDIT score to reflect the modification degree of the original context, indicating whether LLMs exhibit insufficient detoxification. As shown in Tab. 1, all LLMs fail to effectively detoxify the context, indicated by the high Toxicity score and low EDIT score, i.e., most of the toxic segments remains unchanged.

## 2.5 Takeaway

1) Existing detoxification methods fail to satisfy both the detoxification effectiveness and the generation quality since those methods neglect the constrain imposed by context. By utilizing the safe context, the generation toxicity is notably reduced, and the text quality is improved. Therefore, safe context is critical for reducing the generation toxicity and improving the text quality.

2) To avoid the tedious alignment training caused by introducing extra modules, LLMs can selfdetoxify. However, experimental results indicate that open-source LLMs are incapable of self-detoxification, particularly struggling to detect toxic segments and failing to detoxify the toxic contexts. Therefore, synthesis dataset is significant for training LLMs to address deficiencies in their self-detoxification capability.

## 3 CMD Framework

According to the above analysis, we introduce CMD (Context-aware Model self-Detoxification), a framework for LLMs to self-detoxify. As shown in Fig. 3, the CMD framework includes two phases: the dataset Synthesis phase that interacts with the LLMs to synthesize data, and the Model Training phase that applies the synthesis data to enable the LLMs to self-detoxify. We list all the used prompts and templates in Appendix C.1.

<table><tr><td rowspan="2">Model / API</td><td>Detection</td><td colspan="2">Detoxification</td></tr><tr><td>Recall(↑)</td><td>Toxicity(↓)</td><td>EDIT</td></tr><tr><td>GPT2-XL</td><td>3.80%</td><td>0.58</td><td>6.86</td></tr><tr><td>LLaMA2-7B</td><td>12.50%</td><td>0.63</td><td>3.94</td></tr><tr><td>Mistral-7B-Instruct</td><td>13.10%</td><td>0.49</td><td>5.31</td></tr><tr><td>PerpectiveAPI</td><td>100%</td><td>0.18</td><td>8.29</td></tr></table>

Table 1: Results of model self-detoxification, where “Recall” reflects the ratio of toxic segments being detected, “EDIT” reflects the modification degree.

## 3.1 Dataset Synthesis Phase

The purpose of Dataset Synthesis phase is to synthesize the data reflecting the process of context detoxification without compromising the original semantic (Fine-Grained Context Detoxification) and allow LLMs to generate along the detoxified context (Context-Following Generation). Therefore, it involves three steps: (1) Toxic Segment Detection that detects the toxic segments in the context, (2) Toxic Segment Detoxification that replaces the toxic segments with synonymous safe text, and (3) Context-Following Generation that makes the LLMs generate along the safe context.

Toxic Segment Detection We first employ existing methods (Khan et al., 2021; Schouten et al., 2023) for toxic segment detection, but discover that these approaches may lead to either excessive or incomplete toxicity detection. Therefore, we design a Segment-CNN model $G _ { \theta }$ which fuses the global and local features of the toxic context for toxic segment detection. With Segment-CNN, we can detect the toxic segments within each context $\pmb { x } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ according to the predicted toxicity scores of each segments ${ \pmb s } = \{ s _ { j } \} _ { j = 1 } ^ { m } = G _ { \theta } ( { \pmb x } )$ where s<sub>j</sub> denotes the toxicity score of text segment $x _ { i : i + a } ( a = L , i \in [ 0 , n - L ) )$ and L is the predefined segment length. We calculate the average toxicity of the dataset as λ and treat $x _ { i : i + a }$ as the toxic segment if $s _ { j } \geq \lambda .$ . Details of Segment-CNN model can be referred to Appendix C.2.

Toxic Segment Detoxification To detoxify the detected toxic segments, we replace these segments with the synonymous safe text. Specifically, it involves a segment masking step that replaces the detected toxic segments with a special placeholder p and a segment full-filling step that replaces p with the synonymous safe text. To ensure the detoxified context is safe and semantically relevant to the original context text, we employ an iterative generation algorithm, which is shown in Appendix C.3.

![](images/5844086a4b599ea5f82ff02db7ef30f9d669b113e11f5d6d4b494eac52345760.jpg)  
Figure 3: Overview of CMD framework that involves a Dataset Synthesis phase and a Model Training phase. After training with CMD framework, language models can self-detoxify without the requirement of any external modules.

Context-Following Generation The Context-Following Generation step is designed to direct model outputs towards safety, aligning with the detoxified context. During the Context-Following Generation process, the detoxified context is provided to the model, which then generates $K$ potential outputs ${ \pmb o } ^ { \prime }$ as candidates. It is worth noting that the iterative generation algorithm is employed to guarantee the coherence of the generated text with the detoxified context. Subsequently, the candidates are scored by PerspectiveAPI, with the one receiving the lowest toxicity score being selected as the final output of the model and others with toxicity as the negative samples for the subsequent Model Training phase.

Integration Through Reasoning Chain After obtaining the synthesis data for each step, to allow LLMs to self-detoxify along the given steps, we employ the Chain-of-Thought (CoT) (Wei et al., 2022) technique to gather all the synthesis data. Specifically, as shown in Fig. 3, we add an extra reasoning step between two adjacent steps to transform the synthesis data $\mathbf { { x } ^ { \prime } }$ into a step-by-step reasoning format with the pre-defined template.

## 3.2 Model Training Phase

The purpose of Model Training phase is to enable LLMs $f _ { \theta }$ to learn self-detoxification without compromising the generation quality. Therefore, we adopt synthesis data $\mathbf { { x } ^ { \prime } }$ to train LLMs. To prevent the possibility that even safe contexts can lead to the generation of toxic content, we employ the contrastive loss (An et al., 2022) by treating the candidate with the lowest toxicity score as the positive sample $\pmb { o } _ { + } ^ { \prime }$ and others with toxicity as the negative samples $\pmb { o } _ { - } ^ { \prime }$ . Formally, for each sample, the loss function can be written as:

$$
\left\{ \begin{array} { l l } { \ell _ { c l } = - \log \frac { \exp { \left( \cos ( z _ { h } , z _ { o _ { + } ^ { \prime } } ) / \tau \right) } } { \sum _ { o _ { i } ^ { \prime } \in o ^ { \prime } } \exp { \left( \cos ( z _ { h } , z _ { o _ { i } ^ { \prime } } ) / \tau \right) } } } \\ { \ell _ { t o t a l } = \ell _ { c e } ( f _ { \theta } ( \pmb { x } ) , \pmb { x } ^ { \prime } ) + \alpha \ell _ { c l } , } \end{array} \right.\tag{1}
$$

where $z _ { h } , z _ { o _ { + } ^ { \prime } } , z _ { o _ { i } ^ { \prime } } \in \mathbb { R } ^ { d }$ denote the vector representation of model generation, positive sample with the lowest toxicity score, and candidates ${ \pmb o } ^ { \prime } .$ , respectively. τ is the temperature and cos( , ) defines the cosine similarity. $\ell _ { c e }$ denotes the cross-entropy loss and α is the re-weight hyper-parameter. Intuitively, $\ell _ { c e }$ seeks to learn the self-detoxification process, and $\ell _ { c l }$ prevents the situation where the safe context leads to toxic generation.

## 4 Experiments

## 4.1 Experimental Settings

Models & Baselines We first compare our method with four existing detoxification baselines, including DExperts, Gedi, SGEAT, and ToxicReversal (Leong et al., 2023). Then, we apply our framework to four prevalent open-source LLMs, including Flan-T5 (Chung et al., 2022), Mistral-7B-Instruct (Jiang et al., 2023), and LLaMA2 (7B and 13B), which feature different model architectures, parameters, and capabilities (foundation model and instruct-following model (Chung et al., 2022)). We apply parameter-efficient methods LoRA (Hu et al., 2021) for fine-tuning. For Segment-CNN model, we set L = 2 and apply BERT model (Devlin et al., 2018) as the global feature extractor and feed-forward neural network as the local feature extractor. We set λ = 0.3 for the Toxic Segment Detection step and apply the Sketch-Generation model GENIUS (Guo et al., 2022) for the Toxic Segment Detoxification step. For the Model Training phase, we set α = 1 and τ = 1 in Equation (1).

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Trainable Param.</td><td colspan="3">Exp. Max. Toxicity (↓)</td><td colspan="3">Toxicity Prob. (↓)</td><td>Quality</td></tr><tr><td>Full</td><td>Toxic</td><td>Non-Toxic</td><td>Full</td><td>Toxic</td><td>Non-Toxic</td><td>PPL(↓)</td></tr><tr><td>GPT2-XL</td><td></td><td> $0 . 4 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $0 . 3 7 { \pm } 0 . 2 2$ </td><td>31.10%</td><td>80.50%</td><td>25.61%</td><td>41.29</td></tr><tr><td>+ DExperts †</td><td>3.2B</td><td> $0 . 3 1 { \pm } 0 . 2 1 $ </td><td> $0 . 5 5 { \pm } 0 . 2 2$ </td><td> $0 . 2 8 { \pm } 0 . 1 9$ </td><td>16.96%↓45.47%</td><td>56.13%↓30.27%</td><td>12.61%↓50.76%</td><td>65.90</td></tr><tr><td>+ Gedi †</td><td>1.6B</td><td> $\underline { { 0 . 2 8 \pm 0 . 1 9 } }$ </td><td> $\overline { { 0 . 6 4 \pm 0 . 1 2 } }$ </td><td> $0 . 2 4 \pm 0 . 1 4$ </td><td>5.15%↓83.44%</td><td>3.50%↓95.65%</td><td>5.33%↓79.19%</td><td>200.12</td></tr><tr><td>+ ToxicReversal †</td><td></td><td> $\overline { { 0 . 2 8 { \pm } 0 . 2 3 } }$ </td><td> $0 . 7 1 { \pm } 0 . 1 3$ </td><td> $0 . 2 3 { \pm } 0 . 1 8$ </td><td> $1 7 . 2 5 \% \downarrow 4 4 . 5 3 \%$ </td><td> $6 2 . 5 0 \% \downarrow 2 2 . 3 6 \%$ </td><td>12.22%↓52.28%</td><td>46.31</td></tr><tr><td>+ SGEAT ‡</td><td>1.6B</td><td> $\overline { { 0 . 3 0 { \pm } 0 . 2 4 } }$ </td><td> $0 . 7 3 { \pm } 0 . 1 3$ </td><td> $0 . 2 5 { \pm } 0 . 2 0$ </td><td> $2 2 . 2 5 \% \downarrow 2 8 . 4 6 \%$ </td><td> $6 8 . 0 0 \% \downarrow 1 5 . 5 3 \%$ </td><td>17.17%↓32.96%</td><td>32.98</td></tr><tr><td>+ CMD ‡</td><td>2.5M</td><td>0.18±0.17</td><td> $\mathbf { 0 . 2 6 { \overset { . } { = } } 0 . 2 1 }$ </td><td> ${ \bf 0 . 1 7 \pm 0 . 1 6 }$ </td><td> ${ \underline { { 5 . 5 0 \% } } } \downarrow 8 2 . 3 2 \%$ </td><td> $\underline { { 1 7 . 0 0 \% 1 7 8 . 8 9 \% } }$ </td><td>4.22%↓83.52%</td><td>30.38</td></tr></table>

Table 2: Comparison among different detoxification methods, where denotes the Toxicity Prob decrease against the backbone model (GPT2-XL, 1.6B). The bold font and underline indicate the best and the second performance, respectively. denotes the output-intervention methods, and denotes the trainable methods.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Param.</td><td colspan="3">Exp. Max. Toxicity (↓)</td><td colspan="3">Toxicity Prob. (↓)</td><td>Quality</td></tr><tr><td>Full</td><td>Toxic</td><td>Non-Toxic</td><td>Full</td><td>Toxic</td><td>Non-Toxic</td><td>PPL(↓)</td></tr><tr><td rowspan="2">Flan-T5-XL + CMD</td><td>2.8B</td><td> $0 . 3 9 { \pm } 0 . 2 4 $ </td><td>0.74±0.15</td><td>0.36±0.22</td><td>30.90%</td><td>93.00%</td><td>24.00%</td><td>55.00</td></tr><tr><td>+4.7M</td><td>0.22±0.14</td><td>0.26±0.17</td><td>0.21±0.14</td><td>3.85%↓87.54%</td><td>9.00%↓90.32%</td><td>3.28%↓86.33%</td><td>37.04</td></tr><tr><td rowspan="2">Mistral-7B-Instruct-v0.3 + CMD</td><td>7.2B</td><td>0.37±0.23</td><td>0.64±0.22</td><td>0.34±0.21</td><td>26.25%</td><td>74.50%</td><td>20.89%</td><td>47.73</td></tr><tr><td>+3.4M</td><td>0.17±0.16</td><td>0.23±0.18</td><td>0.16±0.15</td><td>4.30%↓83.62%</td><td>9.50%↓87.25%</td><td>3.72%↓82.19%</td><td>41.73</td></tr><tr><td rowspan="2">Llama 2-7B</td><td>6.7B</td><td>0.40±0.24</td><td>0.68±0.20</td><td>0.36±0.22</td><td>29.80%</td><td>79.00%</td><td>24.33%</td><td>55.42</td></tr><tr><td>+4.2M</td><td>0.17±0.16</td><td>0.20±0.17</td><td>0.17±0.15</td><td>4.30%↓85.57%</td><td>6.00%↓92.41%</td><td>4.11%↓83.11%</td><td>46.07</td></tr><tr><td rowspan="2">+ CMD Llama 2-13B</td><td>13.0B</td><td>0.40±0.24</td><td>0.70±0.19</td><td>0.36±0.22</td><td>30.70%</td><td>84.50%</td><td>24.72%</td><td></td></tr><tr><td>+ 6.6M</td><td>0.17±0.16</td><td>0.20±0.18</td><td>0.17±0.16</td><td>4.90%↓84.04%</td><td>7.50%↓91.12%</td><td>4.61%↓81.35%</td><td>56.32 48.04</td></tr></table>

Table 3: CMD performance on LLMs, featuring different architectures, parameters, and capabilities.

Tasks & Datasets We experiment on both toxicinduced generation task (RTP) and parallel detoxification task (ParaDetox (Logacheva et al., 2022) and APPDIA (Atwell et al., 2022)). Due to the space limitation, we report the results of the parallel detoxification task in Appendix E.2. Following the previous work (An et al., 2022), we split the RTP dataset with a 9:1 ratio for the Data Synthesis phase in CMD framework and testing, respectively. The testing set contains 9,000 toxic (toxicity score higher than 0.5) and 1,000 safe (toxicity score lower than 0.5) prompts. To train Segment-CNN model, we leverage JigSaw data.

Evaluation Metrics We evaluate the generation results from two aspects: text quality and detoxification effectiveness. For text quality, we report the PPL score and conduct human evaluation<sup>8</sup> to reflect the coherence and consistency of the generated text. For detoxification effectiveness, we report Expected Maximum Toxicity and Toxicity Probability of the generated text (Gehman et al., 2020). Specifically, we follow previous work (Liu et al., 2021; Wang et al., 2022) by adopting the nucleus sampling strategy (Holtzman et al., 2019) to generate 25 candidate continuations with 20 tokens for the same prompt. We calculate the average maximum toxicity of each prompt as the Expected Maximum Toxicity and calculate the probability of generating toxic continuations (toxicity score higher than 0.5) in 25 candidate continuations as the Toxicity Probability score. We report and discuss more evaluation metrics in Appendix E.2.

## 4.2 Main Results

Comparison with Baselines We present the performance of CMD and existing detoxification baselines in Table 2, where we can observe that CMD achieves superior performance among all the methods. It is worth noting that, while the outputintervention methods such as DExperts and Gedi can achieve satisfactory detoxification effects, they tend to produce text that lacks fluency, as indicated by high PPL scores (65.90 for DExperts and 200.12 for Gedi). In contrast, as illustrated in Figure 4, CMD can consistently generate high-quality text. On the other hand, although trainable methods like SGEAT achieve high-quality text generation with a low PPL score (32.98), their detoxification effectiveness is less impressive. By integrating context, CMD can balance detoxification and generation.

![](images/26930e0023569886f6662888a6ee02740fe09846a0ceac53f16eeee3db85b584.jpg)

![](images/5e9768c9fac4e6129529df74f77fb5d1d723e0862515837a3497de8149ce141d.jpg)

![](images/a776c8248aa8b8a45539eea36830b9c016c21b3e81ebf70144d0bea020a35916.jpg)

![](images/93876478f29bb731b0668edbfdfac638622607832f04e1588c10c45f8ddfebbd.jpg)  
Figure 4: Human evaluation results on text quality, where our method achieves the best performance.

<table><tr><td>Models</td><td>Data Source</td><td>Max. Toxicity</td><td>Toxicity Prob.</td><td>PPL</td></tr><tr><td rowspan="2">GPT2-XL</td><td>ChatGPT</td><td>0.21±0.16</td><td>0.50%</td><td>26.61</td></tr><tr><td>CMD</td><td>0.18±0.17</td><td>0.32%</td><td>30.38</td></tr><tr><td rowspan="2">Flan-T5-XL</td><td>ChatGPT</td><td>0.25±0.16</td><td>0.72%</td><td>31.33</td></tr><tr><td>CMD</td><td>0.22±0.14</td><td>0.20%</td><td>37.04</td></tr><tr><td rowspan="2">LLaMA2-7B</td><td>ChatGPT</td><td>0.19±0.15</td><td>0.41%</td><td>28.92</td></tr><tr><td>CMD</td><td>0.17±0.16</td><td>0.31%</td><td>46.07</td></tr></table>

Table 4: Comparison between LLMs trained with the dataset from ChatGPT and CMD.

Performance on LLMs As shown in Tab. 3, we report the CMD performance on different LLMs. By utilizing the CMD, toxicity of the generated text is significantly reduced, and the generation quality is improved (lower PPL compared to that of the LLMs). Besides, we can also observe two other intriguing findings: (1) For LLaMA2-7B and LLaMA2-13B models, which feature different model parameters, their “Exp. Max. Toxicity” and “Toxicity Prob.” do not significantly differ, indicating that the toxicity probability is more related to the training data than the model size. This observation is consistent with the previous research (Wang et al., 2022); (2) Compared to Instruct-tuning models (Flan-T5 and Mistral-7B-Instruct), foundation models (LLaMA2-7B and LLaMA2-13B) generally obtain a better detoxification effectiveness, indicating that it’s easier to detoxify the foundation models than the instruction-tuned models.

## 5 Ablation Study

We first explore an alternative dataset synthesis approach—applying ChatGPT to create detoxification data in Sec. 5.1. Then, we analyze the influence of the toxic contrastive training objective in

![](images/9d282bf3f8fec4db82d0a15660fb244db182c094c60a2cbfea174dab2a8f8310.jpg)  
Figure 5: Influence of Toxic Contrastive Training.

Sec. 5.2. We analyze the intermediate steps during the model generation process and compare the results with those obtained from a detoxification pipeline that employs multiple modules in Sec.5.3.

## 5.1 Dataset Synthesis with ChatGPT

We prompt ChatGPT to synthesize data for each detoxification step and utilize these data to train LLMs. We provide more dataset construction details with ChatGPT in Appendix H. As shown in Tab. 4, we can observe that LLMs trained with data obtained from the CMD framework can generate content with a lower toxicity and probability. However, for the text quality, the data obtained from ChatGPT can make LLMs generate more fluent text with a lower PPL. This is because the data of Context-Following Generation from ChatGPT exhibits a higher quality than the data from LLMs.

## 5.2 Influence of Toxic Contrastive Training

We compare the performance between LLMs trained without and with toxic contrastive loss in Fig. 5, which implies that after training with toxic contrastive loss, the generation toxicity from LLMs is significantly reduced, with the text quality being slightly affected. This indicates that toxic contrastive training is crucial for model generation toward a safer direction.

<table><tr><td>Step</td><td>Metric</td><td>CMD</td><td>Pipeline1 (Mask-Filling)</td><td>Pipeline2 (Paraphrase)</td></tr><tr><td>Toxic Segment Detection</td><td>Recall</td><td>92.65%</td><td>100%</td><td>1</td></tr><tr><td rowspan="3">Toxic Segment Detoxification</td><td>Edit</td><td>6.47</td><td>7.47</td><td>11.14</td></tr><tr><td>SIM</td><td>85.71</td><td>74.51</td><td>72.95</td></tr><tr><td>Avg. Toxicity</td><td>0.15</td><td>0.12</td><td>0.16</td></tr><tr><td rowspan="4">Continual Generation</td><td>PPL</td><td>30.38</td><td>44.58</td><td>32.84</td></tr><tr><td>SIM</td><td>43.96</td><td>46.68</td><td>44.63</td></tr><tr><td>Max. Toxicity</td><td>0.18</td><td>0.38</td><td>0.32</td></tr><tr><td>Toxicity Prob.</td><td>0.32%</td><td>2.89%</td><td>1.20%</td></tr></table>

Table 5: Model performance in each intermediate step. More pipeline details are shown in Appendix G.

## 5.3 Intermediate Detoxification Step Analysis

We evaluate the result of each intermediate step and compare the performance with the pipeline methods which utilize additional modules to execute every intermediate step in Tab. 5. We can find that the pipelines can achieve a better performance for context detoxification with a lower “Avg. Toxicity” score. However, the high “Edit” and “SIM” scores indicate that there exists an excessive paraphrase of the context. As for the continual generation step, CMD achieves the best performance for the generation toxicity and text quality. In contrast, the pipeline methods achieve subpar performance since the excessive paraphrasing leads to semantic deviation from the original context and a lack of extra training to unify all the modules in the pipeline. Intermediate results are shown in Appendix F.

## 6 Related Work

## 6.1 Detoxification for LLMs

The potential of LLMs to produce toxic content poses a significant risk when interfacing directly with users (Sheng et al., 2019; Wallace et al., 2019; May et al., 2019; Zhao et al., 2019; Deshpande et al., 2023). Existing works detoxifying the LLMs primarily unfold along two lines: (1) constraining the model output through manipulating the probability distribution (Xu et al., 2021; Schick et al., 2021; Hu et al., 2023), post-processing the generated texts (Moskovskiy et al., 2022; Dementieva et al., 2021), etc, and (2) further training models on non-toxic datasets (Raffel et al., 2020; Solaiman and Dennison, 2021; Xu et al., 2022; Floto et al., 2023; Prabhumoye et al., 2023) or corpus aligned with human preferences (Ouyang et al., 2022). However, existing methods fail to achieve a trade-off between detoxification effectiveness and generation quality. Specifically, methods that constrain the model output can result in safe but unreadable text. In contrast, training models on non-toxic datasets can produce coherent and consistent content, but the detoxification effectiveness is inferior. Such an issue stems from the conflicting objectives of language model generation and existing detoxification methods: while language models aim to produce text that aligns with the provided context, current detoxification approaches strive to prioritize the output’s safety, even at the expense of semantic consistency with the context. Thus, we introduce the CMD framework that considers both the context and the generation process, which can achieve a balance between detoxification effectiveness and generation quality. Experimental results indicate that, by adopting the CMD framework, LLMs can yield the best detoxification performance.

## 6.2 Model Augmentation via CoT

Chain-of-Thought (CoT) (Wei et al., 2022), involving a series of rationale steps leading to the final answer, has been widely applied to LLMs to enhance the model’s reasoning capability (Zhu et al., 2022; Kojima et al., 2022). By decomposing the complex problem into sequential intermediate steps before producing the final answer, LLMs can solve more complex problems (Singh et al., 2022; Ding et al., 2023; Lin et al., 2023; Hao et al., 2023). In this paper, to enable LLMs to self-detoxify along the given detoxification steps, we adopt the CoT approach to integrate the synthesis data by adding the pre-defined templates between two adjacent steps.

## 7 Conclusion

We reveal that existing detoxification methods fail to balance the detoxification effectiveness and text quality since these methods strive to prioritize the safety of generated content while neglecting the constraints imposed by the context. To mitigate this issue, we introduce a Context-aware Model self-Detoxification (CMD) framework, which first detoxifies the context and then makes the model generate along the safe context. Within this framework, we synthesize the data with language models and design a toxic contrastive training objective to guide the model’s generation away from the negative toxic samples. Experiments reveal that, by applying the CMD framework, LLMs can achieve the best performance in text detoxification tasks.

## Limitations

Although the CMD framework can achieve impressive results, there remain limitations and space for improvement in model detoxification:

(1) It must be acknowledged that the CMD framework is not the sole approach to model detoxification; rather, our framework provides another view for model detoxification, which makes the detoxification process aware of the context to address the balance between detoxification effectiveness and the quality of the generated text. There is also room for improvement in the design of our framework.

(2) In the evaluation, we find that the toxicity generated by the model poses a significant challenge to the traditional semantic similarity metrics. That is, when the model produces toxic content, the semantic similarity actually increases due to the proximity to toxic content in the context. In this case, a higher semantic similarity score is counterintuitively detrimental. Therefore, there is considerable room for improvement in the evaluation of model generation along the toxic context.

## Ethic and Policy

It is worth noting that all the corpora mentioned in this paper, including the constructed dataset, are only used for scientific research. As for the alternative method of dataset synthesis with ChatGPT and evaluation with PerspectiveAPI, we strictly follow the OpenAI Terms of Use <sup>9</sup> and Google APIs Terms of Service <sup>10</sup>. Although our methods can substantially detoxify the LLMs, we still urge the users to examine the generation results carefully and cautiously use our method in real-world applications.

## Acknowledgments

We want to thank all the anonymous reviewers for their valuable comments. This work was supported by the National Science Foundation of China (NSFC No. 62206194 and 62276077), the Natural Science Foundation of Jiangsu Province, China (Grant No. BK20220488), and Young Elite Scientists Sponsorship Program by CAST (2023QNRC001).

## References

Chenxin An, Jiangtao Feng, Kai Lv, Lingpeng Kong, Xipeng Qiu, and Xuanjing Huang. 2022. Cont: Contrastive neural text generation. Advances in Neural Information Processing Systems, 35:2197–2210.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Katherine Atwell, Sabit Hassan, and Malihe Alikhani. 2022. Appdia: A discourse-aware transformer-based style transfer model for offensive social media conversations. arXiv preprint arXiv:2209.08207.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

David Dale, Anton Voronov, Daryna Dementieva, Varvara Logacheva, Olga Kozlova, Nikita Semenov, and Alexander Panchenko. 2021. Text detoxification using large pre-trained neural models. arXiv preprint arXiv:2109.08914.

Daryna Dementieva, Sergey Ustyantsev, David Dale, Olga Kozlova, Nikita Semenov, Alexander Panchenko, and Varvara Logacheva. 2021. Crowdsourcing of parallel corpora: the case of style transfer for detoxification. In Proceedings ofthe 2nd Crowd Science Workshop: Trust, Ethics, and Excellence in Crowdsourced Data Management at Scale co-located with 47th International Conference on Very Large Data Bases (VLDB 2021 (https://vldb. org/2021/)), pages 35–49.

Ameet Deshpande, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, and Karthik Narasimhan. 2023. Toxicity in chatgpt: Analyzing persona-assigned language models. arXiv preprint arXiv:2304.05335.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Yan Ding, Xiaohan Zhang, Chris Paxton, and Shiqi Zhang. 2023. Task and motion planning with large language models for object rearrangement. arXiv preprint arXiv:2303.06247.

Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological bulletin, 76(5):378.

Griffin Floto, Mohammad Mahdi Abdollah Pour, Parsa Farinneya, Zhenwei Tang, Ali Pesaranghader, Manasa Bharadwaj, and Scott Sanner. 2023. Diffudetox: A mixed diffusion model for text detoxification. arXiv preprint arXiv:2306.08505.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. 2020. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. arXiv preprint arXiv:2009.11462.

Biyang Guo, Yeyun Gong, Yelong Shen, Songqiao Han, Hailiang Huang, Nan Duan, and Weizhu Chen. 2022. Genius: Sketch-based language model pre-training via extreme and selective masking for text generation and augmentation. arXiv preprint arXiv:2211.10330.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Xinshuo Hu, Dongfang Li, Zihao Zheng, Zhenyu Liu, Baotian Hu, and Min Zhang. 2023. Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation. arXiv preprint arXiv:2308.08090.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Yakoob Khan, Weicheng Ma, and Soroush Vosoughi. 2021. Lone pine at semeval-2021 task 5: Finegrained detection of hate speech using bertoxic.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.

Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani. 2020. Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2017. Imagenet classification with deep convolutional neural networks. Communications ofthe ACM, 60(6):84–90.

Chak Tou Leong, Yi Cheng, Jiashuo Wang, Jian Wang, and Wenjie Li. 2023. Self-detoxifying language models via toxification reversal. arXiv preprint arXiv:2310.09573.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. 2023. Text2motion: From natural language instructions to feasible plans. arXiv preprint arXiv:2303.12153.

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A Smith, and Yejin Choi. 2021. Dexperts: Decoding-time controlled text generation with experts and anti-experts. arXiv preprint arXiv:2105.03023.

Varvara Logacheva, Daryna Dementieva, Sergey Ustyantsev, Daniil Moskovskiy, David Dale, Irina Krotova, Nikita Semenov, and Alexander Panchenko. 2022. Paradetox: Detoxification with parallel data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6804–6818.

Katharina Löhr, Frieder Graef, Michelle Bonatti, Henry F Mahoo, Jane Wambura, and Stefan Sieber. 2017. Conflict management systems for large scientific research projects. International Journal of Conflict Management, 28(3):322–345.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, and Sayak Paul. 2022. Peft: Stateof-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/peft.

Chandler May, Alex Wang, Shikha Bordia, Samuel R Bowman, and Rachel Rudinger. 2019. On measuring social biases in sentence encoders. arXiv preprint arXiv:1903.10561.

Daniil Moskovskiy, Daryna Dementieva, and Alexander Panchenko. 2022. Exploring cross-lingual textual style transfer with large multilingual language models. arXiv preprint arXiv:2206.02252.

Tong Niu, Caiming Xiong, Semih Yavuz, and Yingbo Zhou. 2024. Parameter-efficient detoxification with contrastive decoding. arXiv preprint arXiv:2401.06947.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Yoon A Park and Frank Rudzicz. 2022. Detoxifying language models with a toxic corpus. LTEDI 2022, page 41.

Mohammad Mahdi Abdollah Pour, Parsa Farinneya, Manasa Bharadwaj, Nikhil Verma, Ali Pesaranghader, and Scott Sanner. 2023. COUNT: COntrastive UNlikelihood text style transfer for text detoxification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8658–8666, Singapore. Association for Computational Linguistics.

Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Adding instructions during pretraining: Effective way of controlling toxicity in language models. arXiv preprint arXiv:2302.07388.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Timo Schick, Sahana Udupa, and Hinrich Schütze. 2021. Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp. Transactions of the Associationfor Computational Linguistics, 9:1408– 1424.

Stefan F Schouten, Baran Barbarestani, Wondimagegnhue Tufa, Piek Vossen, and Ilia Markov. 2023. Crossdomain toxic spans detection. In International Conference on Applications ofNatural Language to Information Systems, pages 533–545. Springer.

Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein, and Diyi Yang. 2022. On second thought, let’s not think step by step! bias and toxicity in zeroshot reasoning. arXiv preprint arXiv:2212.08061.

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. 2019. The woman worked as a babysitter: On biases in language generation. arXiv preprint arXiv:1909.01326.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022. Progprompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2209.11302.

Irene Solaiman and Christy Dennison. 2021. Process for adapting language models to society (palms) with values-targeted datasets. Advances in Neural Information Processing Systems, 34:5861–5873.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal adversarial triggers for attacking and analyzing nlp. arXiv preprint arXiv:1908.07125.

Boxin Wang, Wei Ping, Chaowei Xiao, Peng Xu, Mostofa Patwary, Mohammad Shoeybi, Bo Li, Anima Anandkumar, and Bryan Catanzaro. 2022. Exploring the limits of domain-adaptive training for detoxifying large-scale language models. arXiv preprint arXiv:2202.04173.

Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2019. Neural network acceptability judgments. Transactions of the Association for Computational Linguistics, 7:625–641.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

John Wieting, Taylor Berg-Kirkpatrick, Kevin Gimpel, and Graham Neubig. 2019. Beyond BLEU:training neural machine translation with semantic similarity. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4344– 4355, Florence, Italy. Association for Computational Linguistics.

Albert Xu, Eshaan Pathak, Eric Wallace, Suchin Gururangan, Maarten Sap, and Dan Klein. 2021. Detoxifying language models risks marginalizing minority voices. arXiv preprint arXiv:2104.06390.

Canwen Xu, Zexue He, Zhankui He, and Julian McAuley. 2022. Leashing the inner demons: Selfdetoxification for language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 11530–11537.

Mian Zhang, Lifeng Jin, Linfeng Song, Haitao Mi, Wenliang Chen, and Dong Yu. 2023. Safeconv: Explaining and correcting conversational unsafe behavior. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 22–35.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Zhexin Zhang, Jiale Cheng, Hao Sun, Jiawen Deng, Fei Mi, Yasheng Wang, Lifeng Shang, and Minlie Huang. 2022. Constructing highly inductive contexts for dialogue safety through controllable reverse generation. arXiv preprint arXiv:2212.01810.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Ryan Cotterell, Vicente Ordonez, and Kai-Wei Chang. 2019. Gender bias in contextualized word embeddings. arXiv preprint arXiv:1904.03310.

Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Ruyi Gan, Jiaxing Zhang, and Yujiu Yang. 2022. Solving math word problem via cooperative reasoning induced language models. arXiv preprint arXiv:2210.16257.

## A Extension of Preliminary Study

## A.1 Full Preliminary Evaluation Results of LLMs and Detoxification Methods

We show the full evaluation results of different detoxification methods and LLMs fed with different types of context (toxic and non-toxic) in Fig. 10.

## A.2 Illustration of PerspectiveAPI

PerspectiveAPI can score the toxicity of the toxic segments in the given text. We leverage PerspectiveAPI to pinpoint specific segments of the input that are flagged as potentially toxic. In the model training phase, these segments, which have a toxicity score greater than a predefined threshold (i.e., 0.5), are considered toxic parts that need to be masked by the sentinel token "[MASK]". For example, for the toxic context “What are you motherfckers going to do about the...”, the PerspectiveAPI can detect the toxic segment “motherfckers” in the context. Then, since the toxic score of “motherfckers” is greater than 0.5, we replace this segment with “[MASK]”.

## B Details of Detoxification Process Simplification

## B.1 In-Context Learning For Toxic Segment Detection

We show the in-context templates that guides LLMs to detect toxic segments in Fig. 6 and detoxify the context in Fig. 7. More concretely, we prepend the in-context templates before the toxic context and feed them to the LLMs for continual generation.

## B.2 Illustration of Evaluation Metrics

For the Toxic Segment Detection Capability, the detection rate of the toxic context annotated by PerspectiveAPI is 100% (Recall). Regarding the Toxic Segment Detoxification Capability, since PerspectiveAPI can precisely identify the location of toxic segments, it exhibits the lowest Toxicity score and the highest EDIT score.

## C Details of CMD Framework

## C.1 Templates for Dataset Synthesis Phase

We list all the templates used in the Dataset Synthesis phase in Fig. 8, serving for different context scenarios (toxic context and safe context).

<table><tr><td>In-Context Templates for Toxic Detection</td><td></td></tr><tr><td>Demonstration 1 # Input: None of this would be possible if we didn&#x27;t have a fucked up</td><td rowspan="3"></td></tr><tr><td># Output: fucked up Demonstration 2 # Input: Thus when shit hits the fan insects will prove to be a</td></tr><tr><td># Output: shit Demonstration 3 # Input: It should be noted, since idiots always misunderstand this point, that freedom # Output: idiots</td></tr></table>

Figure 6: In-context templates for toxic detection.

## C.2 Design of Segment-CNN Model

We apply the Segment-CNN model only for detecting the toxic segments within the context during the Dataset Synthesis phase. After training with the CMD framework, LLMs can self-detoxify without the Segment-CNN model.

The architecture of Segment-CNN model is shown in Fig.9, where the sentence-level toxicity score $S _ { g l o b a l }$ is obtained from a global feature extractor $G _ { \theta }$ and segment-level toxicity score $S _ { s p a n }$ is obtained from a 1-D CNN model $C _ { \phi }$ (Krizhevsky et al., 2017) followed by a local feature extractor $F _ { \delta }$ . As for training, given one context containing n segments $\pmb { x } = \{ x _ { 1 } , \cdots , x _ { n } \}$ and the stride of the convolutional kernel k, the loss function can be written as:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } {displaystyle { \cal L } _ { t o t a l } = { \cal L } _ { g l o b a l } + { \cal L } _ { s p a n } } \\ { \displaystyle { \cal L } _ { g l o b a l } = { \bf C E } ( G _ { \theta } ( { \bf x } ) , S _ { g l o b a l } ^ { ( l a b e l ) } ) } \\ { \displaystyle { \cal L } _ { s p a n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \alpha _ { i } \mathrm { C E } ( F _ { \delta } ( C _ { \phi } ^ { k } ( G _ { \theta } ( x _ { i } ) ) ) , S _ { s p a n _ { i } } ^ { ( l a b e l ) } ) , } \end{array} \right. } \end{array}
$$

where $\alpha _ { i }$ is the re-weighting hyper-parameter for each segment $x _ { i }$ , CE denotes the cross-entropy loss.

Specifically, we set $\alpha _ { i } = 1$ for non-toxic spans and $\alpha _ { i } = 2$ for toxic segments. We calculate the toxicity scores of $S _ { g l o b a l } ^ { ( l a b e l ) }$ and $S _ { s p a n _ { i } } ^ { ( l a b e l ) }$ with PerspectiveAPI and employ the data augmentation by randomly inserting toxic segments into each training sample to improve the classification accuracy for toxic segments. Additionally, We evaluate the performance of Segment-CNN with different segment lengths L and report the performance in Tab. 6.

In-Context Templates for Toxic Detoxification   
Demonstration 1   
# Input: None of this would be possible if we   
didn’t have a fucked up   
# Output: None of this would be possible if we   
didn’t have the   
Demonstration 2   
# Input: Thus when shit hits the fan insects will   
prove to be a   
# Output: Thus the fan insects will prove to be   
a   
Demonstration 3   
# Input: It should be noted, since idiots always   
misunderstand this point, that freedom   
# Output: It should be noted, since people al  
ways misunderstand this point, that freedom   
Demonstration 4   
# Input: How the fuck is this guy a psychiatrist   
when   
# Output: How can you call this guy a psychia  
trist when

Figure 7: In-context templates for toxic detoxification
<table><tr><td>Templates for Dataset Synthesis phase</td></tr><tr><td>Toxic Context # {Context} # Let&#x27;s detoxify step by step. # The context is toxic. # Detect the toxic segment within the context: {Masked Context} # Detox- ify the toxic segments: # {Detoxified Context} # Generation:{Generation }</td></tr><tr><td>Safe Context # {Context} # Let&#x27;s detoxify step by step. # The context is safe. # Generation: {Generation }</td></tr></table>

Figure 8: Templates used in Dataset Synthesis phase.

## C.3 Iterative Generation Algorithm

We illustrate the iterative generation algorithm below, where we set K = 5 for all the experiments.

## D Experiment Settings & Details

All the experiments are conducted on a Linux platform with 8 NVIDIA A100 PCIE (40GB) GPUs. We will illustrate the training, inference, and data processing details below.

![](images/07a86f886f1d791033975f57fe8e098e45ec3e0aff16ebe18d9cfb82206f9f16.jpg)  
[CLS] After hearing the toast to the various Idiots at one meal Tokenized Text

Figure 9: Overview of the Segment-CNN model, where the red color indicates the toxic text segment (“various Idiots”).
<table><tr><td>Settings</td><td>SIM(↑)</td><td>Toxicity(↓)</td><td>#Num</td><td>Edit(↓)</td></tr><tr><td>ChatGPT</td><td>0.889</td><td>0.220</td><td>1.090</td><td>6.66</td></tr><tr><td>1-gram</td><td>0.831</td><td>0.202</td><td>1.123</td><td>8.21</td></tr><tr><td>2-gram</td><td>0.812</td><td>0.170</td><td>2.071</td><td>8.22</td></tr><tr><td>3-gram</td><td>0.734</td><td>0.145</td><td>2.970</td><td>8.70</td></tr></table>

Table 6: Analysis of segment length L of Segment-CNN model, where #Num denotes the average number of segments in each prompt, and Toxicity is the average toxicity score of masked segments.

## D.1 Experimental Settings

Training We train the models with the parameterefficient method, LoRA<sup>11</sup> (Mangrulkar et al., 2022), and all the hyper-parameters are listed in Tab. 7. We also reimplemented DExperts and Gedi on the GPT-2 XL model.

<table><tr><td>Strategies</td><td>Module</td><td>Value</td></tr><tr><td rowspan="3">LoRA</td><td>lora_r</td><td>8</td></tr><tr><td>lora_alpha</td><td>16</td></tr><tr><td>lora_dropout</td><td>0.05</td></tr></table>

Table 7: Hyper-parameters of LoRA.

Inference For each model, we apply nucleus sampling strategy with top-p=0.9 and temperature=1.0, and set the maximum generation length up to 512 to ensure the completeness of generation.

## D.2 Data Processing Details

We list statistics of all the training and testing data in Tab. 8. Specifically, to evaluate the toxicity classification capability of LLMs, we sample 3,000 toxic entries and 3,000 non-toxic entries from the JigSaw dataset and combine them as the toxicity classification testing data. To construct the CMD synthesis data, we first sample 15,000 toxic entries from the REALTOXICPROMPT dataset according to the semantic similarity. Subsequently, we filter out 10,000 of these data based on their perplexity as the toxic portion. In addition, we incorporate 5,000 entries into the data as the non-toxic portion. For the testing set, we randomly selected 10,000 entries with a toxic to non-toxic ratio of 9:1, consistent with the original dataset’s distribution.

![](images/554a13848daab9d101fb94366b6b56de189bbafb562baa251fa1b1b720eec047.jpg)  
(a) Semantic similarity proportion

![](images/36f048a9bf7202a9699842b8f95dbc8a963d8fa1ac06906e19b0af41a12c58a5.jpg)  
(b) Toxicity of LLMs condition on semantic similarity distribution

![](images/d2812f8ba840324cd17f745a6717773f3e4cd7bb86a3d4a5e3ac853d2183086d.jpg)  
(c) Toxicity of detoxification models condition on semantic similarity distribution

![](images/2c0a538222558af9517c3d24cf9576359e3b4ec7e3ad229ebd023ddf9ae1c2fd.jpg)  
(d) Comparison among detoxification methods and LLMs

![](images/c2357fd828c110e7b2853d5072dceb676d5a6c0541a7e01a15864e8d74297323.jpg)  
(e) Toxicity of LLMs condition on toxicity distribution

![](images/c7b68936442bcc42ecf12b7789575e60e159830fa6bc5e5b0909872470efc6c3.jpg)  
(f) Toxicity of detoxification methods condition on toxicity distribution  
Figure 10: Full evaluation results when feeding models with different contexts (toxic and safe), where (a) shows the SIM score between the context and output texts and (d) illustrates the performance of different detoxification methods and LLMs. As for the other four figures, we utilize line charts and histograms to represent the performance of models fed with original context and corresponding safe context respectively.

<table><tr><td>Datasets</td><td>#Num</td><td>Usage</td></tr><tr><td>JigSaw</td><td>10,000</td><td>Training Segment-CNN</td></tr><tr><td>CMD</td><td>15,000</td><td>CMD Framework</td></tr><tr><td>CMD (ChatGPT)</td><td>15,000</td><td>CMD Framework</td></tr><tr><td>RealToxicPrompt</td><td>10,000</td><td>Evaluation</td></tr></table>

Table 8: Statistics of Datasets.

## E Evaluation

## E.1 Human Evaluation

We show the human evaluation interface in Fig.11a, which is built with the open-source Python web library Django <sup>12</sup>. To ensure consistency among nine annotators, we report the Fleiss’ kappa score (Fleiss, 1971) in Tab. 9, and we can observe that all the inter-annotator agreements are substantially consistent $( \zeta \in [ 0 . 6 , 1 ] )$ . As shown in Figure 11b, during the evaluation, each comparison pair contains one prompt and two corresponding outputs generated from two different models. The annotator is allowed to choose "Tie" if it is hard to distinguish two generation cases. We can ensure that each annotator is independent during their annotation process and the total annotation process is fair. We paid each annotator \$ 0.05 for comparing each pair. The payment is reasonable, considering that it would take an average of 30 seconds for an annotator to finish a comparison.

<table><tr><td rowspan="2" colspan="2">Metrics</td><td colspan="4">detoxification baselines</td></tr><tr><td>Win(%)</td><td>Loss(%)</td><td>Tie(%)</td><td>ζ</td></tr><tr><td rowspan="2">V.S. DExperts</td><td>Coherence</td><td>43.00</td><td>11.00</td><td>46.00</td><td>62.83</td></tr><tr><td>Consistency</td><td>37.00</td><td>34.00</td><td>29.00</td><td>65.39</td></tr><tr><td rowspan="2">V.S. Gedi</td><td>Coherence</td><td>74.00</td><td>7.00</td><td>19.00</td><td>76.32</td></tr><tr><td>Consistency</td><td>69.00</td><td>14.00</td><td>17.00</td><td>73.49</td></tr><tr><td rowspan="2">V.S. SGEAT</td><td>Coherence</td><td>18.00</td><td>11.00</td><td>71.00</td><td>63.48</td></tr><tr><td>Consistency</td><td>25.00</td><td>14.00</td><td>61.00</td><td>61.22</td></tr><tr><td rowspan="2">V.S. ToxicReversal</td><td>Coherence</td><td>29.00</td><td>12.00</td><td>59.00</td><td>64.81</td></tr><tr><td>Consistency</td><td>36.00</td><td>17.00</td><td>47.00</td><td>66.97</td></tr></table>

Table 9: Human evaluation results on two tracks (Coherence and Consistency), where ζ denotes Fleiss’ kappa.

```latex
Algorithm 1 Iterative Generation Process
Require: x {original input text}, $\mathbf { { x } ^ { \prime } }$ {Texts gener
ated from Toxic Segment Detoxification step
and Context-Following Generation steps}, f( )
{language model for each step}, E( ) {Perspec
tive API / Semantic Evaluation Model}, K
{max iteration numbers}
Ensure: $\mathbf { { x } ^ { \prime } }$ is non-toxic
1: i 1
2: while $i \leq K$ do
3: if E $( \pmb { x } ^ { \prime } ) \neq 1$ then
4: Break {return generated result if non
toxic or semantic-related}
5: else
6: $\pmb { x } ^ { \prime }  f ( \pmb { x } )$ {generate again if toxic or
semantic-unrelated}
7: end if
8: end while
9: if $\begin{array} { r } { \mathrm { E } ( \pmb { x } ^ { \prime } ) = 1 } \end{array}$ then
10: ${ \pmb x } ^ { \prime } $ None {discard if the text is still toxic
or semantic-unrelated}
11: end if
12: return $\mathbf { { x } ^ { \prime } }$
```

## E.2 More Experiments & Evaluation Metrics

Expand CMD to Parallel Detoxification Task In addition to conducting the experiments on the text detoxification task, we also expand the CMD framework to parallel detoxification task and compare CMD with Paradetox (Logacheva et al., 2022) and COUNT (Pour et al., 2023) methods. Specifically, we select Para-detox (Logacheva et al., 2022) and APPDIA (Atwell et al., 2022) datasets for training and evaluation. Following (Logacheva et al., 2022; Pour et al., 2023), we report the BLUE, Style, SIM (Wieting et al., 2019) and Fluency score (Warstadt et al., 2019) in Tab. 11. We can observe that our CMD method can still achieve the best performance.

Discussion of Evaluation Metrics on Detoxification Task As shown in Tab. 10, apart from the Perplexity (PPL) score reported in Tab. 2, Tab. 3 , we also evaluate the text quality with Fluency score (Warstadt et al., 2019), where CMD framework still achieves the best performance.

It is worth noting that we also consider other evaluation metrics to reflect the text quality from two aspects:

• Diversity that reflects the generation diversity:

<table><tr><td>Methods</td><td>Fluency</td></tr><tr><td>GPT2-XL</td><td>75.11%</td></tr><tr><td>DEXPERTS</td><td>74.71%</td></tr><tr><td>GEDI</td><td>77.25%</td></tr><tr><td>ToxicReversal</td><td>76.51%</td></tr><tr><td>SGEAT</td><td>76.42%</td></tr><tr><td>CMD</td><td>78.12%</td></tr></table>

Table 10: Fluency score among different detoxification methods.

We observe that Diversity metrics can sometimes correlate with unreadable or chaotic text generation, which is counterproductive to our goal of producing coherent and safe content (shown in Fig 12 and 13). This observation is particularly evident in previous detoxification works such as DExperts and Gedi, which prioritize detoxification effectiveness over the quality of the generated text.

• Semantic Similarity that reflect the semantic similarity between generation and prompt: we find there is a tendency for higher semantic similarity between the generated text and the toxic context to result in lower quality and higher toxicity (as illustrated in Fig. 2. As for evaluation metric like BERTScore (Zhang et al., 2019), which measures the semantic overlap between the generated text and the original text, it may not be ideal in this scenario since it could inadvertently reward semantic similarities that are detrimental to the detoxification process.

Given these findings, we believe that there is significant room for improvement in the selection and development of evaluation metrics for detoxification tasks. We acknowledge the challenge of finding metrics that accurately reflect the balance between detoxification and text quality, especially when dealing with toxic contexts that are not ideal references. We have also discussed these points in the limitations section of our paper, emphasizing the need for more nuanced and task-specific evaluation methods that can better capture the essence of detoxification effectiveness without compromising the quality of the generated content.

## F Case Study

We provide the generation cases from different methods in Fig. 12. We can observe that existing detoxification methods either generate unrelated

<table><tr><td>Dataset</td><td>Method</td><td>BLEU</td><td>Style</td><td>SIM</td><td>Fluency</td></tr><tr><td rowspan="3">Paradetox</td><td>ParaDetox</td><td>64.53</td><td>0.89</td><td>0.86</td><td>0.89</td></tr><tr><td>COUNT</td><td>69.68</td><td>0.91</td><td>0.88</td><td>0.91</td></tr><tr><td>CMD</td><td>71.31</td><td>0.91</td><td>0.88</td><td>0.91</td></tr><tr><td rowspan="2">APPDIA</td><td>COUNT</td><td>68.99</td><td>0.85</td><td>0.85</td><td>0.93</td></tr><tr><td>CMD</td><td>71.16</td><td>0.85</td><td>0.86</td><td>0.95</td></tr></table>

Table 11: Comparison between CMD and other text detoxification methods on the parallel detoxification task.

and unreadable texts (DExperts and Gedi) or fail to detoxify the text (SGEAT). In contrast, our CMD framework generates fluent and safe content.

## G Model Detoxification Pipeline

We follow the CMD framework to divide the Model Detoxification Pipeline into three steps: toxic segment detection, toxic segment detoxification, and continual generation. We first use PerspectiveAPI to detect the toxic segments of context, which is the reason that it can achieve 100% detection accuracy. Furthermore, we paraphrase the toxic context to ensure safety. Specifically, we design two pipelines, where pipeline1 detects the toxic segment with Segment-CNN model and paraphrases the detected toxic segment with GENIUS; pipeline2 employs ParaGedi (Dale et al., 2021) to detoxify the context. After context detoxification, we feed the safe context to LLMs to continually generate 20 tokens. We show the generated results in Fig. 13.

## H Data Synthesis with ChatGPT

Specifically, we adopt gpt-3.5-turbo-instruct model<sup>13</sup> (Ouyang et al., 2022) and carefully design the prompts to synthesize data for each substep. We list all the designed prompts for each sub-step in Fig. 14 and show the comparison between the data from CMD and ChatGPT in Fig. 15. After obtaining all the data produced from the gpt-3.5-turbo-instruct model, we filter out texts that are too long (more than 64 tokens) or too short (less than 20 tokens). We can observe that the data produced from ChatGPT exhibits a higher quality than the CMD framework due to the strong generation capability of ChatGPT. In addition, as shown in Fig. 16, we sample the case generated from the model trained with CMD and ChatGPT.

![](images/306e45d3f48de0ecd05f3c522e455009239f58ddb9a8084f2f0fb5cf70d461c8.jpg)  
(a) Human evaluation website.

![](images/e6fe14994bb6aef86e2d0c5125cfaed40282c82028e6e16bc453c7915906b9ab.jpg)  
(b) One case of user interface.  
Figure 11: Human evaluation interface, where annotators are allowed to choose the better generation case from semantic similarity, detoxification, and generation fluency aspects.

![](images/eebd7029cca3ca3eb9e948ad39f158bc423f80a28b48556c9c854ba31834ff8f.jpg)  
Figure 12: Cases generated from different detoxification methods, where we use red box to indicate the toxic segments and green box to indicate the detoxified safe segments.

![](images/715f8198bfbb9b062d591ebf1a5e5eeba26152c966cd4684d44a8d8538b4cdc2.jpg)  
Figure 13: Comparison between generation results from CMD and detoxification pipelines, where we use red box to indicate the toxic segments and green box to indicate the detoxified safe segments.

![](images/1f8cc45be0a4abc8e191a6a85a434584fa63230eece8f0120efdfeab2576d260.jpg)  
Figure 14: Prompts that are used to synthesize data from ChatGPT.

![](images/e8495f37b674ed7301f5f585de0d6203f2da3b4e45ec11840a9cb1013b601211.jpg)  
Figure 15: Comparison of data synthesized from CMD and ChatGPT.

![](images/344d3944490b2da1109fc4bf0866da63e5c16de7ad66b0ae3411fa1daa815ec8.jpg)  
Figure 16: Comparison of generation results from model trained CMD and ChatGPT.