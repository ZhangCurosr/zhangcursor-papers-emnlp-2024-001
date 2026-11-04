# Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection

Xiaomeng Hu<sup>1</sup>, Yiming Zhang<sup>1</sup>, Ru Peng<sup>1</sup>, Haozhe Zhang<sup>2</sup>,

Chenwei Wu<sup>2</sup>, Gang Chen<sup>1</sup>, Junbo Zhao<sup>1</sup>

<sup>1</sup>ZheJiang University

<sup>2</sup>Huawei Technologies Co., Ltd, China {xm.hu,rupeng,yimingz,cg,j.zhao}@zju.edu.cn} zhanghaozhe7@huawei.com wucw14@mails.tsinghua.edu.cn

## Abstract

In recent years, large language models (LLMs) have achieved remarkable success in the field of natural language generation. Compared to pre vious small-scale models, they are capable of generating fluent output based on the provided prefix or prompt. However, one critical challenge — the hallucination problem — remains to be resolved. Generally, the community refers to the undetected hallucination scenario where the LLMs generate text unrelated to the input text or facts. In this study, we intend to model the distributional distance between the regular conditional output and the unconditional output, which is generated without a given input text. Based upon Taylor Expansion for this distance at the output probability space, our approach manages to leverage the embedding and first-order gradient information. The resulting approach is plug-and-play that can be easily adapted to any autoregressive LLM. On the hallucination benchmarks HADES and other datasets, our approach achieves state-of-the-art performance.

## 1 Introduction

In recent years, large language models, such as GPT-4 (OpenAI, 2023), LLaMa (Touvron et al., 2023) and PaLM (Chowdhery et al., 2022), have achieved tremendous successes. LLMs generate fluent output based on given texts and prompts. They have been applied in various real-world scenarios, such as dialogue systems and information retrieval systems (Mialon et al., 2023). However, despite their powerful generative capabilities, LLMs still suffer from the problem of hallucinations: Given an input, the model generates text unrelated to the source or even contradictory (Ji et al., 2023). Even the most advanced language model currently available, GPT-4, has been shown to experience hallucinations (Bang et al., 2023). These hallucination issues severely compromise the fairness and safety of LLMs, hindering their

large-scale deployment.

Indeed, to solve this issue, the task of hallucination detection, aiming to determine whether hallucination occurs, has attracted more and more attention from the community. The existing methods for hallucination detection primarily focus on detecting at the token level in the output (Zhang et al., 2023). Some methods utilize uncertainty metrics at the token level to determine hallucination (Yuan et al., 2021; Fu et al., 2023), while some other methods design prompts to query the model multiple times to assess the hallucination (Mündler et al., 2023). Despite the prior efforts, we believe that white-box hallucination detection is not much studied. The white-box detection setting can both benefit the open-sourced LLM community such as LLaMA or Mistral (Jiang et al., 2023) — and prompt the LLM owners to provide extra services upon the current forms of API. Contrary to the black-box setting, a white-box setup may utilize the full token probability matrices and the embedding layers in the transformer. In this work, we dive into this white-box setup and prove that this internal information may significantly benefit detecting hallucinations.

However, given the current parameter scale of the LLM, devising a white-box detector is nontrivial. This means we need to systematically determine what portion of the information in the LLM is discriminative to raise the hallucination case. The motivation for this study stems from a rather simple and straightforward idea, as follows. We translate the verbal definition of hallucination to be more formal. Take the QA task as an example, where Q and A represent the question and the model’s answer, respectively. $P ( A | Q )$ and $P ( A )$ represent the probability distributions of the model outputting A under conditional Q and unconditional scenarios. The hallucination detection task can be considered as a task to determine how much the model answer A depends on the question Q, that is, modelling the difference between $P ( A )$ and $P ( A | Q )$ as denoted as $D ( P ( A ) , P ( A | Q ) )$ ).

![](images/48718d1ef130059790bef890aec56dabc3c9ff2f942710ba11563155e3053a7d.jpg)  
Figure 1: Two query forms of our method. Conditional query jointly inputs the question-answer text, while unconditional query only inputs the answer text. We use zero padding in the unconditional query to align the dimension.

Indeed, one can directly use metrics like KL divergence, cross-entropy or others to calculate the difference between $P ( A )$ and $P ( A | Q )$ . However, these measurements would only offer a singledimensional output as the indicator, which we found quite imprecise towards the detection of hallucination. We argue that simply using the probability space for this problem overlooks the complexity of the hallucination issue due to the massive amount of information wasted and dropped. In our approach, by contrast, we first conduct Taylor series of $D ( P ( A ) , P ( A | Q ) )$ . We further show that the subitems from the resulting series can be modelled by the embeddings in the transformer combined sophistically with the first-order gradient information. This result paves the way to effectively detect hallucinations.

Figure 1 provides a visual example of our approach, named EGH (Embedding and Gradientbased Hallucination detection method). More specifically, inspired by Filippova (2020), we construct a dual form, (i)-the conditional tuple jointing the source-output text versus (ii)-its unconditional counterpart with only the output text. After feeding both forms through the LLM, we yield the $D ( P ( A ) , P ( A | Q ) )$ mentioned above. We cache the corresponding necessary information, including embedding and gradient, alongside the two feedforward passes. On top of them, we train a simple classifier to discriminate if the hallucination occurs.

Our method is a model-agnostic solution for white-box hallucination detection, which relies on model internal signals (embedding and gradient information) to perform. In order to make our method solid, we managed to experiment with multiple LLMs as base models. On several common hallucination detection datasets, including HaluEval (Li et al., 2023), SelfCheckGPT (Manakul et al., 2023), and HADES (Liu et al., 2022), EGH establishes a set of state-of-the-art performances.

## 2 Related Work

## 2.1 Hallucination in LLMs

In recent years, large language models, such as GPT-4 (OpenAI, 2023), LLaMa (Touvron et al., 2023), and PaLM (Chowdhery et al., 2022) have become the mainstream research direction in the field of natural language processing. According to the scaling law (Kaplan et al., 2020), as the number of parameters increases, the capabilities of these large language models (LLMs) also improve, allowing them to generate fluent output based on existing text. However, due to the uncertainty in the outputs, large language models (LLMs) also face hallucination issues, which significantly hinder the development of LLMs.

Ji et al. (2023) define hallucination as natural language generation models generating unfaithful or nonsensical text. In a more granular classification, we can categorize hallucinations into two types: intrinsic hallucination and extrinsic hallucination. Intrinsic hallucination refers to generated text that contradicts the input, while extrinsic hallucination refers to outputs that cannot be derived from the input, meaning unrelated to the input. While Zhang et al. (2023) categorize hallucinations into three types: Input-conflicting hallucination, where LLMs generate content that deviates from the source input provided by users; Contextconflicting hallucination, where LLMs generate content that conflicts with previously generated information by itself; Fact-conflicting hallucination, where LLMs generate content that is not faithful to established world knowledge.

Recent work indicates that hallucinations in large language models are inevitable (Kalai and Vempala, 2023; Kang et al., 2024; Xu et al., 2024). Therefore, we need to detect hallucinations during the model’s output process.

## 2.2 Hallucination Detect Method

One simple detection method utilizes numerical metrics such as ROUGE (Lin, 2004) and PAR-ENT (Dhingra et al., 2019). Some researchers have also explored using models to detect hallucinations. FActScore (Min et al., 2023) retrieves external knowledge based on the question, and after obtaining this external knowledge, it utilizes a language model (such as LLaMa-65b) to assess the consistency between the answer and the external knowledge in order to determine the hallucination. Filippova (2020) proposed a language model-based detection method where they trained a conditional LM and an unconditional LM separately, using their respective losses to measure the presence of hallucination.

Besides, some methods allow LLMs to autonomously assess hallucinations by designing prompts to query the model multiple times. Mündler et al. (2023) design dedicated prompts to query an evaluator LLM (e.g.ChatGPT) whether the subjective LLM contradicts itself under the same context and report classification metrics, including precision, recall, and F1 score.

Existing hallucination detection methods mostly perform hallucination detection at the token level. However, when the model loses much of the internal information while outputting tokens, it undoubtedly significantly increases the difficulty of hallucination detection. Therefore, this paper designs a white-box hallucination detection method that utilizes the internal information of the model to detect hallucinations.

## 3 Method

## 3.1 Preliminary

To simplify representation, in this section, we take a question-answering task as an example for illustration. The input source text is the question in the QA task, denoted as $Q = \{ Q _ { 1 } , Q _ { 2 } , \dotsb , Q _ { m } \}$ , and the text generated by the model is the answer to the question, denoted as $A = \left\{ A _ { 1 } , A _ { 2 } , \cdot \cdot \cdot , A _ { n } \right\}$ Here, m and n represent the number of tokens of $Q$ and A, respectively. Our task is to classify $A$ and obtain its hallucination label $y _ { h a l }$ , where $y _ { h a l } \in \{ 0 , 1 \}$

## 3.2 Overview

Our method is based on the assumption that during the process of generating hallucinations, the model tends to incorporate less information from the source text, and the output is unrelated to the source text. Therefore, the extent to which the model accesses the source text can, to a certain degree, represent the level of hallucination. For a language model LLM, question $Q ,$ and output text A, we feed Q and A into the model in two different forms,where [, ] represent the concatenation operation:

• Concatenate Q and A together and feed into the model. This input method simulates the model’s output in the presence of conditions Q.

• Only feed A into the model. This input method corresponds to simulating the model’s output without any conditions. To align with the dimension in the first case, We adopt the zero-padding method, which involves inserting 0 of length m before A, where ${ \textbf { 0 } } =$ $\underset { \{ 0 , 0 , \cdots , 0 \} } { \overset { m } { \overbrace { 0 , 0 , \ldots , 0 } } }$ . Since the zero-padding parts are masked in the forward pass, this input method is equivalent to directly inputting A.

After completing the input in the two aforementioned forms, We have the following definition:

Definition $( D [ Q , A ] )$ . the difference in output between the conditional input and the unconditional input.

Consider the probability distributions of the answer part in two input modes, denoted as $P ( A | Q )$ and $P ( A | \mathbf { 0 } ) , D [ Q , A ]$ can be represented as the distribution difference between $P ( A | Q )$ and $P ( A | \mathbf { 0 } )$

$$
D ( [ Q , A ] ) = { \mathrm { D i f f e r e n c e } } ( P ( A | Q ) , P ( A | \mathbf { 0 } ) )\tag{1}
$$

where Difference() is the differentiable indicators representing differences in probability distributions, such as KL divergence, cross-entropy, and so on. During the generation of hallucinated outputs, we believe that the model receives less information from $\mathrm { Q } ,$ resulting in a smaller relationship between the model’s output and Q. The gap $D ( [ Q , A ] )$ is smaller. Conversely, when the model outputs normally, it fully utilizes information from Q, leading to a larger gap $D ( [ Q , A ] )$ . Therefore, $D ( [ Q , A ] )$ can be the extent to which the model accesses the source text and then represents the degree of hallucination further.

![](images/868ce4cdf19c9bcd6bf6049416f3a2432fcb7b6a2d2ad620a5cd76c845cea83e.jpg)  
Figure 2: The algorithm schematic of the EGH. The step 1 part of the figure represents the extraction process of feature $E = E ( A | Q ) - E ( A | \mathbf { 0 } )$ , while the step 2 part represents the extraction process of feature $G = \nabla D ( [ \mathbf { 0 } , A ] )$ Only the step 3 part (Hallucination Detector) in the figure undergoes parameter updates during the training process.

Since Difference() is differentiable by definition, together with the language model itself being differentiable, $D ( [ Q , A ] )$ is a differentiable function. Consider the first-order Taylor polynomial of $D ( [ Q , A ] )$ at point $[ 0 , A ] \ =$ m

$$
\{ \overbrace { 0 , 0 , \cdots , 0 } ^ { \cdots } , A _ { 1 } , A _ { 2 } , \cdots , A _ { n } \} \colon
$$

$$
\begin{array} { r l r } {  { D ( [ Q , A ] ) = D ( [ \mathbf { 0 } , A ] ) } } \\ & { } & { + [ \nabla D ( [ \mathbf { 0 } , A ] ) ] ^ { T } ( [ Q , A ] - [ \mathbf { 0 } , A ] ) } \\ & { } & { + R _ { 1 } ( [ Q , A ] ) \quad \quad \quad ( } \end{array}\tag{2}
$$

It’s clear that $D ( [ \mathbf { 0 } , A ] )$ is a constant. $R _ { 1 } ( [ Q , A ] )$ is a term of higher order, such as a second derivative, with a high computational cost to model. Considering the high computational cost and difficulty of $R _ { 1 }$ , we ignore the use of $R _ { 1 }$ in this article and only focus on whether hallucination detection can be performed based on the other terms. And the remaining two terms in Equation 2, $[ Q , A ] - [ 0 , A ]$ and $\nabla D ( [ \mathbf { 0 } , A ] )$ , can be used as factors influecing $D ( [ \mathbf { 0 } , A ] )$ . EGH aims to identify which internal parts affect hallucination generation rather than fully modelling the entirety of $D ( [ \mathbf { 0 } , A ] )$ ). Therefore, we only extract features to represent the two factors above. In the following two sections, we demonstrate the features used to represent both entities.

## 3.3 Embedding Information

For the factor $[ Q , A ] - [ 0 , A ]$ , as shown in step 1 of Figure 2, it means the difference between $[ Q , A ]$ and $[ \mathbf { 0 } , A ]$ . When it comes to the token level, subtracting two tokens has no practical significance. Therefore, we consider using embedding layers as an alternative. $E ( A | Q )$ means the embedding under condition Q, while $E ( A | \mathbf { 0 } )$ means the embedding under condition 0 (unconditional). The difference vectors of two embeddings can be used as representations of $[ Q , A ] - [ 0 , A ]$ . Let $E ( A | Q )$ and $E ( A | \mathbf { 0 } )$ be the embedding vectors corresponding to the answer part in two input modes, we use the differential embedding vector between $E ( A | Q )$ and $E ( A | \mathbf { 0 } )$ to represent this feature, denoted as $E = E ( A | Q ) - E ( A | \mathbf { 0 } )$

## 3.4 Gradient Information

For the factor $\nabla D ( [ \mathbf { 0 } , A ] )$ , as shown in step 2 of Figure 2, since we used the embedding layer above, we calculate the gradient of its corresponding layer as the representation. Consider the probability distributions of the answer part in two input modes, $D ( [ Q , A ] )$ can be expressed as the sum of KL divergences corresponding to each token in the two probability distributions:

$$
D ( [ Q , A ] ) = \sum _ { i = 1 } ^ { n } D _ { K L } [ P ( A _ { i } | Q ) | | P ( A _ { i } | \mathbf { 0 } ) ]\tag{3}
$$

After computing the KL divergence, we perform gradient backpropagation to obtain the gradients

Algorithm 1 Algorithm of EGH   
Input: The Question: $Q = \{ Q _ { 1 } , Q _ { 2 } , \cdots , Q _ { m } \}$   
The Answer $A = \{ A _ { 1 } , A _ { 2 } , \cdot \cdot \cdot , A _ { n } \}$ , The Func  
tion $e _ { \mathrm { L L M } } ( )$ is to get the final embedding of the   
LLM. The Function $p _ { \mathrm { L L M } } ( )$ is to get the Proba  
bility of the LLM’s outputs. The Function $d _ { \mathrm { K L } } ( \ v u )$   
means Calculate KL Divergence. The Weight Pa  
rameter λ.   
Output: The hallucination label $y _ { h a l } \in \{ 0 , 1 \}$   
1: $[ Q , A ]  \{ Q _ { 1 } , Q _ { 2 } , \cdot \cdot \cdot , Q _ { m } , A _ { 1 } , A _ { 2 } , \cdot \cdot \cdot , A _ { n } \}$   
m   
人   
2: $[ \mathbf { 0 } , A ]  \{ \overleftarrow { 0 , 0 , \cdot \cdot \cdot , 0 } , A _ { 1 } , A _ { 2 } , \cdot \cdot \cdot , A _ { n } \}$   
3: $P ( A | Q )  p _ { \mathrm { L L M } } ( [ Q , A ] )$   
4: $P ( A | \mathbf { 0 } )  p _ { \mathrm { L L M } } ( [ \mathbf { 0 } , A ] )$   
5: $\begin{array} { r } { D ( [ Q , A ] ) \xleftarrow { } \sum _ { i = 1 } ^ { n } d _ { \mathrm { K L } } [ P ( A _ { i } | Q ) | | P ( A _ { i } | \mathbf { 0 } ) ] } \end{array}$   
6: $G \gets \nabla ( D [ \mathbf { 0 } , A ] )$   
7: $E ( A | Q )  e _ { \mathrm { L L M } } ( [ Q , A ] )$   
8: $E ( A | \mathbf { 0 } )  e _ { \mathrm { L L M } } ( [ \mathbf { 0 } , A ] )$   
9: $E  E ( A | Q ) - E ( A | \mathbf { 0 } )$   
10: $y _ { h a l }  f ( \lambda E + ( 1 - \lambda ) G )$   
11: return $y _ { h a l }$

corresponding to the embedding layer, which is the feature that represents the factor $\nabla D ( [ \mathbf { 0 } , A ] )$ ). We denote it as $G = \nabla D ( [ \mathbf { 0 } , A ] )$

## 3.5 Embedding and Gradient-based Hallucination Detection Method

The overall algorithm is shown in Figure 2. Following the input methods described in Section 3.2, we input Q and A to the LLM in two different forms. For feature $E ,$ we take the last hidden layer as the embedding, extract them, and obtain the difference vector $E \in \mathbb { R } ^ { n \times h }$ , where h is the hidden size. For feature G, we first calculate the Kullback-Leibler divergence between the two output probability distributions and then perform gradient backpropagation to obtain $G \in \mathbb { R } ^ { n \times h }$ . So far, our operations have been solely focused on obtaining the two features, and the parameters involved during this process are frozen.

After extracting the aforementioned two features, E and G, the label of hallucination, $y _ { h a l } \in \{ 0 , 1 \}$ can be represented as:

$$
y _ { h a l } = f ( \alpha E + \beta G ) ,\tag{4}
$$

where $f ( )$ is a trainable function instantiated by a simple MLP as the end-point hallucination detector. $\alpha , \beta$ are hyper-parameters weighing E and G. To simplify the representation, we introduce a variable $\textstyle \lambda = { \frac { \alpha } { \alpha + \beta } }$ to rewrite the above formula into a more standardized form:

$$
y _ { h a l } = f ( \lambda E + ( 1 - \lambda ) G )\tag{5}
$$

In the actual implementation process, after fusing the E and G, we take the average pooling of the outputs to obtain a representation vector $\mathbb { R } ^ { h }$ for the whole answer text. Finally, we simply use a three-layer MLP model as f to obtain the final hallucination label, $y _ { \mathrm { h a l } } \in \{ 0 , 1 \}$ . We use the training data to train the parameters of this part and ultimately obtain the hallucination detector.

## 4 Experiment

## 4.1 Dataset

HaluEval (Li et al., 2023) HaluEval is one of the benchmarks for hallucination detection. It consists of 5000 general examples and 30000 task-specific examples, covering three tasks: question answering, knowledge-grounded dialogue, and text summarization. For general examples, the questions come from the Alpaca (Taori et al., 2023) instruction fine-tuning dataset, where each data consists of ChatGPT’s response and hallucination labels. For task-specific examples, each data includes a question, ChatGPT’s generated correct answer, and a hallucinated answer.

SelfcheckGPT (Manakul et al., 2023) The dataset contains 1908 sentences from 298 articles generated by GPT-3. Each sentence is labelled with one of the three veracity labels: “Accurate”, “Minor Inaccurate” and “Major Inaccurate”. In the experiments, “Major Inaccurate” and “Minor Inaccurate” sentences are labelled as hallucination class, while sentences labelled as “Accurate” are considered non-hallucination class.

HADES (Liu et al., 2022). HADES is a tokenlevel hallucination detection dataset consisting of 8,754 training and 1,000 validation examples. The dataset annotates certain specific tokens in a sentence to determine whether these labelled tokens contain hallucinations.

## 4.2 Experiment Setup

Selection of LLM Ideally, EGH should detect during the hallucination generation process, directly utilizing a model that generates hallucinations. However, since we cannot reproduce the generation process of these datasets, we opt to use other language models to simulate the generation process. This language model must be open source since we need to leverage white-box information. We chose LLaMa-2-7B and OPT-6.7B as base language models for the feature extraction step. For the HADES dataset, to align the settings in the baselines, we use BERT, RoBERTa and GPT-2 as the base model, which are the same as baselines.

<table><tr><td>Method</td><td>Model</td><td>QA</td><td>Dialogue</td><td>Summary</td><td>General</td></tr><tr><td rowspan="4">Baseline</td><td>GPT-3</td><td>49.71</td><td>50.42</td><td>51.03</td><td>77.47</td></tr><tr><td>text-davinci-002</td><td>59.56</td><td>61.23</td><td>47.77</td><td>48.23</td></tr><tr><td>text-davinci-003</td><td>49.65</td><td>68.37</td><td>48.10</td><td>87.05</td></tr><tr><td>ChatGPT</td><td>63.03</td><td>67.82</td><td>61.74</td><td>86.01</td></tr><tr><td rowspan="2">CoNLI (Lei et al., 2023)</td><td>GPT-3.5</td><td>85.00</td><td>-</td><td>64.10</td><td>一</td></tr><tr><td>GPT-4</td><td>86.20</td><td>-</td><td>72.50</td><td></td></tr><tr><td>KnowHalu (Zhang et al., 2024)</td><td>GPT-3.5</td><td>80.70</td><td></td><td>68.50</td><td></td></tr><tr><td>SAPLMA (Azaria and Mitchell, 2023)</td><td>LLaMa2-7B</td><td>95.31</td><td>73.59</td><td>93.68</td><td>79.23</td></tr><tr><td rowspan="2">EGH</td><td>LLaMa2-7B</td><td>97.19</td><td>77.10</td><td>95.08</td><td>83.01</td></tr><tr><td>OPT-6.7B</td><td>96.26</td><td>74.96</td><td>92.23</td><td>81.46</td></tr></table>

Table 1: Results in HaluEval Benchmark. The baseline method utilizes prompts to query the model, prompting it to determine whether the answer exhibits hallucinations. We reproduced the experiments in the baseline on our partitioned test set. EGH has achieved state-of-the-art results in three task-specific tasks.

<table><tr><td>Method</td><td>Setup</td><td>PR-AUC</td></tr><tr><td rowspan="5">SelfCheckGPT</td><td>w/BERTScore</td><td>81.96</td></tr><tr><td>w/QA</td><td>84.26</td></tr><tr><td>w/Unigram(max)</td><td>85.63</td></tr><tr><td>Combination</td><td>87.33</td></tr><tr><td>trained on SelfCheckGPT</td><td>86.45</td></tr><tr><td rowspan="2">Wang et al. (2023) EGH</td><td>trained on HaluEval</td><td>87.23</td></tr><tr><td>trained on SelfCheckGPT</td><td>89.82</td></tr></table>

Table 2: Results on the SelfCheckGPT dataset. We train two models on two different datasets, 10% of the Self-CheckGPT and HaluEval. The latter results demonstrate that our method possesses a certain degree of generalization.

Training Process For the HaluEval dataset, due to there being four tasks in HaluEval (for uniform representation, we also consider the general examples as a task), we trained a separate model for each task. For a specific task, We randomly select 10% of the data as the training set to train the hallucination detector and use the remaining 90% of the data as the testing set for testing. For the Self-CheckGPT dataset, we conducted separate tests with and without training: the untrained test utilized a model trained on HaluEval for evaluation, while the trained test involved training on 10% of the data from SelfCheckGPT and testing on the remaining 90% of the data. For the HADES dataset, we use the training data of HADES to obtain the hallucination detector model and test it on the validation set.

Baselines For the HaluEval dataset, we adopt the LLM test methods from the original paper as the baselines. The baseline experiments utilized the following prompts: You should try your best to determine ifthe answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUSTbe “Yes” or “No”. We evaluate four state-of-the-art LLMs as baselines: GPT-3(davinci), text-davinci-002, textdavinci-003 and ChatGPT. We reproduced the experiments in the baseline on our partitioned test set (as described above) as the final baseline result. We also compare the results in Zhang et al. (2024) and Lei et al. (2023). Additionally, since our method is a white-box method, we used the method from Azaria and Mitchell (2023) as a white-box baseline for comparison with our approach. For the Self-CheckGPT Dataset, we use the results from the original paper as the baseline. We also compare the results in Wang et al. (2023). For the HADES dataset, since the HADES dataset does not have a publicly available labelled test set, we replicated the benchmark experiments on the validation set, which serves as the baseline.

Training Hyperparameters Our hallucination detector is a three-layer MLP, where the first two layers scale the feature dimension to half of the preceding layer, and the final layer transforms it into probabilities. The ReLU (Glorot et al., 2011) function is chosen as the activation function. In Equation 5, λ is set to 0.8, respectively. In Section

<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td rowspan="2">Acc</td><td rowspan="2">G-Mean (↑)</td><td rowspan="2">BSS (↓)</td><td rowspan="2">AUC</td><td colspan="3">Not Hallucination</td><td colspan="3">Hallucination</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">Benchmark</td><td>GPT-2</td><td>70.60</td><td>69.66</td><td>19.68</td><td>75.86</td><td>68.16</td><td>80.11</td><td>73.66</td><td>74.31</td><td>60.57</td><td>66.74</td></tr><tr><td>BERT</td><td>71.20</td><td>70.09</td><td>20.61</td><td>75.56</td><td>68.35</td><td>81.67</td><td>74.42</td><td>75.71</td><td>60.16</td><td>67.05</td></tr><tr><td>RoBERTa</td><td>71.00</td><td>69.78</td><td>19.46</td><td>77.39</td><td>68.01</td><td>82.06</td><td>74.38</td><td>75.85</td><td>59.34</td><td>66.59</td></tr><tr><td rowspan="3">EGH</td><td>GPT-2</td><td>71.30</td><td>70.16</td><td>19.34</td><td>75.87</td><td>70.35</td><td>75.44</td><td>72.81</td><td>71.49</td><td>65.88</td><td>68.56</td></tr><tr><td>BERT</td><td>73.00</td><td>72.69</td><td>18.86</td><td>78.84</td><td>73.62</td><td>72.91</td><td>73.26</td><td>71.75</td><td>72.48</td><td>72.11</td></tr><tr><td>RoBERTa</td><td>72.60</td><td>71.77</td><td>19.24</td><td>77.62</td><td>70.18</td><td>80.51</td><td>74.99</td><td>75.19</td><td>63.22</td><td>68.68</td></tr></table>

Table 3: We utilize the same base model to obtain gradient information and compare the results of our method with the benchmark. Our approach consistently outperforms the benchmark in terms of performance.

5.1, we provide a detailed discussion of the impact of λ on the results.

## 4.3 Results in HaluEval

Table 1 presents the results of our experiments on HaluEval. EGH has achieved state-of-the-art (SOTA) results in three task-specific tasks. In the QA task, EGH achieved a 97% accuracy rate, which is 35% higher than the baseline and 10% higher than the CONLI method. In the Dialogue task, EGH achieved a 77.1% accuracy rate, which is 5% higher than the baseline. In the Summary task, EGH achieved a 95% accuracy rate, which is 35% higher than the baseline. In the general question-answering task, our approach achieved an accuracy of 82%, slightly lower than the baseline. Due to the imbalance of labels in the general question-answering task, with only 815 out of 5000 data instances being hallucination data, we consider this result to be a normal phenomenon. Compared with the white-box method SAPLMA, EGH has a 2-3% improvement on all four tasks.

## 4.4 Results in SelfCheckGPT

Table 2 presents the results of our experiments on SelfCheckGPT. The model trained on the HaluEval achieved performance comparable to the baseline and method Wang et al. (2023) on PR-AUC. After training on 10% of the data, our approach outperformed the baseline and method Wang et al. (2023) by 2-3% on PR-AUC. The results indicate that our method exhibits a certain degree of generalization and is applicable to various tasks and datasets.

## 4.5 Results in HADES

Table 3 presents the results of our experiments on HADES. EGH has shown improvements in accuracy compared to the benchmark, using the same base model. BERT achieved the highest accuracy, reaching 73%, and it also had the highest F1 score, surpassing the benchmark by 5%. RoBERTa achieved the highest F1 score on non-hallucination data. The experimental results indicate that EGH is also applicable to traditional language models such as BERT and GPT-2.

<table><tr><td>λ</td><td>0.0</td><td>0.2</td><td>0.5</td><td>0.8</td><td>1.0</td></tr><tr><td>Accuracy F1</td><td>74.22 75.36</td><td>75.47 76.62</td><td>77.12 77.74</td><td>77.39 78.28</td><td>76.54 76.68</td></tr></table>

Table 4: Ablation Study on Weights on dialogue task

## 5 Discussion

## 5.1 Weights for the embedding and Gradient

Recalling Equation 5, $y _ { h a l } = f ( \lambda E + ( 1 - \lambda ) G )$ For the two extracted features, the difference embedding E and the gradient G, we respectively use weight λ to perform weighted summation. To verify the effectiveness of the two features, we conduct ablation experiments using different λ, with a set of values for 0, 0.2, 0.5, 0.8, and 1. Since the accuracy on QA and summary tasks is higher, making the ablation effects less noticeable, we conducted experiments on the dialogue task instead. The experimental results are presented in the Table 4:

As shown in Table 4, when using only embedding or gradient $( \lambda = 0 \mathrm { o r } \lambda = 1 )$ , both accuracy and F1 score are lower than when using both features simultaneously. Therefore, both features contribute to the classification process. When the value of λ is relatively large, the classification accuracy is higher. Specifically, when λ = 0.8, the accuracy is the highest, indicating that embedding plays a primary role in classification.

## 5.2 Visualization: Difference $( P ( A ) , P ( A | Q ) )$

Our method is based on the assumption that during the process of generating hallucinations, the model tends to rely less on information from the input and engage in more “free expression”. Due to these two input forms representing conditional and unconditional scenarios, respectively, the extent of the model’s creativity, namely its access to source conditions, can be indicated by the difference between these two outputs. Therefore, we introduce Difference(P(A), P(A Q)) to represent the difference between the output probability distributions under two input scenarios. Mathematical quantities such as KL divergence, cross-entropy, etc., can directly express the difference between two probability distributions, thus serving as a form of Difference(P(A), P(A Q)). Consequently, we compute the values of these two statistical quantities in both hallucination and non-hallucination samples and plot the probability distribution figure.

![](images/b70fa8ccd4bd6ae769cd76262913ae1ac8ac53a1faf5f900026b8e9c0b99e71d.jpg)

![](images/c617b9dd28439a957699d1641331959431104512b848b2d622bf611e1c6b1f41.jpg)

![](images/687fbf6ea7e2c2cb2d165280f409e0eb7958666d8f6810ae0644caf77b208a6b.jpg)  
(a) KL Divergence

![](images/f7f2fdf75e39d1a59ef7be251be1b9aed7ca2cb40643000875aab915622997c3.jpg)

![](images/811a1a939228570f449d79910e795360b310f56921de0cecb4e32b413a1d04da.jpg)

![](images/ef0e6ce4c591d8ecf79d8520cb8ab77a90506dcfad34c3ca8c0744f13b56a4bb.jpg)

(b) Cross Entropy  
![](images/84446771a102d36fcb184b527cb32ec7dd0b3859d7884babec9b4bd9587841e8.jpg)

![](images/9e0ab7b1e6493cfdbe6a92190fd4603a55e39c9cbfd0a620027d82b95bff2ca4.jpg)

Figure 3: Probability Distribution for KL Divergence and Cross Entropy. There is a significant difference in the distribution of non-hallucination and hallucination samples, and the peaks also conform to our hypothesis.
<table><tr><td>Difference Form</td><td>Metric</td><td>Hal</td><td>Non-Hal</td></tr><tr><td>KL Divergence</td><td>mean std</td><td> $\overline { { 6 . 7 \times 1 0 ^ { - 5 } } }$   $2 . 6 \times 1 0 ^ { - }$  5</td><td> $\overline { { 9 . 6 \times 1 0 ^ { - 5 } } }$   $9 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Cross entropy</td><td>mean std</td><td> $\overline { { 3 . 7 \times 1 0 ^ { - 3 } } }$   $1 . 0 \times 1 0 ^ { - }$  -3</td><td> $\overline { { 5 . 7 \times 1 0 ^ { - 3 } } }$   $1 . 8 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 5: Average(mean) and standard deviations(std) of KL divergence and cross entropy

As shown in Figure 3, we randomly sampled 1,000 examples for each task in HaluEval and obtained the probability distributions corresponding to hallucinated and non-hallucinated data. The distribution of KL divergence and Cross Entropy have certain differences between nonhallucination and hallucination samples. Since hallucinatory responses tend to receive less information from the questions, the disparity Difference(P(A), P(A Q)) between the two output probability distributions will be relatively smaller. Therefore, the KL divergence and crossentropy will be smaller for hallucination samples compared to non-hallucination samples. The mean values in the Table 5 above also confirm this observation.

<table><tr><td>Method</td><td>QA</td><td>Summary</td></tr><tr><td>LR with KL and CE</td><td>67.71</td><td>63.00</td></tr><tr><td>EGH</td><td>97.19</td><td>95.08</td></tr></table>

Table 6: The results of directly comparing KL divergence and cross-entropy using logistic regression and EGH indicate that applying these two features directly for hallucination detection is not advisable.

However, as mentioned earlier, using a threshold value for the KL divergence or cross-entropy is imprecise to determine whether hallucinations occur. Besides, due to the loss of a significant amount of information during the process of outputting probabilities, directly utilizing features such as probabilities, KL divergence, cross-entropy, etc., cannot effectively model hallucinations. To demonstrate this, we designed a simple experiment: we used KL divergence and cross-entropy as two-dimensional features, with Logistic Regression as the classifier for detection. The results of the QA and Summary tasks in HaluEval are shown in Table 6. The experimental results support our viewpoint. Therefore, our approach extracts deeper features from these two input modes for hallucination detection.

## 6 Conclusion

This paper proposes a white-box hallucination detection method named EGH that utilizes the internal embedding and gradient of the model to determine hallucination. EGH is based on the following assumptions: during hallucination generation, the model tends to generate responses without considering the input question directly. The model’s understanding of the input question represents the degree of hallucination. We designed both conditioned and unconditioned inputs and utilized the Taylor expansion method to demonstrate that embeddings and gradient features can represent this degree. EGH achieves state-of-the-art results on hallucination detection datasets such as HaluEval, SelfCheckGPT, and HADES, validating the effectiveness of our method. Our work provides valuable insights into detecting hallucinations in LLM.

## 7 Limitation

The main limitation of this method lies in the fact that, as our approach requires leveraging gradient information and necessitates gradient backpropagation, it undoubtedly increases the time and space complexity of the method. In terms of time, our method requires two inputs, although it can be alleviated through batch input. In terms of space, our method requires gradient calculation and additional space to store information. In the future, we will explore more lightweight methods to detect hallucinations.

## Acknowledgements

This work is supported by the Pioneer R&D Program of Zhejiang (No. 2024C01035), and the NSFC Grants (No. 62206247), and in part by the Fundamental Research Funds for the Central Universities 226-2024-00049. This work is also partially supported by ZJU Kunpeng&Ascend Center of Excellence.

## References

Amos Azaria and Tom Mitchell. 2023. The internal state of an LLM knows when it’s lying. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 967–976, Singapore. Association for Computational Linguistics.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt

on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Bhuwan Dhingra, Manaal Faruqui, Ankur Parikh, Ming-Wei Chang, Dipanjan Das, and William W Cohen. 2019. Handling divergent reference texts when evaluating table-to-text generation. arXiv preprint arXiv:1906.01081.

Katja Filippova. 2020. Controlled hallucinations: Learning to generate faithfully from noisy data. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 864–870.

Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. 2011. Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 315– 323. JMLR Workshop and Conference Proceedings.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Adam Tauman Kalai and Santosh S Vempala. 2023. Calibrated language models must hallucinate. arXiv preprint arXiv:2311.14648.

Katie Kang, Eric Wallace, Claire Tomlin, Aviral Kumar, and Sergey Levine. 2024. Unfamiliar finetuning examples control how language models hallucinate. arXiv preprint arXiv:2403.05612.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Deren Lei, Yaxi Li, Mingyu Wang, Vincent Yun, Emily Ching, Eslam Kamal, et al. 2023. Chain of natural language inference for reducing large language model ungrounded hallucinations. arXiv preprint arXiv:2310.03951.

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. Halueval: A largescale hallucination evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Tianyu Liu, Yizhe Zhang, Chris Brockett, Yi Mao, Zhifang Sui, Weizhu Chen, and William B Dolan. 2022. A token-level reference-free hallucination detection benchmark for free-form text generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6723–6737.

Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.

Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. 2023. Augmented language models: a survey. arXiv preprint arXiv:2302.07842.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251.

Niels Mündler, Jingxuan He, Slobodan Jenko, and Martin Vechev. 2023. Self-contradictory hallucinations of large language models: Evaluation, detection and mitigation. arXiv preprint arXiv:2305.15852.

OpenAI. 2023. Gpt-4 technical report.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Xiaohua Wang, Yuliang Yan, Longtao Huang, Xiaoqing Zheng, and Xuan-Jing Huang. 2023. Hallucination detection for generative large language models by bayesian sequential estimation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 15361–15371.

Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. 2024. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021. Bartscore: Evaluating generated text as text generation. Advances in Neural Information Processing Systems, 34:27263–27277.

Jiawei Zhang, Chejian Xu, Yu Gai, Freddy Lecue, Dawn Song, and Bo Li. 2024. Knowhalu: Hallucination detection via multi-form knowledge based factual checking. arXiv preprint arXiv:2404.02935.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. 2023. Siren’s song in the ai ocean: A survey on hallucination in large language models. arXiv preprint arXiv:2309.01219.

## A Analysis of Our Experimental Results

Our method achieved state-of-the-art results on various datasets. Among all tasks, the QA task yielded the best performance, with an accuracy reaching 97%. In QA tasks, answers are highly dependent on the questions, and there is a certain degree of similarity between questions. Models are likely to make "heuristic errors," answering questions based on past learned knowledge and experiences without looking at the question stem. Therefore, our approach focuses on extracting the model’s understanding of the question, addressing the common hallucination type in QA tasks: Input-conflicting hallucination. We have achieved excellent results in QA tasks. For dialogue tasks they typically involve longer contexts, and the questions from different samples are relatively independent. Instances of encountering the same question are less frequent, so the model is less likely to rely on "memorizing answers". Consequently, occurrences of Input conflicting hallucinations are less common, result ing in less effective performance compared to QA tasks. In general QA tasks, our approach falls be low the baseline. We believe there are two main reasons for this: First, there is an imbalance be tween hallucination and non-hallucination samples, with only 815 hallucination samples available. This leads to a significant bias towards non-hallucination samples in the classifier, resulting in poor classification of hallucination samples. Second, in general datasets, questions tend to be common and relatively short. Under this task, models are most likely to experience Fact-conflicting hallucination, where the output answers contain factual errors that contradict objective facts. Since our approach does not rely on external knowledge bases, its ability to detect such hallucinations is not very strong. We consider this to be a normal phenomenon.