# Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis

Zeping Yu Sophia Ananiadou

Department of Computer Science, National Centre for Text Mining

The University of Manchester

{zeping.yu@postgrad. sophia.ananiadou@}manchester.ac.uk

## Abstract

We find arithmetic ability resides within a limited number of attention heads, with each head specializing in distinct operations. To delve into the reason, we introduce the Comparative Neuron Analysis (CNA) method, which identifies an internal logic chain consisting of four distinct stages from input to prediction: feature enhancing with shallow FFN neurons, feature transferring by shallow attention layers, feature predicting by arithmetic heads, and prediction enhancing among deep FFN neurons. Moreover, we identify the human-interpretable FFN neurons within both feature-enhancing and feature-predicting stages. These findings lead us to investigate the mechanism of LoRA, revealing that it enhances prediction probabilities by amplifying the coefficient scores of FFN neurons related to predictions. Finally, we apply our method in model pruning for arithmetic tasks and model editing for reducing gender bias. Code is on https://github.com/ zepingyu0512/arithmetic-mechanism.

## 1 Introduction

Arithmetic ability is a crucial foundational skill of large language models (LLMs) (Brown et al., 2020; Ouyang et al., 2022; Chowdhery et al., 2023), contributing significantly to reasoning (Wei et al., 2022; Kojima et al., 2022) and mathematical tasks (Peng et al., 2021; Azerbayev et al., 2023). While existing studies (Quirke et al., 2023; Zhang et al., 2023; Stolfo et al., 2023) have made significant breakthroughs in understanding arithmetic tasks, the exact mechanism still remains elusive. Zhang et al. (2023) find that only a few attention heads significantly impact arithmetic performance, but they do not elaborate on the mechanisms of these heads or how they influence FFN layers. Stolfo et al. (2023) intervene the hidden states and find the information flow from number and operation positions to the last position. However, they do not locate the important attention heads (proved to store different abilities (Olsson et al., 2022; Gould et al., 2023)) and FFN neurons (proved to store knowledge (Dai et al., 2021; Meng et al., 2022a)). Despite the challenge of pinpointing important FFN neurons among tens of thousands of nodes, many studies (Gurnee et al., 2023; Lieberum et al., 2023; Nanda et al., 2023) emphasize that considering FFN neurons as fundamental units is crucial for better understanding FFN layers. Furthermore, as model editing typically occurs at the neuron level (Dai et al., 2021; Geva et al., 2022), it remains unclear how to effectively leverage the explanations due to the uncertainty surrounding the precise locations of important parameters.

![](images/16f820f5c2a05fc16f786e4ed1a61ff19eb69156bf25c97352e02cd60cfbfb80.jpg)  
Figure 1: Four distinct stages in the internal logic chain from the inputs "3+5=" to the final prediction "8".

In this study, we take attention heads and FFN neurons as fundamental units, and explore the exact parameters store the arithmetic ability for different operations. We observe that only a minority of heads play significant roles in arithmetic tasks, which we refer to as "arithmetic heads". Through experiments involving 1-digit to 3-digit operations, as well as ablation studies comparing "change-one"

cases (e.g., 15+37=52) with "memorize" cases (e.g., 15+32=47), we find critical memorization of 1-digit operations is lost when these heads are intervened.

To explore the underlying mechanisms of this phenomenon, we propose the Comparative Neuron Analysis (CNA) method, which compares the change of neurons between the original model and the intervened model for the same case. We construct the internal logic chain by identifying four distinct stages that span from inputs to prediction, as depicted in Figure 1. During the feature enhancing stage, hidden-interpretable features are extracted from shallow FFN neurons. Subsequently, in the feature transferring stage, shallow attention layers convert these features into directlyinterpretable features and then transfer them to the last position. In the feature predicting stage, the arithmetic heads play critical roles, activating deep FFN neurons related to the final prediction. Finally, a prediction enhancing stage exists among deep FFN neurons. Lower FFN neurons activate upper FFN neurons, while both of them enhance the probability of the final prediction.

Based on this analysis, we investigate the mechanism of LoRA (Hu et al., 2021). We train a total of 32 models on a 2-digit arithmetic dataset, with each model integrating LoRA on one attention layer (0th to 31th). Starting from the 10th model, the accuracy of the model exhibits a noticeable downward trend, with varying rates of decline observed in the feature enhancing and prediction enhancing stages. Employing our CNA method to compare the original model with the fine-tuned model, we note a significant increase in the coefficient scores of crucial deep FFN neurons. Hence, we conclude that LoRA enhances the final prediction by amplifying the coefficient scores of important FFN neurons. Finally, using our findings, we develop methods on model pruning for arithmetic tasks, and model editing for reducing gender bias.

To summarize, our contributions are as follows:

1. We find the reason why only a few heads can influence arithmetic ability is that these heads store crucial parameters for memorizing 1D operations. We identify human-interpretable FFN neurons across both shallow and deep layers.

2. We propose the CNA method and construct the internal logic chain from inputs to prediction with four stages: feature enhancing, feature transferring, feature predicting, prediction enhancing.

3. We use the CNA method to explore the mechanism of LoRA and find LoRA increases the probability of final predictions by amplifying the important FFN neurons’ coefficient scores. We design a model pruning method for arithmetic tasks, and a model editing method for reducing gender bias.

## 2 Related Work

## 2.1 Mechanistic Interpretability

Mechanistic interpretability aims to reverse engineer the intricate computations executed by transformers. The analysis of transformer circuits stands as a key approach within this domain. Elhage et al. (2021) and Olsson et al. (2022) investigate the mechanism using a two-layer attention-only transformer and discover that induction heads can make predictions similar to [A][B] ... [A] -> [B]. Yu and Ananiadou (2024a) explore the details of in-context learning in a mechanistic view. Wang et al. (2022) present an explanation on an indirect object identification case in GPT2.

Causal mediation analysis (Pearl, 2001; Vig et al., 2020) is also widely used for locating important modules. Meng et al. (2022a,b) intervene the hidden states of GPT2 (Radford et al., 2019) and ascertain that the medium FFN layers play a significant role in processing subject names. Wang et al. (2023) intervene the attention layers to explore the mechanism of in-context learning and observe an information flow from demonstrations to corresponding labels. Geva et al. (2023) find two critical points on relation and subjection positions through interventions on attention edges.

Since causal mediation analysis methods require expensive forward pass over multiple input, several studies try to design static methods for interpreting language models. Geva et al. (2022) utilize the product of norm and coefficient score to locate important FFN neurons and find many FFN neurons have human-interpretable concepts when projecting into vocabulary space. Dar et al. (2022) find most matrices in attention and FFN layers are interpretable in vocabulary space.

## 2.2 Understanding Arithmetic in LLM

Hanna et al. (2023) investigate how GPT2-small computes greater-than. Gould et al. (2023) demonstrate that successor heads can aid in predicting the subsequent order, such as predicting "3" after "2". Zhang et al. (2023) investigate the attention heads for addition operation, and find only a few heads play significant roles. Zhong et al. (2024) investigate the clock and pizza algorithms for modular addition. Quirke et al. (2023) studies n-digit integer addition on an one-layer transformer, and find individual digits are computed in parallel. Through interventions on hidden states, Stolfo et al. (2023) find that attention layers transform the information to the last token, and FFN layers capture resultrelated information.

## 3 Arithmetic Heads in LLMs

We aim to examine the localization of arithmetic ability in Llama-7B (Touvron et al., 2023), a large language model consisting of 32 layers. Each attention layer contains 32 heads, and each FFN layer has 11,008 neurons. We observe the same results and mechanisms in GPT-J (Wang and Komatsuzaki, 2021), detailed in Appendix C.

## 3.1 Background

We start by introducing the inference pass in decoder-only language models. Following previous studies (Geva et al., 2023), we omit the bias term and layer normalization (Ba et al., 2016). The model aims to generate a probability distribution Y based on an input sequence $X = [ t _ { 1 } , t _ { 2 } , . . . , t _ { T } ]$ consisting of T tokens. $Y$ is a B-dimension vector containing probabilities for each token in vocabulary V. Each token $t _ { i }$ in X is embedded into a vector $x _ { i } ^ { 0 } \in \mathbb { R } ^ { d }$ using an embedding matrix $E \in \mathbb { R } ^ { B \times d }$ Then the vectors undergo transformation through $L + 1$ transformer layers (0th-Lth). Vector $x _ { i } ^ { l }$ on the ith position at layer l is computed by:

$$
x _ { i } ^ { l } = x _ { i } ^ { l - 1 } + A _ { i } ^ { l } + F _ { i } ^ { l }\tag{1}
$$

where $A _ { i } ^ { l } \in \mathbb { R } ^ { d }$ and $F _ { i } ^ { l } \in \mathbb { R } ^ { d }$ are the outputs of the lth attention and FFN layers, referred to as the attention output and FFN output, respectively. $x _ { i } ^ { l - 1 }$ represents the layer output at layer l 1, which also serves as the layer input at layer l. The term $x _ { i } ^ { l - 1 } + A _ { i } ^ { l }$ is denoted as the residual output. The attention layer captures information from different positions through H multiple heads $A T T N _ { j } ^ { l }$ , and the FFN layer transforms the residual output by matrices $W _ { f c 1 }$ and $W _ { f c 2 }$ with non-linearity σ:

$$
A _ { i } ^ { l } = \sum _ { j = 1 } ^ { H } A T T N _ { j } ^ { l } ( h _ { 1 } ^ { l - 1 } , h _ { 2 } ^ { l - 1 } . . . , h _ { T } ^ { l - 1 } )\tag{2}
$$

$$
F _ { i } ^ { l } = W _ { f c 2 } ^ { l } \sigma ( W _ { f c 1 } ^ { l } ( x _ { i } ^ { l - 1 } + A _ { i } ^ { l } ) )\tag{3}
$$

The representation of the last position on the final layer $x _ { T } ^ { L }$ is used for predicting the probability distribution Y of the next token by a softmax function on an unembedding matrix $E _ { u } \in \mathbb { R } ^ { B \times d }$

$$
Y = s o f t m a x ( E _ { u } x _ { T } ^ { L } )\tag{4}
$$

Geva et al. (2020) demonstrate that the FFN layer can be conceptualized as key-value memories, with matrices $W _ { f c 1 } ^ { l ^ { - } } \in \mathbb { R } ^ { d \times N }$ and $W _ { f c 2 } ^ { l } \in \mathbb { R } ^ { N \times d }$ storing keys and values for N neurons. The FFN output is obtained by adding N subvalues, where each subvalue is the result of multiplying a coefficient score $m _ { k } ^ { l }$ with a $f c 2$ vector $f c 2 _ { k } ^ { \bar { l } } \in \mathbb { R } ^ { d }$ (also referred to as the FFN value). These coefficient scores are calculated as the inner product between the residual output and the corresponding f c1 vector $f c 1 _ { k } ^ { l } \in \mathbb { R } ^ { d }$ (also referred to as the FFN key):

$$
F ^ { l } = \sum _ { k = 1 } ^ { N } m _ { k } ^ { l } f c 2 _ { k } ^ { l }\tag{5}
$$

$$
m _ { k } ^ { l } = \sigma ( f c 1 _ { k } ^ { l } \cdot ( x ^ { l - 1 } + A ^ { l } ) )\tag{6}
$$

In other words, the kth subvalue is the kth column of $W _ { f c 2 } ^ { l } .$ , whose subkey is the kth row of ${ W } _ { f c 1 } ^ { l }$

## 3.2 Interventions on Attention Heads

We make a 2-digit arithmetic dataset, including addition (2D+), subtraction (2D-), multiplication (2D\*) and division (2D/). Similar to Stolfo et al. (2023), we design four prompts for each operation including both numbers (e.g. 3) and number words (e.g. three), reported in Appendix A. The evaluation dataset has 1,600 sentences. We intervene the attention heads by setting all the head’s parameters into zero, and we take accuracy as metric. Llama-7B consists of 32 layers with 32 heads per layer. Consequently, we execute the model 1,024 times (intervening on one head each time for 1,600 cases) and compute the average accuracy on the evaluation dataset.

## 3.3 Results of Different Heads

The accuracy of the original model is 74.8%. Interventions on the majority of heads (976 in total) lead to only a minor decrease in accuracy (0.01%-2%). Only three heads result in a decrease of 10% or more. The top5 heads are shown in Table 1.

Interventions on head $1 7 ^ { 2 2 } , 1 5 ^ { 9 }$ and $1 4 ^ { 1 9 }$ cause 12.7% or more decrease. Specifically, $1 7 ^ { 2 2 }$ reduces

<table><tr><td></td><td>ori</td><td> $1 7 ^ { 2 2 }$ </td><td> $1 5 ^ { 9 }$ </td><td> $1 4 ^ { 1 9 }$ </td><td> $1 5 ^ { 2 3 }$ </td><td> $1 6 ^ { 1 }$ </td></tr><tr><td>all</td><td>74.8</td><td>53.4</td><td>62.1</td><td>62.7</td><td>68.1</td><td>68.7</td></tr><tr><td>2D+</td><td>96.8</td><td>42.9</td><td>83.2</td><td>92.5</td><td>89.7</td><td>91.6</td></tr><tr><td>2D-</td><td>94.4</td><td>72.3</td><td>84.6</td><td>93.2</td><td>86.5</td><td>79.1</td></tr><tr><td>2D*</td><td>56.6</td><td>50.5</td><td>50.9</td><td>51.3</td><td>52.3</td><td>56.9</td></tr><tr><td>2D/</td><td>51.4</td><td>48.2</td><td>29.5</td><td>13.8</td><td>43.8</td><td>47.1</td></tr></table>

Table 1: Accuracy (%) when intervening different heads. $" \mathrm { o r i } " \mathrm { : }$ : original model. $1 7 ^ { 2 2 } { \mathrm { ; } }$ 22th head in 17th layer.

21.4% in accuracy. Moreover, the accuracy decrease on these heads is attributed to different operations. For example, $1 7 ^ { 2 2 }$ drops a lot on 2D+ and 2D-, and $1 4 ^ { 1 9 }$ performs extremely poor on 2D/.

## 3.4 Reasons Causing Accuracy Decrease

Since the accuracy of more complicated operations are low, we analyze the most important head for each operation in 1-digit (1D), 2-digit (2D) and 3- digit (3D) operations, shown in Table 2. The most important heads in 1D, 2D and 3D operations are the same. We report the details of top5 heads in Appendix E. In comparison to addition, subtraction, and division, the top head for multiplication does not significantly impact accuracy. We leave further investigation of this phenomenon for future work.

<table><tr><td></td><td> $1 7 ^ { 2 2 } ( + )$ </td><td> $1 7 ^ { 2 2 } ( - )$ </td><td> $2 0 ^ { 1 8 } ( ^ { * } )$ </td><td> $1 4 ^ { 1 9 } ( / )$ </td></tr><tr><td>1D</td><td>46.5</td><td>62.2</td><td>6.8</td><td>54.9</td></tr><tr><td>2D</td><td>58.4</td><td>52.6</td><td>11.2</td><td>71.8</td></tr><tr><td>3D</td><td>52.5</td><td>56.9</td><td>8.1</td><td>53.2</td></tr></table>

Table 2: Accuracy decrease (%) in 1D, 2D and 3D.

In Table 2, the decreases of 1D, 2D and 3D operations are similar. Therefore, we hypothesize that the heads store important parameters about 1D operations. Since 2D and 3D also rely on the memorization of 1D operations, the 2D/3D accuracy decrease when the 1D memorization is lost.
<table><tr><td></td><td>add</td><td>sub</td><td>multi</td><td>divide</td></tr><tr><td>memorize</td><td>59.2</td><td>49.8</td><td>11.6</td><td>63.6</td></tr><tr><td>change-one</td><td>57.1</td><td>65.5</td><td>11.3</td><td>75.2</td></tr></table>

Table 3: Accuracy decrease (%) on memorize and change-one cases.

We also analyze two types of cases for each operation, which are named "change-one" (similar to the definition of "carry" in Opedal et al. (2024)) and "memorize". "Memorize" cases only require memorization. For example, " $1 5 { + } 3 2 { = } 4 7 "$ requires memorization about $" 5 + 2 = 7 "$ and $" 1 + 3 = 4 "$ , thus $" 1 5 + 3 2 = - > 4 "$ and $" 1 5 + 3 2 { = } 4  7 "$ are two "memorize" cases. "Change-one" cases require the change-one ability. For example, $" 1 5 + 3 7 =  5 "$ is a "change-one" case, as the output is based on $" 5 \mathrm { = } 1 \mathrm { + } 3 \mathrm { + } 1 "$ . For multiplication and division cases, we take the last token as "memorize" cases, and others as "change-one" cases. We compute the accuracy decrease between the original model and the intervened model for each operation. The results are shown in Table 3. If the heads only store change-one abilities, the decrease of "memorize" cases should be much smaller than "change-one" cases. However, the accuracy decrease of "memorize" cases and "change-one" cases are similar. Hence, we hypothesize the heads store parameters for memorizing 1D operations.

## 4 Comparative Neuron Analysis for Mechanistic Interpretability

In this section, we investigate how head $1 7 ^ { 2 2 }$ influence 1D+ and 1D- operations. Analysis of head $1 4 ^ { 1 9 }$ for 1D/ operations is shown in Appendix B, resulting the same stages with Section 4.2-4.4.

## 4.1 Methodology

The core idea of our proposed CNA method is comparing the same neuron across different models given the same input, or comparing the same neuron across different inputs within the same model. Due to the computational intensity of the forward pass, employing causal mediation analysis methods on every neuron is impractical. Therefore, we take the increase of log probability (Yu and Ananiadou, 2024b) as importance score for each neuron. The importance score of a FFN neuron $m _ { k } ^ { l } f c 2 _ { k } ^ { l }$ is $l o g ( p ( \bar { w } | x _ { T } ^ { l - 1 } + A _ { T } ^ { l } + m _ { k } ^ { l } f c 2 _ { k } ^ { l } ) ) - l o g ( p ( \stackrel { . . . } { w } | x _ { T } ^ { l - 1 } +$ $A _ { T } ^ { l } ) )$ , where w is the final predicted token and the probability is computed by the softmax function when multiplying the vectors with the unembedding matrix (Eq.4). Then we compute the change of each neuron’s importance score between the original model and the intervened model (intervening head $1 7 ^ { 2 2 } )$ ), and sort the change score to locate the most important neurons causing the final prediction probability decrease. We only intervene one head because this head can result very much decrease in accuracy. In later sections, we introduce the analysis process focusing on a specific case $" 3 + 5 = "$ , and devise various methods to prove these findings are applicable to all 1D+ and 1D- cases.

## 4.2 Feature Predicting via Arithmetic Head

For case $" 3 + 5 = "$ with prediction "8", we compute the importance score change for each neuron, and find the most important neurons are in FFN layers. We project these neurons in vocabulary space (Geva et al., 2022) by multiplying the FFN neurons v and unembedding matrix: $P _ { v } = s o f t m a x ( E _ { u } v )$ The top tokens when projecting into the unembedding space are shown in Table 4. $2 8 _ { 3 6 9 6 }$ means the 3696th neuron in the 28th FFN layer. "ori" and "inv" denote the original and the intervened model ("mdl"). "imp" and "coef" represent the importance score and coefficient score of each neuron.

<table><tr><td>FFNv</td><td>mdl</td><td>imp</td><td>coef</td><td>top10 tokens</td></tr><tr><td> $2 8 _ { 3 6 9 6 }$ </td><td>ori</td><td>0.82</td><td>6.21</td><td>[8, eight, VIII,</td></tr><tr><td>283696</td><td>inv</td><td>0.13 0.31</td><td>0.95 8.44</td><td>huit, acht, otto] [six, eight, acht,</td></tr><tr><td> $2 5 _ { 7 1 6 4 }$   $2 5 _ { 7 1 6 4 }$ </td><td>ori inv</td><td>0.07</td><td>2.08</td><td>Four, twelve, six, four, vier]</td></tr><tr><td> $1 9 _ { 5 7 6 9 }$ </td><td>ori</td><td>0.20</td><td>3.79</td><td>[eight, VIII, 8,</td></tr><tr><td> $1 9 _ { 5 7 6 9 }$ </td><td>inv</td><td>0.06</td><td>1.28</td><td>II, huit, acht]</td></tr></table>

Table 4: Importance scores and coefficient scores of located important FFN neurons for input $" 3 + 5 = "$

All these neurons contain concepts about "eight" and "8" in top tokens. The importance scores and coefficient scores drop a lot in the intervened model. From the interpretable results, we hypothesize that the reason why the accuracy decreases a lot in the intervened model is that head $1 7 ^ { 2 2 }$ stores important parameters for activating the important FFN neurons related to the final prediction. To verify this hypothesis, we conduct two experiments on all 1D+ and 1D- cases. For each case, we employ the CNA method to identify the important FFN neurons. Then in the original model we only intervene the most important FFN neurons ("mask") or intervene all the other FFN neurons within the 17th 31th layers ("keep"). The accuracy decrease on all 1D+ and 1D- cases is presented in Table 5.

<table><tr><td></td><td>top99</td><td>top50 top30</td><td>top20</td><td>top10</td></tr><tr><td>mask</td><td>100.0</td><td>96.0 89.5</td><td>86.8</td><td>68.4</td></tr><tr><td>keep</td><td>3.9</td><td>7.8 13.2</td><td>18.4</td><td>38.2</td></tr><tr><td>coef</td><td>49.1</td><td>60.4 67.2</td><td>72.7</td><td>77.1</td></tr></table>

Table 5: Decrease (%) of accuracy and coefficient score on all 1D+ and 1D- cases when intervening and keeping the most important FFN neurons.

When intervening the top99 FFN neurons, the accuracy decreases 100%. When intervening all the other neurons in deep FFN layers, the accuracy only decreases 3.9%. This suggests that almost all important information for predicting the final token is contained within the FFN neurons identified by our CNA method. We also report the decrease of the top neurons’ coefficient scores ("coef") between the intervened model and the original model in Table 5. In all situations, the coefficient scores drop much. Therefore, our hypothesis is verified: head $1 7 ^ { 2 2 }$ stores important parameters for activating the important FFN neurons related to final predictions. When head $1 7 ^ { 2 2 }$ is intervened, coefficient scores of important FFN neurons drop a lot, thus final predictions’ probabilities drop much.

## 4.3 Prediction Enhancing among Deep FFN Neurons

In case $" 3 + 5 = "$ , we observe that there is a prediction enhancing stage among the most important FFN neurons $2 8 _ { 3 6 9 6 } , 2 5 _ { 7 1 6 4 }$ and $1 9 _ { 5 7 6 9 }$ . The inner product scores between the FFN value of $1 9 _ { 5 7 6 9 }$ and the FFN keys of $2 5 _ { 7 1 6 4 }$ and $2 8 _ { 3 6 9 6 }$ are large. Additionally, the inner product between the FFN value of $2 5 _ { 7 1 6 4 }$ and the FFN key of 28<sub>3696</sub> is also large. Therefore, a prediction enhancing directed acyclic graph (PE-DAG) exists among the three neurons, where 19 is the root. Activation of the lower FFN neuron recursively triggers activations of upper semantic-related FFN neurons.

To explore whether the prediction enhancing stage also exists in other 1D+ and 1D- cases, we compute the coefficient score change of important FFN neurons when intervening the lowest neuron among the most important neurons. If there are many neurons in the lowest layer, we intervene the neuron with the largest importance score in the lowest layer. Decrease of coefficient score when intervening the lowest important neuron in the original model are shown in Table 6.

<table><tr><td></td><td>top99</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td>coef</td><td>15.8</td><td>14.8</td><td>12.5</td><td>9.5</td><td>4.4</td></tr></table>

Table 6: Decrease (%) of coefficient score when intervening the lowest neuron among important neurons.

Intervening only one neuron among top99 neurons can reduce the coefficient scores by 15.8%. The results indicate that the prediction enhancing stage exists among the identified deep FFN neurons. Among 1D+ and 1D- cases, comparing with intervening the lowest neuron among top10 and top20 neurons, the coefficient score decreases more when intervening the lowest neuron among top50 and top99 important neurons. This phenomenon maybe because the lowest neuron among top99 and top50 neurons typically resides on lower FFN layers compared to those on top10 and top20 neurons.

## 4.4 Feature Enhancing with Hidden-Interpretable Shallow FFN Neurons

Stolfo et al. (2023) utilize causal mediation analysis and find the model processes numbers and operators on early FFN layers and transfer into last position via attention layers. In this section, our objective is to locate the specific neurons fulfilling this function and to analyze the roles of shallow FFN layers and attention layers in this process. To identify the important shallow FFN neurons for case $" 3 + 5 = " - > " 8 "$ , we sort the neurons by computing the inner products between the PE-DAG root $1 9 _ { 5 7 6 9 }$ and the attention transformation of each FFN neuron. We find that the neurons (on residual streams of $" 3 "$ and "5") with highest inner products are hidden-interpretable. When projecting the original neurons into vocabulary space, they do not contain human-interpretable concepts in top tokens. However, after the transformation of attention layers, these neurons become interpretable. Moreover, we find that the word embeddings of "3" and "5" are also hidden-interpretable. The top vocabulary tokens of original and 15th attention layer transformation are shown in Table 7.

<table><tr><td>FFNv</td><td>origin</td><td colspan="2">attn transform</td></tr><tr><td>124072</td><td>[rd, quarters, PO, Constraint, ran, avas]</td><td>[III, Three, triple]</td><td>three, 3,</td></tr><tr><td>112258</td><td>[enz, Trace, lis, vid, suite, HT, ung, icano]</td><td>[XV, fifth, Fif, avas, Five, five, abase, fif]</td><td></td></tr><tr><td>word &quot;3&quot;</td><td>[rd, rum, quar- ters, Af, EX- ISTS, raum]</td><td>[three, Three, RGB, triple, 3, triangle]</td><td></td></tr><tr><td>word &quot;5&quot;</td><td>[th, esa, gi, AXI, gal, ides, Inject, abase, ipage, san, IDE]</td><td>vos, fif, fifth]</td><td>[Fif, XV, engo,</td></tr></table>

Table 7: Hidden-interpretable FFN neurons’ top10 tokens transformed by 15th attention layer.

We hypothesize that these hidden-interpretable FFN neurons are crucial for enhancing input features. We develop a zero-shot method to identify these hidden-interpretable shallow FFN neurons. For each FFN neuron on 0th 15th layer, we compute the transformation by $0 t h - 1 6 t h$ attention layers’ value-output matrices, and project these vectors into vocabulary space. If the top50 tokens contain M or more concepts related to numbers or operations, we add this neuron into a hiddeninterpretable neuron set. Then we intervene all the neurons in this neuron set in the original model, and compute the accuracy decrease on all 1D+ and 1D- cases. The number of neurons and accuracy under different M are shown in Table 8.

<table><tr><td></td><td>M=0</td><td>M=1</td><td>M=2</td><td>M=3</td></tr><tr><td>number</td><td>51,980</td><td>10,426</td><td>1,953</td><td>510</td></tr><tr><td>accuracy</td><td>98.7</td><td>68.4</td><td>53.9</td><td>43.4</td></tr></table>

Table 8: Decrease (%) of accuracy on 1D+ and 1Dcases when intervening hidden-interpretable neurons.

There are 176,128 neurons in $0 t h - 1 5 t h$ FFN layers. Intervening with only 1,953 neurons (M=2) results in a decrease of 53.9%. This strongly suggests that these hidden-interpretable neurons play a significant role in enhancing features and are valuable for final predictions. Further supporting this notion is the observation that randomly intervening 1,953 neurons on the $0 t h - 1 5 t h$ FFN layers only results in an accuracy decrease of 2.6%. Compared to directly interpretable neurons in deep FFN layers, hidden-interpretable neurons in shallow FFN layers are more widely distributed. When intervening 10,426 neurons (about 6% of all neurons in 0th 15th layers), the accuracy decreases 68.4%.

## 4.5 Constructing the Internal Logic Chain from Inputs to Prediction

In Section 4.2-4.4, we apply our CNA method to identify the important neurons for the case "3+5", and also design experiments to verify the generality across other 1D+ and 1D- cases. In this section, we conclude the internal logic chain from inputs to prediction for case $" 3 + 5 = " > " 8 "$

First, in feature enhancing stage, shallow FFN neurons containing hidden-interpretable features (e.g. 11<sub>2258</sub>, 12<sub>4072</sub>) are extracted. In feature transferring stage, the hidden-interpretable features (word embeddings and shallow FFN neurons) are transformed into directly-interpretable features by attention layers and then transferred to the last position. In feature predicting stage, head $1 7 ^ { 2 2 }$ activates deep FFN neurons associated with the concept of $" 8 " ( \mathrm { e . g . ~ 2 8 _ { 3 6 9 6 } , 2 5 _ { 7 1 6 4 } , 1 9 _ { 5 7 6 9 } } )$ based on the enhanced features. Finally, in the prediction enhancing stage, lower FFN neurons activate higher FFN neurons, which collectively contribute to the probability of $" 8 "$ in the final prediction.

Through our CNA method, we precisely identify crucial parameters (attention heads and FFN neurons) for predicting final tokens. Compared to prior studies, our approach enables the discovery of more detailed locations and offers a clearer explanation of the information flow. Given our method’s ability to pinpoint precise parameters, it can be effectively leveraged for downstream tasks such as model pruning and model editing, which we discuss in Section 6.

## 5 Understanding the Mechanism of LoRA

LoRA (Hu et al., 2021) is a commonly used parameter-efficient fine-tuning method (Houlsby et al., 2019; Li and Liang, 2021; Lester et al., 2021). By adding trainable low-rank matrices into attention layers, models are fine-tuned with only 0.5% additional parameters, yielding favorable outcomes. Intuitively, LoRA is similar to a head. Inspired by the analysis on arithmetic heads, we apply the CNA method to understand the mechanism of LoRA.

We first investigate whether LoRA plays distinct roles when added into various layers. We fine-tune 32 models on the 2-digit arithmetic dataset, with each model incorporating a low-rank matrix into a distinct attention layer. Notably, we introduce negative numbers in 2D cases such as $" 3 - 5 = - 2 "$ as the original Llama model does not learn this concept well. The training and testing set consist of 18,000 and 2,000 sentences, respectively. We determine the optimal learning rate from choices of 0.001, 0.0005, and 0.0001. The maximum epoch is set to 4. The results are depicted in Figure 2.

![](images/5201d2e4933f47dbd98406afe7a315acfe8647f8ae5a72e12691523c06fe2b90.jpg)  
Figure 2: Accuracy: adding LoRA in different layers. All the fine-tuned models exhibit superior ac-

curacy compared to the original model (62.96%). The 0th and the 31th layer may have special use, since the accuracy of the 0th and 31th models differs much from their neighboring models. The accuracy of the $1 s t - 9 t h$ models is around 90%. Starting from the 10th model, the accuracy keeps decreasing. The average slope during the 10th to 16th models differs from that of the 17th to 30th models. Motivated by LoRA’s accuracy curve and the analysis of arithmetic heads, we hypothesize that LoRA enhances the correct predictions’ probabilities by amplifying the deep FFN neurons related to final predictions. We apply our CNA method on the original model and five LoRA models analyzing the case $" 3 + 5 = "$ , detailed in Table 9.

<table><tr><td></td><td>ori</td><td>9th</td><td>15th</td><td>16th</td><td>19th</td><td>20th</td></tr><tr><td> $2 8 _ { 3 6 9 6 }$ </td><td>6.2</td><td>3.6</td><td>6.3</td><td>3.9</td><td>5.7</td><td>4.1</td></tr><tr><td> $2 5 _ { 7 1 6 4 }$ </td><td>8.4</td><td>16.1</td><td>11.8</td><td>11.0</td><td>13.9</td><td>9.7</td></tr><tr><td> $1 9 _ { 5 7 6 9 }$ </td><td>3.8</td><td>9.2</td><td>7.7</td><td>6.1</td><td>5.1</td><td>3.8</td></tr></table>

Table 9: Important neurons’ coefficient scores on the original model and five fine-tuned models for $" 3 + 5 = "$

Across all five fine-tuned models, the coefficient scores of $2 5 _ { 7 1 6 4 }$ and 19<sub>5769</sub> surpass those of the original model. The scores are higher in shallowlayer models compared to deep-layer models. The significant decrease in the coefficient score observed in $2 5 _ { 7 1 6 4 }$ in the 20th model can be attributed to its failure to leverage the features of $1 9 _ { 5 7 6 9 }$
<table><tr><td>LoRA layer</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td> $1 s t - 9 t h$ </td><td>42%</td><td>49%</td><td>57%</td><td>59%</td></tr><tr><td> $1 0 t h - 1 6 t h$ </td><td>29%</td><td>36%</td><td>44%</td><td>53%</td></tr><tr><td>17th − 30th</td><td>2%</td><td>11%</td><td>14%</td><td>28%</td></tr></table>

Table 10: Coefficient score increase (%) of different fine-tuned models compared with the original model.

For all cases, we compute the average coefficient score increase of $1 s t - 9 t h , 1 0 t h - 1 6 t h$ , and 17th 30th models on the most important neurons, detailed in Table 10. Across all scenarios, the coefficient scores of significant FFN neurons surpass those of the original model. Notably, fine-tuning LoRA in shallow layers yields a greater amplification of FFN neurons’ coefficient scores compared to deep layers. This observation validates our hypothesis: LoRA enhances the probabilities of final predictions by amplifying the coefficient scores of deep FFN neurons relevant to final predictions.

## 6 Applications

In this section, we utilize our method for model pruning on arithmetic tasks and for model editing aimed at mitigating gender bias.

## 6.1 Model Pruning for Arithmetic Tasks

As recent powerful models boast tens of billions of parameters, the extraction of sub-networks from these large models for various downstream tasks has become crucial. This approach is based on the assumption that only a small subset of parameters in an over-parameterized model are pertinent to a specific task and similar tasks share similar sub-networks (Pfeiffer et al., 2023). Recent works (Stanczak et al.´ , 2022; Foroutan et al., 2022) in multilingual models can support these hypotheses.

In this section, we apply our findings on model pruning for arithmetic tasks. As discussed in Section 4, important information for final predictions is concentrated in only a few deep FFN neurons. Therefore, we design a simple method to prune useless neurons in deep FFN layers. We apply our CNA method between the original model and the 9th LoRA model on all the 1D+, 1D-, $1 \mathrm { D ^ { * } }$ and 1D/ cases, to find the important top500 neurons for each case. Then we prune all the other FFN neurons among $1 7 t h - 3 1 t h$ layers, thus only 5% deep FFN neurons are saved in the pruned model. Finally, we add LoRA on the 9th layer of the pruned model, and fine-tune on the training set. The parameters on deep FFN layers are reduced to 5%, and only 0.015% LoRA parameters are added.

<table><tr><td></td><td>origin</td><td>LoRA9</td><td>LoRA9-p</td><td>LoRA9-r</td></tr><tr><td>acc</td><td>62.9</td><td>89.3</td><td>82.3</td><td>17.1</td></tr></table>

Table 11: Accuracy on 2-digit datasets.

The results are shown in Table 11. The accuracy of the fine-tuned pruned model (LoRA9- p) is 82.3%, better than original Llama (62.9%). While our method do not reach the performance of the fine-tuned model without pruning (LoRA9), it still offers a promising avenue for model pruning. Furthermore, although 2-digit arithmetic is an easy task, fine-tuning LoRA on a randomly-pruned model (LoRA9-r) with the same number of neurons fails to yield satisfactory results (only 17.1%). This further underscores the significance of our method.

## 6.2 Model Editing for Reducing Gender Bias

Even though LLMs have achieved great success, they can learn, perpetuate, and amplify harmful social biases (Gallegos et al., 2023). In this section, we focus on gender bias, which is observed in different models (de Vassimon Manela et al., 2021; Kotek et al., 2023). We apply our CNA method analyzing similar cases with different genders in the same model. For example, we identify the important neurons for predicting "nurse" by calculating the change of importance scores between sentences "A woman works as a" and "A man works as a". Since the other words are the same except "woman" and "man", these neurons contain much gender bias causing p(nurse woman) > p(nurse man). The neurons’ top tokens of are shown in Table 12. For example, the top tokens of FFN neuron 19<sub>8436</sub> are all professions. Under the input "A woman works as a", this neuron’s coefficient score is 3.39. While the neuron’s coefficient score is only 0.14 activated by "A man works as a", proving that this neuron contains much gender bias.

<table><tr><td>FFNv</td><td>gend</td><td>imp</td><td>coef</td><td>top tokens</td></tr><tr><td> $2 2 _ { 2 6 5 1 }$ </td><td>F</td><td>0.24</td><td>6.48</td><td>[maid, domestic,</td></tr><tr><td> $2 2 _ { 2 6 5 1 }$ </td><td>M</td><td>0.06</td><td>1.53</td><td>servant, servitor]</td></tr><tr><td> $1 9 _ { 8 4 3 6 }$ </td><td>F</td><td>0.16</td><td>3.39</td><td>[nurse, secretary,</td></tr><tr><td> $1 9 _ { 8 4 3 6 }$ </td><td>M</td><td>0.01</td><td>0.14</td><td>typing, reception]</td></tr></table>

Table 12: FFN neurons contain gender bias. "F":woman.

We then apply our CNA method on 32 common professions contain gender bias (detailed in Appendix D). Designing four prompts, we identify top18 important FFN neurons and edit them by setting their parameters to zero. The average perplexity difference log(p(prof gend1)) log(p(prof gend2)) is shown in Table 13, reduced by 35.7% when only 18 neurons are edited.

<table><tr><td></td><td>total bias</td><td>woman bias</td><td>man bias</td></tr><tr><td>origin</td><td>1.26</td><td>1.45</td><td>1.08</td></tr><tr><td>edited</td><td>0.81</td><td>1.04</td><td>0.59</td></tr></table>

Table 13: Gender bias of original and edited model.

These results can demonstrate that our proposed CNA method can be utilized in different tasks. It is also important to note that the utilized gender bias datasets may not comprehensively represent general scenarios. We leave the explorations on different datasets in future work.

## 7 Discussion and Conclusion

We aim to discuss the mechanisms behind causal mediation analysis and static interpretation methods. Causal mediation analysis methods can find the "root cause" (head $1 7 ^ { 2 2 } )$ of the probability change, which are usually not interpretable. Static methods can locate the interpretable "direct cause" (FFN neurons), but many elements can activate these neurons. Our CNA method can locate both "root cause" and "direct cause", and reconstruct the whole logic chain from inputs to prediction.

Overall, we identify the important attention heads and FFN neurons for arithmetic operations. We propose the comparative neuron analysis (CNA) method and construct the internal logic chain from inputs to prediction, including the feature enhancing stage, feature transferring stage, feature predicting stage, and prediction enhancing stage. Based on these findings, we find LoRA increases the final predictions’ probabilities by enlarging the important FFN neurons’ coefficient scores. Finally, we apply our method and findings on model pruning for arithmetic tasks, and model editing for reducing gender bias. Our method and analysis offer a comprehensive insight for understanding LLM.

## Limitations

The case studies rely on projecting vectors in vocabulary space, which is widely used in previous studies (Elhage et al., 2021; Ram et al., 2022; Geva et al., 2022; Dar et al., 2022). While the results are empirically interpretable, the theories of this method are incomplete. Therefore, we utilize this method in our case studies and supplement our findings with additional methods to strengthen our conclusions, thus enhancing their persuasiveness.

Another limitation lies in the lack of standardization across various studies regarding attribution methods. Different intervention methods (zero intervention, noise intervention, replace intervention, etc.) may get different results. Apart from causal mediation analysis methods and static interpretation methods, gradient-based methods (Sundararajan et al., 2017) and SHAP values (Lundberg and Lee, 2017) are also widely utilized for attributing important modules. However, these methods often demand substantial computational resources, rendering them unsuitable for our work.

A potential risk of our work is that attackers can identify the important neurons and edit these neurons to change the output probability distribution.

For instance, instead of reducing the gender bias by setting the neurons’ parameters to zero, they can amplify the gender bias professions’ probabilities by enlarging the identified neurons in Section 6.2. Hence, it is important to distinguish whether a model is edited, and we leave this exploration in future work.

## 8 Acknowledgements

This work is supported by the project JPNP20006 from New Energy and Industrial Technology Development Organization (NEDO). This work is supported by the computational shared facility and the studentship from the Department of Computer Science at the University of Manchester.

## References

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. arXiv preprint arXiv:1607.06450.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2021. Knowledge neurons in pretrained transformers. arXiv preprint arXiv:2104.08696.

Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. 2022. Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535.

Daniel de Vassimon Manela, David Errington, Thomas Fisher, Boris van Breugel, and Pasquale Minervini. 2021. Stereotype and skew: Quantifying gender bias in pre-trained and fine-tuned language models. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2232–2242.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. 2021. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1:1.

Negar Foroutan, Mohammadreza Banaei, Rémi Lebret, Antoine Bosselut, and Karl Aberer. 2022. Discovering language-neutral sub-networks in multilingual language models. arXiv preprint arXiv:2205.12672.

Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. 2023. Bias and fairness in large language models: A survey. arXiv preprint arXiv:2309.00770.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. arXiv preprint arXiv:2304.14767.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. arXiv preprint arXiv:2203.14680.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2020. Transformer feed-forward layers are keyvalue memories. arXiv preprint arXiv:2012.14913.

Rhys Gould, Euan Ong, George Ogden, and Arthur Conmy. 2023. Successor heads: Recurring, interpretable attention heads in the wild. arXiv preprint arXiv:2312.09230.

Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. 2023. Finding neurons in a haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610.

Michael Hanna, Ollie Liu, and Alexandre Variengien. 2023. How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. In Thirty-seventh Conference on Neural Information Processing Systems.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Hadas Kotek, Rikker Dockum, and David Sun. 2023. Gender bias and stereotypes in large language models. In Proceedings ofThe ACM Collective Intelligence Conference, pages 12–24.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Tom Lieberum, Matthew Rahtz, János Kramár, Geoffrey Irving, Rohin Shah, and Vladimir Mikulik. 2023. Does circuit analysis interpretability scale? evidence from multiple choice capabilities in chinchilla. arXiv preprint arXiv:2307.09458.

Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022a. Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems, 35:17359–17372.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. 2022b. Massediting memory in a transformer. arXiv preprint arXiv:2210.07229.

Neel Nanda, Senthooran Rajamanoharan, Janos Kramar, and Rohin Shah. 2023. Fact finding: Attempting to reverse-engineer factual recall on the neuron level. In Alignment Forum, page 6.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. 2022. In-context learning and induction heads. arXiv preprint arXiv:2209.11895.

Andreas Opedal, Alessandro Stolfo, Haruki Shirakami, Ying Jiao, Ryan Cotterell, Bernhard Schölkopf, Abulhair Saparov, and Mrinmaya Sachan. 2024. Do language models exhibit the same cognitive biases in problem solving as human learners? arXiv preprint arXiv:2401.18070.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Judea Pearl. 2001. Direct and indirect effects. Probabilistic and Causal Inference: The Works of Judea Pearl, page 373.

Shuai Peng, Ke Yuan, Liangcai Gao, and Zhi Tang. 2021. Mathbert: A pre-trained model for mathematical formula understanding. arXiv preprint arXiv:2105.00377.

Jonas Pfeiffer, Sebastian Ruder, Ivan Vulic, and´ Edoardo Maria Ponti. 2023. Modular deep learning. arXiv preprint arXiv:2302.11529.

Philip Quirke et al. 2023. Understanding addition in transformers. arXiv preprint arXiv:2310.13121.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Ori Ram, Liat Bezalel, Adi Zicher, Yonatan Belinkov, Jonathan Berant, and Amir Globerson. 2022. What are you token about? dense retrieval as distributions over the vocabulary. arXiv preprint arXiv:2212.10380.

Karolina Stanczak, Edoardo Ponti, Lucas Torroba Hen-´ nigen, Ryan Cotterell, and Isabelle Augenstein. 2022. Same neurons, different languages: Probing morphosyntax in multilingual pre-trained models. arXiv preprint arXiv:2205.02023.

Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. 2023. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7035–7052.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319– 3328. PMLR.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388– 12401.

Ben Wang and Aran Komatsuzaki. 2021. Gpt-j-6b: A 6 billion parameter autoregressive language model.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. 2022. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593.

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. 2023. Label words are anchors: An information flow perspective for understanding in-context learning. arXiv preprint arXiv:2305.14160.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Zeping Yu and Sophia Ananiadou. 2024a. How do large language models learn in-context? query and key matrices of in-context heads are two towers for metric learning. arXiv preprint arXiv:2402.02872.

Zeping Yu and Sophia Ananiadou. 2024b. Neuronlevel knowledge attribution in large language models. arXiv preprint arXiv:2312.12141.

Wei Zhang, Wan Chaoqun, Yonggang Zhang, Xinmei Tian, Xu Shen, and Jieping Ye. 2023. Interpreting the inner mechanisms of large language models in mathematical addition.

Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. 2024. The clock and the pizza: Two stories in mechanistic explanation of neural networks. Advances in Neural Information Processing Systems, 36.

## A Four Prompts in Arithmetic Dataset

<table><tr><td>type</td><td>prompt</td></tr><tr><td>addition-1 addition-2 addition-3 addition-4</td><td>The sum of n1 and n2 is Q: What is n1 plus n2? A: n1 plus n2 is  $\mathrm { n } 1 + \mathrm { n } 2 =$ </td></tr><tr><td>subtract-1 subtract-2 subtract-3 subtract-4</td><td>The difference between n1 and n2 is Q: What is n1 minus n2? A: n1 minus n2 is  $\mathrm { n } 1 - \mathrm { n } 2 =$ </td></tr><tr><td>multiply-1 multiply-2 multiply-3 multiply-4</td><td>The product of n1 and n2 is Q: What is n1 times n2? A: n1 times n2 is n1  $^ { * } { \bf n } 2 =$ </td></tr><tr><td>division-1 division-2 division-3 division-4</td><td>The ratio of n1 and n2 is Q: What is n1 divides n2? A: n1 divides n2 is n1  $/ { \mathfrak { n } } 2 =$ </td></tr></table>

Table 14: Four prompts for 2-digit arithmetic operations.

## B Results of Interventions on Head $1 4 ^ { 1 9 }$

We conduct the same experiments as discussed in Section 4.2-4.4. The results of head $1 4 ^ { 1 9 }$ is shown in Table 15 (corresponding to Table 5), Table 16 (corresponding to Table 6), and Table 17 (corresponding to Table 8).

<table><tr><td></td><td>top99</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td>mask</td><td>84.6</td><td>82.1</td><td>74.4</td><td>66.7</td><td>51.3</td></tr><tr><td>keep</td><td>48.7</td><td>51.3</td><td>53.9</td><td>53.9</td><td>64.2</td></tr><tr><td>coef</td><td>50%</td><td>61%</td><td>67%</td><td>70%</td><td>73%</td></tr></table>

Table 15: Decrease (%) of accuracy and coefficient score on all 1D/ cases when masking and keeping the top FFN neurons.

In Table 15, when head $1 4 ^ { 1 9 }$ is intervened, coefficient scores of important neurons in deep FFN layers are reduced, causing the accuracy decrease. Also, the top identified neurons contain much information. Interventions on top99 neurons result in an accuracy decrease of 84.6%.
<table><tr><td></td><td>top99</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td>coef</td><td>1.3</td><td>0.9</td><td>3.2</td><td>4.9</td><td>7.0</td></tr></table>

Table 16: Decrease (%) of coefficient score when intervening the lowest neuron among important FFN neurons.

The results of Table 16 also demonstrate that among the identified important neurons, the lower neurons can enhance higher neurons’ coefficient scores among deep FFN neurons. Therefore, the prediction enhancing stage also exists.

<table><tr><td></td><td>M=0</td><td>M=1</td><td>M=2</td><td>M=3</td></tr><tr><td>number</td><td>51,980</td><td>10,426</td><td>1,953</td><td>510</td></tr><tr><td>acc</td><td>97.5</td><td>82.1</td><td>30.8</td><td>25.7</td></tr></table>

Table 17: Decrease (%) of accuracy on 1D/ cases when intervening hidden-interpretable neurons.

In Table 17, The hidden-interpretable neurons in shallow FFN layers are important for 1D/ cases $( \mathbf { e . g . \ ^ { \prime \prime } } 7 2 / 8 { = } " )$ . When intervening 10,426 hiddeninterpretable shallow FFN neurons, the accuracy reduces 82.1%. For comparison, we randomly intervene 10,426 FFN neurons in shallow FFN layers, and the interventions only cause a decrease of 5.1%.

Overall, head $1 4 ^ { 1 9 }$ shares the same mechanism with head $1 7 ^ { 2 2 }$ . Head $1 4 ^ { 1 9 }$ stores important parameters for division operations, while head $\bar { 1 7 } ^ { 2 2 }$ is responsible for addition and subtraction.

## C Results of Interventions in GPT-J

We conduct the same experiments in Section 3.3 in GPT-J. The accuracy when intervening each head is presented in Table 18.

<table><tr><td></td><td>ori</td><td> $7 ^ { 0 }$ </td><td> $1 3 ^ { 9 }$ </td><td> $0 ^ { 1 1 }$ </td><td> $1 5 ^ { 6 }$ </td><td> $1 4 ^ { 1 4 }$ </td></tr><tr><td>all</td><td>74.5</td><td>63.6</td><td>64.4</td><td>65.0</td><td>68.8</td><td>70.8</td></tr><tr><td>2D+</td><td>97.0</td><td>95.0</td><td>94.0</td><td>98.0</td><td>95.0</td><td>97.0</td></tr><tr><td>2D-</td><td>78.6</td><td>63.6</td><td>41.8</td><td>63.6</td><td>74.5</td><td>80.0</td></tr><tr><td>2D*</td><td>71.0</td><td>54.0</td><td>72.0</td><td>53.0</td><td>59.0</td><td>72.0</td></tr><tr><td>2D/</td><td>51.5</td><td>42.0</td><td>50.0</td><td>45.6</td><td>46.7</td><td>34.4</td></tr></table>

Table 18: Accuracy (%) when intervening different heads in GPT-J.

In GPT-J, we also observe that different heads store important parameters for various operations. For instance, the accuracy of 2D- decreases significantly when intervening in head $1 3 ^ { 9 }$ , whereas head $1 4 ^ { 1 4 }$ holds significant parameters for 2D/.

Then we apply the CNA method between the original model and the intervened model on head $1 3 ^ { 9 }$ on 2D- cases. The results are shown in Table 19 (corresponding to Table 5), Table 20 (corresponding to Table 6), and Table 21 (corresponding to Table 8).

<table><tr><td></td><td>top99</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td>mask</td><td>58.4</td><td>45.9</td><td>33.4</td><td>25</td><td>25</td></tr><tr><td>keep</td><td>37.5</td><td>50</td><td>54.1</td><td>83.4</td><td>95.9</td></tr><tr><td>coef</td><td>17%</td><td>20%</td><td>21%</td><td>23%</td><td>29%</td></tr></table>

Table 19: Decrease (%) of accuracy and coefficient score when masking and keeping the top FFN neurons.
<table><tr><td></td><td>top99</td><td>top50</td><td>top30</td><td>top20</td><td>top10</td></tr><tr><td>coef</td><td>1.9</td><td>1.7</td><td>1.6</td><td>1.2</td><td>1.9</td></tr></table>

Table 20: Decrease (%) of coefficient score when intervening the lowest neuron among important neurons.

In Table 19, the top FFN neurons also play a large role in GPT-J. When intervening the top99 neurons, the accuracy decreases 58.4%. Compared with Llama, the degrees of coefficient decrease and accuracy change are both smaller. In Table 20, when intervening the lowest neuron among the important neurons identified by our CNA method, the deep neurons’ coefficient scores decrease.

<table><tr><td></td><td>M=0</td><td>M=1</td><td>M=2</td><td>M=3</td></tr><tr><td>number</td><td>一</td><td>4,272</td><td>1,228</td><td>564</td></tr><tr><td>acc</td><td>-</td><td>100.0</td><td>83.4</td><td>20.8</td></tr></table>

Table 21: Decrease (%) of accuracy when intervening hidden-interpretable neurons.

Results in Table 21 indicate that the hiddeninterpretable shallow FFN neurons also exist in GPT-J. When intervening 4,272 neurons, the accuracy decreases 100%.

Overall, we observe similar results in GPT-J. Similar to Llama, GPT-J also exhibits the presence of four stages: feature enhancing, feature transferring, feature predicting, and prediction enhancing.

## D Details for Evaluating Gender Bias

We design eight prompts to find the most common professions causing the gender bias. The prompts are shown in Table 22, where <gend> is "man" or "woman".

We compute the top100 predictions of each prompt for different genders, and compare the different professions, which are shown in Table 23. These professions contain much gender bias. We then apply our CNA method between cases with different genders under the first prompt, and identify the top18 important neurons causing the difference. Finally, we edit the top18 neurons by setting their parameters to zero, and then compute the perplexity difference between different genders for all prompts in both the original and edited model (results are shown in Table 13).

<table><tr><td>prompt</td></tr><tr><td>A &lt;gend&gt; works as a A &lt;gend&gt; is employed as a</td></tr><tr><td>A &lt;gend&gt; holds a job as a</td></tr><tr><td>A &lt;gend&gt;&#x27;s occupation is</td></tr><tr><td>The job of a &lt;gend&gt; is The work of a &lt;gend&gt; is</td></tr><tr><td>The profession of a &lt;gend&gt; is The work of a &lt;gend&gt; involves</td></tr></table>

Table 22: Eight prompts for gender bias professions.

<table><tr><td>gend</td><td>profession</td></tr><tr><td>woman</td><td>cleaner, nurse, secretary, domestic helper, maid, reception, seller, server, librarian, pharmacist, translator, beauti- cian, dental assistant, hairdresser, vol-</td></tr><tr><td>man</td><td>unteer, bookkeeper police, guard, delivery, labour, driver, machinist, roofer, machine operator, lumberjack, technician, miner, night- watch, painter, photographer, builder,</td></tr></table>

Table 23: Professions with gender bias.

## E Important Heads for 1-Digit, 2-Digit and 3-Digit Operations

We report the top5 important heads for 1D, 2D and 3D operations in this section. For each operation, the experiments are conducted on the last prompt in Table 14. The results are shown in Table 24-27.

<table><tr><td>1D+</td><td>ori 88.9</td><td> $1 7 ^ { 2 2 }$  47.6</td><td> $1 5 ^ { 2 3 }$  82.8</td><td> $6 ^ { 2 0 }$  84.1</td><td> $1 3 ^ { 1 5 }$  84.1</td><td> $1 4 ^ { 1 9 }$  84.1</td></tr><tr><td>2D+</td><td>ori 94.5</td><td>1722 39.3</td><td>159 86.0</td><td>132 87.9</td><td>620 88.6</td><td>1216 89.2</td></tr><tr><td></td><td>ori</td><td> $1 7 ^ { 2 2 }$ </td><td> $8 ^ { 1 0 }$ </td><td> $1 5 ^ { 2 3 }$ </td><td>1216</td><td>159</td></tr><tr><td>3D+</td><td>96.1</td><td>46.4</td><td>82.7</td><td>83.5</td><td>85.2</td><td>87.2</td></tr><tr><td>1D-</td><td>ori 82.0</td><td> $1 7 ^ { 2 2 }$   $\bf { 3 1 . 0 }$ </td><td> $1 6 ^ { 1 }$  51.0</td><td> $1 5 ^ { 2 3 }$   $5 3 . 0$ </td><td> $2 ^ { 2 6 }$   $5 7 . 0$ </td><td> $1 3 ^ { 2 }$   $6 5 . 0$ </td></tr><tr><td>2D-</td><td>ori 80.0</td><td> $1 6 ^ { 1 }$  33.9</td><td> $1 7 ^ { 2 2 }$  37.9</td><td> $1 3 ^ { 2 }$  61.8</td><td> $1 5 ^ { 2 3 }$  63.3</td><td> $1 2 ^ { 1 6 }$  70.6</td></tr><tr><td></td><td>ori</td><td> $1 6 ^ { 1 }$ </td><td> $1 7 ^ { 2 2 }$ </td><td> $1 5 ^ { 2 3 }$ </td><td> $1 3 ^ { 2 }$ </td><td> $1 2 ^ { 1 6 }$ </td></tr><tr><td>3D-</td><td>57.1</td><td>19.6</td><td>22.9</td><td>29.3</td><td>34.3</td><td>40.7</td></tr></table>

Table 24: Results of most important heads for 1D+, 2D+, and 3D+.

Table 25: Results of most important heads for 1D-, 2D-, and 3D-.

<table><tr><td>1D*</td><td>ori 93.0</td><td> $3 ^ { 5 }$   $\mathbf { 8 5 . 4 }$ </td><td> $2 0 ^ { 1 8 }$   $\mathbf { 8 6 . 7 }$ </td><td> $6 ^ { 2 4 }$   $8 7 . 3$ </td><td> $1 7 ^ { 2 2 }$   $8 9 . 2$ </td><td> $0 ^ { 3 0 }$   $8 9 . 2$ </td></tr><tr><td>2D*</td><td>ori 56.9</td><td> $1 5 ^ { 9 }$  49.3</td><td> $1 4 ^ { 1 9 }$  50.1</td><td> $1 7 ^ { 2 2 }$  50.5</td><td> $2 0 ^ { 1 8 }$  50.5</td><td> $3 ^ { 5 }$  51.6</td></tr><tr><td></td><td>ori</td><td> $3 ^ { 5 }$ </td><td> $1 5 ^ { 9 }$ </td><td> $1 4 ^ { 1 9 }$ </td><td> $1 3 ^ { 1 9 }$ </td><td> $2 ^ { 1 4 }$ </td></tr><tr><td>3D*</td><td>32.8</td><td>25.9</td><td>29.7</td><td>30.3</td><td>31.1</td><td>31.1</td></tr></table>

Table 26: Results of most important heads for 1D\*, 2D\*, and 3D\*.

<table><tr><td>1D/</td><td>ori 78.9</td><td> $1 4 ^ { 1 9 }$   $\mathbf { 3 5 . 6 }$ </td><td> $1 5 ^ { 9 }$  61.1</td><td> $2 1 ^ { 2 4 }$   $6 5 . 6$ </td><td> $6 ^ { 2 4 }$   $6 7 . 8$ </td><td> $1 6 ^ { 2 1 }$   $6 8 . 9$ </td></tr><tr><td>2D/</td><td>ori 48.6</td><td> $1 4 ^ { 1 9 }$  13.7</td><td> $1 2 ^ { 1 }$  30.2</td><td> $3 ^ { 3 }$  31.4</td><td> $1 6 ^ { 2 1 }$  36.9</td><td> $1 ^ { 2 9 }$  38.8</td></tr><tr><td></td><td>ori</td><td> $1 4 ^ { 1 9 }$ </td><td> $3 ^ { 3 }$ </td><td> $1 2 ^ { 1 }$ </td><td> $6 ^ { 2 2 }$ </td><td> $1 5 ^ { 9 }$ </td></tr><tr><td>3D/</td><td>19.0</td><td>9.67</td><td>12.7</td><td>13.0</td><td>13.3</td><td>13.7</td></tr></table>

Table 27: Results of most important heads for 1D/, 2D/, and 3D/.