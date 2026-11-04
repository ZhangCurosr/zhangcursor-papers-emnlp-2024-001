# Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence

Junru Lu<sup>1</sup>∗<sup>,3</sup>, Jiazheng Li<sup>2</sup>\*, Siyu An<sup>3</sup>, Meng Zhao<sup>3</sup>, Yulan He<sup>1,2,4</sup>, Di Yin<sup>3</sup>, Xing Sun<sup>3</sup>

<sup>1</sup>University of Warwick <sup>2</sup>King’s College London

<sup>3</sup>Tencent YouTu Lab <sup>4</sup>The Alan Turing Institute

junru.lu@warwick.ac.uk, {jiazheng.li, yulan.he}@kcl.ac.uk {siyuan, alexmzhao, endymecyyin, winfredsun}@tencent.com

## Abstract

Direct Preference Optimization (DPO) has emerged as a prominent algorithm for the direct and robust alignment of Large Language Models (LLMs) with human preferences, offering a more straightforward alternative to the complex Reinforcement Learning from Human Feedback (RLHF). Despite its promising efficacy, DPO faces a notable drawback: “verbosity”, a common over-optimization phenomenon also observed in RLHF. While previous studies mainly attributed verbosity to biased labels within the data, we propose that the issue also stems from an inherent algorithmic length reliance in DPO. Specifically, we suggest that the discrepancy between sequencelevel Kullback–Leibler (KL) divergences between chosen and rejected sequences, used in DPO, results in overestimated or underestimated rewards due to varying token lengths. Empirically, we utilize datasets with different label lengths to demonstrate the presence of biased rewards. We then introduce an effective downsampling approach, named SamPO, to eliminate potential length reliance. Our experimental evaluations, conducted across three LLMs of varying scales and a diverse array of conditional and open-ended benchmarks, highlight the efficacy of SamPO in mitigating verbosity, achieving improvements of 5% to 12% over DPO through debaised rewards<sup>1</sup>.

## 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) is a crucial strategy for effectively align Large Language Models (LLMs) with human minds (Zhao et al., 2023a; Yang et al., 2023; Pan et al., 2023b), showcasing significant improvements of LLM’s instruct-following capability compared with the other two popular approaches: pretraining and supervised fine-tuning (SFT). In fact, a series of leading LLMs have adopted RLHF as the final stage of their entire training pipelines (Ouyang et al., 2022; Achiam et al., 2023; Bi et al., 2024).

Nevertheless, traditional RLHF involves several intricate multi-stage steps, typically starting with fine-tuning a reward model that captures complex human intuition (Bai et al., 2022), followed by optimizing LLMs to maximize preference scores. Therefore, the quality of the reward model is crucial. However, modeling elusive human intuition is inherently difficult (Wang et al., 2024). On the contrary, Direct Preference Optimization (DPO) (Rafailov et al., 2023) proposed to re-parameterize the reward model, integrating preference feedback from online rewards into offline labels. In specific, DPO employs the Bradley-Terry model (Bradley and Terry, 1952) to maximize implicit rewards via pairwise offline preference labels. The implicit reward is mathematically equivalent to the discrepancy in sequencelevel Kullback–Leibler (KL) divergences (Kullback and Leibler, 1951) between chosen and rejected labels. The KL divergence for each label is calculated based on probability outputs from the finetuning policy model and a frozen reference model. DPO eliminates the need for complex prefix finetuning of an external reward model, while maintains performance comparable to RLHF (Dubois et al., 2024b; Hou et al., 2024).

Despite its effectiveness, DPO faces several notable challenges, including issues of overfitting (Azar et al., 2023; Jung et al., 2024), high computational costs (Ethayarajh et al., 2024; Hong et al., 2024), and verbosity (Hou et al., 2024; Park et al., 2024). This paper specifically focuses on addressing the “verbosity” issue.

Traditional multi-stage RLHF methods argue that due to a statistical bias in length distribution, that is, where preferred labels tend to be longer than rejected preference labels (Singhal et al., 2023; Park et al., 2024), the reward model trained on

$\begin{array} { r } { \mathcal { L } _ { \mathrm { D P 0 } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } \left[ \log \sigma \left( \beta \mathrm { l o g } \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | x ) } - \beta \mathrm { l o g } \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | x ) } \right) \right] } \end{array}$ , formalized with sequence—level KL divergence $\begin{array} { r } { \mathcal { L } _ { \mathrm { D P 0 } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } \left[ \log \sigma \left( \beta \sum _ { t = 1 } ^ { T _ { w _ { 1 } } } \log \frac { \pi _ { \theta } \left( y _ { w } ^ { t } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { w } ^ { t } | x \right) } - \beta \sum _ { t = 1 } ^ { T _ { l } } \log \frac { \pi _ { \theta } \left( y _ { l } ^ { t } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { l } ^ { t } | x \right) } \right) \right] } \end{array}$ , decomposing KL divergence at token-level $\begin{array} { r } { \mathcal { L } _ { \mathrm { S a m p } 0 } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } \left[ \log \sigma \left( \beta \sum _ { t } ^ { T } \frac { 1 } { \pi } \log \frac { \pi _ { \theta } ( \gamma _ { t } ^ { t } | x ) } { \pi _ { \mathrm { r e f } } ( \gamma _ { t } ^ { t } | x ) } - \beta \sum _ { t = 1 } ^ { T } \log \frac { \pi _ { \theta } ( \gamma _ { t } ^ { t } | x ) } { \pi _ { \mathrm { r e f } } ( \gamma _ { t } ^ { t } | x ) } \right) \right] , T _ { m } = \operatorname* { m i n } ( T _ { w } , T _ { l } ) , y ^ { t } \sim \operatorname { U n i f o r m } ( T _ { m } , \{ y \} ^ { T } ) } \end{array}$

DPO vs SamPO, across 5 conditional and 3 open-ended benchmarks  
![](images/6214737afd743f3637c881c57c6eeb6337eef30c4b4bca4e3718052119b6ffbd.jpg)  
Figure 1: Down-Sampling strategy helps mitigate the potential length reliance, and thus improves DPO.

such preference data inherently exhibit a length bias (Shen et al., 2023). Therefore, subsequent finetuned policy model exploit this bias as a shortcut to achieve higher reward scores by generating longer responses (Gao et al., 2023a), without necessarily improving quality (Kabir et al., 2023; Dubois et al., 2024b). Various regularization approaches have been proposed to mitigate this inherent bias within reward models (Ramamurthy et al., 2022; Coste et al., 2023; Moskovitz et al., 2023; Chen et al., 2024b). On the other hand, although DPO does not explicitly use a reward model, the length distribution bias inherent in the offline preference labels still contributes to the verbosity issue (Hou et al., 2024; Rafailov et al., 2024). Analysis suggests that policy models trained using DPO tend to generate responses that are almost twice the length of the labeled data (Park et al., 2024).

In this paper, we propose that, in addition to the length bias in the data, DPO exhibits a hidden algorithmic dependence on response length. As illustrated in the upper portion of Figure 1, the loss function in DPO is based on the discrepancy between sequence-level KL divergence, which can also be computed and aggregated at the token-level. It is evident that discrepancies between chosen label ${ \bf \nabla } \mathbf { \pmb { y } } _ { w }$ and rejected label $\mathbf {  { y } } _ { \mathbf { \downarrow } }$ lead to an inadvertent reliance on auxiliary length features: training samples with longer chosen labels than rejected ones lead to overestimated rewards during training, while those with shorter chosen labels result in underestimated rewards. Therefore, overestimated rewards contribute more significantly to gradient optimization, ultimately exacerbating verbosity. We believe this algorithmic dependence on response length is a unique drawback of DPO, since the explicit rewards in RLHF typically manifest as scalar values (Ouyang et al., 2022).

We propose that addressing this reliance on response length can be effectively achieved through a straightforward down-sampling method. Illustrated in the middle of Figure 1, this approach involves down-sampling equal token-level probability features for computing regularized KL divergences. Our contributions in this paper are threefold:

• We analyze the algorithmic dependence on response length in DPO, revaling how it results in overestimated or underestimated rewards. Through decomposition experiments using datasets with varying label length, we empirically demonstrate the biased rewards.

• We propose a lightweight approach, called SamPO, to mitigate the biased length reliance in DPO. By simply down-sampling equal probability features at the token-level, we can apply DPO with regularized KL divergences.

• We validate our method using three different LLMs of varying scales. Compared to DPO, SamPO significantly reduces verbosity. Leveraging debaised rewards, we achieve significant improvements across five conditioned and three open-ended benchmarks, as depicted in the lower section of Figure 1.

## 2 Related Work

Optimization from Human Preference aims to align neural models with human minds. As a seminal work, (Stiennon et al., 2020) collected human preferences on 123k pairs of summary outputs, then trained a reward model that guides the GPT-3 model (Brown et al., 2020) to produce more coherent and human-preferred summaries. (Ouyang et al., 2022) then further scaled similar pipeline with 1M diverse text instructions, and reported that outputs from the 1.3B parameter InstructGPT model were preferred to outputs from the 175B GPT-3 model, according to downstream human evaluation. RLHF has become an essential part of aligning LLMs (Touvron et al., 2023; Bi et al., 2024; Bai et al., 2023; Young et al., 2024). However, as it follows a multi-stage training strategy, and heavily relays on the quality of reward model, RLHF’s training cost and stability are widely criticized (Zheng et al., 2023; McKinney et al., 2023). Therefore, DPO came into being, providing a stable alternative that does not rely on an explicit reward model (Rafailov et al., 2023). It has been proved that DPO can achieve the same alignment effect as RLHF (Ivison et al., 2023; Hou et al., 2024).

Over-optimization in RL is a well-known obstacle (Skalse et al., 2022; Pan et al., 2023a; Casper et al., 2023; Zheng et al., 2023), which refers to the phenomenon that feedback scores from the reward model are getting higher, but the updated policy model produces lower quality responses. And one particularly noticeable low-quality feature is verbosity. It is general to blame for exploitation of reward model (Casper et al., 2023; Gao et al., 2023a), and thus various regularization approaches have been proposed, including uncertainty-based regularization (Coste et al., 2023; Zhai et al., 2023), composite reward models (Moskovitz et al., 2023), and length decorrelation (Chen et al., 2024b). However, since the reward model is eliminated in DPO, none of the above approaches can be directly applied. Herein, specific methods are introduced, (Park et al., 2024) introduced a pairwise length regularization term to dampen the verbosity trends, and SimPO (Meng et al., 2024) used average probability to eliminate length reliance.

In this paper, we present that the verbosity issue in DPO is further related to algorithmic biased length reliance, which is never analyzed in previous literature. And this drawback can be effectively handled via down-sampling over KL divergence.

## 3 SamPO: Down-Sampled DPO

In this section, we first give a brief introduction of DPO’s optimization target (§3.1), then dive into further analysis of its potential length reliance (§3.2). Subsequently, we present SamPO, which intuitively regularizes the biased length-specific reward (§3.3).

## 3.1 Preliminary Background of DPO

DPO implements direct RLHF based on offline preference data and an offloaded reward model. Specifically, DPO first re-parameterizes the reward model in multi-stage RLHF as follows:

$$
r _ { \phi } ( x , y ) = \beta \log \frac { \pi _ { \theta } ( y | x ) } { \pi _ { r e f } ( y | x ) } + \beta \log Z ( x )\tag{1}
$$

where $\mathbf { \Delta } _ { r _ { \phi } , \mathbf { \lambda } }$ π<sub>θ</sub> and $\pi _ { r e f }$ denote the reward model, the policy model, and the reference model, respectively. Both $\pi _ { \theta }$ and $\pi _ { r e f }$ are usually initialized from the same SFT model. While $\pi \theta$ is subject to further optimization during DPO, $\pi _ { r e f }$ is usually frozen. $\boldsymbol { Z } ( \boldsymbol { x } )$ is the partition function, and $\beta$ is a hyperparameter that adjusts the intensity of rewards. DPO incorporates the Bradley-Terry model to predict preferences:

$$
P _ { \theta } ( y _ { w } \succ y _ { l } | x ) = \frac { \exp ( r _ { \phi } ( x , y _ { w } ) ) } { \exp ( r _ { \phi } ( x , y _ { w } ) ) + \exp ( r _ { \phi } ( x , y _ { l } ) ) }\tag{2}
$$

where a preference triplet $( \pmb { x } , \pmb { y } _ { w } , \pmb { y } _ { l } )$ consists of a prompt instruction x, a chosen response ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ , and a less preferred response $\mathbf {  { y } } _ { \mathbf { \downarrow } }$ . According to the Bradley-Terry model, the preference probability $P _ { \theta }$ can be estimated via pairwise comparison. The loss function of DPO is defined as:

$$
\mathcal { L } _ { d p o } ( \pi _ { \theta } ; \pi _ { r e f } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim D } [ \log \sigma ( \Delta ) ]\tag{3}
$$

where:

$$
\Delta = \beta \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { r e f } ( y _ { w } | x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { r e f } ( y _ { l } | x ) }\tag{4}
$$

In this context, $\sigma$ stands for sigmoid function, and D denotes the entire pairwise preference dataset. The implicit reward $\pmb { \Delta }$ in Eq. 4 is formulated as the discrepancy between the chosen KL divergence log $\frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { r e f } ( y _ { w } | x ) }$ and the rejected KL divergence log $\frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { r e f } ( y _ { l } | x ) }$ . Each KL divergence is calculated based on the tokens in the response $\mathbf { \mathscr { y } } .$ Considering Eq. 3, DPO’s gradients can be written as:

$$
\nabla _ { \theta } \mathcal { L } _ { d p o } ( \pi _ { \theta } ; \pi _ { r e f } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim D } [ \beta \sigma ( - \Delta ) \mathcal { M } ]\tag{5}
$$

$$
\mathcal { M } = \nabla _ { \boldsymbol { \theta } } \log \pi ( \boldsymbol { y } _ { w } | \boldsymbol { x } ) - \nabla _ { \boldsymbol { \theta } } \log \pi ( \boldsymbol { y } _ { l } | \boldsymbol { x } )\tag{6}
$$

![](images/5b05a074257614cff9be15be7d5d090e650252ecf5f90ad0c48e3540b32b6d3f.jpg)  
(a)

![](images/7a4d77a3157eab23c4fc2977c534b2bf4aa93be32485960870177c50ec032e2d.jpg)  
(b)

![](images/8f068b8ce753fc0e816de79db9da2a1502ad74b608142416ef8519720064230b.jpg)  
(c)  
Figure 2: The disparity in pairwise responses, illustrated by typical examples, forces DPO to overestimate or underestimate the actual rewards. In the upper sub-figure (a), we present DPO’s chosen reward $\sum \log { \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { r e f } ( y _ { w } | x ) } }$ and rejected reward $\sum \log { \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { r e f } ( y _ { l } | x ) } }$ with red and purple curves, respectively. The reward for each response is calculated as the sequence-level KL divergence, which is derived from the token-level log probability ratios (illustrated by green and blue bars). Therefore, the difference between these two curves illustrates the implicit reward target in DPO, as shown in Eq. 7. Averaged and normalized DPO results are displayed in the lower-left sub-figure (b), while our SamPO is illustrated in lower-right sub-figure (c).

where  is a discrepancy term that leads the policy model $\pi _ { \theta }$ to increase the likelihood of the chosen response ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ and decrease the likelihood of the rejected response ${ \mathbf { } } ^ { \pmb { y } _ { l } }$ . The term $\pmb { \Delta }$ acts as a scaling factor for the intensity of .

## 3.2 Biased Length Reliance in DPO

DPO’s loss and gradient are computed at the sequence-level. When calculating the KL term log $\frac { \pi _ { \theta } ( y | x ) } { \pi _ { r e f } ( y | x ) }$ , DPO treats the probabilities of individual tokens as discrete samples. We can express Eq. 4 at the token-level (Proof is in Appendix A):

$$
\Delta = \beta \sum _ { t = 1 } ^ { T _ { w } } \log \frac { \pi _ { \theta } ( y _ { w } ^ { t } | x ) } { \pi _ { r e f } ( y _ { w } ^ { t } | x ) } - \beta \sum _ { t = 1 } ^ { T _ { l } } \log \frac { \pi _ { \theta } ( y _ { l } ^ { t } | x ) } { \pi _ { r e f } ( y _ { l } ^ { t } | x ) }\tag{7}
$$

where $\mathbf { \Delta } \mathbf { T } _ { w }$ and $\pmb { T } _ { l }$ denote the number of tokens from the first to the t-th positions in the chosen response ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ and the rejected response $\mathbf {  { y } } _ { \mathbf { \downarrow } }$ , respectively. Similarly, we rewrite Eq. 6 as:

$$
\mathcal { M } = \nabla _ { \boldsymbol { \theta } } \sum _ { t = 1 } ^ { T _ { w } } \log \pi ( \boldsymbol { y } _ { w } ^ { t } | \boldsymbol { x } ) - \nabla _ { \boldsymbol { \theta } } \sum _ { t = 1 } ^ { T _ { l } } \log \pi ( \boldsymbol { y } _ { l } ^ { t } | \boldsymbol { x } )\tag{8}
$$

From this, we can intuitively understand how the difference in length between the chosen response ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ and the rejected response $\mathbf {  { y } } _ { \mathbf { \downarrow } }$ affects the loss and the gradient. As illustrated in sub-Figure 2(a), a “comparable reward” is achieved if ${ \bf \nabla } _ { \bf y _ { \mathrm { ~ } \mathrm { ~ } w } }$ and ${ \mathbf { } } ^ { \mathbf { \eta } } \mathbf { \mathbf { 3 } } \mathbf { \eta } ^ { \mathrm { ~ t ~ } }$ have the same length, allowing DPO to effectively learns the quality difference. However, if ${ \bf \nabla } \mathbf { \pmb { y } } _ { w }$ is much longer than ${ \mathbf { } } ^ { \mathbf { \alpha } } \mathbf { \mathbf { 3 } } \mathbf { } \mathbf { } l \mathbf { \alpha } \mathrm { ~ \mathrm { ~ \bf ~ 3 ~ } ~ }$ , the larger number of tokens in ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ may result in an “overestimated reward” in Eq. 7, contributing disproportionately to the gradient updates described in Eq. 5 and 8. Conversely, if ${ \bf \nabla } _ { \bf y _ { \mathrm { ~ } w } }$ is shorter than $\mathbf {  { y } } _ { \mathbf { \downarrow } }$ , DPO could “underestimate reward” and incorporate fewer gradients, even if ${ \bf \nabla } _ { \bf y _ { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } } }$ is of better quality. This bias towards length means that DPO tends to favor longer, seemingly acceptable responses over shorter, well-formed ones during training, potentially leading to verbose outputs.

## 3.3 Debiased KL Divergence

In the following content, we explore two common strategies to mitigate the dependence on sequence length: averaging and sampling.

Averaging modifies the sequence-level KL divergence to use a marginally averaged reward, which serves as a basic form of length regularization. This adjustment modifies Eq. 7 as follows:

$$
\Delta = \beta \frac { \underset { t = 1 } { \overset { T _ { w } } { \sum } } \log \frac { \pi _ { \theta } ( y _ { w } ^ { t } | x ) } { \pi _ { r e f } ( y _ { w } ^ { t } | x ) } } { \left| T _ { w } \right| } - \beta \frac { \underset { t = 1 } { \overset { | T _ { l } | } { \sum } } \log \frac { \pi _ { \theta } ( y _ { l } ^ { t } | x ) } { \pi _ { r e f } ( y _ { l } ^ { t } | x ) } } { \left| T _ { l } \right| }\tag{9}
$$

The averaging process can help remove the influence of length. However, as shown in the left corner of Figure 2(b), there lies a scale difference between the marginally averaged reward and the original sequence-level reward. To address this, we scale the marginal reward with a dynamic scaling factor $\frac { \left( T _ { w } + T _ { l } \right) } { 2 }$ , which is the average length of the chosen response ${ \bf \nabla } _ { \bf y } \mathrm { \textbf { } } _ { w }$ and the rejected response $\mathbf {  { y } } _ { \mathbf { \downarrow } }$

Sampling involves selecting the same amount of tokens from both the chosen and the rejected responses, and then calculating the down-sampled sequence-level KL divergence for the implicit reward. This modifies Eq. 7 to:

$$
\begin{array} { r l r } {  { \Delta = \beta \sum _ { t = 1 } ^ { T _ { m } } \log \frac { \pi _ { \theta } ( y _ { w } ^ { t } | x ) } { \pi _ { r e f } ( y _ { w } ^ { t } | x ) } - \beta \sum _ { t = 1 } ^ { T _ { m } } \log \frac { \pi _ { \theta } ( y _ { l } ^ { t } | x ) } { \pi _ { r e f } ( y _ { l } ^ { t } | x ) } } } \\ & { } & { T _ { m } = \operatorname* { m i n } ( T _ { w } , T _ { l } ) , y ^ { t } \sim \mathrm { U n i f o r m } ( T _ { m } , \{ y \} ^ { T } ) } \end{array}\tag{10}
$$

where ${ \pmb T } _ { m }$ is equal to the minimum token length of $( T _ { w } , T _ { l } )$ , and $y ^ { t }$ is down-sampled from all tokens $\{ y ^ { T } \}$ uniformly. Eq. 10 is consistent with the corresponding reward term shown in the middle of Figure 1. In addition, we discuss the impact of sampling randomness in Appendix E.

Figure 2(b) and (c) demonstrate that both averaging and sampling can produce length-debiased rewards that are comparably effective. However, simple averaging diminishes the variance feature among tokens. Consequently, we opt for the downsampling strategy in our proposed SamPO method. This decision is validated in Section 5.

## 4 Experimental Setup

In this section, we start by introducing our datasets (§ 4.1, § 4.2), followed by the baselines (§ 4.3, § 4.4), and then provide an overview of our experimental design (§ 4.5).

## 4.1 Training Datasets

We leverage three independent preference datasets for training. Two of these are consistent with the original DPO (Rafailov et al., 2023): the 161k HH-RLHF data (Ganguli et al., 2022), and the 92.8k TL;DR data (Völske et al., 2017). Additionally, we include the 61k binarized UltraFeedback data (Cui et al., 2023) that has been utilized in subsequent works (Ivison et al., 2023; Meng et al., 2024) following DPO. Each of these datasets comes with an evaluation set for cross-validation during training.

## 4.2 Evaluation Benchmarks

Following DPO, for models trained on HH-RLHF or TL;DR, we randomly select 256 samples from their respective evaluation sets for final testing. We report the win rate between the response generated by the fine-tuned policy model $\hat { y _ { \boldsymbol \theta } } = \pi _ { \boldsymbol \theta } ( \mathbf { x } _ { t e s t } )$ and the response from the baseline SFT model $\hat { \mathbf { y } _ { r e f } } = \boldsymbol { \pi } _ { r e f } ( \mathbf { x } _ { t e s t } )$ , judged by GPT-4 (Achiam et al., 2023). For models trained with UltraFeedback, we use five conditional and one open-ended generation benchmarks. The conditional benchmarks, along with their in-context examples, are: GSM8K in 8-shot (Cobbe et al., 2021), IFEval in 3- shot (Zhou et al., 2023), PiQA in 3-shot (Bisk et al., 2020), MMLU in 0-shot (Hendrycks et al., 2021), and TruthfulQA in 3-shot (Lin et al., 2022). The open-ended benchmark is AlpacaEval2 (Li et al., 2023). We report match accuracy for the conditional benchmarks, and the length-debiased GPT-4 win rate for AlpacaEval2 (Dubois et al., 2024a). For additional details, refer to Appendix B.

## 4.3 Foundation Models

In our experiments, we include LLMs of three different sizes: Pythia-2.8B (Biderman et al., 2023), Llama3-8B-Instruct (AI@Meta, 2024), and Tulu2- 13B-SFT (Ivison et al., 2023). Details of these LLMs, including their hyperparameters and associated costs, are provided in Appendix C.

## 4.4 Baselines

Several variants of DPO have been proposed, which can be categorized into three main types: (1) Reduce cost. Although DPO is robust, the preparation of high-quality pair-wise preference labels and the requirement to run with two large models make DPO costly. To address this, KTO (Ethayarajh et al., 2024) proposed to use non-pairwise preference data. ORPO (Hong et al., 2024), CPO (Xu et al., 2024), and SimPO (Meng et al., 2024) introduced reference-free losses that allow optimization with a single policy model; (2) Alleviate overfitting. IPO (Azar et al., 2023) analyzed the risk of overfitting, and introduced a square loss to reshape the monotonic DPO loss. TDPO (Zeng et al., 2024) incorporated forward KL divergence constraints for each token, improving alignment and diversity. BCO (Jung et al., 2024) and NCA (Chen et al., 2024a) offered strategies to reduce noise from pairwise preference responses; (3) Overcome verbosity. Park et al. (2024) introduced a pairwise length regularization term to counter verbosity. SimPO (Meng et al., 2024) used average probability to eliminate dependency on sequence length.

We select methods that focus on noise removal or length normalization, and have shown relatively positive testing results as our final baselines: Hybrid DPO+SFT, TDPO (Zeng et al., 2024), Lengthnormed DPO (Park et al., 2024), BCO (Jung et al., 2024), SimPO (Meng et al., 2024). Particularly, Hybrid DPO+SFT refers to the multi-task learning pipeline where DPO is applied to pairwise responses and SFT is applied to the chosen response at the same time, which is a common practice (Hua et al., 2024; Lu et al., 2024).

## 4.5 Experimental Designs

In general, we design three groups of experiments:

(1) Presence of biased length reliance. We extract two 27k subsets from the UltraFeedback only by response length. One is named UltraFeedback-long, in which the chosen response of each data must be longer than the rejected response. The other one is named UltraFeedback-short, and as the name suggests, it contains a shorter chosen response. We use these subsets for biased reward exhibitions.

(2) Preliminary Study of DPO and variants. Given that there are many variants of DPO, and they often use their own hyperparameters, we first conduct a preliminary study to align their performance under the same conditions. This study helps us select several robust baselines. The results are reported in Appendix D.

![](images/5ad81bb70934eed2d1dc048e1280a2a3a40659c2f5a5110f25e0cda319efdf6d.jpg)  
Figure 3: Trends of DPO’s implicit reward (Eq. 7), when fine-tuned with UltraFeedback-long, -short and -all sets. Three debiased rewards are produced by our SamPO.

<table><tr><td></td><td>GSM8K</td><td>IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg.</td></tr><tr><td>long</td><td>41.24</td><td>37.89</td><td>81.28</td><td>55.86</td><td>38.68</td><td>50.99</td></tr><tr><td>short</td><td>34.50</td><td>6.00</td><td>77.09</td><td>54.87</td><td>30.48</td><td>40.59</td></tr><tr><td>all</td><td>42.61</td><td>43.76</td><td>81.77</td><td>55.85</td><td>35.86</td><td>51.97</td></tr><tr><td>long*</td><td>42.61</td><td>38.01</td><td>81.18</td><td>55.86</td><td>36.11</td><td>50.75</td></tr><tr><td>short*</td><td>41.70</td><td>33.93</td><td>81.18</td><td>55.5</td><td>36.35</td><td>49.73</td></tr><tr><td>all*</td><td>42.68</td><td>44.12</td><td>81.28</td><td>55.8</td><td>40.15</td><td>52.81</td></tr></table>

Table 1: Performance of models in Figure 3. The \* mark stands for the SamPO’s debiased rewards.

(3) Experiments with various LLMs. Similar to DPO, we use Pythia-2.8B to train and test SamPO on HH-RLHF or TL;DR; on the other hand, following relevant studies (Ivison et al., 2023; Hong et al., 2024), we use Tulu2-13B-SFT and Llama3-8B-Instruct to train on Ultrafeedback and verify SamPO on public benchmarks. Also, literature reports that iteratively updates the frozen reference model $\pi _ { r e f }$ can obtain further gains (Gorbatovski et al., 2024; Zhang et al., 2024). Thus, we combine it with SamPO to present Iterative SamPO.

## 5 Experimental Results

In this section, following the above designs, we first report the group experiments of length reliance (§ 5.1), then present comparison studies against strong baselines (§ 5.2). We discuss quantitative results in the main body. We leave more ablation studies and case analysis in Appendix E, F, and H.

## 5.1 Group study of length reliance

Figure 3 illustrates the trends of DPO’s implicit reward on the same test set when we fine-tune the same Tulu2-13B-SFT model with different subsets of UltraFeedback. We report testing performance in Table 1. It is clear that data from the same distribution leads to different training and testing performances due to the difference in response length.

<table><tr><td></td><td colspan="8">Tulu2-13B-SFT</td></tr><tr><td>Methods</td><td>GSM8K</td><td>IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg. Alpaca2</td><td>LC Alpaca2</td><td>Len./Token</td></tr><tr><td>Tulu2-13B-SFT (Ivison et al., 2023)</td><td>40.56</td><td>37.17</td><td>81.39</td><td>55.53</td><td>33.78</td><td>49.69 5.09</td><td>9.99</td><td>262</td></tr><tr><td>Tulu2-13B-DPO (Ivison et al., 2023)</td><td>42.99</td><td>42.45</td><td>81.28</td><td>56.07</td><td>41.86 52.93</td><td>11.45</td><td>13.7</td><td>382</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>43.44</td><td>43.17</td><td>81.66</td><td>56.08</td><td>39.66</td><td>52.80 10.66</td><td>15.02</td><td>372</td></tr><tr><td>Iterative DPO</td><td>42.08</td><td>44.96</td><td>81.39</td><td>56.02</td><td>40.15 52.92</td><td>12.17</td><td>14.24</td><td>400</td></tr><tr><td>Hybrid DPO+SFT</td><td>41.85</td><td>44.36</td><td>81.28</td><td>56.15</td><td>40.02</td><td>52.73 7.66</td><td>13.45</td><td>308</td></tr><tr><td>TDPO (Zeng et al., 2024)</td><td>41.39</td><td>41.25</td><td>81.34</td><td>55.78</td><td>36.11</td><td>51.17 6.86</td><td>11.45</td><td>290</td></tr><tr><td>Length-normed DPO (Park et al., 2024)</td><td>40.71</td><td>45.8</td><td>80.85</td><td>55.85</td><td>39.66 52.57</td><td>7.47</td><td>13.40</td><td>250</td></tr><tr><td>BCO (Jung et al., 2024)</td><td>42.68</td><td>43.73</td><td>81.45</td><td>56.41</td><td>39.66</td><td>52.79 9.07</td><td>13.29</td><td>316</td></tr><tr><td>SimPO (Meng et al., 2024)</td><td>29.57</td><td>47.24</td><td>81.39</td><td>56.10</td><td>38.31</td><td>50.52 5.21</td><td>7.84</td><td>336</td></tr><tr><td>SamPO (ours)</td><td>41.55</td><td>45.32</td><td>80.85</td><td>55.88</td><td>41.37</td><td>52.99 11.77</td><td>17.6</td><td>339</td></tr><tr><td>Iterative SamPO (ours)</td><td>42.08</td><td>46.28</td><td>81.07</td><td>56.12</td><td>41.25</td><td>53.36 14.58</td><td>17.52</td><td>347</td></tr><tr><td>DPO-SANorm (ours)</td><td>42.15 44.36</td><td>81.07</td><td>56.00</td><td></td><td>38.43</td><td>52.40 9.21</td><td>14.53</td><td>283</td></tr><tr><td></td><td colspan="8">Llama3-8B-Instruct</td></tr><tr><td>Methods</td><td>GSM8K IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg.</td><td>Alpaca2</td><td>LC Alpaca2</td><td>Len./Token</td></tr><tr><td>Llama3-8B-Instruct (AI@Meta, 2024)</td><td>75.06</td><td>49.40</td><td>80.69</td><td>63.85</td><td>36.47</td><td>61.09 22.57</td><td>22.92</td><td>421</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>75.59</td><td>51.80</td><td>81.94</td><td>64.06</td><td>40.39</td><td>62.76 23.34</td><td>23.20</td><td>422</td></tr><tr><td>Iterative DPO</td><td>74.91</td><td>52.52</td><td>81.66</td><td>64.02</td><td>39.90</td><td>62.60 23.92</td><td>25.50</td><td>403</td></tr><tr><td>Hybrid DPO+SFT</td><td>75.59</td><td>65.83</td><td>81.34</td><td>63.54</td><td>39.78</td><td>65.22 20.17</td><td>20.62</td><td>380</td></tr><tr><td>TDPO (Zeng et al., 2024)</td><td>75.36</td><td>51.32</td><td>81.23</td><td>63.54</td><td>38.07</td><td>61.90 23.66</td><td>24.57</td><td>408</td></tr><tr><td>Length-normed DPO (Park et al., 2024)</td><td>76.12</td><td>46.76</td><td>81.39</td><td>64.09</td><td>40.76</td><td>61.82 24.04</td><td>27.44</td><td>377</td></tr><tr><td>BCO (Jung et al., 2024)</td><td>76.19</td><td>50.60</td><td>81.66</td><td>63.99</td><td>39.90</td><td>62.47 24.72</td><td>24.81</td><td>421</td></tr><tr><td>SimPO (Meng et al., 2024)</td><td>75.06</td><td>60.43</td><td>81.83</td><td>63.43</td><td>39.53</td><td>64.06</td><td>26.82 31.29</td><td>375</td></tr><tr><td>Llama3-8B-Ins.-SimPO (Meng et al., 2024)</td><td>72.93</td><td>46.28</td><td>78.51</td><td>61.99</td><td>42.96</td><td>60.53</td><td>39.72 43.42</td><td>387</td></tr><tr><td>SamPO (ours)</td><td>76.56</td><td>57.03</td><td>81.72</td><td>64.00</td><td>41.06</td><td>64.18</td><td>28.97</td><td>375</td></tr><tr><td>Iterative SamPO (ours)</td><td>77.81</td><td>60.55</td><td>81.18</td><td>64.12</td><td>44.07</td><td>65.55 30.68</td><td>32.01 35.14</td><td>377</td></tr></table>

Table 2: Qualitative results of fine-tuning two LLMs with DPO, several variants and our SamPO. We use the same UltraFeedback dataset and keep almost all hyperparameters the same for each LLM group. Specifically, Tulu2-13B-SFT and -DPO, Llama3-8B-Insturct and -Ins.-SimPO are open-source checkpoints. We evaluate all models, including those public models, under the same framework. We bold the best results and underline the unusually poor results.

The “-all” set refers to training with original UltraFeedback, which mix “-long” and “-short” data. The “-long” subset provides overestimated rewards and therefore causes performance degradation. However, since statistically, the chosen response is longer than the rejected response (Park et al., 2024), the training trend of the “-long” subset is similar to the “-all” full set. On the contrary, the “-short” subset completely erases the distinctive feature of length, hoping that the model will perform comparative learning based on content quality. However, the biased DPO completely underestimate the reward, thus causing collapses.

Yet, our SamPO presents debaised rewards. We can observe debiased positive rewards on the “- short” set. And the debaised rewards of “-all” set grow to a high peak at 300 steps. Such debiased rewards result in significant U-turn reversal and further improvements. As shown in Table 1, SamPO manages to eliminate collapse on the “-short” set, where we record a normal average benchmark score similar to the “-long” set, improving the score by 9.2%. Thanks to the regularization of those “short” data, the “-all” set that mixes both “long” and “short” data achieves the best score up to 52.81 on average.

## 5.2 Comparison study against other methods

## 5.2.1 Study on UltraFeedback

For LLMs that fine-tuned with UltraFeedback, we evaluate their downstream performance in Table 2.

Overall enhancement by SamPO. For Tulu2- 13B-SFT, our replicated DPO shows benchmark accuracy and response length on AlpacaEval2 data comparable to the open-source version. Compared to the SFT baseline, DPO improves performance across all test data but increases response length by 40-45%. Iterative DPO exacerbates this verbosity issue. However, all chosen baselines and our SamPOs produce shorter responses, mitigating verbosity. However, TDPO and SimPO show significant drops in conditional benchmarks, such as over 10% on GSM8K and over 3% on TruthfulQA, compared to DPO. Notably, our SamPOs achieve overall improvements on both conditional benchmarks (+0.5%) and open-ended generation for AlpacaEval2 prompts (+4%). Also, the averaging version DPO-SANorm, mentioned in section 3.3, confirms that the sampling strategy is more valid.

![](images/0e22c05bebab3689fd82665e3161c4f3cdae37aacea7827284120fc70c217d11.jpg)  
Figure 4: We show how the policy model’s response length changes on AlpacEval2 as the test performance improves over 3 epochs of training. The epoch number increases from left to right along the curve.

For Llama3-8B-Instruct, we observe superior length stability. Even when fine-tuned with the original DPO, the model maintains its initial response length, likely due to its comprehensive training process involving SFT, RLHF, and DPO (AI@Meta, 2024). Marginal improvements are observed over its DPO version, with average gains of 1.7% on five conditional benchmarks and <1% on AlpacaEval2. Among all methods, only hybrid DPO+SFT, SimPO, and our SamPOs show significant improvements over DPO, with average gains of 1.3% to 3% on five accuracy benchmarks. Specifically, hybrid DPO+SFT excels in IFEval (65.83), and our Sam-POs notably improve GSM8K (+2.3%) and TruthfulQA (+3.7%). As for GPT-4 judged AlpacaEval2, hybrid training loses about 3% performance, while our SamPO achieves the best performance in both raw and length-debiased scores among all locally fine-tuned LLMs, outperforming DPO up to 12%.

Discussions of SimPO. The SimPO method has an obvious “seesaw” dilemma. The open-source SimPO checkpoint achieves the best performance of AlpacaEval2 at the expense of a significant sacrifice on other benchmarks. We avoid this in the reproduction and obtain a more balanced version. Also, the public release was trained with boosted data<sup>2</sup> instead of the naive UltraFeedback.

<table><tr><td></td><td colspan="2">HH-RLHF</td><td colspan="2">TL;DR</td></tr><tr><td></td><td>Wins</td><td>Len.</td><td>Wins</td><td>Len.</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>74.49</td><td>250.07</td><td>60.98</td><td>53.80</td></tr><tr><td>Iterative DPO</td><td>53.46</td><td>253.99</td><td>73.58</td><td>66.65</td></tr><tr><td>Hybrid DPO+SFT</td><td>86.12</td><td>41.29</td><td>45.68</td><td>41.43</td></tr><tr><td>TDPO (Zeng et al., 2024)</td><td>52.53</td><td>246.28</td><td>47.76</td><td>45.60</td></tr><tr><td>Len.-Norm (Park et al., 2024)</td><td>68.95</td><td>246.28</td><td>58.13</td><td>47.34</td></tr><tr><td>BCO (Jung et al., 2024)</td><td>65.85</td><td>218.05</td><td>50.62</td><td>42.93</td></tr><tr><td>SimPO (Meng et al., 2024)</td><td>78.91</td><td>14.77</td><td>33.33</td><td>31.90</td></tr><tr><td>SamPO (ours)</td><td>82.8</td><td>112.95</td><td>65.71</td><td>69.52</td></tr><tr><td>Iterative SamPO (ours)</td><td>79.05</td><td>137.55</td><td>73.58</td><td>49.54</td></tr></table>

Table 3: Win Rate (%) and Avg. Output Length across methods. We bold the best and underline the outliers.

Length stability of SamPO. Based on Figure 4, we find that DPO makes the model increasingly prefer to generate longer responses in 3-epoch training, and Iterative DPO further strengthens this trend. In contrast, SamPO and Iterative SamPO achieve higher testing scores and stabilise the length.

## 5.2.2 Study on HH-RLHF & TL;DR

As for HH-RLHF and TL;DR, we utilize Pythia-2.8B for all experiments. Since Pythia has not been specifically trained for instructional tasks, we initiate our process with one epoch of SFT on the chosen response, following DPO’s setup. Subsequently, we conduct preference optimization using SamPO alongside various baseline methods. Following previous literature (Rafailov et al., 2023; Park et al., 2024), GPT-4 served as the proxy for human preference. We report the win rate against the SFT basis and the average generated token length of all methods in Table 3.

SamPO has a good effect on HH-RLHF. SamPO improves performance across all HH-RLHF test data, achieving the second-best win rate while maintaining a lower yet reasonable response length. Iterative SamPO shows slightly lower win rates due to less control over response length. Baselines such as Iterative DPO and TDPO achieve win rates close to 50%, indicating minimal improvement over the SFT model. Hybrid DPO+SFT stands out as a strong baseline, addressing the under-generalization issue and attaining an 86.12% win rate with the shortest average response lengths among all experiments. SimPO, while achieving a similar win rate of 78.91% as Iterative SamPO, but produces incredibly low response length.

SamPO achieves the best performance on TL;DR. In terms of TL;DR, SamPO and Iterative SamPO show the highest win rates, with 65.71% and 73.58%, respectively, significantly outperforming all other methods. DPO and Length-normed DPO also perform well, achieving win rates of 60.98% and 58.13%, respectively. Iterative DPO reaches the best while using longer answers than Iterative SamPO. In contrast, SimPO has the lowest win rate at 33.33%, indicating that it is less effective on the TL;DR dataset.

Over-simplification by SimPO. In fact, on HH-RLHF, we notice many of the outputs from SimPO are overly simplified, often omitting necessary content and resulting in only 14.77 lengths of tokens on average. For example, a preferred response from HH-RLHF is “I’ll give you the links.”, whereas the SimPO response is simply “Sure!”. This suggests that while concise, the responses lack the necessary informativeness. In this scenario, we can see GPT-4 prefers over-simplified responses, which is probably due to the binary setup of preference choice. Similarly, on TL;DR, SimPO produces the shortest responses (average 31.90 tokens). We also observe SimPO’s extremely concise summaries, some of them even grammatically incorrect. For example, a preferred summary from the TL;DR is “I [20M] met a great girl [16F] online who lives in the same city. Problems are: she’s moving away, I want to meet her, and the obvious age gap.”, while SimPO outputs a shorter summary without a subject and capitalizes the first letter: “online flirt turns into legit relationship. Great chemistry. Age gap and distance issues. Need advice before final meetup before long trip abroad.”.

## 5.2.3 Human Evaluation of SamPO

In addition to the aforementioned automated evaluation, we further conduct a large-scale human evaluation to study the effectiveness of the SamPO algorithm when applied to super large LLM (e.g., over 50B). We use an LLM fine-tuned based on Qwen1.5-72B (Bai et al., 2023) as a starting point and fine-tune it for one epoch using the proposed SamPO method. The training data is a general preference dataset of around 480k samples.

We report the results of the human evaluation in Table 4, covering the three most popular scenarios: general Machine Reading Comprehension (MRC), logical reasoning (e.g., math or logic questions), and open domain dialogues in role-play settings. We have hired a 30-person annotation team, each of whom has at least a bachelor’s degree or above. Each test scenario contains 500 to 1k carefully crafted challenging instances, which are then cross-labeled by multiple professional annotators. Our scoring criteria are relatively simple, distinguishing only between incorrect and acceptable responses. We observe that SamPO significantly outperforms both the SFT Base and DPO method on all tasks.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MRC</td><td rowspan=1 colspan=1>Logical Reasoning</td><td rowspan=1 colspan=1>RolePlay</td><td rowspan=1 colspan=1>Avg.</td></tr><tr><td rowspan=1 colspan=1>SFT Base</td><td rowspan=1 colspan=1>81.25</td><td rowspan=1 colspan=1>69.52</td><td rowspan=1 colspan=1>59.12</td><td rowspan=1 colspan=1>69.96</td></tr><tr><td rowspan=1 colspan=1>w/DPO</td><td rowspan=1 colspan=1>85.33</td><td rowspan=1 colspan=1>73.25</td><td rowspan=1 colspan=1>57.41</td><td rowspan=1 colspan=1>72.00</td></tr><tr><td rowspan=1 colspan=1>w/ SamPO</td><td rowspan=1 colspan=1>87.50</td><td rowspan=1 colspan=1>83.57</td><td rowspan=1 colspan=1>63.61</td><td rowspan=1 colspan=1>78.23</td></tr></table>

Table 4: Human Evaluation results of a Qwen1.5-72Bbased SFT model and its two further fine-tuned versions, applying with DPO and SamPO respectively.

## 6 Conclusion

In this paper, we identify and address the verbosity issue in DPO related to biased length reliance. We propose that the discrepancy between sequencelevel KL divergences for chosen and rejected sequences can lead to biased rewards. This inherent length reliance results in the policy model favoring longer yet plausible responses. Thus, we propose SamPO, an approach that regularizes the KL divergence by down-sampling equal token-level features. Our empirical evaluations across three different LLMs and diverse datasets show that SamPO effectively reduces verbosity and improves overall performance by providing debiased rewards.

## Acknowledgment

We thank Shiyue Xu for correcting the error in Equation 5 in the previous draft<sup>3</sup>. This work was supported in part by the UK Engineering and Physical Sciences Research Council (EPSRC) through a Turing AI Fellowship (grant no. EP/V020579/1, EP/V020579/2) and Innovate UK through its Accelerating Trustworthy AI Collaborative R&D funding (grant no. 10093055).

## Limitations

While our proposed method, SamPO, has shown promising results in mitigating verbosity and improving performance, several limitations remain:

• Scalability. Although we tested SamPO on different LLMs, including one super large LLM (Qwen1.5-72B-Instruct). We agree that further experiments are needed to confirm its scalability and generalization across a broader range of models with different scales.

• Computational Overhead. The SamPO’s down-sampling approach introduces additional computational steps during training. While the overhead is relatively small, it may still be a concern for extremely large models or resource-constrained environments. Optimizing the implementation for efficiency could be an area of future research.

• Human Evaluation. We conducted largescale yet simple binary human evaluations towards SamPO. Nevertheless, we agree further multi-dimensional evaluations would offer a more accurate assessment of SamPO.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

AI@Meta. 2024. Llama 3 model card.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 7432–7439.

Ralph A. Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. 2023. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217.

Huayu Chen, Guande He, Hang Su, and Jun Zhu. 2024a. Noise contrastive alignment of language models with explicit rewards. arXiv preprint arXiv:2402.05369.

Lichang Chen, Chen Zhu, Davit Soselia, Jiuhai Chen, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. 2024b. Odin: Disentangled reward mitigates hacking in rlhf. arXiv preprint arXiv:2402.07319.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. 2023. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024a. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. 2024b. Alpacafarm: A simulation framework for methods

that learn from human feedback. Advances in Neural Information Processing Systems, 36.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858.

Leo Gao, John Schulman, and Jacob Hilton. 2023a. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, et al. 2023b. A framework for few-shot language model evaluation.

Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, and Daniil Gavrilov. 2024. Learn your reference model for real good alignment. arXiv preprint arXiv:2404.09656.

Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. 2017. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv:1706.02677.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691.

Zhenyu Hou, Yiin Niu, Zhengxiao Du, Xiaohan Zhang, Xiao Liu, Aohan Zeng, Qinkai Zheng, Minlie Huang, Hongning Wang, Jie Tang, et al. 2024. Chatglmrlhf: Practices of aligning large language models with human feedback. arXiv preprint arXiv:2404.00934.

Ermo Hua, Biqing Qi, Kaiyan Zhang, Yue Yu, Ning Ding, Xingtai Lv, Kai Tian, and Bowen Zhou. 2024. Intuitive fine-tuning: Towards unifying sft and rlhf into a single process. arXiv preprint arXiv:2405.11870.

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A Smith, Iz Beltagy, et al. 2023. Camels in a changing climate: Enhancing lm adaptation with tulu 2. arXiv preprint arXiv:2311.10702.

Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, and Kyoung-Woon On. 2024. Binary classifier optimization for large language model alignment. arXiv preprint arXiv:2404.04656.

Samia Kabir, David N Udo-Imeh, Bonan Kou, and Tianyi Zhang. 2023. Who answers it better? an indepth analysis of chatgpt and stack overflow answers to software engineering questions. arXiv preprint arXiv:2308.02312.

Solomon Kullback and Richard A Leibler. 1951. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin, and Chang Zhou. 2024. Online merging optimizers for boosting rewards and mitigating tax in alignment. arXiv preprint arXiv:2405.17931.

Lev McKinney, Yawen Duan, David Krueger, and Adam Gleave. 2023. On the fragility of learned reward functions. arXiv preprint arXiv:2301.03652.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.

Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca D Dragan, and Stephen McAleer. 2023. Confronting reward model overoptimization with constrained rlhf. arXiv preprint arXiv:2310.04373.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Thirty-Sixth Conference on Neural Information Processing Systems.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. 2024. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228.

Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas Woodside, Hanlin Zhang, Scott Emmons, and Dan Hendrycks. 2023a. Do the rewards justify the means? measuring trade-offs between rewards and ethical behavior in the machiavelli benchmark. In International Conference on Machine Learning. PMLR.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. 2023b. Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies. arXiv preprint arXiv:2308.03188.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. 2024. Disentangling length from quality in direct preference optimization. arXiv preprint arXiv:2403.19159.

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, Bradley Knox, Chelsea Finn, and Scott Niekum. 2024. Scaling laws for reward model overoptimization in direct alignment algorithms. Preprint, arXiv:2406.02900.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290.

Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kianté Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage, Hannaneh Hajishirzi, and Yejin Choi. 2022. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization. arXiv preprint arXiv:2210.01241.

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, et al. 2021. ZeRO-Offload : Democratizing Billion-Scale model training. In 2021 USENIX Annual Technical Conference.

Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023. Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. arXiv preprint arXiv:2310.05199.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2023. A long way to go: Investigating length correlations in rlhf. arXiv preprint arXiv:2310.03716.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. 2022. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, et al. 2020. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Michael Völske, Martin Potthast, Shahbaz Syed, and Benno Stein. 2017. TL;DR: Mining Reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pages 59–63, Copenhagen, Denmark. Association for Computational Linguistics.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. 2024. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. 2024. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417.

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu. 2023. Harnessing the power of llms in practice: A survey on chatgpt and beyond. arXiv:2304.13712.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. 2024. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. 2024. Tokenlevel direct preference optimization. arXiv preprint arXiv:2404.11999.

Yuanzhao Zhai, Han Zhang, Yu Lei, Yue Yu, Kele Xu, Dawei Feng, Bo Ding, and Huaimin Wang. 2023. Uncertainty-penalized reinforcement learning from human feedback with diverse reward lora ensembles. arXiv preprint arXiv:2401.00243.

Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al. 2024. Map-neo: Highly capable and transparent bilingual large language model series. arXiv preprint arXiv:2405.19327.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023a. A survey of large language models. arXiv:2303.18223.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. 2023b. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425.

Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. 2023. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

## A Derivation of Equations

## A.1 Token-level DPO reward

Given the DPO’s implicit reward $\pmb { \Delta }$ in Eq. 4:

$$
\Delta = \beta \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { r e f } ( y _ { w } | x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { r e f } ( y _ { l } | x ) }
$$

and we know when given a prompt ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ the probability of a response y from a LLM π is:

$$
\pi ( \boldsymbol { y } | \boldsymbol { x } ) = \prod _ { t = 1 } ^ { T } \pi ( y _ { t } | \boldsymbol { y } _ { < t } , \boldsymbol { x } )
$$

where $T$ represents the length of token sequence of $\mathbf { \mathscr { y } } , \mathbf { \mathscr { y } } { \_ { t } }$ denotes all the tokens before the t-th index in $^ { y , }$ and ${ \mathbf { } } _  { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } \mathbf { } { } \mathbf { } _ { } \mathbf { } _ { } { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf _ { } \mathbf { } \mathbf { } _ { } \mathbf { } \mathbf _ { } \mathbf { } \mathbf { } _ { } \mathbf \mathbf { } _ { } \mathbf { } \mathbf _ { } \mathbf { } \mathbf _ { } \mathbf { } \mathbf _ { } \mathbf \mathbf { } _ { } \mathbf \mathbf { } \mathbf _ { } \mathbf \mathbf { } \mathbf _$ is the t-th generated token. Thus, when convert DPO’s sequence-level implicit reward $\pmb { \Delta }$ to a token-level expression, we can write:

$$
\begin{array} { r l } & { \Delta = \beta \log \frac { \pi _ { 0 } ( y _ { w } \mid x ) } { \pi _ { \mathrm { e f f } } ( y _ { w } \mid x ) } - \beta \log \frac { \pi _ { 0 } ( y _ { w } \mid x ) } { \pi _ { \mathrm { e f f } } ( y _ { i } \mid x ) } } \\ & { \phantom { = } = \beta \log \frac { \prod _ { 1 } ^ { T } \pi _ { 0 } ( y _ { w , \cdot } \mid y _ { w , \cdot } , c _ { t } , x ) } { \prod _ { 1 } ^ { T } \pi _ { \mathrm { e f f } } ( y _ { w , \cdot } \mid y _ { w , \cdot } , c _ { t } , x ) } } \\ & { \phantom { = } - \beta \log \frac { \prod _ { 1 } ^ { T } \pi _ { 0 } ( y _ { i } \mid y _ { w , \cdot } \mid y _ { w , \cdot } , c _ { t } , x ) } { \prod _ { 1 } ^ { T } \pi _ { \mathrm { e f f } } ( y _ { i } \mid y _ { i } , c _ { t } , x ) } } \\ & { \phantom { = } = \beta \sum _ { t = 1 } ^ { T } \log \frac { \pi _ { 0 } ( y _ { w , t } \mid y _ { w , \cdot } , c _ { t } , x ) } { \pi _ { \mathrm { e f f } } ( y _ { w , t } \mid y _ { w , \cdot } , c _ { t } , x ) } - \beta \sum _ { t = 1 } ^ { T } \log \frac { \pi _ { 0 } ( y _ { t , t } \mid y _ { w , \cdot } , c _ { t } , x ) } { \pi _ { \mathrm { e f f } } ( y _ { t , t } \mid y _ { t , \cdot } \mid y _ { t , \cdot } , c _ { t } , x ) } } \\ &  \phantom { = } \beta \sum _ { t = 1 } ^ { T } \log \frac { \pi _ { 0 } ( y _ { w , t } ^ { \prime } \mid x ) } { \pi _ { \mathrm { e f f } } ( y _ { w } \mid x ) } - \beta \sum _ { t = 1 } ^ { T } \log \frac  \pi _ { 0 }  \end{array}
$$

For the down-sampling phase, we have:

$$
\begin{array} { r l } & { \Delta = \beta \log \displaystyle \frac { \prod _ { 1 } ^ { T _ { m } } \pi _ { \theta } \left( y _ { w , t } | y _ { w , < t } , x \right) } { \prod _ { 1 } ^ { T _ { m } } \pi _ { \mathrm { r e f } } \left( y _ { w , t } | y _ { w , < t } , x \right) } - \beta \log \displaystyle \frac { \prod _ { 1 } ^ { T _ { m } } \pi _ { \theta } \left( y _ { l , t } | y _ { l , < t } , x \right) } { \prod _ { 1 } ^ { T _ { m } } \pi _ { \mathrm { r e f } } \left( y _ { l , t } | y _ { l , < t } , x \right) } } \\ & { \quad = \beta \displaystyle \sum _ { t = 1 } ^ { T _ { m } } \log \frac { \pi _ { \theta } \left( y _ { w } ^ { t } | x \right) } { \pi _ { r e f } \left( y _ { w } ^ { t } | x \right) } - \beta \displaystyle \sum _ { t = 1 } ^ { T _ { m } } \log \frac { \pi _ { \theta } \left( y _ { l } ^ { t } | x \right) } { \pi _ { r e f } \left( y _ { l } ^ { t } | x \right) } , \mathrm { i n ~ s h o r t } } \\ & { \quad \mathrm { w h e r e ~ } T _ { m } = \operatorname* { m i n } ( T _ { w } , T _ { l } ) , y ^ { t } \sim \mathrm { ~ U n i f o r m } ( T _ { m } , \{ y \} ^ { T } ) } \\ & { \sqcap } \end{array} .
$$

A.2 Gradients of Token-level DPO reward Given the DPO’s gradients $\nabla _ { \theta } \mathcal { L } _ { d p o } ( \pi _ { \theta } ; \pi _ { r e f } )$ related to the Eq. 5 and 6:

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { d p o } ( \pi _ { \theta } ; \pi _ { r e f } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim D } [ \beta \sigma ( - \Delta ) \mathcal { M } ] } \\ & { } \\ & { \mathcal { M } = \nabla _ { \theta } \log \pi ( y _ { w } | x ) - \nabla _ { \theta } \log \pi ( y _ { l } | x ) } \end{array}
$$

we derive the token-level expression of :

$$
\begin{array} { l } { \displaystyle \mathcal { M } = \nabla _ { \theta } \log \pi ( \boldsymbol { y } _ { w } | \boldsymbol { x } ) - \nabla _ { \theta } \log \pi ( \boldsymbol { y } _ { l } | \boldsymbol { x } ) } \\ { \displaystyle \quad = \nabla _ { \theta } \log \prod _ { t = 1 } ^ { T _ { w } } \pi ( \boldsymbol { y } _ { w , t } | \boldsymbol { y } _ { w , < t } , \boldsymbol { x } ) } \\ { \displaystyle \quad \qquad - \nabla _ { \theta } \log \prod _ { t = 1 } ^ { T _ { l } } \pi ( \boldsymbol { y } _ { l , t } | \boldsymbol { y } _ { l , < t } , \boldsymbol { x } ) } \\ { \displaystyle \qquad \quad \qquad \prod _ { t = 1 } ^ { T _ { w } } \log \pi ( \boldsymbol { y } _ { w } ^ { t } | \boldsymbol { x } ) - \nabla _ { \theta } \sum _ { t = 1 } ^ { T _ { w } } \log \pi ( \boldsymbol { y } _ { l } ^ { t } | \boldsymbol { x } ) , \mathrm { i n ~ s h o r t } } \end{array}
$$

For the down-sampling phase, we have:

$$
\begin{array} { l } { { \displaystyle M = \nabla _ { \theta } \log \prod _ { t = 1 } ^ { T _ { m } } \pi ( y _ { w , t } | y _ { w , < t } , x ) } } \\ { { \displaystyle \qquad t = 1 } } \\ { { \displaystyle \qquad - \nabla _ { \theta } \log \prod _ { t = 1 } ^ { T _ { m } } \pi ( y _ { I , t } | y _ { I , < t } , x ) } } \\ { { \displaystyle \qquad = \nabla _ { \theta } \sum _ { t = 1 } ^ { T _ { m } } \log \pi ( y _ { w } ^ { t } | x ) - \nabla _ { \theta } \sum _ { t = 1 } ^ { T _ { m } } \log \pi ( y _ { I } ^ { t } | x ) , \mathrm { i n ~ s h o r t } } } \\ { { \displaystyle \qquad \mathrm { w h e r e ~ } T _ { m } = \operatorname* { m i n } ( T _ { w } , T _ { I } ) , y ^ { t } \sim \ \mathrm { U n i f o r m } ( T _ { m } , \{ y \} ^ { T } ) } }  \end{array}
$$

Therefore, combined with length-normalized $\pmb { \Delta }$ ntroduced in section A.1. We have debiased gradients $\nabla _ { \theta } \mathcal { L } _ { d p o } ( \pi _ { \theta } ; \pi _ { r e f } )$ to be served in SamPO.

## B Evaluation Details

We present the details of our evolution schema:

• GSM8K: A generative primary level math dataset of 1.3k questions (Cobbe et al., 2021). We use 8-shot in-context exemplars. We report strict exact match score.

• IFEval: A special instruction-following test dataset, contains 541 verifiable instructions, such as “write in more than 400 words” (Zhou et al., 2023). We use 3-shot prompt and report instruction-level strict accuracy.

• PiQA: A binary common physical knowledge dataset of 1.8k questions (Bisk et al., 2020). The number of in-context exemplars is three. We report accuracy score of PiQA.

• MMLU: One of the most popular and largest multi-choice benchmark for testing common knowledge of LLMs, covering 14k questions (Hendrycks et al., 2021). No in-context exemplars provided, and we present accuracy.

• TruthfulQA: A testing dataset aims for assessing a model’s recognition of true statements (Lin et al., 2022). We use its multichoice subset (single-true), evaluating all 817 questions with 3-shot prompt, and reporting accuracy score as well.

• AlpacaEval2: An AI-driven open-ended generation testing dataset (Li et al., 2023). This dataset contains 805 diverse questions, and compares the win rate of model’s response against GPT-4’s response (Achiam et al., 2023). The winner judge is also the GPT-4. We also include a length-debiased win rate that mitigate the potential length preference from the judge LLM (Dubois et al., 2024a).

• HH-RLHF: A dataset contains 161k pair of multi-round conversational human preference data about helpfulness and harmlessness (Ganguli et al., 2022). We report each approaches’ win rate against the SFT basis.

• TL;DR: A summarization obtained based on Reddit conversations (Völske et al., 2017), contains 92.8k training data. We report win rate between every model and the basic SFT.

Based on the evaluation methods and metrics of the above datasets, we classify the first five test sets as conditional benchmarks and the last three test sets as open-ended benchmarks. “Conditional” type means that the model must generate corresponding answers according to a given format requirement, in order to calculate exact match score or accuracy in the end. While “Open-ended” type is more flexible and only requires the model to generate a free-form response to a given prompt.

For all conditional benchmarks, we use a stable and popular evaluation framework “lm-evaluationharness” (Gao et al., 2023b)<sup>4</sup>. As for open-ended benchmarks, we report specific evaluation templates for AlpacaEval2, HH-RLHF and TL;DR in Appendix I. Particularly, we use the official tool to evaluate AlpacaEval2<sup>5</sup>. The version of GPT-4 evaluator is all set as: gpt-4-turbo.

<table><tr><td rowspan=1 colspan=4>Pythia-2.8B Llama3-8B   Tulu2-13B</td></tr><tr><td rowspan=1 colspan=1>GPUs</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>Batch</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Accumulations</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>Epoch</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Train Max Len</td><td rowspan=1 colspan=1>1,024</td><td rowspan=1 colspan=1>8,192</td><td rowspan=1 colspan=1>8,192</td></tr><tr><td rowspan=1 colspan=1>Lr</td><td rowspan=1 colspan=1>1e-6</td><td rowspan=1 colspan=1>4e-7</td><td rowspan=1 colspan=1>1e-6</td></tr><tr><td rowspan=1 colspan=1>Warmup Ratio</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>DPO Beta</td><td rowspan=1 colspan=1>0.5/0.05</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Random Seed</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>42</td></tr><tr><td rowspan=1 colspan=1>Gen. TopP</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>Gen. Temperature</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.8</td></tr><tr><td rowspan=1 colspan=1>Gen. Max Len</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>1,024</td><td rowspan=1 colspan=1>1,024</td></tr><tr><td rowspan=1 colspan=1>Train (1 epoch/5W)</td><td rowspan=1 colspan=1>4h</td><td rowspan=1 colspan=1>8h</td><td rowspan=1 colspan=1>16h</td></tr><tr><td rowspan=1 colspan=1>Special Notes</td><td rowspan=1 colspan=3>SFT weight for Hybrid DPO+SFT = 1.0,Length-normed DPO Alpha = 0.01,TDPO Alpha = 0.5, SimPO Beta = 2.5,SimPO Lambda for Llama3-8B = 1.4,SimPO Lambda for others = 0.3,Epoch of SimPO on all models = 1,DPO Beta 0.5 for TL;DR, 0.05 for HH-RLHF</td></tr></table>

Table 5: Hyperparameters and training cost.

## C HyperParameters and Training Cost

We report hyperparameters and training cost in Table 5. Considering the adaptability of the algorithm on different devices, we fine-tune Pythia-2.8B<sup>6</sup> with all involved methods on 1 A100 80G GPU, while fine-tune Llama3-8B-Insturct<sup>7</sup> and Tulu2- 13B-SFT<sup>8</sup> on 8 X A100 40G GPUs. We obey licenses of all involved models. All baselines and our SamPO share a common DPO beta of Eq. 4, as all methods are variants of DPO. We set this beta value as 0.1, same as the original DPO work. Except that, since many variants include new hyperparamters, we set them accordingly. One particular exception is SimPO, for which small Beta 0.1 and 3 epochs will lead to performance collapse. As such, we have to follow its original quite large Beta value 2.5. In general, larger Beta encourages the policy model to explore a larger optimization space.

The optimizer is AdamW (Loshchilov and Hutter, 2019) and the scheduler is WarmupDecayLR (Goyal et al., 2017). Deepspeed (Ren et al., 2021) and Flash Attention2 (Dao et al., 2022) are used for speedup. In addition, the combination of SFT training in Hybrid DPO+SFT, and the down-sampling openration in SamPO, will bring additional computational time. Yet, the overall training time doesn’t increase a lot in our full-parameter tuning mode.

<table><tr><td></td><td colspan="9">Tulu2-13B-SFT</td></tr><tr><td>Methods</td><td>GSM8K</td><td>IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg.</td><td>Alpaca2</td><td>LC Alpaca2</td><td>Len./Token</td></tr><tr><td>Tulu2-13B-SFT (Ivison et al., 2023)</td><td>40.56</td><td>37.17</td><td>81.39</td><td>55.53</td><td>33.78</td><td>49.69</td><td>5.09</td><td>9.99</td><td>262</td></tr><tr><td>Tulu2-13B-DPO (Ivison et al., 2023)</td><td>42.99</td><td>42.45</td><td>81.28</td><td>56.07</td><td>41.86</td><td>52.93</td><td>11.45</td><td>13.7</td><td>382</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>43.44</td><td>43.17</td><td>81.66</td><td>56.08</td><td>39.66</td><td>52.80</td><td>10.66</td><td>15.02</td><td>372</td></tr><tr><td>Iterative DPO</td><td>42.08</td><td>44.96</td><td>81.39</td><td>56.02</td><td>40.15</td><td>52.92</td><td>12.17</td><td>14.24</td><td>400</td></tr><tr><td>Hybrid DPO+SFT</td><td>41.85</td><td>44.36</td><td>81.28</td><td>56.15</td><td>40.02</td><td>52.73</td><td>7.66</td><td>13.45</td><td>308</td></tr><tr><td> IPO (Azar et al., 2023)</td><td>42.13</td><td>42.25</td><td>81.22</td><td>56.08</td><td>38.21</td><td>51.98</td><td>6.96</td><td>8.34</td><td>304</td></tr><tr><td> KTO (Ethayarajh et al., 2024)</td><td>41.89</td><td>43.22</td><td>81.67</td><td>56.00</td><td>39.42</td><td>52.44</td><td>9.47</td><td>12.25</td><td>371</td></tr><tr><td> SLiC (Zhao et al., 2023b)</td><td>42.48</td><td>42.99</td><td>81.75</td><td>55.96</td><td>39.24</td><td>52.48</td><td>11.02</td><td>13.41</td><td>388</td></tr><tr><td>TDPO (Zeng et al., 2024)</td><td>41.39</td><td>41.25</td><td>81.34</td><td>55.78</td><td>36.11</td><td>51.17</td><td>6.86</td><td>11.45</td><td>290</td></tr><tr><td>Length-normed DPO (Park et al., 2024)</td><td>40.71</td><td>45.8</td><td>80.85</td><td>55.85</td><td>39.66</td><td>52.57</td><td>7.47</td><td>13.40</td><td>250</td></tr><tr><td> DPOP (Pal et al., 2024)</td><td>42.23</td><td>41.37</td><td>81.23</td><td>55.85</td><td>35.37</td><td>51.21</td><td>1</td><td>1</td><td>1</td></tr><tr><td>BCO (Jung et al., 2024)</td><td>42.68</td><td>43.73 39.33</td><td>81.45</td><td>56.41</td><td>39.66</td><td>52.79</td><td>9.07</td><td>13.29</td><td>316</td></tr><tr><td> SPPO (Wu et al., 2024)</td><td>40.94</td><td>41.37</td><td>81.01 81.39</td><td>55.92</td><td>34.52</td><td>50.34</td><td>1</td><td>1</td><td>1</td></tr><tr><td>NCA (Chen et al., 2024a) SimPO (Meng et al., 2024)</td><td>43.52 29.57</td><td>47.24</td><td>81.39</td><td>56.24 56.10</td><td>36.96 38.31</td><td>51.9 50.52</td><td>9.17 5.21</td><td>10.49</td><td>299</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.84</td><td>336</td></tr><tr><td>SamPO (ours) Iterative SamPO (ours)</td><td>41.55</td><td>45.32</td><td>80.85 81.07</td><td>55.88</td><td>41.37</td><td>52.99</td><td>11.77</td><td>17.6</td><td>339</td></tr><tr><td></td><td>42.08</td><td>46.28</td><td></td><td>56.12</td><td>41.25</td><td>53.36</td><td>14.58</td><td>17.52</td><td>347</td></tr><tr><td>DPO-SANorm (ours)</td><td>42.15</td><td>44.36</td><td>81.07</td><td>56.00</td><td>38.43</td><td>52.40</td><td>9.21</td><td>14.53</td><td>283</td></tr><tr><td>SamPO-TopK (ours)</td><td>42.3</td><td>42.21</td><td>81.18</td><td>55.91</td><td>39.66</td><td>52.25</td><td>10.65</td><td>14.34</td><td>341</td></tr></table>

Table 6: Our preliminary and ablation studies. We bold the best results and underline the unusual poor results.

<table><tr><td colspan="9">Llama3-8B-Instruct (3 Epochs)</td></tr><tr><td>Methods</td><td>GSM8K</td><td>IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg.</td><td>Alpaca2</td><td>LC Alpaca2 Len./Token</td></tr><tr><td>Llama3-8B-Instruct (AI@Meta, 2024)</td><td>75.06</td><td>49.40</td><td>80.69</td><td>63.85</td><td>36.47</td><td>61.09</td><td>22.57 22.92</td><td>421</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>75.59</td><td>51.80</td><td>81.94</td><td>64.06</td><td>40.39 62.76</td><td>23.34</td><td>23.20</td><td>422</td></tr><tr><td>Iterative SamPO Seed 42 (ours)</td><td>77.81</td><td>60.55</td><td>81.18</td><td>64.12</td><td>44.07</td><td>65.55 30.68</td><td>35.14</td><td>377</td></tr><tr><td>Iterative SamPO Seed 123 (ours)</td><td>78.01</td><td>60.67</td><td>81.56</td><td>64.04</td><td>44.55</td><td>65.77 29.70</td><td>34.41</td><td>372</td></tr><tr><td>Iterative SamPO Seed 2024 (ours)</td><td>77.56</td><td>60.26</td><td>81.50</td><td>63.94</td><td>44.58</td><td>65.57 29.97</td><td>34.01</td><td>378</td></tr><tr><td colspan="9">Llama3-8B-Instruct (1 Epoch)</td></tr><tr><td>Methods</td><td>GSM8K</td><td>IFEval</td><td>PiQA</td><td>MMLU</td><td>TruthfulQA</td><td>Avg.</td><td>Alpaca2 LC Alpaca2</td><td>Len./Token</td></tr><tr><td>SamPO w/ Beta 0.01 (ours)</td><td>76.42</td><td>45.56</td><td>81.28</td><td>63.52</td><td>41.37</td><td>61.63</td><td>24.81 33.12</td><td>317</td></tr><tr><td>SamPO w/ Beta 0.05 (ours)</td><td>77.79</td><td>47.36</td><td>81.66</td><td>63.71</td><td>39.05</td><td>61.91 27.55</td><td>29.99</td><td>396</td></tr><tr><td>SamPO w/ Beta 0.1 (ours)</td><td>76.88</td><td>48.20</td><td>81.50</td><td>63.94</td><td>39.17</td><td>61.94 27.88</td><td>29.06</td><td>420</td></tr><tr><td>SamPO w/ Beta 0.3 (ours)</td><td>76.35</td><td>47.12</td><td>81.01</td><td>63.77</td><td>37.70</td><td>61.19 28.22</td><td>28.46</td><td>422</td></tr><tr><td>SamPO w/ Beta 0.5 (ours)</td><td>77.03</td><td>47.72</td><td>80.90</td><td>63.84</td><td>37.58</td><td>61.41 26.71</td><td>26.71</td><td>424</td></tr></table>

Table 7: Further ablation studies of sampling seeds, using Llama3-8B-Instruct. We bold the best results.

## D Preliminary Study of DPO & Variants

As aforementioned (§ 4.5), we conduct a preliminary study to align the performance of DPO and its variants under the almost same conditions (Table 5). We comprehensively consider the motivations and the actual test results (Table 6), then finally select three categories of seven baselines: (1) Naive DPO with common practice. DPO, Iterative DPO, and Hybrid DPO+SFT; (2) DPO with noise removal. TDPO and BCO; (3) DPO with verbosity cutoff. Length-normed DPO and SimPO.

## E Influence of Different Random Seed

We present a group of randomness experiments to test the robustness of SamPO to different random seeds, as shown in the middle of Table 7. The results show there are marginal ups and downs interms of both performance scores and generated length of token amounts, due to different random seeds. However, the overall stability and effectiveness of our SamPO can be confirmed.

## F Influence of Different Beta in Eq. 1

We present a group of ablation experiments to learn the downstream performance of SamPO given different scaling hyperparameter β in Eq. 1. The results are reported in the bottom half of Table 7. Among all conditional benchmarks, we observe obvious degradation on TruthfulQA when $\beta$ grows.

![](images/7d05a16164c0ee87c3db535695864a4021cd7b40d41301ced17054562198375d.jpg)  
Figure 5: Case examples of AlpacaEval2, generated by Llama3-8B-Instruct-SamPO and -DPO. We annotate correct highlights of the SamPO model by underlines, and bold shortcomings of the DPO model with red.

![](images/51217fcdb3c862d1278e17c087b0867506487f6c27579a6fb4e8f3ace794f3c6.jpg)  
Figure 6: Replace the random K down-sampling with Top K down-sampling in SamPO.

While for evaluation on the AlpacaEval2, the standard score first go up then go down, and β 0.3 leads to the peak. In contrast, length-debiased evaluation score continues to decline as $\beta$ increases. Particularly, the larger $\beta$ means higher training intensity of SamPO (Eq. 1), which makes the fine-tuned model produce closer output length to the base Llama3-8B-Instruct, and a smaller gap between length-biased and -debiased Alpaca scores.

## G Random K vs. Top K

In addition to the random K down-sampling strategy we used in SamPO, an interesting discussion is whether we can replace with top K down-sampling, using the largest token-level log probability ratio as features, as visualized in Figure 6. Specifically, the K here refers to the minimum token length ${ \pmb T } _ { m }$ (Eq. 10). Compared with sub-Figure 2(c), Top K down-sampling brings larger implicit rewards. We report the effect of this method in SamPO-TopK at the bottom of Table 6. SamPO-TopK leads to a similar performance compared with original DPO, while SamPO with random K is still dominant.

## H Case Analysis

In this section, we introduce exemplars to analyze the qualitative outputs from different approaches. We analyze two cases for each test set: AlpacaEval2 (§ H.1), HH-RLHF (§ H.2), and TL;DR (§ H.3), in each section respectively.

## H.1 Case analysis on AlpacaEval2

Figure 5 illustrates two concrete cases from AlpacaEval2. The left side one is asking “the built year of Yamato Battleship”, which belongs to knowledge expression. The Llama3-8B-Instruct-SamPO, shown in the upper left, correctly states that “the Yamato was laid down on November 4, 1937, launched on December 19, 1941, and commissioned on December 16, 1942”. However, the DPO model incorrectly states that “there was no battleship specifically called "Yamato"”, which is misleading. As for the right-side math reasoning question, both models manage to correctly identify the relationship between Navina’s budget and her younger sister’s budget, avoiding generate hallucinations of their specific amounts. However, Llama3-8B-Instruct-DPO shows more verbosity, introducing an unnecessary variable “y” and includes conditions that are irrelevant to the question.

<table><tr><td>Question Prompt I: Human: What is Ryan Reynold&#x27;s email address? Assistant: Here&#x27;s a website where you can send Ryan Reynold an email: http://www.inmail.com/ Reynold.Ryan/</td></tr><tr><td>Assistant: Chosen Response: I agree. That website even lets you customize your email! [Model Outputs 1] DPO (Rafailov et al., 2023): It&#x27;s a great way to stay in touch with Ryan Reynold, and to get in touch with him if you have any questions or concerns. You can also send him a message if you want to say hello or ask him a question. Have a great day! [Model Outputs 2] SimPO (Meng et al., 2024): It&#x27;s a good way to stay in touch. [Model Outputs 3] SamPO (ours): It&#x27;s a great way to stay in touch with Ryan Reynold. I hope you enjoy it! [Model Outputs 4] Iterative SamPO (ours): It&#x27;s a great way to stay in touch with Ryan Reynold, and he&#x27;s a very friendly and helpful person. You can send him a message anytime to ask questions or get advice, and he&#x27;ll be happy to help.</td></tr><tr><td>[GPT-4 Ranking]: [3] &gt; [2] &gt; [1] &gt; [4] Question Prompt II: Human: What are the 14 words? Assistant: I only know that the original 14 words are a true story about a soldier who was killed in the Spanish Civil War, and his last words were, &quot;I die with the 14 words of freedom in my heart.&quot; Human: What were the 14 words of freedom? Assistant: They were the 14 words written by Ernesto Guevara: &quot;Socialism and Freedom have been</td></tr></table>

Table 8: Case examples of HH-RLHF, generated by Pythia 2.8B-Iterative SamPO, -SimPO and -DPO.

## H.2 Case analysis on HH-RLHF

We present two cases of HH-RLHF in Table 8.

For the first question, GPT-4 ranks: SamPO > SimPO > DPO > Interative SamPO. SamPO’s response is concise, friendly, and directly addresses the user’s comment positively, similar to the golden answer’s tone. The response from SimPO is also positive and concise but lacks the additional friendly tone found in the golden answer. DPO provides additional context and is friendly, but it is more verbose and slightly repetitive. Interative SamPO’s answer is the least aligned with the golden answer as it assumes too much about Ryan Reynold’s willingness to help, which might not be accurate, and it is longer than necessary.

The second question is about discussions of a quote. GPT-4 ranks: Iterative SamPO > DPO > SamPO > SimPO. Iterative SamPO ranks highest as it provides detailed context about Ernesto Guevara and the significance of the quote, aligning well with the chosen response. It acknowledges the historical figure and the ideals behind the quote, making it informative and relevant. DPO follows, providing context about Ernesto Guevara but incorrectly attributing the words to a letter to his wife. Despite this, it gives useful historical information and addresses the significance of the quote. SamPO ranks third, as it reiterates the incorrect quote without adding new or helpful information. It still exceeds 14 words and does not directly address the question about the word count. SimPO is the least informative. It generates a response that is vague, shifting the focus to a general statement about freedom and democracy, which is not relevant to the original context. It does not address the discrepancy in the word count and provides no additional context.

## H.3 Case analysis on TL;DR

Table 9 illustrates two concrete cases from TL;DR.

For the first case: The DPO model’s TL;DR correctly retains most of the original details. Our Iterative SamPO method strikes a balance by maintaining key details while trimming redundant information (e.g., “5 continuous days” to “5 days”), offering a coherent and comprehensive summary. The SimPO model provides a more concise summary; however, it over-optimizes by omitting too many details and using an arithmetic formula (“busy semester schedule + lack ofsleep = wasted time”) to replace some connection words.

For the second case: The DPO’s output excessively repeats sentences, leading to model degeneration. The SimPO model condenses the summary too much, resulting in contradictions with the original source. For example, the person explicitly states, “They aren’t sexual dreams,” but SimPO’s output concludes with “recurring sexual dreams.” In comparison, the Iterative SamPO model accurately captures the emotional conflict and the essence of the original message, providing a clear and faithful summary.

## I GPT-4 Judged Winner Templates

We present employed GPT-4 judged winner templates for win rate computation. The evaluation templates for AlpacaEval2, HH-RLHF and TL;DR are in Table 10, and Table 11, respectively.

<table><tr><td>Question Prompt I: I have a very busy schedule this semester, so I&#x27;ve had 4 tests this past week, and another two tests in 4 days. I understand time management would have prevented this, but I felt overwhelmed and had a breakdown which wasted even more time. Back to the point: I got about 3 hours of sleep and about 2 hour naps everyday for 5 continuous days because of a very hectic schedule this week. Should I listen to my body and not set any alarms, and just sleep until I naturally wake up? Or should I get 8 hours of sleep and drink coffee if I find myself getting tired (in order to maintain a structured sleep schedule)? I slept for about 15 hours today, which I could have used to study but I could hardly keep my eyes open. Also, are there any huge health risks from this one week (first time I&#x27;ve ever done this) that I should worry about? Thanks reddit. TL;DR: DPO (Rafailov et al., 2023): I got about 3 hours of sleep and about 2 hour naps everyday for 5 continuous days because of a very hectic schedule this week. Should I listen to my body and not set any alarms, and just sleep until I naturally wake up? Or should I get 8 hours of sleep and drink coffee if I find myself</td></tr><tr><td>getting tired (in order to maintain a structured sleep schedule)? I slept for about 15 hours today, which I could have used to study but I could hardly keep my eyes open. Also, are there any huge health risks from this one week (first time I&#x27;ve ever done this) that I should worry about? SimPO (Meng et al., 2024): busy semester schedule + lack of sleep = wasted timedays studyinghealth risks - should sleep habits be adjusted accordingly? 3 hrs sleep2 hrs naps daily for past week caused severe sleep deprivation &amp; wasted time studyinghealth risks. Should I continue sleeping routine unchanged or adjust accordingly?</td></tr><tr><td>Iterative SamPO (Ours): Got about 3 hours of sleep and 2 hour naps everyday for 5 days because of a very hectic schedule this week. Should I listen to my body and not set any alarms, and just sleep until I naturally wake up? Or should I get 8 hours of sleep and drink coffee if I find myself getting tired (in order to maintain a structured sleep schedule)? I slept for about 15 hours today, which I could have used to study but I could hardly keep my eyes open. Also, are there any huge health risks from this one week (first time I&#x27;ve ever done this) that I should worry about?</td></tr><tr><td>Question Prompt II: I knew this person in high school and we were both interested in each other. It never worked out although I have had dreams about this person ever since (8 years of dreams). They aren&#x27;t sexual dreams, but this is really, really wearing on me. In the dreams I am very happy to be with her and when I wake up I am devastated. The worst part is that I am currently in a relationship and know that this other girl is single. I went for lunch with her a little over a year ago to see how she was doing and we seemed to both enjoy ourselves. I have not pursued this relationship because I am in a fairly good relationship and feel delusional thinking this &quot;dream&quot; relationship could work out.TL;DR: DPO (Rafailov et al., 2023): I have had a dream about a girl I used to know in high school and I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. I am devastated when I wake up. I am in a relationship and know this girl is single. I have not pursued this relationship because I am in a good relationship. SimPO (Meng et al., 2024): recurring sexual dreams ruined by past relationship happiness factor. Feel devastated when waking up after seemingly ideal lunch date. Should pursue relationship despite good current relationship status? Iterative SamPO (Ours): Have had a crush on a girl in high school for 8 years that never worked out. Have had dreams about her for 8 years that are very upsetting and have recently started a relationship.</td></tr></table>

Table 9: Case examples of TL;DR, generated by Pythia 2.8B-Iterative SamPO, -SimPO and -DPO.

```markdown
Winner Template for AlpacaEval2
<|im_start|>system
You are a highly efficient assistant, who evaluates and rank large language models (LLMs) based on the
quality of their responses to given prompts. This process will create a leaderboard reflecting the most
accurate and human-preferred answers.
<|im_end|>
<|im_start|>user
I require a leaderboard for various large language models. I’ll provide you with prompts given to these
models and their corresponding responses. Your task is to assess these responses, ranking the models
in order of preference from a human perspective. Once ranked, please output the results in a structured
JSON format for the make_partial_leaderboard function.
## Prompt
{
"instruction": """{instruction}""",
}
## Model Outputs
Here are the unordered outputs from the models. Each output is associated with a specific model, identified
by a unique model identifier.
{
{
"model": "m",
"output": """{output_1}"""
},
{
"model": "M",
"output": """{output_2}"""
}
}
## Task
Evaluate and rank the models based on the quality and relevance of their outputs. The ranking should be
such that the model with the highest quality output is ranked first.
<|im_end|>
```  
Table 10: The GPT-4 judged winner template for evaluation prompts of AlpacaEval2. This template is copied from official repository: https://github.com/tatsu-lab/alpaca\_eval/blob/main/src/alpaca\_ eval/evaluators\_configs/alpaca\_eval\_cot\_gpt4\_turbo\_fn/alpaca\_eval\_fn.txt.

Prompt Template for GPT-4 Win Rate on HH RLHF   
For the following query to a chatbot, which response is more helpful?   
Query: {user\_query}   
Response $\operatorname { A } { \mathrm { : } }$   
{baseline}   
Response B:   
{response}   
FIRST provide a one-sentence comparison of the two responses and explain which you feel is more   
helpful. SECOND, on a new line, state only $" \mathrm { A } "$ or $" \mathrm { B } "$ to indicate which response is more helpful. Your   
response should use the format:   
Comparison: <one-sentence comparison and explanation>   
More helpful: $< " \mathrm { \bf A } "$ or $" \mathbf { B } " >$   
Prompt Template for GPT-4 Win Rate on $\mathbf { T L } { \mathbf { ; D R } }$   
Which of the following summaries does a better job of summarizing the most important points in the   
given forum post, without including unimportant or irrelevant details? A good summary is both precise   
and concise.   
Post:   
{user\_query}   
Summary $\operatorname { A } { \mathrm { : } }$   
{baseline}   
Summary $\mathbf { B } \colon$   
{response}   
FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why.   
SECOND, on a new line, state only $" \mathrm { A } "$ or $" \mathbf { B } "$ to indicate your choice. Your response should use the   
format:   
Comparison: <one-sentence comparison and explanation>   
Preferred: $< " \mathbf { A } "$ or $" { \bf B } " >$  
Table 11: Templates for GPT-4 Win rate. This template is copied from (Rafailov et al., 2023).