# An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification

Zhuowei Chen<sup>1</sup>, Lianxi Wang<sup>1,2</sup> ,

Yuben Wu<sup>1</sup>, Xinfeng Liao<sup>1</sup>, Yujia Tian<sup>1</sup>, Junyang Zhong<sup>1</sup>

<sup>1</sup>Guangdong University of Foreign Studies, Guangzhou, China.

<sup>2</sup>Guangzhou Key Laboratory of Multilingual Intelligent Processing, Guangzhou, China. wanglianxi@gdufs.edu.cn

## Abstract

Sentiment classification (SC) often suffers from low-resource challenges such as domainspecific contexts, imbalanced label distributions, and few-shot scenarios. The potential of the diffusion language model (LM) for textual data augmentation (DA) remains unexplored, moreover, textual DA methods struggle to balance the diversity and consistency of new samples. Most DA methods either perform logical modifications or rephrase less important tokens in the original sequence with the language model. In the context of SC, strong emotional tokens could act critically on the sentiment of the whole sequence. Therefore, contrary to rephrasing less important context, we propose DiffusionCLS to leverage a diffusion LM to capture in-domain knowledge and generate pseudo samples by reconstruct ing strong label-related tokens. This approach ensures a balance between consistency and diversity, avoiding the introduction of noise and augmenting crucial features of datasets. DiffusionCLS also comprises a Noise-Resistant Training objective to help the model generalize. Experiments demonstrate the effectiveness of our method in various low-resource scenarios including domain-specific and domain-general problems. Ablation studies confirm the effec tiveness of our framework’s modules, and visualization studies highlight optimal deployment conditions, reinforcing our conclusions.

## 1 Introduction

Sentiment classification is a crucial application of text classification (TC) in Natural Language Processing (NLP) and can play a crucial role in multiple areas. However, NLP applications in domainspecific scenarios, such as disasters and pandemics, often meet with low-resource conditions, especially domain-specific problems, imbalance data distribution, and data deficiency (Sedinkina et al., 2022;

Lakshmi and Velmurugan, 2023; Nabil et al., 2023; Gatto and Preum, 2023). Recently, the birth of pre-trained language models (PLMs) and large language models (LLMs) have advanced the NLP field, giving birth to numerous downstream models based on them. On the one hand, these PLMs take the models to a new height of performance, on the other hand, since these models are highly data-hungry, they struggle to perform satisfactorily on most tasks under noisy, data-sparse and low-resource conditions (Patwa et al., 2024; Chen et al., 2023b; Wang et al., 2024; Yu et al., 2023).

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Textual Sample</td></tr><tr><td rowspan=2 colspan=1>OtherCTR Methods(GENIUS)</td><td rowspan=1 colspan=1>Cor.</td><td rowspan=1 colspan=1>[sad] [M] the traffic [M] a nightmare.[M][M][M]frustrating.</td></tr><tr><td rowspan=1 colspan=1>Gen.</td><td rowspan=1 colspan=1>[sad] Navigating the traffic was literallya nightmare. Truly frustrating.</td></tr><tr><td rowspan=2 colspan=1>Diffusion-CLS(ours)</td><td rowspan=1 colspan=1>Cor.</td><td rowspan=1 colspan=1>[sad] Today, the [M] was [M][M].It was [M][M].</td></tr><tr><td rowspan=1 colspan=1>Gen.</td><td rowspan=1 colspan=1>[sad] Today, the journey was a disaster.It was utterly chaotic.</td></tr><tr><td rowspan=1 colspan=3>Today, the traffic was a nightmare.Original Text:It was really frustrating.</td></tr></table>

Table 1: Examples of CTR methods. Most CTR methods rephrase minor tokens while DiffusionCLS reconstructs strong label-related tokens. Cor. and Gen. denotes the corrupted sequence and generated sequence respectively.

To address these challenges, one effective approach is data augmentation (DA), which enriches the diversity of the dataset without explicitly collecting new data (Feng et al., 2021). Classic rulebased DA methods employ logical modifications to obtain pseudo samples, such as EDA (Wei and Zou, 2019), and AEDA (Karimi et al., 2021). Modelbased DA methods develop rapidly as the transformer architecture dominates the NLP field, most of these methods execute DA through corrupt-thenreconstruct (CTR), as examples shown in Table 1. Namely, masked language model (MLM) (Wu et al., 2019; Kumar et al., 2020), and GENIUS (Guo et al., 2022) which applies BART as the sample generator. Also, Anaby-Tavor et al. (2020) proposed LAMBADA, which finetunes GPT-2 and generates pseudo samples with label prompts.

However, these methods often struggle with domain-specific tasks and uneven label distributions. Some methods generate samples solely relying on pre-trained knowledge, like GENIUS. The other though finetuned on the downstream dataset, these methods generate samples only conditioned on the label itself, such as LAMBADA, leading to strong label inconsistency, especially in data-sparse settings. Also, most CTR methods focus on replacing minor tokens in sequences but keeping the crucial tokens stationary to generate high-quality pseudo samples.

In contrast, we corrupt the most label-related tokens first and reconstruct the whole sentence conditioned on the context and label prompt, as shown in Table 1, to diversify the key label-related tokens rather than less important contexts. This approach not only augments sample diversity but also upholds consistency through selective masking. Inspired by DiffusionBERT (He et al., 2023), which is designed to recover the most informative tokens from those with less informatics, we propose DiffusionCLS. Additionally, building upon the findings of Guo et al. (2022), we further introduce consistency and diversity as crucial elements for quality of samples. High-quality pseudo samples must align with their labels and domain contexts, minimizing noise introduction. Integrating these samples enhances dataset diversity, thereby positively impacting the model performance.

DiffusionCLS initially finetunes PLM with a diffusion objective, functioning as a sample generator, followed by training the TC model in a noiseresistant manner. By fine-tuning the diffusion LM, we can then input original samples with their crucial tokens corrupted and use the label as a generation prompt to get new samples. This method diversifies the original dataset by replacing strong label-related tokens and also steers the model towards producing high-quality pseudo samples that comply with the diversity-consistency rule. Also, experimental codes have been released on GitHub<sup>1</sup>.

The major contributions of this paper can be summarized as follows:

• We propose DiffusionCLS, which comprises a diffusion LM-based data augmentation module for SC, generating diverse but consistent pseudo samples by substituting diverse strong label-related contexts.

• Designed and integrated a noise-resistant training method within the proposed DiffusionCLS, which significantly improves the SC model’s performance with pseudo samples.

• Comprehensive experiments on domainspecific and multilingual datasets validate DiffusionCLS’s superior performance in SC tasks. Detailed ablation studies highlights the effectiveness of its individual modules.

• A visualization study is conducted to discuss the diversity-consistency trade-off, which further validates the effectiveness of Diffusion-CLS.

## 2 Related Work

## 2.1 Low-Resource Text Classification

Motivated by the observation that data is often scarce in specific domains or emergent application scenarios, low-resource TC (Chen et al., 2018) has recently attracted considerable attention. Lowresource TC involves effectively categorizing text in scenarios where data is scarce or limited. Goudjil et al. (2018) and Tan et al. (2019) have explored several methods for low-resource TC, which mainly involve traditional machine learning techniques to increase data quantity and diversity.

Recently, since the studies by Lan et al. (2019) and Sun et al. (2020) demonstrated the impressive performance of PLMs across various NLP tasks, a significant amount of work has leaned towards using PLMs to address low-resource TC problems (Wen and Fang, 2023; Ogueji et al., 2021; Liu et al., 2019; Devlin et al., 2018). However, PLMs requires amounts of annotated samples for finetuning, data-sparce significantly impacts models’ performances and DA could mitigate such problems.

## 2.2 Textual Data Augmentation

To address low-resource challenges, various data augmentation methods have been proposed, including Easy-Data-Augmentation (EDA) (Wei and Zou, 2019), Back-Translation (BT) (Shleifer, 2019), and CBERT (Wu et al., 2019). However, these methods, relying on logical replacements and external knowledge, often introduce out-domain knowledge and domain inconsistency. Moreover, these methods focus only on a specific original input, resulting in limited diversity.

![](images/3fca467823cf2b26d6d5c7069f48b6c65c66a281ec9a3455443d22adb4adb1ac.jpg)  
Figure 1: Overview of the proposed method. DiffusionCLS comprises four core components: Label-Aware Noise Schedule, Label-Aware Prompting, Conditional Sample Generation, and Noise-Resistant Training.

Another type of data augmentation method includes representation augmentation approaches. These methods generate pseudo-representation vectors by interpolating or perturbing the representations of original samples. For instance, Zhang et al. (2017) proposed the groundbreaking technique known as mixup, and Chen et al. (2023a) recently proposed AWD, an advanced approach in textual DA.

Recent advancements in generative models have led to research on GPT-based paraphrasing data augmentation methods, such as LAMBADA (Anaby-Tavor et al., 2020), which fine-tuned GPT-2 model to generate new samples. However, LAM-BADA generates new samples based solely on specific labels, neglecting information from the original samples. Another research direction involves not fine-tuning PLMs but combining the language modeling capability of pretrained models with the generative diversity of diffusion models (He et al., 2023), which significantly improves the capability of the generative encoder, i.e., MLM.

Since diffusion LMs can generate new sequences from masked original sequences, which matches the goal of retaining key information and rephrasing secondary information in generative data augmentation. Therefore, on top of diffusion LM, we propose DiffusionCLS, simultaneously considering label and domain consistency and generating pseudo samples by partially paraphrasing strong label-related tokens. Extensive experiments verify the effectiveness of our method and hopefully be extended to numerous NLP tasks.

## 3 Methodology

Sentiment classification models often overfit and lack generalization due to sample deficiency. To address this, we propose DiffusionCLS, consisting of Label-Aware Noise Schedule, Label-Aware Prompting, Conditional Sample Generation, and Noise-Resistant Training. A diffusion LM-based sample generator is integrated to generate new samples from the original dataset, enhancing TC model performance.

Figure 1 illustrates DiffusionCLS. The diffusion LM-based sample generator generates new samples for data augmentation, while the TC model is trained for the specific task. Label-Aware Prompting and Label-Aware Noise Schedule are crucial for training the sample generator, and Conditional Sample Generation and Noise-Resistant Training contribute to the training of the TC model.

## 3.1 Sample Generator

To generate usable samples for further TC model training, there are two crucial rules of success to satisfy, diversity and consistency. Therefore, we expect the generated samples to be as diverse as possible with consistency to the TC label and original domain simultaneously. However, higher diversity also leads to a higher difficulty in maintaining consistency.

As He et al. (2023) excavated the potential of combining diffusion models with LMs for sequence generation, we built the sample generator from the discrete diffusion model scratch. Precisely, we design the Label-Aware Noise Schedule for the diffusion LM, which helps the model to generate diverse and consistent samples. Additionally, we integrate

Label-Aware Prompting into the training regime, enabling the model to grasp label-specific knowledge, subsequently serving as the guiding condition for sample generation. These two modules help the generator to surpass the diversity-consistency challenge and excel in performance.

## 3.1.1 Label-Aware Noise Schedule

A proper algorithm of noise schedule could guide the diffusion LM to capture more accurate semantic relations. Moreover, the effectiveness of timeagnostic decoding has been demonstrated, indicating that incorporating implicit time information in the noise schedule process is effective (Ho et al., 2020; Nichol and Dhariwal, 2021; He et al., 2023). Since the generated samples are also expected to stay consistent with the TC label and the original domain, we proposed Label-Aware Noise Schedule.

The Label-Aware Noise Schedule begins by integrating a proxy model that has been fine-tuned for the TC task. This proxy model allows for the determination of the importance of each token in the TC process, quantified through attention scores between the [CLS] token and other tokens, which are derived from the last layer of proxy model and calculated as follows.

$$
w _ { i } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } s _ { i } ^ { h } ,\tag{1}
$$

where $s _ { i } ^ { h }$ represents the i-th token attention score in the h-th attention head, and $w _ { i }$ denotes the weight measuring the importance of the i-th token.

Motivated by He et al. (2023)’s DiffusionBERT, we incorporates absorbing state in the LM noise schedule. In our method, during the masking transition procedure, each token in the sequence remains unchanged or transitions to [MASK] with a certain probability. The transition probability of token i at step t can be denoted as:

$$
q _ { t } ^ { i } = 1 - \frac { t } { T } - \lambda \cdot S ( t ) \cdot w _ { i } ,\tag{2}
$$

$$
S ( t ) = \mathrm { s i n } \frac { t \pi } { T } ,\tag{3}
$$

where $q _ { t } ^ { i }$ represents the probability that a token is being masked, and T denotes the total step number. λ is introduced to control the impact of $w _ { i }$ , as a hyper-parameter.

By introducing strong label-related $w _ { i } .$ , the diffusion model is guided to recover the tokens with lower weight first, then recover the tokens that are strongly related to the classification task later.

![](images/4beb845fccfa8595b12ab9ee832b704bc39bb9ad691f46668ecb4c96d9befb17.jpg)  
Figure 2: The probability of a token remaining unmasked, with λ set to 0.5.

The probability of a token being masked is tied to its attention score relative to the [CLS] token, reflecting its contribution to the TC objective. Figure 2 shows that masking probabilities depend on the token’s label-related information. Label-Aware Noise Scheduling guides the model to recover the most label-related key tokens from those less crucial to the classification task.

## 3.1.2 Label-Aware Prompting

However, such a noise schedule still poses a challenge to the conditional generation process. The diversity-consistency trade-off becomes more intense when important tokens are masked. With fewer unmasked tokens provided, the model naturally has a higher possibility of generating tokens that would break the label consistency.

![](images/3c514021995b86c199b205568c961163ffe6d53f2d1c7924e054f044068897dd.jpg)  
Figure 3: Label-Aware Prompting, each masked sequence is concatenated with their corresponding label.

To address this challenge, we propose Label-Aware Prompting, a method that offers supplementary conditional information during both training and inference phrases. This additional information aids the model in generating samples that uphold label consistency.

As Figure 3 illustrated, following the masking of samples in the noise schedule process, the labels of these samples are concatenated with their respective masked sequences.

## 3.2 Text Classification Model

In this work, we adopt encoder-based PLM as our backbone model and finetuned them for the TC task. Though diffusion LM is strong enough to maintain consistency and diversity at the same time, the introduction of pseudo samples unavoidably introduced noise data to the training of the TC model. To mitigate such a problem, we design a contrastive learning-based noise-resistant training method, further improving the scalability of the proposed DiffusionCLS.

## 3.2.1 Reflective Conditional Sample Generation

We implement label prompting as a prior for the sample generator, akin to Label-Aware Prompting. Additionally, we introduce a novel reflective conditional sample generation module within the training loop of the TC model. This module dynamically generates masked sequences for the sample generator, integrating insights from label annotations and attention scores derived from the TC model simultaneously, calculating weights for each token with Eq.1.

However, generating pseudo samples from varying degrees of masking will result in various degrees of context replacement flexibility, thus impacting the consistency and diversity of pseudo samples. Essentially, providing a proper amount of conditional information will lead to plausible samples. Thus, we perform multiple experiments to search for the best condition, which will be further discussed in Section 4.5.

## 3.2.2 Noise-Resistant Training

The introduction of pseudo samples unavoidably introduced noise data to the training of the TC model. To mitigate such a problem, we design a contrastive learning-based noise-resistant training method, further improving the scalability of the proposed DiffusionCLS.

Figure 4 demonstrates the Noise-resistant Training. Specifically, besides including supervision signals from labels of original and generated samples, we also guide the model to enlarge the gap between samples with different labels.

Consider a dataset comprising m distinct categories $\textit { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { m } \}$ , we can obtain k samples from the original training set, and the corresponding subscript list is $I = \{ 1 , 2 , . . . , k -$

1, k . Essentially, a batch of sentences S = $\left\{ s _ { 1 } , s _ { 2 } , . . . , s _ { k - 1 } , s _ { k } \right\}$ , their corresponding label sequence $L = [ l _ { 1 } , l _ { 2 } , . . . , l _ { k - 1 } , l _ { k } ]$ with $l _ { i } \in C$ , and negative set for each sample $N _ { i } = \{ j \in I | l _ { j } \neq l _ { i } \}$ From this, we derive semantic representations $H = \left\{ h _ { 1 } , h _ { 2 } , . . . , h _ { k - 1 } , h _ { k } \right\}$ from the TC model. Furthermore, employing a sample generator yields B new samples for each original sample $s _ { i }$ , denoted as $G _ { i } = \{ g _ { 0 } ^ { s _ { i } } , g _ { 1 } ^ { s _ { i } } , . . . , g _ { B - 1 } ^ { s _ { i } } , g _ { B } ^ { s _ { i } } \}$ , where $g _ { 0 } ^ { s _ { i } } = s _ { i }$

![](images/c993d7995eed43472dc3d0f7831ddbe27596e7815517550b31091cf69cf64795.jpg)  
Figure 4: Noise-resistant contrastive learning. Cross points are generated samples while round dots denote original samples. Train-with-noise objective aiming at enlarging the gap between original samples with different labels.

Contrastive Loss. To avoid expanding the impact of noise samples, we calculate contrastive loss from only the original samples. With the aim to enlarge the gap between samples from different categories, the contrastive loss can be calculated as:

$$
L _ { c } = \frac { 1 } { K } \mathrm { l o g } \sum _ { i \in I } \sum _ { j \in N _ { i } } \exp ( \frac { \sin ( h _ { i } , h _ { j } ) } { \tau } ) ,\tag{4}
$$

where sim $( )$ denotes the consine similarity function and τ is a hyper-parameter as a scaling term.

Classification Loss. We also allows supervision signals directly affects the training of the TC model through the cross entropy loss, which can be denoted as:

$$
L _ { e } = - \frac { 1 } { K ( B + 1 ) } \sum _ { i \in I } \sum _ { b = 0 } ^ { B } \sum _ { c \in C } y _ { b , c } ^ { i } \mathrm { l o g } ( \hat { y } _ { b , c } ^ { i } ) ,\tag{5}
$$

where $y _ { b , c } ^ { i }$ is the label indicator, and $\hat { y } _ { b , c } ^ { i }$ is the predicted probability of b-th pseudo sample of the original sample i being of class $c .$

Training Objective. From two losses mentioned above, we formulated the overall training objective for the TC model, which can be denoted as:

$$
L = L _ { c } + L _ { e } .\tag{6}
$$

## 4 Experiments

## 4.1 Datasets and Baselines

To measure the efficiency of the propose DiffusionCLS, we utilize both domain-specific and domain-general datasets comprising samples in Chinese, English, Arabic, French, and Spanish. Namely, domain-specific SMP2020-EWECT<sup>2</sup>, India-COVID-X<sup>3</sup>, SenWave (Yang et al., 2020), and domain-general SST-2 (Maas et al., 2011). Additionally, to compare with the most cutting-edge low-resource TC methods, we utilize SST-2 dataset to evaluate our method in the few-shot setting. Dataset statistic and descriptions are demonstrated in Appendix A.

To thoroughly explore and validate the capabilities of DiffusionCLS, we compare our method with a range of data augmentation techniques, from classic approaches to the latest advancements for lowresource TC. Specifically, we take Resample, Back Translation (Shleifer, 2019), Easy Data Augmentation (EDA) (Wei and Zou, 2019), SFT GPT-2 referenced to LAMBADA (Anaby-Tavor et al., 2020), AEDA (Karimi et al., 2021), and GENIUS (Guo et al., 2022) as our baselines. Also, we compare our method in the few-shot setting with a couple of cutting-edge methods, namely, SSMBA (Ng et al., 2020), ALP (Kim et al., 2022), and SE (Zheng et al., 2023). More details of our baselines are demonstrated in Appendix B.

## 4.2 Experiment Setup

We set up two experimental modes, entire data mode and partial data mode, to reveal the effectiveness of our method in different scenarios. Since severe imbalanced distribution challenges existed, we take macro-F1 and accuracy as our major evaluation metrics.

Also, we conduct 5-shot and 10-shot experiments on SST-2 to investigate the performance of DiffusionCLS in extreme low-resource conditions. For evaluation, we use accuracy as the metric and report the average results over three random seeds to minimize the effects of stochasticity.

Additionally, we setup comparisons between variant augmentation policies, namely, generate new samples until the dataset distribution is balanced, and generate n pseudo samples for each sample (n-samples-each), which denoted as B/D and G/E in Table 2, and n=4 in our experiments.

Other related implementation details are described in Appendix A.

## 4.3 Results and Analysis

The results of entire-data-setting experiments on datasets SMP2020-EWECT and India-COVID-X are mainly demonstrated in Table 2, which we compare DiffusionCLS with other strong DA baselines. For experiments with partial-data and few-shot settings, results are majorly showed in Figure 5 and Table 9.

Results under Entire Data Mode. As shown in Table 2, in general, the proposed DiffusionCLS outperforms most DA methods on domain-specific datasets SMP2020-EWECT and India-COVID-X, especially under G/E policy. Notably, the DiffusionCLS positively impacts the TC model across all policies and datasets, which most baselines fail.

Our method excels in dealing with the challenge of uneven datasets. Under severe uneven distribution and domain-specific scenarios, i.e., the dataset SMP2020-EWECT, most DA baselines fail to impact the classification model positively except DiffusionCLS, which achieves the best performance. Also, our method achieves competitive performance under data-sparse and domain-specific scenarios, i.e., in the dataset India-COVID-X, most DA methods bring improvement to the classification model, and our DiffusionCLS ranked second.

Rule-based DA methods such as EDA, rather lack diversity bringing overfit problems or solely relying on out-domain knowledge therefore breaking consistency and impacting the task model negatively. For model-based methods, though most methods significantly increase the diversity of the generated samples, they rather generate samples solely depending on pretraining knowledge and incontext-learning techniques or generate samples only conditioned on the label itself, posing a challenge of maintaining consistency.

Results under Partial Data Mode and Fewshot Settings. As shown in Figure 5 and Table 9 in Appendix C, the proposed DiffusionCLS method consistently improves the classification model. Notably, DiffusionCLS matches the PLM baseline performance on the Arabic SenWave dataset using only 50% of the data samples.

We also compare DiffusionCLS with the most cutting-edge few-shot methods on SST-2 dataset under 5-shot and 10-shot setting, the results are shown in Table 3. Though our method fails to surpass all few-shot baselines, it still achieves competitive performance with those designed for the few-shot task.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Policy</td><td colspan="4">SMP2020-EWECT</td><td colspan="4">India-COVID-X</td></tr><tr><td>Macro-F</td><td>Acc</td><td>∆F</td><td>∆Acc</td><td>Macro-F</td><td>Acc</td><td>∆F</td><td>∆Acc</td></tr><tr><td>Raw PLM</td><td>N/A</td><td>65.87%</td><td>79.17%</td><td></td><td></td><td>70.99%</td><td>70.63%</td><td></td><td></td></tr><tr><td>+ Resample</td><td>B/D</td><td>64.84%</td><td>78.17%</td><td>-1.03%</td><td>-1.00%</td><td>72.74%</td><td>72.57%</td><td>1.75%</td><td>1.94%</td></tr><tr><td>+ BT (2019)</td><td>B/D</td><td>64.03%</td><td>77.93%</td><td>-1.84%</td><td>-1.24%</td><td>72.93%</td><td>72.79%</td><td>1.94%</td><td>2.16%</td></tr><tr><td>+ EDA (2019)</td><td>B/D</td><td>65.88%</td><td>78.87%</td><td>0.01%</td><td>-0.30%</td><td>66.83%</td><td>66.41%</td><td>-4.16%</td><td>-4.22%</td></tr><tr><td>+ AEDA (2021)</td><td>B/D</td><td>66.58%</td><td>79.50%</td><td>0.71%</td><td>0.33%</td><td>72.90%</td><td>72.89%</td><td>1.91%</td><td>2.26%</td></tr><tr><td>+ GENIUS (2022)</td><td>B/D</td><td>64.27%</td><td>78.23%</td><td>-1.60%</td><td>-0.94%</td><td>72.84%</td><td>72.46%</td><td>1.85%</td><td>1.83%</td></tr><tr><td>+ DiffusionCLS (ours)</td><td>B/D</td><td>66.47%</td><td>79.43%</td><td>0.60%</td><td>0.26%</td><td>72.80%</td><td>72.57%</td><td>1.81%</td><td>1.94%</td></tr><tr><td>+ BT (2019)</td><td>G/E</td><td>65.15%</td><td>77.93%</td><td>-0.72%</td><td>-1.24%</td><td>74.40%</td><td>74.30%</td><td>3.41%</td><td>3.67%</td></tr><tr><td>+ EDA (2019)</td><td>G/E</td><td>50.12%</td><td>71.87%</td><td>-15.75%</td><td>-7.30%</td><td>74.15%</td><td>73.87%</td><td>3.16%</td><td>3.24%</td></tr><tr><td>+ GPT-2 (2020)</td><td>G/E</td><td>65.06%</td><td>77.80%</td><td>-0.81%</td><td>-1.37%</td><td>69.55%</td><td>69.58%</td><td>-1.44%</td><td>-1.05%</td></tr><tr><td>+ AEDA (2021)</td><td>G/E</td><td>65.81%</td><td>78.93%</td><td>-0.06%</td><td>-0.24%</td><td>75.49%</td><td>75.27%</td><td>4.50%</td><td>4.64%</td></tr><tr><td>+ GENIUS (2022)</td><td>G/E</td><td>64.30%</td><td>78.07%</td><td>-1.57%</td><td>-1.10%</td><td>74.28%</td><td>74.08%</td><td>3.29%</td><td>3.45%</td></tr><tr><td>+ DiffusionCLS (ours)</td><td>G/E</td><td>67.98%</td><td>80.23%</td><td>2.11%</td><td>1.06%</td><td>74.65%</td><td>74.41%</td><td>3.66%</td><td>3.78%</td></tr></table>

Table 2: Experiment results on SMP2020-EWECT and India-COVID-X datasets, with N/A indicating no augmentation, B/D for balancing pseudo samples, and G/E for the n-samples-each policy. We adopt bert-base as the English PLM and wwm-roberta as the Chinese PLM. + denotes the model is trained with the corresponding augmentation method. $\Delta \mathsf { A c c }$ and ∆F represent performance variance between training with augmentation and without.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">#Shot</td><td colspan="10">Data Augmentaion Methods</td></tr><tr><td>N/A</td><td>+EDA† (2019)</td><td> $+ \mathbf { B } \mathbf { T } ^ { \dagger }$  (2019)</td><td>+SSMBA† (2020)</td><td>+ALP† (2022)</td><td>+SE† (2023)</td><td>+GPT-2 (2020)</td><td>+mixup (2017)</td><td>+AWD (2023a)</td><td>+DiffusionCLS</td></tr><tr><td rowspan="2">SST-2</td><td>5</td><td>54.38</td><td>56.22</td><td>55.77</td><td>56.34</td><td>63.40</td><td>-</td><td>52.18</td><td>61.81</td><td>58.86</td><td>65.30</td></tr><tr><td>10</td><td>61.82</td><td>53.96</td><td>62.05</td><td>59.05</td><td>69.72</td><td>57.56</td><td>54.17</td><td>61.55</td><td>64.62</td><td>68.29</td></tr></table>

Table 3: Performances of TC models on dataset SST-2 under the few-shot setting. denotes that results are from previous research. All of our results are collected from 5 runs with different seeds.

![](images/28bff689ea103a2244f551f9ff95372d3d79d4c7c6cf4d8878ec8a8ff234ccf1.jpg)  
(a) Arabic

![](images/21bc772fa6c0800eb9384b40b19b5229e29c505eaadea70da460dd139df8afe6.jpg)  
(b) Spanish

![](images/64f5a45c18efc11568bc30fdf1e8d1ab045652e9d755e1a923b9e0a81e66993a.jpg)  
(c) French  
Figure 5: Performances of SC models on dataset SenWave under the partial data setting. Red lines denote the raw PLM results and blue lines represent models trained with DiffusionCLS.

Since DiffusionCLS requires diffusion training to adapt to domain-specific tasks, extreme sample insufficiency may introduce noise, negatively impacting the model. However, our method positively impacts the TC model in most low-resource cases by effectively utilizing pre-trained and in-domain knowledge, from severe imbalanced label distribution to severe sample insufficiency.

## 4.4 Ablation Study

To validate the effectiveness of modules in the proposed DiffusionCLS, we conduct ablation studies to study the impacts of each module. Table 4 presents the results of the ablation experiments. In each row of the experiment results, one of the modules in DiffusionCLS has been removed for discussion, except D.A., which removes all modules related to the generator and only applies noiseresistance training.

Overall, all modules in the proposed Diffusion-CLS works positively to the TC model, compared with the pure PLM model, the application of DiffusionCLS leads to 2.11% and 3.66% rises in F1 values on dataset SMP2020-EWECT and India-COVID-X respectively.

The results of ablation studies further validate that the Label-Aware Prompting effectively improves the quality of pseudo samples. Also, the Noise-Resistant Training reduces the impact of noise pseudo samples.

<table><tr><td>Dataset</td><td>Methods</td><td>Macro-F</td><td>Acc</td></tr><tr><td>SMP2020- EWECT</td><td>DiffusionCLS -w/o D.A. -w/o L.A.P. -w/o N.R.T.</td><td>0.6798 0.6637 0.6671 0.6695</td><td>0.8023 0.7957 0.7930 0.7963</td></tr><tr><td>India- COVID-X</td><td>DiffusionCLS -w/o D.A. -w/o L.A.P. -w/o N.R.T.</td><td>0.7465 0.7206 0.7298 0.7361</td><td>0.7441 0.7181 0.7268 0.7354</td></tr></table>

Table 4: Experiment results of ablation study, where -w/o is the abbreviation of without. D.A., L.A.P., and N.R.T. correspond to data augmentation, label-aware prompting, and noise-resistant training. D.A. removes the generator.

## 4.5 Discussions and Visualizations

Generating pseudo samples from more masked tokens provides more flexibility for generation and tends to result in more diverse samples, however, it will enlarge the possibility of breaking the consistency since less information is provided.

To analyze the optimal amount of masks for generating new pseudo samples, we conduct experiments on the India-COVID-X dataset. During conditional sample generation, we gather masked sequences from 32 noise-adding steps, group them into sets of eight, and evaluate how varying masking levels impact the model’s performance.

As shown in Figure 6, our observations indicate a unimodal trend. The model’s performance improves with increased masking, peaks at the 4th group, and then declines with further masking. This reflects the diversity-consistency trade-off, more masked tokens create more diverse samples, but overly diverse samples may be inconsistent with original labels or domain.

To explore the relationship between generated pseudo samples and original samples, we conduct 2D t-SNE visualization. Figure 7 shows that as masking increases, pseudo samples gradually diverge from the original samples, indicating increased diversity.

![](images/684eb48c8f0ca56d79d90be454f1836bf8df8d71468b0643c6714de35c2f97c0.jpg)  
Figure 6: Performances of models with pseudo samples generated from different groups of masked sequences, in which step one will result in original sequences and step 32 will result in generating pseudo samples from fully masked sequences.

![](images/026750052f9f2fd774a3123b178f845bf8c1ccddd8b50acfa91fcca344170aab.jpg)  
Figure 7: 2D t-SNE visualization on the India-COVID-X dataset.

## 5 Conclusion

In this work, we propose DiffusionCLS, a novel approach tackling SC challenges under low-resource conditions, especially in domain-specific and uneven distribution scenarios. Utilizing a diffusion LM, DiffusionCLS captures in-domain knowledge to generate high-quality pseudo samples maintaining both diversity and consistency. This method surpasses various kinds of data augmentation techniques. Our experiments demonstrate that DiffusionCLS significantly enhances SC performance across various domain-specific and multilingual datasets. Ablation and visualization studies further validate our approach, emphasizing the importance of balancing diversity and consistency in pseudo samples. DiffusionCLS presents a robust solution for data augmentation in low-resource NLP applications, paving a promising path for future research.

## Limitations

Like most model-based data augmentation methods, the performance of data generators is also limited in extreme low-resource scenarios. This limitation persists because the model still necessitates training on the training data, even with the potential expansion of the dataset through the inclusion of unlabeled data, data deficiency impacts the data generator negatively.

## Acknowledgments

This work was supported by the National Social Science Fund of China (No. 22BTQ045).

## References

Ateret Anaby-Tavor, Boaz Carmeli, Esther Goldbraich, Amir Kantor, George Kour, Segev Shlomov, Naama Tepper, and Naama Zwerdling. 2020. Do not have enough data? deep learning to the rescue! In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 7383–7390.

Junfan Chen, Richong Zhang, Zheyan Luo, Chunming Hu, and Yongyi Mao. 2023a. Adversarial word dilution as text data augmentation in low-resource regime. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 12626–12634.

Xilun Chen, Yu Sun, Ben Athiwaratkun, Claire Cardie, and Kilian Weinberger. 2018. Adversarial deep averaging networks for cross-lingual sentiment classification. Transactions ofthe Associationfor Computational Linguistics, 6:557–570.

Zhuowei Chen, Yujia Tian, Lianxi Wang, and Shengyi Jiang. 2023b. A distantly-supervised relation extraction method based on selective gate and noise correction. In China National Conference on Chinese Computational Linguistics, pages 159–174. Springer.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Steven Y Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, and Eduard Hovy. 2021. A survey of data augmentation approaches for nlp. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 968–988.

Joseph Gatto and Sarah M Preum. 2023. Not enough labeled data? just add semantics: A data-efficient method for inferring online health texts. arXiv preprint arXiv:2309.09877.

Mohamed Goudjil, Mouloud Koudil, Mouldi Bedda, and Noureddine Ghoggali. 2018. A novel active

learning method using svm for text classification. International Journal of Automation and Computing, 15:290–298.

Biyang Guo, Yeyun Gong, Yelong Shen, Songqiao Han, Hailiang Huang, Nan Duan, and Weizhu Chen. 2022. Genius: Sketch-based language model pre-training via extreme and selective masking for text generation and augmentation. arXiv preprint arXiv:2211.10330.

Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. 2023. DiffusionBERT: Improving generative masked language models with diffusion models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4521–4534, Toronto, Canada. Association for Computational Linguistics.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851.

Akbar Karimi, Leonardo Rossi, and Andrea Prati. 2021. AEDA: An easier data augmentation technique for text classification. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2748–2754, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Hazel H Kim, Daecheol Woo, Seong Joon Oh, Jeong-Won Cha, and Yo-Sub Han. 2022. Alp: Data augmentation using lexicalized pcfgs for few-shot text classification. In Proceedings of the aaai conference on artificial intelligence, volume 36, pages 10894– 10902.

Varun Kumar, Ashutosh Choudhary, and Eunah Cho. 2020. Data augmentation using pre-trained transformer models. In Proceedings ofthe 2nd Workshop on Life-long Learningfor Spoken Language Systems, pages 18–26.

S Deepa Lakshmi and T Velmurugan. 2023. Classification of disaster tweets using natural language processing pipeline. Acta Scientific COMPUTER SCIENCES Volume, 5(3).

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Associationfor Computational Linguistics: Human

Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

Alvi Ahmmed Nabil, Dola Das, Md Shahidul Salim, Shamsul Arifeen, and HM Abdul Fattah. 2023. Bangla emergency post classification on social media using transformer based bert models. In 2023 6th International Conference on Electrical Information and Communication Technology (EICT), pages 1–6. IEEE.

Nathan Ng, Kyunghyun Cho, and Marzyeh Ghassemi. 2020. SSMBA: Self-supervised manifold based data augmentation for improving out-of-domain robustness. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1268–1283, Online. Association for Computational Linguistics.

Alexander Quinn Nichol and Prafulla Dhariwal. 2021. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR.

Kelechi Ogueji, Yuxin Zhu, and Jimmy Lin. 2021. Small data? no problem! exploring the viability of pretrained multilingual language models for lowresourced languages. In Proceedings ofthe 1st Workshop on Multilingual Representation Learning, pages 116–126.

Parth Patwa, Simone Filice, Zhiyu Chen, Giuseppe Castellucci, Oleg Rokhlenko, and Shervin Malmasi. 2024. Enhancing low-resource LLMs classification with PEFT and synthetic data. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 6017–6023, Torino, Italia. ELRA and ICCL.

Marina Sedinkina, Martin Schmitt, and Hinrich Schütze. 2022. Domain adaptation for sparse-data settings: What do we gain by not using bert? arXiv e-prints, pages arXiv–2203.

Sam Shleifer. 2019. Low resource text classification with ulmfit and backtranslation. arXiv preprint arXiv:1903.09244.

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, and Haifeng Wang. 2020. Ernie 2.0: A continual pre-training framework for language understanding. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 8968–8975.

Ming Tan, Yang Yu, Haoyu Wang, Dakuo Wang, Saloni Potdar, Shiyu Chang, and Mo Yu. 2019. Out-ofdomain detection for low-resource text classification tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3566–3572.

Lianxi Wang, Yujia Tian, and Zhuowei Chen. 2024. Enhancing Hindi feature representation through fusion of dual-script word embeddings. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 5966–5976, Torino, Italia. ELRA and ICCL.

Jason Wei and Kai Zou. 2019. EDA: Easy data augmentation techniques for boosting performance on text classification tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6382–6388, Hong Kong, China. Association for Computational Linguistics.

Zhihao Wen and Yuan Fang. 2023. Augmenting lowresource text classification with graph-grounded pretraining and prompting. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 506–516.

Xing Wu, Shangwen Lv, Liangjun Zang, Jizhong Han, and Songlin Hu. 2019. Conditional bert contextual augmentation. In Computational Science–ICCS 2019: 19th International Conference, Faro, Portugal, June 12–14, 2019, Proceedings, Part IV 19, pages 84–95. Springer.

Qiang Yang, Hind Alamro, Somayah Albaradei, Adil Salhi, Xiaoting Lv, Changsheng Ma, Manal Alshehri, Inji Jaber, Faroug Tifratene, Wei Wang, Takashi Gojobori, Carlos M. Duarte, Xin Gao, and Xiangliang Zhang. 2020. Senwave: Monitoring the global sentiments under the covid-19 pandemic. Preprint, arXiv:2006.10842.

Jianfei Yu, Qiankun Zhao, and Rui Xia. 2023. Crossdomain data augmentation with domain-adaptive language modeling for aspect-based sentiment analysis. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1456–1470, Toronto, Canada. Association for Computational Linguistics.

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2017. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.

Haoqi Zheng, Qihuang Zhong, Liang Ding, Zhiliang Tian, Xin Niu, Changjian Wang, Dongsheng Li, and Dacheng Tao. 2023. Self-evolution learning for mixup: Enhance data augmentation on few-shot text classification tasks. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 8964–8974, Singapore. Association for Computational Linguistics.

## A Experiment Setup, Implementation, and Dataset Statistics

## A.1 Experiment Setup

The low-resource challenge in TC includes problems like insufficient annotated samples, domainspecific adaptation problems, and imbalanced distribution. To measure the capability of the proposed DiffusionCLS to mitigate these problems, we conduct experiments on three domain-specific datasets with respect to the problems mentioned above, as shown in Table 7.

## A.2 Implementation

For implementation, we take bert-base-uncased<sup>4</sup> and chinese-roberta-wwm<sup>5</sup> from the huggingface platform respectively for English and Chinese dataset training. Also, hyper-parameters settings of our work are demonstrated in Table 5 and Table 6.

## A.3 Datasets

For our experiments, we utilize multilingual datasets, both domain-specific and domain-general, to evaluate the proposed DiffusionCLS. Data statistics and their challenges are demonstrated in Table 7 and Table 8.

• SMP2020-EWECT<sup>6</sup>. This Chinese dataset includes 8,606 pandemic-related posts, categorized into neutral, happy, angry, sad, fear, and surprise, with highly imbalanced label distribution.

• India-COVID-X<sup>7</sup>. This dataset contains cleaned English tweets from India X platform on topics such as coronavirus, COVID-19, and lockdown. The tweets have been labeled into four sentiment categories with relatively balanced label distribution.

• SenWave(Yang et al., 2020). This dataset includes about 5,000 English tweets and approximately 3,000 Arabic tweets in the specific domain of the pandemic and lockdown, which are annotated with sentiment labels. Englishtranslated French and Spanish annotated samples are also included. We extract all single label samples for experiments.

• SST-2(Maas et al., 2011). It includes 11,855 movie review sentences parsed by the Stanford parser, with 215,154 unique phrases annotated by three human judges.

## B Baselines

• Non-Generative Methods

– SSMBA(Ng et al., 2020). Uses a corruption and reconstruction function to augment data by filling in masked portions.

– ALP(Kim et al., 2022). Employs Lexicalized Probabilistic Context-Free Grammars to generate syntactically diverse augmented samples.

– SE(Zheng et al., 2023). Utilizes a selfevolution learning-based mixup technique to create adaptive pseudo samples for training.

– AEDA(Karimi et al., 2021). Randomly insert punctuations into the original sentences to produce new samples.

## • Generative Methods

– GPT-2(Anaby-Tavor et al., 2020). Finetunes GPT-2 with prompt-based SFT, prompting labels to generate pseudo samples.

– GENIUS(Guo et al., 2022). A conditional text generation model using sketches as input, which can fill in the missing context for a given sketch.

## • Representation Augmentation Methods

– mixup(Zhang et al., 2017). Mixup is a representational DA technique that creates new training samples by linearly interpolating between pairs of examples and their labels.

– AWD(Chen et al., 2023a). AWD generates challenging positive examples for low-resource text classification by diluting strong positive word embeddings with unknown-word embeddings.

## C Experiment Results with Partial Data Mode

The proposed DiffusionCLS method consistently enhances the classification model, achieving higher accuracy with only 50% training data than the raw PLM on dataset SMP2020-EWECT. Detailed results are shown in Table 9.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Epoch of Proxy Model</td><td>15</td></tr><tr><td># Diffusion Steps</td><td>32</td></tr><tr><td>Index of Diffusion Group</td><td>4</td></tr><tr><td># Aug. Samples</td><td>4</td></tr><tr><td>Learning Rate</td><td>4e-06</td></tr><tr><td>Weight Decay</td><td>0.01</td></tr></table>

Table 5: Settings of hyperparameters, all values are identical across all datasets.

<table><tr><td>Dataset</td><td>PLM Name</td><td>Epoch of DiffusionLM</td><td>Batch Size</td></tr><tr><td>SMP2020-EWECT</td><td>chinese-roberta-wwm-ext</td><td>1</td><td>60</td></tr><tr><td>India-COVID-X</td><td>bert-base-uncased</td><td>1</td><td>40</td></tr><tr><td>SenWave-Arabic</td><td>CAMeL-Lab/bert-base-arabic-camelbert-ca</td><td>2</td><td>60</td></tr><tr><td>SenWave-France</td><td>dbmdz/bert-base-french-europeana-cased</td><td>2</td><td>60</td></tr><tr><td>SenWave-Spanish</td><td>dccuchile/bert-base-spanish-wwm-uncased</td><td>2</td><td>60</td></tr><tr><td>SST-2</td><td>bert-base-uncased</td><td>2</td><td>20</td></tr></table>

Table 6: Settings of hyperparameters across datasets, all PLMs are directly loaded from the Huggingface platform.

<table><tr><td>Challenge</td><td>SMP2020-EWECT</td><td>India-COVID-X</td><td>SenWave</td><td>SST-2</td></tr><tr><td rowspan="3">Insufficient Samples Domain-Specific</td><td>√</td><td>√</td><td>√</td><td>×</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Imbalanced Distribution Multilingual</td><td>X</td><td>X</td><td>√</td><td>X</td></tr></table>

Table 7: Low-resource challenges of datasets.

<table><tr><td>Dataset</td><td>Language</td><td>#Train</td><td>#Test</td><td>#Label</td><td>Avg. Length</td><td>S/D</td></tr><tr><td>India-COVID-X</td><td>English</td><td>2164</td><td>926</td><td>4</td><td>25.23</td><td>0.0127</td></tr><tr><td>SMP2020-EWECT</td><td>Chinese</td><td>8606</td><td>3000</td><td>6</td><td>54.44</td><td>0.1634</td></tr><tr><td rowspan="3">SenWave</td><td>Arabic</td><td>2210</td><td>553</td><td>6</td><td>26.07</td><td>0.1069</td></tr><tr><td>Spanish</td><td>4116</td><td>1029</td><td>6</td><td>19.25</td><td>0.1284</td></tr><tr><td>French</td><td>4116</td><td>1029</td><td>6</td><td>18.90</td><td>0.1284</td></tr><tr><td>SST-2</td><td>English</td><td>67000</td><td>18000</td><td>2</td><td>10.41</td><td>0.0578</td></tr></table>

Table 8: Data statistics. S/D represents the standard deviation of label distributions.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Percentage</td><td colspan="2">DiffusionCLS</td><td colspan="2">PLM</td><td rowspan="2"> $\Delta F$ </td><td rowspan="2">∆Acc</td></tr><tr><td>Macro-F</td><td>Acc</td><td>Macro-F</td><td>Acc</td></tr><tr><td rowspan="5">SMP2020-EWECT</td><td>0.05</td><td>54.93%</td><td>74.20%</td><td>54.88%</td><td>73.87%</td><td>0.05%</td><td>0.33%</td></tr><tr><td>0.20</td><td>64.35%</td><td>78.23%</td><td>63.60%</td><td>77.70%</td><td>0.75%</td><td>0.53%</td></tr><tr><td>0.35</td><td>64.49%</td><td>78.23%</td><td>63.65%</td><td>78.00%</td><td>0.84%</td><td>0.23%</td></tr><tr><td>0.50</td><td>65.09%</td><td>78.90%</td><td>65.09%</td><td>78.03%</td><td>0.01%</td><td>0.87%</td></tr><tr><td>1.00</td><td>67.98%</td><td>80.23%</td><td>66.14%</td><td>78.77%</td><td>1.85%</td><td>1.47%</td></tr><tr><td rowspan="5">India-COVID-X</td><td>0.05</td><td>46.33%</td><td>47.73%</td><td>44.89%</td><td>48.16%</td><td>1.45%</td><td>-0.43%</td></tr><tr><td>0.20</td><td>63.44%</td><td>63.17%</td><td>61.60%</td><td>61.34%</td><td>1.84%</td><td>1.84%</td></tr><tr><td>0.35</td><td>66.04%</td><td>65.98%</td><td>65.16%</td><td>65.12%</td><td>0.87%</td><td>0.86%</td></tr><tr><td>0.50</td><td>70.17%</td><td>69.98%</td><td>69.32%</td><td>69.01%</td><td>0.84%</td><td>0.97%</td></tr><tr><td>1.00</td><td>74.65%</td><td>74.41%</td><td>70.99%</td><td>70.63%</td><td>3.66%</td><td>3.78%</td></tr></table>

Table 9: Experiment results on dataset SMP2020-EWECT and India-COVID-X with partial data mode, with the percentage column indicating how much data is used in the training process. ∆Acc and ∆F represent the performance variance between training with a data augmentation method and their corresponding baselines, i.e., without data augmentation methods.