# Collaborative Performance Prediction for Large Language Models

Qiyuan Zhang City University of Hong Kong qzhang732-c@my.cityu.edu.hk

Xue Liu McGill University xueliu@cs.mcgill.ca

Comprehensively understanding and accurately predicting the performance of large language models across diverse downstream tasks has emerged as a pivotal challenge in NLP research. The pioneering scaling law on downstream works (Hu et al., 2024; Isik et al., 2024) demonstrated intrinsic similarities within model families and utilized such similarities for performance prediction. However, they tend to overlook the similarities between model families and only consider design factors listed in the original scaling law. To overcome these limitations, we introduce a novel framework, Collaborative Performance Prediction (CPP), which significantly enhances prediction accuracy by leveraging the historical performance of various models on downstream tasks and other design factors for both model and task. We also collect a collaborative data sourced from online platforms containing both historical performance and additional design factors. With the support of the collaborative data, CPP not only surpasses traditional scaling laws in predicting the performance of scaled LLMs but also facilitates a detailed analysis of factor importance, an area previously overlooked. Our code is available here<sup>1</sup>.

## Abstract

## 1 Introduction

Large Language Models (LLMs) (Brown et al., 2020; Ouyang et al., 2022) have emerged as one of the most important AI research powered by largescale parameters, high computational resources, and massive training data. With the substantial increase in model sizes, the evaluation cost of LLMs’ performance becomes even more significant. For example, testing a single LLM on certain benchmarks often requires \$10K+ and 4K+ GPU hours (Liang et al., 2023). Therefore, understanding the behaviors and predicting the capabilities of LLMs across scales under various tasks

Fuyuan Lyu∗ McGill University & MILA fuyuan.lyu@mail.mcgill.ca

Chen Ma∗ City University of Hong Kong chenma@cityu.edu.hk

becomes a vital question (Ganguli et al., 2022a; Owen, 2024; Finnveden, 2020; Hu et al., 2024) for both researchers and engineers.

Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022; Hernandez et al., 2022; Gordon et al., 2021; Bahri et al., 2024; Muennighoff et al., 2023) have been powerful tools for predicting the capabilities of LLMs. It indicates a power-law correlation between the model performance and design factors such as computational measure (FLOPs) utilized during training. Although the scaling law was originally proposed as a strong intuitive guide for designing LLM, researchers (Hu et al., 2024; Ruan et al., 2024; Isik et al., 2024) have extended its utility into predicting model performances on various metrics, such as BLEU in Machine Translation, and different tasks. These works can accurately predict model performances by utilizing the similarity within each model family, e.g., models within each family are usually trained on the same dataset. However, there are several issues rooted in their methods: the performance prediction 1) requires transparent design factors that consume substantial training resources to fit the curve, 2) is only tailored to a certain model family and a specific task metric, and 3) neglects the connections among different models and tasks.

The aforementioned limitations motivate us to design more effective methods for predicting the performance of LLMs on downstream tasks. Two observations sparked our attention. Firstly, A strong similarity exists between model families, e.g.LLama-family and GPT family. Models from different families behave similarly in prediction distribution (Shrivastava et al., 2023) and emergent phenomenon (Wei et al., 2022). Secondly, with the emerging LLM models and the increasingly diverse tasks, the cost of enumerating and benchmarking models with tasks increases exponentially. Therefore, we aim to utilize the similarities across model families in order to collaboratively predict <sup>�</sup>the model performance over diverse tasks in an accurate yet efficient way.

![](images/ab434fea97cd678cca8b2065f6c0ce77f5186d9acc42296561dc26ceabe81c5d.jpg)  
Figure 1: Framework for Collaborative Performance Prediction of Large Language Models. This schematic delineates two principal components: (1) Collaborative Data, which encompasses a score matrix illustrating the �<sub>�</sub>1performance of various LLMs across downstream tasks, along with external descriptive factors of both models …and tasks; (2) Collaborative Prediction Method, given the model and task IDs to leverage this collaborative data, enabling accurate score prediction.

To incorporate the aforementioned intuitions, we propose a new scheme, Collaborative Performance Prediction (CPP), to efficiently predict the performance of LLMs on evaluation tasks. This scheme learns the latent representations of LLMs and tasks, which captures the intrinsic similarity among different models and tasks. The interaction (e.g., inner product) between the latent representations of LLMs and tasks can be utilized to predict the performance of LLMs on certain tasks. To fulfil the proposed scheme, we collect the LLM performance data from academic papers, technical reports, and open leaderboards covering 72 models and 29 tasks. To summarize, our scheme has several advantages:

• Low Training Cost: Compared with methods (Hu et al., 2024) that extend scaling law to various downstream tasks, no pre-training or finetuning of LLM is required in our scheme.

• Prediction over proprietary model: Unlike previous methods (Ruan et al., 2024), our scheme supports prediction over proprietary models without knowing key design factors, such as computational measures.

• Prediction from small to large: By utilizing cross-family information, our scheme can accurately estimate model performance, e.g., emergent ability, of large models on downstream tasks given the information from small models.

• Beyond Scaling Laws: Our scheme is more general and can incorporate diverse factors, such as task description factors.

• Factor-level Interpretability: Our scheme can provide interpretability by analyzing the factors importance of LLMs.

Under our scheme, multiple customized prediction methods (e.g., COLLABORATIVE FITER-ING (Koren et al., 2022)) can be incorporated to predict the performance of LLMs, further validating the feasibility and generality. Our method enables more diverse factors as input, ranging from traditional LLM design factors to task design factors, e.g., targeted ability and few-shot setting.

Upon extensive experimentation within the openreleased core leaderboard of HELM (Liang et al., 2023) and our collected historical matrix, our predictive performance demonstrated exceptionally well. Specifically, even without any input of model factors or task factors: in HELM, we use 50% of the scores to predict the other 50%, the predictive ranking (derived from predicted scores) achieves Accuracy =10%, and MAE@2 =39%; in our collected matrix (characterized by a 44% sparsity level) achieves an Accuracy =45%, and the MAE@2 =84%. Notably, the accuracy of our prediction from small to large LMs significantly exceeded that predicted by scaling laws. Using an analysis method similar to SHAPLEY-VALUES (Lundberg and Lee, 2017; Shapley, 1952), we elucidate the importance of different factors, which surprisingly does not fully align with scaling law (Kaplan et al., 2020). Therefore, our method is undoubtedly more versatile.

## 2 Related Work

## 2.1 Downstream Scaling Law and Performance Predictability of LLM

Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022; Hernandez et al., 2022; Bahri et al., 2024; Muennighoff et al., 2023) for LLMs have increas ingly become a focal point in understanding and guiding critical design decisions, such as model size and the characteristics and volume of pre training data. Traditionally, most research in this area has concentrated on how measures like cross entropy loss or perplexity scale. Subsequent studies have extended these efforts to the scaling be havior on translation (Isik et al., 2024; Ghorbani et al., 2021; Zhuocheng et al., 2023) and other downstream tasks modeling (Caballero et al., 2023; Henighan et al., 2020). The high predictability in LLMs capability has directly spurred extensive research work (see Survey Anwar et al. (2024)) exploring whether LLMs can demonstrate predictability on downstream tasks, which are considered highly unpredictable in traditional ML knowl edge (Ganguli et al., 2022a). Particularly, the “emergence” phenomenon (Suzgun et al., 2022; Wei et al., 2022) has challenged predictability, where models suddenly exhibit striking capabilities at specific training reources. Recent studies (Schaeffer et al., 2023) have made remarkable achievements in breaking the discontinuities in perfor mance brought about by emergence, and Ganguli et al. (2022a); Owen (2024); Finnveden (2020) demonstrated the predictability on downstream tasks, for instance, Hu et al. (2024) directly fits a curve of training resources and downstream task performance by repeatedly pretraining a specific model. Furthermore, Arora and Goyal (2023) pre dicted the performance through decomposing the complex capabilities of LMs to some base skills.

Given that predictability has now been established, we reassess the underlying premises that enable this predictability: the prevailing similarities across multiple models and various downstream tasks (Liu et al., 2023; Perlitz et al., 2024; Polo et al., 2024; Torregrossa et al., 2020; Ilic´, 2023). Based on this, we step beyond the limitations defined by scaling laws and propose a new methodology to predict the performance of LLMs on various downstream tasks.

## 2.2 Collaborative Filtering

Collaborative filtering (CF) (Koren et al., 2022) is a widely used technique in recommendation systems that predicts users’ preferences by collecting the historical preferences of many other users. The underlying assumption of $\mathrm { C F }$ is that similar users will share similar preferences on similar items. A seminal method in CF is matrix factorization (Koren et al., 2009) (MF). It reduces the dimensionality of the user-item matrix by learning the latent factors associated with users and items, respectively. This approach helps handle sparse data and improves scalability. The factorization of the user-item matrix R can be represented as

$$
\mathbf { R } \approx \mathbf { P } ^ { \top } \cdot \mathbf { Q } ,\tag{1}
$$

where each column vector in $\mathbf { P }$ and $\mathbf { Q }$ represents a specific user or item, respectively, with hidden dimension $d .$ The latent representations of users and items capture the user preferences and item properties in the latent space, and the inner product can be utilized to predict the interaction between users and items. To optimize the latent feature vectors, the following loss function is employed:

$$
\operatorname* { m i n } _ { \mathbf { P } , \mathbf { Q } } \quad \sum _ { ( u , i ) \in \Omega } ( r _ { u i } - \mathbf { p } _ { u } ^ { \top } \cdot \mathbf { q } _ { i } ) ^ { 2 } ,\tag{2}
$$

which measures the squared differences between the observed ratings $r _ { u i }$ and the ratings predicted by the model $\mathbf { p } _ { u } ^ { \top } \cdot \mathbf { q } _ { i }$ for each user-item pair $( u , i )$ in the set Ω of observed interactions.

Here, Yang et al. (2019) transferred the collaborative filtering for ML model selection by predicting the cross-valided errors, which demonstrates CF’s adaptability and efficiency in diverse application areas.

## 3 Background and Pilot Demonstration

## 3.1 Scaling Law on Downstream Tasks

For classic scaling laws, researchers propose a hypothesized power-law relationship between a model’s computational measures $C _ { m }$ (e.g., training FLOPs) and their performance loss $L _ { m } \left( e . g . \right.$ perplexity). Specifically, for a model m within a family f (e.g., Llama-2 7B, 13B, and 70B), the relationship is hypothesized as

$$
\log ( L _ { m } ) \approx \omega _ { f } \log ( C _ { m } ) + b _ { f } ,\tag{3}
$$

where $\omega _ { f }$ and $b _ { f }$ are scaling coefficients customized for each model family. Researchers fit

Latent Factor=7

Training/Validation = 50%/50%  
Training/Validation = 10%/90%  
![](images/b3172d9271b3dced501fbb231dc5c7e47214a70bdd80d9cbb07681e50a72a346.jpg)

![](images/5eb2adce5a10bb5de4f4c2dd781dcc182726a2244273103356f45ba114ee7509.jpg)

![](images/36ed7c142425cac41d928a34b0b113383dd86eaa36f4fbff1e3a8b9734d4396a.jpg)

![](images/431403b87b63653cc5b4bf3a5f9b88b275d1a0ddb3e95bd8906d3dc62fd13606.jpg)

![](images/93f514c1eb61026abd27cb27743003c4ccc2195bbec318772156ba57745759cb.jpg)

![](images/cd9f83a149c8af099663160978fd8196af344426080ee86a9c36b7f3486f50f5.jpg)

![](images/3d7dcd7553896a1f1a630748307d381a642981bbaa361bc3dedbaea04527445d.jpg)

![](images/1406c47836b2ebaa8b0fb27b604529aaed2d0cf006d57020eb12d4e58abfc1e5.jpg)  
Latent Factor=10  
Figure 2: Error Distribution of Predictions (Normalized Score and Rank Derived by Score) Based on the HELM Lite Leaderboard Using Matrix Factorization: We evaluate the effectiveness of Matrix Factorization (MF) using two latent factors, 7 and 10, across 2 training/validation split percentages. Accuracy is the percentage of instances where the predicted rank equals the actual rank. MAE@2 is defined as the percentage of instances where the absolute difference between the predicted and actual ranks is 2.

this formula through repeated scaling experiments, then use it to accurately predict performance when larger-scale $( C ^ { \prime } > C )$ . Some studies (Finnveden, 2020; Owen, 2024) have adapted scaling laws to specific downstream task metrics, proposing that sigmoidal functions are more suitable for predictions, as follows:

$$
\sigma ^ { - 1 } ( S _ { m } ) \approx \omega _ { f } \log ( C _ { m } ) + b _ { f } ,\tag{4}
$$

where $S _ { m }$ refers to the normalized downstream scores of models within the range [0, 1]. However, applying scaling laws across different model families on various specific tasks presents a tradeoff: fitting unique coefficients for each evaluation scenario $( e . g .$ , Llama 2 on MMLU) is a resourceintensive endeavor (Hu et al., 2024); alternatively, estimating these coefficients using a limited number (3-5) of models within the same family may compromise the accuracy of the predictions. Moreover, the recent work (Ruan et al., 2024) extends scaling law by incorporating latent variables to capture the patterns across model families and tasks.

## 3.2 Pilot Demonstration on HELM

Scaling laws reveal that models from any family exhibit a similar performance trend as computational measures increase. This insight suggests there are commonalities and connections between different models. These motivate us to employ the MF method to explore more similarities beyond computational measures, $e . g .$ ., the relationship among the different model families and tasks.

We perform the aforementioned MF on the benchmark matrix to observe the error gap between predicted and truth (normalized) scores. Specifically, we select the core leaderboard provided by HELM for our exploratory experiments with only model name, task name, and performance scores. This leaderboard, 68 models and 16 tasks, presented in a score matrix with a density of 82.5%, which includes both open-source and proprietary models, $e . g .$ , GPT-4 and Jurassic-2. Our method treats all models and tasks as independent entities without introducing any prior similarity factors. We hope to observe whether MF can predict the remaining scores, giving a small part of the matrix, where we evaluate two training/validation sets split strategies: 10%/90%, 50%/50%. As illustrated in Figure 2, MF can accurately predict most of the missing scores within a low error range, which proves that it can encode the similarity across the model and the task without regression depending on explicit computational measures variable.

## 4 Collaborative Performance Prediction

## 4.1 Definition

Motivated by pilot experiments, we introduce the concept of “Collaborative Performance Prediction” (CPP) to facilitate the performance prediction of LLMs.

Definition 1. Let $\mathcal { M } = \{ M _ { 1 } , M _ { 2 } , \dots , M _ { n } \}$ be a set of n LLMs, and $\mathcal { T } = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { m } \}$ be a suite of m tasks. Define the Score Matrix $\mathbf { S } ,$ which is an n m matrix where each element $s _ { i j }$ represents the performance score ofmodel $M _ { i }$ on task $T _ { j } . ~ s _ { i j }$ is defined as

$$
s _ { i j } = { \left\{ \begin{array} { l l } { s c o r e } & { i f t e s t e d , } \\ { u n k n o w n } & { o t h e r w i s e . } \end{array} \right. }
$$

Function: Employ an prediction method to estimate the unknown elements of S, denoted by $\widehat { s } _ { i j }$ based on the known values.

Extention: Accommodate model design factors ${ { \mathcal { V } } _ { m } } = \{ V _ { m } ^ { 1 } , V _ { m } ^ { 2 } , . . . , V _ { m } ^ { M } \}$ , such as common computational meatures, and task design factors $\nu _ { t } =$ $\{ V _ { t } ^ { 1 } , V _ { t } ^ { 2 } , \ldots , V _ { t } ^ { T } \}$ , such as targeted capabilities and few-shot settings.

Based on this definition, our framework consists of two components: 1) collaborative performance data, 2) collaborative prediction methods. We anticipate that an accurate score can be predicted based on the historical performance of various models on downstream tasks and other design factors for both model and task. Moreover, we can incorporate or solely rely on the factors describing the LLM and the associated downstream tasks.

## 4.2 Collaborative Data

![](images/ad6cefb114d0eaed6e5f88433905bbbf14d825815f97cdf643b1db561e01f366.jpg)

![](images/3b3d24193466741c94b9aa18cd7675fe7714dd6ecc35cee0785f8c24413eaf2c.jpg)  
Figure 3: Distribution of Testing Coverage Across Models and Tasks. The left bar shows the number of tasks each model has been tested on; The right bar illustrates the number of models tested in each specific task.

Unlike the scaling law approach, which requires training resource factors to obtain the correlation between metric scores and factors at a high training cost, our proposed method makes use of evaluation results and other design factors reported from existing studies, referred to as collaborative data. Open-source leaderboards, such as Open-$\mathrm { L L M } ^ { 2 }$ , HELM, and OpenCompass<sup>3</sup>, have made tremendous efforts on this issue in fairly evaluating different LLMs. Our efforts extend beyond merely (Ruan et al., 2024) utilizing data from opensource leaderboards with matrix sparsity of 0%. We also extract test results from different models’ papers, technical reports, and model cards. Ultimately, we have collected a score matrix of $n = 7 2$ $m = 2 9$ with a density of only 56%. Furthermore, we collected 12 and 4 detailed design factors for models and tasks. These details are listed in Appendix B.1. Our data analysis is shown in Figure 3 and Figure 8.

Data Analysis. Based on the collective data, we can make the following observations: a) Uneven distribution of testing resources. We observe significant variability in the deployment of testing efforts, as shown in Figure 3. For instance, models from the LLAMA series have undergone extensive testing across various tasks, in contrast to models like GOPHER, where testing has largely stagnated. A similar disparity is also evident among tasks, with MMLU and HELLASWAG receiving considerable evaluation, whereas RACE has been relatively underexplored. This trend suggests that as LLMs proliferate and tasks evolve, scores across the matrix will increasingly skew. This leads to a pronounced long-tail effect in testing coverage for many tasks, barring a few that consistently receive comprehensive evaluations. b) Widespread variations in the scores. It is noteworthy that identical models yield varying scores on the same tasks across different studies (Shrivastava et al., 2023; AI@Meta, 2024), a variation often attributed to differences in prompt settings, model versions, and the volume of test samples employed. Typically, these score variations range within 0.1, with scores normalized between [0, 1]. This phenomenon underscores the importance of public leaderboards and highlights researchers’ need to articulate their testing frameworks when performing customized evaluations. When conflicted, we prefer the results from the Open-LLM leaderboard in the collective data. c) Missing description/model card.

We advocate for consistently providing complete model cards for open-source and proprietary models. Such a phenomenon is shown in Figure 8 and, unsurprisingly, a long-tail distribution is witnessed. While it is understandable that proprietary models might withhold specific details about parameters, they can still divulge information about parameter scale and the extent of pre-training. Furthermore, we recommend a more thorough description of testing tasks, including suggested few-shot settings and detailed descriptions of targeted capabilities.

## 4.3 Prediction Methods

In Section 2.2, classical collaborative filtering methods are inspired to conduct the performance prediction. In principle, most collaborative filtering methods can be applied. Here, in addition to the abovementioned MF, we also leverage neural collaborative filtering (He et al., 2017) (NCF) methods, which uses a multi-layer perceptron to learn the model-task interaction function to predict the score $\widehat { s } _ { i j }$ for a model i on a task $j ,$ providing a way bto learn non-linearities in the data:

$$
\begin{array} { r l } & { \widehat { s } _ { i j } = f ( i , j | \mathcal { M } , \mathcal { T } , [ \mathcal { V } _ { i } , \mathcal { V } _ { j } ] , \theta ) } \\ & { \quad \quad = \mathbf { M } \mathbf { L } \mathbf { P } ( p _ { i } , q _ { j } , [ e _ { v i } , e _ { v j } ] ) , } \end{array}\tag{5}
$$

where and $\tau$ denote the sets of collaborative models and tasks, and their descriptive factors $\nu _ { i }$ $\nu _ { j }$ optionally enrich the input data. Here, $p _ { j }$ and $q _ { j }$ are the latent vectors for model i and task $j$ that capture the intrinsic properties of models and tasks, as well as embeddings $[ e _ { v i } , e _ { v j } ]$ derived from their descriptive factors, and θ represents the parameters of NCF.

Moreover, we further simplify the model to verify whether it is feasible to predict a score when only inputting the descriptive factors $\nu _ { i } , \nu _ { j }$ into the prediction model:

$$
\begin{array} { r } { \widehat { s } _ { i j } = f ( i , j | \mathcal { V } _ { i } , \mathcal { V } _ { j } , \boldsymbol { \theta } ) } \\ { = \mathbf { M } \mathbf { L } \mathbf { P } ( e _ { v i } , e _ { v j } ) , } \end{array}\tag{6}
$$

For both settings, where the goal is to predict the scores accurately, the loss function can be defined as follows:

$$
L ( \theta ) = \frac { 1 } { N } \sum _ { ( i , j ) \in \mathcal { D } } ( \widehat { s } _ { i j } - s _ { i j } ) ^ { 2 } ,\tag{7}
$$

where $N$ is the total number of scores set for training, and $s _ { i j }$ is the true score for model i and task j.

## 5 Experiments

In this section, we evaluate the feasibility of CPP from an overall benchmark perspective and a model perspective in Section 5.1 and 5.2, respectively; we then analyze the importance of factors for both models and tasks in Section 5.3. Additionally, a substantial amount of ablation and analysis is placed in the appendix D, such as sparsity, the correlations in tasks and models, and which models and tasks are more critical for prediction.

Experimental Setting. Our validation framework utilizes the aforementioned collaborative dataset as the score matrix . We partition scores $\{ s _ { i j } \}$ into train and validation set, detailed in Appendix C.2.

Evaluation Metric. To accurately evaluate CPP, we adopt two types of metrics: 1) SCORE-LOSS metrics including MSE LOSS and L1 LOSS between predicted scores and true scores (normalized) on downstream tasks and 2) RANK-ACCURACY metrics including ACCURACY and MAE@2 between the rank of predicted scores and true scores. We elaborate on these metrics in Appendix C.1.

## 5.1 Evaluation from Benchmark Perspective

In this study, we select the abovementioned methods, MF and NCF, to verify whether $s _ { i j }$ can be accurately predicted based on the input of model i and task $j .$ . To examine whether enhancements are helpful, we modify NCF to support the input of design factors, detailed in Appendix C.2. Based on Figure 4 and Table 1, we can make the following observations:

First, all methods accurately predicted model performance, demonstrating that collaborative filtering mechanisms can predict model outcomes based on collaborative data across different models and tasks. This prediction is achieved without explicit scaling factors or fitting a log-power curve. Second, from MF to NCF, the transformation in interaction mechanisms further enhances accuracy, suggesting that model improvements can further augment the efficacy of our methodology. Additionally, we further increased accuracy by incorporating factors, such as model scaling variables and task descriptions, into the NCF framework alongside ID information. This confirms that incorporating explicit factors can enhance model and task similarities. Finally, among all metrics, we particularly noted that the accuracy of the predictive ranking was acceptable. In other words, researchers can use our method to accurately predict the ranking range of their developed models on test tasks,0.8 thereby enhancing model performance on specific tasks.c<sup>o</sup>

![](images/45608439895ef2bcf0a1bb44bb6c76980383a0a0bc73cb17e2db772787574491.jpg)  
Matrix Factorization

![](images/353d44de31372d21df65bb7a9ca4f8f0b6f431c7dee3f028b164189ebb68abed.jpg)  
Neural Collaborative Filtering

![](images/aa0408422d4e18901c0548a2c96c3f2ae26d071b1fba30732cd6048b3c5cdd29.jpg)  
Neural Collaborative Filtering (Factor-enhanced)

![](images/d94e2710787659d7a2c4afb93a0f414cc4c6a1a68db3c27b5c4a25374ecb635b.jpg)  
Neural Collaborative Filtering (only Factor)

Figure 4: Comparative visualization of predictive accuracy across various scoring methods. From left to right: c<sup>o</sup>MF, NCF, NCF with Factor Enhancement, and NCF based solely on Factors. Each plot displays the regression u<sup>a</sup>between predicted and actual scores, where the solid line represents the regression fit and the shaded area denotes <sup>A</sup>the confidence interval (CI). A line closer to the diagonal indicates perfect prediction and higher prediction accuracy. These plots demonstrate the enhanced performance in score prediction achieved by integrating factors into the NCF method.
<table><tr><td rowspan="2">Prediction Method</td><td colspan="2">Score-Loss</td><td colspan="2">Rank-Acc</td></tr><tr><td> $\overline { { { \mathrm { M S E } } { \mathrm { L o s s } } \downarrow } }$ </td><td> $\overline { { \mathrm { M e a n } \ L 1 \ L \mathrm { o s s } \ \downarrow } }$ </td><td>Mean Prec.(%) ↑</td><td>MAE@2(%) ↑</td></tr><tr><td>Matrix Factorization</td><td> $\overline { { 2 . 1 6 e ^ { - 2 } ( 1 . 1 9 e ^ { - 4 } ) } }$ </td><td> $\overline { { 9 . 4 7 e ^ { - 2 } ( 2 . 8 9 e ^ { - 4 } ) } }$ </td><td>44.33(0.69)</td><td>83.16(0.73)</td></tr><tr><td>Neural Collaborative Filtering</td><td> $1 . 5 8 e ^ { - 2 } ( 4 . 2 2 e ^ { - 5 } )$ </td><td> $8 . 9 4 e ^ { - 2 } ( 3 . 1 0 e ^ { - 4 } )$ </td><td>41.76(1.22)</td><td>84.98(0.42)</td></tr><tr><td>+ Factor Enhanced</td><td> $\mathbf { 1 . 2 5 e ^ { - 2 } ( 3 . 3 5 e ^ { - 6 } ) }$ </td><td> $\mathbf { 7 . 8 8 e ^ { - 2 } ( 6 . 3 1 e ^ { - 5 } ) }$ </td><td>45.45(0.33)</td><td>84.54(0.27)</td></tr><tr><td>Only Factor</td><td> $\overline { { 1 . 7 5 e ^ { - 2 } ( 2 . 0 7 e ^ { - 5 } ) } }$ </td><td> $\overline { { 8 . 5 7 e ^ { - 2 } ( 1 . 4 8 e ^ { - 4 } ) } }$ </td><td>33.47(0.12)</td><td>84.08(0.37)</td></tr></table>

Table 1: Comparison of prediction methods for LLM performance. Bold indicates the best-performed.

Predictability with Only Description Factors.t<sup>u</sup> 0.4 We validate whether high predictive accuracy can still be achieved by only inputting the models’ andMSE\_Loss@CPP: 0 tasks’ design factors. As demonstrated in Table 1, the accuracy of predicted rankings (derived from0.0 0.2 0.4 0.6 0.8 1.0 predicted scores) remains high, affirming that our method supports predictions based solely on factors. However, the accuracy is lower than other<sub>0.8</sub> models, suggesting that finer-grained latent similarities remain encoded as potential factors within the<sub>c</sub>o 0.6 identity information across different models andal tasks.<sub>A</sub><sup>c</sup>

## 5.2 Evaluation from Model PerspectiveMSE\_

To mimic the utilization of CPP in the real world,0.0 this section takes a model perspective to investigate<sup>0.0</sup> <sup>0.2</sup> <sup>0.4</sup> <sup>0.6</sup> <sup>0.8</sup> <sup>1.0</sup> the predictive accuracy of CPP upon each model. Specifically, we propose two scenarios: (i) prediction with no prior testing information and (ii) prediction with prior testing information on 2 tasks. These two scenarios correspond to real-world cases when the model has not been developed or is tested on a few tasks and expects an accurate prediction of its ability on other tasks. In both scenarios, we focus on larger LLMs, e.g., LLama2-70b, as they are more computationally expensive to develop and test, requiring an accurate LLM prediction.

(a) With no prior testing information (CPP-0)  
![](images/9f353a19d4f5992fca78c85d281e0644d903ea98160cad717defa9aeee76cfe0.jpg)

![](images/f91a6bfb4e564901344f7a4a89090e12a4555c4af7861e4fca3aafc79d7e46a7.jpg)  
Figure 5: Comparison of the predictive performance of collaborative performance prediction (CPP) versus traditional scaling laws (SL) for LLMs: (a) CPP-0, with no prior testing information, and (b) CPP-2, with prior testing on two tasks.

We report the results of CPP and SL on both scenarios in Figure 5 and can draw the following conclusions. Under the CPP-0 scenario, CPP demonstrated greater adaptability across different tasks compared to SL, with points closely aligned along the $y = x$ line (“perfect prediction”) in Figure 5 (a). This suggests that CPP has effectively captured task-specific characteristics, such as value ranges, whereas SL, despite achieving a lower MSE-LOSS, tends to concentrate its predictions around 0.5. Under the CPP-2 scenario, the distribution of points of CPP is noticeably closer to $y = x ,$ , as shown in Figure 5 (b), and its MSE-LOSS is also lower than that of SL. This indicates that leveraging performance data from other tasks considerably enhances the model’s cross-task prediction capabilities, underscoring a degree of consistency across tasks for the same model. This approach demonstrates that predictions for scaling LLMs on downstream tasks can be dynamically improved by evaluating performance on less computationally intensive tasks and using those outcomes to predict scores on subsequent tasks more accurately.

## 5.3 Factor Importance Analysis via SHAPLEY-VALUE

In this section, we aim to analyze each design factor’s importance over CPP. The Shapley value, a concept derived from cooperative game theory (Shapley, 1952), offers a systematic approach to measuring individual factors’ contribution in predictive models (Lundberg and Lee, 2017; Covert et al., 2021). Appendix C.3 shows a detailed formulation of the Shapley value. Visualization for Shapley values of each design factor is shown in Figure 6.

Based on Figure 6 (a), we can make the following observations regarding model factors. First, we have discovered that in addition to traditionally important factors such as training data size and parameter size mentioned in scaling law (Kaplan et al., 2020), other design factors significantly influence predictive outcomes. These include the model family, context window size, and batch size. Second, the importance of the model family cannot be overlooked, as it may relate to differences in data quality across models, including proprietary data or specific architectural details. For instance, using a particular model family might mean adopting architectures or optimization techniques better suited to specific tasks. Moreover, the size of the context window also significantly affects model performance. A larger context window allows the model to better understand the context in long texts, which is particularly crucial for long-context LLMs (Xiong et al., 2023). Experience (Google, 2024) has shown that such models perform better across various tasks. Batch size, as another crucial factor, affects the stability and speed of model training. An appropriate batch size ensures a balance between the accuracy of gradient estimation and computational efficiency during training.

![](images/9f1fadf4ab00c2fb3d1bfb799c0d6a4c01c076f1e4b78ae1e5ea0ebd83c490c8.jpg)

(b) Task Factor  
![](images/b286e5a4e39a617e1c2d4356f5c2e67fa432377a50cb69bc0ef0f5c0626a6354.jpg)  
Figure 6: Mean Shapley Value on Each Factor.

As for the importance of task factors, results in Figure 6 (b) show that the target ability among all factors is more important. This also implies that similarities between the domains of different tasks can help predict outcomes. This conclusion is consistent with previous observations (Ruan et al.,

2024; Perlitz et al., 2024; Polo et al., 2024)

In summary, these findings indicate that LLMs performance prediction should not rely solely on traditional design factors limited by scaling law but also on other key factors that might impact overall model performance.

## 6 Conclusion and Discussion

Advancing beyond traditional scaling laws on downstream tasks, we propose a collaborative performance prediction framework for large language models. It offers significant advantages, including easy deployment, low training costs, and superior predictive accuracy. Uniquely, it enables incorporating additional design factors and supports an in-depth analysis of their impact, including factor importance and correlations in models and tasks. For prediction, we collect collaborative data containing many historical performances and factors.

Our method’s predictive accuracy is expected to improve as it benefits from an expanding pool of collaborative data. Moreover, this approach highlights the potential to identify neglected but vital factors beyond traditional scaling laws, such as task design factors, thereby enriching our comprehension of LLM performance predictability on downstream tasks.

## Limitations

“Single-source-of-truth”. When collecting the collaborative data, we hypothesize that each model’s performance on each task is identical. However, in the real world, the detailed testing setting, for instance, the testing prompt writing, can influence LLM’s performance variance. Although we observed this, we only saved one score from different sources. How to incorporate the setting of testing as an additional dimension remains to be solved in future works.

Susceptibility to data quality. The prediction accuracy of CPP highly depends on the quality of collaborative data. The current version passively collects collaborative data from online resources. Should information from either of these data sources be incorrect, the prediction capability of CPP would decrease correspondingly. To overcome such a limitation, jointly considering passive information collected from data sources and active information, such as performances of models tested on some tasks by the user, might be a solution.

Utilizing techniques such as efficient benchmarking (Perlitz et al., 2024; Polo et al., 2024) could alleviate the cost of obtaining active information.

## Ethics Statement

The data we use are collected from public papers, technical reports, open leaderboards, and model cards on GitHub.

## Acknowledgements

This work was supported by the Start-up Grant (No. 9610564), the Donations for Research Projects (No. 9229129) of the City University of Hong Kong, and the Early Career Scheme (No. CityU 21219323) of the University Grants Committee (UGC).

## References

AI@Meta. 2024. Llama 3 model card.

Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Yoshua Bengio, Danqi Chen, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, and David Krueger. 2024. Foundational challenges in assuring alignment and safety of large language models. In arXiv.

Sanjeev Arora and Anirudh Goyal. 2023. A theory for emergence of complex skills in language models. In arXiv.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models. In arXiv.

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. 2024. Explaining neural scaling laws. In arXiv.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances

in Neural Information Processing Systems, pages 1877–1901.

Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. 2023. Broken neural scaling laws. In International Conference on Learning Representations.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. In arXiv.

François Chollet. 2019. On the measure of intelligence. In arXiv.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. In arXiv.

Ian C. Covert, Scott Lundberg, and Su-In Lee. 2021. Explaining by removing: a unified framework for model explanation. The Journal ofMachine Learning Research.

Lukas Finnveden. 2020. Extrapolating gpt-n performance.

Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, Sheer El Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Scott Johnston, Andy Jones, Nicholas Joseph, Jackson Kernian, Shauna Kravec, Ben Mann, Neel Nanda, Kamal Ndousse, Catherine Olsson, Daniela Amodei, Tom Brown, Jared Kaplan, Sam McCandlish, Christopher Olah, Dario Amodei, and Jack Clark. 2022a. Predictability and surprise in large generative models. In Conference on Fairness, Accountability, and Transparency. ACM.

Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, Sheer El Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Scott Johnston, Andy Jones, Nicholas Joseph, Jackson Kernian, Shauna Kravec, Ben Mann,

Neel Nanda, Kamal Ndousse, Catherine Olsson, Daniela Amodei, Tom Brown, Jared Kaplan, Sam McCandlish, Christopher Olah, Dario Amodei, and Jack Clark. 2022b. Predictability and surprise in large generative models. In ACM Conference on Fairness, Accountability, and Transparency, pages 1747––1764.

Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, and Colin Cherry. 2021. Scaling laws for neural machine translation. In arXiv.

Google. 2024. Gemini 1.5 blog.

Mitchell A Gordon, Kevin Duh, and Jared Kaplan. 2021. Data and parameter scaling laws for neural machine translation. In Empirical Methods in Natural Language Processing, pages 5915–5922.

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In International Conference on World Wide Web, pages 173–182.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, and Sam McCandlish. 2020. Scaling laws for autoregressive generative modeling. In arXiv.

Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Ben Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, and Sam McCandlish. 2022. Scaling laws and interpretability of learning from repeated data. In arXiv.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack William Rae, and Laurent Sifre. 2022. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems.

Shengding Hu, Xin Liu, Xu Han, Xinrong Zhang, Chaoqun He, Weilin Zhao, Yankai Lin, Ning Ding, Zebin Ou, Guoyang Zeng, Zhiyuan Liu, and Maosong Sun.

2024. Predicting emergent abilities with infinite resolution evaluation. In International Conference on Learning Representations.

David Ilic. 2023.´ Unveiling the general intelligence factor in language models: A psychometric approach.

Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii, and Sanmi Koyejo. 2024. Scaling laws for downstream task performance of large language models. In arXiv.

Neil Jethani, Mukund Sudarshan, Ian Covert, Su-In Lee, and Rajesh Ranganath. 2022. Fastshap: Real-time shapley value estimation. In arXiv.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. In arXiv.

Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37.

Yehuda Koren, Steffen Rendle, and Robert Bell. 2022. Advances in Collaborative Filtering, pages 91–142. Springer.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic evaluation of language models. In arXiv.

Nelson F. Liu, Tony Lee, Robin Jia, and Percy Liang. 2023. Do question answering modeling improvements hold across benchmarks? In arXiv.

Scott M. Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. In International Conference on Neural Information Processing Systems, pages 4768–4777.

Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. 2023. Scaling data-constrained language models. In Conference on Neural Information Processing Systems.

Frank Nielsen. 2016. Hierarchical Clustering, pages 195–211. Springer.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, pages 27730–27744.

David Owen. 2024. How predictable is language model benchmark performance? In arXiv.

Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv, Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal Shmueli-Scheuer, and Leshem Choshen. 2024. Efficient benchmarking of language models. In arXiv.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. 2024. tinyBenchmarks: evaluating llms with fewer examples. In arXiv.

Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. 2024. Observational scaling laws and the predictability of language model performance. In arXiv.

Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. 2023. Are emergent abilities of large language models a mirage? In Conference on Neural Information Processing Systems.

Lloyd S. Shapley. 1952. A Valuefor N-Person Games. RAND Corporation.

Vaishnavi Shrivastava, Percy Liang, and Ananya Kumar. 2023. Llamas know what gpts don’t show: Surrogate models for confidence estimation. In arXiv.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. In arXiv.

François Torregrossa, Vincent Claveau, Nihel Kooli, Guillaume Gravier, and Robin Allesiardo. 2020. On the correlation of word embedding evaluation metrics. In Language Resources and Evaluation Conference, pages 4789–4797.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent abilities of large language models. In arXiv.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. In arXiv.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao Ma. 2023. Effective long-context scaling of foundation models. In arXiv.

Chengrun Yang, Yuji Akimoto, Dae Won Kim, and Madeleine Udell. 2019. Oboe: Collaborative filtering for automl model selection. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, page 1173–1183.

Zhang Zhuocheng, Shuhao Gu, Min Zhang, and Yang Feng. 2023. Scaling law for document neural machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8290–8303.

## A Pilot Demonstrations using Neural Collaborative Filtering

In this section, we supplemented the error distribution in Figure 7, which is generated using neural collaborative filtering on the HELM lite leaderboard. Compared to Figure 2, it is evident that neural collaborative filtering consistently outperforms MF across each setting.

## B Collaborative Data

## B.1 Data Description

List of Models and Tasks. The table 2 contains all the models and tasks we have collected.

Description Factors for Models and Tasks We have collected the characteristics of models and tasks in relevant aspects through model cards, technical reports, and academic papers. We have organized and introduced these characteristics, as well as the corresponding embedding methods, as listed in Table 3.

Note that during data collection, not all factors are available. For these missing factors, such as CO2 and GPU hours, we replace them as zero when entering data.

## B.2 Data Analysis

We conducted a statistical analysis of the data we collected, specifically examining the number of models tested for each task, the number of tasks tested for each model, and the number of models described by each factor. Since each task is consistently associated with four factors, we did not create a distribution chart for this aspect.

## C Experimental Setup

## C.1 Evaluation Metrics

Apart from visualization, we also evaluate the method based on two types of metrics: 1) SCORE-LOSS Metric: we calculate MSE LOSS and L1 LOSS between predicted scores and true scores (normalized) on downstream tasks; 2) RANK-ACCURACY Metric: researchers are sometimes not concerned with detailed scores but rather the rankings the model is in, so we calculate the accuracy of rank derived from the predicted scores, ACCU-RACY and MAE@2. ACCURACY refers to the percentage of instances where the predicted rank equals the true rank, and MAE@2 refers to the percentage of instances where the absolute difference between the predicted rank and the true rank is in 2, the formulation as below:

$$
\mathrm { A c c u r a c y } = \left( { \frac { \sum _ { i = 1 } ^ { N } \mathbf { 1 } ( r _ { i } = { \widehat { r } } _ { i } ) } { N } } \right) \times 1 0 0 \% ,\tag{8}
$$

$$
\mathrm { M A E @ 2 } = \left( \frac { \sum _ { i = 1 } ^ { N } \mathbf { 1 } ( | r _ { i } - \widehat { r } _ { i } | \leq 2 ) } { N } \right) \times 1 0 0 \% ,\tag{9}
$$

where N is the total number of validation instances, $r _ { i }$ is the true rank, $\widehat { r _ { i } }$ is the predicted rank derived bby the predicted score; 1( ) is the indicator function that evaluates to 1 if the argument is true and 0 otherwise;    denotes the absolute value.

## C.2 Detailed Setting of Validation Prediction Accuracy Experiments

In this section, we detail the setup of each experiment in 5.

Different Prediction Methods. Due to the 44% sparsity of the collected collaboration matrix, we used 5% of the known data as the validation set, with the remaining data serving as the observed training set. We trained each model five times through random splitting, deriving an average performance and variance. We configured our models with latent factors = 10, learning rate = 0.01, and iteration = 250, 000. The Figure 4 is the results when random\_seed = 1.

Predicting from Small to Large LMs. The focus here is on deriving the scaling law applicable to specific task metrics. Undeniably, traditional methods do not provide a directly usable scaling law across all downstream tasks for comparative analysis. However, we observed in the literature (Ruan et al., 2024) that a sigmoidal curve with a single coefficient and a single bias value represents the scaling law for downstream tasks. Moreover, this curve’s coefficients and bias values have a general range across all tasks, $w = [ 0 . 5 , 2 ] , b = [ - 1 0 , - 3 ]$ Consequently, we set this range of coefficients and bias for this curve. Then we used the normalized scores of smaller models within the same model family and their corresponding parameter sizes to fit the scaling law curve for each task. This approach generally follows a “pretrain-finetune” methodology. Additionally, CPP-2 refers to randomly selecting two scores from the observed performances of the model to be included in the training data. In this experiment, we use factorenhanced NCF (setting is same as above).

![](images/142ead64f3e37756092ac15f55c276532aa7c25eb292f39dc67227abc6f55232.jpg)  
Training/Validation=10%/90%

![](images/2ecbd7412e12b8a3e84b3676fcc1186ce4a4aea771a7ac3d0c265c543a1fc020.jpg)

![](images/15f8bd9f23d9cd554c1e8a5189a9fc95182cd92d13735a51e1505ceeb3e4bccc.jpg)

![](images/1830e9e3ec03978e48f2a7457f13af80db1c6dc912e7f7c565683e1c7b7bdcfd.jpg)

![](images/496f7b1a358dd31fd6913bd240345f1438eb635fff7e6d6957c178ebb6a1141d.jpg)

![](images/96d3b58adc5163230cc3d596c9d5c946e337f6c9cd3cdddcd6d54a4856fa7928.jpg)

![](images/b65dce09f60d6220f9bc01bbb6fb7414021d2770b9d4393342426db615775830.jpg)  
Latent Factor=7

![](images/807b72136b3c4cfce750e4ade876f3f9ba089e2426f700ba62277f831b8c4001.jpg)

![](images/384d3f54f1dc9fccb9584ec30405e9c86a1e9a634a0ff5eb8540da2bf3fd0477.jpg)  
Latent Factor=10

Figure 7: Error Distribution of Predictions (Normalized Score and Rank Derived by Score) Based on the HELM Lite Leaderboard Using Neural Collaborative Filtering: We evaluate the effectiveness of Matrix Factorization (MF) using two latent factors, 7 and 10, across 2 training/validation split percentages. Accuracy is defined as the percentage of instances where the predicted rank equals the actual rank. MAE@2 is defined as the percentage of instances where the absolute difference between the predicted rank and the actual rank is 2.  
![](images/3a808c5bfdfe2bd96778b0d9f6ab4fb33912d69ff4676541c7ffa0deaaa21bf5.jpg)  
Table 2: List of Models and Tasks

## C.3 Detailed Setting of Analysis Experiments

Shapley-Value for Factor Importance Analysis. Given a predictive model f and a set of factors N, the Shapley value of a factor i is computed as follows:

$$
\phi _ { i } ( v ) = \sum _ { \begin{array} { l } { S \subseteq N \setminus \{ i \} } \\ { \cdot \left[ v \left( S \cup \{ i \} \right) - v \left( S \right) \right] , } \end{array} } { \frac { | S | ! ( | N | - | S | - 1 ) ! } { | N | ! } }\tag{10}
$$

where:

• N is the total set of factors.

• S is a subset of factors excluding factor i.

• S is the number of factors in subset S.

• v(S) is the prediction model’s output when only the factors in subset $S$ are used.

<table><tr><td colspan="3">Model</td></tr><tr><td>Factors</td><td>Description</td><td>Embedding</td></tr><tr><td>Model Family</td><td>Type of model family, e.g., LLAMA 2, PYTHIA</td><td>Categorical Embedding</td></tr><tr><td>Pretraining Dataset Size (B)</td><td>Data size in millions of tokens</td><td>Numerical Embedding</td></tr><tr><td>Parameter Size (M)</td><td>Number of model parameters in millions</td><td>Numerical Embedding</td></tr><tr><td>GPUh</td><td>GPU hours consumed</td><td>Numerical Embedding</td></tr><tr><td>FLOPs</td><td>Floating-point operations count</td><td>Numerical Embedding</td></tr><tr><td>Context Window</td><td>Max context size in tokens, e.g., 1024, 2048</td><td>Categorical Embedding</td></tr><tr><td>Batch Size (M)</td><td>Size of batches in millions,e.g., 1M, 2M</td><td>Categorical Embedding</td></tr><tr><td>Layers</td><td>Number of layers in the model</td><td>Numerical Embedding</td></tr><tr><td>Number Heads</td><td>Number of attention heads</td><td>Numerical Embedding</td></tr><tr><td>Key/Value Size</td><td>Size of key/value in attention mechanism</td><td>Numerical Embedding</td></tr><tr><td>Bottleneck Activation Size</td><td>Size of activation in bottleneck layers</td><td>Numerical Embedding</td></tr><tr><td>Carbon Emission (tCO2Eq)</td><td>Carbon footprint of training</td><td>Numerical Embedding</td></tr><tr><td colspan="3">Task</td></tr><tr><td>Ability</td><td>Type of targeted cognitive ability, e.g., reasoning</td><td>Categorical Embedding</td></tr><tr><td>TaskFamily</td><td>Related task family ,e.g., ARC</td><td>Categorical Embedding</td></tr><tr><td>Output Format</td><td>Format of task output, e.g., binary</td><td>Categorical Embedding</td></tr><tr><td>Few-Shot Setting</td><td>Description of few-shot learning setting,e.g., zero-shot, 32-shot</td><td>Categorical Embedding</td></tr></table>

Table 3: Design Factors of Models and Tasks
<table><tr><td rowspan="2">Scaled LLMs</td><td rowspan="2">Prior Tasks</td><td colspan="2">Score-Loss</td><td colspan="2">Rank-Acc</td></tr><tr><td>MSE Loss</td><td>Mean L1 Loss</td><td>Mean Prec.(%)</td><td>MAE@2(%)</td></tr><tr><td rowspan="2">LLaMA 2-70B</td><td>CF-0</td><td> $\overline { { 1 . 3 4 e ^ { - 2 } } }$ </td><td> $\overline { { 8 . 8 3 e ^ { - 2 } } }$ </td><td>16.7</td><td>50.0</td></tr><tr><td>CF-2</td><td> $1 . 7 9 e ^ { - 2 } ( 1 . 3 e ^ { - 3 } )$ </td><td> $1 . 7 9 e ^ { - 2 } ( 5 . 6 e ^ { - 4 } )$ </td><td>9.1(7.5e−3)</td><td>54.5(5.7e−4)</td></tr><tr><td rowspan="2">LLaMA 3-70B</td><td>CF-0</td><td> $5 . 6 3 e ^ { - 2 }$ </td><td> $\overline { { 1 9 . 2 7 e ^ { - 2 } } }$ </td><td>14.3</td><td>71.4</td></tr><tr><td>CF-2</td><td> $1 . 7 e ^ { - 2 } ( 1 . 4 1 e ^ { - 4 } )$ </td><td> $1 0 . 7 e ^ { - 2 } ( 1 . 6 8 e ^ { - 3 } )$ </td><td> $2 0 . 0 ( 4 . 0 e ^ { - 2 } )$ </td><td>90.0(9.0e−2)</td></tr><tr><td rowspan="2">LLaMA-65B</td><td>CF-0</td><td> $1 . 7 3 e ^ { - 2 }$ </td><td> $9 . 7 8 e ^ { - 2 }$ </td><td>24.0</td><td>80.0</td></tr><tr><td>CF-2</td><td> $1 . 8 8 e ^ { - 2 } ( 1 . 4 2 e ^ { - 5 } )$ </td><td> $1 0 . 0 2 e ^ { - 2 } ( 4 . 1 e ^ { - 4 } )$ </td><td> $1 7 . 3 ( 1 . 9 e ^ { - 3 } )$ </td><td> $7 1 . 7 ( 4 . 7 e ^ { - 4 } )$ </td></tr><tr><td rowspan="2">Luminous Supreme-70B</td><td>CF-0</td><td> $6 . 0 6 e ^ { - 2 }$ </td><td> $\overline { { 2 0 . 1 4 e ^ { - 2 } } }$ </td><td>27.27</td><td>63.63</td></tr><tr><td>CF-2</td><td> $1 . 4 5 e ^ { - 2 } ( 1 . 1 e ^ { - 5 } )$ </td><td> $1 0 . 7 9 e ^ { - 2 } ( 6 . 4 e ^ { - 7 } )$ </td><td> $1 6 . 7 ( 3 . 1 e ^ { - 3 } )$ </td><td>83.3(3.5e−3)</td></tr><tr><td rowspan="2">Pythia-12B</td><td>CF-0</td><td> $\overline { { 2 . 1 9 e ^ { - 2 } } }$ </td><td> $\overline { { 1 1 . 2 e ^ { - 2 } } }$ </td><td>21.42</td><td>71.42</td></tr><tr><td>CF-2</td><td> $1 . 5 7 e ^ { - 2 } ( 2 . 1 e ^ { - 6 } )$ </td><td> $1 0 . 8 8 e ^ { - 2 } ( 4 . 6 e ^ { - 8 } )$ </td><td> $3 3 . 3 ( 2 . 7 e ^ { - 2 } )$ </td><td> $6 6 . 7 ( 6 . 9 e ^ { - 3 } )$ </td></tr><tr><td rowspan="2">Yi-9b</td><td>CF-0</td><td> $3 . 2 0 e ^ { - 2 }$ </td><td> $\overline { { 1 4 . 6 6 e ^ { - 2 } } }$ </td><td>44.4</td><td>100.0</td></tr><tr><td>CF-2</td><td> $0 . 9 e ^ { - 2 } ( 3 . 1 e ^ { - 4 } )$ </td><td> $8 . 1 e ^ { - 2 } ( 5 . 1 e ^ { - 6 } )$ </td><td> $7 1 . 4 ( 9 . 1 e ^ { - 2 } )$ </td><td>100(0)</td></tr><tr><td rowspan="2">Baichuan 2-13B-Base</td><td>CF-0</td><td> $\overline { { 2 . 7 0 e ^ { - 2 } } }$ </td><td> $\overline { { 1 2 . 8 4 e ^ { - 2 } } }$ </td><td>57.14</td><td>100.0</td></tr><tr><td>CF-2</td><td> $1 . 0 e ^ { - 2 } ( 4 . 9 e ^ { - 4 } )$ </td><td> $7 . 5 e ^ { - 2 } ( 4 . 7 e ^ { - 4 } )$ </td><td>40.0(6.2e−4)</td><td>100.0(0)</td></tr><tr><td rowspan="2">Qwen-14B</td><td>CF-0</td><td> $\overline { { 1 . 0 5 e ^ { - 2 } } }$ </td><td> $7 . 9 6 e ^ { - 2 }$ </td><td>33.3</td><td>100.0</td></tr><tr><td>CF-2</td><td> $3 . 1 e ^ { - 2 } ( 1 . 8 e ^ { - 3 } )$ </td><td> $1 1 . 1 e ^ { - 2 } ( 6 . 6 e ^ { - 3 } )$ </td><td> $2 5 . 0 ( 7 . 1 e ^ { - 3 } )$ </td><td>91.7(6.9e−3)</td></tr><tr><td rowspan="2">TigerBot-70B</td><td>CF-0</td><td> $\overline { { 8 . 0 2 e ^ { - 2 } } }$ </td><td> $\overline { { 1 9 . 2 6 e ^ { - 2 } } }$ </td><td>12.5</td><td>75.0</td></tr><tr><td>CF-2</td><td> $4 . 4 e ^ { - 2 } ( 2 . 9 e ^ { - 6 } )$ </td><td> $1 5 . 3 e ^ { - 2 } ( 6 . 6 e ^ { - 5 } )$ </td><td> $2 5 . 0 ( 6 . 9 e ^ { - 3 } )$ </td><td> $8 3 . 3 ( 6 . 1 e ^ { - 3 } )$ </td></tr><tr><td rowspan="2">Gamma-7B</td><td>CF-0</td><td> $\overline { { 4 . 9 4 e ^ { - 2 } } }$ </td><td> $\overline { { 1 7 . 6 2 e ^ { - 2 } } }$ </td><td>15.79</td><td>47.36</td></tr><tr><td>CF-2</td><td> $1 0 . 2 e ^ { - 2 } ( 3 . 2 e ^ { - 5 } )$ </td><td> $2 5 . 9 e ^ { - 2 } ( 1 . 6 e ^ { - 4 } )$ </td><td> $2 6 . 4 ( 8 . 6 e ^ { - 4 } )$ </td><td> $5 8 . 8 ( 1 . 4 e ^ { - 2 } )$ </td></tr><tr><td rowspan="2">Falcon-180B</td><td>CF-0</td><td> $5 . 0 0 e ^ { - 2 }$ </td><td> $\overline { { 1 7 . 9 1 e ^ { - 2 } } }$ </td><td>14.58</td><td>57.14</td></tr><tr><td>CF-2</td><td> $3 . 2 e ^ { - 2 } ( 2 . 1 e ^ { - 5 } )$ </td><td> $1 0 . 4 2 e ^ { - 2 } ( 7 . 8 e ^ { - 5 } )$ </td><td> $2 3 . 9 4 ( 8 . 5 e ^ { - 2 } )$ </td><td> $6 3 . 6 ( 2 . 1 e ^ { - 5 } )$ </td></tr><tr><td rowspan="2">Gopher-280B</td><td>CF-0</td><td> $\overline { { 1 4 . 4 8 e ^ { - 2 } } }$ </td><td> $3 0 . 7 6 e ^ { - 2 }$ </td><td>15.38</td><td>61.53</td></tr><tr><td>CF-2</td><td> $1 0 . 8 7 e ^ { - 2 } ( 3 . 6 e ^ { - 5 } )$ </td><td> $2 3 . 5 9 ( 4 . 2 e ^ { - 4 } )$ </td><td> $2 7 . 3 3 ( 1 . 8 e ^ { - 3 } )$ </td><td>66.49(6.8e−3)</td></tr></table>

Table 4: The accuracy of Predicting Scaled Large LMs in CPP-0, CPP-2.

• ${ \cal { S } } \cup \{ i \} )$ is the model’s output when the factors in subset S plus factor i are used.

• The factorial terms S ! and $( | N | - | S | - 1 ) !$ ! weigh the contribution of each subset according to the number of factors included or excluded, ensuring a fair allocation across all possible combinations.

The Shapley value, $\phi _ { i } ( v )$ , quantifies the average marginal contribution of a factor i across all possible combinations of factors. The formula takes every subset S of the total factor set N that does not include i, calculates the difference in the model’s prediction output with and without factor i and averages this difference over all subsets. The averaging is weighted by the factor $\frac { | S | ! ( | N | - | S | - 1 ) ! } { | N | ! }$ which corresponds to the number of permutations in which subset S appears as a prefix or suffix of the total set when factor i is added.

Number of tested models on each task  
![](images/73cb9dc0ade826c6dffd31b10894aeb6a0728d6a1257c9ce77c3d6447f431692.jpg)

Number of tested tasks per model  
![](images/09ab7f68ead6a29ff05b1cf18a92561d45ba3f7347d1905d461344184d706ae0.jpg)

Number of Models Containing the Factors  
![](images/98451caec2fd1b315e62d0fb6b96be56c6d9ad83ca37aa3e211b38e86b609680.jpg)  
Figure 8: The detailed distribution of collaborative data.

This approach ensures that each factor’s contribution is assessed fairly and comprehensively, accounting for interactions with other factors and their unique impact when combined in different ways. Shapley values are particularly useful for factor importance analysis because they provide a solid theoretical foundation and are less biased than simpler importance metrics.

The Shapley value algorithm for analyzing feature (factor) importance is computationally intensive, which has led to the development of various approximation methods (Jethani et al., 2022).

Fortunately, our predictive model involves a manageable number of factors, allowing us to use the most accurate direct computation method of Shapley values. Specifically, we apply an enumeration approach to compute Shapley values on a pretrained factor-enhanced neural collaborative filter-Complex Reasoning and CoT Ting model during the inference stage. This involves systematically masking factors to assess their impact.

For the implementation, we mask factors differently based on their data type as outlined in Table 3:

• numerical factors: we set the input factor values to zero;

S• categorical factors: we set the corresponding <sub>t</sub>uembedding layer parameters to zero.

AWe then compute the difference in validation loss with and without each factor present, providing us with each factor’s marginal contribution. This detailed approach allows us to quantify precisely how much each factor contributes to the model’s predictive performance, providing valuable insights into factor importance and model behavior.

## D Ablation Study

## D.1 Ablation on Sparsity Threshold

To ascertain whether matrices composed of collaborative performance data can accurately predict the performance of LLMs, it is essential to consider the critical variable: the matrix sparsity. We assessed the impact of sparsity on prediction accuracy by manipulating the sparsity of the training matrix via masking. This method allowed us to obtain a reliable measure of average accuracy, as illustrated in Figure. 9. It is noteworthy that our method of controlling sparsity only reduces the number of training samples. We ensured fairness in each comparative experiment by maintaining a consistent validation set throughout. During the experiment, we maintained the same settings for the learning rate and number of iterations as in the main experiment. To ensure the robustness of our experimental results, each reported outcome represents the average score after conducting five random splits.

The data we collected inherently has a sparsity of 44%. Hence, we only have the remaining 46% of collaborative data. As sparsity levels range from 49.60% to 88.80%(masking 10% to 80% of the collaborative data), the graph shows a pronounced increase in L1 Loss and a decrease in Accuracy, indicating deteriorating model performance with higher sparsity, especially when sparsity exceeds 60%, where there is a significant drop in accuracy. Conversely, MAE@2 remains relatively stable before experiencing fluctuations, suggesting varying impacts on this metric. Interestingly, accuracy even improves when sparsity reaches 50%. We think the possible reason for this might be the presence of an optimal level of information reduction that removes redundant or noisy data without significantly compromising signal integrity. This phenomenon suggests that a moderate level of sparsity could potentially enhance model performance by focusing on more relevant factors.Number

![](images/998321a28c0c28a72fea7176d90d09041901c1f56e79e9eaaf388f7e3b2368ae.jpg)  
Figure 9: Relationship between matrix sparsity and three key performance metrics: L1 Loss, Accuracy, and MAE@2.

## D.2 Ablation on Predicting Performance on Complex Reasoning and CoT TasksM<sup>o</sup>

From the model perspective, it is crucial for validat-u<sup>m</sup> ing the feasibility of predictive methodologies to assess the predictive accuracy on special tasks potentially exhibiting “emergent” phenomena (Suzgun et al., 2022; Wei et al., 2022), including complex<sup>Number</sup> <sup>of</sup> <sup>tested</sup> <sup>tasks</sup> <sup>per</sup> <sup>model</sup> reasoning and Chain of Thought (CoT) tasks (Wei et al., 2023). “Emergent’ phenomena refers to the<sub>a</sub>s<sup>k</sup> challenges associated with predicting performanceer <sup>o</sup> from smaller models when the scale of a model ex-N pands significantly, resulting in discontinuous leaps in model capabilities. The existence of this phenomenon is subject to ongoing debate. Nonetheless, recent efforts (Ganguli et al., 2022b; Hu et al., 2024; Owen, 2024; Ruan et al., 2024; Schaeffer et al.,<sub>s</sub> 2023) continue to focus on how scaling laws can<sub>f</sub> M<sup>o</sup> be modified to mitigate the “gap” between smaller<sub>m</sub>b<sup>e</sup> and larger models. This may involve modifying metrics or incorporating additional data points to linearize the growth curve or alternatively opting for a sigmoidal curve.

Theoretically, these challenges are not too difficult for our prediction method, as the underlying mechanism of “emergent” abilities reflects a type of similarity. This commonality manifests when models exceed a certain scale. By analyzing cross-model similarities—how other larger models demonstrate emergent capabilities compared to their smaller counterparts—we can enhance our predictive accuracy for the current model. Overall, these tasks are pivotal for comprehensive validation processes, e.g., GSM8K (Cobbe et al., 2021), BBH (Suzgun et al., 2022), HUMANEVAL (Chen et al., 2021) and MBPP (Austin et al., 2021).

In detail, if we want to evaluate the performance of predicting a model on these special tasks, the training data is the performance information from other model families, the smaller model of the same family, and the randomly selected two non-special tasks prior to the performance of this model. In our experiment, we tested the 4 models on these tasks, and then we plotted the test results on Figure 10. As illustrated in Figure 10, our predictive scores are more adaptive to each task, where the points are close along the “perfect prediction” line, which means our prediction method captures the similarity in the specific task across models. Our proposed method’s MSE Loss is comparable to the scaling law, which shows the feasibility of CPP (in CPP-2).

![](images/fce5fb9a981b1bc954c2cc3f6f903e1d32f7c063db2f872d98ea5cfc5e12f45a.jpg)  
Figure 10: Comparison of the predictive performance of collaborative performance prediction (CPP) versus traditional scaling laws (SL) for Large Language Models (LLMs) in Complex Reasoning and CoT Tasks.

Generalization to Completely New Tasks. As presented in Tab. 5, CPP-T0 and CPP-T2 have a relative small error, demonstrating our method CPP shows reliable generalization. When CPP-T2 has the prior performance of two models in this task, it has a significant drop compared to CPP-T0. These two experimental results inspire us that prediction and evaluation should be interactive, i.e., we should evaluate two small models or tasks to get true but low-cost results, and then the accuracy of prediction can be improved after obtaining the results.

## D.3 Correlation between Models

Experiment. We conducted a “leave-one-out” experiment to test the impact of Model A on the predictive performance of Model B. This involved masking Model A and using the performance of other models to train predictive methods, which were then validated on Model B to observe changes in loss. This approach generated a matrix with the masked model names on the X-axis and the validation model names on the Y-axis, with the values representing the change in loss.

The “Leave-one-out” experiment is a robust method commonly used in statistical analysis. To assess the impact of different models on the predictive performance of a specific model, we implemented a strategy where we systematically masked each selected model in the training set. The procedure involved masking each model individually and then training and testing the loss on a validation model. This process was repeated across all models, culminating in creating a matrix where axis=0 represents the masked model ID, and axis=1 represents the validation model ID. The values in the matrix correspond to the loss observed. This experiment was conducted under three different random seeds to ensure the stability and reliability of the results.

Subsequently, each model was used as a validation set, with the remaining data serving as the training set to calculate the loss for each model. This also resulted in a matrix where axis=1 indicates the validation model ID, and the columns[:, valid model id] represent the corresponding loss for that validation model. We derived a delta loss matrix by calculating the difference between these two matrices.

Given that each validation model has its own range of loss variations, we normalized the delta loss matrix. We then performed a row-based correlation analysis on this normalized matrix to assess each model’s impact on predictive performance. The higher the correlation value between the two models, their effects on predictions are more similar.

Analysis. Based on this correlation matrix, we further conducted a hierarchical clustering analysis (Nielsen, 2016). The results indicate that a set of models exists that are similar in their impact on the predictive performance of the model. Other models are far away from them. (Details in Table 6)

This analysis not only helps us understand each model’s specific contributions to predictive performance but also reveals the similarities and differences in functionality among the models, providing a crucial basis for model optimization and selection.

We performed a row-wise correlation analysis 13 on this matrix and discovered that models from the same family tend to have similar impacts on predictions, as do models of the same size. After conducting a hierarchical distance analysis, we concluded that a group of models exists that, when more performance data is available, can significantly enhance the accuracy of the predictive models. There are also what might be termed “noise model performances” in our analysis D.3.

![](images/645405d1c70c01379bdebdda61fb129b5110ab8272e746928b6b97720d814c2f.jpg)  
Figure 11: Instance Distribution of the model factor Shapley value. X-axis represents the Shapley value, which indicates the degree of prediction loss change; Y-axis indicates the factor names in order of importance from top to bottom. Each point represents an instance.

## D.4 Correlation between Tasks

We also conducted “leave-one-out” experiments on these tasks and created a heatmap figure. 14 of the correlations. Tasks with similar targeted ability testing capabilities demonstrated similar influences, such as GSM8K, MATH (Hendrycks et al., 2021), ARC (Chollet, 2019), and HUMANEVAL, which all require complex reasoning abilities.

Table 5: The predictive performance (MSE) of CPP in the predictions of the completely new task. Here, CPP-T0 refers to the predictive performance of CPP in the predictions of the completely new task, and CPP-T2 refers to the predictive performance of CPP in the predictions of the task when we only know two models’ performance on this task, indicating CPP has no prior knowledge and few cases.
<table><tr><td>Models</td><td>BoolQ(0-shot)</td><td>BIG-bench hard(3-shot)</td><td>HellaSwag(10-shot)</td><td>HumanEval(pass@1)</td></tr><tr><td>CPP-T0</td><td>0.02201</td><td>0.07103</td><td>0.03414</td><td>0.1244</td></tr><tr><td>CPP-T2</td><td>0.0182</td><td>0.00725</td><td>0.02506</td><td>0.0763</td></tr></table>

![](images/71f781fb457cbe1c2055dcfb5b9f4e48b0d52e1be285f02dd853dd3224124f41.jpg)  
Shapley Value (by prediction Loss)  
Figure 12: Instance Distribution of the task factor Shapley value. X-axis represents the Shapley value, which indicates the degree of prediction loss change; Y-axis indicates the factor names in order of importance from top to bottom. Each point represents an instance.

## E Others

## E.1 Visualization

The figure 15 is the visualization for the prediction performance of scaled language models on downstream tasks.

![](images/1020a5afb95bd850d4c28d9c324266211eeb2c5e0230126855c58ec5293f83e7.jpg)  
Figure 13: The correlation heatmap of impacts of different models on prediction performance.

![](images/5bc10d7d51638cf1b02370ace960af35e927c0916d83e0ee0442b6c7fc32283f.jpg)  
Figure 14: The correlation heatmap of impacts of different tasks on prediction performance.

![](images/a003317f57a3acba13211e629f8fac7750c00b91b5461909bc77bccede1d5717.jpg)  
Figure 15: Prediction performance of various scaled Language Models on downstream tasks. This figure illustrates regression plots comparing the predicted versus actual performance normalized scores for a series of large language models, including Llama-2-70B, Llama-65B, Falcon-180B, Gopher-280B, Pythia-12B, Gemma-7B, TigerBot-70B, Qwen-14B, Luminous Supreme 70B, and Llama-3-70B. Each subplot displays a regression line with a shaded 95% confidence interval and includes the L1 loss for each model’s predictions, highlighting the accuracy and variability of predictive capabilities across different models.

<table><tr><td rowspan=1 colspan=1>Distance Cluster</td><td rowspan=1 colspan=1>Models</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>LLama-2-7B, LLama-2-13B, LLama-2-70B, Llama 3 8B,LLaMA-7B, LLaMA-65B, Claude-V3 Haiku, Claude-V3 Sonnet,Claude-V3 Opus, GPT-4, BLOOM-176B, Luminous Extended-30B,Luminous Supreme-70B, OPT-175B, GPT-NeoX-20B, sheared llama-2.7B,sheared llama-1.3B, INCITE-Base-3B, INCITE-Base-7B, OpenLLaMA-3B-v1, Pythia-1.4B,Pythia-2.8B, Pythia-70M, Pythia-410M, Pythia-6.9B,Gopher - 280B, Gopher - 44M, Gopher - 117M, MT-NLG 530B, GLaM,Baichuan 1-7B, Baichuan 1-13B-Base, Baichuan 2-7B-Base, Baichuan 2-13B-Base,Skywork-13B, Qwen-7B, Qwen-14B, TigerBot-13b,Gemma-2b, Gemma-7b</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>gpt-3.5, Falcon-7B, Pythia-1B, Gropher - 1.4B, Yi-9b, TigerBot-70b</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>LLaMA-33B</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Yi-6b</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>BlueLM-7B</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Falcon-40B</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>MPT-7B</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Falcon-180B</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>PaLM-540B</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Pythia-160M</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>GPT-J-6B</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>GPT-3-175B, Luminous Base-13B</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>Gopher - 417M</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>Llama 3 70B</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>LLaMA-13B</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>TinyLlama-1.1B</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>Phi-1.5-1.3B</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>Gopher - 7.1B</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>InternLM2-20B</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>GLM-130B</td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>MPT-30B</td></tr><tr><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>chinchilla</td></tr><tr><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>Mistral 7B</td></tr><tr><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>InternLM2-7B</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>OpenLLaMA-3B-v2</td></tr><tr><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>Phi-2-2.7B</td></tr><tr><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>Pythia-12B</td></tr></table>

Table 6: Distance Cluster of Models