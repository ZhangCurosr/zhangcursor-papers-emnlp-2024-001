# MetaGPT: Merging Large Language Models Using Model Exclusive Task Arithmetic

Yuyan Zhou∗ Baichuan Inc.

Liang Song\* Baichuan Inc.

Bingning Wang<sup>†</sup> Baichuan Inc.

Weipeng Chen Baichuan Inc.

## Abstract

The advent of large language models (LLMs) like GPT-4 has catalyzed the exploration of multi-task learning (MTL), in which a single model demonstrates proficiency across diverse tasks. Task arithmetic has emerged as a costeffective approach for MTL. It enables performance enhancement across multiple tasks by adding their corresponding task vectors to a pre-trained model. However, the current lack of a method that can achieve optimal performance with low computational cost and protecting the data privacy, which limits their application to LLMs. In this paper, we propose Model Exclusive Task Arithmetic for merging GPT-scale models (MetaGPT), which formalizes the objective of model merging into a multi-task learning framework, aiming to minimize the average loss difference between the merged model and each individual task model. Since data privacy limits the use of multi-task training data, we leverage LLMs’ local linearity and task vectors’ orthogonality to separate the data term and scaling coefficients term and derive a model-exclusive task arithmetic method. Our proposed MetaGPT is dataagnostic and bypasses the heavy search process, making it cost-effective and easy to implement for LLMs. Extensive experiments demonstrate that MetaGPT leads to improvements in task arithmetic and achieves state-of-the-art performance on multiple tasks.

## 1 Introduction

In recent years, a well-established paradigm for AI has been to pre-train models using large-scale datasets and then to fine-tune the models on different tasks through supervised learning with taskspecific datasets, which can lead to improved performance while requiring less labeled data (Devlin et al., 2018; OpenAI, 2023; Dodge et al.,

![](images/99a7c5923ea7cb98bc7fbbeb361548fd5a1a1f25617c35a776f25081edbf46b2.jpg)  
Figure 1: Existing methods face the trilemma of performance, data privacy, and computational costs, which hinders its application to LLMs. Our MetaGPT can solve these problems under careful approximation and thus can scale to GPT3-scale LLMs.

2020). However, for each new application, a separate model has to be fine-tuned and deployed, which is computationally expensive and resourceintensive (Fifty et al., 2021; Zhang and Yang, 2021). Thus, Multi-Task Learning (MTL) methods have been proposed and developed to enable a single model to solve multiple tasks concurrently.

Conventional MTL approaches typically involve collecting raw data across multiple tasks and then jointly training a single model (Caruana, 1997; Yang et al., 2023a). However, the fine-tuning process becomes extremely computationally intensive with the development of large language models (LLMs) that may comprise billions or even trillions of parameters. Therefore, researchers have explored merging various task-specific models with the expectation that the merged model can handle multiple tasks simultaneously.

One of the outstanding merging methods is task arithmetic (Ilharco et al., 2023). For a given task, the element-wise difference between the weights of the pre-trained model and the fine-tuned model is referred to as the task vector. Recent studies have shown that linearly adding multiple scaled task vectors to the pre-trained model can improve performance across those tasks (Ilharco et al., 2023; Yang et al., 2023b). Nevertheless, previous task arithmetic methods face a trilemma in practice. 1) The best-performing task arithmetic methods require extra training to obtain optimal hyper-parameters, but the high computational costs hinder their application to GPT3-scale LLMs. 2) Some training-free methods heuristically set the scaling coefficient to a constant (e.g., 0.3), which is efficient but leads to sub-optimal performance. 3) Some methods conduct grid search on the training/validation set, which is sometimes impractical and faces the risk of data privacy concerns. In summary, as illustrated in Figure 1, there is essentially no task arithmetic method suitable for billion-scale models that perform satisfactorily in practice.

To address the aforementioned problems, in this paper, we propose MetaGPT: an optimal and efficient task arithmetic method for MTL without any data (model exclusive task arithmetic). We begin by providing a detailed theoretical analysis of the task loss difference and average loss difference introduced by the task arithmetic algorithm. Since we aim to choose parameters that minimize the average loss difference, we first separate the data term and scaling coefficients, which also establishes a performance upper bound for task arithmetic. After separating the scaling coefficients, the final result is quadratic for each scaling coefficient, leading to a closed-form solution that is simple and effective to implement.

The experimental results on the LLaMA-2 (Touvron et al., 2023) and Mistral (Jiang et al., 2023) series demonstrate that the MetaGPT approach is superior to previous merging methods on several tasks. MetaGPT provides an efficient avenue to optimally implement task arithmetic for large-scale multi-task learning (MTL) and push the frontiers of language model merging. To sum up, our contributions include:

1. We provide the mathematical formulation of the optimization objective for task arithmetic and the first theoretical analysis of the performance bound for task arithmetic.

2. To achieve optimal performance under the setting of efficient and protecting data privacy, we separate the data term and scaling coefficients in the optimization objective, which leads to a closed-form solution for the scaling coefficients.

3. Our MetaGPT is orthogonal to existing task vector-improving methods and can be integrated to achieve higher performance.

4. Extensive experiments demonstrate that our MetaGPT can improve task arithmetic and achieve state-of-the-art performance.

## 2 Related Work

Model Merging. Currently, model merging has been developed for multiple uses such as improving performance on a single target task (Izmailov et al., 2018; Wortsman et al., 2022), improving outof-domain generalization (Ramé et al., 2023; Cha et al., 2021; Arpit et al., 2022), and improving the performance of multi-task learning (Ilharco et al., 2023; Yadav et al., 2024; Yu et al., 2023), which is the core focus of our research. The range of applications has led to a proliferation of methods to improve beyond simple parameter averaging. Fisher merging (Matena and Raffel, 2022) tries to weight the importance of individual models using Fisher Information Matrix and uses it to merge different models. RegMean (Jin et al., 2022) formulate the merging problem as a regression problem and leads to an optimal solution for linear models. Task Arithmetic (Ilharco et al., 2023) presents a method for merging models by adding task vectors to the pretrained model to improve multi-task performance. Ties Merging (Yadav et al., 2024) and DARE (Yu et al., 2023) propose to refine the task vectors by resolving the interference and removing extremely redundant components. Ortiz-Jimenez et al. (2024) propose that fine-tuning the models in their tangent space can amplify weight disentanglement and lead to substantial performance improvements.

Multi-Task Learning. Multi-task learning is a powerful method for solving multiple correlated tasks simultaneously (Caruana, 1997). Current MTL works mainly focus on learning the shared representations from designing specific architecture (Misra et al., 2016; Sun et al., 2020) or using specific optimization methods (Sener and Koltun, 2018; Liu et al., 2021). The former focuses on learning the shared representation using different methods such as designing specific representation sharing module (Liu et al., 2019; Ding et al., 2021), learning to branch (Lu et al., 2017; Guo et al., 2020), and based selection criteria (Ma et al., 2018; Hazimeh et al., 2021). And the latter focuses on balancing multiple tasks from the perspectives of task training weights (Sener and Koltun, 2018; Liu et al., 2019), gradient dominance (Chen et al., 2018; He et al., 2022; Yang et al., 2023a), and solving gradient conflicts (Yu et al., 2020; Chen et al., 2020; Liu et al., 2021). However, the conventional MTL approaches for collecting raw data across multiple tasks for joint training are not suitable for LLMs. The factors contributing to this issue are twofold: first, computational inefficiency due to the substantial computational costs associated with updating pre-trained models; second, a significant number of data proprietors are reluctant to disclose valuable or privacy-sensitive raw data.

## 3 Preliminaries

## 3.1 Notation

Let $f : X \times \Theta $ be a neural network taking inputs $x \in \chi$ and parameterized by a set of weights $\pmb { \theta } \in \Theta$ . We assume $\mathcal { X } \subseteq \mathbb { R } ^ { p } , \Theta \subseteq \mathbb { R } ^ { m }$ and $\mathcal { Y } \subseteq \mathbb { R } ^ { q }$ . We consider fine-tuning a pre-trained model $f ( \cdot , \pmb \theta _ { 0 } )$ on � different tasks, with each task � consisting of a triplet $( \mathcal { D } _ { t } , \mathcal { L } _ { t } , \pmb { \theta } _ { t } )$ , where $\mathcal { D } _ { t } ~ = ~ ( \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } , \mathcal { D } _ { t } ^ { \mathrm { v a l } } , \mathcal { D } _ { t } ^ { \mathrm { t e s t } } )$ is the training, validation and test data of task $t , \mathcal { L } _ { t }$ is the loss function of task $t ,$ and $\pmb { \theta } _ { t }$ is the model parameters fine-tuned on task � based on the pre-trained weight $\pmb { \theta } _ { 0 }$

## 3.2 Task Arithmetic

Let the task vector of task � be the difference between the fine-tuned and the pre-trained weights:

$$
\pmb { \tau } _ { t } = \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } .\tag{1}
$$

Task arithmetic aims to solve the multi-task learning problem by directly adding the scaled task vectors to the pre-trained model weight $\pmb { \theta } _ { 0 }$ :

$$
\pmb { \theta } _ { \mathrm { f i n a l } } = \pmb { \theta } _ { 0 } + \sum _ { i = 1 } ^ { T } \lambda _ { i } \pmb { \tau } _ { i }\tag{2}
$$

where $\lambda _ { i }$ is the scaling coefficient of task vector $\tau _ { i } . \mathrm { A s }$ illustrated in Eq. 2, the task arithmetic introduces � hyper-parameters $\{ \lambda _ { i } | i = 1 , \cdots , T \}$ and the choice of these scaling coefficients has a significant influence on the performance of the merged model. Thus, selecting the appropriate scaling coefficients for different task vectors remains a challenging problem.

## 3.3 Existing Methods

Earlier task arithmetic (Ilharco et al., 2023; Yadav et al., 2024) propose to perform a grid search (G-Task Arithmetic) on the validation set to choose the optimal scaling coefficients. However, as the number of tasks increases, exploring all the scaling coefficient combinations faces the curse of dimensionality. Therefore, to simplify the problem, they use the same value for multiple scaling coefficients, thereby reducing the computational complexity. In the absence of the training/validation data, they set $\lambda = 0 . 3$ as the default setting for dataless arithmetic. Moreover, Adamerging (Yang et al., 2023b) aims to autonomously learn the coefficients from unlabeled test samples using entropy minimization.

## 3.4 Scalability Challenges for LLMs

The methods mentioned above are not suitable for scaling to LLMs: The grid search method requires extra validation/training data, which faces the risk of data privacy concerns and the curse of dimensionality when the number of tasks increases. For instance, conducting a grid search for three hyperparameters, each with a discretization interval of 0.01, would require $1 0 ^ { 6 }$ forward passes across the entire dataset. Setting a fixed value such as 0.3 for all the $\lambda _ { i }$ is time-efficient and can be applied to LLMs, but it leads to sub-optimal performance. Using test data input to unsupervised optimize these hyper-parameters can lead to an optimal solution but requires extra data and necessitates loading multiple models for training. This process is both time and memory consuming, making it challenging to apply to LLMs. For example, merging three LLMs requires loading three LLMs simultaneously to optimize, which is extremely costly. The statement above suggests that scaling up existing optimal task arithmetic to LLMs remains a challenging problem.

## 4 Our Proposed MetaGPT

## 4.1 Overview

To solve the problems above, we propose a new algorithm MetaGPT, based on careful approximations to a closed-form solution, which easily scales to giant models both in terms of runtime as well as performance while protecting data privacy. In this section, we state the motivation and optimization problem and solve it step by step. All proofs of lemmas and theorems are provided in the appendix.

![](images/0cf41d87fbe740fba9d5b59d32d461830d7fabec771028fcadf0ebfbb18dd198.jpg)  
Figure 2: Current task arithmetic based methods face the problems of sub-optimal performance, huge computational and memory cost, curse of dimensionality and data privacy, which makes it difficult to scale to LLMs. Our method solves the aforementioned problems and provides an avenue to scale task arithmetic to LLMs.

## 4.2 MetaGPT Optimization Objective

Definition 1 (Single Task Loss Difference). For the fine-tuned model $\pmb \theta _ { i }$ and the task arithmetic merged model $\pmb { \theta } _ { \mathrm { f i n a l } }$ . The Task Loss Difference in task � (TLD<sub>�</sub>) is defined as:

$$
\begin{array} { r l } & { \mathrm { T L D } _ { t } ( \lambda _ { 1 } , \cdots , \lambda _ { T } , \tau _ { 1 } , \cdots , \tau _ { T } ) } \\ & { \phantom { \mathrm { T L D } _ { t } ( \lambda _ { 1 } , \cdots , \lambda _ { T } , \tau _ { 1 } , \cdots , \tau _ { T } ) } = \mathcal { L } _ { t } ( \pmb \theta _ { \mathrm { f i n a l } } , \pmb x ) - \mathcal { L } _ { t } ( \pmb \theta _ { t } , \pmb x ) \mathrm { . } } \end{array}\tag{3}
$$

It is obvious that smaller $\mathrm { T L D } _ { t }$ suggests that the loss of the merged model is close or even lower than the fine-tuned model on task �, which indicates a better task arithmetic performance.

However, for task arithmetic, it aims to improve the average performance of the final model on all the tasks. Thus, we define the average of all the task loss differences as Average Loss Difference (ALD), which can be formulated as follows:

Definition 2 (Average Task Loss Difference). For the fine-tuned models $\{ \pmb { \theta } _ { i } | i = 1 , \cdots , T \}$ and task arithmetic merged model $\pmb { \theta } _ { \mathrm { f i n a l } }$ . The average loss difference for all tasks is defined as:

$$
\begin{array} { r l } & { \displaystyle \mathrm { A L D } ( \lambda _ { 1 } , \cdot \cdot \cdot , \lambda _ { T } , \tau _ { 1 } , \cdot \cdot \cdot , \tau _ { T } ) } \\ & { \displaystyle \quad \quad = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mathcal { L } _ { t } ( \pmb \theta _ { \mathrm { f i n a l } } , \pmb x ) - \mathcal { L } _ { t } ( \pmb \theta _ { t } , \pmb x ) \right) . } \end{array}\tag{4}
$$

Thus, the optimization objective of MetaGPT is to find the optimal scaling coefficients that can minimize the ALD, which can be formulated as:

Definition 3 (Optimization objective of MetaGPT). Our MetaGPT aims at finding the scaling coefficients $\{ \lambda _ { i } | i = 1 , \cdots , T \}$ , which minimizes the average loss difference ALD:

$$
\operatorname * { a r g m i n } _ { \lambda _ { 1 } , \cdots , \lambda _ { T } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mathcal { L } _ { t } ( \theta _ { \mathrm { f i n a l } } , \pmb x ) - \mathcal { L } _ { t } ( \pmb \theta _ { t } , \pmb x ) \right) .\tag{5}
$$

## 4.3 Separating Data and Coefficients

Before analyzing ALD, we start with reformulating $\mathrm { T L D } _ { t }$ by its Taylor expansion.

Lemma 4. Using Taylor expansionfor $\mathcal { L } ( \pmb \theta _ { \mathrm { f i n a l } } , \pmb x )$ at $\pmb { \theta } _ { t }$ , the TLD in Eq. 3 can be reformulated as a quadraticform with respect to the linear combination of� and �:

$$
\mathrm { T L D } _ { t } = \frac { 1 } { 2 } \pmb { h } _ { t } ^ { \top } \left( \int _ { 0 } ^ { 1 } \nabla ^ { 2 } \mathcal { L } _ { t } ( \gamma _ { t } ( \beta ) ) d \beta \right) \pmb { h } _ { t } ,\tag{6}
$$

where $\gamma _ { t } ( \beta ) = \pmb { \theta } _ { t } + \beta ( \pmb { \theta } _ { \mathrm { f i n a l } } - \pmb { \theta } _ { t } )$ and $\mathbf { } _ { \mathbf { } ^ { h _ { t } } }$ is the linear combination of� and �:

$$
\pmb { h } _ { t } = \sum _ { k \neq t } \lambda _ { k } ( \pmb \theta _ { k } - \pmb \theta _ { 0 } ) - ( 1 - \lambda _ { t } ) ( \pmb \theta _ { t } - \pmb \theta _ { 0 } ) .\tag{7}
$$

Single $\mathrm { T L D } _ { t }$ is associated with the data, models, and scaling coefficients. As we can see in Eq. 6, we have transformed the data term $\boldsymbol { x } _ { t }$ to the Hessian, the coefficients $\lambda = [ \lambda _ { 1 } , \cdots , \lambda _ { T } ]$ and models term $[ \pmb { \theta } _ { 1 } , \cdots , \pmb { \theta } _ { T } ]$ to �. As our method tends to achieve model-exclusive task arithmetic, the final result should not correlate with the data term. Thus, we first provide a property, which will be used latter in our theorem proofs to separate the data term and scaling coefficients and models term. In general, if a pre-trained network $f ( \cdot ; \pmb { \theta } _ { 0 } )$ demonstrates kernel behavior during fine-tuning, i.e., fine-tuning occurs in the linear regime, the following property must be satisfied (Jacot et al., 2018):

Property 5 (NTK linearization). Around the initialization weights $\pmb { \theta } _ { 0 } ,$ , a neural network can be approximated with a linear approximation:

$$
f ( \pmb { x } ; \pmb { \theta } _ { 0 } + \alpha ( \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } ) ) \approx f ( \pmb { x } ; \pmb { \theta } _ { 0 } ) + \alpha \cdot C .\tag{8}
$$

where $\boldsymbol { C } = ( \mathbf { \nabla } \mathbf { \theta } \mathbf { \theta } \mathbf { \theta } - \mathbf { \theta } \mathbf { \theta } \mathbf { \theta } ) ^ { \top } \nabla f ( \mathbf { x } , \pmb { \theta } _ { 0 } )$ is a data and model dependent constant.

It is worth noting that, as the network width approaches infinity, Eq. 8 becomes exact and remains valid throughout training (Jacot et al., 2018; Arora et al., 2019; Lee et al., 2019), which is specifically suitable for the LLMs arithmetic scenario.

The second property is observed by (Ilharco et al., 2023), which states that the different task vectors are orthogonal:

Property 6 (Orthogonality of Task Vectors). For task vector $\tau _ { i } = \theta _ { i } - \theta _ { 0 }$ and $\tau _ { j } = \pmb { \theta } _ { j } - \pmb { \theta } _ { 0 } ( i \neq j )$ we have the following equation:

$$
\pmb { \tau } _ { i } ^ { \top } \pmb { \tau } _ { j } = ( \pmb { \theta } _ { i } - \pmb { \theta } _ { 0 } ) ^ { \top } ( \pmb { \theta } _ { j } - \pmb { \theta } _ { 0 } ) = 0 .\tag{9}
$$

Now, as we previously introduce our first Lemma to transform the $\mathrm { T L D } _ { t }$ in Eq. 3 into a quadratic form with respect to the linear combination of � and �. Next, using Property 5,6 and Lemma 7, we can upper bound the $\mathrm { T L D } _ { t }$ and separate the data term and scaling coefficients and models term.

Theorem 7. The $\mathrm { T L D } _ { t }$ can be upper bounded by:

$$
\begin{array} { r l } & { \displaystyle \mathrm { T L D } _ { t } ( \lambda _ { 1 } , \cdots , \lambda _ { T } , \tau _ { 1 } , \cdots , \tau _ { T } ) } \\ & { \le \frac { \delta _ { t } ^ { 2 } } { 2 } \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| _ { 2 } ^ { 2 } \bigg \{ \displaystyle \sum _ { k \neq t } ^ { T } \mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } ) \| \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \| ^ { 2 } \bigg \} , } \end{array}\tag{10}
$$

where $\delta _ { t }$ is a data-dependent constant and we use $\mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } )$ to denote $( \lambda _ { k } ^ { 2 } ) \mathbb { 1 } \left( k \neq t \right) + ( 1 - \lambda _ { k } ^ { 2 } ) \mathbb { 1 } \left( k = t \right)$

Now, after separating the data-related term to $\delta _ { t }$ the scaling coefficients and models term to $\mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } )$ By summing all the $\mathrm { T L D } _ { t } \mathbf { s }$ , we can separate the two terms for ALD:

Theorem 8. By summing all the $\mathrm { T L D } _ { t }$ , we can separate the correlation between data term and scaling coefficients term in ALD:

$$
\operatorname { A L D } ( \lambda _ { 1 } , \cdot \cdot \cdot , \lambda _ { T } , \tau _ { 1 } , \cdot \cdot \cdot , \tau _ { T } )\tag{11}
$$

$$
\leq \sum _ { t = 1 } ^ { T } \delta _ { t } ^ { 2 } \left. \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \right. _ { 2 } ^ { 2 } \left\{ \sum _ { k \neq t } ^ { T } \mathbb { 1 } \left( \lambda _ { k } ^ { 2 } \right) \left. \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \right. ^ { 2 } \right\} ,
$$

## 4.4 The Optimal Solution

After separating the data term and the scaling coefficients term, we can now reformulate our optimization objective Eq. 11 and derive the closed-form optimal solution of the scaling coefficients.

Theorem 9 (� decomposition of ALD). For each $\lambda _ { t }$ , we use it to decompose Eq. 11 as:

$$
\mathrm { A L D } \leq \sum _ { t = 1 } ^ { T } \mathrm { A L D } _ { \lambda _ { t } } ,\tag{12}
$$

where $\mathrm { \ A L D } _ { \lambda _ { t } }$ is:

$$
\mathrm { A L D } _ { \lambda _ { t } } = \frac { \delta _ { 0 } ^ { 2 } } { 2 } \Vert \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \Vert ^ { 2 } \left[ \sum _ { k = 1 } ^ { T } \mathbb { 1 } _ { t } ( \lambda ) \Vert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \Vert ^ { 2 } \right] ,\tag{13}
$$

where $\delta _ { 0 } = \operatorname* { m a x } _ { t } \delta _ { t }$ . The equation above easily leads to a model-exclusive closed-form solution:

Theorem 10 (Optimal Scaling Coefficients). We can solve $\lambda _ { t }$ form Eq 13 by:

$$
\lambda _ { t } = \underset { \lambda _ { t } } { \arg \operatorname* { m i n } } \ \lVert \pmb { \theta _ { t } } - \pmb { \theta _ { 0 } } \rVert ^ { 2 } \left[ \sum _ { k = 1 } ^ { T } \mathbb { 1 } _ { t } ( \lambda ) \lVert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \rVert ^ { 2 } \right] .\tag{14}
$$

The above equation is quadratic on $\lambda _ { t }$ and the optimal solutionfor $\lambda _ { t }$ is:

$$
\lambda _ { t } = \frac { \| \pmb \theta _ { t } - \pmb \theta _ { 0 } \| ^ { 2 } } { \sum _ { k = 1 } ^ { n } \| \pmb \theta _ { k } - \pmb \theta _ { 0 } \| ^ { 2 } } .\tag{15}
$$

![](images/d51e2cde5af8edc67bccd7535ab813ff24d208122cb4ec82ec8218f2d83b869e.jpg)  
Figure 3: Verification of NTK linearization. We randomly sampled the outputs of Llama-2-7b-chat-hf with different �. We can see that the sampled outputs are linearly with � as expected.

## 5 Property Verification

In Section 4, we introduced two properties essential to our proof. In this section, we conduct experiments to verify these properties.

## 5.1 NTK Linearization

Jacot et al. (2018) have proved that when the width of the neural network approaches infinity, it demonstrates kernel behavior and the optimization proceeds in the linear regime. We test Llama-2-7b-chat-hf (Touvron et al., 2023) on AGIEval (Zhong et al., 2023) dataset to verify its linearity. We have randomly sampled three outputs of the Llama-2-7b-chat-hf when � in Eq. 8 gets value of 0, 0.1,    , 1 . For better visualization, we also subtract all the outputs using max �<sub>�</sub> , ensuring they have the same endpoint. From the results in Figure 3, we can see that all the outputs are almost linear with �, which indicates that LLMs do exhibit a kernel behavior during finetuning.

## 5.2 Task Vector Orthogonality

Ilharco et al. (2023); Yang et al. (2023b) have performed experiments to verify this property for vision models. For LLMs, we also observe similar results: these task vectors are almost orthogonal to each other. The result has been shown in Figure 4. We can see that different task vectors are almost orthogonal, and their cosine similarity is nearly 0 as Eq.9 expected, which verifies the property we have used for our proof.

![](images/26a87d90af9917528fd2cdeb8f5a912525ae8b5bb760a6fc8eb6b64fb6668db5.jpg)  
Figure 4: Verification of orthogonality. We calculate the cosine similarity between six different task vectors and find that their cosine similarity is nearly 0.

## 6 Experiments

In this section, we conduct experiments to demonstrate the effectiveness of our MetaGPT. In the first section, we demonstrate that our MetaGPT consistently achieves optimal average performance across diverse datasets and is robust for model series with varying parameter sizes and architectures. DARE and Ties-Merging are task vector-improving methods that resolve conflicts and redundant parameters between task vectors. We conduct experiments to demonstrate that our method is orthogonal to theirs and can be integrated to improve the average performance further. Finally, we show that the model merged by our MetaGPT has better out-ofdistribution generalization ability.

## 6.1 Merging Models Using MetaGPT

Dataset and Models. To test the effectiveness of our method, we use Llama-2-7b-chat-hf (Touvron et al., 2023), MAmmoTH-7B (Yue et al., 2023) and llama-2-coder-7b (Manuel Romero, 2023) as models fine-tuned on general knowledge, math, and code datasets using the pre-trained model Llama-2- 7B-hf (Touvron et al., 2023). Moreover, we use a different model architecture: Mistral-7B-Instructv0.2 (AI), MAmmoTH2-7B-Plus (Yue et al., 2024) and Mistral-7B-codealpaca-lora (Nondzu) as models fine-tuned on general knowledge, math, and code datasets using pre-trained model Mistral 7B (Jiang et al., 2023). We also provide experiments using models with larger sizes: Llama-2- 13b-chat-hf (Touvron et al., 2023), MAmmoTH-13B (Yue et al., 2023), and llama-2-13b-codechat (TA¸SAR, 2023) as models fine-tuned on general knowledge, math, and code datasets using the pre-trained model Llama-2-13B-hf (Touvron et al., 2023). We use WinoGrande (Sakaguchi et al., 2021) and AGIEval (Zhong et al., 2023) for evaluating general knowledge performance, GSM8K (Cobbe et al., 2021) and MATH (Saxton and Hill, 2019) for testing mathematical reasoning ability, HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) for estimating codegeneration capacity.

Evaluation Metrics. We use common evaluation settings for a single task: 5-shot accuracy for AGIEval, 4-shot accuracy for GSM8K and MATH, 3-shot accuracy for MBPP, and zero-shot accuracy for HumanEval and WinoGrande. We employ two key metrics in evaluating different merging methods: absolute average performance and normalized average accuracy.

Quantitative Evaluation for LLaMA-2-7B. We use the metrics and datasets we introduced above to evaluate the performance of different methods. We use Weight Average (Wortsman et al., 2022), Task Arithmetic (Ilharco et al., 2023), Ties-Merging (Yadav et al., 2024) and DARE (Yu et al., 2023), which are also model exclusive and computationally efficient methods, to compare with our method by merging LLaMA-2-7B. The scores in Table 1 show that for WinoGrande, AGIEval, GSM8k, and MATH dataset, our method scores 64.25, 32.71, 45.41, and 7.80, which outperforms other methods. For the HumanEval dataset, DARE performs best, and for the MBPP dataset, the Weight Average method achieves the highest score. Since our method aims to achieve the average best performance, we use absolute average performance score and normalized average performance score to compare the five methods. We can see that our MetaGPT achieves the rank-1 score 31.51, 1.31 in both absolute average performance and normalized average performance.

<table><tr><td>Model</td><td>WinoGrande AGIEval</td><td></td><td>GSM8k MATH MBPP</td><td></td><td></td><td>HumanEval</td><td>Abs. Avg Nor. Avg</td><td></td></tr><tr><td>LM</td><td>62.67</td><td>34.01</td><td>28.66</td><td>4.00</td><td>22.00</td><td>7.31</td><td>26.44</td><td>0.91</td></tr><tr><td>Math</td><td>61.64</td><td>29.40</td><td>47.16</td><td>2.40</td><td>17.40</td><td>11.58</td><td>28.26</td><td>0.84</td></tr><tr><td>Code</td><td>61.88</td><td>27.41</td><td>17.21</td><td>2.20</td><td>24.80</td><td>21.92</td><td>25.90</td><td>0.84</td></tr><tr><td>Weight Average</td><td>63.93</td><td>31.36</td><td>37.68</td><td>7.00</td><td>23.40</td><td>20.12</td><td>30.58</td><td>1.25</td></tr><tr><td>Task Arithmetic</td><td>63.54</td><td>31.70</td><td>37.53</td><td>5.20</td><td>23.20</td><td>19.51</td><td>30.11</td><td>1.12</td></tr><tr><td>Ties Merging</td><td>62.67</td><td>32.10</td><td>37.93</td><td>7.40</td><td>22.80</td><td>18.29</td><td>30.20</td><td>1.26</td></tr><tr><td>DARE</td><td>63.27</td><td>32.25</td><td>37.86</td><td>7.00</td><td>24.40</td><td>19.51</td><td>30.72</td><td>1.26</td></tr><tr><td>MetaGPT(ours)</td><td>64.25</td><td>32.71</td><td>45.41</td><td>7.80</td><td>21.20</td><td>17.68</td><td>31.51</td><td>1.31</td></tr></table>

Table 1: Performance comparison of merging different LLaMA-2-7B fine-tuned models on different datasets.
<table><tr><td>Model</td><td>WinoGrande AGIEval</td><td></td><td>GSM8k MATH MBPP</td><td></td><td></td><td>HumanEval</td><td>Abs. Avg Nor. Avg</td></tr><tr><td>LM</td><td>69.30</td><td>37.55</td><td>47.54</td><td>7.80</td><td>34.40</td><td>34.75</td><td>38.56 0.776</td></tr><tr><td>Math</td><td>63.46</td><td>38.06</td><td>68.46</td><td>28.00</td><td>24.00</td><td>25.00</td><td>41.16 0.854</td></tr><tr><td>Code</td><td>67.32</td><td>40.69</td><td>60.73</td><td>15.60</td><td>43.40</td><td>39.02</td><td>44.46 0.917</td></tr><tr><td>Weight Average</td><td>67.88</td><td>41.12</td><td>62.77</td><td>17.40</td><td>40.20</td><td>38.41</td><td>44.63 0.921</td></tr><tr><td>Task Arithmetic</td><td>67.88</td><td>41.41</td><td>63.38</td><td>18.80</td><td>40.20</td><td>38.40</td><td>45.01 0.932</td></tr><tr><td>Ties Merging</td><td>67.72</td><td>41.06</td><td>60.35</td><td>17.80</td><td>40.20</td><td>40.24</td><td>44.56 0.924</td></tr><tr><td>DARE</td><td>67.40</td><td>40.58</td><td>59.67</td><td>19.00</td><td>36.00</td><td>40.85</td><td>43.92 0.913</td></tr><tr><td>MetaGPT(ours)</td><td>68.35</td><td>41.86</td><td>66.03</td><td>20.80</td><td>39.00</td><td>35.37</td><td>45.24 0.936</td></tr></table>

Table 2: Performance comparison of merging different Mistral-7B fine-tuned models on different datasets.

Using Different Model Architecture. We also use a different model architecture, Mistral-7B, for evaluation, and the result has been shown in Table 2. The scores have shown similar results to LLaMA-2-7B: For WinoGrande, AGIEval, GSM8k, and MATH dataset, our MetaGPT scores 41.86, 68.35, 66.03, 20.8, which outperforms existing methods, for HumanEval dataset Weight Average, Task Arithmetic, and Ties Merging performs best and for MBPP dataset, DARE method achieves the highest score.

Using Larger Model Size. We also test our method using a larger model LLaMA-2-13B (Touvron et al., 2023). The scores in Table 3 demonstrate that for AGIEval, Math, and MBPP datasets, our method outperforms other methods. For Wino-Grand, GSM8K, and HumanEval dataset, DARE, Weight Average and Ties-Merging achieves the highest score. Similarly, under the average measure absolute average performance and normalized average performance, our method also outperforms the other five methods.

Integrate with Ties/DARE As there are conflicts and redundant parameters between task vectors, DARE (Yu et al., 2023) and Ties-Merging (Yadav et al., 2024) are two methods trying to solve the interfaces, reducing the redundancy and thereby improving the performance of task arithmetic. Since our method is also based on the framework of task arithmetic, Ties-merging and DARE are expected to improve the performance of our MetaGPT further. As we can see in Table 4, under the baseline of Ties-Merging and DARE methods, our method is orthogonal to Ties-Merging and DARE and can integrate them into our MetaGPT, thus leading to further improvement. For example, the average absolute performance of DARE has been improved by our MetaGPT from 30.72 to 31.57. And the normalized absolute performance of DARE has been improved by our MetaGPT from 1.26 to 1.3. Tiesmerging also leads to a similar conclusion: the average absolute performance of DARE has been improved by our MetaGPT from 30.20 to 31.57. And the normalized absolute performance of DARE has been improved by our MetaGPT from 1.26 to 1.33.

<table><tr><td>Model</td><td>WinoGrande AGIEval</td><td></td><td>GSM8K MATH MBPP</td><td></td><td></td><td>HumanEval</td><td>Abs. Avg Nor. Avg</td><td></td></tr><tr><td>LM</td><td>64.80</td><td>35.04</td><td>42.84</td><td>4.80</td><td>27.00</td><td>15.24</td><td>31.62</td><td>1.02</td></tr><tr><td>Math</td><td>60.38</td><td>36.74</td><td>55.27</td><td>3.40</td><td>22.60</td><td>12.80</td><td>31.87</td><td>0.93</td></tr><tr><td>Code</td><td>63.93</td><td>32.04</td><td>36.47</td><td>5.00</td><td>26.60</td><td>16.46</td><td>30.08</td><td>1.01</td></tr><tr><td>Weight Average</td><td>64.88</td><td>37.23</td><td>53.15</td><td>7.60</td><td>29.80</td><td>21.95</td><td>35.77</td><td>1.29</td></tr><tr><td>Task Arithmetic</td><td>65.11</td><td>35.48</td><td>50.34</td><td>7.20</td><td>29.80</td><td>21.95</td><td>34.98</td><td>1.25</td></tr><tr><td>Ties Merging</td><td>65.23</td><td>36.02</td><td>51.23</td><td>7.40</td><td>30.20</td><td>23.17</td><td>35.54</td><td>1.28</td></tr><tr><td>DAREs</td><td>65.70</td><td>36.87</td><td>51.85</td><td>7.60</td><td>30.00</td><td>22.56</td><td>35.76</td><td>1.29</td></tr><tr><td>MetaGPT(ours)</td><td>65.04</td><td>37.33</td><td>52.92</td><td>7.80</td><td>30.40</td><td>21.95</td><td>35.91</td><td>1.30</td></tr></table>

Table 3: Comparison of performance of merging fine-tuned LLaMA-2-13B on different datasets.
<table><tr><td>Method</td><td colspan="8">WinoGrande AGIEval GSM8k MATH MBPP HumanEval |Abs. Avg Nor. Avg</td></tr><tr><td>Ties-Merging</td><td>62.67</td><td>32.10</td><td>37.93</td><td>7.40</td><td>22.80</td><td>18.29</td><td>30.20</td><td>1.26</td></tr><tr><td>Ties + MetaGPT</td><td>62.35</td><td>32.91</td><td>46.10</td><td>8.00</td><td>22.40</td><td>17.68</td><td>31.57</td><td>1.33</td></tr><tr><td>Dare</td><td>63.27</td><td>32.25</td><td>37.86</td><td>7.00</td><td>24.40</td><td>19.51</td><td>30.72</td><td>1.26</td></tr><tr><td>Dare + MetaGPT</td><td>62.99</td><td>33.01</td><td>45.72</td><td>7.60</td><td>21.80</td><td>18.29</td><td>31.57</td><td>1.30</td></tr></table>

Table 4: MetaGPT can be integrated with DARE and Ties-Merging, thereby leading to further improvment.

## 6.2 Out of Distribution Generalization

Following (Yang et al., 2023b; Jin et al., 2022), we also compare the out-of-distribution generalization ability of different merging methods. We evaluate different methods using JEC-QA (Zhong et al., 2020), FinanceIQ (DI, 2023), and MedQA (Jin et al., 2021) dataset. All three datasets use 5-shot accuracy as the evaluation metric. Table 5 summarizes out-of-distribution generalization performance when merging all domain specific models using different methods. As we can see, MetaGPT outperforms current methods on these unseen datasets, which demonstrates that MetaGPT is more robust to the test data distribution shifts.

## 7 Details of different Methods

We give a detailed comparison of the current merging method below from the perspective of extra data information, time complexity, and optimal performance. The time complexity for forward and backward processes is denoted as FW and BP. For RegMean, it requires the inner product data matrices for layer input to calculate the updated parameters. It only requires a forward process, but loading all the inner products of the layer input matrix requires  � memory. For Fisher merge, it also requires the data to calculate the Fisher Matrix, which requires the forward process to calculate the Fisher matrix and  � memory to store the approximated diagonal Fisher matrix. Grid-search Task Arithmetic (G-Task Arithmetic) requires $O ( G ^ { \mathrm { T } } \times \mathcal { T } _ { \mathrm { F W } } )$ forward process to evaluate, where G is the grid number (G = 100 means 100 girds from 0 to 1) and T is the number of tasks. The space complexity is also equal to the memory requirement of the forward process. For Adamerging, it simultaneously loads T LLMs to optimize, whose time complexity is $O ( \mathcal { T } _ { \mathrm { B P } } )$ and space complexity is: $O ( S _ { \mathrm { B P } } \times T )$ . For weight average, task arithmetic, and MetaGPT, they all do not need extra data information, which is model exclusive. Their time and space complexity is  1 and $O ( n )$ , but only our MetaGPT achieves optimal performance.

<table><tr><td>Model</td><td>JEC-QA FinancelQ MedQA</td><td></td><td>Avg</td></tr><tr><td>LM Math</td><td>31.32 32.83 25.56 30.25</td><td>30.20 24.73</td><td>31.45 26.85</td></tr><tr><td>Code</td><td>29.23</td><td>30.87 26.25 34.17</td><td>28.78</td></tr><tr><td>Weight Average Task Arithmetic</td><td>30.73 30.85</td><td>29.90 33.89</td><td>31.60 31.62</td></tr><tr><td></td><td></td><td>30.13</td><td></td></tr><tr><td>Ties Merging</td><td>30.80</td><td>33.53 30.02</td><td>31.45</td></tr><tr><td>DARE</td><td>30.79</td><td>33.93 30.17</td><td>31.63</td></tr><tr><td>MetaGPT(ours)</td><td>30.97</td><td>34.31 30.07</td><td>31.78</td></tr></table>

Table 5: Out of Distribution Generalization

<table><tr><td colspan="6">Extra Data Info Time Complexity Space Complexity Optimal Apply to LLMs</td></tr><tr><td>RegMean</td><td>√</td><td> $O ( \mathcal { T } _ { \mathrm { F W } } )$ </td><td>O(θ)</td><td>√</td><td>X</td></tr><tr><td>Fisher Merge</td><td>√</td><td> $O ( \mathcal { T } _ { \mathrm { F W } } )$ </td><td>O(θ)</td><td>√</td><td>X</td></tr><tr><td>G-Task Arithmetic</td><td>√</td><td> $O ( G ^ { \mathrm { T } } \times \mathcal { T } _ { \mathrm { F W } } )$ </td><td> ${ \cal O } ( S _ { \mathrm { F W } } )$ </td><td>√</td><td>X</td></tr><tr><td>AdaMerging</td><td>√</td><td> $O ( \mathcal { T } _ { \mathrm { B P } } )$ </td><td> $O ( S _ { \mathrm { B P } } \times T )$ </td><td>√</td><td>X</td></tr><tr><td>Task Arithmetic</td><td>X</td><td>O(1)</td><td>O(θ)</td><td>X</td><td>√</td></tr><tr><td>Weight Average</td><td>X</td><td>O(1)</td><td>O(θ)</td><td>X</td><td>√</td></tr><tr><td>MetaGPT</td><td>X</td><td>O(1)</td><td>O(θ)</td><td>√</td><td>√</td></tr></table>

Table 6: Extra data information requirement, time and space complexity, and optimally of current methods. The time complexity for forward process and back propagation are denote by $\mathcal { T } _ { \mathrm { F W } } , \mathcal { T } _ { \mathrm { B P } }$ . The space complexity for forward process and back propagation are denote by <sub>FW</sub>, <sub>BP</sub>. T is the number of task, � is the number of parameters and G is the grid number $( \mathrm { G } = 1 0 0 $ means 100 girds from 0 to 1).

## 8 Conclusion

In this paper, we have provided a novel model merging method named MetaGPT, an optimal task arithmetic algorithm under the setting of efficient and protecting data privacy, which is specifically designed for LLMs. We provide the mathematical formulation of task arithmetic’s optimization objective and the theoretical analysis of the task arithmetic performance bound. By separating the data and scaling coefficient term under careful approximation, the closed-form solution provides an avenue for optimally achieving task arithmetic under the setting of applicable to LLMs and without using any data. Extensive experiment results show that our MetaGPT outperforms the existing state-ofthe-art model-exclusive merging method and can be integrated with task vector-improving methods such as Ties-Merging and DARE to achieve better performance.

## 9 Limitations

(1) Our works share the same general limitation of existing task arithmetic based methods: Our merging method relies on common initialization and model architecture, which ensures that the task vectors are orthogonal. (2) Moreover, since our method is specifically designed for LLMs and relies on the NTK linearization, for small size models, our method may not perform well.

## References

Mistral AI. Mistral-7b-instruct-v0.2.

Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. 2019. On exact computation with an infinitely wide neural net. In Advances in Neural Information Processing Systems (NeurIPS).

Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. 2022. Ensemble of averages: Improving model selection and boosting performance in domain generalization. Advances in Neural Information Processing Systems, 35:8265–8277.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Rich Caruana. 1997. Multitask learning. Machine learning, 28:41–75.

Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. 2021. Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems, 34:22405–22418.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter,

Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML, pages 794–803. PMLR.

Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. 2020. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In NeurIPS.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Duxiaoman DI. 2023. Financeiq.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Ke Ding, Xin Dong, Yong He, Lei Cheng, Chilin Fu, Zhaoxin Huan, Hai Li, Tan Yan, Liang Zhang, Xiaolu Zhang, et al. 2021. Mssm: a multiple-level sparse sharing model for efficient multi-task learning. In SIGIR, pages 2237–2241.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. 2021. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34:27503– 27516.

Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. 2020. Learning to branch for multi-task learning. In ICML, pages 3854–3863. PMLR.

Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, and Ed Chi. 2021. Dselectk: Differentiable selection in the mixture of experts with applications to multi-task learning. NeurIPS, 34:29335–29347.

Yun He, Xue Feng, Cheng Cheng, Geng Ji, Yunsong Guo, and James Caverlee. 2022. Metabalance: Improving multi-task recommendations via adapting gradient magnitudes of auxiliary tasks. WWW, pages 2205–2215.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. The Twelfth International Conference on Learning Representations.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. 2018. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407.

Arthur Jacot, Franck Gabriel, and Clément Hongler. 2018. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. 2022. Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849.

Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. 2019. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in Neural Information Processing Systems (NeurIPS).

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. 2021. Conflict-averse gradient descent for multi-task learning. NeurIPS, 34:18878–18890.

Shikun Liu, Edward Johns, and Andrew J. Davison. 2019. End-to-end multi-task learning with attention. In CVPR, pages 1871–1880. Computer Vision Foundation / IEEE.

Yongxi Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, Tara Javidi, and Rogerio Feris. 2017. Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification. In CVPR, pages 5334–5343.

Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixtureof-experts. In SIGKDD, pages 1930–1939. ACM.

Manuel Romero. 2023. llama-2-coder-7b (revision d30d193).

Michael S Matena and Colin A Raffel. 2022. Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems, 35:17703– 17716.

Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-stitch networks for multi-task learning. In CVPR, pages 3994–4003. IEEE Computer Society.

Nondzu. Mistral-7b-codealpaca-lora.

OpenAI. 2023. GPT-4 technical report. Preprint, arXiv:2303.08774.

Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. 2024. Task arithmetic in the tangent space: Improved editing of pre-trained models. Advances in Neural Information Processing Systems, 36.

Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David Lopez-Paz. 2023. Model ratatouille: Recycling diverse models for outof-distribution generalization. In International Conference on Machine Learning, pages 28656–28679. PMLR.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106.

Grefenstette Saxton and Kohli Hill. 2019. Analysing mathematical reasoning abilities of neural models. arXiv:1904.01557.

Ozan Sener and Vladlen Koltun. 2018. Multi-task learning as multi-objective optimization. In NeurIPS, pages 525–536.

Ximeng Sun, Rameswar Panda, Rogerio Feris, and Kate Saenko. 2020. Adashare: Learning what to share for efficient deep multi-task learning. NeurIPS, 33:8728– 8740.

Davut Emre TA¸SAR. 2023. llama-2-13b-code-chat.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. 2024. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36.

Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo. 2023a. Adatask: A task-aware adaptive learning rate approach to multi-task learning. In AAAI, volume 37, pages 10745–10753.

Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. 2023b. Adamerging: Adaptive model merging for multi-task learning. In The Twelfth International Conference on Learning Representations.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2023. Language models are super mario: Absorbing abilities from homologous models as a free lunch. arXiv preprint arXiv:2311.03099.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. NeurIPS, 33:5824–5836.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. 2024. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548.

Yu Zhang and Qiang Yang. 2021. A survey on multitask learning. IEEE Transactions on Knowledge and Data Engineering, 34(12):5586–5609.

Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. 2020. Jecqa: a legal-domain question answering dataset. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 9701–9708.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364.

## Appendix

## A Proof

## A.1 Proof of Lemma 4

Using Taylor expansion for $\mathcal { L } ( \pmb \theta _ { \mathrm { f i n a l } } , \pmb x )$ at $\pmb \theta _ { 0 } \mathbf { : }$ :

$$
\mathcal { L } ( \pmb \theta _ { \mathrm { f i n a l } } , \pmb x )\tag{16}
$$

$$
\ O = \mathcal { L } _ { t } ( \sum _ { k = 1 } ^ { n } \lambda _ { k } ( \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } ) + \pmb { \theta } _ { 0 } , \pmb { x } _ { t } )\tag{17}
$$

$$
\scriptstyle = { \mathcal { L } } _ { t } \left( h _ { t } + \theta _ { t } , x _ { t } \right)\tag{18}
$$

$$
\begin{array} { r l } {  { = \mathcal { L } _ { t } \bigl ( \pmb { \theta } _ { t } , \pmb { x } _ { t } \bigr ) + \nabla \mathcal { L } _ { t } \bigl ( \pmb { \theta } _ { t } , \pmb { x } _ { t } \bigr ) \pmb { h } _ { t } } } \end{array}
$$

$$
+ \left. \frac { 1 } { 2 } \pmb { h } _ { t } ^ { \top } \left( \int _ { 0 } ^ { 1 } \nabla ^ { 2 } \mathcal { L } _ { t } ( \gamma _ { t } ( \beta ) ) d \beta \right) \pmb { h } _ { t } \right.\tag{19}
$$

where $\gamma _ { t } ( \beta ) = \pmb { \theta } _ { t } + \beta ( \pmb { \theta } _ { \mathrm { f i n a l } } - \pmb { \theta } _ { t } )$ and $\mathbf { } _ { \mathbf { } ^ { h _ { t } } }$ is the linear combination of � and �:

$$
\pmb { h } _ { t } = \sum _ { k \neq t } \lambda _ { k } ( \pmb \theta _ { k } - \pmb \theta _ { 0 } ) - ( 1 - \lambda _ { t } ) ( \pmb \theta _ { t } - \pmb \theta _ { 0 } )\tag{20}
$$

Because the $\pmb { \theta } _ { t }$ is fine-tuned using loss $\mathcal { L } _ { t }$ , the gradient of $\mathcal { L } _ { t }$ at $\pmb { \theta } _ { t }$ is zero, and the first order expansion is 0. Substituting Eq. 19 to Eq. 3, we have:

$$
\mathrm { T L D } _ { t } = \mathcal { L } _ { t } ( \pmb { \theta } _ { \mathrm { f i n a l } } , \pmb { x } _ { t } ) - \mathcal { L } _ { t } ( \pmb { \theta } _ { t } , \pmb { x } _ { t } )\tag{21}
$$

$$
= \frac { 1 } { 2 } \pmb { h } _ { t } ^ { \top } \left( \int _ { 0 } ^ { 1 } \nabla ^ { 2 } \mathcal { L } _ { t } ( \gamma _ { t } ( \beta ) ) d \beta \right) \pmb { h } _ { t }\tag{22}
$$

Thus, we have completed the proof.

## A.2 Proof of Theorem 7

Before starting the proof, we first introduce a lemma:

Lemma 11. Under the Property. 5, the task vector is linearly with the gradient.

$$
\delta _ { t } ( \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } ) = \nabla _ { \pmb { \theta } _ { 0 } } f ( \pmb { x } , \pmb { \theta } _ { 0 } )\tag{23}
$$

Proof: For gradient descent, we have:

$$
\pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } = \sum _ { i = 1 } ^ { n } l r _ { i } \nabla \mathcal { L } _ { t } ^ { i }\tag{24}
$$

$$
\digamma = \sum _ { i = 1 } ^ { n } l r _ { i } { \frac { \partial { \mathcal { L } } _ { t } ^ { i } } { \partial f } } \nabla f _ { i }\tag{25}
$$

where $l r _ { i }$ and $\nabla { \mathcal { L } } _ { t } ^ { i }$ and $\nabla f _ { i }$ is the learning rate, gradient loss, gradient of $f$ at step i. From Property 5, we can see that the fine-tuning process of $f$ occurs in the linear regime, which indicates that the first order derivative in the task vector direction is an constant. We derivative at $\pmb { \theta } _ { t }$ :

$$
\nabla _ { \pmb { \theta } _ { t } } f ( \pmb { x } , \pmb { \theta } _ { t } ) = \nabla _ { \pmb { \theta } _ { 0 } } f ( \pmb { x } , \pmb { \theta } _ { 0 } )\tag{26}
$$

Thus, we substitute all the gradient of $f _ { i }$ using $\nabla _ { \pmb { \theta } _ { 0 } } f ( \pmb { x } , \pmb { \theta } _ { 0 } )$ :

$$
\delta _ { t } ( \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } ) = \nabla _ { \pmb { \theta } _ { 0 } } f ( \pmb { x } , \pmb { \theta } _ { 0 } )\tag{27}
$$

$$
\mathrm { w h e r e } \quad { \frac { 1 } { \delta _ { t } } } = \sum _ { i = 1 } ^ { n } l r _ { i } { \frac { \partial { \mathcal { L } } _ { t } ^ { i } } { \partial f } } .
$$

Thus, we have completed the proof of the Lemma.

For the of loss function, using Property 5 we have:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { t } ( \pmb { \theta } _ { t } , \pmb { x } _ { t } ) = \frac { 1 } { 2 } \| f ( \pmb { x } _ { t } , \pmb { \theta } _ { t } ) - y \| ^ { 2 } } } \\ & { } & { = \frac { 1 } { 2 } \| ( \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } ) ^ { \top } \nabla f ( \pmb { x } _ { t } ; \pmb { \theta } _ { 0 } ) + C _ { 0 } \| ^ { 2 } } \end{array}\tag{8}
$$

(29)

For the Hessian of loss function, it can be represented as:

$$
\nabla _ { \theta _ { t } } ^ { 2 } \mathcal { L } _ { t } = \nabla _ { \theta _ { 0 } } f ( \pmb { x } _ { t } ; \pmb { \theta } _ { 0 } ) \nabla _ { \theta _ { 0 } } ^ { \top } f ( \pmb { x } _ { t } ; \pmb { \theta } _ { 0 } )\tag{30}
$$

Using Eq. 30 the $\mathrm { T L D } _ { t }$ can be represented as:

$$
2 \mathrm { T L D } _ { t }
$$

$$
= \pmb { h } _ { t } ^ { \top } \left( \int _ { 0 } ^ { 1 } \nabla ^ { 2 } \mathcal { L } _ { t } ( \gamma _ { t } ( \beta ) ) d \beta \right) \pmb { h } _ { t }\tag{31}
$$

(32)

$$
\mathbf { \Sigma } = \pmb { h } _ { t } ^ { \top } \left( \nabla ^ { 2 } \mathcal { L } _ { t } ( \tilde { \pmb { \theta } } ) \right) \pmb { h } _ { t }\tag{33}
$$

$$
{ \bf \Gamma } = { \pmb { h } } _ { t } ^ { \top } \left( \nabla _ { { \theta } _ { 0 } } f ( { \pmb { \theta } } _ { 0 } , { \pmb x } _ { t } ) \nabla _ { { \theta } _ { 0 } } f ^ { \top } ( { \pmb { \theta } } _ { 0 } , { \pmb x } _ { t } ) \right) { \pmb { h } } _ { t }\tag{34}
$$

$$
\mathrm { \ d } = \mathrm { t r } \bigg \{ \pmb { h } _ { t } ^ { \top } \left( \nabla _ { \pmb { \theta } _ { 0 } } f ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \nabla _ { \pmb { \theta } _ { 0 } } f ^ { \top } ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \right) \pmb { h } _ { t } \bigg \}\tag{35}
$$

$$
\leq \mathrm { t r } ( \pmb { h } \pmb { h } ^ { \top } ) \mathrm { t r } \left( \nabla _ { \theta _ { 0 } } f ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \nabla _ { \theta _ { 0 } } f ^ { \top } ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \right)\tag{36}
$$

<sup>For</sup> <sup>tr</sup>(<sup>��⊤</sup>)<sup>,</sup> <sup>using</sup> <sup>Property.</sup> <sup>6,</sup> <sup>we</sup> <sup>have:</sup>

$$
\mathrm { t r } (  { \boldsymbol { h } }  { \boldsymbol { h } } ^ { \top } )\tag{37}
$$

$$
= \left\| \sum _ { k \neq t } ^ { T } \lambda _ { k } ( \pmb \theta _ { k } - \pmb \theta _ { 0 } ) - ( 1 - \lambda _ { t } ) ( \pmb \theta _ { t } - \pmb \theta _ { 0 } ) ^ { \top } \right\| ^ { 2 }\tag{38}
$$

$$
= \sum _ { k \neq t } ^ { T } \left[ \mathbb { 1 } _ { k \neq t } ( \lambda _ { k } ^ { 2 } ) + \mathbb { 1 } _ { k = t } ( 1 - \lambda _ { k } ^ { 2 } ) \right] \lVert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \rVert ^ { 2 }\tag{39}
$$

$$
= \sum _ { k \neq t } ^ { T } \big [ \mathbb { 1 } ( \lambda _ { k } ^ { 2 } ) \lVert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \rVert ^ { 2 } \big ]\tag{40}
$$

where $( \lambda _ { k } ^ { 2 } ) \mathbb { 1 } ( k \neq t ) + ( 1 - \lambda _ { k } ^ { 2 } ) \mathbb { 1 } ( k = t ) : = \mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } )$

For the second part: $\mathrm { t r } ( \nabla _ { \boldsymbol { \theta } _ { 0 } } f ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \nabla _ { \boldsymbol { \theta } _ { 0 } } f ^ { \top } ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) )$ , using Lemma 11 we can have:

$$
\begin{array} { r l } { \mathrm { t r } \left( \nabla _ { \theta _ { 0 } } f ( \theta _ { 0 } , \pmb { x } _ { t } ) \nabla _ { \theta _ { 0 } } f ^ { \top } ( \pmb { \theta } _ { 0 } , \pmb { x } _ { t } ) \right) = } & { { } \delta _ { t } ^ { 2 } \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| ^ { 2 } } \end{array}\tag{41}
$$

Thus, for $\mathrm { T L D } _ { t }$ we can upper bound it by

$$
\mathrm { T L D } _ { t } \leq \frac { \delta _ { t } ^ { 2 } } { 2 } \left. \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \right. _ { 2 } ^ { 2 } \left\{ \sum _ { k \neq t } ^ { T } \mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } ) \left. \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \right. ^ { 2 } \right\}\tag{42}
$$

## A.3 Proof of Theorem 8

By summing Eq.42 from 1 to T, we can complete the proof.

$$
\mathrm { A L D } \leq \sum _ { t = 1 } ^ { T } \frac { \delta _ { t } ^ { 2 } } { 2 } \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| _ { 2 } ^ { 2 } \left\{ \sum _ { k \neq t } ^ { T } \mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } ) \| \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \| ^ { 2 } \right\}\tag{43}
$$

## A.4 Proof of Theorem 9

First, for Eq. 43, we have:

$$
\mathrm { A L D } \leq \frac { \delta _ { t } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| _ { 2 } ^ { 2 } \left\{ \sum _ { k \neq t } ^ { T } \mathbb { 1 } _ { t } ( \lambda _ { k } ^ { 2 } ) \| \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \| ^ { 2 } \right\}\tag{44}
$$

where $\delta _ { 0 } = \operatorname* { m a x } \{ \delta _ { i } \}$ For Eq. 44, it is easy to verify that the terms containing $\lambda _ { t }$ can be represented as:

$$
\mathrm { A L D } _ { \lambda _ { t } } = \frac { \delta _ { t } ^ { 2 } } { 2 } \Vert \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \Vert ^ { 2 } \left[ \sum _ { k = 1 } ^ { T } \mathbb { 1 } _ { t } ( \lambda ) \Vert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \Vert ^ { 2 } \right]\tag{45}
$$

Thus, the ALD can be upper bounded by

$$
\mathrm { A L D } \leq \sum _ { t = 1 } ^ { T } \mathrm { A L D } _ { \lambda _ { t } }\tag{46}
$$

## A.5 Proof of Theorem 10

Because each $\mathrm { \mathbf { A L D } } _ { \lambda _ { t } }$ does not contain other scaling coefficients. We can solve each optimal $\lambda _ { t }$ from $\mathrm { A L D } _ { \lambda _ { t } }$

$$
\lambda _ { t } = \underset { \lambda _ { t } } { \arg \operatorname* { m i n } } \frac { \delta _ { 0 } ^ { 2 } } { 2 } \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| ^ { 2 } \left[ \sum _ { k = 1 } ^ { T } \mathbb { 1 } _ { t } ( \lambda ) \| \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \| ^ { 2 } \right]\tag{47}
$$

$$
= \underset { \lambda _ { t } } { \arg \operatorname* { m i n } } \ \lVert \pmb { \theta _ { t } } - \pmb { \theta _ { 0 } } \rVert ^ { 2 } \left[ \sum _ { k = 1 } ^ { T } \mathbb { 1 } _ { t } ( \lambda ) \lVert \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \rVert ^ { 2 } \right]\tag{48}
$$

The RHS of the above equation is quadratic on $\lambda _ { t }$ and and the optimal solution for $\lambda _ { t }$ is:

$$
\lambda _ { t } = \frac { \| \pmb { \theta } _ { t } - \pmb { \theta } _ { 0 } \| ^ { 2 } } { \sum _ { k = 1 } ^ { n } \| \pmb { \theta } _ { k } - \pmb { \theta } _ { 0 } \| ^ { 2 } }\tag{49}
$$

## B Details of Models and Datasets

Table 7 shows the versions and correspondence with pre-trained backbones of fine-tuned LLMs. Table 8 shows the details of the datasets we use in our paper.

## C Infra and hardware details

We use PyTorch as the deep learning framework. We merge and evaluate the neural networks using A100 GPUs.

## D Hyper-parameter Setting

For both DARE and TIES-Merging, the density of 0.55 is used, and the open-source tool MergeKit<sup>1</sup> is employed for the merging process.

Table 7: Details of datasets we used for our evaluation.
<table><tr><td>Dataset</td><td>Number of</td><td>Number of</td><td>Number of Training Examples Validation Examples Testing Examples</td><td>Evaluate Metric</td></tr><tr><td>WinoGrande</td><td>9,248</td><td>1,267</td><td>1,767</td><td>0-shot accuracy</td></tr><tr><td>AGIEval</td><td>N/A</td><td>N/A</td><td>8,062</td><td>5-shot accuracy</td></tr><tr><td>GSM8k</td><td>7,473</td><td>N/A</td><td>1,319</td><td>4-shot accuracy</td></tr><tr><td>Math</td><td>7,500</td><td>N/A</td><td>1,500</td><td>4-shot accuracy</td></tr><tr><td>MBPP</td><td>374</td><td>30</td><td>500</td><td>3-shot accuracy</td></tr><tr><td>HumanEval</td><td>N/A</td><td>N/A</td><td>164</td><td>0-shot accuracy</td></tr><tr><td>JEC-QA</td><td>N/A</td><td>N/A</td><td>26,365</td><td>5-shot accuracy</td></tr><tr><td>FinancelQ</td><td>N/A</td><td>N/A</td><td>7,173</td><td>5-shot accuracy</td></tr><tr><td>MedQA</td><td>N/A</td><td>N/A</td><td>61,097</td><td>5-shot accuracy</td></tr></table>

Table 8: Details of models we used for our evaluation.
<table><tr><td>Pre-trained Model</td><td>Task</td><td>Fine-tuned-Models</td></tr><tr><td>LLaMA-2-7b</td><td>General Knowledge Mathematical Reasoning Code Generating Chinese Spanish Japanese</td><td>meta-llama/Llama-2-7b-chat-hf TIGER-Lab/MAmmoTH-7B mrm8488/llama-2-coder-7b hfl/chinese-llama-2-7b clibrain/Llama-2-7b-ft-instruct-es elyza/ELYZA-japanese-Llama-2-7b</td></tr><tr><td>Mistral-7b</td><td>General Knowledge Mathematical Reasoning Code Generating</td><td>mistralai/Mistral-7B-Instruct-v0.2 TIGER-Lab/MAmmoTH2-7B Nondzu/Mistral-7B-codealpaca-lora</td></tr><tr><td>LLaMA-2-13b</td><td>General Knowledge Mathematical Reasoning Code Generating</td><td>meta-llama/Llama-2-13b-chat-hf TIGER-Lab/MAmmoTH-13B emre/llama-2-13b-code-chat</td></tr></table>