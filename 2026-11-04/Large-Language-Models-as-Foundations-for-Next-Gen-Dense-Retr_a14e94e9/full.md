# Large Language Models as Foundations for Next-Gen Dense Retrieval: A Comprehensive Empirical Assessment

Kun Luo<sup>1,2,3</sup>† Minghao Qin<sup>2</sup>† Zheng Liu<sup>2</sup>∗ Shitao Xiao<sup>2</sup> Jun Zhao<sup>1,3</sup> Kang Liu<sup>1,2,3</sup>∗

<sup>1</sup>The Key Laboratory of Cognition and Decision Intelligence for Complex Systems,

Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>2</sup>Beijing Academy of Artificial Intelligence, Beijing, China

<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China luokun695, zhengliu1026 @gmail.com kliu@nlpr.ia.ac.cn

## Abstract

Pre-trained language models like BERT and T5 serve as crucial backbone encoders for dense retrieval. However, these models often exhibit limited generalization capabilities and face challenges in improving in-domain accuracy. Recent research has explored using large language models (LLMs) as retrievers, achieving state-of-the-art performance across various tasks. Despite these advancements, the specific benefits of LLMs over traditional retrievers and the impact of different LLM configurations—such as parameter sizes, pre-training duration, and alignment processes—on retrieval tasks remain unclear.

In this work, we conduct a comprehensive empirical study on six key dimensions of dense retrieval capabilities, including in-domain accuracy, data efficiency, zero-shot generalization, lengthy retrieval, instruction-based retrieval, and multi-task learning. We evaluate over 15 different backbone LLMs and non-LLMs. Our findings reveal that larger models and extensive pre-training consistently enhance in-domain accuracy and data efficiency. Additionally, larger models demonstrate significant potential in zero-shot generalization, lengthy retrieval, instruction-based retrieval, and multi-task learning. These results underscore the advantages of LLMs as versatile and effective backbone encoders in dense retrieval, providing valuable insights for future research and development in this field.

## 1 Introduction

Dense retrieval, a novel paradigm in Information Retrieval (IR), has emerged with the advancement of deep neural networks. Unlike traditional IR methods, dense retrieval encodes both queries and documents as embeddings within a shared latent space, capturing their semantic relationships through embedding similarities. Dense retrieval models have become the predominant choice in recent neural retrieval approaches and are widely applied in various downstream tasks such as web search, question answering, and sentence similarity (Karpukhin et al., 2020; Xiong et al., 2020; Muennighoff et al., 2022).

In the past few years, dense retrieval models intensively adopted pre-trained language models, such as BERT (Devlin et al., 2018) and T5 (Raffel et al., 2020), as their backbone encoders. These models excel in identifying semantic similarities between queries and documents. However, they still face significant challenges in becoming versatile enough to handle a wide range of retrieval tasks (Muennighoff et al., 2022). Their in-domain retrieval accuracy is often constrained by the capacity of their backbone encoders, such as the number of parameters (Ni et al., 2021). Additionally, dense retrieval models typically struggle to generalize to unseen data, necessitating fine-tuning with a large amount of labeled data to perform well in the target domain. Finally, achieving versatility in dense retrieval models requires training on multiple retrieval tasks simultaneously, which demands sufficient capacity from the backbone encoder (Zhang et al., 2023; Xiao et al., 2023).

Recently Large Language Models (LLMs) have been prompted or fine-tuned as dense retrieval models and achieved improved performance across a wide range of retrieval tasks, thanks to their superior capability for semantic understanding and rich world knowledge (Li et al., 2023; Wang et al., 2023; Zhuang et al., 2024; Muennighoff et al., 2024). These models vary in parameters from 2 billion to 56 billion, with pre-training sufficiency ranging from hundreds of billions to tens of trillions of tokens, and include both base models and human preference aligned chat models. Despite the common understanding that larger models generally yield better performance (Kaplan et al., 2020; Biderman et al., 2023), the specific benefits of varying parameter numbers, pre-training sufficiency, and alignment processes of backbone LLMs for different retrieval tasks still remain unclear.

In this study, we focus on the following two research questions: 1) For different retrieval tasks, what specific benefits can LLMs offer compared to non-LLMs as the backbone encoders? 2) For LLMs with varying configurations (i.e., different parameter numbers, pre-training sufficiency and alignment processes), what contributes more to different retrieval tasks as the backbone encoder. We conduct comprehensive empirical investigation across a wide range of retrieval tasks, assessing various critical retrieval capabilities: in-domain accuracy, data efficiency, zero-shot generalization, lengthy retrieval generalization, instruction-based retrieval, and multi-task learning. Our study explore over 15 different backbone LLMs and non-LLMs, with parameter numbers ranging from 0.1 billion to 32 billion and varying pre-training sufficiency, including both base LLMs and chat LLMs.

Previous dense retrieval models have demonstrated inferior in-domain accuracy due to the limited capacity of their backbone encoders (Ni et al., 2021). We employ MS MARCO (Nguyen et al., 2016), one of the largest web search datasets, to train and evaluate the in-domain accuracy of dense retrieval models with different backbone encoders. Our results indicate that both increasing the model size and enhancing pre-training sufficiency can consistently improve the upper limit of in-domain accuracy. Notably, we discover that both base LLMs and human-preference-aligned chat LLMs show comparable potential as backbone encoders for dense retrieval tasks. By training with different proportions of MS MARCO, we explore data efficiency and find that scaling up model size facilitates convergence, allowing LLMs to converge swiftly even with limited annotated data, without the need for intricate multi-stage training processes.

We examine generalization ability from three perspectives: zero-shot generalization, lengthy retrieval generalization, and instruction-based retrieval generalization. First, we evaluate zero-shot generalization using BEIR benchmark (Thakur et al., 2021). Our findings indicate that model size is the most crucial factor for zero-shot retrieval generalization. Moreover, traditional dense retrieval models are limited by the maximum input length used during pre-training and retrieval training. We investigate whether LLM-based retrievers, pre-trained with longer context windows, can effectively generalize to lengthy retrieval tasks even when trained with shorter passage lengths. Finally, dense retrieval models often lack flexibility in handling varying retrieval intents (Su et al., 2022). We explore the capability of different models to incorporate instructions during retrieval, discovering that training with instruction benefits LLMs but not non-LLMs, and that human-preference alignment does not significantly improve performance compared to base LLMs.

We further explore the multi-task learning capabilities of models with different backbone encoders, essential for developing versatile retrievers (Zhang et al., 2023; Xiao et al., 2023). We adopt five distinct retrieval tasks, where interference exists due to varying retrieval intents. Our findings reveal that although all models experience performance decreases with multi-task training compared to training on each single-task, increasing model size consistently mitigates this gap.

To summarize, we make the following contributions: 1) We conduct a thorough experimental study using more than 15 backbone encoders with different configurations for dense retrieval across six distinct retrieval tasks. 2) We demonstrate that LLM-based retrievers consistently enhance performance across all retrieval tasks compared to non-LLM-based retrievers. 3) We investigate how different configurations of backbone LLMs impact each retrieval task, focusing on distinct retrieval capabilities.

## 2 Related Work

The related works are reviewed from two aspects: dense retrieval, LLM-based retriever.

First of all, in the realm of neural retrievers, dense retrieval models have consistently demonstrated superior performance over traditional sparse models like BM25 across a wide array of retrieval tasks (Karpukhin et al., 2020; Ni et al., 2021; Muennighoff et al., 2022). A critical factor contributing to the success of dense retrieval models is the utilization of powerful pre-trained language models as their initialization.

Over the past few years, pre-trained language models such as BERT (Devlin et al., 2018) and T5 (Raffel et al., 2020) have been intensively used as backbone encoders for dense retrieval. For instance, GTR (Ni et al., 2021) highlights the indomain accuracy and generalization capabilities of T5-based dense retrieval models, with model parameters reaching up to 4.8 billion. Fang et al. (2024) explores scaling laws for dense retrieval models but restricts their study to BERT backbones with up to 110 million parameters and only explores the in-domain situation. Currently, state-ofthe-art dense retrievers employ models with more than 7 billion parameters or more as backbones. Neelakantan et al. (2022) discuss large-scale unsupervised text embedding pre-training, observing consistent performance improvements when scaling up GPT-based dense retrieval model sizes from 300 million to 175 billion parameters. Additionally, recent studies such as Wang et al. (2023) have shown that fine-tuning directly with labeled data can achieve strong performance. Our study focuses on fine-tuning directly using labeled data while comparing various backbone encoders.

Large Language Models (LLMs) have recently demonstrated significant potential as backbone encoders for dense retrieval, attributed to their vast number of parameters and extensive pre-training. Repllama (Ma et al., 2023) fine-tuned Llama-2-7B and Llama-2-13B to function both as dense retrievers and pointwise rerankers. LLaRA (Li et al., 2023) introduced two pretraining tasks specifically designed to better adapt the backbone Llama-2- 7B model for dense retrieval, resulting in notable improvements in both supervised and zero-shot scenarios. E5-mistral and Gecko (Wang et al., 2023; Lee et al., 2024) enhanced the training of LLMbased dense retrievers using synthetic data, employing models with 1.5 billion and 7 billion parameters to achieve notable results across various retrieval tasks. GRIT (Muennighoff et al., 2024) successfully unified text embedding and generation within a single LLM, maintaining performance levels comparable to those of specialized embedding-only and generative-only models, using a model with 56 billion parameters (14 billion activation parameters). LLM2Vec (BehnamGhader et al., 2024) presented an unsupervised method for transforming decoderonly LLMs into dense retrievers, demonstrating significant promise for adapting LLM backbone encoders for dense retrieval in an unsupervised manner. PromptReps (Zhuang et al., 2024) employed human preference-aligned chat LLMs to produce high-quality dense representations unsupervised.

These models vary in parameters from 1.5 billion to 56 billion, with pre-training covering hundreds of billions to tens of trillions of tokens, and include both base LLMs and human preference-aligned chat LLMs. Despite the exciting advancements in retrieval tasks achieved by leveraging various LLMs with distinct configurations and diverse training strategies, the specific benefits of variations in parameter count, pre-training extent, and alignment processes of backbone LLMs for retrieval tasks remain still uncertain.

## 3 Preliminary

Dense retrieval leverages an encoder to project both the query q and the candidate passage p into a shared dense embedding space, resulting in embeddings $\mathrm { h } _ { q }$ and $\mathrm { h } _ { p } .$ A scoring function, such as the inner product or cosine similarity, is then applied to these dense vectors to model relevance:

$$
\mathrm { s } ( \mathrm { q } , \mathrm { p } ) = \langle \mathrm { h } _ { q } , \mathrm { h } _ { p } \rangle\tag{1}
$$

This allows for the retrieval of relevant documents by performing approximate nearest neighbor (ANN) search within the embedding space.

In our study, we compare more than 15 backbone encoders, varying in model architecture (encoderonly and decoder-only), model size (0.1B to 32B), and pre-training sufficiency (up to 15T tokens). Consistent with prior research, we utilize the [CLS] token to obtain text representations for the BERT model and employ mean-pooling for the T5 model. For instance, BERT tokenizes the input text into a sequence T: [CLS], $\mathrm { t } _ { 1 } , . . . , \mathrm { t } _ { N }$ , [EOS]. This tokenized sequence is subsequently encoded by BERT, generating output embeddings that are combined to form the text embedding, with the [CLS] token performing this integration:

$$
\mathrm { h } _ { t } = \mathrm { B E R T ( T ) [ C L S ] }\tag{2}
$$

When using large language model (LLM) as the backbone encoder, text embeddings need to be created differently. Most LLMs use a decoder-only architecture and causal attention mechanism, meaning that only the last token in the input sequence can access the global context. As a result, the text embedding is taken from the output embedding of the special token [EOS]:

$$
\mathrm { h } _ { t } = \mathrm { L L M ( T ) [ E O S ] }\tag{3}
$$

Given the query-passage pair $( \mathrm { q } _ { i } , \mathrm { p } _ { i } ^ { + } )$ , we adopt the standard InfoNCE (Izacard et al., 2021) loss L

over the in-batch negatives and hard negatives for training:

$$
\mathrm { L } = - \log \frac { \exp ( \mathrm { s } ( \mathrm { q } _ { i } , \mathrm { p } _ { i } ^ { + } ) ) } { \exp ( \mathrm { s } ( \mathrm { q } _ { i } , \mathrm { p } _ { i } ^ { + } ) ) + \sum _ { j } \exp ( \mathrm { s } ( \mathrm { q } _ { j } , \mathrm { p } _ { j } ^ { - } ) ) }\tag{4}
$$

where $\mathrm { p } _ { j } ^ { - }$ is the set of negative passages and s(q, p) is the scoring function of query and passage. In this paper, we adopt the temperature-based cosine similarity function as follows:

$$
\mathrm { s } ( \mathrm { q } , \mathrm { p } ) = { \frac { 1 } { \tau } } \cos ( \mathrm { h } _ { q } , \mathrm { h } _ { p } )\tag{5}
$$

τ is a temperature hyper-parameter, which is fixed to 0.02 in all experiments.

## 4 Empirical Study

In this section, we aim to address two key research questions: 1) For different retrieval tasks, what specific benefits can LLMs offer compared to non-LLMs as the backbone encoders? 2) For LLMs with varying configurations (i.e., different parameter numbers, pre-training sufficiency, and alignment processes), what contributes more to different retrieval tasks as the backbone encoder. To answer these questions, we conduct a comprehensive empirical study across six critical dimensions of dense retrieval, each encompassing several specific retrieval tasks. These dimensions are investigated using various pre-trained language models as backbone encoders, focusing on: in-domain accuracy (Section 4.1), data efficiency (Section 4.2), zeroshot generalization (Section 4.3), lengthy retrieval generalization (Section 4.4), instruction-based retrieval (Section 4.5), and multi-task learning (Section 4.6).

## 4.1 In-domain Accuracy

Setting We utilize MS MARCO (Nguyen et al., 2016) to train and evaluate the in-domain accuracy of dense retrieval models with varying backbones encoders. Specifically, we employ BERT (Devlin et al., 2018) with 110M and 330M parameters (BERT-base and BERT-large), T5 (Raffel et al., 2020) encoders with parameter numbers ranging from 110M to 4.8B, and a diverse set of LLMs including the Llama, Phi, Gemma, and Qwen1.5 series (Touvron et al., 2023; Gunasekar et al., 2023; Bai et al., 2023; Team et al., 2024). It is important to note that different LLMs have varying configurations. For instance, the phi-1.5 model is a lightweight LLM with 1.3B parameters and is pre-trained on a relatively small amount of tokens (150B), indicating less pre-training sufficiency. In contrast, the Llama-3-8B model is extensively pretrained on over 15T tokens, significantly more than the 2T tokens used for Llama-2-7B. The Qwen1.5 series offers a variety of models in different sizes, all pre-trained on the same corpus, enabling direct comparisons of the effects of scaling up model size.

All models are trained with a batch size of 128 and incorporate 7 hard negative samples to ensure fair comparisons of in-domain retrieval accuracy. All training operations take place on 8xA800 (80GB) GPUs. We use the Adam optimizer with an initial learning rate of 3e-4 and linear decay. For training LLM retrievers, we employ LoRA (Hu et al., 2021), which has demonstrated similar efficacy to full-parameter fine-tuning for retrieval tasks (Ma et al., 2023). The in-domain accuracy of each model is evaluated using the MS MARCO development set, comprising 6,980 queries. We use NDCG@10, MRR@10, Recall@10, and Recall@1000 as evaluation metrics, providing a comprehensive analysis of in-domain performance.

Results and Analysis As presented in Figure 1, the results indicate that model performance generally improves with an increase in parameter numbers. This trend is particularly noticeable within models from the same series. For instance, the Qwen1.5 series demonstrates this progression: Qwen1.5-0.5B model scores 36.7, while the Qwen1.5-32B model achieves 42.6, representing an improvement of 5.9 points. This trend suggests that increasing model size is a feasible way to yield better in-domain accuracy. Detailed results are presented in Table 5.

Additionally, the results demonstrate that LLMbased retrievers significantly outperform non-LLM retrievers. The performance of Gemma-2B has already surpassed all BERT and T5-based models despite having fewer parameters than the T5-xxl model. This suggests that LLMs’ extensive pretraining and advanced language understanding capabilities offer significant advantages as backbone encoders for dense retrieval.

An interesting observation is that smaller models can sometimes marginally outperform larger ones. The Qwen1.5-0.5B model, with fewer parameters, surpasses the Phi-1.5-1.3B model and competes closely with the Phi-2-2.7B model. This performance discrepancy may be attributed to differences in pre-training sufficiency. The Qwen1.5 models benefit from more extensive and diverse pre-training data, totaling over 3 trillion tokens, whereas the Phi models are pre-trained on a smaller amount of high-quality data, with 150 billion tokens for the Phi-1.5 and 1.4 trillion tokens for the Phi-2. This extensive pre-training enables the Qwen1.5-0.5B model to perform better when finetuned for retrieval tasks. A similar conclusion can be drawn from the comparison between the Llama-3-8B and Llama-2-7B models, as well as between LLMs and non-LLMs. Extensive and varied pretraining of backbone encoders can significantly enhance in-domain retrieval accuracy, even compensating for a smaller parameter count.

![](images/84b8b2973ec73d5074ddefbf2582d96388f089fb5a679676f1c069317529ac18.jpg)  
Figure 1: In-domain accuracy (measured by MRR@10)

![](images/9b451cd4c2978a6bcd4b758c3c1e55ebb1cdbc2727cc971c9e93d5f51772f6b0.jpg)  
Figure 2: Data efficiency

## 4.2 Data Efficiency

Setting We use checkpoints from models trained on MS MARCO for different numbers of steps to evaluate their performance on the development set, in order to better understand the impact of parameter number and pre-training sufficiency on data efficiency and convergence speed.

We compare BERT-large, Qwen1.5-0.5B, and Llama-2-7B to explore the impact of data efficiency with model parameter number and pre-training sufficiency. Notably, BERT-large and Qwen1.5-

![](images/bb018215cecfc3d5f0885e99b41c9bcbeadfdf382dd631f0b5b68cf7173bcfe5.jpg)  
Figure 3: Lengthy retrieval

0.5B have similar non-embedding parameter number, while Qwen1.5-0.5B is based on decoder architecture and has undergone more extensive pretraining.

Results and Analysis As presented in Figure 2, our findings indicate that larger model sizes lead to higher data efficiency and faster convergence. Specifically, after 100 training steps on MS MARCO, Llama-2-7B outperforms Qwen1.5-0.5B by 5.4 points and BERT-large by 14.4 points. This suggests that with an increase in parameter number, better performance can be achieved with less labeled data. Furthermore, as shown in Table 1, when comparing the relative score difference between 100 steps and the full training of 3700 steps, Llama-2-7B shows a score difference of 8.8 points, which is smaller than the 9.7 points for Qwen1.5- 0.5B and 15.3 points for BERT-large. This indicates that larger models are able to converge faster.

The experiment results also demonstrate that LLMs have better data efficiency compared to non-LLMs, even with similar parameter sizes. For example, after 100 training steps on MS MARCO, Qwen1.5-0.5B outperforms BERT-large by 9 points. Despite having a similar number of parameters, Qwen1.5-0.5B has undergone more extensive pre-training (over 3 trillion tokens compared to BERT’s 3.3 billion tokens) and employs a decoder architecture, which enhances its language understanding ability and enables faster convergence in the retrieval task where text discriminative ability is crucial.

![](images/488e0d61bb3ae9d20c8f0c0d1416313b19fa57d3202aaf9dabf36aeac3faf149.jpg)  
Figure 4: Zero-shot performance (measured by NDCG@10)

<table><tr><td>Model</td><td>Parameter Number</td><td>NDCG@10</td><td>MRR@10</td><td>Recall@10</td></tr><tr><td colspan="5">100 Steps</td></tr><tr><td>Bert-large</td><td>0.3 B</td><td>24.6(δ = 15.3)</td><td>20.0</td><td>40.5</td></tr><tr><td>Qwen1.5-0.5B</td><td>0.5 B</td><td>33.6(δ = 9.7)</td><td>27.9</td><td>53.2</td></tr><tr><td>Llama-2-7B</td><td>7B</td><td>39.0(δ = 8.8)</td><td>32.4</td><td>61.0</td></tr><tr><td colspan="5">Full 3700 Steps</td></tr><tr><td>Bert-large</td><td>0.3 B</td><td>39.9</td><td>33.8</td><td>60.3</td></tr><tr><td>Qwen1.5-0.5B</td><td>0.5 B</td><td>43.3</td><td>36.7</td><td>65.5</td></tr><tr><td>Llama-2-7B</td><td>7B</td><td>47.8</td><td>40.8</td><td>70.9</td></tr></table>

Table 1: Model convergence speed.

## 4.3 Zero-Shot Generalization

Setting Dense retrieval models typically struggle with zero-shot retrieval on unseen data (Ni et al., 2021). We investigate the specific benefits that LLM-based retrievers can bring to zero-shot generalization, focusing on varying model sizes and pre-training sufficiency.

We evaluate all models on 13 zero-shot retrieval tasks in the BEIR (Thakur et al., 2021) evaluation suite, which encompasses a diverse range of retrieval tasks and domains, including medical retrieval, financial retrieval, and duplication detection. All models are directly transferred for zeroshot evaluation on BEIR after being trained on MS MARCO. During the evaluations, we set the maximum length of the query to 64 tokens and the maximum length of the passage to 256 tokens.

Results and Analysis The results are shown in Figure 4, measured by average performance of NDCG@10 across 13 retrieval tasks. LLM retrievers significantly outperform non-LLM retrievers in zero-shot retrieval tasks, indicating that the extensive knowledge and robust generalization capabilities of LLMs are highly advantageous for zero-shot retrieval. Notably, this improvement is not merely a result of increased model size: even the Qwen1.5- 0.5B model, which has a similar non-embedding parameter count, demonstrates much better generalization (+1.6%) than the BERT-large model. This highlights the potential of LLMs to serve as robust encoders for various retrieval domains.

<table><tr><td>Model</td><td>Parameter Number</td><td>MSMARCO-ID</td><td>MSMARCO-OOD</td></tr><tr><td>Bert-large</td><td>0.3 B</td><td>40.0</td><td>39.3</td></tr><tr><td>Qwen1.5-0.5B</td><td>0.5 B</td><td>43.5</td><td>43.6</td></tr><tr><td>Qwen1.5-4B</td><td>4B</td><td>47.0</td><td>47.0</td></tr><tr><td>Qwen1.5-14B</td><td>14 B</td><td>48.9</td><td>48.9</td></tr><tr><td>Llama-3-8B</td><td>8B</td><td>49.6</td><td>49.6</td></tr></table>

Table 2: Unseen instruction comparison. ”ID” means instructions are seen during training, ”OOD” means the instructions are unseen during training.

For different configurations of LLMs, model size is the primary factor influencing their generalization capability. Unlike in-domain accuracy, where both model size and pre-training sufficiency are important, generalization performance is almost directly correlated with the number of parameters. For example, the Qwen-0.5B model, despite benefiting from more extensive pre-training, performs worse than the Phi-1.5-1.3B and Phi-2-2.7B models with larger parameter sizes but less pre-training sufficiency. This suggests that larger models, with better capacity, can prevent overfitting to domainspecific retrieval data, resulting in better generalization to unseen data.

## 4.4 Lengthy Retrieval Generalization

Setting Traditional dense retrieval models are constrained by the maximum input length used during pre-training and retrieval training, while extending this length significantly increases computational costs (Chen et al., 2024). Given that LLMs are pre-trained with longer context windows, we investigate if they can be trained with shorter passage lengths while effectively generalizing to longer lengths during retrieval. We use MS MARCO for training and set the maximum query length to 64 tokens and the maximum passage length to 256 tokens. All other hyperparameters are aligned with those used in Section 4.1.

<table><tr><td>Model</td><td>Hotpot</td><td>NQ</td><td>MSM</td><td>FiQA</td><td>NFCorpus</td><td>SciFact</td><td>Average</td></tr><tr><td>BERT-large</td><td> $4 6 . 8 ( - 4 . 6 )$ </td><td> $4 7 . 3 ( + 0 . 9 ) $ </td><td> $4 0 . 0 ( + 0 . 1 ) $ </td><td> $2 4 . 3 ( - 2 . 0 )$ </td><td> $2 4 . 7 ( - 2 . 0 )$ </td><td> $5 5 . 5 ( + 0 . 9 ) $ </td><td> $3 9 . 8 ( - 1 . 0 ) $ </td></tr><tr><td>Qwen1.5-0.5B</td><td> $5 9 . 3 ( + 2 . 7 ) $ </td><td> $5 0 . 5 ( + 7 . 1 ) $ </td><td> $4 3 . 5 ( + 0 . 2 ) $ </td><td> $3 3 . 5 ( - 0 . 4 ) $ </td><td> $3 1 . 8 ( + 1 . 5 ) $ </td><td> $6 6 . 2 ( - 0 . 6 )$ </td><td> $4 7 . 4 ( + 1 . 7 ) $ </td></tr><tr><td> $\mathrm { Q w e n 1 . 5 – 4 B }$ </td><td> $6 3 . 6 ( - 0 . 1 )$ </td><td> $5 7 . 7 ( + 7 . 4 )$ </td><td> $4 7 . 0 ( + 0 . 2 ) $ </td><td> $3 9 . 8 ( + 0 . 4 ) $ </td><td> $3 4 . 8 ( - 0 . 6 ) $ </td><td> $7 2 . 1 ( + 1 . 3 )$ </td><td> $5 2 . 5 ( + 1 . 4 ) $ </td></tr><tr><td>Qwen1.5-14B</td><td> $6 9 . 5 ( + 3 . 2 ) $ </td><td> $6 3 . 0 ( + 3 . 7 ) $ </td><td> $4 8 . 9 ( + 0 . 2 ) $ </td><td> $4 5 . 6 ( + 0 . 6 ) $ </td><td> $3 7 . 0 ( + 0 . 6 ) $ </td><td> $7 5 . 9 ( + 1 . 7 ) $ </td><td> $5 6 . 7 ( + 1 . 8 )$ </td></tr><tr><td>Llama-3-8B</td><td>70.9(+4.9)</td><td> $6 3 . 1 ( + 6 . 7 ) $ </td><td> $4 9 . 6 ( + 0 . 9 ) $ </td><td> $4 4 . 8 ( + 3 . 1 ) $ </td><td> $3 7 . 8 ( + 2 . 6 ) $ </td><td> $7 5 . 4 ( + 1 . 4 )$ </td><td> $5 6 . 8 ( + 3 . 2 ) $ </td></tr><tr><td>Qwen1.5-0.5B-Chat</td><td>57.5</td><td>49.5</td><td>43.6</td><td>32.8</td><td>31.7</td><td>65.0</td><td>46.7</td></tr><tr><td> $\mathrm { Q w e n 1 . 5 - 4 B { \cdot } C h a t }$ </td><td>64.0</td><td>58.1</td><td>47.2</td><td>40.2</td><td>36.1</td><td>71.3</td><td>52.8</td></tr><tr><td> $\mathrm { Q w e n 1 . 5 – 1 4 B – C h a t }$ </td><td>69.4</td><td>63.5</td><td>49.0</td><td>44.4</td><td>37.1</td><td>76.0</td><td>56.6</td></tr><tr><td> $\mathrm { L l a m a } { - } 3 { - } 8 \mathrm { B - C h a t }$ </td><td>70.6</td><td>63.0</td><td>49.6</td><td>44.8</td><td>38.2</td><td>75.5</td><td>56.9</td></tr></table>

Table 3: Instruction-based retrieval performance measured by NDCG@10. The average performance discrepancy is compared to training without instruction.

<table><tr><td>Model</td><td>Hotpot</td><td>STS</td><td>MSM</td><td>Tool</td><td>QReCC</td><td>Average</td></tr><tr><td>BERT-large</td><td> $6 2 . 1 ( - 2 . 4 ) $ </td><td> $8 0 . 2 ( + 2 . 7 ) $ </td><td> $3 8 . 8 ( - 1 . 1 ) $ </td><td> $7 6 . 6 ( - 5 . 2 )$ </td><td> $4 7 . 3 ( - 4 . 1 ) $ </td><td> $6 1 . 0 ( - 2 . 0 ) $ </td></tr><tr><td>Qwen1.5-0.5B</td><td>72.1(-1.5)</td><td> $8 0 . 1 ( + 1 . 0 ) $ </td><td> $4 3 . 7 ( + 0 . 2 ) $ </td><td> $8 4 . 8 ( - 4 . 8 ) $ </td><td> $5 0 . 7 ( - 3 . 9 )$ </td><td> $6 6 . 3 ( - 1 . 8 )$ </td></tr><tr><td>Qwen1.5-4B</td><td>79.8(-0.6)</td><td> $8 2 . 0 ( + 2 . 2 ) $ </td><td> $4 6 . 8 ( + 0 . 0 ) $ </td><td> $8 6 . 1 ( - 4 . 2 )$ </td><td> $5 4 . 9 ( - 4 . 4 ) $ </td><td> $6 9 . 9 ( - 1 . 4 ) $ </td></tr><tr><td>Llama-3-8B</td><td> $8 5 . 7 ( + 0 . 3 )$ </td><td> $8 2 . 8 ( + 1 . 3 ) $ </td><td> $4 8 . 9 ( + 0 . 2 ) $ </td><td> $8 9 . 9 ( - 2 . 7 ) $ </td><td> $5 9 . 5 ( - 3 . 3 ) $ </td><td> $7 3 . 4 ( - 0 . 8 ) $ </td></tr></table>

Table 4: Multi-task learning performance measured by NDCG@10. The performance discrepancy is compared to training on each single task.

For evaluation, we utilize NarrativeQA (Kociskˇ y\` et al., 2018), which requires long context information to accurately retrieve target queries. The evaluation was conducted with maximum lengths ranging from 256 to 8192 tokens for passages, with the goal of thoroughly assessing each model’s length generalization capabilities in the retrieval task.

Results and Analysis The results are illustrated in Figure 3. The long context window of LLMs improves length generalization compared to BERT. When evaluated with a context length of 256 tokens on the NarrativeQA Retrieval task, BERT-large outperforms Qwen1.5-0.5B by 0.4 points. However, with a length of 512 tokens, Qwen1.5-0.5B exceeds the performance of BERT-large by 0.9 points. This interesting finding demonstrates that LLM retrievers consistently generalize better with increasing input lengths, while non-LLM retrievers like BERT struggle with longer inputs and are constrained by a 512-token limit unless explicitly extended. Detailed results are presentend in Table 7

Furthermore, increasing the parameter number of LLM retrievers consistently enhances performance with longer inputs. This indicates that scaling up LLMs is an effective strategy for improving lengthy retrieval generalization, obviating the need for specific training on longer retrieval inputs.

## 4.5 Instruction-Based Retrieval

Setting Dense retrieval models often lack flexibility in adapting to varying retrieval intents of users, which is both common and critical in real-world retrieval scenarios (Su et al., 2022). We incorporate instructions into the training of dense retrieval models, aiming to evaluate the instruction comprehension capabilities of models with different backbone encoders. Specifically, we prepare five retrieval instructions and prepend them to queries during training on MS MARCO. We conduct evaluation on six retrieval tasks, including both in-domain and out-of-domain scenarios, to determine whether incorporating instructions can enhance the understanding of retrieval intent thus improving general performance of different models. The instructions are presented in Figure 5.

Results and Analysis As shown in Table 3, training with instructions significantly improves the performance of LLM retrievers, whereas for BERT retrievers results in decreased performance. This suggests that LLMs have superior semantic understanding, enabling them to adjust retrieval objectives based on instructions.

We evaluate models on MS MARCO (Nguyen et al., 2016) development set using instructions not seen during training. The result is presented in Table 2. These instructions are complex modifications of the training instructions (Figure 5), designed to test the models’ robustness. The results show that LLM retrievers exhibit strong robustness to these new instructions, while BERT experience performance degradation due to interference from the unseen instructions. This implies that LLMs can better utilize their capabilities in real-world retrieval scenarios as backbone encoder for dense retrieval, offering better customizability and adaptability to meet diverse user retrieval needs.

Furthermore, we adopt chat LLMs as backbone encoders to investigate if these aligned models could better utilize retrieval instructions, the result is shown in Table 3. Contrary to expectations, chat LLMs do not show further improvements when trained and tested under the same setting as base models. Thus, given the superior scalability of base LLMs across various downstream tasks, the base LLMs remain more suitable as backbone encoders for dense retrieval models.

## 4.6 Multi-Task Learning

Setting Training a versatile dense retrieval model is challenging due to the specific semantic information required by various retrieval tasks, often causing mutual interference (Zhang et al., 2023; Xiao et al., 2023; Neelakantan et al., 2022). We explore the multi-task learning capacity of different backbone encoders, which is essential for developing robust retrievers.

Our study encompasses four distinct retrieval tasks alongside a text similarity task: 1) ToolLLM (Qin et al., 2023): This task evaluates the ability of retrievers to identify necessary tools based on provided instructions and tool descriptions. Performance is measured using NDCG@5 on the test set. 2) QReCC (Anantha et al., 2020): This task involves retrieving relevant knowledge based on the concatenation of conversation context and the most recent query. Performance is assessed using NDCG@3, in line with previous studies (Mao et al., 2023). 3) NLI (Bowman et al., 2015): We utilize the NLI training set to establish text similarity capabilities and evaluate models on STS tasks from the MTEB (Muennighoff et al., 2022). 4) HotpotQA (Yang et al., 2018): This task tests retrieval performance in a multi-hop question-answering scenario. 5) MS MARCO (Nguyen et al., 2016): This task assesses the web search capabilities of different models.

Results and Analysis As shown in Table 4, the results demonstrate a clear trend: as model size increases, the average performance across the five distinct retrieval tasks improves. This indicates that larger models exhibit enhanced universality and capacity, suggesting their greater potential to serve as versatile embedding models in multi-task scenarios.

In addition to comparing the absolute performance of each model across multiple tasks, we conducted experiments contrasting the performance of models trained on each individual task versus joint multi-task training. Table 4 presents the relative performance discrepancy. We observed that multi-task training results in a relative performance decrease compared to single-task training across all tasks. This aligns with the hypothesis proposed by (Neelakantan et al., 2022), suggesting that certain retrieval tasks might have inherently conflicting definitions, such as search and sentence similarity tasks. Notably, the performance decrease diminishes as model size increases, indicating that larger models might be capable of learning the intrinsic relationships and distinctions between tasks during multi-task training. This capability potentially allows these models to narrow the performance gap between multi-task and single-task training, and in some cases even achieve improvements over singletask training. This suggests that LLMs with more parameter numbers have the potential to serve as versatile general-purpose retrievers across multiple retrieval tasks.

## 5 Conclusions

In this paper, we conduct a comprehensive empirical investigation into the benefits and configurations of LLMs as backbone encoders for dense retrieval tasks. Our focus is on comparing LLMs with non-LLMs and analyzing the impact of various LLM configurations, such as parameter count, pre-training sufficiency, and alignment processes. Our study highlights the significant advantages of utilizing LLMs as backbone encoders for dense retrieval tasks. We find that increasing the parameter count and ensuring sufficient pre-training of backbone encoders enhance in-domain accuracy. Additionally, adopting larger models consistently yields performance gains in zero-shot retrieval generalization, lengthy retrieval generalization, and multitask learning. These insights provide a foundation for future research aimed at optimizing dense retrieval models by balancing model size and pretraining sufficiency of backbone LLMs to achieve superior performance across diverse retrieval scenarios.

## 6 Limitations

While our study provides valuable insights into the benefits and configurations of LLMs as backbone encoders for dense retrieval tasks, several limitations should be considered: Firstly, some experiments lack comparisons with all other backbone models in the same series, such as in data efficiency and multitask performance. Secondly, there are still some capability dimensions of retrieval models that haven’t been examined, such as multilingual retrieval and robustness against noisy data. Additionally, certain characteristics of LLMs, such as whether they use unidirectional or bidirectional attention mechanisms, and the overlap between pretraining data and downstream retrieval task data, have not been explored. Addressing these aspects in future studies could provide a more complete, general conclusion.

## 7 Ethical consideration

Our research explores the use of various Large Language Models (LLMs) as backbone encoders for dense retrieval tasks. Despite undergoing additional fine-tuning in various experiments, these models retain ethical and social risks inherent in their pretraining data. Notably, open-source LLMs may incorporate private or contentious data during the training phase, thereby raising additional ethical concerns.

## 8 Ackonwledgements

We would like to thank all the reviewers for their helpful feedback, and EMNLP 2024 and ACL Rolling Review organizers for their efforts. This work was supported by Beijing Natural Science Foundation (L243006) and CCF-BaiChuan-Ebtech Foundation Model Fund.

## References

Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, and Srinivas Chappidi. 2020. Open-domain question answering goes conversational via question rewriting. arXiv preprint arXiv:2010.04898.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and

Siva Reddy. 2024. Llm2vec: Large language models are secretly powerful text encoders. arXiv preprint arXiv:2404.05961.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR.

Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun Liu. 2024. Scaling laws for dense retrieval. arXiv preprint arXiv:2403.18684.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allie Del Giorno, Sivakanth´ Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. 2023. Textbooks are all you need. arXiv preprint arXiv:2306.11644.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.

Toma´s Koˇ ciskˇ y, Jonathan Schwarz, Phil Blunsom, Chris\` Dyer, Karl Moritz Hermann, Gabor Melis, and Ed-´ ward Grefenstette. 2018. The narrativeqa reading comprehension challenge. Transactions ofthe Associationfor Computational Linguistics, 6:317–328.

Jinhyuk Lee, Zhuyun Dai, Xiaoqi Ren, Blair Chen, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Boratko, Rajvi Kapadia, Wen Ding, et al. 2024. Gecko: Versatile text embeddings distilled from large language models. arXiv preprint arXiv:2403.20327.

Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. 2023. Making large language models a better foundation for dense retrieval. arXiv preprint arXiv:2312.15503.

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023. Fine-tuning llama for multi-stage text retrieval. arXiv preprint arXiv:2310.08319.

Kelong Mao, Hongjin Qian, Fengran Mo, Zhicheng Dou, Bang Liu, Xiaohua Cheng, and Zhao Cao. 2023. Learning denoised and interpretable session representation for conversational search. In Proceedings of the ACM Web Conference 2023, pages 3193–3202.

Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2024. Generative representational instruction tuning. arXiv preprint arXiv:2402.09906.

Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. 2022. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316.

Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pretraining. arXiv preprint arXiv:2201.10005.

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading comprehension dataset.

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez ´ Abrego, Ji Ma, Vincent Y Zhao,<sup>´</sup> Yi Luan, Keith B Hall, Ming-Wei Chang, et al. 2021. Large dual encoders are generalizable retrievers. arXiv preprint arXiv:2112.07899.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, and Tao Yu. 2022. One embedder, any task: Instruction-finetuned text embeddings. arXiv preprint arXiv:2212.09741.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale,\` Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Nandan Thakur, Nils Reimers, Andreas Ruckl¨ e, Ab-´ hishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600.

Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023. Retrieve anything to augment large language models. arXiv preprint arXiv:2310.07554.

Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon. 2024. Promptreps: Prompting large language models to generate dense and sparse representations for zero-shot document retrieval. arXiv preprint arXiv:2404.18424.

<table><tr><td>Model</td><td>Dimension</td><td>NDCG@10</td><td>MRR@10</td><td>R@10</td><td>R@1000</td></tr><tr><td>BERT-base</td><td>768</td><td>37.5</td><td>31.6</td><td>57.4</td><td>95.2</td></tr><tr><td>BERT-large</td><td>1024</td><td>39.9</td><td>33.8</td><td>60.3</td><td>96.0</td></tr><tr><td>T5-base</td><td>768</td><td>40.1</td><td>33.7</td><td>61.5</td><td>97.3</td></tr><tr><td>T5-xl</td><td>2048</td><td>42.3</td><td>35.8</td><td>64.0</td><td>98.3</td></tr><tr><td>T5-xxl</td><td>4096</td><td>44.2</td><td>37.6</td><td>66.2</td><td>98.6</td></tr><tr><td>Phi-1.5-1.3B</td><td>2048</td><td>40.6</td><td>34.1</td><td>62.2</td><td>98.0</td></tr><tr><td>Phi-2-2.7B</td><td>2560</td><td>43.3</td><td>36.6</td><td>65.8</td><td>98.6</td></tr><tr><td>Gemma-2B</td><td>2048</td><td>46.8</td><td>39.8</td><td>70.1</td><td>99.2</td></tr><tr><td>Gemma-7B</td><td>3072</td><td>48.7</td><td>41.7</td><td>72.1</td><td>99.4</td></tr><tr><td>Llama-2-7B</td><td>4096</td><td>47.8</td><td>40.8</td><td>70.9</td><td>99.4</td></tr><tr><td>Llama-3-8B</td><td>4096</td><td>49.0</td><td>42.1</td><td>71.9</td><td>99.5</td></tr><tr><td>Llama-2-13B</td><td>5120</td><td>48.7</td><td>42.0</td><td>71.4</td><td>99.5</td></tr><tr><td>Qwen1.5-0.5B</td><td>1024</td><td>43.3</td><td>36.7</td><td>65.5</td><td>98.2</td></tr><tr><td>Qwen1.5-4B</td><td>2048</td><td>46.8</td><td>40.0</td><td>69.7</td><td>99.2</td></tr><tr><td>Qwen1.5-14B</td><td>5120</td><td>48.3</td><td>41.3</td><td>71.5</td><td>99.4</td></tr><tr><td>Qwen1.5-32B</td><td>5120</td><td>49.5</td><td>42.6</td><td>72.7</td><td>99.5</td></tr><tr><td>Qwen1.5-0.5B-Chat</td><td>1024</td><td>43.3</td><td>36.8</td><td>65.1</td><td>98.1</td></tr><tr><td>Qwen1.5-4B-Chat</td><td>2048</td><td>47.0</td><td>40.1</td><td>70.0</td><td>99.2</td></tr><tr><td>Qwen1.5-14B-Chat</td><td>5120</td><td>48.6</td><td>41.5</td><td>71.8</td><td>99.4</td></tr><tr><td>Llama-3-8B-Chat</td><td>4096</td><td>48.7</td><td>41.8</td><td>71.6</td><td>99.4</td></tr></table>

Table 5: Detailed result of in-domain accuracy on MS MARCO.

<table><tr><td>Model</td><td>ArguAna</td><td>ClimateFEVER</td><td>DBPedia</td><td>FEVER</td><td>FiQA2018</td><td>HotpotQA</td><td>NFCorpus</td><td>NQ</td><td>Quora</td><td>SCIDOCS</td><td>SciFact</td><td>Touche2020</td><td>TRECCOVID</td><td>Avg</td></tr><tr><td>Bert-base</td><td>42.9</td><td>19.9</td><td>30.3</td><td>69.4</td><td>24.4</td><td>50.2</td><td>25.3</td><td>42.3</td><td>84.8</td><td>13.1</td><td>50.6</td><td>21.8</td><td>57.4</td><td>40.9</td></tr><tr><td>Bert-large</td><td>43.1</td><td>21.7</td><td>31.9</td><td>68.1</td><td>26.4</td><td>51.4</td><td>26.7</td><td>46.4</td><td>85.7</td><td>13.8</td><td>54.7</td><td>20.7</td><td>59.2</td><td>42.2</td></tr><tr><td>t5-v1_1-xxl</td><td>44.0</td><td>24.6</td><td>35.2</td><td>63.4</td><td>36.1</td><td>57.5</td><td>31.4</td><td>50.3</td><td>85.1</td><td>15.1</td><td>62.0</td><td>22.7</td><td>52.9</td><td>44.6</td></tr><tr><td>Phi-v1.5-1.3B</td><td>45.4</td><td>26.3</td><td>28.0</td><td>64.9</td><td>32.1</td><td>54.5</td><td>31.7</td><td>42.5</td><td>86.6</td><td>16.2</td><td>65.9</td><td>23.6</td><td>65.0</td><td>44.8</td></tr><tr><td>Phi-v2-2.7B</td><td>49.4</td><td>31.2</td><td>34.4</td><td>70.7</td><td>38.4</td><td>62.2</td><td>36.5</td><td>50.8</td><td>86.9</td><td>18.5</td><td>67.2</td><td>23.3</td><td>66.1</td><td>48.8</td></tr><tr><td>Gemma-2B</td><td>47.9</td><td>31.5</td><td>40.2</td><td>72.9</td><td>39.0</td><td>61.9</td><td>36.0</td><td>52.5</td><td>84.8</td><td>18.1</td><td>72.4</td><td>18.7</td><td>55.7</td><td>48.5</td></tr><tr><td>Gemma-7B</td><td>49.9</td><td>31.3</td><td>42.8</td><td>73.5</td><td>44.0</td><td>67.3</td><td>38.1</td><td>60.4</td><td>86.9</td><td>18.7</td><td>74.7</td><td>21.5</td><td>58.3</td><td>51.2</td></tr><tr><td>Llama-2-7B</td><td>48.7</td><td>31.2</td><td>44.4</td><td>76.2</td><td>42.3</td><td>68.1</td><td>36.2</td><td>57.3</td><td>86.8</td><td>18.3</td><td>73.8</td><td>19.6</td><td>47.8</td><td>50.0</td></tr><tr><td>Llama-2-13B</td><td>57.4</td><td>30.7</td><td>43.9</td><td>70.4</td><td>45.6</td><td>67.7</td><td>37.1</td><td>60.9</td><td>85.8</td><td>17.7</td><td>74.6</td><td>21.8</td><td>55.0</td><td>51.4</td></tr><tr><td>Llama-3-8B</td><td>56.1</td><td>30.8</td><td>41.6</td><td>72.7</td><td>41.7</td><td>66.0</td><td>35.2</td><td>56.4</td><td>85.8</td><td>17.8</td><td>74.0</td><td>20.6</td><td>56.9</td><td>50.4</td></tr><tr><td>Qwen1.5-0.5B</td><td>46.0</td><td>26.6</td><td>32.9</td><td>68.1</td><td>31.9</td><td>56.6</td><td>29.8</td><td>43.4</td><td>84.6</td><td>15.8</td><td>65.4</td><td>13.5</td><td>54.7</td><td>43.8</td></tr><tr><td>Qwen1.5-4B</td><td>50.2</td><td>30.5</td><td>40.5</td><td>72.9</td><td>39.4</td><td>63.7</td><td>35.4</td><td>54.3</td><td>85.3</td><td>17.5</td><td>70.8</td><td>18.3</td><td>58.6</td><td>49.0</td></tr><tr><td>Qwen1.5-14B</td><td>56.5</td><td>30.1</td><td>43.0</td><td>73.4</td><td>45.0</td><td>64.4</td><td>36.4</td><td>59.3</td><td>85.7</td><td>19.3</td><td>74.2</td><td>21.9</td><td>60.8</td><td>51.5</td></tr><tr><td>Qwen1.5-32B</td><td>57.5</td><td>31.3</td><td>44.5</td><td>75.3</td><td>47.9</td><td>68.0</td><td>37.1</td><td>59.7</td><td>86.0</td><td>18.8</td><td>75.6</td><td>24.5</td><td>60.3</td><td>52.8</td></tr></table>

Table 6: Detailed result of zero-shot retrieval generalization.

<table><tr><td>Model</td><td>256</td><td>512</td><td>1024</td><td>2048</td><td>4096</td><td>8192</td></tr><tr><td>BERT-large</td><td>18.0</td><td>18.1</td><td>=</td><td>=</td><td>-</td><td></td></tr><tr><td>Qwen1.5-0.5B</td><td>17.6</td><td>19.0</td><td>20.1</td><td>21.1</td><td>37.1</td><td>44.9</td></tr><tr><td>Qwen1.5-4B</td><td>22.8</td><td>23.9</td><td>25.4</td><td>27.1</td><td>49.1</td><td>54.9</td></tr><tr><td>Qwen1.5-7B</td><td>24.3</td><td>26.4</td><td>27.8</td><td>28.2</td><td>52.3</td><td>55.9</td></tr><tr><td>Qwen1.5-32B</td><td>26.9</td><td>28.4</td><td>28.7</td><td>30.8</td><td>54.8</td><td>59.0</td></tr><tr><td>Llama3-8B</td><td>28.4</td><td>29.2</td><td>29.9</td><td>30.4</td><td>53.4</td><td>57.9</td></tr></table>

Table 7: Detailed result of lengthy retrieval on narrativeqa with varying maximum input passage length.

<table><tr><td>MSMARCO Train Instructions</td><td>Given a web search query, retrieve relevant passages that answer the query. Retrieve pertinent passages from web searches to address the query. Obtain relevant excerpts from online searches to provide answers. Access passages on the web that directly respond to the search query. Find pertinent text snippets online that address the given search query. Access relevant passages from internet searches that answer the query at hand.</td></tr><tr><td>MSMARCO Evaluate Instructions</td><td>Locate specific passages on the web that not only answer the given query but also provide in-depth analysis and contextual information. Identify and extract relevant text from online sources that comprehensively respond to the query, including supporting evidence. Obtain detailed excerpts from reputable online searches that provide thorough answers and insights related to the query. Find relevant online snippets that respond to the given query. Retrieve and synthesize passages from websites that directly and extensively</td></tr><tr><td>NQ</td><td>address the search query. Given a question, retrieve Wikipedia passages that answer the question.</td></tr><tr><td>FiQA</td><td>Given a financial question, retrieve user replies that best answer the question.</td></tr><tr><td>HotpotQA</td><td>Given a multi-hop question, retrieve documents that can help answer the question.</td></tr><tr><td>NFCorpus</td><td>Given a question, retrieve relevant documents that best answer the question.</td></tr></table>

Figure 5: Instrctions used in instruction-based retrieval.