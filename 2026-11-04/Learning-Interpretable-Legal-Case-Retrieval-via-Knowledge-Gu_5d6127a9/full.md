# Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation

Chenlong Deng, Kelong Mao, Zhicheng Dou\* Gaoling School of Artificial Intelligence, Renmin University of China {dengchenlong,dou}@ruc.edu.cn

## Abstract

Legal case retrieval for sourcing similar cases is critical in upholding judicial fairness. Different from general web search, legal case retrieval involves processing lengthy, complex, and highly specialized legal documents. Existing methods in this domain often overlook the incorporation of legal expert knowledge, which is crucial for accurately understanding and modeling legal cases, leading to unsatisfactory retrieval performance. This paper introduces KELLER, a legal knowledge-guided case reformulation approach based on large language models (LLMs) for effective and interpretable legal case retrieval. By incorporating professional legal knowledge about crimes and law articles, we enable large language models to accurately reformulate the original legal case into concise sub-facts of crimes, which contain the essential information of the case. Extensive experiments on two legal case retrieval benchmarks demonstrate superior retrieval performance and robustness on complex legal case queries of KELLER over existing methods.

## 1 Introduction

Legal case retrieval is vital for legal experts to make informed decisions by thoroughly analyzing relevant precedents, which upholds justice and fairness (Hamann, 2019). This practice is crucial in both common law and civil law systems globally (Lastres, 2015; Harris, 2002). In civil law, although following past cases (known as "stare decisis") is not mandatory, judges are still highly advised to consider previous cases to improve the accuracy and trustworthiness of their judgments.

In legal case retrieval, both the query and the document are structured legal cases, distinguishing the task from other information retrieval (IR) tasks. Specifically, as shown in Figure 1, a legal case document comprises several sections, such as procedure, facts, and the court’s decision, making it much longer than typical queries and passages in the standard ad-hoc search tasks. Its average text length often exceeds the maximum input limits of popular retrievers, such as 512 tokens (Devlin et al., 2019). Moreover, a legal case may encompass multiple, distinct criminal behaviors (Deng et al., 2024b). Comprehensively considering all criminal behaviors of a legal case is important in determining its matching relevance with a query case. However, these key criminal descriptions are usually dispersed throughout the lengthy contents, which can significantly affect the effectiveness of traditional long document modeling strategies like FirstP and MaxP (Dai and Callan, 2019) in the legal

![](images/9c8a6fbdf91970c5fcd0762b6b76edf191fb887860155fc49c0bf3460afc2b20.jpg)  
Figure 1: The query case and candidate document case examples. The query case typically contains only partial content since it has not been adjudicated. Extractable crimes and law articles are highlighted in red.

domain.

To tackle the challenge of comprehending long and complex legal cases, previous works mainly fall into two categories. The first approach focuses on expanding the context window size (Xiao et al., 2021) or splitting legal cases into passages (Shao et al., 2020). However, given the specialized and complex nature of legal texts, merely increasing the context window size still proves insufficient for significantly improving the retrieval performance. In contrast, the second approach performs direct text summarization (Askari and Verberne, 2021; Tang et al., 2023) or embedding-level summarization (Yu et al., 2022) on the legal case, aiming to only keep the most crucial information for assessing the relevance between legal cases. However, they typically only rely on heuristic rules or the models’ inherent knowledge for summarization. As the legal domain is highly specialized, existing approaches that overlook professional legal knowledge (e.g., law articles) are likely to perform inaccurate summarization.

In this paper, we present a Knowledge-guidEd case reformuLation approach for LEgal case Retrieval, named KELLER. Our main idea is to leverage professional legal knowledge to guide large language models (LLMs) to summarize the corresponding key sub-facts for the crimes of the legal cases, and then directly learn to model case relevance based on these crucial and concise sub-facts.

Due to the specialization and complexity of the legal case, it is quite challenging to directly summarize the corresponding key sub-facts for all the crimes from the legal case, even using advanced LLMs (Tang et al., 2023). To address this problem, we propose a two-step legal knowledge-guided prompting method, as illustrated in the left side of Figure 2. In the initial step, we prompt LLM to extract all of the crimes and law articles contained in the legal case and then perform post-processing on them to establish correct mappings between the crimes and law articles by referring to the legal expert database. In the next step, we prompt LLM with the extracted “crime-article ” pairs to summarize the sub-fact of the crime from the legal case. The intermediate law articles, serving as high-level abstractions of the actual criminal events, can largely reduce the difficulty of identifying the corresponding sub-fact for the crime and improve accuracy. Figure 5 shows an example of three summarized sub-facts from a legal case.

Then, we directly model the case relevance based on these sub-facts because they are not only the most crucial information for relevance judgment in legal case retrieval but are also concise enough to meet the text length limitations of popular pre-trained retrieval models. For the comprehensive consideration of effectiveness, efficiency, and interoperability, we adopt the simple MaxSim and Sum operators to aggregate the relevance scores between query and document sub-facts to get the final case relevance score. The model is trained with dual-level contrastive learning to comprehensively capture the matching signals at the case level and the sub-fact level. On two widely-used datasets, we show that KELLER achieves new state-of-the-art results in both zero-shot and fine-tuning settings. Remarkably, KELLER demonstrates substantial improvements in handling complex queries.

Our main contributions can be summarized as:

(1) We propose to leverage professional legal knowledge about crimes and law articles to equip LLM with much-improved capabilities for summarizing essential sub-facts from complex cases.

(2) We suggest performing simple MaxSim and Sum aggregation directly on those refined sub-facts to achieve effective and interpretable legal retrieval.

(3) We introduce dual-level contrastive learning that enables the model to capture multi-granularity matching signals from both case-level and sub-factlevel for enhanced retrieval performance.

## 2 Related Work

Legal case retrieval. Existing legal case retrieval methods are categorized into statistical and neural models. Statistical models, notably the BM25 algorithm, can be enhanced by incorporating legal expert knowledge such as legal summarization (Tran et al., 2020; Askari and Verberne, 2021), issue elements (Zeng et al., 2005) and ontology (Saravanan et al., 2009). Neural models have been advanced through deep learning and the use of pre-trained language models (Devlin et al., 2019; Zhong et al., 2019; Chalkidis et al., 2020; Zhang et al., 2023). Recent advancements in this domain include the design of specialized pre-training tasks tailored for legal case retrieval, which yields remarkable improvements in retrieval metrics (Li et al., 2023a; Ma et al., 2023b; Deng et al., 2024a).

Due to the limitations of neural models in handling long texts, researchers mainly focus on processing lengthy legal documents by isolating the "fact description" section and truncating it to fit the model’s input constraints (Ma et al., 2021; Yao et al., 2022; Ma et al., 2023b; Li et al., 2023a). To overcome the long-text problem, some other strategies include segmenting texts into paragraphs for interaction modeling (Shao et al., 2020), employing architectures like Longformer for extensive pre-training on legal texts (Xiao et al., 2021), and transforming token-level inputs into sentence-level encoding (Yu et al., 2022).

Query rewriting with LLMs. Recently, researchers naturally employ LLMs to enhance the effectiveness of query rewriting and intent understanding (Zhu et al., 2023; Mao et al., 2023; Ma et al., 2023a; Wang et al., 2023; Jagerman et al., 2023; Mao et al., 2024). For instance, HyDE (Gao et al., 2023) creates pseudo passages for better query answers, integrating them into a vector for retrieval, while Query2Doc (Wang et al., 2023) employs few-shot methods to generate precise responses. Furthermore, Jagerman et al. (2023) explores LLMs’ reasoning capacities to develop "Chain-of-Thoughts" responses for complex queries. However, the above methods struggle with legal case retrieval, where both queries and documents are lengthy cases. In the legal domain, PromptCase (Tang et al., 2023) attempts to address this by summarizing case facts within 50 words, but this approach often misses important details as many cases feature multiple independent facts.

## 3 Methodology

In this section, we first introduce some basic concepts in legal case retrieval. Then we delve into the three core parts of our KELLER, including legal knowledge-guided case reformulation, relevance modeling, and dual-level contrastive learning.

## 3.1 Preliminaries

In legal case retrieval, both queries and candidate documents are real structured legal cases that can extend to thousands of tokens in length. Figure 1 shows an illustration of the typical case structure. Specifically, a case usually contains several sections, including procedure, fact, reasoning, decision, and tail. Notably, the candidate documents are completed legal cases that have been through the adjudication process and therefore contain all sections. In contrast, the query cases are not yet adjudicated, so they usually only include the procedure andfact sections.

Formally, given a query case q and a set of document cases D, the objective of legal case retrieval is to calculate a relevance score s between the query case and each document case in D, and then rank the document cases accordingly.

## 3.2 Knowledge-Guided Case Reformulation

When assessing the relevance between two legal cases, the key facts of their crimes are the most crucial things for consideration. Therefore, given the complexity of the original legal cases which makes direct learning challenging, we try to first refine the legal cases into shorter but more essential “crime-fact” snippets. For example, we can get such a snippet from the case shown in Figure 1, whose crime is “the crime of arson” and the fact is “Yan took advantage ofMu’s absence and setfire ...”.

However, the description of a crime and its corresponding facts are often scattered throughout the lengthy case, and a single case may contain multiple crimes and facts, significantly complicating the extraction process. To tackle this problem, we propose a two-step prompting method leveraging professional legal knowledge to guide LLM to achieve accurate extraction.

Crime and law article extraction. First, we prompt LLM to extract all crimes and all law articles from the case. This step is relatively straightforward for LLM, as each crime and law article is a distinct, identifiable element within the text. For example, the extracted crime and law article for the case shown in Figure 1 are “the crime of arson” and “Article 114 and Paragraph 1 of Article 67 of the Criminal Law of the People’s Republic of China”, respectively. Our extraction prompt is shown in Appendix B.

Post-Processing. The extracted law articles may just be the titles. We then expand these titles into full articles by gathering their detailed provision content from the Web based on the titles. Then, we establish a mapping between each crime and its relevant law articles by referring to a database built by our legal experts. Note that the correlation between specific crimes and their corresponding legal articles is objective, as it is clearly defined by law. After post-processing, we can obtain all the “crime-articles” pairs for a legal case.

Fact summarization. Next, we leverage the extracted crimes and their relevant law articles to guide LLM in summarizing the specific facts of each crime from the original legal case. The law articles, serving as high-level abstractions of the actual criminal events, can considerably simplify the task of identifying the corresponding specific facts. The prompt for fact summarization is shown in Appendix B.2.

![](images/c02e5081181c8572f977c0923a9a06989093b2c548b616fcd716cace32294927.jpg)  
Figure 2: Overview of KELLER. We first perform legal knowledge-guided prompting to reformulate the legal cases into a series of crucial and concise sub-facts. Then, we directly model the case relevance based on the sub-facts. The model is trained at both the coarse-grained case level and the fine-grained sub-fact level via contrastive learning.

Through our legal knowledge-guided reformulation, we can accurately distill a series of crimes and their corresponding specific facts from the originally lengthy legal cases. Finally, we form a subfact snippet, with the crime as the title and its facts as the main body. These refined sub-facts are not only the most crucial information for relevance judgment in legal case retrieval but are also concise enough to meet the text length limitations of popular pre-trained retrieval models. Please note that, since the required legal knowledge is present in criminal case documents from mainstream countries (e.g., China and the United States), our approach is actually internationally applicable. Our materials in Appendix D further prove this.

## 3.3 Relevance Modeling

We directly model the relevance of legal cases using the refined sub-facts, rather than relying on the full text of the original legal cases. Specifically, given a query case $q = \{ q _ { 1 } , . . . , q _ { m } \}$ and a candidate case $d = \{ d _ { 1 } , . . . , d _ { n } \}$ , where $q _ { i }$ represents the i-th sub-fact of q and $d _ { j }$ represents the j-th subfact of d. We utilize a pre-trained text encoder to

encode them:

$$
\begin{array} { r } { E _ { q _ { i } } = \mathrm { { P o o l } } _ { [ \mathrm { { C L S } } ] } \left( \mathrm { { E n c o d e r } } ( q _ { i } ) \right) , } \\ { E _ { d _ { j } } = \mathrm { { P o o l } } _ { [ \mathrm { { C L S } } ] } \left( \mathrm { { E n c o d e r } } ( d _ { j } ) \right) , } \end{array}\tag{1}
$$

where Pool<sub>[CLS]</sub> means extracting the embedding output at the [CLS] token position. Then, we compute the similarity matrix ${ \bf M } _ { m \times n }$ using the L2- norm dot product. Each element $M _ { i , j }$ of M is the similarity calculated between the normalized embeddings of the i-th sub-fact in the reformulated query case and j-th sub-fact in the reformulated document case:

$$
\begin{array} { r l } & { M _ { i , j } = \mathrm { { S i m } } ( E _ { q _ { i } } , E _ { d _ { j } } ) } \\ & { \qquad = \mathrm { { N o r m } } ( E _ { q _ { i } } ) \cdot \mathrm { { N o r m } } ( E _ { d _ { j } } ^ { T } ) . } \end{array}\tag{2}
$$

Finally, we aggregate this similarity matrix to derive the matching score. There are various sophisticated choices for aggregation, such as using attention or kernel pooling (Xiong et al., 2017). In this paper, we opt to employ the MaxSim and Sum operators (Khattab and Zaharia, 2020):

$$
s _ { q , d } = \sum _ { i = 1 } ^ { m } \mathbf { M a x } _ { j = 1 } ^ { n } M _ { i , j } ,\tag{3}
$$

where $s _ { q , d }$ is the final predicted relevance score. We choose these two operators because of their advantages in effectiveness, efficiency, and interpretability over the other aggregation approaches for our scenario:

(1) Effectiveness: Typically, each query’s subfact $q _ { i }$ matches one document sub-fact $d _ { j }$ at most in practice, which is well-suited for MaxSim of applying the Max operation across all document’s subfacts for a given query’s sub-fact. For instance, considering a query sub-fact about “drug trafficking”, and the document sub-facts about “drug trafficking” and “the discovery of privately stored guns and ammunition”, only the “drug trafficking” sub-fact of the document is relevant for providing matching evidence. In contrast, using soft aggregation methods (e.g., kernel pooling (Xiong et al., 2017)) may introduce additional noise in this scenario.

(2) Efficiency: Maxsim and Sum operations on tensors are quite efficient for both re-ranking and large-scale top-k retrieval supported by multivector-based Approximate Nearest Neighbor algorithms (Khattab and Zaharia, 2020). This high efficiency is important for meeting the low-latency requirements of the practical use.

(3) Interpretability: MaxSim provides clear interpretability by revealing the quantitative contribution of each query and document sub-fact towards the final relevance score, which can aid in understanding the ranking strategies and justifying the retrieval results. We further illustrate this advantage by studying a real case in Section 4.6.

## 3.4 Dual-Level Contrastive Learning

We incorporate matching signals from both the coarse-grained case level and the fine-grained sub-fact level to comprehensively enhance the model performance in legal case matching.

Case-level contrastive learning. At the case level, we consider directly optimizing toward the final matching score between the query case and the document cases. Specifically, we employ the classical ranking loss function to promote the relevance score between the query and the positive document while reducing it for negative documents:

$$
\mathcal { L } _ { \mathrm { R } } = - \log \frac { \exp ( s _ { q , d ^ { + } } / \tau ) } { \exp ( s _ { q , d ^ { + } } / \tau ) + \sum _ { d ^ { - } } \exp ( s _ { q , d ^ { - } } / \tau ) } ,\tag{4}
$$

where $d ^ { + }$ is the positive document of the query q and each $d ^ { - }$ is from the in-batch negatives. τ is a temperature parameter.

Sub-fact-level contrastive learning. At the subfact level, we incorporate intermediate relevance signals among sub-facts to fine-grainedly enhance the model’s effectiveness in understanding subfacts’ content and their matching relationships.

However, only the case-level relevance labels are available in the dataset. Naively considering all the sub-fact pairs between the query and the positive documents as positives and all the sub-fact pairs between the query and the negative documents as negatives will introduce substantial false positive and negative noise. To mitigate this issue, we propose a heuristic strategy to obtain high-quality relevance labels for the query’s sub-facts $\{ q _ { 1 } , . . . , q _ { m } \}$ . The core idea of this strategy is to combine the caselevel relevance and the charges of each sub-fact to accurately identify true positive and negative samples. We introduce the details of this strategy in Appendix C due to the space limitation.

After getting the sub-fact level relevance labels, we also adopt the ranking loss function for sub-fact level contrastive learning:

$$
\mathcal { L } _ { \mathrm { S } } = - \log \frac { \exp ( s _ { M _ { i , j } + } / \tau ) } { \exp ( s _ { M _ { i , j } + } / \tau ) + \sum _ { J ^ { - } } \exp ( s _ { M _ { i , j } - } / \tau ) } ,\tag{5}
$$

where $M _ { i , j ^ { + } }$ are the similarity score between $q _ { i }$ and its positive document. $M _ { i , j ^ { - } }$ are the similarity score between $q _ { i }$ and its negative document subfact. $J ^ { - }$ − is the collection of all negative document sub-facts for $q _ { i }$ . The final learning objective is the combination of $\mathcal { L } _ { \mathrm { R } }$ and ${ \mathcal { L } } _ { \mathrm { { S } } } .$

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { R } } + \alpha \mathcal { L } _ { \mathrm { S } } , } \end{array}\tag{6}
$$

where $\alpha$ is a hyper-parameter to adjust the weights of two losses.

## 4 Experiments

## 4.1 Experimental Setup

Dataset and evaluation metrics. We conduct extensive experiments on two widely-used datasets: LeCaRD (Ma et al., 2021) and LeCaRDv2 (Li et al., 2023b), whose statistics are listed in Appendix A.1. Considering the limited number of queries in LeCaRD, we directly evaluate all the queries of LeCaRD using the best model trained on LeCaRDv2, thereby avoiding the need for dataset split. Following the previous studies (Li et al., 2023a,b), we regard label=3 in LeCaRD and label 2 in LeCaRDv2 as positive. For the query whose candidate documents are all annotated as positive, we supplement the candidate pool by sampling 10 document cases from the top 100-150 BM25 results. To exclude the effect of unlabeled potential positives in the corpus, we rank the candidate pools and adopt MAP, P@k (k=3), and

NDCG@k (k=3, 5, 10) as our evaluation metrics.

Baselines. We compare KELLER against the following baselines across three categories. The first is traditional probabilistic models, including TF-IDF and BM25. The second is ranking methods based on pre-trained language models, including BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), BGE (Xiao et al., 2023) and SAILER (Li et al., 2023a). The third is ranking methods designedfor handling long (legal) text, including BERT-PLI (Shao et al., 2020), Lawformer (Xiao et al., 2021), and PromptCase (Tang et al., 2023).

Implementations. We introduce the selected language models, hyperparameter settings and other details in Appendix A.2.

## 4.2 Main Results

The main results are as shown in Table 1 and we have the following observations:

(1) KELLER outperforms all baseline methods across all metrics on both datasets. Compared with previous methods tailored for the long-text problem, KELLER employs knowledgeguided case reformulation to address the challenge of long-text comprehension. This demonstrates the effectiveness of separating comprehension and matching tasks in the domain of legal case retrieval.

(2) After fine-tuning on legal case retrieval datasets, the performance gap between generalpurpose and retrieval-oriented PLMs becomes less distinct. This observation may stem from two reasons. First, the scarcity of training data in the legal case retrieval task can induce overfitting to annotation signals, which hampers the model’s generalization capabilities. Second, Naive truncation of lengthy texts can make the model’s inputs lose sufficient matching signals, leading to inconsistencies between relevance annotations and matching evidence.

(3) We observe that these long-text-oriented baseline methods do not show significant advantages. Despite BERT-PLI and Lawformer processing more text than other methods, their input capacity was still insufficient for the average length of legal cases. Handling both long-text processing and complex semantic understanding within one retriever presents a significant challenge. To address this issue, our approach offloads a portion of the long-text comprehension task via knowledgeguided case reformulation and improves the rank-

ing performance.

## 4.3 Zero-shot Evaluation

Considering the inherent data scarcity problem in legal case retrieval, we evaluate the zero-shot performance (i.e., without fine-tuning on the training set of LeCaRDv2) of models on LeCaRDv2.

Results are shown in Table 2 and we find that KELLER consistently outperforms baselines in both zero-shot and fine-tuning settings. Upon comparing the performance of each method under zeroshot and fine-tuned settings, we observe that most methods benefit from fine-tuning except SAILER. Intuitively, models trained in a general domain or task could be enhanced through fine-tuning. In specific domains, continued fine-tuning of models generally does not lead to a significant decrease in performance. We posit that the unexpected outcomes in the SAILER model primarily arise from overfitting the limited data used for fine-tuning, which impairs the generalization capabilities established in the pre-training phase.

## 4.4 Ablation Study

We design the following six ablations: (1) KGCR NS: We replace our Knowledge-Guided Case Reformulation (KGCR) with a Naive Summarization (NS), which produces case summaries without hierarchical structure. We subsequently optimize the dual encoders with this text as the input. (2)MS Mean: We replace MaxSim and Sum (MS) with Mean to capture the average relevance of each sub-fact in the candidate cases to the query. (3) MS  NC: We Naively Concatenate (NC) all the reformulated sub-facts into a text sequence and subsequently optimize the dual-encoders. (4) MS KP: We employ kernel pooling (Xiong et al., 2017) on the score matrix to capture relevance signals. (5) w/o sfCL: Training without the sub-fact-level contrastive learning. (6) w/o SfCL: Training without the case-level contrastive learning.

Results are shown in Table 3 and we can observe:

(1) Every ablation strategy results in a decline in the model’s performance, demonstrating the effectiveness of each module within KELLER. This outcome indicates that KELLER’s architecture is both comprehensive and synergistic, with each module contributing to the model’s overall performance.

(2) The replacement of the KGCR module exhibits the most significant impact on performance. This highlights the pivotal role of the KGCR module in KELLER. The KGCR module decomposes cases into structured sub-facts, which are crucial for the model’s learning process.

Table 1: Main results of the fine-tuned setting on LeCaRD and LeCaRDv2. “ ” indicates our approach outperforms all baselines significantly with paired t-test at $\mathsf { p } < 0 . 0 5$ level. The best results are in bold.
<table><tr><td rowspan="2">Model</td><td colspan="5">LeCaRD</td><td colspan="5">LeCaRDv2</td></tr><tr><td>MAP</td><td>P@3</td><td>NDCG@3</td><td>NDCG@5</td><td>NDCG@10</td><td>MAP</td><td>P@3</td><td>NDCG@3</td><td>NDCG@5</td><td>NDCG@10</td></tr><tr><td colspan="10">Traditional ranking baselines</td></tr><tr><td>BM25</td><td>47.30</td><td>40.00</td><td>64.45</td><td>65.59 69.15</td><td>55.20</td><td>48.75</td><td></td><td>72.11</td><td>72.51</td><td>79.85</td></tr><tr><td>TF-IDF</td><td>42.59</td><td>36.19</td><td>58.14</td><td>59.98</td><td>63.37</td><td>55.19</td><td>47.92</td><td>71.38</td><td>72.70</td><td>75.04</td></tr><tr><td colspan="10">PLM-based neural ranking baselines</td></tr><tr><td>BERT</td><td>53.83</td><td>50.79</td><td>73.19</td><td>73.43</td><td>75.54</td><td>60.66</td><td>53.12</td><td>77.78</td><td>78.73</td><td>80.85</td></tr><tr><td>RoBERTa</td><td>55.79</td><td>53.33</td><td>74.40</td><td>74.33</td><td>76.70</td><td>59.75</td><td>53.12</td><td>78.15</td><td>78.97</td><td>80.70</td></tr><tr><td>BGE</td><td>54.98</td><td>53.33</td><td>74.29</td><td>74.09</td><td>75.65</td><td>60.64</td><td>51.87</td><td>76.99</td><td>78.43</td><td>80.90</td></tr><tr><td>SAILER</td><td>57.98</td><td>56.51</td><td>77.55</td><td>77.04</td><td>79.41</td><td>60.62</td><td>54.58</td><td>78.67</td><td>78.99</td><td>81.41</td></tr><tr><td colspan="10">Neural ranking baselines designed for long text</td></tr><tr><td>BERT-PLI</td><td>48.16</td><td>43.80</td><td>65.74</td><td>68.14</td><td>71.32</td><td>55.34</td><td>46.67</td><td>71.62</td><td>73.68</td><td>76.63</td></tr><tr><td>Lawformer</td><td>54.58</td><td>50.79</td><td>73.19</td><td>73.43</td><td>75.54</td><td>60.17</td><td>54.17</td><td>78.23</td><td>78.99</td><td>81.40</td></tr><tr><td colspan="10">Case reformulation with LLMs</td></tr><tr><td>PromptCase</td><td>59.71</td><td>55.92</td><td>78.75</td><td>78.44</td><td>80.71</td><td>62.25</td><td>54.19</td><td>78.51</td><td>79.07</td><td>81.26</td></tr><tr><td>KELLER</td><td>66.84†</td><td>57.14</td><td>81.24†</td><td>82.42†</td><td>84.67†</td><td>68.29†</td><td>63.13†</td><td>84.97†</td><td>85.63†</td><td>87.61†</td></tr></table>

Table 2: Zero-shot performance on LeCaRD and LeCaRDv2. “ ” indicates our approach outperforms all baselines significantly with paired t-test at $\mathsf { p } < 0 . 0 5$ level. The best results are in bold.
<table><tr><td rowspan="2">Model</td><td colspan="5">LeCaRD</td><td colspan="5">LeCaRDv2</td></tr><tr><td>MAP</td><td>P@3</td><td>NDCG@3</td><td>NDCG@5</td><td>NDCG@10</td><td>MAP</td><td>P@3</td><td>NDCG@3</td><td>NDCG@5</td><td>NDCG@10</td></tr><tr><td colspan="10">General PLM-based baselines</td></tr><tr><td>BERT</td><td>42.92</td><td>37.78</td><td>60.11</td><td>61.37</td><td>64.10</td><td>56.46</td><td>52.08</td><td>75.82</td><td>77.05</td><td>79.39</td></tr><tr><td>RoBERTa</td><td>51.50</td><td>47.62</td><td>69.21</td><td>71.07</td><td>73.60</td><td>57.89</td><td>52.08</td><td>75.48</td><td>76.33</td><td>78.38</td></tr><tr><td>Lawformer</td><td>42.80</td><td>38.41</td><td>59.46</td><td>61.61</td><td>64.13</td><td>55.05</td><td>49.58</td><td>74.42</td><td>74.31</td><td>76.96</td></tr><tr><td colspan="10">Retrieval-oriented pre-training baselines</td></tr><tr><td>BGE</td><td>51.81</td><td>47.62</td><td>68.57</td><td>69.91</td><td>72.61</td><td>57.21</td><td>50.42</td><td>73.59</td><td>75.36</td><td>77.80</td></tr><tr><td>SAILER</td><td>60.62</td><td>56.19</td><td>79.93</td><td>78.99</td><td>81.41</td><td>62.80</td><td>55.00</td><td>79.38</td><td>81.17</td><td>83.83</td></tr><tr><td>KELLER</td><td>64.17†</td><td>57.78</td><td>80.47</td><td>81.43†</td><td>84.36†</td><td>65.87†</td><td>61.67†</td><td>83.33†</td><td>83.75†</td><td>86.06†</td></tr></table>

(3) Among different aggregation strategies, MS Mean demonstrates the least performance degradation. This is primarily because the dataset mainly consists of simple cases with single charges, where Mean and MS become essentially equivalent. Conversely, MS  NC exhibits the most notable performance decline. This is mainly because the model no longer maintains a cross-matching architecture after the concatenation operation. Merging multiple facts into a single representation negatively impacts representation learning.

## 4.5 Evaluations on Different Query Types

We investigate the two query types presented in both LeCaRD and LeCaRDv2: common and controversial. Common queries are similar to initial trials, and controversial queries to retrials, which are typically more complex and require additional expert review. We evaluated multiple models on these query types. Notably, SAILER’s performance declined after fine-tuning, so we included its zeroshot results for comparison, alongside the finetuned outcomes of other models. Results as shown in Figure 3 and we find:

Table 3: Results of ablation study on LeCaRDv2.
<table><tr><td>Strategy</td><td>MAP</td><td>P@3</td><td>NDCG@3</td><td>NDCG@5</td><td>NDCG@10</td></tr><tr><td colspan="6">Effect of knowledge-guided case reformulation</td></tr><tr><td>KGCR→NS</td><td>61.91</td><td>55.13</td><td>79.50</td><td>79.11</td><td>81.47</td></tr><tr><td colspan="6">Effect of different aggregation strategy</td></tr><tr><td>MS→Mean</td><td>67.15</td><td>61.81</td><td>81.58</td><td>84.42</td><td>86.74</td></tr><tr><td>MS→NC</td><td>63.35</td><td>57.92</td><td>80.37</td><td>81.99</td><td>84.04</td></tr><tr><td>MS→KP</td><td>65.47</td><td>60.06</td><td>79.87</td><td>83.61</td><td>85.39</td></tr><tr><td colspan="6">Effect of contrastive learning</td></tr><tr><td>w/o SfCL</td><td>67.39</td><td>61.93</td><td>81.24</td><td>84.73</td><td>86.91</td></tr><tr><td>w/o CaCL</td><td>67.18</td><td>61.67</td><td>82.76</td><td>84.45</td><td>86.51</td></tr><tr><td>KELLER</td><td>68.29</td><td>63.13</td><td>84.97</td><td>85.63</td><td>87.61</td></tr></table>

![](images/6f556186d4683e558a5c23e5c1094aba8fb32ea01b7bbc7c7649af70c605000c.jpg)

![](images/2a6c41ed157d122a7b73cac0fc231a22d07f9fa59152b7705220b1856ef39535.jpg)  
Figure 3: Evaluation on different query types. We evaluate four models on (a) LeCaRD and (b) LeCaRDv2.

![](images/4346807f99311f78ef21bb1a8571b1e79c32ca66d3dc82fcf277feab014a9820.jpg)  
Figure 4: An example of the interpretability of KELLER. We can observe that each sub-fact of the query finds a correct match in the candidate document (in red).

![](images/a7be106b37a404e588c39a5b00373f250035027beb1212a4fb7a256d17e3b3d5.jpg)  
Figure 5: Comparison of the original text, naive summarization, and our proposed knowledge-guided case reformulation. The original text is manually abbreviated due to its length. Important sentences are marked in red.

(1) KELLER outperformed other models on both query types, showing more substantial gains in controversial queries with improvements of 24.04% and 13.41% in the LeCaRD and LeCaRDv2 datasets, respectively. This enhanced performance is credited to KELLER’s novel case reformulation, which simplifies complex scenarios into sub-facts, aiding in better comprehension and matching.

(2) In the LeCaRD dataset, lexical-based models showed consistent performance across different queries, unlike representation-based models which varied significantly. For example, BERT outperformed BM25 on common queries but was less effective on controversial ones, a difference attributed to the models’ limited ability to handle multifaceted cases. KELLER’s cross-matching architecture successfully addresses this limitation.

## 4.6 Case Studies

Case reformulation. We provide an illustrative comparison between the original case description, naive summarization, and our knowledge-guided case reformulation in Figure 5. The case centers on complex issues of drug transport and firearm possession. Most details focus on drug transportation, with brief mentions of firearms found at the defendant’s residence towards the end. Given the 512-token limit of most retrievers, crucial information about the firearms is often inaccessible. While naive summarization captures the main points, it overlooks specifics about the firearms in the context of drug offenses. In contrast, our KGCR method segments the case into three topics—drug transportation, illegal drug possession, and illegal firearms possession—thus detailing each criminal aspect comprehensively.

Interpretability. In KELLER, each sub-fact in a query represents a specific intent of the query, with the highest match score from a candidate case indicating how well this intent is met. KELLER allows users to see which sub-fact in a candidate case matches their intent. For example, in a case involving robbery and harboring crimes shown in Figure 4, KELLER accurately matches sub-facts in the query to those in the candidate case, demonstrating the alignment of KELLER’s scoring with the underlying legal facts of the case. The matching is shown in a matrix, where the positions $( q _ { 1 } , d _ { 1 } )$ and (q<sub>2</sub>, d<sub>2</sub>) highlight the defendant’s actions in the query and the candidate case, respectively, establishing a direct correlation between the computed scores and the case ranking.

## 5 Conclusion

In this paper, we introduce KELLER, a ranking model that effectively retrieves legal cases with high interpretability. KELLER structures legal documents into hierarchical texts using LLMs and determines relevance through a cross-matching module. Our tests on two expert-annotated datasets validate its effectiveness. In the future, we will enhance KELLER by incorporating additional specialized knowledge and generative models to refine performance and produce language explanations.

## Limitations

External Knowledge base Construction. Our method requires constructing a legal knowledge base to assist in case reformulation, which introduces an extra step compared to the out-of-the-box dense retrievers. This issue is common in most domain-specific knowledge-enhanced methods.

Computing Efficiency. Our approach needs to call large language models when processing the query case, which may bring additional computational costs. In our experiments, we have employed techniques such as vLLM to achieve high-speed inference. Furthermore, we believe that with ongoing advancements in techniques in both hardware and algorithms, the computational of utilizing LLMs for processing individual query cases online will be acceptable. For example, Llama3-8B can achieve a speed exceeding 800 tokens per second on the Groq platform, while recent inference services provided by Qwen and DeepSeek require less than \$0.0001 per 1,000 tokens.

## Ethical Discussion

The application of artificial intelligence in the legal domain is sensitive, requiring careful examination and clarification of the associated ethical implications. The two datasets utilized in our experimental analysis have undergone anonymization processes, particularly with regard to personally identifiable information such as names.

Although KELLER demonstrates superior performance on two human-annotated datasets, its recommendations for similar cases may sometimes be imprecise when dealing with intricate real-world queries. Additionally, the case databases in existing systems may not consistently include cases that fully satisfy user requirements. The choice to reference the retrieved cases should remain at the discretion of the experts.

## Acknowledgement

This work was supported by the National Science and Technology Major Project No. 2022ZD0120103, National Natural Science Foundation of China No.62272467, the fund for building world-class universities (disciplines) of Renmin University of China, Public Computing Cloud of RUC. The work was partially done at the Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE, and School of Interdisciplinary Studies of RUC.

## References

Arian Askari and Suzan Verberne. 2021. Combining lexical and neural retrieval with longformer-based summarization for effective case law retrieval. In Proceedings of the Second International Conference on Design of Experimental Search & Information REtrieval Systems, Padova, Italy, September 15-18, 2021, volume 2950 of CEUR Workshop Proceedings, pages 162–170. CEUR-WS.org.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: the muppets straight out of law school. CoRR, abs/2010.02559.

Zhuyun Dai and Jamie Callan. 2019. Deeper text understanding for ir with contextual neural language modeling. In Proceedings ofthe 42nd international ACM SIGIR conference on research and development in information retrieval, pages 985–988.

Chenlong Deng, Zhicheng Dou, Yujia Zhou, Peitian Zhang, and Kelong Mao. 2024a. An element is worth a thousand words: Enhancing legal case retrieval by incorporating legal elements. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11- 16, 2024, pages 2354–2365. Association for Computational Linguistics.

Chenlong Deng, Kelong Mao, Yuyao Zhang, and Zhicheng Dou. 2024b. Enabling discriminative reasoning in llms for legal judgment prediction. CoRR, abs/2407.01964.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers),

pages 4171–4186. Association for Computational Linguistics.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1762–1777. Association for Computational Linguistics.

Hanjo Hamann. 2019. The german federal courts dataset 1950–2019: from paper archives to linked open data. Journal of empirical legal studies, 16(3):671–688.

Bruce V Harris. 2002. Final appellate courts overruling their own" wrong" precedents: the ongoing search for principle. Law Quarterly Review, 118(July 2002):408–427.

Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui Wang, and Michael Bendersky. 2023. Query expansion by prompting large language models. CoRR, abs/2305.03653.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39– 48.

Steven A Lastres. 2015. Rebooting legal research in a digital age.

Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue Wu, Yiqun Liu, Chong Chen, and Qi Tian. 2023a. SAILER: structure-aware pre-trained language model for legal case retrieval. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, pages 1035–1044. ACM.

Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yixiao Ma, and Yiqun Liu. 2023b. Lecardv2: A large-scale chinese legal case retrieval dataset. arXiv preprint arXiv:2310.17609.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692.

Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023a. Query rewriting for retrieval-augmented large language models. CoRR, abs/2305.14283.

Yixiao Ma, Yunqiu Shao, Yueyue Wu, Yiqun Liu, Ruizhe Zhang, Min Zhang, and Shaoping Ma. 2021. Lecard: a legal case retrieval dataset for chinese law system. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 2342–2348.

Yixiao Ma, Yueyue Wu, Weihang Su, Qingyao Ai, and Yiqun Liu. 2023b. Caseencoder: A knowledgeenhanced pre-trained model for legal case encoding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 7134– 7143. Association for Computational Linguistics.

Kelong Mao, Chenlong Deng, Haonan Chen, Fengran Mo, Zheng Liu, Tetsuya Sakai, and Zhicheng Dou. 2024. Chatretriever: Adapting large language models for generalized and robust conversational dense retrieval. CoRR, abs/2404.13556.

Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou, Haonan Chen, and Hongjin Qian. 2023. Large language models know your contextual search intent: A prompting framework for conversational search. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 1211–1225. Association for Computational Linguistics.

Manavalan Saravanan, Balaraman Ravindran, and Shivani Raman. 2009. Improving legal information retrieval using an ontological framework. Artificial Intelligence and Law, 17:101–124.

Yunqiu Shao, Jiaxin Mao, Yiqun Liu, Weizhi Ma, Ken Satoh, Min Zhang, and Shaoping Ma. 2020. Bert-pli: Modeling paragraph-level interactions for legal case retrieval. In IJCAI, pages 3501–3507.

Yanran Tang, Ruihong Qiu, and Xue Li. 2023. Promptbased effective input reformulation for legal case retrieval. In Databases Theory and Applications - 34th Australasian Database Conference, ADC 2023, Melbourne, VIC, Australia, November 1-3, 2023, Proceedings, volume 14386 of Lecture Notes in Computer Science, pages 87–100. Springer.

Vu Tran, Minh Le Nguyen, Satoshi Tojo, and Ken Satoh. 2020. Encoded summarization: summarizing documents into continuous vector space for legal case retrieval. Artificial Intelligence and Law, 28:441– 467.

Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 9414–9423. Association for Computational Linguistics.

Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun. 2021. Lawformer: A pre-trained language model for chinese legal long documents. AI Open, 2:79–84.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023. C-pack: Packaged resources to advance general chinese embedding. CoRR, abs/2309.07597.

Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings ofthe 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, Shinjuku, Tokyo, Japan, August 7-11, 2017, pages 55–64. ACM.

Feng Yao, Chaojun Xiao, Xiaozhi Wang, Zhiyuan Liu, Lei Hou, Cunchao Tu, Juanzi Li, Yun Liu, Weixing Shen, and Maosong Sun. 2022. LEVEN: A largescale chinese legal event detection dataset. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 183–201. Association for Computational Linguistics.

Weijie Yu, Zhongxiang Sun, Jun Xu, Zhenhua Dong, Xu Chen, Hongteng Xu, and Ji-Rong Wen. 2022. Explainable legal case matching via inverse optimal transport-based rationale extraction. In SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022, pages 657–668. ACM.

Yiming Zeng, Ruili Wang, John Zeleznikow, and Elizabeth A. Kemp. 2005. Knowledge representation for the intelligent legal case retrieval. In Knowledge-Based Intelligent Information and Engineering Systems, 9th International Conference, KES 2005, Melbourne, Australia, September 14-16, 2005, Proceedings, Part I, volume 3681 of Lecture Notes in Computer Science, pages 339–345. Springer.

Kun Zhang, Chong Chen, Yuanzhuo Wang, Qi Tian, and Long Bai. 2023. Cfgl-lcr: A counterfactual graph learning framework for legal case retrieval. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3332– 3341.

Haoxi Zhong, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. 2019. Open chinese language pretrained model zoo. Technical report.

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. CoRR, abs/2308.07107.

## Appendix

## A More Details for Experimental Setup

## A.1 Datasets

The statistics of both datasets are listed in Table 4. LeCaRD comprises 107 queries and 10,700 candidate cases. LeCaRDv2, a more extensive collection, includes 800 queries and 55,192 candidate cases.

## A.2 Implementation Details

For baseline models, we employ the default parameter settings of Okapi-BM25 in the implementation of BM25. For ranking methods based on PLMs, a uniform learning rate of 1e-5 and a batch size of 128 are consistently applied. In BERT-PLI, the numbers of queries and candidate case segments are set to 3 and 4, respectively, with a maximum segment length of 256. For Lawformer, the maximum text input length is set to 3,072, optimized using a learning rate of 1e-5 and a batch size of 64.

Table 4: Basic statistics of the datasets.
<table><tr><td>Dataset</td><td>LeCaRD</td><td>LeCaRDv2</td></tr><tr><td># Train queries</td><td></td><td>640</td></tr><tr><td># Test queries</td><td>107</td><td>160</td></tr><tr><td># Documents</td><td>9,195</td><td>55,192</td></tr><tr><td>Average query length</td><td>445</td><td>4,499</td></tr><tr><td>Average doc length</td><td>7,446</td><td>4,768</td></tr><tr><td>Average golden docs / query</td><td>10.39</td><td>13.65</td></tr></table>

In KELLER, we employ the Qwen-72B-Chat (Bai et al., 2023), which is currently one of the best open-source Chinese LLMs, to perform case reformulation. We do not choose OpenAI API due to concerns about reproducibility and high cost. All prompts, except for the case description, are input as system prompts. In the ranking model, the maximum number of crimes per case is capped at 4, which meets the needs of most cases. We adopt the pre-trained retriever SAILER as the text encoder. The τ in the contrastive learning is 0.01, and the α in the final loss function is 0.9. We conduct model training with a learning rate of 1e-5 and a batch size of 128. All experiments are conducted on four Nvidia Tesla A100-40G GPUs.

## B Prompts

## B.1 Extraction Prompt

Extraction Prompt: You are now a legal expert, and your task is tofind all the crimes and law articles in the procuratorate’s charges (or court judgments) from the provided case. The output format is one line each for crimes and law articles, two lines in total. Multiple crimes (law articles) are separated by semicolons.

![](images/1d2cb4583b08dbb57381f6135a28b66cc67b1a2bcb0c78dce664f685264f88c0.jpg)  
Figure 6: Illustration of our proposed sub-fact-level contrastive learning. The green and red squares represent the positive pairs and negative pairs, respectively. The gray squares are the discarded pairs that are not used for training. The blue rounded rectangles encompass blue squares belonging to the same query/document case. $\{ \mathbf { A } , . . . , \mathbf { G } \}$ are crimes.

## B.2 Summarization Prompt

Summarization Prompt: You are now a legal expert, and you are good at analyzing lengthy legal case texts containing multiple circumstances of crime. Your task is to concisely summarize the causes, procedures, and outcomes associated with a specified crime, ensuring each part does not exceed 100 words.

[Crime]: the specific crime name

[Law Articles]: the specific provisions of law articles

## C Strategy to Obtain Sub-Fact-Level Relevance Labels

Specifically, for a positive document $d ^ { + }$ of query $q ,$ we first check whether any of the document subfacts share the same crimes as any of the query sub-facts:

• If it exists, as shown in Figure 6(a), for a query sub-fact $q _ { i } ^ { . }$ , we treat the document sub-facts that share the same crime as the positives (e.g., the green rectangles in columns $d _ { 1 } ^ { + } , d _ { 2 } ^ { + }$ , and $d _ { 3 } ^ { + } )$ , and all the other document sub-facts as negatives $( \mathrm { e . g . }$ ., the red rectangles in columns $d _ { 1 } ^ { + } , d _ { 2 } ^ { + }$ , and $d _ { 3 } ^ { + } )$ . If the crime of $q _ { i }$ is different from any of the document sub-facts, we will not include $q _ { i }$ for training (e.g., the gray rectangles in row $q _ { 3 } )$

• If not, as shown in Figure 6 (b), we select the $( q _ { i } , d _ { j } ^ { + } )$ which has the highest similarity score as a positive training pair (e.g., the green rectangle), and retain any $( q _ { i } , d _ { k } ^ { + } ( k \neq j ) )$ as negatives (e.g., the red rectangles in columns $d _ { 2 } ^ { + }$ and $d _ { 3 } ^ { + }$ ). All the other query and document sub-fact pairs are discarded (e.g., the gray rectangles in columns $d _ { 1 } ^ { + } , d _ { 2 } ^ { + }$ , and $d _ { 3 } ^ { + } )$ ).

Then, for a negative document $d ^ { - }$ of one query sub-fact $q _ { i } ,$ , we first check whether $q _ { i }$ has one positive sample.

• If not, we discard all the document sub-facts because there doesn’t exist a positive sample for contrastive learning (e.g., the gray rectangles of row $q _ { 3 }$ in Figure 6 (a) and (b)).

• If it exists, we further check whether one of its document sub-facts $d _ { j } ^ { - }$ shares the same crime as a $q _ { i }$ .

1. Both $d _ { j } ^ { - }$ and $q _ { i }$ are implicated to the same crime. we will include all $( q _ { i } , d _ { k } ^ { - } ( k \neq j ) )$ as negatives $( \mathrm { e . g }$ ., the red rectangles of column $d _ { 1 } ^ { - }$ and $d _ { 2 } ^ { - }$ in Figure 6 (a) and (b)). All the other sub-facts are discarded to avoid introducing false negatives (e.g., the gray rectangles of $( q _ { 1 } , d _ { 1 } ^ { - } )$ in Figure 6 (a) and (b)).

2. None of $d _ { j } ^ { - }$ and $q _ { i }$ pertain to the same crime. We will include all $( q _ { i } , d _ { j } ^ { - } )$ as negatives (e.g., the red rectangles of $( q _ { 2 } , d _ { 1 } ^ { - } )$ and $( q _ { 2 } , d _ { 2 } ^ { - }$ ) in Figure 6 (a)).

## D Case Format of Other Regions

To demonstrate the international applicability of our method, we use U.S. legal documents as examples. Figure 7 and Figure 8 depict the formats of a U.S. indictment and a judgment document, respectively. It is evident that the legal knowledge required by our method (a combination of charges and law articles in this paper) is commonly present in the body sections of these documents. our method can be applied to reformulate legal texts in documents from other jurisdictions similarly, thereby enhancing their performance of legal case retrieval.

![](images/ed6fc6e62da14c8b22c1d312f5f541fd1ac0e3d7b318a54574b8bffa4e5edf96.jpg)  
Figure 7: Illustration of the indictment document of US.

![](images/cc37c8a99a789803481e7cb9684305d27e4a7b360e2e0dec4b97acce2d0c7bd8.jpg)  
Figure 8: Illustration of the judgment document of US.