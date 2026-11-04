# CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search

Fengran Mo<sup>1</sup>, Abbas Ghaddar<sup>2</sup>, Kelong Mao<sup>3</sup>,

Mehdi Rezagholizadeh<sup>2</sup>, Boxing Chen<sup>2</sup>, Qun Liu<sup>2</sup>, Jian-Yun Nie<sup>1</sup>

<sup>1</sup>DIRO, Université de Montréal, Québec, Canada

<sup>2</sup>Huawei Noah’s Ark Lab, Montreal Research Center, Canada

<sup>3</sup>Renmin University of China

fengran.mo@umontreal.ca, nie@iro.umontreal.ca

## Abstract

In this paper, we study how open-source large language models (LLMs) can be effectively deployed for improving query rewriting in conversational search, especially for ambiguous queries. We introduce CHIQ, a two-step method that leverages the capabilities of LLMs to resolve ambiguities in the conversation history before query rewriting. This approach contrasts with prior studies that predominantly use closed-source LLMs to directly generate search queries from conversation history. We demonstrate on five well-established benchmarks that CHIQ leads to state-of-the-art results across most settings, showing highly competitive performances with systems leveraging closed-source LLMs. Our study provides a first step towards leveraging open-source LLMs in conversational search, as a competitive alternative to the prevailing reliance on commercial LLMs for query rewriting. Our code is publicly available at https://github.com/ fengranMark/CHIQ.

## 1 Introduction

Conversational search enables users to interact with the system in a multi-turn fashion to satisfy their complex information needs (Gao et al., 2022; Zamani et al., 2023). One of the crucial steps is to compose adequate search queries for each contextdependent utterance. Recent advancements in the task-solving capabilities of Large Language Models (LLMs) (Ouyang et al., 2022; Chen et al., 2024b; Wang et al., 2024a; Huang et al., 2024) have motivated researchers to integrate these models into existing conversational search systems.

Most recent studies (Mao et al., 2023a; Ye et al., 2023) leverage LLMs to directly generate search queries based on the context of the conversation history. Although seemingly straightforward, this technique is shown to achieve higher effectiveness in query rewriting than fine-tuning a smaller language model, such as T5 (Raffel et al., 2020; Chung et al., 2022). However, these performance gains are primarily achieved through the use of commercial, closed-source LLMs (OpenAI, 2023). This is primarily because closed-source LLMs can better perform complex reasoning tasks (Gudibande et al., 2023; Kaddour et al., 2023) compared to open-source models.

One of the main challenges in conversational search resides in the ambiguous nature of the conversation history. Figure 1 illustrates an example where solving co-reference relation in $u _ { 4 }$ and elaborating the response in $r _ { 3 }$ can help generate an adequate search query. Intuitively, performing these tasks requires basic NLP task-solving capabilities, which even small-scale open-source LLMs (e.g., 7B) possess (Touvron et al., 2023; Jiang et al., 2023). The key challenge is to unlock the full capabilities of open-source LLMs for conversational search. This requires carefully preparing the conversation history to enhance its quality, rather than directly using it to generate the search query.

In this paper, we propose CHIQ, a method that aims to enhance the quality of contextual history for improving query rewriting. As illustrated in Figure 1, we leverage the NLP capabilities of LLMs (e.g. solving coreference relation or expanding the context) to make the conversational history less ambiguous, consequently enhancing the relevance of the generated search query. We investigate various methods for integrating refined conversational history into existing frameworks, including ad-hoc query rewriting, generating pseudo supervision signals for fine-tuning query rewriting models, and the fusion of both approaches.

We conduct extensive experiments using the open-source LLM, LLaMA-2-7B (Touvron et al., 2023), across five well-established conversational search benchmarks under both dense and sparse retrieval settings. The experimental results indicate that enhancing the conversational history using our method achieves state-of-the-art performance across most settings, often surpassing systems powered by closed-source LLMs. Our analysis reveals that although closed-source LLMs benefit from enhancing the history, the gap with open-source models is narrower when using the enhanced history with different facets compared to the original. Our contributions are summarized as follows:

![](images/d86d9be877fe4a3ed8323367cbedd2f2236d3686912142d5ebee9d0548afb139.jpg)  
"Something" is a song by band the Beatles from their 1969 album "Abbey Road". It was written by George Harrison, the band's lead guitarist […] as a composer to the level of the Beatles' principal songwriters, John Lennon and Paul McCartney.  
Figure 1: An illustrative example of a conversational history (left box) and the gold positive passage relevant to the last user turn. The enhanced history obtained using our method described in § 3.2 is in the middle box. The right box shows the three search queries generated by LLaMA-2-7B conditioned on the original history, and our CHIQ-FT and CHIQ-AD methods described in § 3.4 and § 3.3, respectively. Underlined terms in the gold passages are those that appear in the query generated by our approaches, which is conditioned on the enhanced history and did not appear in the query generated by the method that uses the original history.

• We propose a two-step method for query rewriting that relies on open-source language models: enhancing the conversation history and then generating the search query.

• We introduce three approaches for generating the search query on top of the enhanced conversation history: ad-hoc query rewriting (CHIQ-AD), fine-tuning a small LM for the task (CHIQ-FT), and a fusion of both approaches (CHIQ-Fusion).

• Experiments conducted on five conversational search benchmarks demonstrate that CHIQ, using open-source LLMs, achieves state-ofthe-art performance across most settings, often surpassing systems that rely on closedsource LLMs.

## 2 Related Work

Different from traditional ad-hoc retrieval, which assumes users submit a stand-alone query, conversational search provides a conversational interface so that users can elaborate more complex search requirements, and interactively perform search. The main challenge lies in accurately understanding the user’s real search intent, which may be embedded within a longer, noisy, and more complex conversational context history. There are two wellestablished approaches in the literature for conversational search: Conversational Dense Retrieval (CDR) (Qu et al., 2020; Yu et al., 2021; Mao et al., 2024; Mo et al., 2024a) and Conversational Query Rewriting (CQR) (Elgohary et al., 2019).

The CDR systems aim to fine-tune an end-toend conversational dense retriever that can directly model the entire conversational history to return relevant documents (Kim and Kim, 2022; Mo et al., 2024b). Conversely, the CQR systems focus on formulating an adequate search query based on the conversational history. This query can then serve as the input to an existing, well-established retrieverranker framework. We base our solution on CQR, leveraging its ability to integrate with existing adhoc search models, which has demonstrated significant practical value (Dalton et al., 2022).

Earlier approaches to CQR attempted to select useful tokens from the conversation context (Kumar and Callan, 2020; Voskarides et al., 2020; Fang et al., 2022) or to train generative rewriter models with conversational sessions to mimic the humanrewritten query (Yu et al., 2020; Lin et al., 2020; Vakulenko et al., 2021). To optimize query rewriting, some studies have adopted reinforcement learning (Wu et al., 2022; Chen et al., 2022), or used the ranking signals with the rewriting model training (Qian and Dou, 2022; Mo et al., 2023a; Mao et al., 2023b). In addition, there have been endeavors to improve the conversion history quality through context denoising (Lin et al., 2021b; Mao et al., 2022a; Krasakis et al., 2022; Mo et al., 2023b; Mao et al., 2023c), and data augmentation (Dai et al., 2022; Mao et al., 2022b; Mo et al., 2024c). Unlike them, we enhance query rewriting by leveraging the NLP capabilities of open-source LLMs to reduce the ambiguity of conversational history. There have been multiple endeavors to integrate LLMs to solve traditional ad-hoc search sub-tasks (Zhu et al., 2023), such as query expansion (Wang et al., 2023; Gao et al., 2023), dense retrieval (Ma et al., 2023; Wang et al., 2024b), and re-ranking (Sun et al., 2023). About conversational search, Jin et al. (2023), Jang et al. (2023), and Chen et al. (2024a) attempt to improve CDR with unsupervised fine-tuning. Mao et al. (2023a), Mo et al. (2024d), and Ye et al. (2023) explore how LLMs can understand users’ contextualized search intents via CQR. Unlike direct rewriting the query using LLMs, we investigate various approaches for integrating refined conversational history into CQR frameworks.

## 3 Methodology

## 3.1 Task Formulation

Let $\mathcal { H } = \{ u _ { k } , r _ { k } \} _ { k = 1 } ^ { n }$ represent the user-system conversational history, where $u _ { k }$ and $r _ { k }$ are the user question and the system response at the k-th turn. Given a new user question $u _ { n + 1 }$ , the goal of a conversational search system is to return a set of passages $\mathcal { P } _ { n + 1 }$ that are relevant to and $u _ { n + 1 }$ which would eventually help generate the model response $r _ { n + 1 }$ . To solve the main challenge of uncovering the real search intents hidden in the user’s context-dependent query, a conversational query rewriting (CQR) module has been commonly employed as an intermediate step to obtain a rewritten query $q _ { n + 1 }$ , which in turn is used as input to an offthe-shelf retriever. Recently, LLMs have become the default option for obtaining $q _ { n + 1 }$ as follows:

$$
q _ { n + 1 }  \mathcal { L L M } ( \mathbb { Z } ^ { C Q R } \oplus \mathcal { H } \oplus u _ { n + 1 } )\tag{1}
$$

where $\oplus$ denotes concatenation and ${ \mathcal { T } } ^ { C Q R }$ is a manually-engineered instruction prompt describing the CQR task. The choice of $\mathcal { L } \mathcal { L } \mathcal { M } ( . )$ has predominantly favored closed-source commercial models, mainly CHATGPT, some of which require multiple iterations to get the optimal query (Mao et al., 2023a; Ye et al., 2023). In this work, we leverage the basic NLP capabilities of open-source LLMs to generate $\mathcal { H } ^ { \prime } .$ , a clearer and less noisy version of . This refined version can then serve as a substitute for in Eq. 1, potentially improving the quality of $q _ { n + 1 } ^ { \prime }$ using an open source LLM.

## 3.2 History Enhancement

In this section, we propose five approaches to tackle the ambiguity problems inherent in conversational history  and map each of them to a fundamental NLP task ability. Then, we explain how we design prompts for an LLM to make part or the entire history clearer. The exact prompts we used, along with illustrative examples for each case, are presented in Appendix A.

## 3.2.1 Question Disambiguation

Users often expect a human-to-human level of interaction with modern conversational systems. They often use acronyms, ambiguous words, or coreference substitutes when asking questions, expecting native understanding and default reasoning capabilities from these systems. For a search system, the search intent is often unclear and ambiguous. Therefore, we propose a prompt to an LLM, denoted as ${ \cal T } ^ { Q D }$ , that takes the conversational history and the subsequent user question $u _ { n + 1 }$ as input, to generate $u _ { n + 1 } ^ { \prime } ,$ a self-contained and unambiguous version of $u _ { n + 1 }$ which can substitute it in Eq. 1.

## 3.2.2 Response Expansion

In conversation sessions, it is common for model responses to be short and concise, especially for the factoid query. While brevity is often convenient for the user to acquire needed information, it makes the response less informative for a search system, which requires abundant rewrite/expansion resources. To handle this issue, we design a prompt ${ \mathcal { T } } ^ { R E }$ which instructs an LLM to enrich the content of the last model response. The goal is to make it self-contained by leveraging the preceding conversational history. The enhanced history $\mathcal { H } ^ { \prime }$ is obtained by replacing the original response by $r _ { n } ^ { \prime } .$

## 3.2.3 Pseudo Response

Given that LLMs have been demonstrated to encapsulate human knowledge, one could employ them to speculate on potential responses directly. The intuition is that, even if the response includes some noise, it may still contain relevant terms, particularly when the LLM is prompted to produce a selfcontained answer. Therefore, we design a prompt ${ \mathcal { T } } ^ { P R }$ that takes the conversational history and $u _ { n + 1 }$ as inputs to generate a pseudo-response $r _ { n + 1 } ^ { \prime }$ The latter can be used to expand the input of Eq. 1 to improve the quality of the query generation.

## 3.2.4 Topic Switch

It is natural in a conversation that different turns may focus on different aspects. Some of them are relevant to the current turn, while others may not. This is especially the case when conversations are long. In such cases, using the full history is highly likely to distract the CQR module, leading to poor query generation. Therefore, we design a prompt $\bar { \mathcal { Z } } ^ { T S }$ that instructs the LLM to determine whether a topic switch happens between $u _ { n + 1 }$ and . If a switch is identified, the enhanced history would only include the last turn to maintain the transition as $\mathcal { H } ^ { \prime } = \{ u _ { n } ^ { \prime } , r _ { n } ^ { \prime } \}$ . The other turns in are deemed to be irrelevant for generating $q _ { n + 1 } ^ { \prime }$ and ignored.

## 3.2.5 History Summary

As the conversation goes on, the historical context becomes longer, which contains more irrelevant/noisy parts. A summary of history context is expected to contain only the most useful information of the original long context and better serve for query expansion. Thus, we propose the prompt ${ \mathcal { T } } ^ { H S }$ , which takes a history context (original or even enhanced) and generates a summary of the conversation $\mathcal { H } ^ { \prime }$

## 3.3 Ad-hoc Query Rewriting

A straightforward method to obtain the enhanced rewritten query $q _ { n + 1 } ^ { \prime }$ is to independently utilize the outputs of the five methods described previously. However, by complementing each other, the outputs of these methods can collectively contribute to a more enhanced conversational history, thereby significantly improving the retrieval performance by generating a better query. We intuitively define multiple combinatory configurations for updating the input in Eq. 1, which are denoted by different symbols. The addition of +QD or +RE indicates that we replace $u _ { n + 1 }$ and $r _ { n }$ with $u _ { n + 1 } \oplus u _ { n + 1 } ^ { \prime }$ (§ 3.2.1) and $r _ { n } ^ { \prime }$ (§ 3.2.2) in $\mathcal { H ^ { \prime } }$ , respectively. +PR signifies that $r _ { n + 1 } ^ { \prime }$ (§ 3.2.3) is concatenated to the input of Eq. 1; Lastly, +TS indicates that $\mathcal { H } ^ { \prime }$ should omit the previous turns except the last one if a topic switch is detected. Lastly, +HS means that is overwritten by the entire $\mathcal { H ^ { \prime } }$ obtained in § 3.2.5. In our default configuration, we first check for topicswitch (TS). If the result is affirmative, we only use the QD+RE+PR configuration on the top of the truncated history in § 3.2.4. Otherwise, we apply QD+RE+PR on top of the original history, followed by +HS which consists of obtaining the history summary on top of the enhanced history. We report the results of models using this configuration in the remaining sections.

## 3.4 Search-Oriented Fine-tuning

In existing studies, fine-tuning conversational query generators based on a small-scale language model, such as T5-base, have proven to be both effective and efficient (Lin et al., 2020). These models consist of using human-rewritten query (Wu et al., 2022) or LLM-generated query (Jang et al., 2023) to serve as supervision signals and take and $u _ { n + 1 }$ as input. However, they do not take the ranking signals into account during the training and the supervision signals might be sub-optimal (Lin et al., 2021b; Mo et al., 2023a).

Considering that oracle search queries are typically unavailable and costly to annotate, we propose extending the existing approach to generate pseudo-supervision signals for query generation by leveraging the outputs produced in § 3.2. More precisely, we propose three modifications to Eq. 1 to obtain a search-oriented $q _ { n + 1 } ^ { \prime }$ as follow:

$$
Q _ { n + 1 } ^ { \prime }  \mathscr { L } \mathscr { L } M ( \hat { \mathbb { Z } } ^ { C Q R } \oplus \mathscr { H } ^ { \prime } \oplus u _ { n + 1 } \oplus p _ { n + 1 } ^ { * } )\tag{2}
$$

where we replace  with the enhanced history $\mathcal { H } ^ { \prime }$ and update the instruction ${ \mathcal { I } } ^ { C Q R }$ to ${ \hat { \mathcal { I } } } ^ { C Q R }$ to condition the query generation on the gold passage $p _ { n + 1 } ^ { * }$ and prompt the LLM to generate multiple pseudo-queries in the same forward pass.

$$
q _ { n + 1 } ^ { \prime }  \arg \operatorname* { m a x } _ { q ^ { \prime } } { S ( Q _ { n + 1 } ^ { \prime } , p _ { n + 1 } ^ { * } ) } , q ^ { \prime } \in Q _ { n + 1 } ^ { \prime }\tag{3}
$$

Then, we select $q _ { n + 1 } ^ { \prime }$ , the one with the highest retrieval score $\boldsymbol { \mathcal { S } }$ determined by an off-the-shelf retriever and relevance judgment from the set of pseudo-queries $Q _ { n + 1 } ^ { \prime }$ as Eq 3, which is used as the supervision signal to fine-tune a rewriting model $\mathcal { M } ( \mathcal { H ^ { \prime } } \oplus u _ { n + 1 } ) = q _ { n + 1 } ^ { \prime }$ by maximum likelihood estimation. It is important to mention that the process described in this section is conducted offline and performed once, for the purpose of generating pseudo-labeled queries to fine-tune a searchoriented query rewriter. During inference,  and $u _ { n + 1 }$ serve as inputs for the fine-tuned model to generate the query $q _ { n + 1 }$ , and no calls are made to the LLM, so that the latency is not much affected.

## 4 Experimental Setup

## 4.1 Datasets and Evaluation Metrics

To be comparable with state-of-the-art systems (Mo et al., 2023a; Mao et al., 2023a), we consider two standard benchmarks for conversational search: TopiOCQA (Adlakha et al., 2022), QReCC (Anantha et al., 2021). TopiOCQA focuses on the challenge of the topic switch under the conversational setting, while QReCC focuses on the query rewriting problem. We run experiments on the official train-test splits and report MRR, NDCG@3, and Recall@10 to evaluate the passage retrieval results as in previous works. In addition, we evaluate three CAsT datasets (Dalton et al., 2020, 2021, 2022) which are used solely as test sets, to further validate the zero-shot or transfer learning ability of our approach, e.g., when CQR models are trained on TopiOCQA and tested on CAsTs.

## 4.2 Baselines

We define three main configurations for the approaches that use our enhanced generated queries:

• CHIQ-AD directly use the queries generated by the ad-hoc method described in § 3.3 as input for an off-the-shelf retriever.

• CHIQ-FT search queries generated by a small LM (e.g., T5) fine-tuned for the CQR task following the approach described in § 3.4.

• CHIQ-Fusion We fuse the rank list retrieved by CHIQ-AD and CHIQ-FT using the resultlevel fusion technique (Lin et al., 2021b).<sup>1</sup>

We compared our methods with a variety of systems that can mainly be classified into three categories. More precisely, we first compare against traditional systems that fine-tune smallscale CQR models (e.g., T5-base) including: QuReTeC (Voskarides et al., 2020) T5QR (Lin et al., 2020), CONQRR (Wu et al., 2022), ConvGQR (Mo et al., 2023a), EDIRCS (Mao et al., 2023b). Then, we compare with the systems that fine-tune an LLM-based retriever, e.g., create the query and document representation by the ending token from a decoder-only model, including RepLLaMA (Ma et al., 2023), E5-Mistral (Wang et al., 2024b), and LLM-Embedder (Zhang et al., 2023), or fine-tune an LLM-based CQR model as RETPO (Yoon et al., 2024) and IterCQR (Jang et al., 2023). Besides, we include the systems that directly obtain the rewritten query by prompting LLMs such as LLM-Aided IQR (Ye et al., 2023), HyDE (Gao et al., 2023), Query2doc (Wang et al., 2023) and LLM4CS (Mao et al., 2023a). Although not directly comparable, we report results of systems that fine-tune an adhoc search retriever for conversational scenarios, including the one without LLMs ConvDR (Yu et al., 2021) and with LLMs InstructorR (Jin et al., 2023). A detailed description of each aforementioned baseline is presented in Appendix B.3.

## 4.3 Implementation Details

We conduct experiments with the instruct-tuning variants<sup>2</sup> of both LLaMA-2-7B (Touvron et al., 2023) and Mistral-2-7B (Jiang et al., 2023) as (.) in Eq. 1 and Eq. 2. We experiment with both BM25 (Robertson et al., 2009) sparse retriever and ANCE dense retriever (Xiong et al., 2020). In addition, we use FlanT5-base<sup>3</sup> (Chung et al., 2022) and large models as the backbone when fine-tuning a CQR model on TopiOCQA and QReCC. The fine-tuning process consists of 10 epochs with a learning rate of 1e-5 and a batch size of 8 for both datasets. More implementation details can be found in Appendix B.2.

## 5 Results and Analysis

## 5.1 Main Results

Table 1 shows both the dense and sparse retrieval performances of systems with diverse properties on the TopiOCQA and QReCC. We report the results of our systems using LLaMA-2-7B as the backbone LLM to make the results comparable with previous work. First, we observe that using our enhanced conversation history significantly improves performance over vanilla baselines that use the original history, for both ad-hoc QR (LLM4CS) and fine-tuning a small QR model (T5QR). For dense retrieval, CHIQ-AD outperforms LLM4CS by 5.5% and 2.2% MRR on TopiOCQA and QReCC respectively, while CHIQ-FT reports a gain of 7.0% and 1.9% over T5QR on the same datasets. Similar gains are also observed using the sparse retriever, indicating the strong effectiveness of our methods.

<table><tr><td rowspan="2">Type</td><td rowspan="2">System</td><td colspan="5">System Properties</td><td rowspan="2">TopiOCQA</td><td colspan="3"></td><td colspan="3">QReCC</td></tr><tr><td>DR</td><td>QR</td><td>CS</td><td>OS</td><td>FT QF</td><td>MRR</td><td>N@3</td><td>R@10</td><td>MRR</td><td>N@3</td><td>R@10</td></tr><tr><td rowspan="10">DEunGe CE)</td><td>ConvDR</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>27.2</td><td>26.4</td><td>43.5</td><td>38.5</td><td>35.7</td><td>58.2</td></tr><tr><td>InstructorR</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>25.3</td><td>23.7</td><td>45.1</td><td>43.5</td><td>40.5</td><td>66.7</td></tr><tr><td>QuReTeC</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>11.2</td><td>10.5</td><td>20.2</td><td>35.0</td><td>32.6</td><td>55.0</td></tr><tr><td>T5QR</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>23.0</td><td>22.2</td><td>37.6</td><td>34.5</td><td>31.8</td><td>53.1</td></tr><tr><td>CONQRR</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td></td><td></td><td></td><td>41.8</td><td>一</td><td>65.1</td></tr><tr><td>ConvGQR</td><td>X</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>25.6</td><td>24.3</td><td>41.8</td><td>42.0</td><td>39.1</td><td>63.5</td></tr><tr><td>EDIRCS</td><td>x</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td><td></td><td></td><td>–</td><td>42.1</td><td>-</td><td>65.6</td></tr><tr><td>IterCQR</td><td>x</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td><td>26.3</td><td>25.1</td><td>42.6</td><td>42.9</td><td>40.2</td><td>65.5</td></tr><tr><td>RETPO</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>30.0</td><td>28.9</td><td>49.6</td><td>44.0</td><td>41.1</td><td>66.7</td></tr><tr><td>LLM-Aided</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td></td><td></td><td></td><td>43.9</td><td>41.3</td><td>65.6</td></tr><tr><td>LLM4CS</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>√</td><td>27.7</td><td>26.7</td><td>43.3</td><td>44.8</td><td>42.1</td><td>66.4</td></tr><tr><td>CHIQ-FT</td><td>X</td><td>√</td><td>X</td><td>√</td><td>√</td><td>X</td><td>30.0†</td><td>28.9†</td><td>51.0†</td><td>36.9</td><td>34.0</td><td>57.6</td></tr><tr><td>CHIQ-AD</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>33.2†</td><td>32.2†</td><td>53.0†</td><td>47.0†</td><td>44.6†</td><td>70.8†</td></tr><tr><td>CHIQ-Fusion</td><td>x</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>38.0†</td><td>37.0†</td><td>61.6†</td><td>47.2†</td><td>44.2†</td><td>70.7†</td></tr><tr><td></td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>8.5</td><td>7.3</td><td>16.0</td><td>34.0</td><td>30.5</td><td>55.5</td></tr><tr><td rowspan="10">(BM25) S parsse)</td><td>QuReTeC T5QR</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>11.3</td><td>9.8</td><td>22.1</td><td>33.4</td><td>30.2</td><td>53.8</td></tr><tr><td>CONQRR</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td><td>X</td><td></td><td></td><td></td><td>38.3</td><td></td><td>60.1</td></tr><tr><td>ConvGQR</td><td>x</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>I</td><td>45.6</td><td>44.1</td><td></td></tr><tr><td>EDIRCS</td><td></td><td></td><td>x</td><td>x</td><td>√</td><td>√</td><td>12.4</td><td>10.7</td><td>23.8</td><td></td><td></td><td>41.0</td></tr><tr><td></td><td>X</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td><td>-</td><td>-</td><td>-</td><td>41.2</td><td>-</td><td>62.7</td></tr><tr><td>IterCQR</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>16.5</td><td>14.9</td><td>29.3</td><td>46.7</td><td>44.1</td><td>64.4</td></tr><tr><td>LLM-Aided</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>一</td><td></td><td>一</td><td>48.9</td><td>46.3</td><td>66.4</td></tr><tr><td>LLM4CS</td><td>x</td><td>√ √</td><td>x x</td><td>√ √</td><td>X √</td><td>√ x</td><td>18.9 17.0†</td><td>17.7 15.4†</td><td>33.7 32.3†</td><td>47.8 37.8</td><td>45.0 35.0</td><td>69.1</td></tr><tr><td>CHIQ-FT CHIQ-AD</td><td>x x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>22.5†</td><td>20.5†</td><td>40.4†</td><td>53.1†</td><td>50.7†</td><td>57.1 77.2†</td></tr><tr><td>CHIQ-Fusion</td><td>x</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>25.6†</td><td>23.5†</td><td>44.7†</td><td>54.3†</td><td>51.9†</td><td>78.5†</td></tr></table>

Table 1: Performance of dense and sparse retrieval on TopiOCQA and QReCC with different systems. We list the attributes of the reported baseline systems, which include: DR based on conversational dense retrieval, QR perform query rewriting, CS leverage close-source LLMs (e.g., ChatGPT or GPT-4), OS leverage open-source LLMs (mainly LLaMA-2-7B), FT fine-tune a small LM (mainly T5-base) for QR, and QF fuse multiple queries for retrieval. RETPO‡ involves high-cost supervised fine-tuning an LLM for QR. denotes significant improvements with t-test at $p < 0 . 0 5$ over all compared baselines (except CONQRR, RETPO, and IterCQR). Bold and underline indicate the best and the second-best results within the categories of dense and sparse retrieval.
<table><tr><td rowspan="2">System</td><td rowspan="2">LLM</td><td colspan="3">CAsT-19</td><td colspan="3">CAsT-20</td><td colspan="3">CAsT-21</td></tr><tr><td>MRR</td><td>N@3</td><td>R@10</td><td>MRR</td><td>N@3</td><td>R@10</td><td>MRR</td><td>N@3</td><td>R@10</td></tr><tr><td>RepLLaMA</td><td>LLaMA-2-7B</td><td>62.4</td><td>31.6</td><td>10.6</td><td>26.8</td><td>18.3</td><td>10.4</td><td>47.4</td><td>32.7</td><td>19.6</td></tr><tr><td>E5-Mistral</td><td>Mistral-7B</td><td>62.2</td><td>31.3</td><td>9.5</td><td>22.0</td><td>15.4</td><td>8.4</td><td>48.2</td><td>32.5</td><td>20.5</td></tr><tr><td>LLM-Embedder</td><td>LLaMA-2-7B</td><td>63.3</td><td>36.6</td><td>11.4</td><td>25.2</td><td>15.4</td><td>8.7</td><td>46.8</td><td>31.2</td><td>17.3</td></tr><tr><td>HyDE</td><td>ChatGPT-3.5</td><td>55.6</td><td>39.2</td><td>10.0</td><td>44.8</td><td>29.3</td><td>16.9</td><td></td><td></td><td></td></tr><tr><td>Query2doc</td><td>ChatGPT-3.5</td><td>58.8</td><td>42.4</td><td>11.6</td><td>48.6</td><td>32.5</td><td>17.3</td><td></td><td></td><td></td></tr><tr><td>InstructorR</td><td>ChatGPT-3.5</td><td>61.2</td><td>46.6</td><td>10.4</td><td>43.7</td><td>29.6</td><td>8.3</td><td>46.7</td><td>32.5</td><td>18.4</td></tr><tr><td rowspan="3">LLM4CS</td><td>LLaMA-2-7B</td><td>68.4</td><td>45.9</td><td>11.2</td><td>52.3</td><td>37.2</td><td>17.9</td><td>57.0</td><td>41.5</td><td>20.2</td></tr><tr><td>Mistral-2-7B</td><td>67.6</td><td>44.5</td><td>10.9</td><td>48.3</td><td>33.5</td><td>17.0</td><td>53.0</td><td>35.3</td><td>19.6</td></tr><tr><td>ChatGPT-3.5</td><td>70.4</td><td>46.8</td><td>11.7</td><td>58.6</td><td>41.5</td><td>19.3</td><td>66.1</td><td>46.9</td><td>24.4</td></tr><tr><td>CHIQ-FT</td><td>LLaMA-2-7B</td><td>68.5</td><td>45.1</td><td>11.9†</td><td>46.3</td><td>31.6</td><td>15.9</td><td>53.9</td><td>36.0</td><td>20.4</td></tr><tr><td>CHIQ-AD</td><td>LLaMA-2-7B</td><td>70.8†</td><td>47.6†</td><td>11.9†</td><td>51.0</td><td>34.4</td><td>17.9</td><td>57.7</td><td>42.0</td><td>22.6</td></tr><tr><td>CHIQ-Fusion</td><td>LLaMA-2-7B</td><td>73.3†</td><td>50.5†</td><td>12.9†</td><td>54.0</td><td>38.0</td><td>19.3</td><td>62.9</td><td>46.5</td><td>25.2†</td></tr></table>

Table 2: Zero-shot retrieval performances of the systems involved with LLMs under the dense retrieval (ANCE). denotes significant improvements with t-test at $p < 0 . 0 5$ over all compared baselines. Bold and underline indicate the best and the second-best results, respectively.

Second, we notice that vanilla QR systems on top of an enhanced history can outperform systems that utilize additional training techniques and sophisticated modules. For instance, CHIQ-AD outperforms both ConvDR and IntructorR, which need relevance judgments to fine-tune a conversational dense retriever on the raw input; ReTPO, which finetunes an LLM for QR and in addition leverages GPT-4 for data augmentation. While CHIQ-FT outperforms its direct competitors, primarily ConvGQR and IterCQR that refine the supervision signals on TopiOCQA, it underperforms on QReCC mainly because previous fine-tuned QR models rely on QReCC’s human-rewritten queries. The contrasting observations between the two datasets suggest that enhancing the history is crucial for performance when no QR-supervised annotations exist.

Third, we observe that by systematically fusing the outputs of our approaches, CHIQ-Fusion outperforms the models that use each component separately, achieving the best performance across most settings. The gains are more significant on the topic-mixed and more challenging TopiOCQA compared to QReCC, with 4.8% and 0.2% MRR score improvements on each dataset. Interestingly, this occurs even though CHIQ-FT significantly underperforms compared to CHIQ-AD, suggesting that CHIQ-FT still generates query content that is complementary and not captured by CHIQ-AD.

## 5.2 Zero-shot Results

We compare the dense retrieval performances of different systems that leverage LLMs under a zero-shot manner on three CAsT datasets in Table 2. We observe a consistent pattern as previous results in Table 1 when comparing the performances within our approaches. More precisely, although CHIQ-FT performs slightly worse compared to CHIQ-AD, fusing their outputs systematically leads to better performances across all three datasets. Besides, we can see that CHIQ-AD outperforms most systems either utilizing open-source or close-source LLMs and yields results competitive with the state-of-the-art system LLM4CS, which requires multiple calling for each query turn. Specifically, CHIQ-AD surpasses LLM4CS with LLaMA-2-7B, on CAsT-19 and CAsT-21. In addition, our top-performing approach, CHIQ-Fusion, outperforms all compared systems, except the LLM4CS with close-source ChatGPT-3.5 on CAsT-20 and CAsT-21, indicating the superior effectiveness of our approaches. We also find that adhoc fine-tuned LLM-based retrievers (RepLLaMA, E5-Mistral, and LLM-Embedder) underperform the systems with LLM-based query generation (HyDE and Query2doc) and the InstructorR with conversational fine-tuning adaption. These systems also underperform CHIQ-FT, which only fine-tunes a small LM on TopiOCQA with enhanced supervision signals. The observation indicates the importance of improving the generalization capabilities of the models to handle complex and diverse conversational scenarios.

## 5.3 Open vs. Close Source LLMs

In addition to conducting experiments with opensource LLMs, we also deploy the closed-source LLM ChatGPT-3.5 to isolate the effects of history enhancement. Table 3 shows the dense retrieval performances on three CAsT test sets when query rewriting (QR) is performed by CHIQ-AD approach on both the original history and the enhanced one. We observe that ChatGPT-3.5 benefits from QR on enhanced history, with NDCG@3 score improvements of 5.4%, 4.3%, and 0.9% across CAsT-19, CAsT-20, and CAsT-21, respectively. Such results indicate that despite the superior reasoning abilities of closed-source LLM, enhancing the conversational history is deemed important for handling complex queries within conversational scenarios.

<table><tr><td rowspan="2">LLM</td><td colspan="2">CAsT-19</td><td colspan="2">CAsT-20</td><td colspan="2">CAsT-21</td></tr><tr><td>MRR</td><td>N@3</td><td>MRR</td><td>N@3</td><td>MRR</td><td>N@3</td></tr><tr><td colspan="7">Original History</td></tr><tr><td>LLaMA Mistral</td><td>67.4</td><td>42.5</td><td>40.9</td><td>27.9</td><td>52.7</td><td>37.2</td></tr><tr><td>ChatGPT</td><td>67.9 69.3</td><td>42.0 40.8</td><td>44.2 53.0</td><td>30.4 36.2</td><td>59.5 60.3</td><td>41.6 41.9</td></tr><tr><td></td><td></td><td></td><td>Our Enhanced History</td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>LLaMA</td><td>70.8</td><td>47.6</td><td>51.0</td><td>34.4</td><td>57.7</td><td>42.0</td></tr><tr><td>Mistral</td><td>71.4</td><td>47.2</td><td>49.2</td><td>34.4</td><td>67.0</td><td>47.2</td></tr><tr><td>ChatGPT</td><td>71.7</td><td>46.4</td><td>55.7</td><td>40.5</td><td>62.2</td><td>42.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Dense retrieval results for systems using various LLMs as backbones, where QR is performed either directly on top of the original conversation history or on our enhanced history using the CHIQ-AD method.

Also, we find that conducting QR on enhanced conversational history helps to narrow the performance gap between open-source and closed-source

LLMs. For instance, the gap of MRR score between LLaMA and ChatGPT-3.5 on the original history is 1.9%, 12.1%, and 7.6% across three CAsT test sets, respectively. In contrast, when utilizing enhanced history, the gaps reduced significantly to 0.9%, 4.7%, and 4.5%, indicating that our designed approach can adequately leverage the capacity of open-source LLMs for conversational search and be competitive with close-source ones.

## 5.4 Search-Oriented Fine-tuning Ablation

We analyze the potential choices for generating search queries in Eq. 2 and Eq. 3 as supervision signals for CHIQ-FT models. Table 4 presents the dense retrieval performances via the queries generated by CHIQ-FT models, which are fine-tuned using manually rewritten queries or the variants of the approach outlined in §3.4. The ablations are based on the results of either without using enhanced history, without generating multiple queries, or not conditioning on the gold passage. We observe that using the queries generated by LLMs as supervision signals outperform the one using manual annotation, which is consistent with previous studies that have identified human-written queries as sub-optimal (Wu et al., 2022; Mo et al., 2023a).

<table><tr><td rowspan="2">Signal</td><td colspan="2">TopiOCQA</td><td colspan="2">QReCC</td></tr><tr><td>MRR</td><td>N@3</td><td>MRR</td><td>N@3</td></tr><tr><td>Manual</td><td>=</td><td></td><td>34.7</td><td>31.9</td></tr><tr><td>CHIQ-FT</td><td>30.0</td><td>28.9</td><td>36.9</td><td>34.0</td></tr><tr><td>W.o. H&#x27;</td><td>27.6</td><td>26.7</td><td>35.4</td><td>33.8</td></tr><tr><td>W.O.  $Q _ { n + 1 } ^ { \prime }$ </td><td>24.2</td><td>23.4</td><td>33.4</td><td>31.7</td></tr><tr><td>W.O.  $p _ { n + 1 } ^ { * }$ </td><td>18.2</td><td>17.2</td><td>26.8</td><td>23.9</td></tr></table>

Table 4: Dense retrieval performances of fine-tuned QR models, which utilize different supervised signals. These include human-written queries and variants from CHIQ-FT, either without enhanced history, without multiple queries, or not conditioned on the gold passage.

Additionally, we observe that all of our proposed adaptive modifications significantly enhance the final performance of the system, especially applying them all to generate the pseudo supervision signals. Such results indicate that improving the quality of supervision signals is crucial for QR model fine-tuning and justifying the effectiveness of our approaches for search-oriented fine-tuning.

## 5.5 History Enhancement Ablation

We study the contribution of each of our proposed prompts in § 3.2 for query enhancement by conducting ablation studies on ′ when one of the prompts is not used. Table 5 presents the ablation performances of dense and sparse retrieval on TopiOCQA and QReCC. The History Summary (HS) is ablated alongside (TS), as HS is not activated if a new topic is detected.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Ablation</td><td colspan="2">TopiOCQA</td><td colspan="2">QReCC</td></tr><tr><td>MRR</td><td>N@3</td><td>MRR</td><td>N@3</td></tr><tr><td rowspan="5">Dense (ANCE)</td><td>CHIQ-AD</td><td>33.2</td><td>32.2</td><td>47.0</td><td>44.6</td></tr><tr><td>W.0. QD</td><td>32.5</td><td>31.4</td><td>44.6</td><td>41.9</td></tr><tr><td>W.O. RE</td><td>28.3</td><td>27.0</td><td>46.6</td><td>44.0</td></tr><tr><td>w.o. PR</td><td>26.4</td><td>25.2</td><td>43.5</td><td>40.8</td></tr><tr><td>w.o. TS</td><td>20.0</td><td>18.7</td><td>46.9</td><td>44.4</td></tr><tr><td rowspan="5">Sparse (BM25)</td><td>CHIQ-AD</td><td>22.5</td><td>20.5</td><td>53.1</td><td>50.7</td></tr><tr><td>W.0. QD</td><td>22.1</td><td>20.1</td><td>47.3</td><td>44.6</td></tr><tr><td>W.o. RE</td><td>19.0</td><td>16.8</td><td>50.1</td><td>47.3</td></tr><tr><td>w.o. TS</td><td>17.8</td><td>16.5</td><td>51.7</td><td>48.8</td></tr><tr><td>W.o. PR</td><td>16.9</td><td>15.3</td><td>46.9</td><td>44.5</td></tr></table>

Table 5: Dense and sparse retrieval results of ablating CHIQ-AD by not using one history enhancement prompt at each line on TopiOCQA and QReCC datasets.

We observe that all our proposed enhancements to the history context contribute positively to the performance of the CHIQ-AD method, although some enhancements are more effective than others. On one hand, detecting topic switching is particularly crucial on TopiOCQA, leading to performance improvements of 13.2% and 5.2% MRR scores in dense and sparse retrieval, respectively. This is mainly due to the multi-topic focus design of the dataset within the same conversation. On the other hand, we notice that while question disambiguation (QD) improves performance, it is less critical compared to predicting a pseudo response (PR) or enhancing the quality of the last system response (RE). In addition, we notice that all our proposed enhancements contribute similarly to the generated search queries across both dense and sparse retrieval settings.

## 5.6 Case Analysis

We manually analyze the content of the enhanced history to better understand the mechanisms and limitations of our approach. This analysis shows the complementary roles each enhancement prompt plays in improving the quality of the original history. QD and RE primarily assist in resolving coreferences and clarifying acronyms to full names, TS helps remove irrelevant content, PR speculates on relevant terms that may occur in the response, and HS not only converts the conversation into plain text but also ensures that key terms from the conversation are preserved. While prompts such as

PR and RE generation generally aid in retrieval, they may also introduce noisy terms due to the wrong fact generated by LLMs that hurt the ranking results. Finally, we also notice that the queries generated by CHIQ-AD and CHIQ-FT are of different styles. The first focuses on expanding more relevant terms to increase the matching scores, while the latter queries are more concise with higher efficiency for retrieval. Nevertheless, aggregating the output rank lists from both approaches helps refine the final results by ranking the relevant passages higher. The concrete examples of these case analyses are presented in Appendix D.

## 6 Conclusion

In this paper, we propose CHIQ, an approach that leverages the basic NLP capabilities of LLMs to enhance the quality of contextual history for improving the query rewriting performance in terms of conversational search. Despite its simplicity, our approach achieves superior performance across various datasets and settings, using open-source LLMs compared to closed-source alternatives. This study shows that instead of simply ask an LLM to generate a search query, it is critical to design strategies to generate different facets of enhancement in view of finding the target information.

## Limitations

Potential limitations of this work include not experimenting with larger open-source LLMs, such as the 56B Mixtral (Mixtral AI team, 2023) or 70B LLaM a, as well as other recent models like Gemma (Team et al., 2024). Additionally, the study did not incorporate more closed-source models such as GPT-4 (OpenAI, 2023) or Claude (AnthropicAI, 2023) to further study the impact of history enhancement. This is mainly due to limitations in computation (open source) and financial (close source) resources. Despite the straightforward and significant gains, some design choices could be further analyzed to potentially boost the performance even more. For instance, adding a backoff filtering strategy could detect when the LLM is producing noisy outputs, or exploring approaches that interpolate between the use of human and pseudoqueries when its higher quality as training signals for CHIQ-FT. Besides, we have considered 5 directions of enhancement in this paper. More strategies can be incorporated so that other useful enhancements can be integrated.

## References

Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries, and Siva Reddy. 2022. Topiocqa: Open-domain conversational question answering with topic switching. Transactions ofthe Associationfor Computational Linguistics, 10:468–483.

Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, and Srinivas Chappidi. 2021. Open-domain question answering goes conversational via question rewriting. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 520–534.

AnthropicAI. 2023. Introducing claude.

Haonan Chen, Zhicheng Dou, Kelong Mao, Jiongnan Liu, and Ziliang Zhao. 2024a. Generalizing conversational dense retrieval via llm-cognition data augmentation. arXiv preprint arXiv:2402.07092.

Pei Chen, Shuai Zhang, and Boran Han. 2024b. Comm: Collaborative multi-agent, multi-reasoningpath prompting for complex problem solving. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1720–1738.

Zhiyu Chen, Jie Zhao, Anjie Fang, Besnik Fetahu, Rokhlenko Oleg, and Shervin Malmasi. 2022. Reinforced question rewriting for conversational question answering.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Y Zhao, Aida Amini, Qazi Mamunur Rashid, Mike Green, and Kelvin Guu. 2022. Dialog inpainting: Turning documents into dialogs. In International Conference on Machine Learning, pages 4558–4586. PMLR.

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2020. Trec cast 2019: The conversational assistance track overview. arXiv preprint arXiv:2003.13624.

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2021. Cast 2020: The conversational assistance track overview. Technical report.

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2022. Trec cast 2021: The conversational assistance track overview. In In Proceedings ofTREC.

Ahmed Elgohary, Denis Peskov, and Jordan Boyd-Graber. 2019. Can you unpack that? learning to rewrite questions-in-context. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5918–5924.

Hung-Chieh Fang, Kuo-Han Hung, Chen-Wei Huang, and Yun-Nung Chen. 2022. Open-domain conversational question answering with historical answers. In Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022, pages 319–326.

Jianfeng Gao, Chenyan Xiong, Paul Bennett, and Nick Craswell. 2022. Neural approaches to conversational information retrieval. arXiv preprint arXiv:2201.05176.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777.

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. 2023. The false promise of imitating proprietary llms. arXiv preprint arXiv:2305.15717.

Kaiyu Huang, Fengran Mo, Hongliang Li, You Li, Yuanchi Zhang, Weijian Yi, Yulong Mao, Jinchen Liu, Yuzhuang Xu, Jinan Xu, et al. 2024. A survey on large language models with multilingualism: Recent advances and new frontiers. arXiv preprint arXiv:2405.10936.

Yunah Jang, Kang-il Lee, Hyunkyung Bae, Seungpil Won, Hwanhee Lee, and Kyomin Jung. 2023. Itercqr: Iterative conversational query reformulation without human supervision. arXiv preprint arXiv:2311.09820.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. CoRR, abs/2310.06825.

Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. 2023. Instructor: Instructing unsupervised conversational dense retrieval with large language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 6649–6675.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547.

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169.

Sungdong Kim and Gangwoo Kim. 2022. Saving dense retriever from shortcut dependency in conversational search. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10278–10287. Association for Computational Linguistics.

Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot query contextualization for conversational search. arXiv preprint arXiv:2204.10613.

Vaibhav Kumar and Jamie Callan. 2020. Making information seeking easier: An improved pipeline for conversational search. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 3971–3980.

Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021a. Pyserini: A python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2356– 2362.

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021b. Contextualized query embeddings for conversational search. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1004–1015.

Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. 2020. Conversational question reformulation via sequence-to-sequence architectures and pretrained language models. arXiv preprint arXiv:2004.01909.

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023. Fine-tuning llama for multi-stage text retrieval. arXiv preprint arXiv:2310.08319.

Kelong Mao, Chenlong Deng, Haonan Chen, Fengran Mo, Zheng Liu, Tetsuya Sakai, and Zhicheng Dou. 2024. Chatretriever: Adapting large language models for generalized and robust conversational dense retrieval. arXiv preprint arXiv:2404.13556.

Kelong Mao, Zhicheng Dou, Haonan Chen, Fengran Mo, and Hongjin Qian. 2023a. Large language models know your contextual search intent: A prompting framework for conversational search.

Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian, Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao Cao. 2023b. Search-oriented conversational query editing. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 4160–4172.

Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022a. Curriculum contrastive context denoising for fewshot conversational dense retrieval. In Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 176–186.

Kelong Mao, Zhicheng Dou, Hongjin Qian, Fengran Mo, Xiaohua Cheng, and Zhao Cao. 2022b. Convtrans: Transforming web search sessions for conversational dense retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2935–2946.

Kelong Mao, Hongjin Qian, Fengran Mo, Zhicheng Dou, Bang Liu, Xiaohua Cheng, and Zhao Cao. 2023c. Learning denoised and interpretable session representation for conversational search. In Proceedings ofthe ACM Web Conference 2023, pages 3193– 3202.

Mixtral AI team. 2023. Mixtral of experts a high quality sparse mixture of experts.

Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, and Jian-Yun Nie. 2023a. Convgqr: Generative query reformulation for conversational search. arXiv preprint arXiv:2305.15645.

Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. 2023b. Learning to relate to previous turns in conversational search. In 29th ACM SIGKDD Conference On Knowledge Discover and Data Mining (SIGKDD).

Fengran Mo, Chen Qu, Kelong Mao, Yihong Wu, Zhan Su, Kaiyu Huang, and Jian-Yun Nie. 2024a. Aligning query representation with rewritten query and relevance judgments in conversational search. arXiv preprint arXiv:2407.20189.

Fengran Mo, Chen Qu, Kelong Mao, Tianyu Zhu, Zhan Su, Kaiyu Huang, and Jian-Yun Nie. 2024b. Historyaware conversational dense retrieval. arXiv preprint arXiv:2401.16659.

Fengran Mo, Bole Yi, Kelong Mao, Chen Qu, Kaiyu Huang, and Jian-Yun Nie. 2024c. Convsdg: Session data generation for conversational search. In Companion Proceedings of the ACM on Web Conference 2024, pages 1634–1642.

Fengran Mo, Longxiang Zhao, Kaiyu Huang, Yue Dong, Degen Huang, and Jian-Yun Nie. 2024d. How to leverage personal textual knowledge for personalized conversational information retrieval. arXiv preprint arXiv:2407.16192.

OpenAI. https://platform.openai.com/docs/models/gpt-3-5-turbo.

OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Hongjin Qian and Zhicheng Dou. 2022. Explicit query rewriting for conversational dense retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4725– 4737.

Chen Qu, Liu Yang, Cen Chen, Minghui Qiu, W Bruce Croft, and Mohit Iyyer. 2020. Open-retrieval conversational question answering. In Proceedings of

the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 539–548.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67.

Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is chatgpt good at search? investigating large language models as re-ranking agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14918–14937.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu, and Raviteja Anantha. 2021. Question rewriting for conversational question answering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pages 355–363.

Christophe Van Gysel and Maarten de Rijke. 2018. Pytrec\_eval: An extremely fast python interface to trec\_eval. In SIGIR. ACM.

Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, and Maarten de Rijke. 2020. Query resolution for conversational search with limited supervision. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 921–930.

Jiayin Wang, Fengran Mo, Weizhi Ma, Peijie Sun, Min Zhang, and Jian-Yun Nie. 2024a. A user-centric benchmark for evaluating large language models. arXiv preprint arXiv:2404.13940.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024b. Improving text embeddings with large language models. CoRR, abs/2401.00368.

Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. arXiv preprint arXiv:2303.07678.

Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reitter, and Gaurav Singh Tomar. 2022. Conqrr: Conversational query rewriting for retrieval with reinforcement learning.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations.

Fanghua Ye, Meng Fang, Shenghui Li, and Emine Yilmaz. 2023. Enhancing conversational search: Large language model-aided informative query rewriting. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5985–6006.

Chanwoong Yoon, Gangwoo Kim, Byeongguk Jeon, Sungdong Kim, Yohan Jo, and Jaewoo Kang. 2024. Ask optimal questions: Aligning large language models with retriever’s preference in conversational search. arXiv preprint arXiv:2402.11827.

Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul Bennett, Jianfeng Gao, and Zhiyuan Liu. 2020. Fewshot generative conversational query rewriting. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 1933–1936.

Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, and Zhiyuan Liu. 2021. Few-shot conversational dense retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 829–838.

Hamed Zamani, Johanne R Trippas, Jeff Dalton, Filip Radlinski, et al. 2023. Conversational information seeking. Foundations and Trends® in Information Retrieval, 17(3-4):244–456.

Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023. Retrieve anything to augment large language models. CoRR, abs/2310.07554.

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. arXiv preprint arXiv:2308.07107.

## A History Enhancement and Query Rewriting Prompts

In this section, we list the prompts that we have carefully designed to enhance different parts of the conversation history, as well as the prompt used for query rewriting and pseudo supervision signals for search-originated fine-tuning. For each prompt, we designed the instruction part through trial and error iterations until we confirmed that both models (LLaMA-2-7B and Mistral-v0.2-7B) could follow the instructions and generate outputs in the required format. We observed no benefits from using in-context examples for any model, as the outputs remained mostly stable, with minor to no changes in the model responses even after adding these examples.

## A.1 Question Disambiguation

You are given a set of question-answers pairs and a new question that is ambiguous. Your goal is to rewrite the question so it becomes clear. Write the new question without any introduction.

## A.2 Response Expansion

Given a series of question-and-answer pairs, along with a new question, your task is to give a one-sentence response to the new question.

## A.3 Pseudo Response

You are given a question-and-answer pair, where the answer is not clear. Your goal is to write a long version of the answer based on its given context. The generated answer should be one sentence only and less than 20 words.

## A.4 Topic Switch

Given a series of question-and-answer pairs, along with a new question, your task is to determine whether the new question continues the discussion on an existing topic or introduces a new topic. Please respond with either "new\_topic" or "old\_topic" as appropriate.

## A.5 History Summary

You are given a context in the form of question-answer pairs. Your goal is to write a paragraph that summarizes the information in the context. The summary should be short with one sentence for each question answer pair.

## A.6 Query Rewriting

Given a series of question-and-answer pairs as context, along with a new question, your task is to convert the new question into a search engine query that can be used to retrieve relevant documents. The output should be placed in a JSON dictionary as follows: {"query": ""}

## A.7 Pseudo Supervision Signals

You are given a relevant passage, a series of question-and-answer pairs as context along with a new question, your task is to generate a set of search queries based on the relevancy between the new question and the relevant passage and also rely on the given context. The output format should be in a list with indexes e.g., 1. 2. 3.

## B Experimental Setup

## B.1 Datasets Details

<table><tr><td>Dataset</td><td>Split</td><td>#Conv.</td><td>#Turns(Qry.)</td><td>#Collection</td></tr><tr><td rowspan="2">TopiOCQA</td><td>Train</td><td>3,509</td><td>45,450</td><td rowspan="2">25M</td></tr><tr><td>Test</td><td>205</td><td>2,514</td></tr><tr><td rowspan="2">QReCC</td><td>Train</td><td>10,823</td><td>29,596</td><td rowspan="2">54M</td></tr><tr><td>Test</td><td>2,775</td><td>8,124</td></tr><tr><td>CAsT-19</td><td>Test</td><td>50</td><td>479</td><td>38M</td></tr><tr><td>CAsT-20</td><td>Test</td><td>25</td><td>208</td><td></td></tr><tr><td>CAsT-21</td><td>Test</td><td>26</td><td>239</td><td>40M</td></tr></table>

Table 6: Statistics of conversational search datasets.

The statistics of each dataset are presented in Table 6. We discard the samples without gold passages. The manually rewritten query for each turn is provided in all datasets except TopiOCQA. The relevance judgments in the CAsT datasets are made by experts with multi-level annotations. The relevance judgment thresholds are set at 1, 2, and 2 for CAsT-19, CAsT-20, and CAsT-21, respectively.

## B.2 Implementation Details

We implement the retrieval evaluation metrics from the pytrec\_eval tool (Van Gysel and de Rijke, 2018). We leverage the Pyserini (Lin et al., 2021a) and Faiss (Johnson et al., 2019) libraries for implementing the BM25 and ANCE retrievers, respectively. Following previous works (Lin et al., 2021b; Mo et al., 2023a), we set BM25 parameters as follows: $k _ { 1 } = 0 . 9 , b = 0 . 4$ on TopiOCQA and $k _ { 1 } = 0 . 8 2 , b = 0 . 6 8$ on the QReCC. The lengths of the query, concatenated input, and passage are truncated to 32, 512, and 384 tokens, respectively.

In all experiments, we use sampling and set the temperature to 0.7 when generating with LLMs.

For the search-oriented fine-tuning, we use the NDCG@3 score as the standard metric to select the generated query as the pseudo supervision signals, while we set the maximum length for the generated query is set to 32, which is the same as (Lin et al., 2020; Wu et al., 2022; Mo et al., 2023a). For the rank-list fusion, we set the balance factor α in Lin et al. (2021b) as 1, which indicates the same importance of different retrieved results.

## B.3 Baselines Details

We provide a more detailed introduction to the following baselines used for comparison:

QuReTeC (Voskarides et al., 2020): A traditional sequence tagger query rewriting approach fine-tuned with weakly supervision signals to determine whether a term in a historical context should be expanded to the current query.

T5QR (Lin et al., 2020): A query rewriting approach fine-tuned with manual annotations provided in QReCC as the supervised signals via the T5-base model.

CONQRR (Wu et al., 2022): A query rewriting approach fine-tuned with manual annotations provided in QReCC and the ranking signals using reinforcement learning via the T5-base model.

ConvGQR (Mo et al., 2023a): A unified framework that integrates query rewriting and query expansion mechanisms by two T5-base models and fine-tuned them with manual annotations and ground-truth response, respectively.

EDIRCS (Mao et al., 2023b): A query rewriting approach based on text editing technique with ranking signals fine-tuned on the T5-base model.

IterCQR (Jang et al., 2023): An iterative query rewriting method using the initial query generated by the ChatGPT-3.5 and refining a query turn multiple times according to the ranking signals feedback during the training stage.

LLM-Aided (Ye et al., 2023): An informative conversational query rewriting by directly prompting ChatGPT-3.5 as both query rewriters and rewrite editors twice to incorporate all the desirable properties for producing the final rewritten queries.

RETPO (Yoon et al., 2024): A retriever preference adapted query rewriting method that finetunes LLaMA-2-7B as a QR model with an external QR dataset generated by GPT-4.

ConvDR (Yu et al., 2021): A traditional conversational dense retrieval method that uses knowledge distillation to learn the session embeddings with relevance judgments from the humanrewritten queries based on the ANCE model.

InstructorR (Jin et al., 2023): A LLM-based general conversational dense retriever tailored to various tasks and domains by fine-tuned with various task-specific instructions and relevance judgments based on FlanT5-XL model.

RepLLaMA (Ma et al., 2023): A large ad-hoc dense retriever fine-tuned on top of the LLaMA-7B model on the MSMARCO dataset.

E5-Mistral (Wang et al., 2024b): A large ad-hoc retriever fine-tuned on top of Mistral-7B model on the synthetic dataset generated by ChatGPT-3.5 and MSMARCO.

LLM-Embedder (Zhang et al., 2023): A unified retrieval model that can support diverse retrieval augmentation needs of LLMs, which is finetuned on various tasks and datasets such as MS-MARCO, NQ, ToolLLM, QReCC, FLAN, Books3, and Multi-Session Chat.

HyDE (Gao et al., 2023): A zero-shot retrieval method, which adopts ChatGPT-3.5 to generate hypothetical documents for the query, then retrieves real documents with hypothetical documents.

Query2doc (Wang et al., 2023): A zero-shot query expansion approach, which expands the original query with the generated documents from ChatGPT-3.5.

LLM4CS (Mao et al., 2023a): A state-of-the-art LLM-based prompting method for conversational query rewriting. We implement it with full aggregation by calling LLMs five times for the query and response generation but without the chain-ofthought (CoT) content because of the efficient annotation consideration in practical scenarios.

## C Results of Mistral-2-7B

Table 7 and Table 8 show the performances of our methods using Mistral-2-7B as the backbone LLM, replacing LLama-2-7B as previously shown in Table 1 and Table 2, respectively. These results exhibit trends similar to those observed with LLama-2-7B, as discussed in Sections 5.1 and 5.2. More specifically, CHIQ-AD consistently outperforms CHIQ-FT across all settings, and combining the results of both methods (CHIQ-Fusion) yields the best performance. It is worth noting that results with Mistral-2-7B are systematically lower than the ones with LLama-2-7B on most datasets (except on CAsT-21). Besides, by comparing the fusion performance between Mistrial and LLaMa, we notice that in most cases when the gap between CHIQ-AD and CHIQ-FT is large, CHIQ-Fusion results are either slightly better or worse than CHIQ-AD. This is mainly because the poor quality of the rank list obtained by CHIQ-FT negatively impacts the one from CHIQ-AD. However, when the gap is smaller, we notice a significant gain for CHIQ-Fusion, suggesting that both variants are generating good and complementary rank lists. It will be interesting to investigate how we can better take advantage of CHIQ-AD and CHIQ-FT in an adaptive fusion. Nevertheless, performing QR on top of the enhanced history with our approach still outperforms most other settings and datasets.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Systems</td><td colspan="3">TopiOCQA</td><td colspan="3">QReCC</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@10</td><td>MRR</td><td>NDCG@3</td><td>Recall@10</td></tr><tr><td rowspan="3">Dense</td><td>CHIQ-FT</td><td>26.2</td><td>25.4</td><td>45.3</td><td>31.1</td><td>28.5</td><td>50.6</td></tr><tr><td>CHIQ-AD</td><td> $2 8 . 9 ^ { \dagger }$ </td><td> $2 8 . 3 ^ { \dagger }$ </td><td> $4 6 . 8 ^ { \dagger }$ </td><td> $4 6 . 7 ^ { \dagger }$ </td><td> $4 4 . 2 ^ { \dagger }$ </td><td> $7 0 . 7 ^ { \dag }$ </td></tr><tr><td>CHIQ-Fusion</td><td> $\mathbf { 3 6 . 3 \AA ^ { \dag } }$ </td><td> $\mathbf { 3 5 . 0 } _ { } ^ { \dagger }$ </td><td>59.6†</td><td> $4 7 . 1 ^ { \dag }$ </td><td> $4 4 . 1 ^ { \dagger }$ </td><td> $7 0 . 3 ^ { \dagger }$ </td></tr><tr><td rowspan="3">Sparse</td><td>CHIQ-FT</td><td>15.2</td><td>14.3</td><td>30.5</td><td>32.9</td><td>30.0</td><td>51.4</td></tr><tr><td>CHIQ-AD</td><td>19.2†</td><td>17.3†</td><td>35.6†</td><td>51.7†</td><td>48.8†</td><td>76.2†</td></tr><tr><td>CHIQ-Fusion</td><td> $2 1 . 4 ^ { \dagger }$ </td><td> $1 9 . 2 ^ { \dagger }$ </td><td>39.4†</td><td> ${ \bf 5 1 . 9 ^ { \dag } }$ </td><td> $\mathbf { 4 9 . 0 } _ { } ^ { \dagger }$ </td><td> ${ \mathbf { 7 6 . 3 } } ^ { \dagger }$ </td></tr></table>

Table 7: Performance of dense and sparse retrieval on TopiOCQA and QReCC datasets based on Mistral-2-7B model. The system properties and the settings are inherited from the Table 1.
<table><tr><td rowspan="2">Systems</td><td colspan="3">CAsT-19</td><td colspan="3">CAsT-20</td><td colspan="3">CAsT-21</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>R@10</td><td>MRR</td><td>NDCG@3</td><td>R@10</td><td>MRR</td><td>NDCG@3</td><td>R@10</td></tr><tr><td>CHIQ-FT</td><td>58.3</td><td>35.4</td><td>9.0</td><td>37.1</td><td>24.7</td><td>12.0</td><td>44.4</td><td>29.1</td><td>16.5</td></tr><tr><td>CHIQ-AD</td><td>71.4†</td><td>47.2†</td><td>12.2†</td><td>49.2</td><td>34.4</td><td>17.5</td><td>67.0†</td><td>47.2†</td><td>25.5†</td></tr><tr><td>CHIQ-Fusion</td><td>71.5†</td><td>47.7†</td><td>11.9†</td><td>51.1</td><td>35.5</td><td>18.4</td><td>66.1</td><td>48.9†</td><td>27.5†</td></tr></table>

Table 8: Performance of dense retrieval on three CAsT datasets based on Mistral-2-7B model. The system properties and the settings are inherited from the Table 2.

sages being ranked higher compared to those based on the original history (LLM-QR). Similar trends are observed in the second case shown in Table 10, where CHIQ-FT and CHIQ-AD outperform QR with the original history. This improvement may be attributed to the enhanced history increasing semantic similarity through references to titles of Tjader’s artworks from PR and HS, such as “Inauguration of the Pleasure Dome”. However, names of various collaborators like “Kenneth Anger and Stan Brakhage” may introduce noise in PR.

## D Case Analysis

Table 9 and Table 10 showcase two examples that support the case study analysis conducted in § 5.6. In the example of Table 9, QD and RE QD and RE contribute by adding the terms “hormone” into “cortisol, glucagon, adrenaline, cytokines orexin, and melatonin” to the enhanced history. In addition, PR enriches the emotional context by including “excitement, anxiety, or fear”, which co-occurs in the gold positive passage, thereby improving the scores. Since a topic switch (TS) is detected, earlier turns containing noisy terms, such as “Adenosine triphosphate (ATP)”, are dropped from the history summary (HS). Therefore, performing QR with our methods on the enhanced history results in top pas-

<table><tr><td rowspan=1 colspan=3>Original History</td><td rowspan=1 colspan=1>Our Enhanced History</td></tr><tr><td rowspan=1 colspan=3> $u _ { 1 } \colon$ The primary high energy mole-cule in human metabolism is? $r _ { 1 } :$ Adenosine Triphosphate (ATP).</td><td rowspan=3 colspan=1>QD: Which hormone among cortisol, glucagon, adrenaline, cytokines orexin,and melatonin is associated with an emotional response?    TS: YesRE: These hormones, including cortisol, glucagon, adrenaline, cytokines, ore-xin, and melatonin, play various roles in regulating metabolic processes invo-lving the breakdown of larger molecules to produce ATP during catabolism.PR: The emotional responses to hormones can vary greatly among individuals.Adrenaline (epinephrine) is often associated with excitement, anxiety, or fear.HS: Adenosine triphosphate (ATP) serves as the primary high energy moleculein human metabolism. Catabolism refers to the metabolic processes that breakdown large molecules into smaller ones. Cortisol, glucagon, adrenaline, cyto-kines, orexin, and melatonin are associated hormones involved in this process.</td></tr><tr><td rowspan=1 colspan=3> $u _ { 2 } \colon$ What is catabolism? $r _ { 2 } \colon$ It is the set of metabolic processesthat breaks down large molecules. $u _ { 3 } \colon$ Which hormones are related to it? $r _ { 3 } \colon$ Cortisol, Glucagon, Adrenaline,</td></tr><tr><td rowspan=1 colspan=3> $u _ { 4 } \colon$ What is the emotional responsedue to the third one?</td><td rowspan=1 colspan=1>Cytokir</td></tr><tr><td rowspan=1 colspan=3>Rewritten Query $\underline { { q _ { 4 } ^ { \prime } } }$ </td><td rowspan=1 colspan=1>Gold Positive Passage $\underline { { p _ { 4 } ^ { * } } }$ </td></tr><tr><td rowspan=1 colspan=3>LLM-QR: What is the emotionalresponse associated with the thirdhormone? (Rank: 41)CHIQ-AD: QD + PR + RE (Rank: 4)CHIQ-FT: How does adrenalineimpact mood? (Rank: 3)CHIQ-Fusion (Rank: 1)</td><td rowspan=1 colspan=1>Every emotional response has a behavioral component, an autonomic compon-ent, and a hormonal component. The hormonal component includes the releaseof adrenaline, an adrenomedullary response that occurs in response to stress andthat is controlled by the sympathetic nervous system. The major emotion studi-ed in relation to adrenaline is fear. In an experiment, subjects who were injectedwith adrenaline expressed more negative and fewer positive facial expressionsto fear films compared to a control group.</td></tr></table>

Table 9: The first case for analyzing the content between the original and our enhanced history, as well as the rewritten queries generated on top of them. The blue tokens and the orange tokens stand for the effective and noisy patterns for matching, respectively. The underline tokens denote the coreference relation between rewritten queries and the original context. The Rank indicates the ranking position of the gold positive passage w.r.t each query.

<table><tr><td rowspan=1 colspan=6>Original History</td><td rowspan=1 colspan=1>Our Enhanced History</td></tr><tr><td rowspan=4 colspan=6> $u _ { 1 } \colon$ Are there any interesting aspects about Cal Tjader? $r _ { 1 } :$ Cal Tjader entered the United States Navy in 1943served as a medical corpsman in the Pacific Theateruntil March 1946. $u _ { 2 } \colon$ Were any albums released?r2: At San Francisco State he met Dave Brubeck, ayoung pianist fresh from a stint in the Army. Brubeckintroduced Tjader to Paul Desmond. $u _ { 3 } \colon$ What were the jazz album released $\operatorname { f o r } ?$ r3: Cal Tjader continued the trio work in California</td><td rowspan=6 colspan=1>QD: What was the outcome or reception of Kenneth Anger&#x27;s collab-oration with Stan Brakhage on their film project in the United States?PR: It is unclear if Kenneth Anger and Stan Brakhage&#x27;s confiscatedfilm was successful as it is presumed to have been destroyed.However, Anger&#x27;s later film, Inauguration of the Pleasure Dome(1954), received critical acclaim for its themes.    TS: NoHS: In 1953, Kenneth Anger returned to the United States followinghis mother&#x27;s death to handle her estate. There, he befriended StanBrakhage and together they produced a controversial film, which wasconfiscated and likely destroyed due to its obscene content. Despitethis setback, Anger went on to create his groundbreaking 38-minutesurrealist work, &quot;Inauguration of the Pleasure Dome,&quot; in 1954,showcasing Crowleyan and Thelemite themes.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>confiscated and lik</td></tr><tr><td rowspan=1 colspan=2> $\cdots$ recording his first</td><td rowspan=1 colspan=4>with them for Fantasy Records.</td><td rowspan=1 colspan=1>this setback, Anger we</td></tr><tr><td rowspan=1 colspan=6> $u _ { 4 } \colon$ What was a title of one of the albums?</td></tr><tr><td rowspan=1 colspan=6>Rewritten Query $\overline { { q _ { 4 } ^ { \prime } } }$ </td><td rowspan=1 colspan=1>Gold Positive Passage $p _ { 4 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=6>LLM-QR: Which album did Cal Tjader record with</td><td rowspan=6 colspan=1>The most obvious deviation from Tjader&#x27;s Latin jazz sound wasSeveral Shades of Jade and the follow-up Breeze From the East.Both albums attempted to combine jazz and Asian music, much asTjader and others had done with Afro-Cuban. The result was ...Other experiments were not so easily dismissed. Tjader teamed upNew Yorker Eddie Palmieri in 1966 to produce El SonidoNuevo A companion LP was recorded for Palmieri&#x27;s ...</td></tr><tr><td rowspan=1 colspan=6>Jack Weeks and John Marabuto in California?(Rank:16)</td></tr><tr><td rowspan=1 colspan=6>CHIQ-AD: QD + PR + HS (Rank: 7)CHIQ-FT: Which specific album title did Cal Tjader</td></tr><tr><td rowspan=1 colspan=6>record with Jack Weeks on bass and either John</td></tr><tr><td rowspan=1 colspan=6>with Marabuto or Vince Guaraldi on piano? (Rank: 10)</td></tr><tr><td rowspan=1 colspan=6>CHIQ-Fusion: (Rank: 5)</td></tr></table>

Table 10: The second case for analyzing the content between the original and our enhanced history, as well as the rewritten queries generated on top of them. The indication is consistent with Table 9.