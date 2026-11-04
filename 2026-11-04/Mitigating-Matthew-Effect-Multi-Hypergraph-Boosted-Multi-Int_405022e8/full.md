# Mitigating Matthew Effect: Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation

Yongsen Zheng<sup>1,2</sup>, Ruilin Xu<sup>3</sup>, Guohua Wang<sup>4†</sup>, Liang Lin<sup>3,5</sup>, Kwok-Yan Lam<sup>1,2†</sup> <sup>1</sup>Nanyang Technological University, Singapore <sup>2</sup>Digital Trust Centre Singapore <sup>3</sup>Sun Yat-sen University <sup>4</sup>South China Agricultural University <sup>5</sup>Peng Cheng Laboratory {yongsen.zheng, kwokyan.lam}@ntu.edu.sg, xurlin5@mail2.sysu.edu.cn wangguohua@scau.edu.cn, linliang@ieee.org

## Abstract

The Matthew effect is a big challenge in Recommender Systems (RSs), where popular items tend to receive increasing attention, while less popular ones are often overlooked, perpetuating existing disparities. Although many existing methods attempt to mitigate Matthew effect in the static or quasi-static recommendation scenarios, such issue will be more pronounced as users engage with the system over time. To this end, we propose a novel framework, Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation (HiCore), aiming to address Matthew effect in the Conversational Recommender System (CRS) involving the dynamic user-system feedback loop. It devotes to learn multi-level user interests by building a set of hypergraphs (i.e., item-, entity-, word-oriented multiple-channel hypergraphs) to alleviate the Matthew effec. Extensive experiments on four CRS-based datasets showcase that HiCore attains a new state-of-the-art performance, underscoring its superiority in mitigating the Matthew effect effectively. Our code is available at https://github.com/zysensmile/HiCore.

## 1 Introduction

Engaging users in ongoing conversations for personalized recommendations, Conversational Recommendation Systems (CRSs) (Qin et al., 2023; Mishra et al., 2023) have become a prevalent strategy utilized in diverse fields (Liu et al., 2023; Epure and Hennequin, 2023). However, CRSs often face a big challenge known as Matthew effect (Liu and Huang, 2021), captured by the adage "the privileged gain more privilege, while the underprivileged fall further behind." This observation underscores that well-received items or categories in past records garner heightened visibility in future suggestions, whereas less preferred ones frequently face neglect or marginalization.

Lately, a multitude of studies have focused on investigating the Matthew effect in relatively unchanging offline recommendation scenarios (Liu and Huang, 2021; Anderson et al., 2020), identifying two root causes for its occurrence. One cause (Liang et al., 2021; Zheng et al., 2021a; Hansen et al., 2021; Anderson et al., 2020) is the heightened vulnerability of individuals with narrower and uniform preferences or interests to succumb to the pervasive influence of the Matthew effect. This susceptibility often stems from a tendency towards familiarity and comfort, leading to a reinforcement of existing patterns and a limited exploration of diverse alternatives. Another cause (Zheng et al., 2021b) arises from the pervasive favoritism towards mainstream items, resulting in a perpetual reinforcement of their prominence, while lesser-known alternatives linger in the shadows. This bias towards popular choices not only perpetuates existing trends but also limits the discoverability of niche or underappreciated options. Thus, the amplification of visibility for widely favored items can overshadow the potential value and diversity offered by less popular but equally deserving alternatives.

Despite their effectiveness, most existing methods still suffer from two major limitations. 1) Interactive Strategy. While many methods have offered valuable insights into the Matthew effect, they often overlook the adverse effects originating from the dynamic user-system feedback loop (Zhang et al., 2021), as they primarily focus on mitigating the Matthew effect in relatively stable offline recommendation settings. In fact, the Matthew effect can intensify as users interact more actively with the system over time, potentially exacerbating concerns such as echo chambers (Ge et al., 2020) and filter bubbles (Steck, 2018). Hence, it is important to address the Matthew effect in the CRS. 2) Interest Exploration. Considering that the root cause of the Matthew effect lies in the confinement of user interests (Zheng et al., 2021a; Liang et al., 2021; Hansen et al., 2021; Anderson et al., 2020), most existing methods focus on leveraging hypergraphs to unveil complex high-order user relationship patterns for exploring user interests. However, these hypergraphs often remain single-channel, constraining their capacity to capture diverse user relation patterns since each hypergraph can only represent a specific type of user patterns. Moreover, these single-channel hypergraphs may risk evolving into traditional Knowledge Graphs (KGs) due to the scarcity of user-item interaction data. Thus, the construction of multi-channel hypergraphs is paramount for exploring multi-level user interests.

To address these limitations, we propose the novel framework, Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation (HiCore), which aims to mitigate the negative impact of Matthew effect when users engage with the system over time in the CRS. It is comprised of Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning and Interest-Boosted CRS. The former devotes to construct multi-hypergraph (i.e., item-oriented, entity-oriented, and word-oriented triple-channel hypergraphs) to learn multi-level user interests (i.e., item-level, entity-level, word-level triple-channel interests), where triple channels contain the group, joint, and purchase channels. The latter aims to utilize the multi-level interests to enhance both conversation and recommendation tasks when users chat with system over time. Concretely, multi-level user interests are used to effectively generate next utterances in the conversational task, and accurately predict users’ interested items in the recommendation task. Extensive experimental results on four benchmarks show that HiCore achieves a new state-of-the-art performance compared all the baselines, and the effectiveness of mitigating Matthew effect in the CRS.

Overall, our main contributions are included:

• To the best of our knowledge, this is the first work to build multi-hypergraph from triple-channel settings for learning multi-interest to mitigate Matthew effect in the CRS.

• We proposed a novel end-to-end framework HiCore, aiming to use multi-interest enhance both recommendation and conversation tasks.

• Quantitative and qualitative experimental results show the effectiveness of HiCore and the superiority of HiCore in mitigating Matthew effect.

## 2 Related Work

## 2.1 Matthew Effect in Recommendation

The Matthew effect poses a formidable challenge in recommendation systems. To combat this issue, there are two primary research lines. One line of research focuses on understanding a diverse range of user interests to enhance recommendation diversification (Anderson et al., 2020; Hansen et al., 2021; Liang et al., 2021; Zheng et al., 2021a). The other line of research (Zheng et al., 2021b) is dedicated to mitigating popularity bias to ensure a balanced exposure of items across various categories. For example, Wang et al. (Wang et al., 2019) conducted a meticulous quantitative analysis, providing valuable insights into the quantitative characteristics of the Matthew effect in collaborative-based recommender systems. Liu et al. (Liu and Huang, 2021) have confirmed the presence and impact of the Matthew effect within the intricate algorithms of YouTube’s recommendation system. However, these methods primarily concentrate on exploring the Matthew effect in static recommendation environments, overlooking the crucial interplay of the user-system feedback loop.

## 2.2 Conversational Recommender System

Conversational Recommendation System aims to uncover users’ genuine intentions and interests through natural language dialogues, thereby offering top-notch recommendations to users. Currently, CRS-based methods can be categorized into two main groups. 1) Attribute-based CRS (Deng et al., 2021a; Lei et al., 2020a,b; Ren et al., 2021; Xu et al., 2021), which seeks to delve into user interests by posing queries about items or their attributes. However, this approach primarily relies on predefined templates for response generation, often falling short in producing fluent, human-like natural language expressions. 2) Generated-based CRS (Chen et al., 2019; Deng et al., 2023; Li et al., 2022; Zhou et al., 2020a, 2022; Shang et al., 2023), which can address the shortcomings of attributecentric CRS by utilizing the Seq2Seq architecture (Vaswani et al., 2017a) to integrate a conversation component and a recommendation component, resulting in the creation of smooth and coherent human-like responses. Despite their effectiveness, they face challenges in grasping the varied interests of users because of the restricted and scarce character of user-item interaction data.

![](images/0aa40c22cb150da9cef0b1f6094c04bf388196e3913eadff288e2ce6e30b225d.jpg)  
Figure 1: Overview of our HiCore framework. It consists of Multi-Hypergraph Boosted Multi-Interest Self Supervised Learning and Interested-Boosted CRS. The former aims to learn multi-level user interests, while the latter devotes to generate responses in the conversation module and predict items in the recommendation module.

## 3 HiCore

Most existing methods (Hussein et al., 2020; Liu et al., 2021a; Nguyen et al., 2014) have consistently revealed that individuals with constrained interests are greatly impacted by Matthew effect. Thus, we propose a novel framework, HiCore, which is comprised of Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning and Interest-Boosted CRS. The overall pipeline of the proposed HiCore is illustrated in Fig.1.

## 3.1 Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning

In this section, we will establish multi-hypergraph to learn multi-level user interests to mitigate Matthew effect in the CRS.

## 3.1.1 Multi-Hypergraph Boosts Multi-Interest

Instead of linking only two nodes per edge as in traditional KGs, hypergraphs extend the notion of edges to connect more than two nodes. By utilizing diverse hypergraphs to encode various highorder user relation patterns, we construct multiple knowledge-oriented triple-channel hypergraphs.

Item-oriented triple-channel Hypergraphs. We first build item-oriented hypergraphs from triple channels, i.e., ‘Group Channel $( g ) ^ { \prime }$ , ‘Joint Channel $( j ) ^ { \flat }$ , and ‘Purchase Channel $( p ) ^ { \prime }$ via the Motif (Milo et al., 2002; Yu et al., 2021), a commonly utilized tool for capturing complex local structures involving multiple nodes, as illustrated in Fig.2.

![](images/90137f09b1816f35fa0cd3a50007b79bfa3ec925b6fec0815b6a7e615d18bd5c.jpg)  
Figure 2: Triangle motifs used in our proposed HiCore.

Group-channel hypergraph. Group-channel hypergraphs aim to analyze users’ social relations to unveil the dynamics among individuals based on their shared interests, preferences, and characteristics. Understanding group preferences not only consolidates individual tastes but also facilitates collective decisions that benefit the entire group. Formally, we utilize a set of triangular motifs (Milo et al., 2002; Yu et al., 2021) to build the item-oriented group-channel hypergraphs $\mathsf { G } _ { g } ^ { ( i ) }$ as:

$$
\mathsf { G } _ { g } ^ { ( i ) } = ( \mathcal { V } _ { g } ^ { ( i ) } , \mathcal { N } _ { g } ^ { ( i ) } , \boldsymbol { A } _ { M _ { k } ^ { g } } ^ { ( i ) } ) .\tag{1}
$$

Here $\mathcal { V } _ { g } ^ { ( i ) }$ represents the set of items derived from the historical conversations, while $\mathcal { N } _ { g } ^ { ( i ) } =$ $\{ M _ { k } ^ { g } | 1 \le k \le 7 \}$ denotes the collection of hyperedges, with each hyperedge representing an occurrence of the specified motif $\boldsymbol { M } _ { k } ^ { g }$ in $\mathrm { F i g } . 2$ $A _ { M _ { k } ^ { g } } ^ { ( i ) } \in | \mathcal { V } _ { g } ^ { ( i ) } | \times | \mathcal { N } _ { g } ^ { ( i ) } |$ is the group-motif-induced adjacency matrices. Firstly, we need to define the matrix computation of each type of motif. Let $\pmb { H } _ { k } ^ { ( i ) }$ be the matrix computation of the motif $\boldsymbol { M } _ { \boldsymbol { k } } ^ { g }$ , then we can obtain:

$$
\left\{ \begin{array} { l l } { H _ { 1 } ^ { ( i ) } = ( I ^ { T } J ) \otimes I ^ { T } + ( J I ) \otimes I + ( I I ^ { T } ) \otimes J , } \\ { H _ { 2 } ^ { ( i ) } = ( I J ) \otimes I + ( J I ^ { T } ) \otimes I ^ { T } + ( I ^ { T } I ) \otimes J , } \\ { H _ { 3 } ^ { ( i ) } = ( I I ) \otimes I + ( I I ^ { T } ) \otimes I + ( I ^ { T } I ) \otimes I , } \\ { H _ { 4 } ^ { ( i ) } = ( J J ) \otimes J , } \\ { H _ { 5 } ^ { ( i ) } = ( J J ) \otimes I + ( J I ) \otimes J + ( I \cdot J ) \otimes J , } \\ { H _ { 6 } ^ { ( i ) } = ( J I ) \otimes I ^ { T } + ( I J ) \otimes I ^ { T } + ( I I ) \otimes J , } \\ { H _ { 7 } ^ { ( i ) } = ( I I ) \otimes I ^ { T } , } \end{array} \right.\tag{2}
$$

where $\otimes$ is the element-wise product, S denote the relation matrix (Yu et al., 2021). $\pmb { J } = \pmb { S } \otimes \pmb { S }$ and $\pmb { I } = \pmb { S } - \pmb { J }$ specify the adjacency matrices of the bidirectional and unidirectional social networks (i.e., group motif), respectively. Then, the groupmotif-induced adjacency matrices $A _ { M _ { k } ^ { g } } ^ { ( i ) }$ is:

$$
\begin{array} { r } { A _ { M _ { k } ^ { ( i ) } } ^ { ( i ) } = \left\{ \begin{array} { l l } { H _ { 1 } ^ { ( i ) } , } & { \mathrm { i f } \quad M _ { 1 } ^ { g } , } \\ { H _ { 2 } ^ { ( i ) } , } & { \mathrm { i f } \quad M _ { 2 } ^ { g } , } \\ { H _ { 3 } ^ { ( i ) } + ( H _ { 5 } ^ { ( i ) } ) ^ { T } , } & { \mathrm { i f } \quad M _ { 3 } ^ { g } , } \\ { H _ { 4 } ^ { ( i ) } , } & { \mathrm { i f } \quad M _ { 4 } ^ { g } , } \\ { H _ { 5 } ^ { ( i ) } + ( H _ { 3 } ^ { ( i ) } ) ^ { T } , } & { \mathrm { i f } \quad M _ { 5 } ^ { g } , } \\ { H _ { 6 } ^ { ( i ) } + ( H _ { 2 } ^ { ( i ) } ) ^ { T } , } & { \mathrm { i f } \quad M _ { 6 } ^ { g } , } \\ { H _ { 7 } ^ { ( i ) } + ( H _ { 1 } ^ { ( i ) } ) ^ { T } , } & { \mathrm { i f } \quad M _ { 7 } ^ { g } . } \end{array} \right. } \end{array}\tag{3}
$$

If $( \boldsymbol { A } _ { M _ { k } ^ { g } } ^ { ( i ) } ) _ { n , r } = 1$ , it signifies that the node n and the node r co-occur in a single instance of $\boldsymbol { M } _ { \boldsymbol { k } } ^ { g }$ When two nodes appear in multiple instances, it turns to be $( { \cal A } _ { M _ { k } ^ { g } } ^ { ( i ) } ) _ { n , r } \ =$ #(n, r occur in the same instance of $M _ { k } ^ { g } )$

Joint-channel hypergraph. The joint channel reflects the scenario of shared behaviors among friends in a social network. When friends purchase the same items, it not only suggests similarities in tastes and interests but also hints at deeper levels of interaction and trust. This phenomenon of "friends purchasing the same item" may facilitate information dissemination and interaction within the social network, strengthening social relationships, and to some extent, reflecting influence and collective behavior within the social network. Therefore, by identifying and analyzing the joint motifs, the itemoriented joint-channel hypergraph $\mathsf { G } _ { j } ^ { ( i ) }$ is:

$$
\left\{ \begin{array} { l l } { \boxed { \hat { \mathbf { G } } _ { j } ^ { ( i ) } = ( \gamma _ { j } ^ { ( i ) } , \mathcal { N } _ { j } ^ { ( i ) } , A _ { M _ { k } ^ { i } } ^ { ( i ) } ) , } } \\ { H _ { 8 } ^ { ( i ) } = ( R R ^ { T } ) \otimes J , } \\ { H _ { 9 } ^ { ( i ) } = ( R R ^ { T } ) \otimes I , } \\ { A _ { M _ { k } ^ { j } } ^ { ( i ) } = H _ { 8 } ^ { ( i ) } , \qquad \mathrm { i f } \quad M _ { 8 } ^ { j } , } \\ { A _ { M _ { k } ^ { j } } ^ { ( i ) } = H _ { 9 } ^ { ( i ) } + [ H _ { 9 } ^ { ( i ) } ] ^ { T } , \quad \mathrm { i f } \quad M _ { 9 } ^ { j } , } \end{array} \right.\tag{4}
$$

where $\mathcal { V } _ { j } ^ { ( i ) }$ , and $\mathcal { N } _ { i } ^ { ( i ) } = \{ M _ { k } ^ { j } | 8 \leq k \leq 9 \}$ denote the item set, and the hyperedge set, respectively. Each hyperedge is induced from each type of joint motif, depicted in Fig.2. R is a binary matrix that records user-item interactions, and $\dot { \boldsymbol { A } } _ { M _ { k } ^ { j } } ^ { ( i ) }$ denotes the joint-motif-induced adjacency matrices.

Purchase-channel hypergraph. Additionally, we should also take into account users who do not have explicit social connections. Therefore, the analysis is non-exclusive and delineates the implicit higher-order social relationships among users who lack direct social ties but still purchase the same items. By considering these users without overt social links, we can uncover hidden patterns of social influence and affiliation that transcend traditional network structures. Thus, the item-oriented purchase-channel hypergraph $\mathsf { G } _ { p } ^ { ( i ) }$ can be induced from the purchase motif $M _ { 1 0 } ^ { p }$ as follows:

$$
\begin{array} { r l } & { \mathsf { G } _ { p } ^ { ( i ) } = ( \mathcal { V } _ { p } ^ { ( i ) } , \mathcal { N } _ { p } ^ { ( i ) } , \boldsymbol { A } _ { M _ { k } ^ { p } } ^ { ( i ) } ) , } \\ & { \boldsymbol { A } _ { M _ { k } ^ { p } } ^ { ( i ) } = \boldsymbol { H } _ { 1 0 } ^ { ( i ) } = \boldsymbol { R } \boldsymbol { R } ^ { T } , \quad \mathrm { i f } \quad M _ { 1 0 } ^ { p } , } \end{array}\tag{5}
$$

here $\mathcal { V } _ { p } ^ { ( i ) }$ and $\mathcal { N } _ { p } ^ { ( i ) } = \{ M _ { k } ^ { p } | k = 1 0 \}$ are the item set and hyperedge set, respectively. Specifically, the hyperedge set, depicted in Fig.2. $\bar { \boldsymbol { A } } _ { M _ { k } ^ { p } } ^ { ( i ) }$ is the purchase-motif-induced adjacency matrices.

Entity-oriented triple-channel hypergraphs. To tackle the sparsity and constraints inherent in historical user-item interaction data, we leverage the rich DBpedia KG (Auer et al., 2007) to build an entity-oriented hypergraph. More precisely, we identify individual items referenced in historical conversations as entities and their k-hop neighbors to construct each hyperedge. This method enables us to capture shared semantic nuances among the broader network of neighbors. Similar to itemoriented triple-channel hypergraphs, we build the entity-oriented hypergraphs from triple channel setting. Formally, the entity-oriented hypergraphs $\mathsf { G } _ { c } ^ { ( \bar { e } ) }$ from triple-channel c can be given as:

$$
\Game _ { c } ^ { ( e ) } = ( \gamma _ { c } ^ { ( e ) } , \mathcal { N } _ { c } ^ { ( e ) } , A _ { M _ { k } ^ { c } } ^ { ( e ) } ) .\tag{6}
$$

Here $c \in \{ g , j , p \}$ represents triple channels $( i . e .$ group, joint, and purchase channel). $\mathcal { V } _ { c } ^ { ( e ) }$ denotes the entities from triple-channel setting. These entities are k-hop neighbors extracted from the historical conversations. $\mathcal { N } _ { c } ^ { ( e ) }$ means the hyperedge set induced from different motifs. Each hyperedge is an instance of the Motif. $A _ { M _ { k } ^ { c } } ^ { ( e ) }$ represents the groupchannel, joint-channel, and purchase-channel adjacency matrices, they are defined as $\operatorname { E q . } ( 3 ) , \operatorname { E q . } ( 4 )$ and Eq.(5), respectively.

Word-oriented triple-channel hypergraphs. The significance of keywords exchanged during conversations is paramount in grasping users’ requirements. By scrutinizing notable words, we can pinpoint specific inclinations, a critical aspect in modeling an array of user tastes. To realize this objective, we construct a lexeme-centric hypergraph utilizing the lexicon-focused ConceptNet (Speer et al., 2017) KG to unveil semantic associations such as synonymy, antonyms, and co-occurrence. Based on these analysis, the word-oriented hypergraphs from group-, joint-, and purchase-channel can be expressed as:

$$
\mathsf { G } _ { c } ^ { ( w ) } = ( \mathcal { V } _ { c } ^ { ( w ) } , \mathcal { N } _ { c } ^ { ( w ) } , A _ { M _ { k } ^ { c } } ^ { ( w ) } ) ,\tag{7}
$$

where $\mathcal { V } _ { c } ^ { ( w ) }$ is the words from k-hop neighbors. $\mathcal { N } _ { c } ^ { ( w ) }$ denotes the hyperedge set from different motifs, including group, joint, purchase motifs. ${ \cal A } _ { M _ { k } ^ { c } } ^ { ( w ) }$ are the word-oriented adjacency matrices induced from triple channels, illustrated in Eq.(3)  Eq.(5).

3.1.2 Multi-Interest Self-Supervised Learning After constructing a series of hypergraphs from triple-channel setting, we will construct multi-level user interests via the hypergraph convolution network (Yu et al., 2021), which can be written as:

$$
\pmb { P } _ { c } ^ { ( l + 1 ) } = \pmb { D } _ { c } ^ { - 1 } \pmb { K } _ { c } \pmb { L } _ { c } ^ { - 1 } \pmb { K } _ { c } ^ { T } \pmb { P } _ { c } ^ { ( l ) } = \pmb { \widehat { D } } _ { c } ^ { - 1 } \pmb { A } _ { c } ^ { ( i ) } \pmb { P } _ { c } ^ { ( l ) } ,\tag{8}
$$

where ${ P } _ { c } ^ { ( l ) }$ and $P _ { c } ^ { ( l + 1 ) }$ represent the output of the l-th and $( l + 1 )$ -th layers, respectively. Specifically, the base user embedding is $P _ { c } ^ { ( 0 ) } = f _ { \mathrm { g a t e } } ^ { c } ( P ^ { ( 0 ) } )$ and $f _ { \mathrm { g a t e } } ^ { c } ( \cdot )$ is self-gating units (SGUs) to control the information flow to different channel from the base user embedding $P ^ { ( 0 ) } . D _ { c }$ is the degree matrix of $A _ { c } ,$ , which is the summation of the motifs without considering self-connections (Yu et al., 2021). In terms of the group motifs, it can be defined as $\begin{array} { r } { \mathbf { { \Sigma } } ^ { ( i ) } = \sum _ { k = 1 } ^ { 7 } \mathbf { { \Sigma } } \mathbf { { \Sigma } } ^ { ( i ) } } \end{array}$ , in terms of joint motifs, $\begin{array} { r } { \pmb { A } _ { c } ^ { ( i ) } = \pmb { A } _ { M _ { 8 } } ^ { ( i ) } + \pmb { A } _ { M _ { 9 } } ^ { ( i ) } \ d , } \end{array}$ , and from the point of the purchase motifs, $\pmb { A } _ { c } ^ { ( i ) } = \pmb { A } _ { M _ { 1 0 } } ^ { ( i ) } - ( \pmb { A } _ { M _ { 8 } } ^ { ( i ) } + \pmb { A } _ { M _ { 9 } } ^ { ( i ) } )$ Based on these analysis, the item-level interests from triple channel setting (i.e., $X _ { g } ^ { ( i ) } , X _ { i } ^ { ( i ) } , X _ { p } ^ { ( i ) } )$ the entity-level interests from triple channel setting $( i . e . , \mathbf { \boldsymbol { X } } _ { g } ^ { ( e ) } , \mathbf { \boldsymbol { X } } _ { j } ^ { ( e ) } , \mathbf { \boldsymbol { X } } _ { p } ^ { ( e ) } )$ , the word-level interests from triple channel setting $( i . e . , X _ { g } ^ { ( w ) } , X _ { j } ^ { ( w ) }$ $X _ { p } ^ { ( w ) } )$ can be defined as:

$$
\begin{array} { r l } & { \pmb { X } _ { g } ^ { ( h ) } = \pmb { \widehat { D } } _ { g } ^ { - 1 } ( \overset { 7 } { \underset { k = 1 } { \sum } } \pmb { A } _ { M _ { k } ^ { g } } ^ { ( h ) } ) \pmb { P } _ { g } ^ { ( L ) } ; } \\ & { \pmb { X } _ { j } ^ { ( h ) } = \pmb { \widehat { D } } _ { j } ^ { - 1 } ( \pmb { A } _ { M _ { 8 } ^ { j } } ^ { ( h ) } + \pmb { A } _ { M _ { 9 } ^ { j } } ^ { ( h ) } ) \pmb { P } _ { j } ^ { ( L ) } ; } \\ & { \pmb { X } _ { p } ^ { ( h ) } = \pmb { \widehat { D } } _ { p } ^ { - 1 } ( \pmb { A } _ { M _ { 1 0 } ^ { p } } ^ { ( h ) } - ( \pmb { A } _ { M _ { 8 } ^ { j } } ^ { ( h ) } + \pmb { A } _ { M _ { 9 } ^ { j } } ^ { ( h ) } ) \pmb { P } _ { p } ^ { ( L ) } . } \end{array}\tag{9}
$$

Here $h \in \{ i , e , w \}$ , and L is the last hypergraph convolution layer. Then, we adopt the attention network $\mathsf { A t t a } ( \cdot )$ and graph convolution GConv( ) to learn final multi-interest $X _ { m }$ as:

$$
\begin{array} { r l } & { X _ { i } = \mathsf { G C o n v } ( \mathsf { A t t a } ( X _ { g } ^ { ( i ) } ; X _ { j } ^ { ( i ) } ; X _ { p } ^ { ( i ) } ) ) , } \\ & { X _ { e } = \mathsf { G C o n v } ( \mathsf { A t t a } ( X _ { g } ^ { ( e ) } ; X _ { j } ^ { ( e ) } ; X _ { p } ^ { ( e ) } ) ) , } \\ & { X _ { w } = \mathsf { G C o n v } ( \mathsf { A t t a } X _ { g } ^ { ( w ) } ; X _ { j } ^ { ( w ) } ; X _ { p } ^ { ( w ) } ) ) , } \\ & { X _ { m } = \mathsf { A t t a } ( X _ { i } ; X _ { e } ; X _ { w } ) . } \end{array}\tag{10}
$$

Here ; is the concatenation operation. Finally, we use InfoNCE (Yu et al., 2021) as our learning objective to conduct the self-supervised learning as:

$$
\begin{array} { r l } & { \mathcal { L } _ { s } = - \displaystyle \sum _ { h } \bigg \{ \sum _ { u \in U } \log \sigma ( f ( X _ { m } ) , z _ { u } ^ { h } ) - f ( X _ { m } , \hat { z } _ { u } ^ { h } ) ) } \\ & { \quad \quad \quad + \displaystyle \sum _ { u \in U } \log \sigma ( f ( z _ { u } ^ { h } , \mathbf { k } ^ { h } ) - f ( \hat { z } _ { u } ^ { h } , \mathbf { k } ^ { h } ) \Big \} . } \end{array}\tag{11}
$$

Here $z _ { u } ^ { h } = f _ { \mathrm { g a t e } } ^ { h } ( f _ { s } ( X _ { h } ; { \mathbf p } _ { u } ^ { h } ) , f _ { s } ( \cdot )$ is the sum operation, $\hat { z } _ { u } ^ { h }$ is the negative example by shuffling both rows and columns of $z _ { u } ^ { h }$ , and h is defined as Eq.(9). $f ( \cdot ) \in \mathbb { R } ^ { d \times d }$ serves as the discriminator, evaluating the alignment between two input vectors. Specifically, $\mathbf { k } ^ { h } = f _ { \mathrm { o u t } } ( Z _ { h } )$ , where $Z _ { h }$ and $X _ { m }$ are ground truths for each other, and $f _ { \mathrm { o u t } } ( \cdot )$ aims to perform permutation invariant (Yu et al., 2021).

## 3.2 Interest-Boosted CRS

To mitigate Matthew effect in the CRS, we employ multi-interest $X _ { m }$ to enhance both recommendation and conversation tasks.

## 3.2.1 Recommendation Module

Recommendation module is to precisely forecast items for users via dynamic natural conversations. To improve recommendation diversity, we use $X _ { m }$ to select target items as $\mathsf { P } _ { \mathrm { r e c } } = X _ { m } \times V _ { \mathrm { c a n d } } $ where $V _ { \mathrm { c a n d } }$ is the embeddings of all candidate items. Finally, we adopt cross-entropy loss (Shang et al., 2023) to learn the recommendation task:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { r e c } } = - \displaystyle \sum _ { b = 1 } ^ { B } \sum _ { a = 1 } ^ { | \mathcal { Z } | } \Big \{ - \left( 1 - \mathsf { Y } _ { a b } \right) \cdot \log ( 1 - \mathsf { P } _ { \mathrm { r e c } } ^ { ( b ) } ( } & { } \\ { + \mathsf { Y } _ { a b } \cdot \log ( \mathsf { P } _ { \mathrm { r e c } } ^ { ( b ) } ( a ) ) \Big \} , } & { } \end{array}\tag{a)}
$$

(12)

where $\mathsf { Y } _ { a b } \in \{ 0 , 1 \}$ , B, and $| \mathcal { T } |$ are the target label, mini-batch size, the size of item set, respectively.

## 3.2.2 Conversation Module

Conversation module centers on crafting appropriate dialogue responses. Next, we use multi-interest $X _ { m }$ to fed into Transformer ${ \mathsf { M H A } } ( \cdot )$ to produce informative responses. Suppose $\mathbf { Y } ^ { n - 1 }$ is the output of the last time unit, then the current one $\mathbf { Y } ^ { n }$ is:

$$
\begin{array} { r l } & { { \bf A } _ { 0 } ^ { n } = { \sf M H } { \sf H } ( { \bf Y } ^ { n - 1 } , { \bf Y } ^ { n - 1 } , { \bf Y } ^ { n - 1 } ) , } \\ & { { \bf A } _ { 1 } ^ { n } = { \sf M H } { \sf H } ( { \bf A } _ { 0 } ^ { n } , { \bf X } _ { m } , { \bf X } _ { m } ) , } \\ & { { \bf A } _ { 2 } ^ { n } = { \sf M H } ( { \bf A } _ { 1 } ^ { n } , { \bf X } _ { \mathrm { c u r } } , { \bf X } _ { \mathrm { c u r } } ) , } \\ & { { \bf A } _ { 3 } ^ { n } = { \sf M H } { \sf A } ( { \bf A } _ { 1 } ^ { n } , { \bf X } _ { \mathrm { h i s } } , { \bf X } _ { \mathrm { h i s } } ) , } \\ & { { \bf A } _ { 4 } ^ { n } = \beta \cdot { \bf A } _ { 2 } ^ { n } + ( 1 - \beta ) \cdot { \bf A } _ { 3 } ^ { n } , } \\ & { { \bf Y } ^ { n } = { \sf F F N } ( { \bf A } _ { 4 } ^ { n } ) , } \end{array}\tag{13}
$$

where $X _ { \mathrm { c u r } }$ and $X _ { \mathrm { h i s } }$ are the current and historical conversations, respectively. $\beta$ is hyper-parameters to control the information flow. Then, we use crossentropy loss to learn the conversation task:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n v } } = - \displaystyle \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T } \log ( \mathsf { P } _ { \mathrm { c o n v } } ( s _ { t } | \{ s _ { t - 1 } \} ) ) , } \\ { \mathsf { P } _ { \mathrm { c o n v } } ( \cdot ) = p _ { 1 } ( s _ { t } | Y _ { i } ) + p _ { 2 } ( s _ { t } | \mathsf { P } _ { \mathrm { r e c } } ) + p _ { 3 } ( s _ { t } | \mathsf { P } _ { \mathrm { r e c } } ) , } \end{array}\tag{14}
$$

where $\mathsf { P } _ { \mathrm { c o n v } } ( \cdot )$ is the probability of the next token when given a sequence $\left\{ s _ { t - 1 } \right\} = s _ { 1 } , s _ { 2 } , \cdot \cdot \cdot , s _ { t - 1 } ,$ where $s _ { t }$ signifies the t-th utterance. $p _ { 1 } ( \cdot ) , p _ { 2 } ( \cdot )$ and $p _ { 3 } ( \cdot )$ denote the vocabulary probability, vocabulary bias, and copy probability, respectively. $T$ is the truncated length of utterances.

## 3.3 Challenges Discussion

Throughout the developmental journey of hypergraphs, we surmounted several significant challenges, elaborated upon below:

1) Hypergraph Construction Challenge: During the project’s initial stages, the real-time construction of hypergraphs presented a bottleneck, resulting in delays. Through the strategic repositioning of this operation to the data preprocessing phase, we adeptly extracted essential subgraphs, leading to a noteworthy reduction in training time. This adjustment enhanced efficiency, streamlined processes, and improved performance.

2) Graph Storage Challenge: The transition to sparse graph storage mechanisms is pivotal in enhancing efficiency, streamlining computation time, and optimizing memory utilization. Embracing this shift not only boosts the system’s performance but also establishes a robust foundation for scalable and resource-efficient operations.

3) Model Training Challenge: With the emergence of a series of hypergraphs, optimizing the efficiency of model training becomes paramount. Consequently, we redefined our strategy by dispersing hypergraphs across multiple computing cards, enabling parallel computation and achieving a significant boost in the model’s runtime speed.

## 4 Experiments and Analyses

To fully evaluate the proposed HiCore, we conduct experiments to answer the following questions:

• RQ1: How does HiCore perform compared with all baselines in the recommendation task?

• RQ2: How does HiCore perform compared with all baselines in the conversation task?

• RQ3: How does HiCore mitigate Matthew effect in the CRS?

• RQ4: How do parameters affect our HiCore?

• RQ5: How do different hypergraphs contribute to the performance?

## 4.1 Experimental Protocol

Datasets. We assess the effectiveness of our proposed HiCore through comprehensive evaluations on four CRS-based benchmarks: REDIAL (Li et al., 2018b), TG-REDIAL (Zhou et al., 2020b), OpenDialKG (Moon et al., 2019), and DuRecDial (Liu et al., 2021b). The REDIAL dataset comprises 11,348 dialogues involving 956 users and 6,924 items, while the TG-REDIAL dataset encompasses 10,000 dialogues with 1,482 users and

<table><tr><td rowspan="2">Model</td><td colspan="7">REDIAL</td><td colspan="6">TG-REDIAL</td></tr><tr><td>R@10</td><td>R@50</td><td>M@10</td><td>M@50</td><td>N@10</td><td>N@50</td><td>R@10</td><td>R@50</td><td></td><td>M@10</td><td>M@50</td><td>N@10</td><td>N@50</td></tr><tr><td>TextCNN</td><td>0.0644</td><td>0.1821</td><td>0.0235</td><td>0.0285</td><td>0.0328</td><td>0.0580</td><td></td><td>0.0097</td><td>0.0208</td><td>0.0040</td><td>0.0045</td><td>0.0053</td><td>0.0077</td></tr><tr><td>SASRec</td><td>0.1117</td><td>0.2329</td><td>0.0540</td><td>0.0593</td><td>0.0674</td><td>0.0936</td><td>0.0043</td><td>0.0178</td><td>0.0011</td><td>0.0017</td><td></td><td>0.0019</td><td>0.0047</td></tr><tr><td>BERT4Rec</td><td>0.1285</td><td>0.3032</td><td>0.0475</td><td>0.0555</td><td>0.0663</td><td>0.1045</td><td></td><td>0.0043</td><td>0.0226</td><td>0.0013</td><td>0.0020</td><td>0.0020</td><td>0.0058</td></tr><tr><td>KGSF</td><td>0.1785</td><td>0.3690</td><td>0.0705</td><td>0.0796</td><td>0.0956</td><td>0.1379</td><td>0.0215</td><td>0.0643</td><td>0.0069</td><td></td><td>0.0087</td><td>0.0103</td><td>0.0194</td></tr><tr><td>TG-ReDial</td><td>0.1679</td><td>0.3327</td><td>0.0694</td><td>0.0771</td><td>0.0924</td><td>0.1286</td><td></td><td>0.0110</td><td>0.0174</td><td>0.0048</td><td>0.0050</td><td>0.0062</td><td>0.0076</td></tr><tr><td>ReDial</td><td>0.1705</td><td>0.3077</td><td>0.0677</td><td>0.0738</td><td>0.0925</td><td>0.1222</td><td></td><td>0.0038</td><td>0.0165</td><td>0.0012</td><td>0.0017</td><td>0.0018</td><td>0.0045</td></tr><tr><td>KBRD</td><td>0.1796</td><td>0.3421</td><td>0.0722</td><td>0.0800</td><td>0.0972</td><td>0.1333</td><td></td><td>0.0201</td><td>0.0501</td><td>0.0077</td><td>0.0090</td><td>0.0106</td><td>0.0171</td></tr><tr><td>BART</td><td>0.1693</td><td>0.3783</td><td>0.0646</td><td>0.0744</td><td>0.0888</td><td>0.1350</td><td></td><td>0.0047</td><td>0.0187</td><td>0.0012</td><td>0.0017</td><td>0.0020</td><td>0.0048</td></tr><tr><td>BERT</td><td>0.1608</td><td>0.3525</td><td>0.0597</td><td>0.0688</td><td>0.0831</td><td>0.1255</td><td>0.0040</td><td></td><td>0.0194</td><td>0.0011</td><td>0.0017</td><td>0.0018</td><td>0.0050</td></tr><tr><td>XLNet</td><td>0.1569</td><td>0.3590</td><td>0.0583</td><td>0.0677</td><td>0.0811</td><td>0.1255</td><td></td><td>0.0040</td><td>0.0187</td><td>0.0011</td><td>0.0017</td><td>0.0017</td><td>0.0048</td></tr><tr><td>KGConvRec</td><td>0.1819</td><td>0.3587</td><td>0.0711</td><td>0.0794</td><td>0.0969</td><td>0.1358</td><td></td><td>0.0220</td><td>0.0524</td><td>0.0088</td><td>0.0102</td><td>0.0119</td><td>0.0185</td></tr><tr><td>MHIM</td><td>0.1966</td><td>0.3832</td><td>0.0742</td><td>0.0830</td><td>0.1027</td><td>0.1440</td><td>0.0300</td><td>0.0783</td><td></td><td>0.0108</td><td>0.0129</td><td>0.0152</td><td>0.0256</td></tr><tr><td>HiCore*</td><td>0.2192</td><td>0.4163</td><td>0.0775</td><td>0.0874</td><td>0.1107</td><td>0.1558</td><td></td><td>0.0270</td><td>0.0769</td><td>0.0880</td><td>0.1074</td><td>0.0152</td><td>0.0225</td></tr></table>

Table 1: Recommendation results on REDIAL and TG-REDIAL datasets. \* indicates statistically significant improvement (p < 0.05) over all baselines.

33,834 items. To provide a holistic evaluation of our proposed methodology, we integrate two cross-domain datasets, OpenDialKG and DuRec-Dial, which cover a wide array of domains including movies, music, books, sports, restaurants, news, and culinary experiences.

Baselines. We compared our HiCore with the following state-of-the-art methods TextCNN (Kim, 2014), SASRec (Kang and McAuley, 2018), BERT4Rec (Sun et al., 2019), KBRD (Chen et al., 2019), Trans. (Vaswani et al., 2017b), ReDial (Li et al., 2018a), KGSF (Zhou et al., 2020a), KG-ConvRec (Sarkar et al., 2020), XLNet (Yang et al., 2019), BART (Lewis et al., 2020), BERT (Devlin et al., 2019), DialoGPT (Zhang et al., 2020), Uni-CRS (Deng et al., 2021b), GPT-3 (Brown et al., 2020), C2-CRS (Zhou et al., 2022), LOT-CRS (Zhao et al., 2023), MHIM (Shang et al., 2023), and HyCoRec (Zheng et al., 2024).

## 4.2 Recommendation Performance (RQ1)

In accordance with (Shang et al., 2023), we utilize Recall@K (R@K), MRR@K (M@K), and NDCG@K (N@K) (K=1, 10, 50) to assess the recommendation task. Analyzing the results presented in Table 1 and Table 2, it is evident that our proposed method, HiCore, consistently outperforms all the comparison baselines.

There exist multiple crucial facets contributing to the advancement of our proposed HiCore method: (a) Diversification of hypergraphs: we introduced a diverse set of hypergraphs, including item-oriented, entity-oriented, and word-oriented hypergraphs. This expansion aims to go beyond the traditional pairwise interactions, broadening the scope of user interest modeling by incorporating interactions among multiple nodes. (b) Exploration of hypergraph configurations: moving beyond the conventional triple-channel model, we delved into various hypergraph configurations like group-channel, joint-channel, and purchasechannel. These configurations were designed to cater not only to social connections but also individual preferences, enhancing the system’s adaptability. (c) Integration of multi-level user interests: transitioning from the triple-channel structure, we integrated these hypergraphs to capture multi-level user interests. This strategic shift helps alleviate the Matthew effect in the CRS involving the dynamic user-system feedback loop. This comprehensive approach highlights the innovation and adaptability of HiCore in addressing the intricacies of user interest modeling and enhancing recommendation system performance.

<table><tr><td rowspan="2">Model</td><td colspan="2">OpenDialKG</td><td colspan="2">DuRecDial</td></tr><tr><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td></tr><tr><td>KBRD</td><td>0.1448</td><td>0.3162</td><td>0.0618</td><td>0.3971</td></tr><tr><td>KGSF</td><td>0.0626</td><td>0.1757</td><td>0.1395</td><td>0.4367</td></tr><tr><td>ReDial</td><td>0.0008</td><td>0.0134</td><td>0.0005</td><td>0.0336</td></tr><tr><td>TGReDial</td><td>0.2149</td><td>0.4035</td><td>0.0956</td><td>0.4882</td></tr><tr><td>HyCoRec</td><td>0.2742</td><td>0.4490</td><td>0.1279</td><td>0.4750</td></tr><tr><td>HiCore*</td><td>0.2628</td><td>0.4526</td><td>0.1735</td><td>0.5471</td></tr><tr><td rowspan="2">Model</td><td>OpenDialKG</td><td></td><td></td><td>DuRecDial</td></tr><tr><td>Dist-2</td><td>Dist-3</td><td>Dist-2</td><td>Dist-3</td></tr><tr><td>KBRD</td><td>0.3192</td><td>1.7660</td><td>0.5180</td><td>1.5500</td></tr><tr><td>KGSF</td><td>0.1687</td><td>0.5387</td><td>0.1389</td><td>0.3862</td></tr><tr><td>ReDial</td><td>0.1579</td><td>0.5808</td><td>0.1095</td><td>0.3981</td></tr><tr><td>TGReDial</td><td>0.4836</td><td>2.1430</td><td>0.5453</td><td>2.0030</td></tr><tr><td>HyCoRec</td><td>2.8190</td><td>4.7710</td><td>1.0820</td><td>2.4440</td></tr><tr><td>HiCore*</td><td>2.8430</td><td>4.8120</td><td>1.0940</td><td>2.4280</td></tr></table>

Table 2: Results on both recommendation and conversation tasks on OpenDialKG and DuRecDial datasets involving various domains. \* indicates statistically significant improvement (p < 0.05) over all baselines.

<table><tr><td rowspan="2">Model</td><td colspan="2">REDIAL</td><td colspan="2">TG-REDIAL</td></tr><tr><td>Dist-2 Dist-3</td><td>Dist-4</td><td>Dist-2</td><td>Dist-3 Dist-4</td></tr><tr><td>ReDial</td><td>0.0214 0.0659 0.1333</td><td></td><td></td><td>0.2178 0.5136 0.7960</td></tr><tr><td>Trans.</td><td>0.05380.1574 0.2696</td><td></td><td>0.2362 0.70631.1800</td><td></td></tr><tr><td>KGSF</td><td>0.0572 0.24830.4349</td><td></td><td></td><td>0.3891 0.88681.3337</td></tr><tr><td>KBRD</td><td>0.07650.33440.6100</td><td></td><td></td><td>0.8013 1.7840 2.5977</td></tr><tr><td>DialoGPT</td><td>0.3542 0.62090.9482</td><td></td><td></td><td>1.1881 2.42693.9824</td></tr><tr><td>GPT-3</td><td>0.3604 0.6399 0.9511</td><td></td><td></td><td>1.2255 2.57134.0713</td></tr><tr><td>UniCRS</td><td>0.24640.42730.5290</td><td></td><td></td><td>0.6252 2.2352 2.5194</td></tr><tr><td>C2-CRS</td><td>0.2623 0.3891 0.6202</td><td></td><td></td><td>0.52351.9961 2.9236</td></tr><tr><td>LOT-CRS</td><td>0.3312 0.61550.9248</td><td></td><td></td><td>0.9287 2.48803.4972</td></tr><tr><td>MHIM</td><td>0.32780.62040.9629</td><td></td><td>1.1100 2.35203.8200</td><td></td></tr><tr><td>HiCore*</td><td></td><td>0.5871 1.1170 1.7500</td><td>2.8610 5.7440 8.4160</td><td></td></tr></table>

Table 3: Conversation results on REDIAL and TG-REDIAL datasets. \* indicates statistically significant improvement (p < 0.05) over all baselines.

## 4.3 Conversational Performance (RQ2)

For the conversation task, we use Distinct n-gram (Dist-n) (Shang et al., 2023) (n=2,3,4) to estimate the conversation task. Table 2 and Table 3 indicate a significant performance superiority of our HiCore. For example, HiCore gains 123.83%, 138.27%, 77.26%, 65.75%, 62.90%, and 79.10% improvements on Dist-2 against the strong baselines including, C2-CRS, UniCRS, LOT-CRS, DialoGPT, GPT-3, and MHIM on the REDIAL dataset, respectively. It also gains 446.51%, 357.61%, 208.07%, 140.80%, 133.46%, and 157.75% improvements on Dist-2 against the strong baselines including, C2-CRS, UniCRS, LOT-CRS, DialoGPT, GPT-3, and MHIM on the REDIAL dataset, respectively.

The improvement in HiCore can be attributed to the following reasons: (a) Our HiCore focuses on constructing a diverse set of hypergraphs, encompassing item-oriented, entity-oriented, and wordoriented triple-channel hypergraphs. These structures effectively capture intricate local patterns through motif analysis, enabling the exploration of high-order user behaviors. This proves invaluable in generating informative and high-quality response utterances. (b) HiCore is dedicated to mitigating the Matthew effect that may occur as users engage with the system over time. By learning multi-level user interests from the hypergraphs, the system can adapt to users’ evolving preferences. This strategic approach enables the CRS to provide a varied array of responses that align with the diverse interests of the users.

![](images/1368bdf8f7d776abd394166f8704906bb542fe38341b1383856f3e3c866efad4.jpg)  
Figure 3: Coverage results of C@k metric.

<table><tr><td rowspan="2">Model</td><td colspan="2">OpenDialKG</td><td colspan="2">DuRecDial</td></tr><tr><td></td><td>A@5 A@15 A@30</td><td>A@5 A@15 A@30</td><td></td></tr><tr><td>KBRD KGSF</td><td></td><td>0.0025 0.0025 0.0088 0.0051 0.0108 0.0182</td><td>0.0318 0.0562 0.0938 0.0276 0.0534 0.0952</td><td></td></tr><tr><td>ReDial TGReDial</td><td></td><td>1.0000 0.9375 0.8333 0.0022 0.0043 0.0070</td><td>1.0000 0.8824 0.9677 0.0137 0.0399 0.0796</td><td></td></tr><tr><td>MHIM HiCore</td><td></td><td>0.0022 0.00440.0075 0.0017 0.0043 0.0065</td><td></td><td>0.0228 0.0434 0.0789 0.0226 0.0423 0.0751</td></tr><tr><td rowspan="2">Model</td><td></td><td>OpenDialKG</td><td>DuRecDial L@5</td><td>L@15 L@30</td></tr><tr><td>L@5</td><td>L@15 L@30 0.2921 0.2782 0.2782</td><td>0.37580.41490.3406 0.23980.2482 0.3343</td><td>0.33140.42430.3302 1.0000 0.8235 0.9677</td></tr></table>

Table 4: Results of Average Popularity (A@K) and Long Tail Ratio (L@K).

## 4.4 Study on Matthew Effect (RQ3)

Given our goal of mitigating the Matthew effect that may arise as users interact with the system over time, we engage in a series of experiments comparing the proposed method with the most robust baselines. This investigation seeks to determine the efficacy of HiCore in effectively alleviating the Matthew effect. Considering the key strategy to mitigate Matthew effect is to improve the recommendation diversification, and thus we use the diversify-based evaluation metrics Coverage@k (C@k), Average Popularity (A@K) of Recommended Items and Long Tail Recommendation Ratio (L@K) to comprehensively evaluate the efficacy of our proposed method in mitigating the Matthew Effect.

Fig.3 illustrates the experimental outcomes, showcasing the consistent superiority of HiCore in achieving the highest levels of Coverage across all datasets in comparison to the most robust baselines. The heightened coverage metric highlights its exceptional ability to encompass a broad spectrum of the recommendation space by incorporating items from diverse categories. Additionally, as outlined in Table 4, our proposed method demonstrates the lowest values for Average Popularity and Long Tail Ratio. This evidence suggests that our method effectively mitigates the adverse effects of item popularity on recommendation outcomes and successfully addresses the long tail distribution of items. These results validate the effectiveness of our proposed approach in combating the Matthew effect in the CRS as users interact with the system over time, attributed to its capability to learn multilevel user interests through a series of hypergraphs from triple-channel setting, including group, joint, and purchase channels.

![](images/c098d43b722bac834d0b5a3b0bd020e58a89359caf51f1158ea29d454ea782da.jpg)  
Figure 4: Impact of different hyperparameteres.

## 4.5 Hyperparameters Analysis (RQ4)

Hyperparameters are parameters in a machine learning algorithm that need to be manually set and tuned to optimize model performance, distinct from the parameters that the model learns during training. Next, we will delve into the research on how various hyperparameters influence the performance of recommendations, including the embedding dimension $d ,$ comparative learning weight $\beta ,$ hypergraph convolution layers N, and the hyperedge threshold P. From Fig.4, we can obtain: (1) Elevating the feature dimensionality enhances outcomes, as higher dimensions can encapsulate more intricate features effectively; (2) Having too few hyperedges may hinder the capture of intricate local patterns, whereas an excess of hyperedges could impede the model’s convergence; (3) A lower beta value signifies a reduced weight for the comparison term, which show that the recommendation term exerts a more significant influence on the results; (4) A two-layer hyperconv network is sufficient to encode high-level features for enhancing recommendation performance.

<table><tr><td rowspan="2">Model</td><td colspan="2">REDIAL</td><td colspan="2">TG-REDIAL</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td>HiCore</td><td>0.2192</td><td>0.4163</td><td>0.0270</td><td>0.0769</td></tr><tr><td>w/o  $\overline { { \mathsf { G } _ { a } ^ { ( i ) } } }$ </td><td>0.2075</td><td>0.4160</td><td>0.0234</td><td>0.0742</td></tr><tr><td>w/o  $\mathsf { G } _ { i } ^ { ( i ) }$ </td><td>0.2012</td><td>0.4026</td><td>0.0217</td><td>0.0706</td></tr><tr><td>w/o  $\mathsf { G } _ { p } ^ { ( i ) }$ </td><td>0.1939</td><td>0.4096</td><td>0.0220</td><td>0.0739</td></tr><tr><td>w/o  $\mathsf { G } _ { g } ^ { ( e ) }$ </td><td>0.2067</td><td>0.4044</td><td>0.0247</td><td>0.0713</td></tr><tr><td>w/o  $\mathsf { G } _ { i } ^ { \left( e \right) }$ </td><td>0.2142</td><td>0.4122</td><td>0.0253</td><td>0.0756</td></tr><tr><td>w/o  $\mathsf { G } _ { p } ^ { ( e ) }$ </td><td>0.1971</td><td>0.4110</td><td>0.0243</td><td>0.0693</td></tr><tr><td>w/o  $\mathsf { G } _ { a } ^ { \mathsf { \tilde { ( } } w \mathsf { ) } }$ </td><td>0.2067</td><td>0.4142</td><td>0.0264</td><td>0.0761</td></tr><tr><td>w/o  $\mathsf { G } _ { i } ^ { \vec { ( \boldsymbol { w } ) } }$ </td><td>0.2151</td><td>0.4145</td><td>0.0223</td><td>0.0733</td></tr><tr><td>w/o  $\mathsf { G } _ { p } ^ { ( w ) }$ </td><td>0.2067</td><td>0.3974</td><td>0.0263</td><td>0.0759</td></tr></table>

Table 5: Ablation studies on the recommendation task.

## 4.6 Ablation Studies (RQ5)

To assess the efficacy of each component within the proposed method, we perform ablation experiments using various iterations of Hicore, including: 1) w/o $\mathsf { G } _ { g } ^ { i } .$ , w/o $\mathsf { G } _ { j } ^ { i }$ , w/o $\mathsf { G } _ { p } ^ { i } \colon \quad \mathsf { G } _ { p } ^ { i } \colon$ removing item-oriented group-channel, joint-channel, purchase-channel hypergraph, respectively; 2) w/o $\mathsf { G } _ { g } ^ { e } ,$ w/o $\mathsf { G } _ { j } ^ { e }$ w/o $\mathsf { G } _ { p } ^ { i } \colon \qquad $ : removing entity-oriented group-channel, joint-channel, purchase-channel hypergraph, respectively; 3) w/o $\mathsf { G } _ { g } ^ { w }$ , w/o $\mathsf { G } _ { j } ^ { w }$ , w/o $\mathsf { G } _ { p } ^ { w }$ : removing word-oriented group-channel, joint-channel, purchase-channel hypergraph, respectively.

Table 5 outlines the experimental findings, indicating that the removal of any hypergraph type results in a performance decrease. This highlights the effectiveness of each hypergraph type and underscores the superiority of HiCore in learning multi-level user interests through a collection of hypergraphs to mitigate Matthew effect in the CRS.

## 5 Conclusion

The Matthew effect poses a significant challenge in the CRS due to the dynamic user-system feedback loop, which tends to escalate over time as users engage with the system. In response to these challenges, we proposed a novel framework, HiCore, aimed at mitigating the Matthew effect by capturing multi-level user interests through a variety of hypergraphs, including item-oriented, entity-oriented, and word-oriented triple-channel hypergraphs. Extensive experiments validate that HiCore outperforms all baselines, demonstrating the effectiveness of HiCore in addressing the Matthew effect as users chat with the system over time in the CRS.

## 6 Limitations

While our HiCore has achieved a remarkable stateof-the-art performance, it does come with certain limitations. Firstly, triple-channel hypergraphs may present challenges due to their computational complexity, interpretational intricacies, and potential issues with sparse data. Secondly, scaling these hypergraphs to larger datasets could introduce scalability hurdles, with a risk of overfitting when the model becomes excessively fine-tuned to the training data. Furthermore, ensuring generalizability and handling resource-intensive computations are crucial factors to consider when leveraging multichannel hypergraphs.

## 7 Ethics Statement

The data used in this paper are sourced from openaccess repositories, and do not pose any privacy concerns. We are confident that our research adheres to the ethical standards set forth by EMNLP.

## 8 Acknowledgements

This research / project is supported by the National Research Foundation, Singapore and Infocomm Media Development Authority under its Trust Tech Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and Infocomm Media Development Authority.

## References

Ashton Anderson, Lucas Maystre, Ian Anderson, Rishabh Mehrotra, and Mounia Lalmas. 2020. Algorithmic effects on the diversity of consumption on spotify. In The Web Conference, pages 2155–2165.

Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary G. Ives. 2007. Dbpedia: A nucleus for a web of open data. In International Semantic Web Conference/Asian Semantic Web Conference, volume 4825, pages 722– 735.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

2020. Language models are few-shot learners. In Conference on Neural Information Processing Systems.

Qibin Chen, Junyang Lin, Yichang Zhang, Ming Ding, Yukuo Cen, Hongxia Yang, and Jie Tang. 2019. Towards knowledge-based recommender dialog system. arXiv preprint arXiv:1908.05391.

Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021a. Unified conversational recommendation policy learning via graph-based reinforcement learning. In Conference on Research and Development in Information Retrieval, pages 1431–1441.

Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021b. Unified conversational recommendation policy learning via graph-based reinforcement learning. In Conference on Research and Development in Information Retrieval, pages 1431–1441.

Yang Deng, Wenxuan Zhang, Weiwen Xu, Wenqiang Lei, Tat-Seng Chua, and Wai Lam. 2023. A unified multi-task learning framework for multi-goal conversational recommender systems. ACM Transactions on Information Systems, 41(3):1–25.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171–4186. Association for Computational Linguistics.

Elena V. Epure and Romain Hennequin. 2023. A human subject study of named entity recognition in conversational music recommendation queries. In European Chapter of the Association for Computational Linguistics, pages 1273–1288.

Yingqiang Ge, Shuya Zhao, Honglu Zhou, Changhua Pei, Fei Sun, Wenwu Ou, and Yongfeng Zhang. 2020. Understanding echo chambers in e-commerce recommender systems. In Conference on Research and Development in Information Retrieval, pages 2261– 2270. ACM.

Christian Hansen, Rishabh Mehrotra, Casper Hansen, Brian Brost, Lucas Maystre, and Mounia Lalmas. 2021. Shifting consumption towards diverse content on music streaming platforms. In Conference on Web Search and Data Mining, pages 238–246. ACM.

Eslam Hussein, Prerna Juneja, and Tanushree Mitra. 2020. Measuring misinformation in video search platforms: An audit study on youtube. ACM on Human-Computer Interaction, 4(CSCW):048:1– 048:27.

Wang-Cheng Kang and Julian J. McAuley. 2018. Selfattentive sequential recommendation. In IEEE International Conference on Data Mining, pages 197– 206.

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Empirical Methods in Natural Language Processing (Demonstrations), pages 1746–1751.

Wenqiang Lei, Xiangnan He, Yisong Miao, Qingyun Wu, Richang Hong, Min-Yen Kan, and Tat-Seng Chua. 2020a. Estimation-action-reflection: Towards deep interaction between conversational and recommender systems. In Web Search and Data Mining, pages 304–312.

Wenqiang Lei, Gangyi Zhang, Xiangnan He, Yisong Miao, Xiang Wang, Liang Chen, and Tat-Seng Chua. 2020b. Interactive path reasoning on graph for conversational recommendation. In International Conference on Knowledge Discovery and Data Mining, pages 2073–2083.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In the Association for Computational Linguistics, pages 7871–7880. Association for Computational Linguistics.

Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018a. Towards deep conversational recommendations. Advances in Neural Information Processing Systems, 31.

Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018b. Towards deep conversational recommendations. In Advances in Neural Information Processing Systems, pages 9748–9758.

Shuokai Li, Ruobing Xie, Yongchun Zhu, Xiang Ao, Fuzhen Zhuang, and Qing He. 2022. User-centric conversational recommendation with multi-aspect user modeling. In Conference on Research and Development in Information Retrieval, pages 223–233.

Yile Liang, Tieyun Qian, Qing Li, and Hongzhi Yin. 2021. Enhancing domain-level and user-level adaptivity in diversified recommendation. In Conference on Research and Development in Information Retrieval, pages 747–756. ACM.

Ping Liu, Karthik Shivaram, Aron Culotta, Matthew A. Shapiro, and Mustafa Bilgic. 2021a. The interaction between political typology and filter bubbles in news recommendation algorithms. In The Web Conference, pages 3791–3801.

Ying Chieh Liu and Min Qi Huang. 2021. Examining the matthew effect on youtube recommendation system. In Conference on Technologies and Applications ofArtificial Intelligence, pages 146–148.

Yuanxing Liu, Weinan Zhang, Baohua Dong, Yan Fan, Hang Wang, Fan Feng, Yifan Chen, Ziyu Zhuang, Hengbin Cui, Yongbin Li, and Wanxiang Che. 2023.

U-NEED: A fine-grained dataset for user needscentric e-commerce conversational recommendation. In Conference on Research and Development in Information Retrieval, pages 2723–2732. ACM.

Zeming Liu, Haifeng Wang, Zhengyu Niu, Hua Wu, and Wanxiang Che. 2021b. Durecdial 2.0: A bilingual parallel corpus for conversational recommendation. In Conference on Empirical Methods in Natural Language Processing EMNLP, pages 4335–4347. Association for Computational Linguistics.

R. Milo, S. Shen-Orr, S. Itzkovitz, N. Kashtan, D. Chklovskii, and U. Alon. 2002. Network motifs: Simple building blocks of complex networks. Science, 298(5594):824–827.

Kshitij Mishra, Priyanshu Priya, and Asif Ekbal. 2023. Help me heal: A reinforced polite and empathetic mental health and legal counseling dialogue system for crime victims. In Association for the Advancement ofArtificial Intelligence, pages 14408–14416.

Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. 2019. Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs. In Conference ofthe Association for Computational Linguistics ACL, pages 845–854. Association for Computational Linguistics.

Tien T. Nguyen, Pik-Mai Hui, F. Maxwell Harper, Loren G. Terveen, and Joseph A. Konstan. 2014. Exploring the filter bubble: the effect of using recommender systems on content diversity. In The Web Conference, pages 677–686. ACM.

Libo Qin, Zhouyang Li, Qiying Yu, Lehan Wang, and Wanxiang Che. 2023. Towards complex scenarios: Building end-to-end task-oriented dialogue system across multiple knowledge bases. In Association for the Advancement of Artificial Intelligence, pages 13483–13491.

Xuhui Ren, Hongzhi Yin, Tong Chen, Hao Wang, Zi Huang, and Kai Zheng. 2021. Learning to ask appropriate questions in conversational recommendation. In Conference on Research and Development in Information Retrieval, pages 808–817.

Rajdeep Sarkar, Koustava Goswami, Mihael Arcan, and John Philip McCrae. 2020. Suggest me a movie for tonight: Leveraging knowledge graphs for conversational recommendation. In Conference on Computational Linguistics, pages 4179–4189.

Chenzhan Shang, Yupeng Hou, Wayne Xin Zhao, Yaliang Li, and Jing Zhang. 2023. Multi-grained hypergraph interest modeling for conversational recommendation. AI Open, 4:154–164.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Association for the Advancement of Artificial Intelligence, pages 4444–4451.

Harald Steck. 2018. Calibrated recommendations. In Conference on Recommender Systems, pages 154– 162.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In International Conference on Information and Knowledge Management, pages 1441–1450. ACM.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017a. Attention is all you need. Advances in neural information processing systems, 30.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017b. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008.

Hao Wang, Zonghu Wang, and Weishi Zhang. 2019. Quantitative analysis of matthew effect and sparsity problem of recommender systems. CoRR.

Kerui Xu, Jingxuan Yang, Jun Xu, Sheng Gao, Jun Guo, and Ji-Rong Wen. 2021. Adapting user preference to online feedback in conversational recommendation. In Web Search and Data Mining, pages 364–372.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in Neural Information Processing Systems, pages 5754–5764.

Junliang Yu, Hongzhi Yin, Jundong Li, Qinyong Wang, Nguyen Quoc Viet Hung, and Xiangliang Zhang. 2021. Self-supervised multi-channel hypergraph convolutional network for social recommendation. In World Wide Web WWW, pages 413–424. ACM / IW3C2.

Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. 2021. Causal intervention for leveraging popularity bias in recommendation. In Conference on Research and Development in Information Retrieval, pages 11–20. ACM.

Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2020. DIALOGPT : Large-scale generative pre-training for conversational response generation. In the Association for Computational Linguistics, pages 270–278. Association for Computational Linguistics.

Zhipeng Zhao, Kun Zhou, Xiaolei Wang, Wayne Xin Zhao, Fan Pan, Zhao Cao, and Ji-Rong Wen. 2023. Alleviating the long-tail problem in conversational recommender systems. In ACM Conference on Recommender Systems, pages 374–385. ACM.

Yongsen Zheng, Ruilin Xu, Ziliang Chen, Guohua Wang, Mingjie Qian, Jinghui Qin, and Liang Lin. 2024. Hycorec: Hypergraph-enhanced multipreference learning for alleviating matthew effect in conversational recommendation. In the Associationfor Computational Linguistics ACL, pages 2526– 2537. Association for Computational Linguistics.

Yu Zheng, Chen Gao, Liang Chen, Depeng Jin, and Yong Li. 2021a. DGCN: diversified recommendation with graph convolutional networks. In The Web Conference, pages 401–412.

Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021b. Disentangling user interest and conformity for recommendation with causal embedding. In The Web Conference, pages 2980–2991.

Kun Zhou, Wayne Xin Zhao, Shuqing Bian, Yuanhang Zhou, Ji-Rong Wen, and Jingsong Yu. 2020a. Improving conversational recommender systems via knowledge graph based semantic fusion. In International Conference on Knowledge Discovery and Data Mining, pages 1006–1014.

Kun Zhou, Yuanhang Zhou, Wayne Xin Zhao, Xiaoke Wang, and Ji-Rong Wen. 2020b. Towards topicguided conversational recommender system. In International Conference on Computational Linguistics, pages 4128–4139.

Yuanhang Zhou, Kun Zhou, Wayne Xin Zhao, Cheng Wang, Peng Jiang, and He Hu. 2022. C<sup>2</sup>-crs: Coarseto-fine contrastive learning for conversational recommender system. In Web Search and Data Mining, pages 1488–1496. ACM.