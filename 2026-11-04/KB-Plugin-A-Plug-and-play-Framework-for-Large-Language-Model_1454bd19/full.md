# KB-Plugin: A Plug-and-play Framework for Large Language Models to Induce Programs over Low-resourced Knowledge Bases

Jiajie Zhang<sup>1</sup> , Shulin Cao<sup>1</sup>, Linmei Hu<sup>2</sup>, Ling Feng<sup>1</sup>, Lei Hou<sup>1</sup>\*, Juanzi Li<sup>1</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Beijing Institute of Technology zhangjj23@mails.tsinghua.edu.cn

## Abstract

Program induction (PI) has become a promising paradigm for using knowledge bases (KBs) to help large language models (LLMs) answer complex knowledge-intensive questions. Nonetheless, PI typically relies on a large number of parallel question-program pairs to make the LLM aware of the schema of a given KB, and is thus challenging for many low-resourced KBs that lack annotated data. To this end, we propose KB-Plugin, a plug-and-play framework that enables LLMs to induce programs over any low-resourced KB. Firstly, KB-Plugin adopts self-supervised learning to encode the detailed schema information of a given KB into a pluggable module, namely schema plugin. Secondly, KB-Plugin utilizes abundant annotated data from a rich-resourced KB to train another pluggable module, namely PI plugin, which can help the LLM extract questionrelevant schema information from the schema plugin of any KB and utilize the information to induce programs over this KB. Experiments show that KB-Plugin outperforms SoTA lowresourced PI methods with 25 smaller backbone LLM on both large-scale and domainspecific KBs, and even approaches the performance of supervised methods.

## 1 Introduction

Recently, the usage of knowledge bases (KBs) as external resources to assist large language models (LLMs) (Brown et al., 2020; Zhao et al., 2023) in answering complex knowledge-intensive questions has gained increasing study (Pan et al., 2023; Li et al., 2023b; Jiang et al., 2023). Among various methods, program induction (PI) has emerged as a promising paradigm due to its good interpretability and capacity to support complex reasoning operations (Cao et al., 2022a; Gu et al., 2023; Li et al., 2023b). Given a KB, PI methods employ LLMs to convert a question into a multi-step program (e.g., KoPL (Cao et al., 2022a) and S-expression (Su et al., 2016)), whose execution against the KB produces the answer. Despite strong capacity, most PI methods rely on individual training for each KB using a large number of manually annotated questionprogram pairs (Xie et al., 2022; Li et al., 2023b; Luo et al., 2023). As for many low-resourced KBs that lack program annotations, how to enable LLMs to utilize their knowledge via PI remains a challenging problem.

![](images/24b17b740210d924d21c55b8cb375ea423f475b1a1616c6ecc65fe3b323ab898.jpg)  
Figure 1: Illustration of KB-Plugin. By simply plugging the schema plugin of a KB and the PI plugin, the LLM is injected with the schema information of this KB and the ability to induce programs over it.

Recent studies (Cao et al., 2022b; Li et al., 2023a) have indicated that the mapping from questions to program sketches (i.e., composed functions without arguments, such as Find Relate FilterConcept) primarily correlates with language compositional structures and is thus transferable across KBs. Hence the main challenge for PI over low-resourced KBs is to determine the argument for each function (Gu and Su, 2022), which requires LLMs to link natural language in a question to corresponding schema items (i.e., pre-defined relations and concepts) in the KB (e.g., in Fig 1, the relation “part of network” and the concept “rail network” are arguments of function Relate and FilterConcept, respectively), so it is important to provide LLMs adequate information of each schema item. A straightforward approach is to directly feed all the schema information to the LLM via a prompt. However, the broad schema of KBs and limited context windows of LLMs make this infeasible (Li et al., 2023a).

Regarding the above challenges, we are inspired by recent studies that claim the parameters of LLMs can encode task-specific knowledge (Saxena et al., 2022; Moiseev et al., 2022; Wang et al., 2022). Our basic idea is to encode detailed schema information of a KB into the parameters of a pluggable module (e.g., LoRA (Hu et al., 2022)), namely schema plugin, so as not to be hampered by limited context windows like the prompt-based approach. Then we use another pluggable module, namely PI plugin, to help the LLM capture question-relevant schema information from the schema plugin and utilize this information to induce programs. As illustrated in Fig. 1, by simply plugging the schema plugin of a KB and the PI plugin, the LLM is injected with the schema information of this KB and the ability to induce programs over it. We name this framework KB-Plugin. To implement KB-Plugin, there remain two key problems: (1) By what task can sufficient information about each schema item in a KB be encoded into its schema plugin? (2) Without annotated data from the low-resource KBs, how can the PI plugin learn to extract and utilize questionrelevant schema information from their schema plugins to induce programs over these KBs?

To solve the above problems, we propose a novel plugin learning and transfer framework. First, inspired by prior studies (Bordes et al., 2013; Lin et al., 2015) which show that schema items in a KB can be well represented by fact triples involving them, we propose to learn schema plugins via a self-supervised triple completion task. Specifically, given a KB, we plug a schema plugin into the LLM and tune the plugin to enable the LLM to complete relevant triples for each schema item in the KB. In this way, the detailed schema information can be encoded into this schema plugin. As for PI plugin learning, inspired by Cao et al. (2022b), we utilize abundant program annotations from a rich-resourced KB. Specifically, we use this KB to generate multiple KBs with different schemas via alias replacement and train a schema plugin for each of them. Given a training question, we plug these schema plugins alone with the

PI plugin into the LLM in turn and train the PI plugin to make the LLM generate the correct program whose arguments conform to the currently plugged schema plugin. In this way, the PI plugin is forced to learn the skills of extracting and utilizing question-relevant schema information from the plugged schema plugin for PI over the corresponding KB. Besides, since the PI plugin is trained to be compatible with different schema plugins, it can be directly transferred to other low-resourced KBs and generalize well with their schema plugins, even if most schema items in these KBs are unseen during its training.

In experiments, we take Wikidata-based KQA Pro as the rich-resourced KB to train the PI plugin, and evaluate our framework on three largescale Freebase-based datasets (WebQSP, GraphQ, and GrailQA) and two domain-specific datasets (MetaQA for movie domain and SoAyBench for academic domain). The results show that KB-Plugin outperforms SoTA low-resourced PI methods with 25 smaller backbone LLM, demonstrating its scalability to extremely complex schemas and adaptability to various domains. On GraphQ, GrailQA, and MetaQA, KB-Plugin even surpasses the performance of several supervised methods.

Our contributions include: (1) proposing KB-Plugin, a novel plug-and-play framework that enables LLMs to induce programs over any lowresourced KB; (2) empirically validating the efficacy of KB-Plugin for both large-scale and domainspecific KBs through extensive experiments.

## 2 Related Work

Low-resourced Program Induction. Recently, there have emerged three types of PI methods for low-resourced KBs that lack program annotations, but each of them has limitations: (1) Few-shot program generation methods (Gu et al., 2023; Li et al., 2023a) utilize in-context learning ability of LLMs to induce programs with a handful of demonstrations. However, they can only determine function arguments based on the schema item names due to limited context windows, so they face challenges in distinguishing similar schema items. They also suffer from long inference time due to excessive LLM calls or executing a vast number of potential programs; (2) Few-shot data generation methods (Li et al., 2023c) also employ in-context learning with LLMs to convert automatically sampled programs into questions, and train a smaller PI model using the generated question-program pairs. Nonetheless, the generated questions may not align with programs and often lack diversity due to the limited number of program templates; (3) Similar to us, program transfer methods (Cao et al., 2022b) also leverage program annotations from a rich-resourced KB to aid PI for low-resourced KBs. However, they mainly focus on program sketch transfer and perform poorly without fine-tuning using annotated question-answer pairs from lowresourced KBs to adapt to their schemas. While KB-plugin obviates the reliance on any annotated data from low-resourced KBs, thereby enabling LLMs to easily utilize their knowledge.

Plug-and-Play Modules for LLMs. In recent years, various parameter-efficient modules have been proposed to adapt LLMs to different downstream tasks (Lester et al., 2021; Hu et al., 2022; Li and Liang, 2021; Pfeiffer et al., 2021) . These modules show plug-and-play characteristics and can inject task-specific knowledge and skills into LLMs (Xiao et al., 2023; Zhang et al., 2023). Some researchers also found that pluggable modules for similar tasks encode knowledge and skills into the parametric space in similar ways (Qin et al., 2021; Su et al., 2022), providing basic rationality for the transferability of our PI plugin.

## 3 Problem Formulation

In this section, we first provide some necessary definitions and then formulate our task.

Knowledge Base. A knowledge base (KB) can be formalized as $\mathcal { K B } = \{ \mathcal { C } , \mathcal { E } , \mathcal { R } , \mathcal { T } \}$ , where ${ \mathcal { C } } , { \mathcal { E } }$ and $\tau$ represent the sets of concepts, entities, relations and fact triples, respectively. Specifically, $\mathcal { R } = \{ r _ { e } , r _ { c } \} \cup \mathcal { R } _ { l }$ , where $r _ { e }$ is “instance $o f ' , r _ { c }$ is “subclass $o f '$ , and $\mathcal { R } _ { l }$ is the set of other general relations. Correspondingly, can be divided into there disjoint subsets: (1) “instance $o f '$ triples $\mathcal { T } _ { e } = \{ ( e , r _ { e } , c ) | e \in \mathcal { E } , c \in \mathcal { C } \}$ ; (2) “subclass $o f '$ triples $\mathcal { T } _ { c } = \{ ( c _ { i } , r _ { c } , c _ { j } ) | c _ { i } , c _ { j } \in \mathcal { C } \}$ ; (3) relational triples $\mathcal { T } _ { l } = \{ ( e _ { i } , r , e _ { j } ) | e _ { i } , e _ { j } \in \mathcal { E } , r \in \mathcal { R } _ { l } \}$ . Elements in $\mathcal { C }$ and are also called the schema items of .

Program Induction. Given a KB and a natural language question $x = \langle w _ { 1 } , w _ { 2 } , \cdot \cdot \cdot , w _ { | x | } \rangle$ program induction (PI) aims to convert x into a program $y ,$ which would return the correct answer when executed against . Formally, y is composed of functions that take a specific type of arguments, and can be serialized as $y =$ $\left. f _ { 1 } ( a r g _ { 1 } ) , \cdots , f _ { t } ( a r g _ { t } ) , \cdots , f _ { | y | } ( a r g _ { | y | } ) \right. , f _ { t } \in$ $\mathcal { F } , a r g _ { t } \in \mathcal { E } \cup \mathcal { C } \cup \mathcal { R } \cup \{ \emptyset \}$ . Here, $\mathcal { F }$ is a set of pre-defined functions that cover basic reasoning operations on KBs. In this work, we use KoPL (Cao et al., 2022a) as our programming language.

Task Formulation. Suppose we have access to (1) source KB $\kappa B ^ { S }$ and source domain data $\mathcal { D } ^ { S } = \{ ( x _ { i } ^ { S } , y _ { i } ^ { S } ) \} _ { i = 1 } ^ { n ^ { S } }$ , which are question-program pairs for $\kappa B ^ { S }$ ; (2) target KB $\kappa B ^ { T }$ , which is lowresourced and has no annotated data. The goal is to learn a PI model $M _ { P I } ^ { T }$ that can translate a question $x ^ { T }$ for $\kappa B ^ { T }$ into program $y ^ { T }$ , whose execution on $\kappa B ^ { T }$ produces the correct answer.

## 4 Methodology

As mentioned in the introduction, to enable a LLM M to induce programs over low-resourced $\kappa B ^ { T }$ KB-Plugin learns two types of pluggable modules for M: (1) KB-specific schema plugin $m _ { s c } ,$ which stores information of schema items of a given KB within its parameters; (2) KB-transferable PI plugin $m _ { P I }$ , which encodes the skill of inducing programs over any KB by extracting and utilizing question-relevant schema information from the schema plugin of this KB. It is trained with $\kappa B ^ { S }$ and $\mathcal { D } ^ { S }$ but can be directly transferred to $\kappa B ^ { T }$ The final PI model for $\kappa B ^ { T }$ can be formulated as

$$
M _ { P I } ^ { T } = \mathrm { p l u g } ( M , \{ m _ { s c } ^ { T } , m _ { P I } \} ) ,\tag{1}
$$

where $m _ { s c } ^ { T }$ is the schema plugin of $\kappa B ^ { T }$ and plug(M, ) means plugging the plugins in into M. In the following, we will first introduce the architecture of two types of plugins, then present our plugin learning and transfer framework.

## 4.1 Plugin Architecture

A host of studies have demonstrated that knowledge and skills can be encapsulated within the parameters of LLMs (Saxena et al., 2022; Moiseev et al., 2022; Wang et al., 2022). Inspired by this, we implement both schema plugin and PI plugin with LoRA (Hu et al., 2022), a popular type of pluggable module for LLMs with a few trainable parameters.

Specifically, let $L _ { M }$ be the set of weight matrices in the self-attention modules and MLP modules of a LLM M. For each $W _ { i } \in \mathbb { R } ^ { d \times k }$ in $L _ { M }$ , LoRA modifies its forward pass from h = W x to h = $( W _ { i } + A _ { i } B _ { i } ) x .$ , where $A _ { i } \in \mathbb { R } ^ { d \times r }$ and $B _ { i } \in \mathbb { R } ^ { r \times k }$ are two matrices with rank $r \ll$ min(d, k). A

![](images/e20640e7119d31cf6941697efb3523ed3ca9f91f8aa23fdd440004264a349c3e.jpg)  
Figure 2: Overview of our plugin learning and transfer framework: (a) Generate multiple source KBs with different schemas and augmented source domain data via alias replacement; (b) Learn an individual schema plugin for each source KB and the target KB via self-supervised schema-relevant triple completion task; (c) Train the PI plugin by inducing program for each source KB when plugging it into the LLM along with the corresponding schema plugin. (d) Transfer the PI plugin by plugging it into the LLM with the schema plugin of the target KB and inducing programs over the target KB with constrained decoding.

LoRA plugin $m _ { j }$ is thus defined as

$$
m _ { j } = \{ ( A _ { i } ^ { m _ { j } } , B _ { i } ^ { m _ { j } } ) | W _ { i } \in L _ { M } \} ,\tag{2}
$$

and $\operatorname { p l u g } ( M , \{ m _ { 1 } , . . . , m _ { N } \} )$ means replacing all $\begin{array} { r l r l r l } { W _ { i } } & { { } } & { \in } & { { } } & { L _ { M } } \end{array}$ with $W _ { i } ~ +$ $\sum _ { j = 1 } ^ { N } \hat { A } _ { i } ^ { m _ { j } } B _ { i } ^ { m _ { j } }$ If we train $\begin{array} { r l } { M ^ { \prime } } & { { } = } \end{array}$ plu $\operatorname { g } ( \operatorname { f z } ( M ) , \left\{ \operatorname { f z } ( m _ { 1 } ) , \ldots , \operatorname { f z } ( m _ { N - 1 } ) , m _ { N } \right\} )$ on a certain task, where $\operatorname { f z } ( \cdot )$ represents parameter freezing, knowledge and skills related to this task will be encoded within $m _ { N }$ . Although other parameter-efficient pluggable modules such as prefix-tuning (Li and Liang, 2021) can also serve as our plugin modules, the advantages of LoRA are that it does not increase input length or inference latency.

## 4.2 Plugin Learning and Transfer Framework

There are two primary challenges for learning schema plugins and the PI plugin: (1) How to encode sufficient information about each schema item of a KB into a schema plugin? (2) How to ensure that the PI plugin can extract and utilize useful schema information for program induction from schema plugins of different KBs, instead of ignoring the schema plugin entirely, directly learning to induce program over source KB during training, and consequently losing transferability?

To handle these challenges, we propose a novel plugin learning and transfer framework, which is illustrated in Fig. 2 and contains four steps: (1) Generate multiple source KBs $\kappa B ^ { S _ { 1 } } , \ldots , \kappa B ^ { S _ { N } }$ with different schemas and augmented data $\begin{array} { c c l } { \mathcal { D } _ { a } ^ { S } } & { = } & { \{ ( x _ { j } ^ { S } , y _ { j } ^ { S _ { 1 } } , \ldots , y _ { j } ^ { S _ { N } } ) \} _ { j = 1 } ^ { n ^ { S } } } \end{array}$ based on $\kappa B ^ { S }$ and $D ^ { S }$ via alias replacement, where $y _ { j } ^ { S _ { i } }$ is the golden program for question $x _ { j } ^ { S }$ on $\dot { \iota } \dot { B } ^ { S _ { i } } ; ( 2 )$ Learn individual schema plugin $\dot { m } _ { s c } ^ { S _ { i } }$ for each $\kappa B ^ { S _ { i } }$ via self-supervised schemarelevant triple-completion task; (3) Train PI plugin m<sub>PI</sub> by requiring $M _ { P I } ^ { S _ { 1 } } , \dots , M _ { P I } ^ { S _ { N } }$ to generate $y _ { j } ^ { S _ { 1 } } , \ldots , y _ { j } ^ { S _ { N } }$ given $\boldsymbol { x } _ { j } ^ { S }$ , respectively, where $M _ { P I } ^ { S _ { i } } \ = \ \mathrm { P l u g } ( \mathrm { f z } ( M )$ $\{ \mathrm { f z } ( m _ { s c } ^ { S _ { i } } ) , m _ { P I } \} )$ , so that m is forced to extract and utilize schema information from each $m _ { s c } ^ { S _ { i } } ; ( 4 )$ Learn schema plugin $m _ { s c } ^ { T }$ for $\kappa B ^ { T }$ using the same method in (2) and take $M _ { P I } ^ { T } = \mathrm { p l u g } ( M , \{ m _ { s c } ^ { T } , m _ { P I } \} )$ as the final PI model for $\kappa B ^ { T }$ . We will introduce each step in detail in the following.

## 4.2.1 KB Generation and Data Augmentation

We utilize the aliases of each schema item to generate multiple KBs with different schemas based on $\mathcal { K } \mathcal { B } ^ { S } = \bar { \{ \mathcal { C } ^ { S } , \mathcal { E } ^ { S } , \mathcal { R } ^ { S } , \mathcal { T } ^ { S } \} }$ . As shown in Fig. 2(a), for each schema item $v \in \mathcal { C } ^ { S } \cup \mathcal { R } ^ { S }$ , we replace v with $v _ { i } ,$ a randomly chosen alias of $v ,$ and record $a _ { i } ( v ) = v _ { i }$ . For example, the concept “basketball team” can be replaced with “basket club” and the relation “member ofsports team” can be replaced with $^ { \mathrm {  } } p l a y s f o r ^ { \mathrm { \Rightarrow } }$ . Relevant triples in $\mathcal { T } ^ { S }$ are also modified with the same alias. In this way, $\kappa B ^ { S _ { i } }$ that has a different schema than $\kappa B ^ { S }$ is created. In practice, we let $\kappa B ^ { S _ { 1 } } = \kappa B ^ { S }$ and repeat above process $N - 1$ times to generate $\kappa B ^ { S _ { 2 } } , \ldots , \kappa B ^ { S _ { N } }$

Similarly, for each question-program pair $( x _ { j } ^ { S } , y _ { j } ^ { S } ) \quad \in \quad { \mathcal { D } } ^ { S }$ , suppose $y _ { j } ^ { S }$ $\Big \langle f _ { 1 } ( a r g _ { 1 } ) , \cdot \cdot \cdot , f _ { t } ( a r g _ { t } ) , \cdot \cdot \cdot , f _ { | y _ { j } ^ { S } | } ( a r g _ { | y _ { j } ^ { S } | } ) \Big \rangle$ we replace every arg<sub>t</sub> $\in \mathcal { C } ^ { S } \cup \mathcal { R } ^ { S }$ with $a _ { i } ( a r g _ { t } )$ to obtain $y _ { j } ^ { S _ { i } }$ , which is the correct program for $\boldsymbol { x } _ { j } ^ { S }$ executable on $\mathcal { \kappa } B ^ { S _ { i } }$ . We repeat the process for $\mathcal { \kappa } B ^ { S _ { 1 } } , \ldots , \mathcal { \kappa } B ^ { S _ { N } }$ to obtain augmented data $\mathcal { D } _ { a } ^ { S } = \{ ( x _ { j } ^ { S } , y _ { j } ^ { S _ { 1 } } , \ldots , y _ { j } ^ { S _ { N } } ) \} _ { j = 1 } ^ { n ^ { S } }$

## 4.2.2 Learning of Schema Plugin

Many studies about knowledge graph embedding show that the information of schema items in a KB can be represented by not only their names but also triples containing them (Bordes et al., 2013; Lv et al., 2018). Inspired by this, we propose to encode schema information into schema plugins via a self-supervised triple completion task. As illustrated in Fig. 2(b), to learn the schema plugin $m _ { s c }$ for a given KB $\mathcal { K B } = \{ \mathcal { C } , \mathcal { E } , \mathcal { R } , \mathcal { T } \}$ , where $\mathcal { T } =$ $\mathcal { T } _ { e } \cup \mathcal { T } _ { c } \cup \mathcal { T } _ { l } .$ , we train $M _ { s c } = \mathrm { P l u g } ( \mathrm { f z } ( M ) , m _ { s c } )$ to complete relevant triples for each concept and relation in in sequence-to-sequence form as follows.

First, for each concept $c \in \mathcal { C }$ , we require $M _ { s c }$ to complete relevant “instance $o f '$ triples to aggregate the semantic features of entities belonging to $c .$ Specifically, we sample $K$ triples $( e _ { k }$ , instance $o f , c )$ from $\mathcal { T } _ { e }$ (see Appendix B for detailed sampling strategy), and use each sampled triple to construct two pairs of verbalized queries and answer as the inputs and expected outputs for $M _ { s c } .$

$\mathbf { \Psi } ^ { 6 6 } \langle e _ { k } \rangle$ || instance $o f ^ { * }  { ^ { * } \langle c \rangle } ^ { * }$ ;

$^ { 6 6 } \langle c \rangle$ || contains instance $\ "  \cdots \langle e _ { k } \rangle \ "$

Here, $\langle e _ { k } \rangle$ and $\langle c \rangle$ means filling in the names of $e _ { k }$ and $^ { c , }$ respectively.

Besides, the information of a concept is also related to its sub- and super-concepts. Therefore, for each triple $( c _ { i }$ , subclass of, $c _ { j } ) \in \mathcal { T } _ { c }$ , we also construct two queries with answers for $M _ { s c } \mathrm { : }$

$^ { 6 6 } \langle c _ { i } \rangle$ || subclass $o f ^ { * }  { ^ { * } \langle c _ { j } \rangle } ^ { * }$

$^ { 6 6 } \langle c _ { j } \rangle$ || contains subclass $"  " \langle c _ { i } \rangle "$

Finally, the information of a relation can be learned from its name and the elements connected by it. Therefore, for each $r \in R _ { l }$ , we sample K triples $( e _ { i } , r , e _ { j } )$ from $\begin{array} { r } { \tau _ { l } , } \end{array}$ choose $c _ { i } , \ : c _ { j }$ such that $( e _ { i } .$ , instance\_ ${ \mathrm { . o f } } , c _ { i } ) , ( e _ { j }$ , instance\_of, $c _ { j } ) ~ \in ~ T _ { e }$ and use each $( e _ { i } , c _ { i } , r , e _ { j } , c _ { j } )$ to construct three queries with answers:

$\bullet \ \tilde { \langle \ e _ { i } \rangle } \ | \ \langle \boldsymbol { c } _ { i } \rangle$ <sup>||</sup> ⟨<sup>r</sup>⟩ <sup>|</sup> <sup>forward”</sup> → <sup>“</sup>⟨<sup>c</sup>j ⟩ <sup>|</sup> ⟨<sup>e</sup>j ⟩<sup>”;</sup>

$\bullet \ ^ { \ast } \langle e _ { j } \rangle \mid \langle c _ { j } \rangle \parallel \langle r \rangle$ | backward” $\ ' \to \cdots \langle c _ { i } \rangle \mid \langle e _ { i } \rangle ^ { \ast } ;$ $\bullet \ \tilde { \ } \langle e _ { i } \rangle \mid \langle c _ { i } \rangle$ || what relation || $\langle c _ { j } \rangle \mid \langle e _ { j } \rangle ^ { \prime \prime } $ $" \langle r \rangle ^ { \flat }$

We empirically find that including $c _ { i } , \ : c _ { j }$ benefits the information encoding for both concepts and relations.

Let the set of all generated queries and answers be $D _ { s c } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { l }$ , then $m _ { s c }$ is trained to minimize

$$
\mathcal { L } _ { s c } = - \sum _ { ( q _ { i } , a _ { i } ) \in D _ { s c } } \log P ( a _ { i } | q _ { i } ) ,\tag{3}
$$

where $P ( a _ { i } | q _ { i } )$ is the likelihood of $M _ { s c }$ generating $a _ { i }$ given $q _ { i } ,$ defined by token-level cross entropy. Note that the learning of $m _ { s c }$ does not rely on any additional data except the KB itself, so we can train a schema plugin for any KB.

## 4.2.3 Learning of PI Plugin

As illustrated in Fig. 2(c), to learn the PI plugin $m _ { P I }$ , we first train individual schema plugin $m _ { s c } ^ { S _ { i } }$ for each $\kappa B ^ { S _ { i } }$ After that, given $( x _ { j } ^ { S } , y _ { j } ^ { S _ { 1 } ^ { S } } , \ldots , y _ { j } ^ { S _ { N } } ) \ \in \ D _ { a } ^ { S }$ , where $x _ { i } ^ { S }$ is a question and $y _ { j } ^ { S _ { i } }$ is the golden program for $\boldsymbol { x } _ { j } ^ { S }$ on $\kappa B ^ { S _ { i } }$ , we train m<sub>PI</sub> by feeding $x _ { i } ^ { S }$ to $M _ { P I } ^ { S _ { 1 } } , \dots , M _ { P I } ^ { S _ { N } }$ and requiring them to generate $y _ { j } ^ { S _ { 1 } } , \ldots , y _ { j } ^ { S _ { N } }$ , respectively. Here, $M _ { P I } ^ { S _ { i } } ~ =$ $\mathrm { P l u g } \bar { ( } \mathrm { f z } ( M ) , \bar { \{ }  \mathrm { f z } ( m _ { s c } ^ { S _ { i } } ) , m _ { P I } \} ,$ ). The overall objective can be formulated as:

$$
\mathcal { L } _ { P I } = - \sum _ { ( x _ { j } ^ { S } , y _ { j } ^ { S _ { 1 } } , . . . , y _ { j } ^ { S _ { N } } ) \in D _ { a } ^ { S } } \sum _ { i = 1 } ^ { N } \log P _ { i } ( y _ { j } ^ { S _ { i } } | x _ { j } ^ { S } ) ,\tag{4}
$$

where $P _ { i } ( y _ { j } ^ { S _ { i } } | x _ { j } ^ { S } )$ is the likelihood of $M _ { P I } ^ { S _ { i } }$ generating $y _ { j } ^ { S _ { i } }$ given $x _ { j } ^ { S }$ , defined by token-level cross entropy. To generate programs conforming to different schemas given the same question, $m _ { P I }$ must learn to (1) choose correct functions according to the compositional structure of the question; (2) extract and utilize question-relevant schema information for argument determination from the corresponding schema plugin, because it is the only difference among $\hat { M _ { P I } ^ { S _ { 1 } } } , \dots , M _ { P I } ^ { S _ { N } }$

## 4.2.4 Plugin Transfer

Once the PI plugin $m _ { P I }$ is trained, we directly transfer it to $\dot { \kappa } \boldsymbol { B ^ { T } }$ as in $\mathrm { F i g } 2 \left( \mathrm { d } \right)$ , and let $M _ { P I } ^ { T } =$ plug $( M , \{ m _ { s c } ^ { T } , m _ { P I } \} )$ be the PI model for $\kappa B ^ { T }$ Here, $m _ { s c } ^ { T }$ is the trained schema plugin for $\kappa B ^ { T }$ using the method in Sec. 4.2.2. Since $m _ { s c } ^ { T }$ and $m _ { s c } ^ { S _ { i } }$ are trained with the same tasks, we expect that they encode schema information into their parameters in similar ways (Qin et al., 2021; Su et al., 2022), so $m _ { P I }$ can also extract schema information from $m _ { s c } ^ { T }$ to help PI over $\kappa B ^ { T }$ . Besides, to guarantee $M _ { P I } ^ { T }$ generating valid programs which do not cause execution error or return an empty answer, we adopt constrained decoding, i.e., after $M _ { P I } ^ { T }$ generates $f _ { 1 } ( a r g _ { 1 } ) , \ldots , f _ { t } ( a r g _ { t } )$ , we enumerate all the valid $f _ { t + 1 } ( a r g _ { t + 1 } )$ following the method of Gu et al. (2023) and restrict $M _ { P I } ^ { \widecheck { T } }$ to only generate one of them. More details are in Appendix C. We also use beam search to retain top-k programs during decoding to provide $M _ { P I } ^ { T }$ with a more global view.

## 5 Experiments

## 5.1 Datasets

Source Domain. We use KQA Pro (Cao et al., 2022a) as the source domain dataset. It provides 117,970 question-program pairs with diverse compositional structures based on a small subset of Wikidata (Vrandecic and Krötzsch, 2014).

Target Domain. We use WebQSP (Yih et al., 2016), GraphQ (Su et al., 2016), GrailQA (Gu et al., 2021), MetaQA (Zhang et al., 2018) and SoAyBench (Wang et al., 2024) as the target domain datasets. Among them, WebQSP, GraphQ, and GrailQA are based on Freebase (Bollacker et al., 2008). Their KBs contain a large number of schema items and cover various domains, thus can evaluate the effectiveness of KB-Plugin for large-scale KBs. MetaQA and SoAyBench are two datasets in movie and academic domains, respectively, and can evaluate the adaptability to specific domains in detail. For MetaQA, since most of the relations in its KB have been covered by KQA Pro, we remove these relations and relevant questionprogram pairs from KQA Pro to avoid data leakage. For SoAyBench which is originally a tool-using dataset based on Aminer (Tang et al., 2008) APIs, we construct its KB by collecting relevant data from these APIs. Table 1 shows the statistics of these datasets and their overlap with source KBs generated from KQA Pro. Most schema items in the target KBs are unseen in source KBs and most test cases also involve unseen schema items.

<table><tr><td>Dataset</td><td>||DM||</td><td>|R|</td><td> $\left| \mathcal { R } _ { u } \right|$ </td><td>|C|</td><td> $| \mathcal { C } _ { u } |$ </td><td> $| \mathcal { D } ^ { \mathrm { t e s t } } |$ </td><td> $| \mathcal { D } _ { u } ^ { \mathrm { t e s t } } |$ </td></tr><tr><td>KQA Pro</td><td>-</td><td>1209</td><td></td><td>794</td><td>1</td><td></td><td></td></tr><tr><td>WebQSP</td><td>56</td><td>412</td><td>296</td><td>446</td><td>363</td><td>1639</td><td>1083</td></tr><tr><td>GraphQ</td><td>70</td><td>9569</td><td>8931</td><td>7298</td><td>7004</td><td>2395</td><td>2340</td></tr><tr><td>GrailQA(dev)</td><td>86</td><td>3938</td><td>3524</td><td>2018</td><td>1868</td><td>6763</td><td>6578</td></tr><tr><td>GrailQA(test)</td><td>86</td><td>3938</td><td>3524</td><td></td><td>20181868</td><td>13231</td><td></td></tr><tr><td>MetaQA</td><td>1</td><td>9</td><td>9</td><td>9</td><td>3</td><td>39093</td><td>39093</td></tr><tr><td>SoAyBench</td><td>1</td><td>17</td><td>11</td><td>5</td><td>3</td><td>792</td><td>756</td></tr></table>

Table 1: Statistics for source and target domain datasets and their overlaps with 16 source KBs generated from KQA Pro. $\left| \mathcal { D M } \right| / \left| \mathcal { R } \right| / \left| \mathcal { C } \right|$ denotes the number of domains / relations / concepts in their KBs. $\left| \mathcal { R } _ { u } \right| / \left| \mathcal { C } _ { u } \right|$ denotes the number of relations / concepts unseen in the source KBs. $| \mathcal { D } ^ { \mathrm { t e s t } } |$ and $| \mathcal { D } _ { u } ^ { \mathrm { t e s t } } |$ denotes the numbers of test cases and test cases that involve unseen schema items, respectively.

## 5.2 Baselines

For WebQSP, GraphQ, GrailQA, and MetaQA, we mainly compare KB-Plugin with low-resourced PI methods including (1) few-shot program generation methods Pangu (Gu et al., 2023) and KB-BINDER (Li et al., 2023a); (2) few-shot data generation method APS (Li et al., 2023c); (3) program transfer method ProgramTrans (Cao et al., 2022b), where we adopt its results without fine-tuning on target KBs for fair comparison. In addition, we also provide the results of several representative supervised models for comparison.

For SoAyBench, we choose tool-using methods that were evaluated on it as baselines, including DFSDT (Qin et al., 2023) and SoAy (Wang et al., 2024). These methods solve questions by prompting LLMs to call Aminer APIs in specific orders via in-context learning. Their processes of determining the composition of APIs and filling in arguments for each API can also be viewed as program induction.

We provide detailed descriptions of all the baselines and our evaluation metrics in Appendix D.1.

## 5.3 Implementation Details

In experiments, we use Llama2-7B (Touvron et al., 2023) as the backbone LLM of KB-Plugin and set the rank r of LoRA to 16. The number of parameters of each plugin is consequently 40M, which is extremely lightweight. The aliases of schema items in KQA Pro are directly obtained from Wikidata. The number of generated source KBs is set to 16 to balance performance and training efficiency. The sampling number K in schema plugin learning is set to be 500, 500, 50, 100, 3000, and 1000 for KQA Pro, WebQSP, GraphQ, GrailQA, MetaQA, and SoAyBench, respectively, to limit the size of the constructed data for schema plugin learning. We use beam size 5 for all experiments. More details can be found in Appendix D.2.

<table><tr><td>Method</td><td>WebQSP</td><td>GraphQ</td><td>Test Dev</td><td>GrailQA</td></tr><tr><td colspan="3">Supervised</td><td></td><td></td></tr><tr><td>QGG</td><td>74.0</td><td></td><td>36.7</td><td></td></tr><tr><td>BERT+Ranking</td><td></td><td>25.0</td><td>58.0</td><td></td></tr><tr><td>ArcaneQA</td><td>75.6</td><td>31.8</td><td>73.7</td><td>76.8</td></tr><tr><td>RnG-KBQA</td><td>75.6</td><td></td><td>74.4</td><td>76.9</td></tr><tr><td colspan="3">Low-resourced</td><td></td><td></td></tr><tr><td>ProgramTrans</td><td>53.8*</td><td></td><td></td><td></td></tr><tr><td>APS</td><td>51.1</td><td></td><td>57.7</td><td>62.1</td></tr><tr><td>KB-BINDER</td><td>53.2</td><td>39.5</td><td>56.0</td><td></td></tr><tr><td>Pangu</td><td>54.5</td><td>43.3</td><td>62.7</td><td></td></tr><tr><td>KB-Plugin</td><td>57.2 / 61.1*</td><td>49.5</td><td>62.7</td><td>65.0</td></tr><tr><td>w/o schema plugin</td><td>41.0</td><td>42.8</td><td></td><td>57.5</td></tr><tr><td>w/  $m _ { s c } ^ { S _ { 0 } }$ </td><td>48.0</td><td>37.9</td><td></td><td>51.0</td></tr></table>

Table 2: F1 results on WebQSP, GraphQ, and GrailQA. \* means using oracle topic entities.
<table><tr><td>Method</td><td>|1-hop</td><td>2-hop</td><td>3-hop</td></tr><tr><td colspan="4">Supervised</td></tr><tr><td>KV-Mem</td><td>96.2</td><td>82.7</td><td>48.9</td></tr><tr><td>PullNet</td><td>97.0</td><td>99.9</td><td>91.4</td></tr><tr><td>EmbedKGQA</td><td>97.5</td><td>98.8</td><td>94.8</td></tr><tr><td>TransferNet</td><td>97.5</td><td>100.0</td><td>100.0</td></tr><tr><td colspan="4">Low-resourced</td></tr><tr><td>KB-BINDER</td><td>93.5</td><td>99.6</td><td>96.4</td></tr><tr><td>KB-Plugin</td><td>97.1</td><td>100.0</td><td>99.3</td></tr><tr><td>w/o schema plugin</td><td>92.6</td><td>99.0</td><td>98.9</td></tr><tr><td>w/  $m _ { s c } ^ { S _ { 0 } }$ </td><td>90.4</td><td>93.6</td><td>88.6</td></tr></table>

Table 3: Hit@1 results on MetaQA.

## 5.4 Main Results

The results are presented in Table 2, 3 and 4. Compared with Pangu, the SoTA PI method for lowresourced KBs, KB-Plugin improves the F1 score by 2.7% and 6.2% on WebQSP and GraphQ, respectively, and achieves comparable performance on GrailQA, despite Pangu using 25 larger model (175B Codex) and 100 annotated examples from each dataset. Moreover, Pangu needs to call Codex hundreds of times for a question to score each candidate program, while our model selects the optimal program via beam search, which is significantly faster and less costly. Besides, since ProgramTrans, KB-BINDER, and Pangu all link questions to schema items according to their names only, the superiority of KB-Plugin also demonstrates the benefits of aggregating additional schema information from relevant triples via schema plugin learning. KB-Plugin even surpasses several supervised models on GraphQ and GrailQA, which demand training using thousands of annotated samples from target KBs, showing the effectiveness of transferring prior knowledge from rich-resourced KBs.

<table><tr><td>Method</td><td>Acc</td></tr><tr><td>DFSDT (gpt-3.5-turbo)</td><td>45.7</td></tr><tr><td>DFSDT (gpt-4) SoAy (gpt-3.5-turbo)</td><td>59.7</td></tr><tr><td>SoAy (gpt-4)</td><td>67.7 88.7</td></tr><tr><td>KB-Plugin</td><td>90.8</td></tr><tr><td>w/o schema plugin</td><td>70.8</td></tr><tr><td>wl msc</td><td>64.0</td></tr></table>

Table 4: Accuracy results on SoAyBench.

<table><tr><td>Dataset</td><td>Method</td><td> $\mathcal { D } _ { \mathrm { s e e n } } ^ { \mathrm { t e s t } }$ </td><td> $\mathcal { D } _ { \mathrm { u n s e e n } } ^ { \mathrm { t e s t } }$ </td></tr><tr><td>WebQSP</td><td>KB-Plugin w/o schema plugin Gain</td><td>64.9 47.6 +17.4</td><td>53.3 37.6 +15.7</td></tr><tr><td>GraphQ</td><td>KB-Plugin w/o schema plugin Gain</td><td>40.0* 70.9* -30.9*</td><td>49.7 42.2 +7.5</td></tr><tr><td>GrailQA-dev</td><td>KB-Plugin 1 w/o schema plugin Gain</td><td>69.0 64.9 +4.1</td><td>64.8 57.3 +7.5</td></tr></table>

Table 5: F1 Results of KB-Plugin with and without schema plugin. $\mathcal { D } _ { \mathrm { u n s e e n } } ^ { \mathrm { t e s t } }$ and $\mathcal { D } _ { \mathrm { s e e n } } ^ { \mathrm { t e s t } }$ denote the sets of test cases that involve and do not involve schema items unseen in the source KBs, respectively. means the results may not be indicative since there are only 55 cases in $\mathcal { D } _ { \mathrm { s e e n } } ^ { \mathrm { t e s t } }$ of GraphQ.

On MetaQA and SoAyBench, KB-Plugin outperforms all the low-resourced baselines even though they use more powerful LLMs (i.e., Codex, gpt-3.5- turbo, and gpt-4), indicating that our framework also performs well for domain-specific KBs. In particular, KB-Plugin achieves strong performance on par with supervised SoTAs on MetaQA even if it does not see any target relations from the source domain.

## 5.5 Ablation Study

To demonstrate the effect of schema plugins, we remove them from our framework, i.e., we directly train a PI plugin using the source domain data and transfer it to the target KBs without training any schema plugins. According to Table 2, 3, 4, and 5, the performance of KB-Plugin without schema plugins is severely degraded, especially on the test cases that involve schema items unseen in the source KBs. The experimental results illustrate that (1) direct PI transfer is difficult due to the substantial difference between the schemas of source and target KBs; (2) schema plugins of target KBs effectively encode adequate schema information via the triple completion task, and the PI plugin can extract and utilize question-relevant schema information from these schema plugins even though it is never trained with them. In addition, if we adopt the schema plugin of a source KB, e.g., $m _ { s c } ^ { S _ { 0 } }$ for the target KBs, the performance of KB-Plugin also drops heavily, showing the necessity of using matched schema plugin.

<table><tr><td>Question I</td><td>Which airport to fly into Rome?</td></tr><tr><td>Pangu</td><td>Find(Rome) Relate(tourist attractions)  $( \pmb { \cal { X } } )$ </td></tr><tr><td>KB-Plugin w/o schema plugin</td><td>Find(Rome) Relate(country) FilterConcept(sovereign state)  $( \pmb { \cal { X } } )$ </td></tr><tr><td>KB-Plugin</td><td>Find(Rome) Relate(transport terminus) FilterConcept(airport)  $( \checkmark )$  (London, transport terminus, Luton airport), (London, instance of, citytown),</td></tr><tr><td>Relevant Triples</td><td>(Luton airport, instance of, airport)</td></tr><tr><td>Question II</td><td>What role did Paul Mccartney play in the Beatles?</td></tr><tr><td>Pangu</td><td>Find(Paul Mccartney) Relate(instruments played) ()</td></tr><tr><td>KB-Plugin</td><td>Find(Beatles) Relate(member) Find(Paul Mccartney) ReverseRelate(member) And() Relate(role)  $( \checkmark )$ </td></tr><tr><td>Source Domain Data Pair</td><td>What is Jane Lynch&#x27;s role in Glee? Find(Glee) Relate(starring) Find(Jane Lynch) ReverseRelate(starring) And() Relate(character role)</td></tr></table>

Table 6: Two typical questions from the test set of WebQSP that KB-Plugin succeeds while Pangu fails. The incorrect functions and arguments are marked as red, while the correct ones are marked as green.

![](images/17d70d783305546a04321e8ab70ba0e71d0e9817aaf472ec4dfc4f112406be6b.jpg)  
Figure 3: KB-Plugin performance with different numbers of generated source KBs.

To show the rationality of our PI plugin learning method, we evaluate the performance of PI plugins trained with different numbers of generated source KBs on WebQSP, GraphQ, and GrailQA, and present the results in Fig. 3. The PI plugin trained with only one source KB performs poorly, implying that it ignores the schema plugin entirely and directly learns PI over this source KB. Once there emerges a new source KB with a different schema, the performance of the trained PI plugin increases substantially, and there is an apparent trend that the performance will increase with more generated source KBs. These results prove that training the PI plugin over multiple source KBs succeeds in forcing the PI plugin to learn to extract and utilize schema information from different schema plugins, and the learned skill can be transferred to target KBs.

## 5.6 Case Study

To better showcase the advantages of KB-Plugin over in-context learning PI methods, we present a case comparison between KB-Plugin and Pangu in Table 6. Question I shows the effect of schema plugin learning and utilization. Both Pangu and KB-Plugin without schema plugin struggle to predict the correct relation “transport terminus” because it is unseen in the demo examples or source KBs. The complete KB-Plugin, however, effectively encodes the information that “transport terminus” is a possible relation between “citytown” and “airport” into the schema plugin via completing relevant triples, and succeeds in predicting this relation by utilizing above information. Question II demonstrates the benefits of harnessing abundant program annotations from the source domain, where Pangu produces a program with incorrect function composition because none of its demo examples has a similar compositional structure, while KB-Plugin induces the correct program by utilizing prior knowledge learned from the source domain. Further analysis can be found in Appendix E and F.

## 6 Conclusion

We propose KB-Plugin, a plug-and-play framework that enables LLMs to induce programs over any low-resourced KBs by learning two types of pluggable modules: KB-specific schema plugin and KB-transferable PI plugin. KB-Plugin achieves better or comparable performance on five heterogeneous KBQA datasets with much smaller backbone LLMs compared to SoTA PI methods for lowresourced KBs, demonstrating its effectiveness for both large-scale and domain-specific KBs. Ablation study and case study also prove the rationality and further showcase the advantage of KB-plugin.

## 7 Limitations

We discuss several limitations of KB-Plugin in this section: (1) In the experiments, we only adopt Llama2-7B as our backbone model due to limited computing resources. Actually, KB-Plugin is model-agnostic and can also be applied to more language models with various sizes and architectures. (2) KB-Plugin requires that the source domain dataset covers questions with diverse various compositional structures, and performs relatively poorly for questions whose compositional structures are unseen in the source domain dataset though they are rare (see Appendix E for details). Future research can focus on improving the transferability of KB-Plugin across compositional structures. In practice, we can also continue to train the PI plugin using some self-training methods such as EGST (Li et al., 2023c) to adapt to these questions. (3) In this work, since both training and evaluation of KB-Plugin require annotated KBQA datasets, we can only take a single dataset KQA Pro as the source dataset and take other datasets as the target datasets, which may limit the upper bounds of KB-Plugin. In the realistic scenario where we need to apply KB-Plugin for a new KB, we can take all these KBQA datasets as the source domain datasets so that the trained source schema plugins would be more diverse and the trained PI plugin would also have stronger transferability and generalizability.

## 8 Ethical Considerations

Though our framework (as well as other PI methods) can effectively reduce the probability of LLMs generating inaccurate answers when faced with questions involving uncommon knowledge, it may still make mistakes if the induced programs are incorrect. In addition, there is a risk of being hacked through targeted means such as injecting harmful or nonfactual knowledge into the KBs. Hence additional care and protective measures should be taken if our framework is deployed in user-facing applications.

All the datasets and encyclopedias used in this work are publicly published with permissible licenses.

## 9 Acknowledgements

This work is supported by Beijing Natural Science Foundation (L243006), Tsinghua University Initiative Scientific Research Program, grants from the Institute for Guo Qiang, Tsinghua University (2019GQB0003) and Zhipu AI.

## References

Kurt D. Bollacker, Colin Evans, Praveen K. Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings ofthe ACM SIG-MOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008, pages 1247–1250. ACM.

Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings ofa meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 2787–2795.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Shulin Cao, Jiaxin Shi, Liangming Pan, Lunyiu Nie, Yutong Xiang, Lei Hou, Juanzi Li, Bin He, and Hanwang Zhang. 2022a. KQA pro: A dataset with explicit compositional programs for complex question answering over knowledge base. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 6101– 6119. Association for Computational Linguistics.

Shulin Cao, Jiaxin Shi, Zijun Yao, Xin Lv, Jifan Yu, Lei Hou, Juanzi Li, Zhiyuan Liu, and Jinghui Xiao. 2022b. Program transfer for answering complex questions over knowledge bases. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8128–8140. Association for Computational Linguistics.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. CoRR, abs/2107.03374.

Yu Gu, Xiang Deng, and Yu Su. 2023. Don’t generate, discriminate: A proposal for grounding language models to real-world environments. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 4928–4949. Association for Computational Linguistics.

Yu Gu, Sue Kase, Michelle Vanni, Brian M. Sadler, Percy Liang, Xifeng Yan, and Yu Su. 2021. Beyond I.I.D.: three levels of generalization for question answering on knowledge bases. In WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021, pages 3477–3488. ACM / IW3C2.

Yu Gu and Yu Su. 2022. Arcaneqa: Dynamic program induction and contextualized encoding for knowledge base question answering. In Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022, pages 1718–1731. International Committee on Computational Linguistics.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spacy: Industrialstrength natural language processing in python.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen. 2023. Structgpt: A general framework for large language model to reason over structured data. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 9237–9251. Association for Computational Linguistics.

Yunshi Lan and Jing Jiang. 2020. Query graph generation for answering multi-hop complex questions from knowledge bases. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 969–974. Association for Computational Linguistics.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 3045– 3059. Association for Computational Linguistics.

Tianle Li, Xueguang Ma, Alex Zhuang, Yu Gu, Yu Su, and Wenhu Chen. 2023a. Few-shot in-context learning for knowledge base question answering. CoRR, abs/2305.01750.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 4582– 4597. Association for Computational Linguistics.

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Lidong Bing, Shafiq R. Joty, and Soujanya Poria. 2023b. Chain of knowledge: A framework for grounding large language models with structured knowledge bases. CoRR, abs/2305.13269.

Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, and Jianyong Wang. 2023c. Flexkbqa: A flexible llm-powered framework for few-shot knowledge base question answering. CoRR, abs/2308.12060.

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA, pages 2181–2187. AAAI Press.

Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting Dong, Meina Song, and Wei Lin. 2023. Chatkbqa: A generate-then-retrieve framework for knowledge base question answering with fine-tuned large language models. CoRR, abs/2310.08975.

Xin Lv, Lei Hou, Juanzi Li, and Zhiyuan Liu. 2018. Differentiating concepts and instances for knowledge graph embedding. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 1971–1979. Association for Computational Linguistics.

Alexander H. Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston.

2016. Key-value memory networks for directly reading documents. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 1400–1409. The Association for Computational Linguistics.

Fedor Moiseev, Zhe Dong, Enrique Alfonseca, and Martin Jaggi. 2022. SKILL: structured knowledge infusion for large language models. In Proceedings of the 2022 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 1581– 1588. Association for Computational Linguistics.

Lunyiu Nie, Shulin Cao, Jiaxin Shi, Jiuding Sun, Qi Tian, Lei Hou, Juanzi Li, and Jidong Zhai. 2022. Graphq IR: unifying the semantic parsing of graph query languages with one intermediate representation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5848–5865. Association for Computational Linguistics.

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2023. Unifying large language models and knowledge graphs: A roadmap. CoRR, abs/2306.08302.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. Adapterfusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 487–503. Association for Computational Linguistics.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. CoRR, abs/2307.16789.

Yujia Qin, Xiaozhi Wang, YuSheng Su, Yankai Lin, Ning Ding, Zhiyuan Liu, Juanzi Li, Lei Hou, Peng Li, Maosong Sun, and Jie Zhou. 2021. Exploring low-dimensional intrinsic task subspace via prompt tuning. CoRR, abs/2110.07867.

Apoorv Saxena, Adrian Kochsiek, and Rainer Gemulla. 2022. Sequence-to-sequence knowledge graph completion and question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 2814– 2828. Association for Computational Linguistics.

Apoorv Saxena, Aditay Tripathi, and Partha P. Talukdar. 2020. Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In Proceedings of the 58th Annual Meeting of

the Associationfor Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 4498–4507. Association for Computational Linguistics.

Jiaxin Shi, Shulin Cao, Lei Hou, Juanzi Li, and Hanwang Zhang. 2021. Transfernet: An effective and transparent framework for multi-hop question answering over relation graph. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 4149–4158. Association for Computational Linguistics.

Yu Su, Huan Sun, Brian M. Sadler, Mudhakar Srivatsa, Izzeddin Gur, Zenghui Yan, and Xifeng Yan. 2016. On generating characteristic-rich question sets for QA evaluation. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 562–572. The Association for Computational Linguistics.

Yusheng Su, Xiaozhi Wang, Yujia Qin, Chi-Min Chan, Yankai Lin, Huadong Wang, Kaiyue Wen, Zhiyuan Liu, Peng Li, Juanzi Li, Lei Hou, Maosong Sun, and Jie Zhou. 2022. On transferability of prompt tuning for natural language processing. In Proceedings of the 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 3949– 3969. Association for Computational Linguistics.

Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. 2019. Pullnet: Open domain question answering with iterative retrieval on knowledge bases and text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 2380–2390. Association for Computational Linguistics.

Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. Arnetminer: extraction and mining of academic social networks. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Las Vegas, Nevada, USA, August 24-27, 2008, pages 990–998. ACM.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Moly-

bog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Commun. ACM, 57(10):78–85.

Xiaozhi Wang, Kaiyue Wen, Zhengyan Zhang, Lei Hou, Zhiyuan Liu, and Juanzi Li. 2022. Finding skill neurons in pre-trained transformer-based language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 11132–11152. Association for Computational Linguistics.

Yuanchun Wang, Jifan Yu, Zijun Yao, Jing Zhang, Yuyang Xie, Shangqing Tu, Huihui Yuan, Jingyao Zhang, Bowen Huang, Yuanyao Li, Juanzi Li, and Jie Tang. 2024. Soay: A service-oriented apis applying framework of large language models.

Chaojun Xiao, Zhengyan Zhang, Xu Han, Chi-Min Chan, Yankai Lin, Zhiyuan Liu, Xiangyang Li, Zhonghua Li, Zhao Cao, and Maosong Sun. 2023. Plug-and-play document modules for pre-trained models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 15713–15729. Association for Computational Linguistics.

Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I. Wang, Victor Zhong, Bailin Wang, Chengzu Li, Connor Boyle, Ansong Ni, Ziyu Yao, Dragomir Radev, Caiming Xiong, Lingpeng Kong, Rui Zhang, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2022. Unifiedskg: Unifying and multi-tasking structured knowledge grounding with text-to-text language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 602–631. Association for Computational Linguistics.

Xuchen Yao. 2015. Lean question answering over freebase from scratch. In NAACL HLT 2015, The 2015 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Denver, Colorado, USA, May 31 - June 5, 2015, pages 66–70. The Association for Computational Linguistics.

Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou, and Caiming Xiong. 2022. RNG-KBQA: generation

augmented iterative ranking for knowledge base question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 6032–6043. Association for Computational Linguistics.

Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. 2016. The value of semantic parse labeling for knowledge base question answering. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 2: Short Papers. The Association for Computer Linguistics.

Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, and Le Song. 2018. Variational reasoning for question answering with knowledge graph. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 6069–6076. AAAI Press.

Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, Huadong Wang, Deming Ye, Chaojun Xiao, Xu Han, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2023. Plug-and-play knowledge injection for pre-trained language models. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 10641–10658. Association for Computational Linguistics.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR, abs/2303.18223.

<table><tr><td>Function</td><td>Input×Args → Output</td><td>Description</td></tr><tr><td>Find</td><td> $E \times \emptyset \to E$ </td><td>find an entity from the KB</td></tr><tr><td>FindAll</td><td> $\emptyset \times \emptyset \to E ^ { \prime }$ </td><td>return all entities in the KB</td></tr><tr><td>Relate</td><td> $( E \cup E ^ { \prime } ) \times R  E ^ { \prime }$ </td><td>a single hop along a relation</td></tr><tr><td>ReverseRelate</td><td> ${ \dot { ( } } E \cup E ^ { \prime } { \dot { ) } } \times R  E ^ { \prime }$ </td><td>a reverse hop along a relation</td></tr><tr><td>FilterConcept</td><td> $E ^ { \prime } \times \overset { \prime } { C }  E ^ { \prime }$ </td><td>return entities in a concept</td></tr><tr><td>And/Or</td><td> $( E ^ { \prime } , E ^ { \prime } ) \times \emptyset  E ^ { \prime }$ </td><td>intersection/union of two sets</td></tr><tr><td>Argmax/Argmin</td><td> $E ^ { \prime } \times R \to E _ { . } ^ { \prime }$ </td><td>superlative aggregations</td></tr><tr><td>LT/LE/GT/GE</td><td> $E \times R { \stackrel { } { \to } } E ^ { \prime }$ </td><td>&lt;1≤1&gt;1≥</td></tr><tr><td>Count</td><td> $E ^ { \prime } \times \emptyset \to N$ </td><td>set cardinality</td></tr></table>

Table 7: KoPL functions used in this work. E: entity; E′: a set of entities; R: relation; C: concept; N: integer.

## A Details of KoPL Functions

We list KoPL functions used in this work in Table 7. We make some modifications to the original (Cao et al., 2022a) for conciseness. Except Find taking topic entities as the argument, other functions either have no arguments or take schema items (i.e., concepts or relations) as their arguments.

## B Triple Sampling Strategy

For WebQSP, GraphQ, and GrailQA, since their KBs are large-scale and relatively sparse, we adopt a popularity-based strategy to sample representative triples for each schema item. Specifically, let the given KB be $\mathcal { K B } = \{ \mathcal { C } , \mathcal { E } , \mathcal { R } , \mathcal { T } \}$ , where $\mathcal { T } = \mathcal { T } _ { e } \cup \mathcal { T } _ { c } \cup \mathcal { T } _ { l }$ . For each $e \in { \mathcal { E } }$ , let cnt(e) be its popularity (i.e., the number of its occurrences in ). When sampling “instance $o f '$ triples for a concept $c \in { \mathcal { C } } .$ , we hope the sampled triples contain representative entities belonging to $^ { c , }$ so we sort all $( e _ { k } , i n s t a n c e o f , c ) \in T _ { e }$ in descending order of cnt $( e _ { k } )$ and select the first K triples. When sampling relational triples for a relation $r \in \mathcal { R } _ { l }$ we take both representativeness and diversity into account. Therefore, we sort all $( e _ { i } , r , e _ { j } ) \in T _ { l }$ in descending order of min $( \mathrm { c n t } ( e _ { i } ) , \mathrm { c n t } ( e _ { j } ) )$ and select the first K triples.

On the other hand, the KBs of KQA Pro, MetaQA, and SoAyBench are dense, so we just randomly sample triples for their schema items.

## C Details of Constrained Decoding

In constrained decoding, after $M _ { P I } ^ { T }$ generates t function chunks $f _ { 1 } ( a r g _ { 1 } ) , \ldots , f _ { t } ( a r g _ { t } )$ , we enumerate all admissible $f _ { t + 1 } ( a r g _ { t + 1 } )$ as the candidate set $P _ { t + 1 }$ following the definition of KoPL functions in Table 7, and constrain $M _ { P I } ^ { T }$ to continue generating one of these candidate or generating the EOS token to end the decoding process.

Specifically, let $E _ { \mathrm { t o p i c } }$ be the set of topic entities in the question obtained using off-theshelf entity linkers $ { \mathrm { ? } } . \mathrm { A t } \ t \ = \ 0 .$ , we enumerate Find(e) for each $e \in \begin{array} { r } { \mathcal { E } _ { \mathrm { t o p i c } } } \end{array}$ as a candidate in $P _ { 1 }$ Specially, around 5% of questions in GraphQ and GrailQA do not have a topic entity (e.g., “Who is the heaviest film director?" from GrailQA, whose target program is FindAll() FilterConcept(director)SelectAmong(weight $k g )$ . For these questions, we follow Pangu (Gu et al., 2023) to start constrained decoding from FindAll()FilterConcept(c), where c is a topic concept provided by Gu and Su (2022).

When $t > 0$ , we execute the current program $p _ { t } ~ = ~ \langle f _ { 1 } ( a r g _ { 1 } ) , \ldots , f _ { t } ( a r g _ { t } ) \rangle$ to get its denotation (i.e., a set of entities) and also the concepts, forward relations, and backward relations that are reachable from the denotation. For each concept $^ { c , }$ we enumerate FilterConcept(c) as a candidate in $P _ { t + 1 }$ . For each forward relation r, we enumerate Relate(r) as a candidate. For each backward relation r, we enumerate ReverseRelate(r) as a candidate, and also include LT(r), LE(r), GT(r), and GE(r) in $P _ { t + 1 }$ if the denotation of $p _ { t }$ is a numerical value such as a quantity or a date. In addition, candidates with superlatives can be enumerated as Argmax(r) and Argmin(r). Also, Count() can always be included to $P _ { t + 1 }$ . If there are multiple topic entities, we enumerate Find(e′) as a candidate to add a new branch, where $e ^ { \prime } \in E _ { \mathrm { t o p i c } }$ is a topic entity not in $p _ { t }$ . When $p _ { t }$ contains multiple branches, we enumerate Or() and And() as candidates to merge the last two branches.

## D Experimental Setup

## D.1 Details of Baselines and Evaluation Metrics

The details of our baselines are as follows:

Pangu (Gu et al., 2023) utilizes potent LLM Codex (Chen et al., 2021) to produce programs in a step-wise fashion via in-context learning. At each step, it first extends existing programs into new valid candidates by enumerating all possible next functions with arguments, then scores each candidate using Codex with several demonstrations and retains the top-k candidates.

KB-BINDER (Li et al., 2023a) first lets Codex generate several "draft" programs for a given question by imitating a few examples, then grounds the arguments in the drafts to the target KB using similarity search to produce hundreds of refined programs. The final answer is decided by the majority vote after executing all these refined programs.

Automatic Program Sampling (APS) (Li et al., 2023c) utilizes gpt-3.5-turbo<sup>2</sup> to translate automatically sampled programs based on a handful of templates into corresponding questions via incontext learning, and subsequently fine-tune a RnG-KBQA (Ye et al., 2022) PI model using the generated question-program pairs.

ProgramTrans (Cao et al., 2022b) is a program transfer method that first uses a seq2seq sketch parser to translate the question into a program sketch, then uses an argument parser to search suitable argument from the KB for each function. We adopt its results without fine-tuning on the target KBs for fair comparison.

DFSDT (Qin et al., 2023) is the SoTA method for general tool using. To solve a question, it employs an LLM to call suitable tool APIs in depth-first order. At each step, the LLM can either (1) call the next API to proceed along a promising path or (2) undo the current call and call another API to expand a new path.

SoAy (Wang et al., 2024) is the SoTA method on SoAyBench. Given a question, it employs LLM to first select the most suitable plan (i.e., API combination) from a candidate pool, then write a Python program with branching and looping structure following the plan to call APIs to get the answer.

Supervised Methods. For WebQSP, GraphQ, GrailQA, and MetaQA, we also provide the fully supervised results of several representative models for comparison, including QGG (Lan and Jiang, 2020), BERT+Ranking (Gu et al., 2021), ArcnaeQA (Gu and Su, 2022), RnG-KBQA (Ye et al., 2022), KV-Mem(Miller et al., 2016), PullNet (Sun et al., 2019), EmbedKGQA (Saxena et al., 2020) and TransferNet Shi et al. (2021).

Evalution Metrics. Following these baselines, we use F1 for WebQSP, GraphQ, and GrailQA, use Hit@1 for MetaQA, and use Accuracy for SoAy-Bench.

## D.2 Implementation Details

We train the schema plugins of the source and target KBs for 3 epochs and 1 epoch, respectively. The batch size and learning rate are set to be 128 and 1e-5, respectively. Besides, we train the PI plugin for 1 epoch with batch size 16 and learning rate 1e-5. For

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Seen</td><td colspan="3">Unseen</td></tr><tr><td>Num</td><td>EM</td><td>F1</td><td>Num</td><td>EM</td><td>F1</td></tr><tr><td>GraphQ</td><td>2148</td><td>71.0</td><td>52.8</td><td>247</td><td>15.4</td><td>20.4</td></tr><tr><td>GrailQA</td><td>6433</td><td>79.9</td><td>67.4</td><td>330</td><td>10.0</td><td>16.4</td></tr></table>

Table 8: Performance of KB-Plugin on test cases whose compositional structures are seen and unseen in the source dataset KQA Pro. EM means the exact match of program sketch.

WebQSP, GraphQ, and GrailQA, we use the same off-the-shelf entity-linker as Pangu to find topic entities; For MetaQA, we follow our baselines to use oracle topic entities; For SoAyBench, we find topic entities using spaCy (Honnibal et al., 2020).

## E Analysis about Question Compositional Structures

For GraphQ and GrailQA, we translate their SPARQL programs to KoPL programs using GraphQ Trans (Nie et al., 2022) and analyze the performance of KB-Plugin on the test cases whose question compositional structures (identified by program sketches) are seen and unseen in the source domain dataset KQA Pro, respectively. From the results in Table 8 we can see that (1) KQA Pro covers most of question compositional structures in the target dataset; (2) KB-Plugin correctly predicts the program sketches for over 70% questions whose compositional structures are seen in the source domain dataset, implying that the mapping from questions to program sketches is largely independent of KB schemas and transferable across KBs, which is consistent with the findings of Cao et al. (2022b) and Li et al. (2023a); (3) KB-Plugin performs poorly on the questions with unseen compositional structures though they are relatively rare, indicating that more advanced transfer techniques across compositional structures remains to be explored.

## F Error Analysis

We analyze 100 incorrect predictions (i.e., F1<1) randomly sampled from the dev set of GrailQA. The major errors are predicting wrong schema items (36%). Specially, when facing several schema items with only subtle differences, e.g., “publisher”(reverse) v.s. “game version published”, KB-plugin tends to prefer to choose the shorter one due to the inherent defects of beam search. Besides, 21% errors are due to a wrong termination check where the model misses the last relation or predicts an additional function. There are also 5% wrong function predictions. Apart from the above errors caused by our model, 27% errors are caused by unidentified or wrongly identified topic entities during entity linking, 9% errors are due to ambiguous or wrong annotations, and the remaining 2% errors are due to the incompletion of KBs.