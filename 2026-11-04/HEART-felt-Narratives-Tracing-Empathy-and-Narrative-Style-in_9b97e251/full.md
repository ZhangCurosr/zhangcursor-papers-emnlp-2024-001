# HEART-felt Narratives: Tracing Empathy and Narrative Style in Personal Stories with LLMs

Jocelyn Shen<sup>1</sup> Joel Mire<sup>2</sup> Hae Won Park<sup>1</sup>

Cynthia Breazeal<sup>1</sup> Maarten Sap<sup>2</sup>

<sup>1</sup>Massachusetts Institute of Technology, Cambridge, MA, USA <sup>2</sup>Carnegie Mellon University, Pittsburgh, PA, USA joceshen@mit.edu, jmire@andrew.cmu.edu, haewon@mit.edu, breazeal@mit.edu, msap2@andrew.cmu.edu

## Abstract

Empathy serves as a cornerstone in enabling prosocial behaviors, and can be evoked through sharing of personal experiences in stories. While empathy is influenced by narrative content, intuitively, people respond to the way a story is told as well, through narrative style. Yet the relationship between empathy and narrative style is not fully understood. In this work, we empirically examine and quantify this relationship between style and empathy using LLMs and large-scale crowdsourcing studies. We introduce a novel, theory-based taxonomy, HEART (Human Empathy and Narrative Taxonomy) that delineates elements of narrative style that can lead to empathy with the narrator of a story. We establish the performance of LLMs in extracting narrative elements from HEART, showing that prompting with our taxonomy leads to reasonable, human-level annotations beyond what prior lexicon-based methods can do. To show empirical use of our taxonomy, we collect a dataset of empathy judgments of stories via a large-scale crowdsourcing study with N = 2, 624 participants.<sup>1</sup> We show that narrative elements extracted via LLMs, in particular, vividness of emotions and plot volume, can elucidate the pathways by which narrative style cultivates empathy towards personal stories. Our work suggests that such models can be used for narrative analyses that lead to human-centered social and behavioral insights.

## 1 Introduction

Empathy, which is a foundational psychological process that drives many prosocial functions, (Zaki, 2019; Morelli et al., 2015), is often delivered through storytelling and sharing of personal experiences (Coplan, 2004; Keen, 2014). Empathetic responses evoked by stories are affected by factors beyond the content of the story alone – delivery, context, and reader characteristics all contribute to the emotional resonance of a narrative. Most studies of narrative empathy and its related constructs focus on reader characteristics,and content of a story (Sharma et al., 2020; Shen et al., 2023). However, intuitively, people also respond to the way a story is told, or the stylistic devices used within a narrative (Figure 1).<sup>2</sup>

![](images/ee2f06ca0bb619902a5fcc88e420f1d161f995faf06af84eee57df4efa656ff2.jpg)  
Figure 1: Narrative empathy can be evoked through the way a story is told (narrative style). This work introduces HEART, a theory-driven taxonomy of narrative elements that contribute to empathy.

A key challenge in narrative analysis within the NLP community is that extracting stylistic features relevant to empathy is not trivial. Prior works use word-count-based (e.g., lexica; Roshanaei et al., 2019; Zhou et al., 2021) or hand-crafted features on extremely limited story sets (Kuzmicová et al.ˇ , 2017; Fernandez-Quintanilla, 2020; Fernandez-Quintanilla and Stradling, 2023; Eekhof et al., 2023; Mangen et al., 2018; Hartung et al., 2016) to quantify narrative elements. However, more complex stylistic narrative devices, such as plot shifts (Nabi and Green, 2015) or vividness of emotions (Pillemer, 1992) are harder to summarize with lexica alone. While a few works have explored using LLMs for more complex narrative analysis tasks (Zhu et al., 2023; Michelmann et al., 2023; Sap et al., 2022), to what extent LLMs can effectively model stylistic devices, and how LLM-extracted features might be leveraged for downstream social insights, remains underexplored.

In this work, we fill this gap by presenting the following contributions. (1) We introduce HEART (Human Empathy and Narrative Taxonomy), a theory-driven taxonomy of narrative style elements that relate to empathy. (2) We use LLMs to quantify aspects of narrativity in our taxonomy and evaluate how well LLMs represent these elements in line with human judgments. For a subset of narrative elements with available lexica, we compare lexical measures with LLM measures, finding that in most cases, GPT-4 and Llama 3 outperform lexica. (3) Through a human study of N = 2, 624 participants, we introduce a new crowdsourced dataset (HEARTfelt Stories Dataset) of empathetic reactions to personal narratives, including annotated narrative style elements, reader characteristics and narrative reactions. (4) With our dataset, we conduct an analysis of pathways through narrative style and reader characteristics leading to empathy, demonstrating the value of HEART in exploring empirical behavioral insights around narrative empathy. In particular, we find that narrative styles with heightened vividness of emotions, character development and action, and plot volume, are tied to narrative empathy. We additionally show that empathy is personalized, with high variability even for the same story, and that beyond narrative style, factors like a reader’s trait empathy and similarity of experiences to the narrator also significantly impact empathy.

## 2 Related Work

Computational linguistic methods can be used to analyze many aspects of narrativity across a large corpus of stories (Sap et al., 2022). Prior works have used lexicon-based approaches to extract psychologically-grounded word categories and relate these to empathy (Roshanaei et al., 2019; Xiao et al., 2016). Zhou et al. (2021) use linguistic style features such as degree of interdependent thinking and integrative complexity (the ability of a person to recognize multiple perspectives and connect them) to predict a viewer’s empathy towards a specific situation. Antoniak et al. (2019) apply narrative analysis techniques to birth stories online, and show patterns of affective and eventbased sequences over time. More recently, Yaden et al. (2024) used linguistic features, such as word phrases and topics, and leveraged LDA to analyze language that separates more empathetic people from more compassionate people, showing that compassionate people use more other-focused language than empathetic people. Other works leverage recent natural language processing (NLP) methods to predict empathy and prosociality from text (Shen et al., 2023; Buechel et al., 2018; Sharma et al., 2020; Bao et al., 2021), but do not explore pathways via which readers feel empathy.

A few works have explored the power of LLMs in characterizing aspects of narrative. In particular, Michelmann et al. (2023) show that LLMs serve as good approximations of human annotators in narrative event segmentation. Other works show that LLMs achieve reasonable performance on character profiling tasks for fictional narratives, particularly in factual consistency and motivation understanding. However, Subbiah et al. (2024) indicate that LLMs fail to perform authentic summarization of stories in line with feedback from writers, apart from successfully drawing on thematic components of the stories. Ultimately, LLMs demonstrate growing potential for narrative understanding tasks (Zhu et al., 2023), but how well they perform, what types of tasks they succeed in, and how they can reveal human behavioral insights, is an active area of research (Agnew et al., 2024).

Our work leverages LLMs to extract narrative style elements that may play a role in narrative empathy through our grounded taxonomy. We evaluate the performance of prompting LLMs to extract such elements against expert human raters. Our empirical study using LLM-extracted narrative elements focuses more on the scientific and behavioral question of how to untangle aspects of narrative style and reader characteristics to understand their contribution towards empathy, rather than improving performance on empathy prediction alone.

## 3 Background

Empathy in the context of narratives has been the subject of many studies in psychology and literary studies. We briefly summarize those below.

Narrative Style and its Role in Empathy. Prior works have theorized how shifts in narrative style impact empathic effect of a story. Keen (2006) proposed a theory of narrative empathy that draws on narrative techniques to enhance empathy, such as flatness or roundness of a character, the character’s mode of consciousness, and vivid use of settings. van Krieken et al. (2017) presented a framework of linguistic cues to measure identification with narrative characters, including character dimensions such as the emotional or perceptual subject of the story. This framework covers both background elements of a story, which can facilitate immersive experiences, and foregrounded elements (such as figurative language), which facilitate aesthetic experiences with the text (Jacobs, 2015).

However, many of these narrative techniques, particularly those that are more abstract in nature, such as plot structure or emotional shifts (Nabi and Green, 2015), have yet to be tested empirically. Researchers in narratology have explored the impact of literary quality on reader empathy, varying aspects such as foregrounding, point of view/viewpoint words, emotion and discourse presentation, and characterisation techniques, but have found mixed results in small-scale studies (Kuzmicová et al.ˇ , 2017; Fernandez-Quintanilla, 2020; Fernandez-Quintanilla and Stradling, 2023; Eekhof et al., 2023; Mangen et al., 2018; Hartung et al., 2016). Other studies have looked at how aspects of literary reading contribute to transportation, or the ability to absorb in a narrative, which further predicts empathy towards a story (Walkington et al., 2020; van Laer et al., 2014, 2019). Koopman (2015) conducted a larger-scale study to investigate the role of genre, personal factors, and affective responses on both empathic understanding and pro-social behavior, finding that genre affected prosocial behaviors. However, narrative style encompasses many aspects beyond genre alone, and each of these elements couples with one another to enhance or diminish narrative empathy.

Reader Characteristics and Narrative Empathy. While narrative style can have an effect on empathy, other factors such as the reader’s characteristics or experiences during reading can affect empathy as well. For example, psychology, economics, and neuroscience have suggested that gender has a significant influence on people’s cognitive empathy, with women exhibiting higher cognitive empathy than men across a variety of age groups (Christov-Moore et al., 2014; Michalska et al., 2013; O’brien et al., 2013). Levels of narrative empathy can also be modulated by one’s trait empathy level (Konrath et al., 2018), emotional state during reading (Roshanaei et al., 2019), or general exposure to literature (Mar et al., 2006). Untangling the effects of these fixed can be challenging, and has been attempted by a few prior works, but with varied results (Koopman and Hakemulder, 2015; Fernandez-Quintanilla, 2020; Roshanaei et al., 2019).

In our work, we propose a taxonomy of narrative empathy based on theories and empirical results presented in the aforementioned works, then scientifically explore what pathways through both narrative style and reader characteristics and life experiences and to overall empathy towards a story. In contrast to prior works, which often vary a single element of narrative style, we construct a thorough taxonomy of narrative elements related to empathy.

## 4 HEART Taxonomy for Empathy and Narrative Style

Based on the aforementioned theoretical and empirical research, we propose HEART, a taxonomy of narrative style elements that can lead to empathy. In A Theory of Narrative Empathy, Keen posits that aspects of characterization, narrative situation, internal perspective, and techniques to represent character consciousness can contribute to narrative empathy. We use these concepts as precursors for developing HEART. Our theoretical model serves as a starting point for understanding what aspects of narrative characteristics might lead to empathy and how we can measure these factors using computational approaches.

Figure 2 shows our full taxonomy, which delineates narrative style as it relates to narrative empathy via four main categories: (1) Character identification (2) Plot (3) Point of view and (4) Setting. In the remainder of this section, we outline each element of our taxonomy and the theoretical and empirical roots of how each element may contribute to narrative empathy.

Character Identification We refer to character identification elements as story aspects that draw readers into the narrator’s perspective, whether this be across internal dimensions (emotion/cognition) or external dimensions (perception/time). We define 6 high-level elements of our taxonomy that can contribute to identification with a character in a story, primarily rooted in (van Krieken et al., 2017)’s work on character identification:

1. Flatness/roundness (Keen, 2006) of the character, including depth of the character expressed through character development over the course of the story or character vulnerability.

![](images/15140467c6d92e395c94fb841c8ea231e389f9a774f7f9199f6fc3ea90a0231a.jpg)  
Figure 2: Narrative Empathy and Style Taxonomy delineating aspects of narrative style that theoretically relate to empathy towards a narrative.

2. Emotional subject (van Krieken et al., 2017; Roshanaei et al., 2019; Pillemer, 1992), refers to the way emotions are expressed both in tone and vividness of emotions.

3. Cognitive subject (Schweitzer and Waytz, 2021; van Krieken et al., 2017), captures expressions of cognition such as thinking, planning, and decision making.

4. Moral subject (van Krieken et al., 2017; Saldias and Roy, 2020) primarily refers to how evaluations or expressions of the narrator’s opinion are conveyed through the story.

5. Action subject (van Krieken et al., 2017), refers to expressions of character action.

6. Subject perception (van Krieken et al., 2017) captures the vividness of perception and bodily sensations experienced by the character.

7. Temporal references (Pillemer, 1992) contain expressed nostalgia (looking to the past) or forecasting and anticipation (looking to the future).

Plot Defining plot has been a key task in narrative analysis (Toubia et al., 2021; Reagan et al., 2016), and can foster empathy through enhancing the narrator’s story via shifts at critical junctures. We delineate 3 aspects of plot that relate to narrative empathy:

1. Plot volume (Keen, 2014; van Laer et al., 2014, 2019) captures the frequency and significance of events in a story.

2. Emotion shifts (Nabi and Green, 2015) indicate fluctuations in the overall emotional trajectory of the story (such as from low to high valence and vice versa).

3. Resolution (Mcadams, 2006) captures the release of tension after the main conflict that a character experiences.

Point of view Prior works suggest that point of view can affect empathy towards a narrator (Eekhof et al., 2023; Fernandez-Quintanilla, 2020; Spitale et al., 2022). For example, first-person perspective can emphasize the personal nature of the story and draw readers into the shoes of the narrator.

Setting Finally, the environment and context of the narrator can facilitate narrative empathy (Pillemer, 1992; van Krieken et al., 2017), for example through world-building to enhance narrative transportation. We capture this element via the vividness of the setting description in a narrative.

## 5 HEART-felt Stories Dataset Annotation

With our theory-grounded taxonomy, we next evaluate how well LLMs can approximate narrative style elements. In order to do so, we annotate the HEART-felt Stories Dataset, a corpus of personal narratives with expert ratings on a subset of stories.

## 5.1 Story Dataset

To empirically observe the narrative elements of HEART, we started with a seed dataset of personal narratives from the EMPATHICSTORIES (Shen et al., 2023) and the EMPATHICSTORIES++ (Shen et al., 2024) dataset, which were specifically designed to include meaningful and vulnerable personal stories with diverse narrators, shared across diverse topics (e.g. relationships, mental health, career and school, etc.). The EMPATHICSTORIES dataset consists of 1,500 personal narratives collected from social media sites (Facebook, Reddit), crowdsourced personal narratives, and transcribed podcasts. The EMPATHICSTORIES++ dataset contains 500 conversational personal stories that were automatically transcribed from storytelling interactions with an AI. We filtered stories to remove potentially harmful topics (e.g. mentions of sexual assault, excessive swearing), and filtered stories that were under 200 words (which might not contain rich narrative style elements), resulting in a final dataset of 874 personal stories.

<table><tr><td>Feature</td><td>KA</td><td>PPA</td><td>ρ</td></tr><tr><td>Optimistic tone</td><td>49.27</td><td>81.50</td><td> $\overline { { 7 2 . 2 1 ^ { * * * } } }$ </td></tr><tr><td>Vivid setting</td><td>48.48</td><td>76.00</td><td> $6 4 . 2 3 ^ { \ast \ast \ast }$ </td></tr><tr><td>Plot volume</td><td>45.97</td><td>83.50</td><td> $6 0 . 3 2 ^ { * * * }$ </td></tr><tr><td>Resolution</td><td>44.29</td><td>79.00</td><td> $5 8 . 9 7 ^ { * * * }$ </td></tr><tr><td>Character vulnerability</td><td>38.17</td><td>75.00</td><td> $5 0 . 0 6 ^ { * * * }$ </td></tr><tr><td>Character development</td><td>28.55</td><td>72.50</td><td> $4 5 . 2 4 ^ { \ast \ast }$ </td></tr><tr><td>Cognition</td><td>27.56</td><td>70.00</td><td> $3 9 . 1 8 ^ { * * }$ </td></tr><tr><td>Evaluations</td><td>26.29</td><td>74.00</td><td> $3 1 . 3 ^ { * }$ </td></tr><tr><td>Emotion shifts</td><td>23.49</td><td>74.50</td><td> $4 6 . 3 4 ^ { * * }$ </td></tr><tr><td>Vivid emotions</td><td>21.17</td><td>66.00</td><td> $3 1 . 8 ^ { * }$ </td></tr><tr><td>Temporal references</td><td>18.29</td><td>77.00</td><td> $2 7 . 9 6 ^ { * }$ </td></tr><tr><td>Bodily sensations</td><td>3.79</td><td>60.33</td><td> $3 4 . 2 5 ^ { \ast }$ </td></tr></table>

Table 1: Agreement between 2 expert human annotators on the narrative elements of our taxonomy. Scores are multiplied by 100 and rounded for readability and sorted by KA. Spearman’s correlation $\rho$ indicates significance.

## 5.2 Expert Narrative Style Annotation

We randomly sampled 50 stories from our final dataset of 874 stories to obtain expert annotations of the narrative elements and validate LLM performance on the task. We selected a subset of 12 narrative elements from our taxonomy that are nontrivial to extract from existing NLP toolkits, and which required human judgments given the subjectivity of the task. Three independent members of our research team with expertise in text analysis and annotation iteratively designed a codebook (Appendix C) with instructions and examples for gauging the presence of each element.

Subsequently, two independent expert annotators rated the presence of each of the 12 narrative elements in the 50 sampled stories. Table 1 shows the agreement between the 2 raters using Krippendorf’s alpha (KA), percent pairwise agreement (PPA), and Spearman’s correlation (ρ). All ratings are positively correlated to each other, but different narrative elements have varying degrees of agreement. We observe the lowest agreement between human annotators for TEMPORAL REFER-

ENCES and BODILY SENSATIONS, where irrealis events and mentions of body sensations across multiple characters caused confusion. Moreover, while some human agreements may appear low using the KA metric, these scores are consistent with prior NLP tasks with more subjectivity (Shen et al., 2023; Rashkin et al., 2018; Sap et al., 2017). In our subsequent empirical analysis, we do not use features with low agreement (below 0.2 KA).

## 6 LLMs for Narrative Style Extraction

Our work explores how LLM-extracted narrative features can be used to yield empirical social insights around empathy and storytelling. As such, we validate whether LLMs are capable of narrative style annotations in line with expert human judgments. To this end, we prompt $\mathrm { G P T } { \cdot } 4 ^ { 3 }$ and the instruction-tuned variant of Llama $3 8 { \mathrm { B } } ^ { 4 }$ with the same instructions and codebook given to human annotators (Appendix C). In Table 2, we report agreement between averaged human ratings and the LLM-based ratings on the same 50 sampled stories.

We observe similar patterns in agreement between GPT-4 and human raters as we do in agreement between our two expert annotators. GPT-4 provides ratings with substantial agreement for narrative features such as CHARACTER VULNERABIL-ITY, OPTIMISTIC TONE, and RESOLUTION. For most features, the GPT-4 ratings are more positively correlated with human annotations than are the Llama 3 ratings. As such, we use GPT-4 to extract the narrative elements for all the remaining stories in our corpus and exclude features that have low agreement with human gold labels in our subsequent empirical study.

## 6.1 Performance of LLMs vs. Lexica

As prior works use lexica (Roshanaei et al., 2019; Zhou et al., 2021) to quantify narrative elements, we compare whether GPT-4 and Llama 3 can outperform psychologically validated lexica in capturing features of HEART. We select 4 dimensions in our taxonomy that readily map to lexicon-based dimensions in LIWC-22 (Boyd, 2022; Pennebaker et al., 1999) and compare correlation to human expert ratings in Table 3. We find that GPT-4- extracted features for OPTIMISTIC TONE, VIVID EMOTIONS, and CHARACTER VULNERABILITY are better aligned with human ratings than LIWC correspondents, although only CHARACTER VUL-NERABILITY is statistically significantly higher $( p \ < 0 . 0 0 1$ as measured by Fisher’s exact test). However, LIWC outperforms GPT-4 in the COG-NITION category, although not statistically significantly so. We discuss the source of potential errors in using GPT-4 to extract COGNITION level of narratives in our error analyses below. Notably, although Llama 3 annotations are generally relatively less correlated with human annotations, the Llama 3 extracted features consistently outperform the LIWC correspondents.

<table><tr><td rowspan="2">Feature</td><td colspan="3">GPT-4</td><td colspan="3">Llama 3 8B Instruct</td></tr><tr><td>KA</td><td>PPA</td><td> $\rho$ </td><td>KA</td><td>PPA</td><td>ρ</td></tr><tr><td>Character vulnerability</td><td>62.89</td><td>86.50</td><td> $8 0 . 1 5 ^ { * * * }$ </td><td>27.08</td><td>79.00</td><td> $7 0 . 5 5 ^ { * * * * }$ </td></tr><tr><td>Optimistic tone</td><td>50.97</td><td>82.25</td><td> $6 8 . 0 6 ^ { * * * }$ </td><td>48.41</td><td>82.75</td><td> $6 7 . 1 4 ^ { * * * }$ </td></tr><tr><td>Resolution</td><td>44.55</td><td>80.00</td><td> $6 1 . 5 9 ^ { * * * }$ </td><td>7.26</td><td>71.83</td><td> $3 4 . 9 3 ^ { * }$ </td></tr><tr><td>Character development</td><td>44.09</td><td>79.25</td><td> $6 1 . 6 4 ^ { * * * }$ </td><td>20.99</td><td>77.25</td><td> $4 6 . 5 1 ^ { \ast \ast }$ </td></tr><tr><td>Vivid setting</td><td>42.12</td><td>78.00</td><td> $6 7 . 3 1 ^ { \ast \ast \ast }$ </td><td>-31.07</td><td>57.03</td><td> $4 1 . 8 4 ^ { * * }$ </td></tr><tr><td>Plot volume</td><td>33.00</td><td>79.25</td><td> $4 4 . 8 8 ^ { * * }$ </td><td>-4.00</td><td>76.08</td><td>27.51</td></tr><tr><td>Emotion shifts</td><td>32.25</td><td>82.25</td><td> $4 5 . 5 ^ { * * }$ </td><td>25.13</td><td>80.58</td><td> $5 2 . 1 3 ^ { * * * }$ </td></tr><tr><td>Vivid emotions</td><td>27.25</td><td>75.00</td><td> $5 9 . 2 1 ^ { \ast \ast \ast }$ </td><td>25.80</td><td>76.00</td><td> $4 2 . 1 3 ^ { * * }$ </td></tr><tr><td>Cognition</td><td>19.83</td><td>73.00</td><td> $3 4 . 9 1 ^ { * }$ </td><td>24.98</td><td>76.00</td><td> $5 2 . 8 9 ^ { * * * }$ </td></tr><tr><td>Evaluations</td><td>-9.76</td><td>75.00</td><td> $2 2 . 6 9$ </td><td>-27.16</td><td>73.00</td><td> $N a N$ </td></tr></table>

Table 2: Agreement between aggregated human annotators (gold ratings) and GPT-4 and Llama 3 8B Instruct ratings of narrative elements in our taxonomy. Rows are sorted by GPT-4 KA.
<table><tr><td>Feature</td><td> $\rho _ { L I W C }$ </td><td> $\underline { { \rho _ { G P T - 4 } } }$ </td><td> $\underline { { \rho _ { L l a m a 3 } } }$ </td></tr><tr><td>Optimistic tone</td><td> $\overline { { 4 7 . 3 5 ^ { * } } } ^ { * * * }$ </td><td> $6 8 . 0 6 ^ { * * * }$ </td><td> $\overline { { 6 7 . 1 4 ^ { * * * } } }$ </td></tr><tr><td>Cognition</td><td> $4 1 . 2 9 ^ { \ast \ast }$ </td><td> $3 4 . 9 1 ^ { * }$ </td><td> $5 2 . 8 9 ^ { * * * }$ </td></tr><tr><td>Vivid emotions</td><td> $3 7 . 6 3 ^ { * * }$ </td><td> $5 9 . 2 1 ^ { \ast \ast \ast }$ </td><td> $4 2 . 1 3 ^ { * * }$ </td></tr><tr><td>Character vulnerability</td><td> $- 6 . 9 5$ </td><td> $8 0 . 1 5 ^ { * * * }$ </td><td> $7 0 . 5 5 ^ { \ast \ast \ast }$ </td></tr></table>

Table 3: Comparison of correlations with human annotations for LIWC, GPT-4, and Llama 3 8B Instruct.

## 6.2 Error Analysis

We observe that GPT-4 consistently over-rates the level of EVALUATIONS and COGNITION expressed in a story as compared to human annotators. Through qualitative examples of stories where GPT-4 and human disagreements are large (Appendix D), GPT-4 typically conflates emotional reactions with evaluations, attributions, or desires (e.g. “...it really got me thinking about when I first went to College...How excited my parents were for me and scared. And I was both excited and scared...”). For COGNITION errors, we see that these systematic errors are typically due to GPT-4 conflating recollection with demonstrations of cognition when overall, the story did not contain more internal thinking processes.

Regarding Llama 3, we observe that when human annotators and GPT-4 agree, but Llama 3 disagrees, it tends to assign higher scores to a minority of features (e.g., CHARACTER VULNERABILITY) while giving lower scores to a majority of features (e.g., VIVID EMOTIONS, VIVID SETTING). The lower ratings for imagery-related features suggest a lesser adeptness with figurative language.

Ultimately, our validation study demonstrates that LLMs – in particular, GPT-4 – can approximate extracting narrative elements relevant to empathy as corroborated by prior work (Shen et al., 2023; Ziems et al., 2024), but some features are more challenging for the model to identify. We show in the following section that GPT-4 narrative ratings still reveal interesting behavioral insights around narrative empathy, even without perfect agreement.

## 7 Human Study for Measuring Empathy

To demonstrate the empirical use of our taxonomy and how extracted narrative elements can be used to explore behavioral insights around narrative empathy, we conduct a large-scale user study presenting stories to different participants and asking them to rate their empathy towards the story. In this section, we discuss our study participants, the task procedure, and our data collection and measures used.

## 7.1 Participants

We recruited N = 2, 624 participants on Prolific<sup>5</sup> to read and rate empathy towards personal stories. An overview of participant demographics is shown in Appendix A. Participants were balanced by sex, predominantly white, and had high trait empathy on average.

## 7.2 Study Procedure

Our study procedure was determined exempt by our institution’s ethics review board. At the beginning of the study, participants rated their current emotional state (arousal/valence), before reading a personal story. After reading the story, they were asked to rate their empathy towards the story, and to check which of the narrative elements within our taxonomy based on which elements contributed most to their emotional reaction towards the story. We asked a qualitative, open-ended question asking what aspects of the narrative’s style made them relate to the story.

After this, we asked participants to answer questions related to (1) narrative-reader interaction effects, which encompass reader factors that are tied to the process of reading the narrative (narrative transportation, prior experience with something that happened in the story, and perceived similarity to the narrator, and (2) reader characteristics (age, gender, ethnicity, trait empathy, how often they read for pleasure, fluent languages, and education level). Survey measurements and reasoning for selecting such measurements are detailed in the following section. All participants were paid \$1 for answering the survey, and participants spent on average 7 minutes completing the entire task. Each of the 874 stories was rated at least 3 times by independent readers, resulting in 2,624 empathetic reactions to stories in total.

## 7.3 Data Collection and Measures

Our user study aims to capture empathy towards a diverse set of narratives with a diverse set of participants with varying reader characteristics in addition to variables that might moderate the effect of narrative style on empathy. Based on related empirical work exploring factors related to empathy (Figure 3), we designed the following surveys (all surveys are included in Appendix E for reproducibility). We make our dataset publicly available to open up deeper research in narrative empathy analysis.

Empathy and Narrative Style Preferences We measure empathy towards the story through the State Empathy Scale (Shen, 2010). To gauge narrative style preferences, participants check off relevant elements from our taxonomy that they felt contributed to empathy towards the story. In addition, we ask for qualitative free-response feedback on what narrative style elements contributed to empathy towards the story.

Narrative-Reader Interaction Effects We define effects at the intersection of reader characteristics and the experience of reading the narrative as narrative-reader interaction effects. These include (1) narrative transportation, measured by the Transportation Scale Short-Form / TS-SF (Appel et al., 2015; Walkington et al., 2020) , (2) prior experience, measured by a Likert scale of how much the reader believes they have been in a similar situation as the narrator, and (3) perceived similarity to the narrator, measured by the Perceived Relational Diversity Scale (Clark, 2002). These features allow us to better understand the pathways via how narrative style elements interplay with narrative-reader interactions to lead to downstream empathy.

![](images/ba8b482788b285460ab3b7c367601102502daa67a1f99f0d0f1576e258cff164.jpg)  
Figure 3: Visualization of how narrative style elements and reader characteristics influence the experience a reader has with a narrative (narrative-reader interaction effects). All of these components combined in turn influence downstream narrative empathy.

Reader Characteristics We collect reader characteristics based on comprehensive literature review of properties that are related to empathy. These features include (1) the emotional state of the reader before reading the story, measured by the arousal/valence scale (Roshanaei et al., 2019), (2) basic demographic information including age, gender, ethnicity (Christov-Moore et al., 2014; Michalska et al., 2013; O’brien et al., 2013), (3) how often participants read for pleasure (Koopman, 2015; Mar et al., 2006), and (4) trait empathy, measured by the Single Item Trait Empathy Scale / SITES (Konrath et al., 2018) and the Toronto Empathy Questionnaire / TEQ (Spreng et al., 2009). Prolific automatically provides additional demographic information on participants such as fluent languages, nationality, and employment and student status.

## 8 Empirical Insights on Narrative Empathy

Next, we demonstrate the efficacy of our taxonomy in exploring empirical questions around empathy with a relevant subset of features from our dataset.

Narrative Style Affects Empathy First, we aggregate empathy ratings for each story by taking the mean across the 3 raters. Then, we split stories into high vs. low presence of each narrative feature and apply Mann-Whitney u-tests to the averaged state empathy for the stories. Figure 5 shows that high aggregated empathy stories have more character development and plot volume. These results are statistically significant, after applying Benjamini-Hochberg correction to account for nine comparisons $( p = 0 . 0 3$ for character development, $p = 0 . 0 3$ for plot trajectory).

![](images/bdf6f926d72059a781f6c6acf3f86bf3b80c9e0fbc468a09558ee0b60400568a.jpg)  
Figure 4: Structural equation modeling of how narrative style elements lead to narrative transportation, combined with effects of the reader sharing a similar experience with the narrator and the reader’s baseline trait empathy.

![](images/c70ad531f8a84b35091839bab945c5ff1243226c0e56fa5ad5857155c33666fc.jpg)  
Figure 5: Comparing average empathy across high vs low presence of each narrative feature, we show that there are significant increases in empathy for stories with more character development and plot volume.

Our work is, to the best of our knowledge, the first to empirically test the effect of character development and plot volume on narrative empathy. While some prior works (van Krieken et al., 2017) propose narrative features that relate to character identification, these are lower level than character development, such as the flatness/roundness or vulnerability of a character. Our findings regarding plot volume are in line with prior works that discuss how salient plot events can mark important moments in narratives that influence the emotional impact of the story (Sap et al., 2022). Prior works primarily from narrative studies use handcrafted features on smaller story sets (Fernandez-Quintanilla, 2020; Eekhof et al., 2023), but do not find significant effects of narrative features such as viewpoint and foregrounding. These studies focus primarily on literary texts rather than narratives that are more common online, and do not take into account other aspects of narrative style and narrative traits that are a part of our theorized taxonomy. These findings suggest future focused works, for example looking at how narrative style relates to empathy across narrative forms (literary vs. personal stories, spoken vs. textual, etc.)

Narrative Empathy is not “One Size Fits All” While our previous analysis captures aggregated empathy, different people can have diverse emotional reactions to the same story. In Figure 6 (Appendix B), we show the standard deviations in state empathy scores for the same story, finding that on average this std. dev. is significantly greater than zero $( p \ < \ 0 . 0 0 1 )$ , indicating that the same narrative can evoke different levels of empathy. To address within-subject variance, we fit mixed effects models of empathy ratings using demographic groups, grouping individuals of similar Age, Sex, Trait Empathy, and Ethnicity and conditioning on multiple ratings for a single story. We find through a likelihood ratio test that empathy predicted by demographic group results in significantly better model fit $( p \ = \ 0 . 0 0 2 )$ These two results indicate that there is high variance in empathy for the same story and that incorporating information regarding diverse demographic profiles can improve empathy model fit, aligning with prior works (August et al., 2020). Our findings have implications in broader empathy prediction tasks within NLP (Buechel et al., 2018; Sharma et al., 2020), which often optimize for a single objective empathy score assigned to a piece of text, aggregating empathy which can overlook individual factors.

Vivid Emotional Expression of Narratives Leads to Narrative Empathy Given our finding that narrative empathy is not “one size fits all,” we conduct analyses taking into account random effects for each story ID with structural equation modeling using the semopy library<sup>6</sup>. Structural equation modeling (SEM) is a standard social science method for structured hypothesis testing and uses a formulation of generalized linear models to account for fixed and random effects when a theoretical model with relationships between elements is proposed.

From our SEM results (Figure 4), we find that vividness of emotions significantly impacts narrative transportation, which in turn influences downstream empathy towards the story. The importance of vividness of emotions in personal stories is supported by other work in psychology. In particular, Pillemer (1992) elaborates that vivid descriptions of emotion in personal stories can convey believability in the experience, more readily evoking empathetic responses. While some computational works explore impact of narrative features on empathy (Roshanaei et al., 2019), they typically focus on positive/negative emotion words, rather than the narrative style or way in which emotions are conveyed through text, and may be better captured by current large-language models.

Figure 4 shows how narrative features contribute to narrative transportation, leading to downstream empathy and taking into account non-stylistic factors like the reader sharing a similar experience as the narrator and the reader’s trait empathy level. We find that both the narrator’s previous experience with something happening in the story as well as their baseline trait empathy are significant predictors of empathy towards the story, but not as much as narrative transportation. In particular, our findings are in line with appraisal theory that suggests that feeling similar emotions is predicated on the target sharing similar experiences (Wondra and Ellsworth, 2015; Yang and Jurgens, 2024). While it is not particularly surprising that similar experience correlates with empathy, very few works have looked at narrative style interactions in tandem with fixed (trait empathy) and more dynamic traits (experiencing something similar), suggesting more holistic consideration of contextual factors related to narrative empathy.

Narrative Style Preferences in Relation to Empathy are Personalized Finally, we show different demographic profiles might prefer different ways of telling a story, where preference is gauged by narrative empathy. Adding the interaction term TRAIT EMPATHY VIVIDNESS OF EMOTIONS to our structural model, we find a significant interaction effect of vivid emotions on the state empathy (est = 0.252, p < 0.001). This indicates that the relationship between vividness of emotions and state empathy increases as trait empathy increases, suggesting that narrative style preferences are personalized across demographic profiles.

While certainly not exhaustive, our empirical analyses show how HEART can be used to yield interesting behavioral insights around how narrative style contributes to empathy. In particular, we note that looking at personalization in narrative empathy, as well as contextualizing reader factors such as their trait empathy level are important for empathy prediction, and are often overlooked in existing empathy tasks.

## 9 Conclusion

In this work, we quantify narrative style as it relates to narrative empathy. We introduce HEART, the first theory-driven taxonomy delineating elements of narrative style that can evoke empathy towards a story. We evaluate the performance of LLMs in extracting narrative elements from HEART, showing that prompting GPT-4 with our taxonomy leads to reasonable, human-level annotations beyond what prior lexicon-based methods can do, but that LLMs struggle in specific tasks, such as GPT-4’s limited ability to extract expressions of cognition and evaluations. Through a crowdsourced study with over 2,000 participants, we demonstrate how HEART can be used to empirically understand the empathic role of narrative style factors. We find that vividness of emotions expressed, character development and plot volume are related to narrative empathy, and contextual factors such as a person’s baseline trait empathy or sharing an experience with the narrator contribute to these effects. Additionally, we show that empathy responses are highly variable even in the same story, and that narrative style preferences are personalized to people with different demographic profiles (such as varying levels of trait empathy). Our findings show the promise of using LLMs for annotating complex story features that can yield interesting social and behavioral insights.

## Limitations

Narrative Style Annotation While most of the features in our taxonomy yielded reasonable consistency across human and LLM annotators, a few elements such as bodily perception and evaluations were less consistent. We excluded these features from our empirical analysis, but future work could make improvements to the annotation process for these specific elements. For example, our codebook makes use of Likert scale ratings for each of the narrative features within an entire story, but more granular annotations such as frequency of occurrences may have more consistency.

Empirical Study Size and Reproducibility Findings in human behavior should be reproducible across different populations and contexts. While we conducted a large scale study with many participants, we did not ask participants to rate multiple stories. Additionally, the demographic distribution of Prolific crowdworkers is predominantly white. Future work should aim to reproduce our empirical insights with diverse populations and different types of stories.

Statistical Modeling Our analysis methods involve interpretable statistical models commonly used in social science research. We chose to use structural equation modeling to gauge behavioral insights around how narrative style contributes to empathy, rather than achieving the best performance on narrative empathy prediction. Future work could improve upon narrative empathy prediction by incorporating narrative features in more complex transformer-based models and ablating different features.

## Ethical Considerations

Personal stories can contain intimate and vulnerable information, in addition to inducing emotions in readers. Our study protocol for showing sensitive stories to crowdworkers was approved by our institution’s ethics review board as an exempt study. Participants gave informed consent that their survey ratings would be collected via Prolific. We ensured that all datasets we used were also collected via IRB-approved protocols, and will only distribute our dataset to IRB-approved protocols.

More broadly, our work aims to advance research in narrative analysis as it relates to realworld human outcomes, such as empathy. Our findings corroborate that empathy is a highly personalized and contextualized experience. As such, in future work, we find that, rather than modeling the average person, it is important to value the rich diversity of human experiences.

We recognize the ethical implications of modeling empathy in stories is double-edged. Empathy can be used in persuasion, marketing, or emotional manipulation. We encourage the findings from our work, and future work on narrative empathy analysis, to focus on improving human empathy for social good. For example, one could develop interactive tools to help a user convey a story more empathetically through understanding the role of narrative devices in reader empathy. Or one could use these insights to understand, at scale, social patterns behind storytelling, and how these might drive empathetic shifts online.

## Acknowledgments

We would like to thank all of our participants and teammates for their invaluable contributions to this project. Special thanks to Sue Holm for narrative annotation and Laura Vianna for study analysis guidance. This work was supported by an NSF GRFP under Grant No. 2141064 and partially funded by NSF grant No. 2230466.

## References

William Agnew, A. Stevie Bergman, Jennifer Chien, Mark Díaz, Seliem El-Sayed, Jaylen Pittman, Shakir Mohamed, and Kevin R. McKee. 2024. The illusion of artificial inclusion. In Proceedings of the CHI Conference on Human Factors in Computing Systems, pages 1–12. ArXiv:2401.08572 [cs].

Maria Antoniak, David Mimno, and Karen Levy. 2019. Narrative Paths and Negotiation of Power in Birth Stories. Proceedings of the ACM on Human-Computer Interaction, 3(CSCW):88:1–88:27.

Markus Appel, Timo Gnambs, Tobias Richter, and Melanie C. Green. 2015. The Transportation Scale–Short Form (TS–SF). Media Psychology, 18(2):243–266.

Tal August, Maarten Sap, Elizabeth Clark, Katharina Reinecke, and Noah A. Smith. 2020. Exploring the Effect of Author and Reader Identity in Online Story Writing: the STORIESINTHEWILD Corpus. In Proceedings of the First Joint Workshop on Narrative Understanding, Storylines, and Events, pages 46–54, Online. Association for Computational Linguistics.

Jiajun Bao, Junjie Wu, Yiming Zhang, Eshwar Chandrasekharan, and David Jurgens. 2021. Conversations Gone Alright: Quantifying and Predicting

Prosocial Outcomes in Online Conversations. In Proceedings ofthe Web Conference 2021, pages 1134– 1145, Ljubljana Slovenia. ACM.

Ryan L Boyd. 2022. The Development and Psychometric Properties of LIWC-22.

Sven Buechel, Anneke Buffone, Barry Slaff, Lyle Ungar, and João Sedoc. 2018. Modeling Empathy and Distress in Reaction to News Stories. arXiv preprint. ArXiv:1808.10399 [cs].

Leonardo Christov-Moore, Elizabeth A Simpson, Gino Coudé, Kristina Grigaityte, Marco Iacoboni, and Pier Francesco Ferrari. 2014. Empathy: Gender effects in brain and behavior. Neuroscience & biobehavioral reviews, 46:604–627.

Mark Andrew Clark. 2002. Perceived relational diversity: A fit conceptualization. Ph.D., Arizona State University, United States – Arizona. ISBN: 9780493434216.

Amy Coplan. 2004. Empathic Engagement with Narrative Fictions. The Journal of Aesthetics and Art Criticism, 62(2):141–152. Publisher: [Wiley, American Society for Aesthetics].

Lynn S. Eekhof, Kobie van Krieken, José Sanders, and Roel M. Willems. 2023. Engagement with narrative characters: the role of social-cognitive abilities and linguistic viewpoint. Discourse Processes, 60(6):411–439. Publisher: Routledge \_eprint: https://doi.org/10.1080/0163853X.2023.2206773.

Carolina Fernandez-Quintanilla. 2020. Textual and reader factors in narrative empathy: An empirical reader response study using focus groups. Language and Literature, 29(2):124–146. Publisher: SAGE Publications Ltd.

Carolina Fernandez-Quintanilla and Fransina Stradling. 2023. Introduction: stylistic approaches to narrative empathy. Journal ofLiterary Semantics, 52(2):103– 121. Publisher: De Gruyter Mouton.

Franziska Hartung, Michael Burke, Peter Hagoort, and Roel M. Willems. 2016. Taking Perspective: Personal Pronouns Affect Experiential Aspects of Literary Reading. PLOS ONE, 11(5):e0154732. Publisher: Public Library of Science.

Arthur M. Jacobs. 2015. Neurocognitive poetics: methods and models for investigating the neuronal and cognitive-affective bases of literature reception. Frontiers in Human Neuroscience, 9. Publisher: Frontiers.

Suzanne Keen. 2006. A Theory of Narrative Empathy. Narrative, 14(3):207–236. Publisher: Ohio State University Press.

Suzanne Keen. 2014. Narrative Empathy. In Narrative Empathy, pages 521–530. De Gruyter.

Sara Konrath, Brian P. Meier, and Brad J. Bushman. 2018. Development and validation of the single item trait empathy scale (SITES). Journal ofResearch in Personality, 73:111–122.

Eva Maria (Emy) Koopman. 2015. Empathic reactions after reading: The role of genre, personal factors and affective responses. Poetics, 50:62–79.

Eva Maria (Emy) Koopman and Frank Hakemulder. 2015. Effects of Literature on Empathy and Self-Reflection: A Theoretical-Empirical Framework. Journal ofLiterary Theory, 9(1):79–111. Publisher: De Gruyter.

Anežka Kuzmicová, Anne Mangen, Hildegunn Støle,ˇ and Anne Charlotte Begnum. 2017. Literature and readers’ empathy: A qualitative text manipulation study. Language and Literature, 26(2):137–152. Publisher: SAGE Publications Ltd.

Anne Mangen, Anne Charlotte Begnum, Anežka Kuzmicová, Kersti Nilsson, Mette Steenberg, andˇ Hildegunn Støle. 2018. Empathy and literary style. Orbis Litterarum, 73(6):471–486.

Raymond A. Mar, Keith Oatley, Jacob Hirsh, Jennifer dela Paz, and Jordan B. Peterson. 2006. Bookworms versus nerds: Exposure to fiction versus non-fiction, divergent associations with social ability, and the simulation of fictional social worlds. Journal ofResearch in Personality, 40(5):694–712.

Dan P. Mcadams. 2006. The Problem of Narrative Coherence. Journal ofConstructivist Psychology. Publisher: Taylor & Francis Group.

Kalina J Michalska, Katherine D Kinzler, and Jean Decety. 2013. Age-related sex differences in explicit measures of empathy do not predict brain responses across childhood and adolescence. Developmental cognitive neuroscience, 3:22–32.

Sebastian Michelmann, Manoj Kumar, Kenneth A. Norman, and Mariya Toneva. 2023. Large language models can segment narrative events similarly to humans. arXiv preprint. ArXiv:2301.10297 [cs, q-bio].

Sylvia A. Morelli, Matthew D. Lieberman, and Jamil Zaki. 2015. The Emerging Study of Positive Empathy. Social and Personality Psychology Compass, 9(2):57–68.

Robin L. Nabi and Melanie C. Green. 2015. The Role of a Narrative’s Emotional Flow in Promoting Persuasive Outcomes. Media Psychology, 18(2):137–162. Publisher: Taylor & Francis Ltd.

Ed O’brien, Sara H Konrath, Daniel Grühn, and Anna Linda Hagen. 2013. Empathic concern and perspective taking: Linear and quadratic effects of age across the adult life span. Journals of Gerontology Series B: Psychological Sciences and Social Sciences, 68(2):168–175.

James Pennebaker, Martha Francis, and Roger Booth. 1999. Linguistic inquiry and word count (LIWC).

David B. Pillemer. 1992. Remembering personal circumstances: A functional analysis. In Affect and accuracy in recall: Studies of"flashbulb" memories, Emory symposia in cognition, 4., pages 236–264. Cambridge University Press, New York, NY, US.

Hannah Rashkin, Antoine Bosselut, Maarten Sap, Kevin Knight, and Yejin Choi. 2018. Modeling Naive Psychology of Characters in Simple Commonsense Stories. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2289–2299, Melbourne, Australia. Association for Computational Linguistics.

Andrew J. Reagan, Lewis Mitchell, Dilan Kiley, Christopher M. Danforth, and Peter Sheridan Dodds. 2016. The emotional arcs of stories are dominated by six basic shapes. EPJ Data Science, 5(1):1–12. Number: 1 Publisher: SpringerOpen.

Mahnaz Roshanaei, Christopher Tran, Sylvia Morelli, Cornelia Caragea, and Elena Zheleva. 2019. Paths to Empathy: Heterogeneous Effects of Reading Personal Stories Online. In 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pages 570–579, Washington, DC, USA. IEEE.

Belen Saldias and Deb Roy. 2020. Exploring aspects of similarity between spoken personal narratives by disentangling them into narrative clause types. arXiv preprint. Number: arXiv:2005.12762 arXiv:2005.12762 [cs].

Maarten Sap, Anna Jafarpour, Yejin Choi, Noah A. Smith, James W. Pennebaker, and Eric Horvitz. 2022. Quantifying the narrative flow of imagined versus autobiographical stories. Proceedings ofthe National Academy ofSciences, 119(45):e2211715119.

Maarten Sap, Marcella Cindy Prasettio, Ari Holtzman, Hannah Rashkin, and Yejin Choi. 2017. Connotation Frames of Power and Agency in Modern Films. page 6.

Shane Schweitzer and Adam Waytz. 2021. Language as a window into mind perception: How mental state language differentiates body and mind, human and nonhuman, and the self from others. Journal ofExperimental Psychology: General, 150(8):1642–1672.

Ashish Sharma, Adam S. Miner, David C. Atkins, and Tim Althoff. 2020. A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support. arXiv preprint. ArXiv:2009.08441 [cs].

Jocelyn Shen, Yubin Kim, Mohit Hulse, Wazeer Zulfikar, Sharifa Alghowinem, Cynthia Breazeal, and Hae Won Park. 2024. Empathicstories++: A multimodal dataset for empathy towards personal experiences. In Findings of the 62nd Annual Meeting of the Association for Computational Linguistics.

Jocelyn Shen, Maarten Sap, Pedro Colon-Hernandez, Hae Park, and Cynthia Breazeal. 2023. Modeling

Empathic Similarity in Personal Narratives. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6237– 6252, Singapore. Association for Computational Linguistics.

Lijiang Shen. 2010. On a Scale of State Empathy During Message Processing. Western Journal of Communication, 74(5):504– 524. Publisher: Routledge \_eprint: https://doi.org/10.1080/10570314.2010.512278.

Micol Spitale, Sarah Okamoto, Mahima Gupta, Hao Xi, and Maja J Mataric. 2022.´ Socially Assistive Robots as Storytellers That Elicit Empathy. ACM Transactions on Human-Robot Interaction, page 3538409.

R. Nathan Spreng, Margaret C. McKinnon, Raymond A. Mar, and Brian Levine. 2009. The Toronto Empathy Questionnaire. Journal ofpersonality assessment, 91(1):62–71.

Melanie Subbiah, Sean Zhang, Lydia B. Chilton, and Kathleen McKeown. 2024. Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers. arXiv preprint. ArXiv:2403.01061 [cs].

Olivier Toubia, Jonah Berger, and Jehoshua Eliashberg. 2021. How quantifying the shape of stories predicts their success. Proceedings of the National Academy ofSciences, 118(26):e2011695118. Publisher: Proceedings of the National Academy of Sciences.

Kobie van Krieken, Hans Hoeken, and José Sanders. 2017. Evoking and Measuring Identification with Narrative Characters – A Linguistic Cues Framework. Frontiers in Psychology, 8. Publisher: Frontiers.

Tom van Laer, Ko de Ruyter, Luca M. Visconti, and Martin Wetzels. 2014. The Extended Transportation-Imagery Model: A Meta-Analysis of the Antecedents and Consequences of Consumers’ Narrative Transportation. Journal ofConsumer Research, 40(5):797– 817.

Tom van Laer, Jennifer Edson Escalas, Stephan Ludwig, and Ellis A van den Hende. 2019. What Happens in Vegas Stays on TripAdvisor? A Theory and Technique to Understand Narrativity in Consumer Reviews. Journal ofConsumer Research, 46(2):267– 285.

Zoë Walkington, Stefanie Ashton Wigman, and David Bowles. 2020. The impact of narratives and transportation on empathic responding. Poetics, 80:101425.

Joshua D. Wondra and Phoebe C. Ellsworth. 2015. An appraisal theory of empathy and other vicarious emotional experiences. 122(3):411–428. 123 citations (Crossref) [2024-09-24].

Bo Xiao, Zac E. Imel, Panayiotis Georgiou, David C. Atkins, and Shrikanth S. Narayanan. 2016. Computational Analysis and Simulation of Empathic Behaviors: A Survey of Empathy Modeling with Behavioral

Signal Processing Framework. Current psychiatry reports, 18(5):49.

David B. Yaden, Salvatore Giorgi, Matthew Jordan, Anneke Buffone, Johannes C. Eichstaedt, H. Andrew Schwartz, Lyle Ungar, and Paul Bloom. 2024. Characterizing empathy and compassion using computational linguistic analysis. Emotion, 24(1):106–115.

Jiamin Yang and David Jurgens. 2024. Modeling empathetic alignment in conversation. Preprint, arxiv:2405.00948 [cs]. 0 citations (Semantic Scholar/arXiv) [2024-09-24].

Jamil Zaki. 2019. The warfor kindness: Building empathy in afractured world. Crown.

Ke Zhou, Luca Maria Aiello, Sanja Scepanovic, Daniele Quercia, and Sara Konrath. 2021. The Language of Situational Empathy. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW1):1–19.

Lixing Zhu, Runcong Zhao, Lin Gui, and Yulan He. 2023. Are NLP Models Good at Tracing Thoughts: An Overview of Narrative Understanding. arXiv preprint. ArXiv:2310.18783 [cs].

Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. 2024. Can Large Language Models Transform Computational Social Science? Computational Linguistics, 50(1):237–291.

## A Participant Demographics

Table 4: Participants’ demographic breakdown.
<table><tr><td rowspan=1 colspan=1>Gender</td><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1>Ethnicity</td><td></td><td rowspan=1 colspan=1>Reading for pleasure</td></tr><tr><td rowspan=1 colspan=1>Female: 1329Male: 1295</td><td rowspan=1 colspan=1>43 ± 14min : 18max : 80</td><td rowspan=1 colspan=1>White : 2234Asian : 150Black : 109Mixed : 86Other : 38NA:8</td><td rowspan=1 colspan=1>4.14 ± 0.88min : 1max : 5</td><td rowspan=1 colspan=1>3.45 ± 1.29min : 1max : 5</td></tr></table>

## B Distribution of Empathy Standard Deviation

## C Codebook and LLM Prompts

## C.1 Character development

We define character development in terms of changes that a character undergoes through the course of narrative events.

We define changes broadly to include cognitive, emotional, behavioral, spiritual, moral, bodily, and social changes.

Notably, we do not consider environmental changes for characters sufficient for character development, but acknowledge that other types of change (e.g. emotional, social) often accompany or are caused by environmental changes.

![](images/ee94cc1b4e65ad8bd5b774b7483f8e92b11dbee3c83c99b8200883d58bbc3312.jpg)  
Figure 6: Distribution of standard deviation in empathy scores for the same story indicate that empathy can differ drastically for the same story.

Rate the narrator’s character development based on the following scale:

• 1 - no change

• 2 - limited change

• 3 - moderate change

• 4 - significant change

• 5 - life-altering, dramatic change

## Examples:

• I watched the birds splashing in the puddle from a bench at the park. They were so playful and content, even as it started to drizzle. (1 - character does not change)

• It wasn’t until my brother told me what he’s been going through that I realized how distant I had been. I broke down in front of him at the time. From that day forward, I decided to be there for my family, no matter what–even if that meant quitting my job and moving home. (5 - multiple dramatic changes)

Respond with a single integer.

Story: [STORY]

## C.2 Character vulnerability

Rate how emotionally vulnerable the narrator is in telling their story. We define vulnerability as how personal or intimate the information shared by the narrator is.

Use the following scale:

• 1 - not vulnerable at all

• 2 - somewhat vulnerable

• 3 - very vulnerable

Examples:

• But I just doubt myself a lot. It’s inevitable. (3 - the author reveals their self doubt)

• I went on a very memorable trip to Crater Lake Oregon on July 8th. (1 - does not share any sensitive information)

Respond with a single integer.

Story: [STORY]

## C.3 Optimistic tone

Rate the level of optimistic/pessimistic tone in the narrator’s story. This should be the tone from the narrator’s perspective, not of other characters in the story.

• -2 - very pessimistic

• -1 - somewhat pessimistic

• 0 - neutral

• 1 - somewhat optimistic

• 2 - very optimistic

Examples:

• I feel alone. It is so frustrating. I used to be fine with it, but then for some reason I actually started wanting to have a friend. I have nobody. And nobody around me seems interesting enough to me. Life gets boring. And frustrating. (rating = -2, very pessimistic)

• He is grown up and I have done my job to get him out into the world. I will miss his teenage years (somewhat), but I am proud of him. (rating = 2, very optimistic)

Respond with a single integer.

Story: [STORY]

## C.4 Vivid emotions

Rate the vividness of emotions described in the story. For example, vividness can be characterized by metaphor, simile, imagery, or strong language.

Use the following scale:

• 1 - not vivid at all

• 2 - somewhat vivid

• 3 - very vivid

## Examples:

• I didn’t feel great about the situation. (1)

• He was a hard-hitter in business, but outside of work he was completely different. (2)

• The pain of losing someone is like being stabbed in the chest. I was devasted when I lost her. (3)

• I was totally exhausted, tears running down my face (3).

Respond with a single integer.

Story: [STORY]

## C.5 Expressions of cognition

Rate how prominent descriptions of cognitive processes are in the story. We define descriptions of cognitive processes as statements that reveal the mental state or thinking pattern of the narrator.

Use the following scale:

• 1 - minimal or no cognitive processes

• 2 - moderate prominence of cognitive processes

• 3 - high prominence of cognitive processes

Examples:

• I was born in the United States. (rating = 1)

• I wondered if I had seen him before. (rating = 2)

• I was thinking about how I could do it but I couldn’t focus because I kept remembering what Sam said to me yesterday. (rating = 3)

Respond with a single integer.

Story: [STORY]

## C.6 Temoral references

Rate the extent to which the character focuses on the past (such as expressing nostalgia or reflections on memories) vs on the future (anticipation, looking forward) in the context of the story.

Note that we are not asking whether the story is a past-tense, present-tense, or future-tense story. We are concerned with the orientation the narrator has toward the past, present, or future. We define ‘extent’ as the amount of narration time oriented toward the relative past, present, or future.

Use the following scale:

• -2 - heavy focus on the past

• -1 - light focus on the past

• 0 - focus on the present

• 1 - light focus on the future

• 2 - heavy focus on the future

Examples:

• “I was stuck in bureaucratic processes for a year, and the whole time I was dreaming of the day my application processing was complete.” (2)

• “I went to the mall and saw a parade on my way.” (0)

• “When I started taking my test, I regretted how little I had studied.” (-1)

Respond with a single integer

Story: [STORY]

## C.7 Plot volume

Stories are structured by a sequence of events.

We define plot trajectory as the amount and significance of events in the story.

If the events are banal or insignificant and do not have a big impact on characters, then the plot trajectory is relatively small. If the events significantly impact characters or setting, then the story has a large plot trajectory.

Rate the degree to which characters and setting are transformed through the course of the story based on the following scale:

• 1 - no change

• 2 - trivial change

• 3 - moderate change

• 4 - significant change

• 5 - life-altering, dramatic change

Examples:

• I stared out the window absent-mindedly for three hours. It was a lovely day. (1)

• I heard a crash outside. I ran outside to see what had happened. It turned out the wind had blown over a box of garden tools. (3)

• After a long and difficult pregnancy, I gave birth to a beautiful baby at 4:15pm. It was a crazy day at the hospital, but thanks to my family and the medical staff, we got through it! (5)

Respond with a single integer.

Story: [STORY]

## C.8 Emotion shifts

Most (but not all) emotions have either a positive (high) or negative (low) valence.

For example, “anger” and “disgust” are low valence, whereas “happy” and “content” are high valence.

Other emotions like “ambivalent” or “surprised” could be neutral, low, high, or ambiguous depending on the context.

We consider 5 different types of emotional shifts that can occur in a story:

• low-to-high valence (e.g. sad to happy)

• high-to-low valence (e.g. happy to sad)

• high-to-high valence (e.g. happy to hopeful)

• low-to-low valence (e.g. sad to angry)

• ambiguous-to-any valence (e.g. bittersweet to excited)

We are interested in relatively straightforward emotion shifts that are either explicitly asserted in the text or easily inferrable based on information in the text. We are less interested in extremely subtle emotional shifts (e.g. joyful to content).

Rate the degree of emotional shifts in the story below.

Use the following scale:

• 1 - no emotional shifts

• 2 - limited or trivial emotional shifts

• 3 - moderate emotional shifts

• 4 - significant emotional shifts

• 5 - life-altering, dramatic emotional shifts

Example Story:

• “I went to college for 1 year before dropping out.” (1)

• “I was surprised to see my friend show up at the cafe where I was working” (2)

• “I was frustrated with Ben for not inviting me, but when I ran into him a few weeks later, our conversation went fine.” (3)

• “I worked hard all semester and was mentally and physically exhausted by the end. It was such a relief to see my grades come in and see that all of my hard work paid off.” (4)

• “I was so excited to get out of class but before the bell rang the principal called me to his office. I was in trouble. I was stressed out of my mind walking to his office, but when I got there, he gave me the good news: I won the school-wide design contest!” (5)

Respond with a single integer.

Story: [STORY]

## C.9 Resolution

In the course of events and interactions between characters, stories introduce conflict. Stories also raise questions about the motives of characters, the meaning of events, and more. Conflict can be explicitly or implicitly referenced by narrators. Alternatively, the reader may subjectively perceive conflict in the situations described by narrators.

Resolution refers to the extent to which conflict is addressed and questions are answered by the end of the story. There are many ways a story may be resolved, partially or completely. Resolution can occur for the narrator, characters within the story, or the reader.

A story with low resolution may not have much conflict or leave conflict unaddressed by the end of the story. A story with high resolution will involve conflict that is addressed or raise questions that are ultimately answered.

Rate the degree of resolution by the end of the story based on the following scale:

• 1 - no resolution

• 2 - limited resolution

• 3 - moderate resolution

• 4 - significant resolution

• 5 - complete resolution

Examples:

• I couldn’t believe that he didn’t apologize. How can someone just pretend that nothing happened? (1)

• I was homeless and finally found a new job, but I hate it and want to find a new one. (3)

• I looked for love my entire life, and had almost given up, when I met them. Now I couldn’t be more in love. (5)

Respond with a single integer. Do not include any words or punctuation marks in your answer.

Story: [STORY]

## C.10 Vividness of setting

Rate the vividness of the setting described in the story. For example, vividness can be characterized by metaphor, simile, imagery, or strong language.

Examples:

• I went to the restaurant to grab a bite to eat. (1)

• The sun cast warm rays onto the concrete in the park. (3)

• The waves in Palos Verdes crashed against the shore, making beautiful ribbons (3)

• There was a house, with music playing in it (2)

• 1 - not vivid at all

• 2 - somewhat vivid

• 3 - very vivid

Story: [STORY]

## D GPT-4 Error Analysis Story Examples

Scores range from 0 to 1, where 0 indicates low presence of narrative feature and 1 indicates high presence.

## D.1 Stories with EVALUATIONS Disagreement

Human: 0.0

GPT-4: 1.0

Yeah, so this is the beginning of the school year, and I’ve seen a lot of people moving into their dorms and apartments, and it really got me thinking about when I first went to College, when

I was moving into the dorm. How excited my parents were for me and scared. And I was both excited and scared, moving away from home and my parents, and knowing that I’d probably get really homesick. Watching all those kids moving in really made me think about how that felt for me. It was really important for me, there was a lot of pressure for me to do well in College because my dad came here from a developing country and wasn’t able to get an education past second grade. I was the first person in my family to go to College, and to make it that far. So I had a lot of emotions, definitely some anxiety, some stress about the pressure of performing and doing well. But also the excitement, and kind of the normal fear that you get doing something you’ve never done before and not having parents who had never experienced College like that before. I didn’t really have anyone to go to, to understand what that meant, like what to expect. So watching those kids just brought me back to that moment.

=================

Human: 0.25   
GPT-4: 1.0

When covid first hit I had to move from my city to my home town. Wasn’t a huge deal as it was for others. It was a year and a half and a lot happened and eventually I came back to my original town, new fiance and cat in tow. I couldn’t find a job once I came back and when I did that place got shut down. My fiance did have one, but any paycheck he had didn’t go to our shared place, mostly his phone bill or groceries once in a while, while he stayed out till 2 am drinking. So I started. Excessively.

I don’t know now how I did, it was only a few months ago, how I managed to. I borrowed from family or friends, I took out loans to make sure rent was paid. It got so bad I was hospitalized for two weeks because for over a day I was throwing up every hour. I had torn my esophagus. No food. My fiance didn’t visit, saying he was scared, but his aunt visited. I cried every night.

When I got home he was working, he came back and was clearly drinking. The next day my mother texted me telling me she was disappointed in me drinking so much I got myself in that situation. I saw that and realized when I was in the hospital, alone, afraid, and not wanting to live like this. Thus me and my mother didn’t speak for months.

Family stuff happened, got back in touch. We both never apologized but there’s a whole lot of cans there to be opened. I ruined relationships I can’t take back. Lost contact with friends.

After this stint my anxiety has been on high alert, making it hard for me to eat or even drink water. Public transit has been scary as I do have those impulse "what if I ran on the track" thoughts which I know are ridiculous.

Thankfully now I do have a job I enjoy, I have my cat, my own place I pay rent.

=================

Human: 0.25   
GPT-4: 1.0

About four months ago, my wife and I sold our first family home. We have a large family. It is my wife and I plus five children. Our oldest daughter started asking us about having her own room. Although we loved that house, we knew it was time to get something bigger.

Luckily, we sold it after being on the market for only three days. We found a house with more bedrooms quickly, and the whole process was as smooth as it could have been.

However, it is bittersweet looking back on everything. That house was very special to me. I did a lot of work on it. I saw my children grow and learn and love there. We made so many good memories. We charted how our children grew on a closet door (which is probably still there). It was a wonderful house while it lasted, but things happen and we had to let it go.

As of today, I can still remember every little nook and cranny of that place. After all, it has only been a few months. However, it is sad to think these memories will eventually fade.

I love our new house, but that first one will always hold a special place in my heart.

=================

Human: 0.0

GPT-4: 1.0

A couple months ago my younger brother got married. I traveled back to my home town of St. Louis, Missouri for the event. I took my girlfriend along with me. It was her first trip to my home town ever. The trip started out great, we picked up our rental car and went to grab a pizza.

The following day was my brother’s wedding ceremony. We thought it had all been planned out thoroughly, but it turns out that nobody had checked the weather report. The day of the wedding came, it started out sunny, a hot day in late

July. Clear blue skies and not a cloud to be seen. We were all so optimistic about the big day. The ceremony was scheduled for 6 PM at sunset, so it would start to cool down and allow for some reprieve from the heat of the day. In theory, that was a great idea.

However, when my girlfriend and I pulled out of the driveway we noticed something new that we hadn’t seen yet on the trip. Storm clouds moving in fast, and lots of them. Dark gray giants rose onto the horizon at a frightening pace. Lightning was visible in the distance as we began our drive to the wedding venue. We hoped and prayed that the storm would blow the other way, and that the outdoor wedding venue would be spared from this particular storm. Would we be able to get away with having the ceremony in decent weather?

It became a race with time. As we drove to the wedding ceremony, it felt as though the clouds were following us and growing larger. As we arrived, I greeted my brother in the parking lot and asked if he thought it would rain. He said maybe, it depends how fast we can get this done. Everyone was present except for the minister, one of the few people who was completely essential to the process.

As the minister arrived, it finally began to rain. It was raining on my brother’s wedding day, I couldn’t believe it, but luckily the ceremony was completed and we had a wonderful sunny reception the next day.

=================

Human: 0.25

GPT-4: 1.0

I was hiking near Lake Ontario with my partner and our two grandkids. It was a beautiful sunny day. The lake sparkled brilliantly. I have a bad knee, so I was struggling with some of the physical activity. My partner suggested that I rest a bit on a fallen log. I was nervous, because I would not be able to get up by myself, but I agreed.

My partner and my granddaugther wanted to hike further to see the bluffs. I said I would be OK for a bit, but my sweet grandson insisted on staying with me. He said, "I won’t let my granny sit in the forest all alone."

Well it was a good thing. My partner and granddaughter didn’t return in a reasonable amount of time. We got very nervous! My grandson is 10 years old, but not strong enough to help me up. He searched for a stout walking stick and found one nearby. I used it to prop myself up, and managed to get my feet underneath me.

Together we went down the shore to find the rest of our party. Luckily all was well, but I would truly have been distraught if I had been all alone waiting for so long!

## D.2 Stories with COGNITION Disagreement

Human: 0.25

GPT-4: 1.0

I’ve been hearing a lot of people saying that MIT students aren’t successful as Stanford or Harvard students because there aren’t as many well-known MIT CEOs. It seems rather unfair of them to say that because MIT students have contributed a lot to this world from nobel-prize winning theorems to groundbreaking algorithms.

Also there are lot of MIT grads like David Siegel who went on to found great companies that don’t necessarily have a face to the brand like Jobs’ Apple or Zuckerberg’s Facebook. And on top of that, there many of MIT grads who go on to be CTOs or other types of product managers (sorry for the emphasis on course 6), and without them, the companies would not be the same.

Above all, out of the "top" institutions, MIT does the most to help lower-income students attain social and economic mobility (I remember reading an article, but can’t find the link).

This is not to say that MIT doesn’t have problems, but at the end of the day, I wish people didn’t equate fame/status with success. You don’t have to be a famous CEO or a CEO in general to be successful. And I’m sure a lot of the people I mentioned didn’t end up becoming crazy famous because they value privacy, which is fine!

And a lot of alumns end up doing what they’re interested in regardless of status, which is amazing and also indicative of success! Success can look different for people.

=================

Human: 0.25

GPT-4: 1.0

When covid first hit I had to move from my city to my home town. Wasn’t a huge deal as it was for others. It was a year and a half and a lot happened and eventually I came back to my original town, new fiance and cat in tow. I couldn’t find a job once I came back and when I did that place got shut down. My fiance did have one, but any paycheck he had didn’t go to our shared place, mostly his phone bill or groceries once in a while, while he stayed out till 2 am drinking. So I started. Excessively.

I don’t know now how I did, it was only a few months ago, how I managed to. I borrowed from family or friends, I took out loans to make sure rent was paid. It got so bad I was hospitalized for two weeks because for over a day I was throwing up every hour. I had torn my esophagus. No food. My fiance didn’t visit, saying he was scared, but his aunt visited. I cried every night.

When I got home he was working, he came back and was clearly drinking. The next day my mother texted me telling me she was disappointed in me drinking so much I got myself in that situation. I saw that and realized when I was in the hospital, alone, afraid, and not wanting to live like this. Thus me and my mother didn’t speak for months.

Family stuff happened, got back in touch. We both never apologized but there’s a whole lot of cans there to be opened. I ruined relationships I can’t take back. Lost contact with friends.

After this stint my anxiety has been on high alert, making it hard for me to eat or even drink water. Public transit has been scary as I do have those impulse "what if I ran on the track" thoughts which I know are ridiculous.

Thankfully now I do have a job I enjoy, I have my cat, my own place I pay rent.

=================

Human: 0.25

GPT-4: 1.0

Today was one of the saddest days of my life. It started early in the day, and my parents came by and picked me up at my house. Everyone was in a very somber mood, but it was sunny and quite warm. We drove out to a church about thirty minutes away near where my mom grew up, and while driving I couldn’t help but think back to all the good memories I had with my cousin. She was always so happy and nice and just fun to be around. But now, that was all gone, and all I had were the memories that were going over in my mind.

Arriving at the church and seeing all of my family, it was hard. It was just so sad, all of it. Seeing my aunt was the hardest part I think, but I knew then that she was strong and was going to be able to get past this. My uncle is an ordained pastor, so he was able to help with the service and I think that helped ease some of the pain.

After the service we all went to the cemetery and gathered up on the hill in the shade. Seeing the final resting place really hit me hard, I started to cry much harder than I had been all day at that point. All of the memories and the final shock to my brain that she was never coming back, made me very sad, and made me miss her dearly.

We then all met at a local place where they served a late lunch and we had some drinks. It was good to see so many of my family, but at the same time, so sad, because I thought that we shouldn’t be seeing each other, at least not for this reason.

I didn’t really know how to feel when we left and I made it back home. I was deeply saddened, and just thought of how my aunt, uncle, and cousins felt. I know that life had changed for them forever, and now life was starting again without their dear one, and that hurt me again. But my family is strong, and stronger together, and I know we will get through this like we will any other tragedy that comes our way.

## E Surveys

## E.1 Empathy and Narrative Style Preferences

State Empathy Scale (Shen, 2010)

Please indicate the level to which you agree with each of the following statements – Strongly disagree (1) to Strongly agree (5)

1. The narrator’s emotions are genuine.

2. I experienced the same emotions as the narrator while reading this story.

3. I was in a similar emotional state as the narrator when reading this story.

4. I can feel the narrator’s emotions.

5. I can see the narrator’s point of view.

6. I recognize the narrator’s situation.

7. I can understand what the narrator was going through in the story.

8. The narrator’s reactions to the situation are understandable.

9. When reading the story, I was fully absorbed.

10. I can relate to what the narrator was going through in the story.

11. I can identify with the situation described in the story.

12. I can identify with the narrator in the story.

## Narrative Style Preferences

Check the aspects of narrative style (the way the story was told) that made you resonate with the story.

1. Flatness/roundness of the character (the character shows development/is vulnerable/subverts expectations)

2. References to the past (nostalgia) or the future

3. The vividness of emotions described in the story

4. The way thoughts/cognition of the character are expressed

5. The way moral judgments of the character are expressed

6. The way the character’s perception and physical sensations are expressed

7. The way the character’s actions are expressed

8. The way the setting of the story is expressed

9. The way the character’s point of view is expressed

10. The overall plot trajectory of the story

11. The presence of a resolution in the story

12. The flow and readibility of the story

13. The overall shifts in the emotional tone of the story

[FREE RESPONSE] What about the narrative style (the way the story was told) made you resonate with it (if any)?

## E.2 Narrative-Reader Interaction

Transportation Scale Short-Form / TS-SF (Appel et al., 2015)

Rate the extent to which you agree with the following statements – Not at all (1) to Very much (7)

1. I could picture myself in the scene of the events described in the narrative.

2. I was mentally involved in the narrative while reading it.

3. I wanted to learn how the narrative ended.

4. The narrative affected me emotionally.

5. While reading the narrative I had a vivid image of the narrator.

## Similar Experience

I have experienced a similar situation as the narrator in my life before – Strongly disagree (1) to Strongly agree (5)

Similar to Narrator (Clark, 2002) I am similar to the narrator – Strongly disagree (1) to Strongly agree (5)

Rate how similar you believe you are to the narrator in terms of the following characteristics – Not similar at all (1) to Highly similar (5)

1. Age

2. Race/ethnicity

3. Sex

4. Religion

5. Sexual orientation

6. Socio-economic status

7. Geographic origin

## E.3 Reader Characteristics

## Education Level

What is the highest level of education you have completed?

1. Some high school or less

2. High school diploma or GED

3. Some college, but no degree

4. Associates or technical degree

5. Bachelor’s degree

6. Graduate or professional degree (MA, MS, PhD, JD, MD, DDS etc.)

7. Prefer not to say

## Reading for pleasure

How often do you read for pleasure?

1. Almost never

2. A couple of times a year

3. A couple of times a month

4. At least once a week

5. Once or more a day

Trait Empathy (Konrath et al., 2018; Spreng et al., 2009)

To what extent does the following statement describe you: "I am an empathetic person" – Strong disagree (1) to Strongly agree (5)

Below is a list of statements. Please read each statement carefully and rate how frequently you feel or act in the manner described – Never (1) to Always (5)

1. When someone else is feeling excited, I tend to get excited too

2. Other people’s misfortunes do not disturb me a great deal

3. It upsets me to see someone being treated disrespectfully

4. I remain unaffected when someone close to me is happy

5. I enjoy making other people feel better

6. I have tender, concerned feelings for people less fortunate than me

7. When a friend starts to talk about his/her problems, I try to steer the conversation towards something else

8. I can tell when others are sad even when they do not say anything

9. I find that I am “in tune” with other people’s moods

10. I do not feel sympathy for people who cause their own serious illnesses

11. I become irritated when someone cries

12. I am not really interested in how other people feel

13. I get a strong urge to help when I see someone who is upset

14. When I see someone being treated unfairly, I do not feel very much pity for them

15. I find it silly for people to cry out of happiness

16. When I see someone being taken advantage of, I feel kind of protective towards him