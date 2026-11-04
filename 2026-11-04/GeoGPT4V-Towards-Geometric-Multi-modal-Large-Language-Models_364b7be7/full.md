# GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation

Shihao Cai<sup>1</sup> \* <sup>‡</sup>, Keqin Bao<sup>1</sup> \*, Hangyu Guo<sup>2</sup>, Jizhi Zhang<sup>1</sup>, Jun Song<sup>2</sup> <sup>†</sup>, Bo Zheng<sup>2</sup>

<sup>1</sup>University of Science and Technology of China, <sup>2</sup>Alibaba Group {caishihao, baokq, cdzhangjizhi}@mail.ustc.edu.cn, hyguo0220@gmail.com, {jsong.sj, bozheng}@alibaba-inc.com

## Abstract

Large language models have seen widespread adoption in math problem-solving. However, in geometry problems that usually require visual aids for better understanding, even the most advanced multi-modal models currently still face challenges in effectively using image information. High-quality data is crucial for enhancing the geometric capabilities of multi-modal models, yet existing open-source datasets and related efforts are either too challenging for direct model learning or suffer from misalignment between text and images. To overcome this issue, we introduce a novel pipeline that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning. We have produced a dataset of 4.9K geometry problems and combined it with 19K open source data to form our GeoGPT4V dataset. Experimental results demonstrate that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks. The code is available at https://github.com/ Lanyu0303/GeoGPT4V\_Project.

## 1 Introduction

With large language models (LLMs) demonstrating formidable performance, their application in solving mathematical problems has become an increasingly popular trend (Toshniwal et al., 2024; Wang et al., 2023b; Gou et al., 2023; Wang et al., 2023a). Prior research has indicated that humans encounter a significant reduction in accuracy when resolving geometric problems devoid of visual aids (Chen et al., 2021). Thus, the integration of visual information from images is imperative for accurately solving of such mathematical problems, necessitating the visual perception capabilities of multimodal large language models (MLLMs). However, even the best batch of MLLMs available now (such as GPT-4V (OpenAI, 2023b), Gemini (Anil et al., 2023)) still lag significantly behind human performance (Wang et al., 2024). Therefore, researchers are eagerly exploring methods to enhance the geometric capabilities of MLLMs.

To enhance the geometric capabilities of MLLMs, an important step is to construct corresponding high-quality data (Gao et al., 2023; Zhou et al., 2023b; Chen et al., 2022). Nevertheless, current data often suffer from two main issues. On the one hand, most open-source datasets are quite challenging, making it difficult for models to directly learn geometric capabilities from them (Bengio et al., 2009; Xu et al., 2020). For instance, the Uni-GEO (Chen et al., 2022) dataset consists of problems extracted from high school textbooks, but the models have not been exposed to the corresponding foundational knowledge. On the other hand, current data augmentation techniques (Gao et al., 2023), using ChatGPT-3.5 to adjust numerical values in the text, fail to harmonize these changes with the corresponding values in images. Consequently, mismatches between the altered text and images can bewilder the model and impede its learning process (Hessel et al., 2021; Yao et al., 2022).

In this paper, we address the aforementioned issues by introducing a straightforward and efficient pipeline for generating geometric problem data. Our objectives are two-fold: (1) to create geometric problems that facilitate the model’s acquisition of basic geometric concepts, and (2) to ensure that the image and the text of the generated geometric problems are well-aligned. In detail, we first employ GPT-4V to create a collection of simplified geometric problems based on open-source datasets. Subsequently, we harness the capabilities of GPT-4 (OpenAI, 2023a) to generate K individual pieces of Wolfram<sup>1</sup> code for each geometric problem previously crafted. The code is then executed to produce K distinct geometric images. Finally, GPT-4V is employed to score these images, allowing us to select the best one that optimally aligns with the associated textual descriptions.

Through the above pipeline, we generate a dataset comprising 4.9K geometric problems characterized by simplicity and image-text matching. We then mix our generated problems with 19K problems from open-source datasets to formulate a dataset with various difficulty levels, named GeoGPT4V. We have conducted comprehensive experiments on the geometry problem subset of MathVista (Lu et al., 2024b) and MathVision (Wang et al., 2024) datasets, two commonly used datasets for multi-modal math. Our experimental results show that models of various sizes and types can achieve significant improvements in geometric capabilities after training with our dataset (achieving 58.2% and 33.8% relative improvement for LLaVA-1.5-7B (Liu et al., 2023b) and ShareGPT4V-7B (Chen et al., 2023a), respectively, on Geometry problem solving (GPS) minitest split of MathVista), which validates the effectiveness of our approach.

In conclusion, the contributions of this paper are summarized as follows:

• We first introduce a novel pipeline capable of automatically generating simple geometric data with aligned image-text pairs.

• We have open-sourced the 4.9K dataset generated by our pipeline, along with the checkpoints of models trained on GeoGPT4V, to facilitate the community’s growth and development.

• Extensive experiments have consistently shown that GeoGPT4V effectively enhances the multimodal geometric capabilities of models of various types and sizes.

## 2 Related Work

In this section, we delve into related studies from two perspectives: multi-modal large language models and mathematical problem solving.

Multi-modal Large Language Models. With the rapid advancement of LLMs, the research community has started to develop multi-modal extensions of these models, known as MLLMs (Bai et al., 2023; OpenAI, 2023b; Liu et al., 2023c). These MLLMs integrate visual information with linguistic data, enhancing their capabilities significantly (Lu et al., 2024a; Li et al., 2023; Ye et al., 2023; Dai et al., 2023). Closed-source models, such as GPT-4V (OpenAI, 2023b), Gemini (Anil et al., 2023), and Qwen-VL-Max (Bai et al., 2023), have demonstrated remarkable proficiency in image comprehension and cognitive tasks. For open-source models, LLaVA (Liu et al., 2023c,b, 2024) utilizes linear projection to bridge the visual encoder and the language model, achieving commendable performance in multi-modal tasks. Building upon the LLaVA architecture, ShareGPT4V (Chen et al., 2023a) employs highquality instructional data to further enhance model capabilities. Moreover, InternVL-Chat (Chen et al., 2023b) upscales its visual encoder to 6 billion parameters. InternLM-XComposer2 (Dong et al., 2024) excels in free-form text-image composition and understanding. Although these MLLMs have shown powerful visual capabilities, MLLMs still confront challenges when it comes to mathematical problem-solving, as highlighted by recent studies (Wang et al., 2024; Lu et al., 2024b; Yue et al., 2023).

Mathematical Problem Solving. The remarkable reasoning capabilities of LLMs have spurred researchers to harness them for solving mathematical problems (Zhou et al., 2023a; Shao et al., 2024; Lightman et al., 2023; Zhao et al., 2023). In the realm of pure text-based mathematical tasks, WizardMath (Luo et al., 2023) enhances model performance by refining instructions through a process of downward and upward instruction evolution. Meta-Math (Yu et al., 2023) approaches the challenge by bootstrapping mathematical questions and rewriting them from various perspectives to improve understanding and problem-solving. However, as previous studies have found, humans’ accuracy significantly decreases when solving geometry problems without images (Chen et al., 2021). Therefore, geometry problems necessitate the visual perception abilities of multi-modal models to fully comprehend and solve them. UniGeo (Chen et al., 2022) addresses this by compiling geometry problems from high school textbooks and introducing a unified multitask geometric transformer framework to tackle calculation and proving problems simultaneously in the form of sequence generation. G-LLaVA (Gao et al., 2023) leverages ChatGPT-3.5 to create geometric question-answer pairs and to rewrite the textual content within questions. Nevertheless, this approach of textual rewriting alone may result in discrepancies between images and text, leading the model to produce incorrect or unrealistic outputs (Liu et al., 2023a). This highlights the ongoing challenge of aligning textual and visual information in multi-modal mathematical problemsolving.

![](images/4b805d8a234486d8d378536ffe6356149954873349a2199e8fec9a58fa276000.jpg)  
Figure 1: Pipeline of our geometric data generation. During the first step, we employ GPT-4V to generate simplified geometric question-answer pairs based on open-source datasets. We highlight the simplified parts compared to the original questions. During the second step, we employ GPT-4 to generate K Wolfram code for each question-answer pair. During the third step, we execute K code to obtain K images. During the fourth step, we employ GPT-4V to score the degree of alignment between the generated images and the questions. We choose the image with the highest score. Finally, we can obtain simplified and image-text matching geometric problems.

## 3 Method

In this section, we will elaborate on the pipeline we have constructed. An overview of our pipeline is depicted in Figure 1. Specifically, our process includes: (1) generating new question-answer pairs (Section §3.1), (2) producing corresponding geometric images (Section §3.2), and (3) scoring and filtering based on the image-text matching degree (Section §3.3).

Formally, the original data from the open-source datasets can be represented as $D \ = \ \{ Q , A , I \}$ where Q represents the question, A represents the answer, and I represents the image.

## 3.1 Question-Answer Pairs Generation

Due to the prevalence of more challenging geometric problems in open-source datasets, to facilitate our model’s learning of basic geometric concepts, we initially simplify these difficult problems to generate easier geometric question-answer (QA) pairs.

In detail, we utilize GPT-4V (OpenAI, 2023b) to generate QA pairs from the dataset $D = \{ Q , A , I \}$ We instruct GPT-4V to craft simplified problems that are derived from the original geometric QA pairs to acquire QA pairs containing fundamental geometric concepts. In detail, we prompt GPT-4V to consider these three perspectives: (1) generating lead-up problems, (2) generating sub-problems, and (3) incorporating the conclusions from the answer into the conditions of the question, which can reduce the complexity of the question. To prevent GPT-4V from generating the same simplified questions, we also ask GPT-4V to generate questions that are as diverse as possible. Additionally, for efficiency, the instruction also asks GPT-4V to generate textual descriptions of images aimed at supporting the subsequent phase of image generation. The detailed prompt can be found in Appendix C.1.

In practice, we generate N (N = 3) new data points based on a single original data point to improve efficiency and reduce API costs. After this phase, the data we obtain can be formally represented as $\tilde { D } _ { 1 } = \{ \tilde { Q } , \tilde { A } , \tilde { D e s } \}$ where Des <sup>˜</sup> represents the image description.

## 3.2 Geometric Images Generation

It is important to highlight that the newly generated QA pairs may not correspond directly to the original images, which could hurt the model’s learning process. To ensure congruity between the textual content and the visual aspects, it is essential to produce new images that align with the generated QA pairs. To address this issue, we employ Wolfram, a powerful software tool capable of executing code to generate geometric images.

In detail, we utilize GPT-4 (OpenAI, 2023a) to generate Wolfram code based on the dataset ${ \tilde { D } } _ { 1 }$ Firstly, we feed the questions, answers, and image descriptions as prompts to GPT-4 to generate Wolfram code. During the generation process, we instruct GPT-4 to explicitly name all variables within the code, with the aim of facilitating a clearer understanding and assisting GPT-4 in recognizing the relationships between code elements and the given questions. The detailed prompt can be found in Appendix C.2. Finally, we execute the Wolfram code, resulting in the generation of new images.

In practice, it is noticed that employing GPT-4 to generate code is unstable. Thus, we generate K $( K = 3 )$ distinct code from the same data to increase the probability of obtaining the correct code. Consequently, we can obtain K distinct images corresponding to K code. It can be represented as $\tilde { D } _ { 2 } = \tilde { \{ Q , A , \tilde { I } ^ { ( 1 ) } , \tilde { I } ^ { ( 2 ) } , . . . , \tilde { I } ^ { ( K ) } \} }$ , where $\tilde { I } ^ { ( i ) }$ represents the i-th image generated for each question.

## 3.3 Scoring and Filtering

After generating K images using Wolfram for each question, we need to select the most suitable one to be used as the final image in our dataset.

Concretely, we employ GPT-4V to assign a score ranging from 0 to 1 that reflects the degree of correspondence between an image generated for the question and the question itself; a higher score signifies a stronger alignment. To augment the scoring proficiency of GPT-4V, drawing inspiration from the Chain-of-Thought (Wei et al., 2022) , we instruct GPT-4V to articulate the rationale underlying its evaluation before determining the ultimate score. The detailed prompt can be found in Appendix C.3.

Finally, for each question associated with K distinct generated images, we obtain K corresponding scores. For each question, we retain the image with the highest score as ${ \tilde { I } } .$ Note that, if this score is less than 0.9, we consider that the image for this question has not been well-generated, and we discard the question. Consequently, we compile a dataset $\tilde { D } = \{ \tilde { Q } , \tilde { A } , \tilde { I } \}$ that consists of questions that are simpler and exhibit a stronger alignment between the images and the associated text.

## 4 Data Analysis

<table><tr><td>Datasets</td><td>Samples</td></tr><tr><td>Open-source Datasets</td><td></td></tr><tr><td>ChartQA UniGEO-Calculation Geometry3K</td><td>7398 3499 2101</td></tr><tr><td>GeoQA+</td><td>6026</td></tr><tr><td>Generated Datasets</td><td></td></tr><tr><td>UniGEO-Proving_Enhanced</td><td>1810</td></tr><tr><td>Geometry3K_Enhanced GeoQA_Enhanced</td><td>1909 1212</td></tr></table>

Table 1: The datasets used in the GeoGPT4V dataset. Column “Samples” is the number of image-text pairs in each dataset. It is worth noting that we only use the training sets of open-source datasets.

In this section, we will present a comprehensive statistical analysis (Section §4.1) and a GPT-4Vbased evaluation (Section §4.2 §4.3) of the datasets generated through our pipeline. Due to space constraints, we also present the results of the human evaluation in Appendix E

## 4.1 Datasets

In this study, to minimize costs, we selected the first 1,500 samples from the training sets of the UniGEO-Proving (Chen et al., 2022), Geometry3K (Lu et al., 2021), and GeoQA (Chen et al., 2021) to create UniGEO-Proving\_Enhanced, Geometry3K\_Enhanced, and GeoQA\_Enhanced for validating the effectiveness of our method. Subsequently, we combine the generated geometric problems with those from open-source datasets, including ChartQA (Masry et al., 2022), UniGEO-Calculation (Chen et al., 2022), the original Geometry3K (Lu et al., 2021), and GeoQA+ (Cao and Xiao, 2022), to form a new dataset with various difficulty levels, dubbed GeoGPT4V. A detailed breakdown of the datasets is provided in Table 1.

## 4.2 Difficulty Evaluation

As mentioned in Section §3, our pipeline will take original data D as input and output generated data $\tilde { D }$ . We aim to generate easier data than the original one to facilitate model learning of basic geometric knowledge. This section demonstrates the efficacy of our pipeline by comparing the difficulty levels of D and $\tilde { D }$

We initiate this by forming a data pair $P _ { 1 } =$ $\{ D , { \tilde { D } } \}$ and utilize GPT-4V to assess the relative difficulty of the data points. To mitigate the bias that GPT-4V may have due to the presentation order, we also consider the pair $P _ { 2 } = \{ \tilde { D } , D \}$ , obtained by swapping the order of the data points. If GPT-4V produces different outputs based on $P _ { 1 }$ and $P _ { 2 } { \mathrm { . } }$ , we conclude that the difficulty of D and $\tilde { D }$ is equal. A detailed prompt can be found in Appendix C.4.

In practice, we randomly sample 500 pairs of generated and corresponding original data points. The outcome, presented in Figure 2a, reveals that over 80% of the questions in the generated dataset are of equal or lesser difficulty compared to the original questions. This indicates that our pipeline is successful in generating data that is simpler than the original dataset.

## 4.3 Image-text Matching Evaluation

As mentioned in the previous section, the alignment between text and images is a critical aspect of geometric problem data. To illustrate that the generated images are better suited for the simplified problems than the original images, we replace the generated images with the original image for each question, resulting in new data $\tilde { D } ^ { \prime } = \{ \tilde { Q } , \tilde { A } , I \}$ Consequently, in this section, we will compare the level of image-text matching in our generated data D<sup>˜</sup> with ${ \tilde { D } } ^ { \prime }$ and the QA data produced by prior methods – G-LLaVA (Gao et al., 2023). Similar to the score function in Section §3.3, we employ GPT4-V to score the degree of alignment between the images and the questions.

In detail, we randomly select 500 data points for each dataset and show the average scores of the three datasets in Figure 2b. The results indicate that our generated data, $\tilde { D } _ { : }$ , exhibits a significantly higher degree of image-text matching than ${ \tilde { D } } ^ { \prime }$ , as well as the dataset enhanced by G-LlaVA (0.9636 for $\tilde { D }$ , 0.7276 for ${ \tilde { D } } ^ { \prime }$ , and 0.6754 for G-LlaVA). Moreover, it is observed that G-LlaVA’s image-text matching score is the lowest, which confirms our hypothesis that simply scaling the size of numbers within problems is an inappropriate approach.

## 5 Experiment

In this section, we conduct experiments to answer the following research questions (RQ):

![](images/eae77fff824341098e64bb4bd435f6c9e1d0fe3c9d802b62b125389adadf3ecd.jpg)  
(a)

![](images/0dc032b8c96c8d8497520ec2e357248d7f95b5be356cd9e38eba889ef3491358.jpg)  
(b)  
Figure 2: The data analysis results. This chart illustrates the simplicity and image-text matching attributes of our dataset. Figure (a) is a comparison chart of the difficulty between the generated and original data. In this figure, “Easier” represents that the generated data is easier than the original data; “Harder” represents that the generated data is harder than the original data; “Equal” represents that the generated and original data have the same difficulty level. Figure (b) shows the average image-text matching scores for the three data types. “Generated Images” represents our generated data. “Original Images” represents the data obtained by replacing generated images in generated data with original images.

• RQ1: Can GeoGPT4V dataset improve geometric capabilities of different models?

• RQ2: Are the generated images better than the original images for model learning?

• RQ3: Is it necessary to score and filter the generated images?

• RQ4: Is the improvement solely due to the original dataset?

## 5.1 Experimental Setup

Benchmarks. We utilize two widely used benchmarks, which encompass numerous multi-model geometric problems, to evaluate the effectiveness of our proposed GeoGPT4V dataset. The detailed information of these benchmarks is as follows:

• MathVista (Lu et al., 2024b) is a mathematical reasoning benchmark in visual contexts. It includes diverse visual contexts, such as natural images, geometry diagrams, charts, etc. Math-Vista includes multiple-choice questions as well as open-ended questions. The MathVista test set comprises 5141 examples without ground truth answers and provides 1000 examples with ground truth answers known as MathVista testmini.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">MathVista</td><td colspan="10">MathVision</td></tr><tr><td>GPS</td><td>GEO</td><td>AVG</td><td>AnaG</td><td>CombG</td><td>DescG</td><td>GrphT</td><td>Angle</td><td>Area</td><td>Len</td><td>SolG</td><td>TransG</td><td>AVG</td></tr><tr><td>LLaVA-1.5</td><td>7B</td><td>20.67*</td><td>20.92*</td><td>20.80*</td><td>7.1</td><td>7.1</td><td>7.7</td><td>10</td><td>15.6</td><td>10.2</td><td>9.8</td><td>5.3</td><td>4.8</td><td>8.62</td></tr><tr><td>LLaVA-1.5</td><td>13B</td><td>24.04*</td><td>23.85*</td><td>23.95*</td><td>14.3</td><td>9.1</td><td>13.5</td><td>5.6</td><td>10.4</td><td>12.6</td><td>14.7</td><td>11.5</td><td>10.7</td><td>11.38</td></tr><tr><td>LLaVA-1.5-G</td><td>7B</td><td>32.69</td><td>32.22</td><td>32.46</td><td>9.52</td><td>16.88</td><td>9.62</td><td>21.11</td><td>19.08</td><td>11.06 17.15</td><td></td><td>9.43</td><td>15.48</td><td>14.37</td></tr><tr><td>LLaVA-1.5-G</td><td>13B</td><td>36.54</td><td>37.24</td><td>36.89</td><td>15.48</td><td>14.29</td><td>12.50</td><td>18.89</td><td>19.65</td><td>13.60</td><td>18.49</td><td>9.02</td><td>11.31</td><td>15.14</td></tr><tr><td>ShareGPT4V</td><td>7B</td><td>|21.63*</td><td>20.50*</td><td>21.07*</td><td>3.6</td><td>10.1</td><td>11.5</td><td>14.4</td><td>16.2</td><td>11.8</td><td>12.3</td><td>9.8</td><td>11.3</td><td>11.22</td></tr><tr><td>ShareGPT4V</td><td>13B</td><td>27.4*</td><td>27.62*</td><td>27.51*</td><td>15.5</td><td>10.7</td><td>11.5</td><td>8.9</td><td>11.6</td><td>13</td><td>17.4</td><td>10.3</td><td>12.5</td><td>12.38</td></tr><tr><td>ShareGPT4V-G ShareGPT4V-G</td><td>7B</td><td>32.69 43.27</td><td>31.80</td><td>32.25 42.77</td><td>11.90 22.62</td><td>12.99</td><td>9.62</td><td>16.67</td><td>17.34</td><td>13.60</td><td>17.59</td><td>10.25</td><td>11.31</td><td>13.47 14.26</td></tr><tr><td></td><td>13B</td><td></td><td>42.26</td><td></td><td></td><td>9.74</td><td>13.46</td><td>11.11</td><td>19.08</td><td>15.80</td><td>13.81</td><td>9.02</td><td>13.69</td><td></td></tr><tr><td>InternVL† InternVL-G†</td><td>40B 40B</td><td>61.1 64.42</td><td>61.1 63.60</td><td>61.10 64.01</td><td>16.67* 16.67</td><td>12.99* 18.18</td><td>15.38* 13.46</td><td>13.33* 16.67</td><td>4.62* 23.12</td><td>5.60* 18.40</td><td>6.46* 18.93</td><td>9.84* 11.89</td><td>10.71* 23.21</td><td>10.62* 17.84</td></tr><tr><td colspan="10">Closed-source Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-VL-Plus</td><td></td><td>38.5</td><td>39.3</td><td>38.90</td><td>17.9</td><td>12.7</td><td>15.4</td><td>8.9</td><td>11.6</td><td>6.4</td><td>10.0</td><td>14.3</td><td>11.31</td><td>12.06</td></tr><tr><td>Qwen-VL-Max</td><td>一</td><td></td><td></td><td></td><td>19.1</td><td>16.9</td><td>16.4</td><td>12.2</td><td>13.3</td><td>14.2</td><td>19.8</td><td>11.5</td><td>17.3</td><td>15.61</td></tr><tr><td>Gemini-1.0-Pro</td><td></td><td>40.4</td><td>41.0</td><td>40.70</td><td>10.7</td><td>20.1</td><td>20.2</td><td>21.1</td><td>19.1</td><td>19.0</td><td>20.0</td><td>14.3</td><td>20.8</td><td>18.37</td></tr><tr><td>Gemini-1.0-Ultra</td><td></td><td>56.2</td><td>55.6</td><td>55.90</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4V</td><td></td><td>50.5</td><td>51.0</td><td>50.75</td><td>32.1</td><td>21.1</td><td>22.1</td><td>14.4</td><td>22.0</td><td>22.2</td><td>20.9</td><td>23.8</td><td>25.6</td><td>22.69</td></tr></table>

Table 2: Overall results of different models on the MathVista and MathVision. We present the detailed scores for all the tasks related to geometry such as “GPS” and “AnaG”, as well as the average score over these tasks in two benchmarks denoted as “AVG”. Due to limited space, we utilize abbreviations for these geometry-related tasks and illustrate the detailed task name in the Appendix A. For the model trained with GeoGPT4V, score increases are marked in red compared to the original model. ∗ indicates our re-implemented test results missed in benchmarks or origin papers. InternVL†represents the abbreviation for InternVL-Chat-V1.2-Plus. The suffix “-G” to the model name indicates a model trained on the GeoGPT4V. For better comparison, we also demonstrate results for five representative closed-source MLLM models.

• MathVision (Wang et al., 2024) is a more challenging multi-modal mathematical benchmark than MathVista. It categorizes all mathematical problems into five difficulty levels and 16 distinct tasks. MathVision also consists of multiplechoice questions and open-ended questions. The MathVision test set comprises 3040 examples with ground truth answers.

Evaluation Method. We strictly follow the evaluation method proposed in MathVista (Lu et al., 2024b) and MathVision (Wang et al., 2024). Firstly, we utilize ChatGPT-3.5 to extract the ultimate response from model outputs in MathVista, while employing regular expressions with MathVision for the same purpose. Consequently, we report the accuracy of the answers as the score for performance evaluation, with a maximum possible score of 100.

Baseline Models. We train the following mainstream open-source models using our proposed GeoGPT4V dataset, with model sizes including 7B, 13B, and 40B.

• LLaVA-1.5 (Liu et al., 2023c,b) utilizes linear layers to connect the vision encoder and the large language model (LLM). In the pre-training stage, LLaVA-1.5 keeps the vision encoder and the LLM frozen, and only trains linear layers. In the fine-tuning stage, it freezes the vision encoder and trains the linear layers and the LLM.

• ShareGPT4V (Chen et al., 2023a) has an architecture similar to LLaVA’s. However, in the pretraining stage of ShareGPT4V, both the vision encoder and the language model remain unfrozen. The training data is high-quality, detailed description data generated by GPT-4V.

• InternVL-Chat-V1.2-Plus (Chen et al., 2023b) utilizes the InternViT (Chen et al., 2023b) as its visual encoder, which has 6 billion parameters. What’s more, it scales LLM to 34B and utilizes a fine-tuning dataset with 12 million samples.

Implementation Details. For data generation, we employ the “gpt-4-vision-preview” and “gpt-4- 1106-preview” API provided by OpenAI for GPT-4V and GPT-4. For model training, all the models are trained on NVIDIA A100 GPUs with PyTorch version 2.0.1. To ensure a fair comparison, we keep the training parameters consistent with those specified by the model’s original authors and train the models for one epoch. Detail training parameters are demonstrated in Appendix B.

<table><tr><td rowspan="2">Model</td><td colspan="3">MathVista</td><td colspan="10">MathVision</td></tr><tr><td>GPS</td><td>GEO</td><td>AVG</td><td>AnaG CombG</td><td></td><td>DescG</td><td>GrphT</td><td>Angle</td><td>Area</td><td>Len</td><td>SolG TransG</td><td>AVG</td></tr><tr><td>LLaVA-1.5-7B</td><td>20.67*</td><td>20.92*</td><td>20.80*</td><td>7.1 7.1</td><td>7.7</td><td></td><td>10</td><td>15.6</td><td>10.2</td><td>9.8</td><td>5.3 4.8</td><td>8.62</td></tr><tr><td>- Image Generation</td><td>30.77</td><td>30.96</td><td>30.87</td><td>8.33 14.94</td><td>8.65</td><td>15.56</td><td>17.34</td><td>12.20</td><td>14.48</td><td>7.79</td><td>19.05</td><td>13.15</td></tr><tr><td>- Image Scoring</td><td>33.65</td><td>31.80</td><td>32.73</td><td>9.52 15.48</td><td>9.62</td><td>20.00</td><td>17.34</td><td>12.20</td><td>15.59</td><td>6.56</td><td>15.48</td><td>13.54</td></tr><tr><td>GeoGPT4V</td><td>32.69</td><td>32.22</td><td>32.46</td><td>9.52 16.88</td><td>9.62</td><td>21.11</td><td>19.08</td><td>11.06</td><td>17.15</td><td>9.43</td><td>15.48</td><td>14.37</td></tr></table>

Table 3: Ablation for image generation and image scoring. “- Image Generation” denotes the exclusion of newly generated geometric images. “- Image Scoring” signifies the random selection of generated images, rather than utilizing GPT4V to score and choose them. For comparison, we also represent the results from the official LLaVA-1.5-7B model in the first line and GeoGPT4V in the last line. Bold results indicate the best results for all models. ∗ indicates our re-implemented test results missed in benchmarks or origin papers.

## 5.2 Main Results (RQ1)

We evaluate the performance of various opensource models on MathVista testmini (short as MathVista) and MathVision test (short as MathVision) benchmarks after training on the GeoGPT4V dataset to demonstrate our proposed method’s effectiveness. For convenience, we append the suffix “-G” to the model name to indicate a model trained on the GeoGPT4V dataset, such as “LLaVA-1.5- $\mathbf { G } ^ { \ast }$ . Since our method focuses on geometric data, we present detailed scores for all the tasks related to geometry and the average score over these tasks in Table 2. The complete set of scores can be found in Appendix D.1 and D.2. In Appendix D.3, we compare the geometric capabilities of our best model, InternVL-Chat-V1.2-Plus-GeoGPT4V, with other open-source and closed-source models.

The experimental results from Table 2 indicate that our dataset can effectively improve different models’ geometric capabilities. First of all, our proposed GeoGPT4V has exhibited an improvement in the average scores across all geometry-related tasks on both MathVista and MathVision benchmarks, indicating that GeoGPT4V can enhance the model’s general geometry performance. Moreover, our proposed GeoGPT4V has brought improvements to most geometry-related tasks in both benchmarks in all scales and types of models. Furthermore, our GeoGPT4V significantly bridges the gap in geometric capabilities between open-source and closed-source models, except InternVL-Chat-V1.2- Plus, which has already employed a substantial amount of customized fine-tuning datasets.

## 5.3 In-depth Analysis

To comprehensively analyze the effectiveness of GeoGPT4V, we design a series of analyzing experiments from various perspectives. Firstly, we design ablation experiments from the standpoint of the efficacy of generating new geometric images and selecting generated images with GPT4V scores. Subsequently, we conduct experiments to demonstrate the substantial performance improvement brought by GeoGPT4V stemming from the generated data rather than the utilization of opensource data. Due to resource and space limitations, we leverage LLaVA-1.5-7B for analytical experiments and conduct evaluations on both MathVista and MathVision.

## 5.3.1 Effect of Generating New Images (RQ2)

We validate the effectiveness of the newly generated geometric images by replacing the images generated in GeoGPT4V with their original counterparts and training the model on them. In detail, we first substitute the newly generated images from GeoGPT4V with the original images while retaining the simplified questions generated, formulating a new dataset denoted as ${ \tilde { D } } ^ { \prime }$ . Subsequently, we train the LLaVA-1.5-7B model on ${ \tilde { D } } ^ { \prime }$ and compare its geometric capabilities with the model trained on GeoGPT4V.

Based on results demonstrated in Table 3, we have following observations: Firstly, the model trained on ${ \tilde { D } } ^ { \prime }$ exhibits inferior performance compared to the model trained on GeoGPT4V, indicating the effectiveness of the newly generated images. Secondly, the model trained on ${ \tilde { D } } ^ { \prime }$ demonstrates stronger performance than the model trained without the use of ${ \tilde { D } } ^ { \prime }$ , thereby validating the efficacy of the easier QA pairs generated by our pipeline.

<table><tr><td>Name</td><td>Base</td><td>Replace</td><td>Mix</td></tr><tr><td rowspan="5">Datasets</td><td>ChartQA</td><td>ChartQA</td><td>ChartQA</td></tr><tr><td>UniGeo-Calculation</td><td>UniGeo-Calculation</td><td>UniGeo-Calculation</td></tr><tr><td>Geometry3K</td><td>Geometry3K_Replace</td><td>Geometry3K_Mix</td></tr><tr><td>GeoQA+</td><td>GeoQA+_Replace</td><td>GeoQA+_Mix</td></tr><tr><td>UniGeo-Proving</td><td>UniGeo-Proving_Replace</td><td>UniGeo-Proving_Mix</td></tr></table>

Table 4: Dataset settings for experiments comparing open-source data and generated data. The suffix “Replace” indicates that we replace the corresponding original data with generated data. The suffix “Mix” indicates that we mix the original data with generated data.
<table><tr><td rowspan="2">Datasets</td><td colspan="3">MathVista</td><td colspan="10">MathVision</td></tr><tr><td>GPS</td><td>GEO</td><td>AVG</td><td>AnaG</td><td>CombG</td><td>DescG</td><td>GrphT</td><td>Angle</td><td>Area</td><td>Len</td><td>SolG</td><td>TransG</td><td>AVG</td></tr><tr><td>Base</td><td>29.33</td><td>28.03</td><td>28.68</td><td>10.71</td><td>15.91</td><td>8.65</td><td>12.22</td><td>16.67</td><td>11.80</td><td>13.59</td><td>8.20</td><td>16.07</td><td>12.65</td></tr><tr><td>Replace</td><td></td><td>33.1732.6432.91</td><td></td><td>7.14</td><td>14.94</td><td>6.73</td><td>20.00</td><td>20.81</td><td>10.80</td><td>15.14</td><td>10.25</td><td>14.29</td><td>13.34</td></tr><tr><td>Mix</td><td></td><td>33.52 34.31 33.92</td><td></td><td>11.90</td><td>15.58</td><td>10.58</td><td>14.44</td><td>17.34</td><td>12.40</td><td>14.48</td><td>9.43</td><td>16.07</td><td>13.58</td></tr></table>

Table 5: Comparison of the effects with and without using the generated datasets. Bold results indicate the best results for all models.

## 5.3.2 Is Scoring Necessary? (RQ3)

As mentioned in Section §3.3, K images are scored, and the one with the highest score is selected from this set. To demonstrate the necessity of scoring, we formulate a new dataset ${ \tilde { D } } ^ { \prime \prime }$ by directly modifying the selection method to randomly choose from the K images while keeping all other aspects unchanged. Consequently, we analyze the performance of the LLaVA-1.5-7B trained on ${ \tilde { D } } ^ { \prime \prime }$

According to results demonstrated in Table 3, we can find that the model trained on ${ \tilde { D } } ^ { \prime \prime }$ exhibits inferior performance on most tasks compared to the model trained on GeoGPT4V. The results indicate that the quality of the images obtained via ranking surpasses those chosen randomly in overall aspects.. It is also worth noting that the model trained on ${ \tilde { D } } ^ { \prime \prime }$ performs better on a few tasks, possibly due to the relative similarity of the generated images in these tasks. While using GPT-4V for selection may introduce bias, random selection has the potential to enhance diversity.

## 5.3.3 Are the Open-source Datasets Enough? (RQ4)

To demonstrate that the performance improvements brought by GeoGPT4V are not solely reliant on open-source data, we compare the performance of models trained using various combinations of opensource and our generated data. In detail, as illustrated in Table 4, we construct three tiers of datasets. Firstly, we combine all open-source datasets to create the “Base” dataset. Subsequently, we replace the original data from the “Base” dataset with the data generated by our pipeline, resulting in the “Replace” dataset. Lastly, we mix the generated data with all the data from the “Base” dataset to form the “Mix” dataset. It is notable that GeoQA is a subset of GeoQA+. Thus we only use GeoQA+ in these three dataset settings, rather than using both GeoQA+ and GeoQA.

We finetune LLaVA-1.5-7B separately on these three datasets and evaluate their performance in Table 5, with observations as follows: Although the “Base” dataset, constructed using open-source data, provides moderate geometric capabilities, our “Replace” and “Mix” datasets exhibit even greater enhancements in geometric performance. This not only demonstrates the effectiveness of the data generated by our pipeline but also indicates that the improvements afforded by GeoGPT4V are not solely derived from open-source data.

## 6 Conclusion

In this study, we propose a novel pipeline to enhance the geometric capabilities of MLLMs. We have proposed data generation methods for multimodal geometric tasks involving problem simplification and the generation of images that match newly generated text. Specifically, we use GPT4V and GPT4 to generate sub-problems or lead-up problems for given geometric tasks, along with the corresponding Wolfram code that can be executed to generate geometric images. Based on the pipeline, we have generated 4.9K simplified and image-text matching geometric problems. We mix our generated data with 19K open-source data to formulate a dataset with various difficulty levels, named GeoGPT4V. After training on the GeoGPT4V dataset, various models have improved geometric scores on both MathVista and MathVision benchmarks. The extensive experimental results demonstrate the effectiveness of the GeoGPT4V dataset. We have open-sourced the GeoGPT4V dataset and the checkpoints of models trained on the GeoGPT4V dataset, with the aim of fostering the community’s growth.

## Limitations

This paper focuses on the generation of geometric images. We employ GPT-4 to generate Wolfram code, which can be executed to generate images. However, this approach is unstable and may result in poor image quality. That’s why we use GPT-4V to score the images, which leads to more API calls and increased costs.

What’s more, this paper only considers simplifying open-source geometric problems. However, generating more complex problems is also worth considering, as it will generate more complex geometric images and help models improve complex reasoning capabilities. Our future work will explore the more accurate generation of complex geometric images.

Finally, multi-modal mathematics is not limited to geometric problems. It also includes tasks such as chart question answering and function question answering. Generating richer charts and function images is also part of our future exploration work.

## Acknowledgement

This work was supported by Alibaba Group through Alibaba Research Intern Program.

## References

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem

Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. 2023. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML 2009, Montreal, Quebec, Canada, June 14-18, 2009, volume 382 of ACM International Conference Proceeding Series, pages 41–48. ACM.

Jie Cao and Jing Xiao. 2022. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022, pages 1511–1520. International Committee on Computational Linguistics.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. 2022. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 3313–3323. Association for Computational Linguistics.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. 2021. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 513–523. Association for Computational Linguistics.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2023a. Sharegpt4v: Improving large multi-modal models with better captions. CoRR, abs/2311.12793.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2023b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. CoRR, abs/2312.14238.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang,

Boyang Li, Pascale Fung, and Steven C. H. Hoi. 2023. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. CoRR, abs/2401.16420.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. 2023. Gllava: Solving geometric problem with multi-modal large language model. CoRR, abs/2312.11370.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving. CoRR, abs/2309.17452.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A referencefree evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 7514–7528. Association for Computational Linguistics.

Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. 2023. Monkey: Image resolution and text label are important things for large multi-modal models. CoRR, abs/2311.06607.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. CoRR, abs/2305.20050.

Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2023a. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v(ision), llava-1.5, and other multi-modality models. CoRR, abs/2310.14566.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023b. Improved baselines with visual instruction tuning. CoRR, abs/2310.03744.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023c. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. 2024a. Deepseek-vl: Towards real-world vision-language understanding. CoRR, abs/2403.05525.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2024b. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR).

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 6774–6786. Association for Computational Linguistics.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. CoRR, abs/2308.09583.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 2263–2279. Association for Computational Linguistics.

OpenAI. 2023a. GPT-4 technical report. CoRR, abs/2303.08774.

OpenAI. 2023b. Gpt-4v(ision) system card. In technical report.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. 2024. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. CoRR, abs/2402.10176.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. 2024. Measuring multimodal mathematical reasoning with math-vision dataset. CoRR, abs/2402.14804.

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023a. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. CoRR, abs/2310.03731.

Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. 2023b. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. CoRR, abs/2312.08935.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan Wang, Hongtao Xie, and Yongdong Zhang. 2020. Curriculum learning for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 6095–6104. Association for Computational Linguistics.

Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. 2022. FILIP: fine-grained interactive language-image pre-training. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. 2023. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. CoRR, abs/2311.04257.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. CoRR, abs/2309.12284.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. CoRR, abs/2311.16502.

James Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Michael Qizhe Xie. 2023. Automatic model selection with large language models for reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 758–783. Association for Computational Linguistics.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023a. Solving challenging math word problems using GPT-4 code interpreter with code-based self-verification. CoRR, abs/2308.07921.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023b. LIMA: less is more for alignment. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

## A Detailed Task Information

Table 6 shows the correspondence between abbreviations and detailed task names.

## B Training Parameters

We keep the same parameters as those specified by the model’s original authors. Detail parameters are shown in Table 7.

## C Prompts

## C.1 Prompt for Question-Answer Pairs Generation

Table 8 shows the prompt for question-answer pairs generation. We prompt GPT-4V to generate simplified geometric problems based on the open-source datasets.

## C.2 Prompt for Wolfram Code Generation

Table 9 shows the prompt for Wolfram code generation. We prompt GPT-4 to generate the Wolfram code based on the information from the question, the answer, and the image description.

## C.3 Prompt for Scoring

Table 10 shows the prompt for scoring. We prompt GPT-4V to score the degree of alignment between the images and the questions.

## C.4 Prompt for Difficulty Comparison

Table 11 shows the prompt for difficulty comparison. We employ GPT-4V to determine which of the two problems is more difficult.

## D Detailed Evaluation Results

## D.1 MathVista Results

We show full MathVista testmini results in Table 12. Although our method focuses on geometric problems, the GeoGPT4V dataset can still improve the overall scores of various models, except InternVL-Chat-V1.2-Plus, which has already employed a customized fine-tuning dataset with 12 million samples.

## D.2 MathVision Results

We show full MathVision test results in Table 13. We can find that the GeoGPT4V dataset can improve the scores of most tasks on MathVision for various models. The results demonstrate the effectiveness of the GeoGPT4V dataset.

## D.3 Comparison with Other Models

We compare the performance of our best model, InternVL-Chat-V1.2-Plus-GeoGPT4V, with other open-source and closed-source models regarding geometric capabilities. Detailed results are in Table 14.

For MathVista, our best model achieves the best geometric scores among all models. For MathVision, our best model achieves the highest scores for average score and most geometric scores among open-source models. The experimental results demonstrate the effectiveness of the GeoGPT4V dataset.

## E Human Evaluation of the Generated Data

In addition to using GPT-4V to evaluate the data we generated, we hired two annotators with sufficient professional knowledge to manually evaluate the data we generated. The following are the evaluation results for difficulty and image-text matching.

## E.1 Difficulty Comparison

We randomly selected 200 generated questions and their corresponding original questions, and asked annotators to compare the difficulty between the generated question and the original question. We display the results in Figure 3a, and the inner agreement between the two annotators is 0.74. In the

Figure, "Easier" indicates that the generated question is easier than the original question, with other symbols following the same pattern. Based on the experimental results, 77.75% of the generated questions are easier or of the same difficulty as the original ones, which indicates that our pipeline can reduce the difficulty of the questions.

## E.2 Image-text Matching Comparison

We randomly selected 200 generated questions and their corresponding original images, and asked annotators to judge which image, the generated one or the original one, better matches the generated question. We display the results in Figure 3b, and the inner agreement between the two annotators is 0.78. In the Figure, "Original" indicates that the original image better matches the question text, with other symbols following the same pattern. Based on the experimental results, we can observe that the generated images match the generated questions better than the original images.

![](images/d5547b70a4cfcc873b19e45192e941d3b81c0292ea88d96720142c4b1cc424fa.jpg)

![](images/ea1a0a0072ae242f8bcdc865525d0023291d9584b81f6e1a2d8b734807073be6.jpg)  
Figure 3: The manual data analysis results. Figure (a) is a manual comparison chart of the difficulty between the generated and original data. In this figure, “Easier” represents that the generated data is easier than the original data; “Harder” represents that the generated data is harder than the original data; “Equal” represents that the generated and original data have the same difficulty level. Figure (b) is a manual comparison chart of the image-text matching between the generated and original images. In this figure, “Original” represents that the original image better matches the question text; “Generated” represents that the generated image better matches the question text; “Equal” represents that the generated image and the original image match the text to the same degree.

<table><tr><td>Abbreviation</td><td>Task</td></tr><tr><td>MathVista</td><td></td></tr><tr><td>FQA GPS MWP TQA VQA ALG ARI GEO LOG NUM SCI</td><td>Figure Question Answering Geometry Problem Solving Math Word Problem Textbook question answering Visual Question Answering Algebraic Reasoning Arithmetic Reasoning Geometry Reasoning Logical Reasoning Numeric Commonsense</td></tr><tr><td>MathVision Algebra</td><td>Statistical Reasoning</td></tr><tr><td>Alg AnaG Ari</td><td>Analytic Geometry Arithmetic</td></tr><tr><td>CombG Comb Cnt DescG GrphT Log</td><td>Combinatorial Geometry Combinatorics Counting Descriptive Geometry</td></tr></table>

Table 6: Correspondence between abbreviations and detailed task names in MathVista and MathVision benchmarks.

<table><tr><td>Parameters</td><td>LLaVA-1.5</td><td>ShareGPT4V</td><td>InternVL-Chat-V1.2-Plus</td></tr><tr><td>Train Epochs</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Global Batch Size</td><td>128</td><td> $1 2 8$ </td><td>128</td></tr><tr><td>Learning Rate</td><td> $2 e ^ { - 5 }$ </td><td> $2 e ^ { - 5 }$ </td><td> $1 e ^ { - 5 }$ </td></tr><tr><td>Learning Rate Schedule</td><td>cosine decay</td><td>cosine decay</td><td>cosine decay</td></tr><tr><td>Weight Decay</td><td>0</td><td>0</td><td>0.05</td></tr><tr><td>Warmup Ratio</td><td>0.03</td><td>0.03</td><td>0.03</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Tune Visual Encoder</td><td>False</td><td>False</td><td>False</td></tr><tr><td>Tune MLP</td><td>True</td><td>True</td><td>True</td></tr><tr><td>Tune LLM</td><td>True</td><td>True</td><td>True</td></tr></table>

Table 7: Training parameters of different models. To make a fair comparison, we keep the training parameters consistent with those specified by the model’s original authors and train the models for one epoch.

![](images/32d8a57e960bd94a7e57954a855a036ba73142ca11c73530cfaa50b89500b036.jpg)  
Table 8: Prompt for Question-Answer Pairs Generation. We prompt GPT-4V to generate simplified questions. We also prompt GPT-4V to generate questions that are as diverse as possible to prevent GPT-4V from generating the same questions.

![](images/468a21765aff11f1198753219e92c4dfd29b80bf2cdca002f5f8deaedb9fe502.jpg)  
Table 9: Prompt for Wolfram Code Generation. When prompting GPT-4, we integrate both image descriptions and question-answer data to refine code generation. Additionally, we prompt GPT-4 to ensure variable naming within the code for clarity, aiming to enhance GPT-4’s grasp of the code’s relationship to the query at hand.

![](images/ded6d437fb2cbbc9624394571bd99f356639bcc0ea13db2708d3eec2cddbf6d7.jpg)  
Table 10: Prompt for Scoring. We employ GPT-4V to score the degree of alignment between the generated images and the questions. Specifically, the score is a decimal that ranges from 0 to 1. We also prompt GPT-4V to give a reason first and then give a final score, hoping this can enhance the accuracy of scoring.

Please act as a difficulty level evaluator.   
Give two geometric data, each consisting of a question, an answer, and an image.   
Please compare these two questions to determine which one is more difficult.   
If the first one is more difficult, output “1”; if the second one is more difficult, output “2”.   
Some useful tips:   
1. You should consider the complexity and difficulty of the questions and images.   
2. Don’t automatically assume that multiple-choice questions are easier.   
3. A shorter answer does not mean it’s easier.   
Input format:   
Question\_1: <the first question>   
Answer\_1: <the first answer>   
Question\_2: <the second question>   
Answer\_2: <the second answer>   
The first image corresponds to the first question, and the second image corresponds to the second question.   
You can only output the number “1” or “2”.

Table 11: Prompt for Difficulty Comparison. We prompt GPT-4V to determine which of the two questions is more difficult. We instruct GPT-4V not to simplistically assume that multiple-choice questions or shorter answers imply an easier question.
<table><tr><td>Model</td><td>Size</td><td>All</td><td>FQA</td><td>GPS</td><td>MWP</td><td>TQA</td><td>VQA</td><td>ALG</td><td>ARI</td><td>GEO</td><td>LOG</td><td>NUM</td><td>SCI</td><td>STA</td></tr><tr><td>LLaVA-1.5</td><td>7B</td><td>25.1*</td><td>23.79*</td><td>20.67*</td><td>12.90*</td><td>39.24*</td><td>32.40*</td><td>24.20*</td><td>22.10*</td><td>20.92*</td><td>16.22*</td><td>18.75*</td><td>36.89*</td><td>22.26*</td></tr><tr><td>LLaVA-1.5</td><td>13B</td><td>27.3*</td><td>22.68*</td><td>24.04*</td><td>16.67*</td><td>42.41*</td><td>35.75*</td><td>27.40*</td><td>24.93*</td><td>23.85*</td><td>18.92*</td><td>25.00*</td><td>39.34*</td><td>22.59*</td></tr><tr><td>LLaVA-1.5-G</td><td>7B</td><td>30.7</td><td>28.25</td><td>32.69</td><td>18.28</td><td>42.41</td><td>34.64</td><td>32.38</td><td>25.78</td><td>32.22</td><td>32.43</td><td>23.61</td><td>42.62</td><td>26.58</td></tr><tr><td>LLaVA-1.5-G</td><td>13B</td><td>32.2</td><td>28.25</td><td>36.54</td><td>19.89</td><td>41.14</td><td>37.99</td><td>35.23</td><td>28.05</td><td>37.24</td><td>27.03</td><td>26.39</td><td>42.62</td><td>27.57</td></tr><tr><td>ShareGPT4V</td><td>7B</td><td>27.3*</td><td>|21.93*</td><td>21.63*</td><td>19.35*</td><td>43.04*</td><td>36.31*</td><td>24.91*</td><td>27.20*</td><td>20.50*</td><td>18.92*</td><td>22.92*</td><td>40.16*</td><td>21.93*</td></tr><tr><td>ShareGPT4V</td><td>13B</td><td>30.4*</td><td>23.97*</td><td>27.40*</td><td>25.81*</td><td>43.67*</td><td>36.87*</td><td>28.83*</td><td>31.16*</td><td>27.62*</td><td>10.81*</td><td>26.39*</td><td>41.80*</td><td>26.91*</td></tr><tr><td>ShareGPT4V-G</td><td>7B</td><td>30.4</td><td>26.77</td><td>32.69</td><td>20.97</td><td>40.51</td><td>34.08</td><td>31.67</td><td>26.91</td><td>31.80</td><td>21.62</td><td>20.83</td><td>40.98</td><td>25.52</td></tr><tr><td>ShareGPT4V-G 13B</td><td></td><td>34.1</td><td>27.51</td><td>43.27</td><td>23.12</td><td>43.04</td><td>36.87</td><td>39.86</td><td>29.18</td><td>42.26</td><td>27.03</td><td>24.31</td><td>44.26</td><td>27.57</td></tr><tr><td>InternVL†</td><td>40B</td><td>59.9</td><td>51.7</td><td>61.1</td><td>79.6</td><td>52.5</td><td>57.0</td><td>54.5</td><td>63.2</td><td>61.1</td><td>16.2</td><td>48.6</td><td>55.7</td><td>60.8</td></tr><tr><td>InternVL-G†</td><td>40B</td><td>56.2</td><td>46.10</td><td>64.42</td><td>75.27</td><td>51.90</td><td>45.81</td><td>57.30</td><td>54.96</td><td>63.60</td><td>18.92</td><td>39.58</td><td>53.28</td><td>55.81</td></tr></table>

Table 12: Overall results of different models on the MathVista. For the model trained with GeoGPT4V, score increases are marked in red compared to the original model. ∗ indicates our re-implemented test results missed in benchmarks or origin papers. InternVL†represents the abbreviation for InternVL-Chat-V1.2-Plus. The suffix “-G” to the model name indicates a model trained on the GeoGPT4V. We present the detailed score for all the tasks such as “FQA” and “GPS”, as well as the overall (All) score for the benchmark. Due to limited space, we utilize abbreviations for the tasks and illustrate the detailed task name in the Appendix A.
<table><tr><td>Model</td><td>Size|</td><td>All</td><td>Alg</td><td>AnaG</td><td>Ari</td><td>CombG Comb</td><td></td><td>Cnt</td><td>DescG</td><td>GrphT</td><td>Log</td><td>Angle</td><td>Area</td><td>Len</td><td>SolG</td><td>Stat</td><td>Topo</td><td>TransG</td></tr><tr><td>LLaVA-1.5</td><td>7B</td><td>8.52</td><td>7.0</td><td>7.1</td><td>10.7</td><td>7.1</td><td>4.8</td><td>10.5</td><td>7.7</td><td>10.0</td><td>9.2</td><td>15.6</td><td>10.2</td><td>9.8</td><td>5.3</td><td>8.6</td><td>4.4</td><td>4.8</td></tr><tr><td>LLaVA-1.5</td><td>13B</td><td>11.12</td><td>7.0</td><td>14.3</td><td>14.3</td><td>9.1</td><td>6.6</td><td>6.0</td><td>13.5</td><td>5.6</td><td>13.5</td><td>10.4</td><td>12.6</td><td>14.7</td><td>11.5</td><td>13.8</td><td>13.0</td><td>10.7</td></tr><tr><td>LLaVA-1.5-G</td><td>7B|</td><td>12.89|</td><td>8.41</td><td>9.52</td><td>9.29</td><td>16.88</td><td>6.55</td><td>10.45</td><td>9.62</td><td>21.11</td><td>12.61</td><td>19.08</td><td>11.06 17.15</td><td></td><td>9.43</td><td>13.79</td><td>13.04</td><td>15.48</td></tr><tr><td>LLaVA-1.5-G</td><td>13B</td><td>13.98</td><td>9.28</td><td>15.48</td><td>16.43</td><td>14.29</td><td>10.71</td><td>10.45</td><td>12.50</td><td>18.89</td><td>11.76</td><td>19.65</td><td>13.6</td><td>18.49</td><td>10.25</td><td>13.79</td><td>17.39</td><td>13.10</td></tr><tr><td>ShareGPT4V</td><td>7B|</td><td>|10.53|</td><td>5.5</td><td>3.6</td><td>12.9</td><td>10.1</td><td>4.8</td><td>7.5</td><td>11.5</td><td>14.4</td><td>10.9</td><td>16.2</td><td>11.8</td><td>12.3</td><td>9.8</td><td>15.5</td><td>17.4</td><td>11.3</td></tr><tr><td>ShareGPT4V</td><td>13B</td><td>11.88</td><td>7.5</td><td>15.5</td><td>16.4</td><td>10.7</td><td>8.9</td><td>9.0</td><td>11.5</td><td>8.9</td><td>7.6</td><td>11.6</td><td>13.0</td><td>17.4</td><td>10.3</td><td>8.6</td><td>8.7</td><td>12.5</td></tr><tr><td>ShareGPT4V-G</td><td>7B|</td><td>|12.80|</td><td>7.83</td><td>11.9</td><td>15.00</td><td>12.99</td><td>5.95</td><td>7.46</td><td>9.62</td><td>16.67</td><td>15.97</td><td>17.34</td><td>13.60 17.59</td><td></td><td>10.25</td><td>15.52</td><td>8.70</td><td>11.31</td></tr><tr><td>ShareGPT4V-G 13B</td><td></td><td>12.63</td><td>8.41</td><td>22.62</td><td>15.00</td><td>9.74</td><td>6.55</td><td>8.96</td><td>13.46</td><td>11.11</td><td>15.13</td><td>19.08</td><td>15.80</td><td>13.81</td><td>9.02</td><td>6.90</td><td>13.04</td><td>13.69</td></tr><tr><td>InternVL†</td><td>40B|9.18*|8.41*</td><td></td><td></td><td>16.67*</td><td>8.57*</td><td>12.99*</td><td>9.52*</td><td>10.45*</td><td>15.38*</td><td>13.33*</td><td>11.76*</td><td>4.62*</td><td>5.60*</td><td>6.46*</td><td>9.84*</td><td>12.07*</td><td>21.74*</td><td>10.71*</td></tr><tr><td>InternVL-G†</td><td>40B</td><td>16.12</td><td>9.57</td><td>16.67</td><td>15.00</td><td>18.18</td><td>10.71</td><td>10.45</td><td>13.46</td><td>16.67</td><td>16.81</td><td>23.12</td><td>18.4</td><td>18.93</td><td>311.89</td><td>6.90</td><td>13.04</td><td>23.21</td></tr></table>

Table 13: Overall results of different models on the MathVision. For the model trained with GeoGPT4V, score increases are marked in red compared to the original model. ∗ indicates our re-implemented test results missed in benchmarks or origin papers. InternVL†represents the abbreviation for InternVL-Chat-V1.2-Plus. The suffix “-G” to the model name indicates a model trained on the GeoGPT4V. We present the detailed score for all the tasks such as “Alg” and “AnaG”, as well as the overall (All) score for the benchmark. Due to limited space, we utilize abbreviations for the tasks and illustrate the detailed task name in the Appendix A.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">MathVista</td><td colspan="10">MathVision</td></tr><tr><td>GPS</td><td>GEO</td><td>AVG</td><td>AnaG</td><td>CombG DescG</td><td></td><td>GrphT Angle</td><td>Area</td><td></td><td>Len</td><td>SolG TransG</td><td>AVG</td></tr><tr><td>InternVL-G†</td><td>40B 64.42</td><td></td><td>63.6</td><td>64.01</td><td>16.67 18.18</td><td>13.46</td><td>16.67</td><td>23.12</td><td>18.40</td><td>18.93</td><td>11.89</td><td>23.21</td><td>17.84</td></tr><tr><td colspan="10">Open-source Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-1.5</td><td>13B</td><td>24.04*</td><td>23.85*</td><td>23.95*</td><td>14.3 9.1</td><td>13.5</td><td>5.6</td><td>10.4</td><td>12.6</td><td>14.7</td><td>11.5</td><td>10.7</td><td>11.38</td></tr><tr><td>ShareGPT4V</td><td>13B</td><td>27.4*</td><td>27.62*</td><td>27.51* 15.5</td><td>10.7</td><td>11.5</td><td>8.9</td><td>11.6</td><td>13</td><td>17.4</td><td>10.3</td><td>12.5</td><td>12.38</td></tr><tr><td>G-LLaVA</td><td>13B</td><td>56.25*</td><td>51.88*</td><td>54.07*</td><td>9.52* 7.79*</td><td>8.65*</td><td>7.78*</td><td>8.67*</td><td>12.20*</td><td>10.02*</td><td>7.38*</td><td>8.93*</td><td>8.99*</td></tr><tr><td>InternLM-VL† InternVL†</td><td>7B 40B</td><td>63.0</td><td>62.3</td><td>62.65</td><td>15.5 15.3</td><td>14.4</td><td>22.2</td><td>19.7</td><td>15.6</td><td>15.0</td><td>11.9</td><td>15.5</td><td>16.12</td></tr><tr><td></td><td></td><td>61.1</td><td>61.1</td><td>61.1 16.67*</td><td>12.99*</td><td>15.38*</td><td>13.33*</td><td>4.62*</td><td>5.60*</td><td>6.46*</td><td>9.84*</td><td>10.71*</td><td>10.62*</td></tr><tr><td colspan="10">Closed-source Models</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-VL-Plus</td><td></td><td>38.5</td><td>39.3</td><td>38.90</td><td>17.9</td><td>12.7 15.4</td><td>8.9</td><td>11.6</td><td>6.4</td><td>10.0</td><td>14.3</td><td>11.31</td><td>12.06</td></tr><tr><td>Qwen-VL-Max</td><td>-</td><td></td><td>一</td><td>19.1</td><td>16.9</td><td>16.4</td><td>12.2</td><td>13.3</td><td>14.2</td><td>19.8</td><td>11.5</td><td>17.3</td><td>15.61</td></tr><tr><td>Gemini-1.0-Pro</td><td></td><td>40.4</td><td>41.0</td><td>40.70 10.7</td><td>20.1</td><td>20.2</td><td>21.1</td><td>19.1</td><td>19.0</td><td>20.0</td><td>14.3</td><td>20.8</td><td>18.37</td></tr><tr><td>Gemini-1.0-Ultra</td><td></td><td>56.2</td><td>55.6</td><td>55.90</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4V</td><td></td><td>50.5</td><td>51.0</td><td>50.75</td><td>32.1 21.1</td><td>22.1</td><td>14.4</td><td>22.0</td><td>22.2</td><td>20.9</td><td>23.8</td><td>25.6</td><td>22.69</td></tr></table>

Table 14: Overall results of our best model and other open-source and closed-source models on the MathVista and MathVision. We present the detailed score for all the tasks related to geometry such as “GPS” and “AnaG”, as well as the average score over these tasks in two benchmarks denoted as “AVG”. Due to limited space, we utilize abbreviations for these geometry-related tasks and illustrate the detailed task name in the Appendix A. Bold results indicate the best results for all models, and the red results indicate the best results among the open-source models. ‡indicates our re-implemented model without an official checkpoint. ∗ indicates our re-implemented test results missed in benchmarks or origin papers. InternVL†represents the abbreviation for InternVL-Chat-V1.2-Plus. InternLM-VL†represents the abbreviation for InternLM-XComposer2-VL. The suffix “-G” to the model name indicates a model trained on the GeoGPT4V.