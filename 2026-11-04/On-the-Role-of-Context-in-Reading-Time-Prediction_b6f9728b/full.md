# On the Role of Context in Reading Time Prediction

Andreas Opedal<sup>1,2</sup> Eleanor Chodroff <sup>3</sup> Ryan Cotterell<sup>1</sup> Ethan Gotlieb Wilcox<sup>1</sup>

<sup>1</sup>ETH Zürich <sup>2</sup>Max Planck ETH Center for Learning Systems <sup>3</sup>University of Zürich {andreas.opedal,ryan.cotterell,ethan.wilcox}@inf.ethz.ch eleanor.chodroff@uzh.ch

## Abstract

We present a new perspective on how readers integrate context during real-time language comprehension. Our proposals build on surprisal theory, which posits that the processing effort of a linguistic unit (e.g., a word) is an affine function of its in-context information content. We first observe that surprisal is only one out of many potential ways that a contextual predictor can be derived from a language model. Another one is the pointwise mutual information (PMI) between a unit and its context, which turns out to yield the same predictive power as surprisal when controlling for unigram frequency. Moreover, both PMI and surprisal are correlated with frequency. This means that neither PMI nor surprisal contains information about context alone. In response to this, we propose a technique where we project surprisal onto the orthogonal complement of frequency, yielding a new contextual predictor that is uncorrelated with frequency. Our experiments show that the proportion of variance in reading times explained by context is a lot smaller when context is represented by the orthogonalized predictor. From an interpretability standpoint, this indicates that previous studies may have overstated the role that context has in predicting reading times.

0 https://github.com/rycolab/ context-reading-time

## 1 Introduction

Surprisal theory (Hale, 2001; Levy, 2008) posits that the amount of effort it takes to process a linguistic unit is an affine function of its in-context information content, i.e., its surprisal. Numerous studies have found empirical support for surprisal theory across different reading measurement methods, languages, and language models (Smith and Levy, 2013; Wilcox et al., 2020; Kuribayashi et al., 2021; Meister et al., 2021; Wilcox et al., 2023; Shain et al., 2024), particularly when controlling for additional effects such as frequency. In this work, we take a critical look at surprisal theory as an adequate explanation for the role of context in reading time prediction, starting from a simple observation: Surprisal is but one quantity that can be derived from a language model to represent the effect of context (Giulianelli et al., 2024). We first show that, as an alternative to surprisal, one could take an association-based view on real-time language comprehension and model it as a function of the pointwise mutual information (PMI) between a unit and its context. Because PMI, surprisal, and frequency are collinear, all linear models with just two of these covariates are equivalent in terms of their predictive power. This simple identity therefore implies that all empirical validation of surprisal theory based on linear regression modeling also lends support for an association-based theory of language processing.

This raises the question of whether there is a more suitable way to estimate the effect that context has on reading time. We argue that, given that frequency is known to play an important role in processing effort (Broadbent, 1967; Inhoff and Rayner, 1986; Rayner and Duffy, 1986; Bybee, 2006), a more interesting construct to analyze should be what context contributes beyond what is already captured by frequency. To obtain a predictor that represents just that, we propose a technique where we project surprisal onto the orthogonal complement of frequency, ensuring that they are uncorrelated. This process effectively disentangles the contextual and non-contextual information into different covariates in our regressions and closely resembles residualization.<sup>1</sup>

To test whether the choice of contextual predictor matters empirically, we measure how much the variance in reading times explained by the contextual predictor changes when substituting surprisal for the orthogonalized context predictor. We find that our proposed predictor results in much smaller explained variance. Our results suggest that empirical work on surprisal theory has overestimated the effect that context has on reading times.

## 2 Predictive Models of Reading Behavior

We seek to model the cognitive processing difficulty of a unit $u , \mathrm { e . g . }$ , a word, drawn from an alphabet Σ. Additionally, we augment Σ to include a unique EOS $\not \in \Sigma$ symbol which indicates the end of an utterance; we further define ${ \overline { { \Sigma } } } \ { \overset { \mathrm { d e f } } { = } } \ \sum \cup \left\{ \operatorname { E O S } \right\}$ Let $\Sigma ^ { * }$ be the set of all strings over the alphabet $\Sigma ;$ we write $\ b { u } \in \Sigma ^ { * }$ for a string, $u _ { t }$ for the $t ^ { \mathrm { { t h } } }$ unit in $\mathbf { \pmb { u } } , | \mathbf { \pmb { u } } |$ for the number of units in ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ and $\mathbf { \Delta } \mathbf { u } v$ for the concatenation of u with another string v. Given a string cu, we are interested in how $u \mathrm { { s } }$ processing effort is shaped by its context of preceding units c.

A common psychometric proxy for the cognitive processing difficulty of u is the time it takes a human to read $u ,$ typically, as measured in an eyetracking study (Rayner, 1998). In general terms, we are interested in empirically assessing some theory of cognitive processing difficulty, which can be thought of as a collection of unit-level properties that are implicated in determining processing effort as measured by reading times. The most common type of evidence adduced to support such theories comes from (generalized) linear modeling. We define a predictor function as a function of type $\mathbf { X } \colon \Sigma ^ { * } \times \overline { { \Sigma } } \to \mathbb { R } ^ { D }$ , i.e., a function that maps a context–unit pair to a D-dimensional real vector. We model the reading time measurements as a linear model $f _ { \beta }$ conditioned on $\mathbf { X } ( c , { \overline { { u } } } )$ , i.e., $r ( \pmb { c } , \overline { { u } } ) \sim f _ { \beta } ( \mathbf { \partial } \cdot \mathbf { \partial } \mathbf { | } \mathbf { X } ( \pmb { c } , \overline { { u } } ) )$ where $\beta \in \mathbb { R } ^ { D }$ is a real-valued parameter vector. A model whose expected value, $\widehat { r } ( \mathbf { c } , \overline { { u } } ) \ = \ \beta ^ { \intercal } \mathbf { X } ( \mathbf { c } , \overline { { u } } )$ , achieves bhigh likelihood on held-out data lends empirical support to the theory that the factors measured by the predictors in X underlie the process of reading.

## 2.1 Language Modeling Background

We are particularly interested in predictors that are derived from language models (LMs) $p _ { \mathrm { H } }$ , which are distributions over $\Sigma ^ { * } { } ^ { 2 } \mathrm { ~ A ~ }$ relevant construct is the probability distribution over prefixes $c \in \Sigma ^ { * }$ called normalized prefix probability:

$$
\pi _ { \mathrm { H } } ( { \pmb c } ) = \frac { 1 } { Z _ { \pi _ { \mathrm { H } } } } \sum _ { { \pmb u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( { \pmb c u } ) ,\tag{1}
$$

where the normalizing constant $Z _ { \pi _ { \mathrm { H } } }$ is

$$
Z _ { \pi _ { \mathrm { H } } } = 1 + \sum _ { { \boldsymbol { \mathbf { u } } } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( { \boldsymbol { \mathbf { u } } } ) | { \boldsymbol { \mathbf { u } } } | .\tag{2}
$$

See Prop. 1 in App. A for a proof of $\operatorname { E q . } \left( 2 \right)$ . In words, this identity says that the normalized prefix probability exists when the expected string length is finite, which is the case for transformer-based LMs (Du et al., 2023). For simplicity, we further assume that $p _ { \mathrm { H } } ( \pmb { u } ) \ > \ 0$ for all $\textbf { \em u } \in \Sigma ^ { * }$ . This assumption holds true in practice due to the softmax function (Boltzmann, 1868; Gibbs, 1902), which enforces the probability estimates to be strictly positive. Then, for all $u \in \Sigma$

$$
p _ { \mathrm { H } } ( u \mid c ) \overset { \mathrm { d e f } } { = } \frac { \pi _ { \mathrm { H } } ( \pmb { c u } ) } { \pi _ { \mathrm { H } } ( \pmb { c } ) } .\tag{3}
$$

The EOS symbol is special in the sense that

$$
p _ { \mathrm { H } } ( \cos \mid { \pmb { c } } ) \overset { \mathrm { d e f } } { = } \frac { p _ { \mathrm { H } } ( { \pmb { c } } ) } { \pi _ { \mathrm { H } } ( { \pmb { c } } ) } .\tag{4}
$$

Thus, $p _ { \mathrm { H } } ( \overline { { u } } \mid c )$ is a probability distribution over $\overline { { \Sigma } }$ . Importantly, note that $p _ { \mathrm { H } } ( \overline { { u } } \ \lvert \textbf { \em c } )$ is not the probability of cu as an entire string given that we know $^ { c , }$ only that u follows $^ { c . }$

2.2 Frequency as a Predictor of Reading Time Previous studies (e.g., Shain, 2019, 2024) have investigated the effect of frequency, operationalized as unigram surprisal, on reading time. A unigram $\mathrm { L M } q _ { \mathrm { H } }$ is a distribution over $\Sigma ^ { * }$ where, when a string is sampled autoregressively, each unit is conditionally independent of the context. In notation, we write $q _ { \mathrm { H } } ( \overline { { u } } )$ for the probability of u independent of context.

We now consider the unigram model that best approximates the human LM $p _ { \mathrm { H } }$ in the sense of the forward Kullback–Leibler divergence $\mathrm { K L } ( p _ { \mathrm { H } } \mid | q )$ We can compute the minimizer $q _ { \mathrm { H } }$ in closed form. We define the following function that counts the number of occurrences of a unit $\overline { { u } } \in \overline { { \Sigma } }$ in u:

$$
\# ( \pmb { u } , \overline { { \boldsymbol { u } } } ) \overset { \mathrm { d e f } } { = } \sum _ { t = 1 } ^ { | \pmb { u } | } \mathbb { 1 } \{ \overline { { \boldsymbol { u } } } = \boldsymbol { u } _ { t } \} + \mathbb { 1 } \{ \overline { { \boldsymbol { u } } } = \mathrm { E O S } \} .\tag{5}
$$

Then, the minimizing unigram LM, factored autoregressively, is given by

$$
q _ { \mathrm { H } } ( \overline { { u } } ) = \frac { 1 } { Z _ { q _ { \mathrm { H } } } } \sum _ { { \pmb u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( { \pmb u } ) \# ( { \pmb u } , \overline { { { u } } } ) ,\tag{6}
$$

where the normalizing constant $Z _ { q _ { \mathrm { H } } }$ is necessarily finite for language models of finite expected length.<sup>3</sup> Then, given an LM p<sub>H</sub> with a unigram

<sup>3</sup>As a counterexample, consider an LM p<sub>H</sub> with $\Sigma =$ $\{ a \}$ and $\textstyle p _ { \mathrm { H } } ( a ^ { n } ) = { \frac { 1 } { \pi ^ { 2 } / 6 } } ~ \cdot ~ { \frac { 1 } { n ^ { 2 } } }$ for n $\in \mathbb { Z } _ { > 0 } , { \bar { \mathrm { i . e . } } }$ ., where the probabilities are globally normalized by $\frac { \pi ^ { 2 } } { 6 }$ , the solution to the Basel problem. The expected count of a would depend on $\scriptstyle \sum _ { n = 1 } ^ { \infty } { \frac { 1 } { n ^ { 2 } } } \cdot n$ , which is divergent. Thus, $Z _ { q _ { \mathrm { H } } } = \infty$

LM $q _ { \mathrm { H } }$ , the unigram surprisal is given by

$$
\begin{array} { r } { \upsilon _ { \mathrm { H } } ( \overline { { u } } ) \stackrel { \mathrm { d e f } } { = } - \log q _ { \mathrm { H } } ( \overline { { u } } ) . } \end{array}\tag{7}
$$

We will refer to unigram surprisal as frequency for the remainder of this paper. Importantly, frequency is often considered as a control variable, rather than the factor being investigated in support of a particular cognitive theory of language processing.<sup>4</sup>

## 2.3 Surprisal as a Predictor of Reading Time

A common claim is that reading is mediated by contextual surprisal (Shannon, 1948), defined as

$$
\iota _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) \stackrel { \mathrm { d e f } } { = } - \log p _ { \mathrm { H } } ( \overline { { u } } \mid \pmb { c } ) .\tag{8}
$$

Indeed, this claim has received much empirical support (Hale, 2001; Demberg and Keller, 2008; Smith and Levy, 2008, inter alia). Importantly, there is evidence that the particular functional relationship, called the linking function, between contextual surprisal and reading time is $\mathrm { a f f i n e } ^ { 5 }$ (Smith and Levy, 2013; Wilcox et al., 2023; Shain et al., 2024), justifying the use of linear regression modeling.

## 2.4 PMI as a Predictor of Reading Time

Next, we point out an alternative way of deriving a contextual predictor from an LM, namely, as the pointwise mutual information (PMI; Fano, 1961) between a unit and its context. PMI measures association, and has been an important notion in NLP (Church and Hanks, 1990; Levy and Goldberg, 2014) and, more recently, psycholinguistics (Tsipidi et al., 2024; Wilcox et al., 2024). The PMI between a unit $\overline { { u } } \in \overline { { \Sigma } }$ and its context $c \in \Sigma ^ { * }$ is

$$
\mu _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) \overset { \mathrm { d e f } } { = } \log \frac { p _ { \mathrm { H } } ( \overline { { u } } \mid \pmb { c } ) \pi _ { \mathrm { H } } ( \pmb { c } ) } { \pi _ { \mathrm { H } } ( \pmb { c } ) q _ { \mathrm { H } } ( \overline { { u } } ) } .\tag{9}
$$

The probability that c and u occur together is expressed in the numerator (rewritten using Eqs. (3) and (4)). The denominator expresses what this probability would be if c and u were independent.

If PMI is predictive of reading times, then that would suggest a theory positing that the strength of association that the observed unit has with its context is part of what determines the effort it takes to process it. It turns out that many of the empirical results that have been published in support of surprisal theory, actually, by courtesy of the assumed affine linking function, provide an equal amount of evidence for a PMI-based theory. To see this, first note that we can rewrite PMI as the difference between frequency and surprisal:

$$
\mu _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) = \log \frac { p _ { \mathrm { H } } ( \overline { { u } } \mid \pmb { c } ) } { q _ { \mathrm { H } } ( \overline { { u } } ) }\tag{10a}
$$

$$
= v _ { \mathrm { H } } ( \overline { { u } } ) - \iota _ { \mathrm { H } } ( c , \overline { { u } } ) .\tag{10b}
$$

This equation shows that υ<sub>H</sub>, ι<sub>H</sub> and $\mu _ { \mathrm { H } }$ are linearly dependent in a certain Hilbert space, which we will introduce in §3. Now, under a linear model $f _ { \beta }$ with only surprisal and frequency as predictors, the expected value of $f _ { \beta }$ , denoted by ${ \widehat { r } } ( c , { \overline { { u } } } )$ , is given by

$$
\boldsymbol { \widehat { r } } ( \boldsymbol { c } , \boldsymbol { \overline { { u } } } ) = \beta _ { 0 } + \beta _ { v _ { \mathrm { H } } } v _ { \mathrm { H } } ( \boldsymbol { \overline { { u } } } ) + \beta _ { \iota _ { \mathrm { H } } } \iota _ { \mathrm { H } } ( \boldsymbol { c } , \boldsymbol { \overline { { u } } } ) .\tag{11}
$$

By adding and subtracting an additional $\beta _ { \iota _ { \mathrm { H } } }$ υ<sub>H</sub>(u) term, this can be rewritten as

$$
\begin{array} { r } { \widehat { r } ( \pmb { c } , \overline { { u } } ) = \beta _ { 0 } + ( \beta _ { v _ { \mathrm { H } } } + \beta _ { v _ { \mathrm { H } } } ) v _ { \mathrm { H } } ( \overline { { u } } ) - \beta _ { v _ { \mathrm { H } } } \mu _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) . } \end{array}\tag{12}
$$

(We suppressed an intermediate step, given in App. B.1.) Thus, it turns out that the very same coefficient that is typically taken to indicate the effect of surprisal also has an alternative interpretation as the negative effect of PMI. Furthermore, the predictive power of a linear model with surprisal and frequency is the same as that of a linear model with PMI and frequency. In other words, if frequency is provided as a predictor, additionally adding surprisal as a predictor is no more predictive than adding PMI ceteris paribus. However, two such models will differ in the coefficient assigned to frequency: $\beta _ { v _ { \mathrm { H } } }$ in the surprisal and frequency model, versus $\beta _ { v _ { \mathrm { H } } } + \beta _ { { \iota } _ { \mathrm { H } } }$ in the PMI and frequency model. As a consequence, they will also differ in terms of the strength of the effect attributed to the predictor that stands in for context, i.e., surprisal or PMI.

## 3 Disentangling the Effect of Context

As there is a large and established body of work showing that frequency plays a major role in explaining the effort it takes to process words (see, e.g., Bybee, 2006), we argue that the interest of surprisal theory lies in understanding what additional effect there is of contextual information beyond frequency. The exposition above implies that neither surprisal nor PMI should receive special status as a measure of the effect of context. Moreover, both surprisal and PMI are correlated with frequency<sup>6</sup> and all three are collinear.

We now present a simple technique to decorrelate frequency from surprisal, resulting in a new predictor that is engineered to be disentangled from frequency. Importantly, our technique attributes the shared effect of frequency and surprisal on reading time to frequency, and then creates a new predictor which represents the effect of surprisal that is not shared with frequency. We argue that this new predictor is more relevant to study than either surprisal or PMI since it represents only the effect of context.

Our technical exposition starts with an underlying probability space $\left( \Sigma ^ { * } { \times } \overline { { \Sigma } } , \mathcal { P } ( \Sigma ^ { * } { \times } \overline { { \Sigma } } ) , \pi _ { \mathrm { H } } \cdot p _ { \mathrm { H } } \right)$ Next, consider the following random variables under this probability space: $\mathbf { I } _ { \mathrm { H } }$ encoding the distribution over surprisals $\iota _ { \mathrm { H } } ( c , \overline { { u } } )$ of the next unit given a context, ${ \bf { M } } _ { \mathrm { { H } } }$ encoding the distribution over PMIs $\mu _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } )$ between the next unit and a context, and ${ \bf Y } _ { \mathrm { H } }$ encoding the distribution over frequencies $v _ { \mathrm { H } } ( \overline { { u } } )$ of a unit. Note that $\mathbf { I } _ { \mathrm { H } } , \mathbf { M } _ { \mathrm { H } }$ and ${ \bf Y } _ { \mathrm { H } }$ are realvalued random variables and that ${ \bf Y } _ { \mathrm { H } }$ is constant in c. They are elements of a Hilbert space  over R containing all random variables under the above probability space that have finite second moment (Rudin, 1987). The inner product on  is given by

$$
\begin{array} { r l r } {  { \langle { \mathbf { X } } , { \mathbf { Y } } \rangle \overset { \mathrm { d e f } } { = } \mathbb { E } [ { \mathbf { X } } { \mathbf { Y } } ] } } \\ & { } & { = \sum _ { c \in \Sigma ^ { * } } \pi _ { \mathrm { H } } ( c ) p _ { \mathrm { H } } ( \overline { { u } } \mid c ) { \mathbf { X } } ( c , \overline { { u } } ) { \mathbf { Y } } ( c , \overline { { u } } ) . } \end{array}
$$

App. C.1 provides further details on why $\mathcal { H }$ is indeed a Hilbert space over R with the above inner product. With  being a Hilbert space, we can take projections on . Taking the projection of $\mathbf { I } _ { \mathrm { H } }$ onto the orthogonal complement of ${ \bf Y } _ { \mathrm { H } }$ we get

$$
\operatorname { p r o j } _ { \mathbf { Y } _ { \mathrm { H } } ^ { \perp } } ( \mathbf { I } _ { \mathrm { H } } ) = \mathbf { I } _ { \mathrm { H } } - \frac { \langle \mathbf { I } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } \rangle } { \langle \mathbf { Y } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } \rangle } \mathbf { Y } _ { \mathrm { H } } .\tag{14}
$$

Projecting in this manner results in an orthogonalization in the sense that $\langle \mathbf { Y } _ { \mathrm { H } } , \mathrm { p r o j } _ { \mathbf { Y } _ { \mathrm { H } } ^ { \perp } } ( \mathbf { I } _ { \mathrm { H } } ) \rangle = 0$ as a consequence of the Hilbert projection theorem (Rudin, 1991, pp. 306-9). See App. C.2 for a proof. If the expected values of at least one of the random variables X and Y is 0, which can be achieved by a simple mean-centering transformation, then

$$
\begin{array} { r } { \langle \mathbf { X } , \mathbf { Y } \rangle = \mathbb { E } [ \mathbf { X } \mathbf { Y } ] \qquad } \\ { = \operatorname { C o v } ( \mathbf { X } , \mathbf { Y } ) + \underbrace { \mathbb { E } [ \mathbf { X } ] \mathbb { E } [ \mathbf { Y } ] } _ { = 0 } . } \end{array}\tag{15}
$$

Thus, if ${ \bf Y } _ { \mathrm { H } }$ and $\mathbf { I } _ { \mathrm { H } }$ are mean-centered the covariance between ${ \bf Y } _ { \mathrm { H } }$ and $\mathrm { p r o j } _ { \mathbf { Y } _ { \mathrm { H } } ^ { \perp } } ( \mathbf { I } _ { \mathrm { H } } )$ will be 0, i.e., they will be decorrelated. The random variable $\mathrm { p r o j } _ { \mathbf { Y } _ { \mathrm { H } } ^ { \perp } } ( \mathbf { I } _ { \mathrm { H } } )$ constitutes a new predictor variable, which we term orthogonalized surprisal. In words, orthogonalized surprisal represents the effect of context and is disentangled from frequency.<sup>7</sup>

## 4 Variance Explained by Context

We now seek an empirical understanding of how the new orthogonalized surprisal predictor influences the importance attributed to context in reading time prediction. In addition to the experiment presented here, we also compared the predictive power for nonlinear models across different predictors. Those experiments are discussed in App. F.

Dataset and Predictors. We use gaze duration times from the Multilingual Eye-movement Corpus (MECO; Siegelman et al., 2022), which consists of word-level eye-tracking measurements from several languages.To obtain surprisal estimates, we approximate $p _ { \mathrm { H } }$ with mGPT (Shliazhko et al., 2024), which is a multilingual LM based on GPT-2. The frequency estimates are from Speer (2022)<sup>8</sup> and PMI is computed through the decomposition given in Eq. (10b). Orthogonalized surprisal is obtained by approximating $\langle \mathbf { I } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } \rangle$ and $\langle \mathbf { Y } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } \rangle$ using only the words that occur in a training corpus. We transform surprisal and frequency to be meancentered, and use the transformed data to compute unbiased sample variances and covariances. Those sample estimates are then plugged into $\operatorname { E q . }$ . (14). App. B.2 gives the formulaic details of this approximation. We also include a word length predictor. Word length is correlated with frequency (Zipf, 1949), so we orthogonalize it in the same manner as surprisal. App. E gives more details on the dataset and predictor variables.

Experimental Setup. We fit three linear models using ordinary least squares, with predictors: (i) surprisal, frequency, and length, (ii) PMI, frequency, and length, and (iii) orthogonalized surprisal, frequency, and orthogonalized length. (An alternative perspective would be to use orthogonalized frequency, which assumes that surprisal is the more fundamental predictor. We provide such results, which are are consistent with the current conclusions, in App. F.1.) We include spillover effects from the previous word and fit models over ten folds of cross-validation. We then measure the relative importance of predictors by averaging over the proportion of variance explained by the predictors across orderings in which they are added to the model (Lindeman et al., 1980; Kruskal, 1987), a technique known as LMG (Grömping, 2007).<sup>9</sup> One advantage of this technique, compared to previous methods (e.g., Delta Log Likelihood; Goodkind and Bicknell, 2018) is that LMG gives a better absolute sense of predictive power.

![](images/8af9ac875988d4591957185c0b7968ade2b9ecd7e8a785221626a2c30131c1ca.jpg)  
Figure 1: Proportion of total variance explained by the predictors (Kruskal, 1987), across languages and linear regression models. Summing the values represented by the four bars yields the coefficient of determination $R ^ { 2 }$ . Note that the $R ^ { 2 }$ values are the same across the three models, as a consequence of the collinearity discussed in §2.4. Error bars are 95% confidence intervals across folds of data. We observe only a small proportion of explained variance by context when excluding the variance it shares with frequency, as in the orthogonalized surprisal predictor.

Results. The LMG values for the different predictors across these models are visualized in Fig. 1. Comparing the plots on the bottom row to the plots on the top row, we observe that the explained variance for orthogonalized surprisal is much lower than for surprisal. These results are consistent across languages. For most languages, when ranking the importance of the predictors, orthogonalization shifts the third most important predictor from context to the previous word’s frequency. Our results suggest that using surprisal therefore overestimates the importance that context has on reading time.<sup>10</sup> We observe values for PMI that for most languages lie between those for surprisal and orthogonalized surprisal, indicating that the extent to which that PMI overestimates the context effect is smaller compared to surprisal. We observe that the mean $R ^ { 2 }$ values of the models across LMG orderings, i.e., the sums of the bars within each facet, range between 0.6–0.8, indicating that the linear models capture a fairly large proportion of the variance observed during the reading process.

## 5 Conclusion

This article discusses predictors that capture how the processing effort of a unit is shaped by its context. We made the observation that there exist alternatives to the widely used surprisal predictor. Surprisal is correlated with non-contextual frequency, so we provided a technique to disentangle contextual and non-contextual information in language models. In so doing, we found that the effect that context has on reading times appears to be small in comparison to non-contextual frequency.

## Limitations

Our approach takes one predictor to remain untouched (i.e., frequency), and modifies others to reflect effects that are disassociated from the first. As suggested above, it would thus be natural to ask what happens in the alternative setting, where surprisal remains untouched and frequency is projected onto the orthogonal complement of surprisal. We provide such an analysis in App. F.1. It turns out that even when attributing the shared effect of frequency and surprisal on reading times to surprisal—which is the case when replacing frequency by an orthogonalized frequency predictor—the variance explained by the frequency predictor is still higher for most languages in comparison to the surprisal predictor. This gives further support to our conclusion that context appears to play a small role in reading time prediction.

Furthermore, our presentation of ideas and discussion largely ignores effects of word lengths. We find word lengths to explain the most variance in reading times in our surprisal and PMI models. In addition, after residualizing word length against frequency, we find length to be the second strongest predictor, with an explained variance ranging from around 10%–21%. One hypothesis is that readers may make multiple saccades within first passes of longer words, and the time it takes to plan and execute these saccades could be the underlying reason why orthogonalized word length remains explanatory even after residualization. Future work could control for this by adding in the number of saccades within a word as an additional predictor into models.

We are unaware of any efficient algorithm to compute $\langle \mathbf { I } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } \rangle$ and $\langle { \bf Y } _ { \mathrm { H } } , { \bf Y } _ { \mathrm { H } } \rangle$ exactly, so in practical settings we must rely on estimation. Thus, it may be that the orthogonalized surprisal predictor is only “close to” being orthogonal to frequency in practice. The manner in which estimation is performed makes our technique similar to residualization—see App. D for a discussion. Moreover, our method only provides guarantees for predictor variables that live in .

Importantly, an off-by-one issue was detected in the trial, sentence and word (interest area) ID scheme for a handful of tokens in MECO (Versions 1.1 and 1.2). This was corrected for our analysis, but should be taken into consideration by other researchers using this dataset. The code to correct the data is included in our repository. In addition, we also identified a few instances of repeated words within a sentence (e.g., “als als”), but given their consistency across participants, we assumed these were typos. These were retained in the analysis, but could have resulted in atypical responses.

In addition, the current paper makes use of a fixed-effects linear regression model with averaged data, as opposed to the more standard mixedeffects regression. Estimation of $R ^ { 2 }$ values from mixed-effects models can differ depending on the researcher assumptions and has historically been under-reported due to this limitation (Nakagawa and Schielzeth, 2013). Nevertheless, some proposals have been made regarding best practices (Nakagawa and Schielzeth, 2013; Rights and Sterba, 2018). Future research should investigate the feasibility of our approach, particularly with the use of partial effect sizes (i.e., the LMG approach), but using mixed-effects models.

Another limitation of this work is that, while we investigate several languages, these are still biased towards Indo-European languages. For example, we present results from one language only for Fino-Uralic, Semitic, Turkic, and Koreanic language families, but seven Indo-European languages. Expanding these results to even more languages would further broaden the impact of this work. In addition, we observe somewhat unique effects for Korean, where, in orthogonalized models, frequency accounts for a lower proportion of the variance, and length and context account for higher proportions, at least compared to other languages. One possible reason for this is the Korean script (Hangul), which combines features of both alphabetic and syllabic writing systems. Future work should conduct similar analyses on different Korean datasets to determine whether this trend is a property of Korean, or just our particular Korean language dataset.

## Ethical Considerations

This work uses previously collected human data from the MECO dataset. Please see the paper that introduces this dataset (Siegelman et al., 2022) for information about the data collection procedure. The authors foresee no ethical problems arising from the work presented here.

## Acknowledgments

We thank Cory Shain for useful discussion. Andreas Opedal acknowledges funding from the Max Planck ETH Center for Learning Systems.

## References

Ludwig Boltzmann. 1868. Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten. K.K. Hof und Staatsdruckerei.

James A. Breaugh. 2006. Rethinking the control of nuisance variables in theory testing. Journal ofBusiness and Psychology, 20(3):429–443.

Donald E. Broadbent. 1967. Word-frequency effect and response bias. Psychological Review, 74 1:1–15.

Joan Bybee. 2006. Frequency ofuse and the organization oflanguage. Oxford University Press.

Kenneth Ward Church and Patrick Hanks. 1990. Word association norms, mutual information, and lexicography. Computational Linguistics, 16(1):22–29.

Vera Demberg and Frank Keller. 2008. Data from eyetracking corpora as evidence for theories of syntactic processing complexity. Cognition, 109(2):193–210.

Li Du, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. 2023. A measure-theoretic characterization of tight language models. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9744–9770, Toronto, Canada. Association for Computational Linguistics.

Robert M. Fano. 1961. Transmission of Information: A Statistical Theory ofCommunication. MIT Press Classics. MIT Press.

Richard Futrell, Edward Gibson, and Roger P. Levy. 2020. Lossy-context surprisal: An informationtheoretic model of memory effects in sentence processing. Cognitive Science, 44(3):e12814.

Catalina B. García, Román Salmerón, Claudia García, and José García. 2019. Residualization: justification, properties and application. Journal ofApplied Statistics, 47(11):1990–2010.

Josiah Willard Gibbs. 1902. Elementary Principles in Statistical Mechanics. Charles Scribner’s Sons.

Mario Giulianelli, Andreas Opedal, and Ryan Cotterell. 2024. Generalized measures of anticipation and responsivity in online language processing. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, Miami, Florida, USA. Association for Computational Linguistics.

Adam Goodkind and Klinton Bicknell. 2018. Predictive power of word surprisal for reading times is a linear function of language model quality. In Proceedings of the 8th Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2018), pages 10–18, Salt Lake City, Utah. Association for Computational Linguistics.

Ulrike Grömping. 2007. Estimators of relative importance in linear regression based on variance decomposition. The American Statistician, 61(2):139–147.

John Hale. 2001. A probabilistic Earley parser as a psycholinguistic model. In Second Meeting of the North American Chapter of the Association for Computational Linguistics.

Jacob Louis Hoover, Morgan Sonderegger, Steven T. Piantadosi, and Timothy J. O’Donnell. 2023. The Plausibility of Sampling as an Algorithmic Theory of Sentence Processing. Open Mind, 7:350–391.

Albrecht Werner Inhoff and Keith Rayner. 1986. Parafoveal word processing during eye fixations in reading: Effects of word frequency. Perception & Psychophysics, 40(6):431–439.

Florian Jaeger. 2010. Redundancy and reduction: Speakers manage syntactic information density. Cognitive Psychology, 61(1):23–62.

William Kruskal. 1987. Relative importance by averaging over orderings. The American Statistician, 41(1):6–10.

Victor Kuperman, Raymond Bertram, and R. Harald Baayen. 2008. Morphological dynamics in compound processing. Language and Cognitive Processes, 23(7-8):1089–1132.

Tatsuki Kuribayashi, Yohei Oseki, Ana Brassard, and Kentaro Inui. 2022. Context limitations make neural language models more human-like. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10421–10436, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Tatsuki Kuribayashi, Yohei Oseki, Takumi Ito, Ryo Yoshida, Masayuki Asahara, and Kentaro Inui. 2021. Lower perplexity is not always human-like. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5203–5217, Online. Association for Computational Linguistics.

Omer Levy and Yoav Goldberg. 2014. Neural word embedding as implicit matrix factorization. In Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc.

Roger Levy. 2008. Expectation-based syntactic comprehension. Cognition, 106(3):1126–1177.

Richard H. Lindeman, Peter F. Merenda, and Ruth Z. Gold. 1980. Introduction to bivariate and multivariate analysis. Glenview (Ill.) : Scott.

Clara Meister, Tiago Pimentel, Patrick Haller, Lena Jäger, Ryan Cotterell, and Roger Levy. 2021. Revisiting the Uniform Information Density hypothesis. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 963– 980, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Shinichi Nakagawa and Holger Schielzeth. 2013. A general and simple method for obtaining r2 from generalized linear mixed-effects models. Methods in ecology and evolution, 4(2):133–142.

Irene Nikkarinen, Tiago Pimentel, Damián Blasi, and Ryan Cotterell. 2021. Modeling the unigram distribution. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3721–3729, Online. Association for Computational Linguistics.

Byung-Doh Oh and William Schuler. 2023a. Transformer-based language model surprisal predicts human reading times best with about two billion training tokens. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 1915–1921, Singapore. Association for Computational Linguistics.

Byung-Doh Oh and William Schuler. 2023b. Why does surprisal from larger transformer-based language models provide a poorer fit to human reading times? Transactions of the Association for Computational Linguistics, 11:336–350.

Byung-Doh Oh, Shisen Yue, and William Schuler. 2024. Frequency explains the inverse correlation of large language models’ size, training data amount, and surprisal’s fit to reading times. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2644–2663, St. Julian’s, Malta. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67.

Keith Rayner. 1998. Eye movements in reading and information processing: 20 years of research. Psychological bulletin, 124 3:372–422.

Keith Rayner and Susan A. Duffy. 1986. Lexical complexity and fixation times in reading: Effects of word frequency, verb complexity, and lexical ambiguity. Memory & Cognition, 14(3):191–201.

Jason D. Rights and Sonya K. Sterba. 2018. A framework of R-squared measures for single-level and multilevel regression mixture models. Psychological Methods, 23(3):434.

Walter Rudin. 1987. Real and complex analysis, 3rd ed. McGraw-Hill, Inc., USA.

Walter Rudin. 1991. Functional Analysis. International series in pure and applied mathematics. McGraw-Hill.

Cory Shain. 2019. A large-scale study of the effects of word frequency and predictability in naturalistic reading. In Proceedings ofthe 2019 Conference of

the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4086–4094, Minneapolis, Minnesota. Association for Computational Linguistics.

Cory Shain. 2024. Word Frequency and Predictability Dissociate in Naturalistic Reading. Open Mind, 8:177–201.

Cory Shain, Clara Meister, Tiago Pimentel, Ryan Cotterell, and Roger Levy. 2024. Large-scale evidence for logarithmic effects of word predictability on reading time. Proceedings of the National Academy of Sciences, 121(10):e2307876121.

Claude E. Shannon. 1948. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379–423.

Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Anastasia Kozlova, Vladislav Mikhailov, and Tatiana Shavrina. 2024. mGPT: Few-Shot Learners Go Multilingual. Transactions of the Association for Computational Linguistics, 12:58–79.

Noam Siegelman, Sascha Schroeder, Cengiz Acartürk, Hee-Don Ahn, Svetlana Alexeeva, Simona Amenta, Raymond Bertram, Rolando Bonandrini, Marc Brysbaert, Daria Chernova, et al. 2022. Expanding horizons of cross-linguistic research on reading: The multilingual eye-movement corpus (MECO). Behavior Research Methods, 54(6):2843–2863.

Nathaniel J. Smith and Roger Levy. 2008. Optimal processing times in reading: A formal model and empirical investigation. In Proceedings of the 30th Annual Conference ofthe Cognitive Science Society.

Nathaniel J. Smith and Roger Levy. 2013. The effect of word predictability on reading time is logarithmic. Cognition, 128(3):302–319.

Robyn Speer. 2022. rspeer/wordfreq: v3.0.

Eleftheria Tsipidi, Franz Nowak, Ryan Cotterell, Ethan Gotlieb Wilcox, Mario Giulianelli, and Alex Warstadt. 2024. Surprise! Uniform information density isn’t the whole story: Predicting surprisal contours in long-form discourse. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, Florida, USA. Association for Computational Linguistics.

Ethan Gotlieb Wilcox, Jon Gauthier, Jennifer Hu, Peng Qian, and Roger P. Levy. 2020. On the predictive power of neural language models for human realtime comprehension behavior. In Proceedings of the 42nd Annual Meeting of the Cognitive Science Society, page 1707–1713.

Ethan Gotlieb Wilcox, Tiago Pimentel, Clara Meister, and Ryan Cotterell. 2024. An information-theoretic analysis of targeted regressions during reading. Cognition, 249:105765.

Ethan Gotlieb Wilcox, Tiago Pimentel, Clara Meister, Ryan Cotterell, and Roger P. Levy. 2023. Testing the Predictions of Surprisal Theory in 11 Languages. Transactions of the Association for Computational Linguistics, 11:1451–1470.

Lee H. Wurm and Sebastiano A. Fisicaro. 2014. What residualizing predictors in regression analyses does (and what it does not do). Journal ofMemory and Language, 72:37–48.

Richard York. 2012. Residualization is not the answer: Rethinking how to address multicollinearity. Social Science Research, 41(6):1379–1386.

George K. Zipf. 1949. Human Behaviour and the Principle of Least Effort. Addison-Wesley.

## A Normalizing the Prefix Probabilities

Here we show that $Z _ { \pi _ { \mathrm { H } } }$ as defined in Eq. (2) is the normalizing constant for the prefix probabilities. Proposition 1. Let $Z _ { \pi _ { \mathrm { H } } }$ be defined as in Eq. (2), i.e.,

$$
Z _ { \pi _ { \mathrm { H } } } = 1 + \sum _ { { \boldsymbol { \mathbf { u } } } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( { \boldsymbol { \mathbf { u } } } ) | { \boldsymbol { \mathbf { u } } } | .\tag{16}
$$

Then, $Z _ { \pi _ { \mathrm { H } } }$ is the normalizing constant for π<sub>H</sub>, i.e.,

$$
\sum _ { c \in \Sigma ^ { * } } \pi _ { \mathrm { H } } ( c ) = 1 .\tag{17}
$$

Proof. First, note that Eq. (2) can be rewritten in the following manner:

$$
Z _ { \pi _ { \mathrm { H } } } = 1 + \sum _ { \boldsymbol { u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( \boldsymbol { u } ) | \boldsymbol { u } |\tag{18a}
$$

$$
= \sum _ { \pmb { u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( \pmb { u } ) + \sum _ { \pmb { u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( \pmb { u } ) | \pmb { u } |\tag{18b}
$$

$$
= \sum _ { \pmb { u } \in \Sigma ^ { * } } ( 1 + | \pmb { u } | ) p _ { \mathrm { H } } ( \pmb { u } )\tag{18c}
$$

$$
= \sum _ { c \in \Sigma ^ { * } } \sum _ { { \pmb u } ^ { \prime } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( { \pmb c u } ^ { \prime } ) .\tag{18d}
$$

The last step follows from the fact that a string u can be segmented into two substrings c and $\mathbf { { \boldsymbol { u } } } ^ { \prime }$ in $1 + | \pmb { u } |$ ways. Note the two cases where either $c = \varepsilon$ and $\mathbf { \nabla } \mathbf { \pmb { u } } ^ { \prime } = \mathbf { \vec { u } }$ , or c = u and ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf } { } { \mathbf } { } { \mathbf } { } { \mathbf } { } { \mathbf } { } \mathbf { }  { \mathbf { } } { \mathbf } { } { \mathbf } { } { \mathbf } { } { \mathbf } { } \mathbf { } { } \mathbf { }  { \mathbf } { } { \mathbf } { } { \mathbf } { } \mathbf { } { } \mathbf { } \mathbf { } { } \mathbf { }  { \mathbf } { } { \mathbf } { } \mathbf { } \mathbf { } { } \mathbf { } \mathbf { } \mathbf { }  { \mathbf { } \mathbf } { } { \mathbf } { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf  { \mathbf \mathbf { } } { \mathbf } { } \mathbf { \mathbf } { \mathbf } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf  { \mathbf \mathbf { } \mathbf } { \mathbf \mathbf } $ (with ε denoting the empty string). It is then easy to see that $Z _ { \pi _ { \mathrm { H } } }$ is a valid normalizing constant:

$$
\sum _ { c ^ { \prime } \in \Sigma ^ { * } } \pi _ { \mathrm { H } } ( c ^ { \prime } ) = \sum _ { c ^ { \prime } \in \Sigma ^ { * } } \frac { \sum _ { u \in \Sigma ^ { * } } p _ { \mathrm { H } } ( c ^ { \prime } u ) } { \sum _ { c \in \Sigma ^ { * } } \sum _ { u ^ { \prime } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( c u ^ { \prime } ) }\tag{19a}
$$

$$
= \frac { \sum _ { c ^ { \prime } \in \Sigma ^ { * } } \sum _ { \pmb { u } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( c ^ { \prime } \pmb { u } ) } { \sum _ { c \in \Sigma ^ { * } } \sum _ { \pmb { u ^ { \prime } } \in \Sigma ^ { * } } p _ { \mathrm { H } } ( c \pmb { u ^ { \prime } } ) }\tag{19b}
$$

(19c)

## B Linear Models

## B.1 Rewriting Linear Models with Surprisal in Terms of PMI

In this section, we demonstrate the equivalence between two linear models, one with surprisal and frequency as predictors, and the other with PMI and frequency as predictors. In particular, we consider a linear model of reading times $f _ { \beta } \left( \mathbf { \partial } \cdot | \mathbf { \partial X } ( c , \overline { { u } } ) \right)$ , in which ${ \bf X } ( c , \overline { { u } } ) \colon \Sigma ^ { * } \times \overline { { \Sigma } }  \mathbb { R } ^ { D }$ includes surprisal, frequency and additional baseline predictors $\mathbf { X _ { b } } ( c , \overline { { u } } ) \colon \Sigma ^ { * } \times \overline { { \Sigma } }  \mathbb { R } ^ { D - 3 }$ and show that it is equivalent to a model in which surprisal is replaced with PMI. Consider the prediction ${ \widehat { r } } ( c , { \overline { { u } } } )$ which is the expected value of $f _ { \beta } \left( \mathbf { \partial } \cdot | \mathbf { \partial X } ( c , \overline { { u } } ) \right)$ b. To demonstrate the equivalence, we simply add and subtract an additional frequency term:

$$
\boldsymbol { \widehat { r } } ( \mathbf { c } , \boldsymbol { \overline { { u } } } ) = \beta _ { 0 } + \beta _ { v _ { \mathrm { H } } } v _ { \mathrm { H } } ( \boldsymbol { \overline { { u } } } ) + \beta _ { { \iota _ { \mathrm { H } } } } { \iota _ { \mathrm { H } } } ( \mathbf { c } , \boldsymbol { \overline { { u } } } ) + \beta _ { \mathrm { b } } ^ { \intercal } \mathbf { X } _ { \mathrm { b } } ( \mathbf { c } , \boldsymbol { \overline { { u } } } )\tag{20a}
$$

$$
= \beta _ { 0 } + ( \beta _ { v _ { \mathrm { H } } } + \beta _ { v _ { \mathrm { H } } } ) v _ { \mathrm { H } } ( \overline { { u } } ) - \beta _ { { \iota } _ { \mathrm { H } } } ( v _ { \mathrm { H } } ( \overline { { u } } ) - { \iota } _ { \mathrm { H } } ( c , \overline { { u } } ) ) + \beta _ { \mathrm { b } } ^ { \intercal } \mathbf { X } _ { \mathrm { b } } ( c , \overline { { u } } )\tag{20b}
$$

$$
\begin{array} { r } { = \beta _ { 0 } + ( \beta _ { v _ { \mathrm { H } } } + \beta _ { v _ { \mathrm { H } } } ) v _ { \mathrm { H } } ( \overline { { u } } ) - \beta _ { v _ { \mathrm { H } } } \mu _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) + \beta _ { \mathrm { b } } ^ { \intercal } \mathbf { X } _ { \mathrm { b } } ( \pmb { c } , \overline { { u } } ) , } \end{array}\tag{20c}
$$

where $\beta = [ \beta _ { 0 } , \beta _ { v _ { \mathrm { H } } } , \beta _ { \iota _ { \mathrm { H } } } , \beta _ { \mathrm { b } } ] \in \mathbb { R } ^ { D }$

## B.2 The Orthogonalized Surprisal Model

Under a linear model of reading time which includes orthogonal surprisal, frequency and (possibly) other predictors $\mathbf { X } _ { \mathrm { b } } ( \pmb { c } , \overline { { u } } )$ , reading time predictions are obtained as follows:

$$
\hat { r } ( c , \overline { { u } } ) = \beta _ { 0 } + \beta _ { t _ { \mathrm { H } } } \left( { \boldsymbol { \iota } } _ { \mathrm { H } } ( c , \overline { { u } } ) - \frac { \mathrm { C o v } ( \mathbf { I } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } ) } { \mathrm { C o v } ( \mathbf { Y } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } ) } { \boldsymbol { v } } _ { \mathrm { H } } ( \overline { { u } } ) \right) + \beta _ { v _ { \mathrm { H } } } v _ { \mathrm { H } } ( \overline { { u } } ) + \beta _ { \mathrm { b } } ^ { \intercal } \mathbf { X } _ { \mathrm { b } } ( c , \overline { { u } } )\tag{21a}
$$

$$
= \beta _ { 0 } + \beta _ { \iota _ { \mathrm { H } } } \iota _ { \mathrm { H } } ( \boldsymbol { c } , \overline { { \boldsymbol { u } } } ) + \left( \beta _ { v _ { \mathrm { H } } } - \beta _ { \iota _ { \mathrm { H } } } \frac { \mathrm { C o v ( \mathbf { I } _ { H } , \mathbf { Y } _ { H } ) } } { \mathrm { C o v ( \mathbf { Y } _ { H } , \mathbf { Y } _ { H } ) } } \right) v _ { \mathrm { H } } ( \overline { { \boldsymbol { u } } } ) + \beta _ { \mathrm { b } } ^ { \intercal } \mathbf { X } _ { \mathrm { b } } ( \boldsymbol { c } , \overline { { \boldsymbol { u } } } ) .\tag{21b}
$$

We estimate $\mathrm { C o v } ( \mathbf { I } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } )$ (and $\operatorname { C o v } ( \mathbf { Y } _ { \mathrm { H } } , \mathbf { Y } _ { \mathrm { H } } ) )$ through the unbiased sample covariance on a training corpus ${ \mathcal { C } } \stackrel { \mathrm { d e f } } { = } \{ ( \mathbf { X } ( c _ { n } , { \overline { { u } } } _ { n } ) , r _ { n } ) \} _ { n = 1 } ^ { N }$ :

$$
\frac { 1 } { N - 1 } \sum _ { n = 1 } ^ { N } ( \iota _ { \mathrm { H } } ( \boldsymbol { c } _ { n } , \overline { { u } } _ { n } ) - \hat { \mu } _ { \iota _ { \mathrm { H } } } ) ( v _ { \mathrm { H } } ( \overline { { u } } _ { n } ) - \hat { \mu } _ { v _ { \mathrm { H } } } ) ,\tag{22}
$$

where $\hat { \mu } _ { \iota _ { \mathrm { H } } }$ and $\hat { \mu } _ { v _ { \mathrm { H } } }$ are the sample means computed over for surprisal and frequency, respectively. In words, this means that we only use the word–context pairs present in the training data for approximating the inner products in Eq. (14). Recall that is infinite-dimensional, and we are not aware of efficient algorithms for exact computations of inner products on .

## C Further Technical Details on the Hilbert Space

In this section we provide formal guarantees for the method presented in §3. In App. C.1, we justify why the Hilbert space introduced in §3 is indeed a Hilbert space. In App. C.2, we show that the projection in Eq. (14) yields a predictor which is orthogonal to frequency.

## C.1 Predictor Variables as Elements of a Hilbert Space

Let $\bigl ( \Sigma ^ { * } \times \overline { { \Sigma } } , \mathcal { P } ( \Sigma ^ { * } \times \overline { { \Sigma } } ) , \pi _ { \mathrm { H } } \cdot p _ { \mathrm { H } } \bigr )$ be a probability space. We further require that $\pi _ { \mathrm { H } } ( \pmb { c } ) p _ { \mathrm { H } } ( \overline { { \ b { u } } } \ | \ \pmb { c } ) > 0$ for all $c \in \Sigma ^ { * }$ and $u \in \Sigma$ , which is equivalent to the assumption that $p _ { \mathrm { H } } ( \pmb { u } ) > 0$ for all $\ b u \in \Sigma ^ { * }$ by Eqs. (3) and (4). We construct a Hilbert space  over R of all random variables of type $\mathbf { X } \colon \Sigma ^ { * } \times \overline { { \Sigma } } \to \mathbb { R }$ with the restriction that $\mathbb { E } [ { \mathbf { X } } ^ { 2 } ] < \infty$ . We require the second moment to be finite since / R. In this paper, we are particularly interested in the random variables $\mathbf { I } _ { \mathrm { H } } ( { \pmb { c } } , { \overline { { u } } } ) = \iota _ { \mathrm { H } } ( { \pmb { c } } , { \overline { { u } } } ) , \mathbf { M } _ { \mathrm { H } } ( { \pmb { c } } , { \overline { { u } } } ) = \mu _ { \mathrm { H } } ( { \pmb { c } } , { \overline { { u } } } )$ $\mathbf { Y } _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) = v _ { \mathrm { H } } ( \overline { { u } } )$ , for $c \in \Sigma ^ { * }$ and $\overline { { u } } \in \overline { { \Sigma } }$ . They encode the distributions over surprisal, frequency and PMI, respectively, and are all distributed according to $\pi _ { \mathrm { H } } \cdot p _ { \mathrm { H } }$ . We have

$$
\mathbb { P } ( \mathbf { I } _ { \mathrm { H } } = \iota ) = \sum _ { \pmb { c } \in \Sigma ^ { * } } \pi _ { \mathrm { H } } ( \pmb { c } ) p _ { \mathrm { H } } ( \overline { { u } } \mid \pmb { c } ) \mathbb { 1 } \{ \iota = \iota _ { \mathrm { H } } ( \pmb { c } , \overline { { u } } ) \}\tag{23a}
$$

$$
\mathbb { P } ( \mathbf { Y } _ { \mathrm { H } } = v ) = \sum _ { \pmb { c } \in \Sigma ^ { * } \atop \overline { { \mathfrak { u } } } \in \overline { { \Sigma } } } \pi _ { \mathrm { H } } ( \pmb { c } ) p _ { \mathrm { H } } ( \overline { { u } } \mid \pmb { c } ) \mathbb { 1 } \left\{ v = v _ { \mathrm { H } } ( \overline { { u } } ) \right\}\tag{23b}
$$

$$
\mathbb { P } ( \mathbf { M } _ { \mathrm { H } } = \mu ) = \sum _ { \pmb { c } \in \Sigma ^ { * } , \atop \overline { { \boldsymbol { u } } } \in \Sigma } \pi _ { \mathrm { H } } ( \pmb { c } ) p _ { \mathrm { H } } ( \overline { { \boldsymbol { u } } } \mid \pmb { c } ) \mathbb { 1 } \{ \mu = \mu _ { \mathrm { H } } ( \pmb { c } , \overline { { \boldsymbol { u } } } ) \} .\tag{23c}
$$

We define the following inner product:

$$
\langle \mathbf { X } , \mathbf { Y } \rangle \stackrel { \mathrm { d e f } } { = } \mathbb { E } \left[ \mathbf { X } \mathbf { Y } \right]  \\  = \sum _ { \stackrel { c \in \Sigma ^ { * } } { \overline { { u } } \in \overline { { \Sigma } } ^ { * } } } \pi _ { \mathrm { H } } ( c ) p _ { \mathrm { H } } ( \overline { { u } } \mid c ) \mathbf { X } ( c , \overline { { u } } ) \mathbf { Y } ( c , \overline { { u } } ) .\tag{24a}
$$

(24b)

Consequently, the norm is given by $| | \cdot | | _ { \mathcal { H } } \overset { \mathrm { d e f } } { = } \sqrt { \langle \cdot , \cdot \rangle }$ and the distance metric between two random variables X and Y is $d ( \mathbf { X } , \mathbf { Y } ) \stackrel { \mathrm { d e f } } { = } | | \mathbf { X } - \mathbf { Y } | | _ { \mathcal { H } } . ^ { 1 1 }$ With this choice of inner product, is a Hilbert space. That is because is the $L ^ { 2 } ( \Sigma ^ { * } \times \overline { { \Sigma } } )$ space, and any such $L ^ { 2 }$ space is a Hilbert space (Rudin, 1987, p. 78).

## C.2 Orthogonal Projection

In this section, we use the Hilbert projection theorem to show that frequency and orthogonalized surprisal are disentangled, i.e., that $\langle \mathbf { Y } _ { \mathrm { H } } , \mathrm { p r o j } _ { \mathbf { Y } _ { \mathrm { H } } ^ { \perp } } ( \mathbf { I } _ { \mathrm { H } } ) \rangle = 0$ . We introduce the Hilbert projection theorem in Prop. 2 and apply it to our technique in Prop. 3.

Proposition 2 (Hilbert Projection Theorem). Let be a Hilbert space and $\mathcal { C } \subseteq \mathcal { H }$ be a nonempty closed convex set. For every $\mathbf { X } \in { \mathcal { H } }$ there exists a unique $| | \mathbf { X } - \mathbf { Y } ^ { \prime } | | _ { \mathcal { H } }$ which is equal to

$$
\operatorname* { i n f } _ { \mathbf { Y } \in { \mathcal { C } } } | | \mathbf { X } - \mathbf { Y } | | _ { \mathcal { H } } .\tag{25}
$$

$_ { I f } \mathcal { C }$ is a closed vector subspace of , then $\mathbf { X } - \mathbf { Y } ^ { \prime }$ is orthogonal to ${ \mathcal { C } } ,$ , i.e., $\left. \mathbf { X } - \mathbf { Y } ^ { \prime } , \mathbf { Y } \right. = 0 f o r$ all $\mathbf { Y } \in { \mathcal { C } } .$

Proof. See Rudin (1991, pp. 306-9).

Proposition 3 (Decorrelated Predictors). Let  be the Hilbert space introduced in $\ S 3$ and App. C.1. Further, consider X, $\mathbf { Z } \in \mathcal { H }$ and define

$$
\operatorname { p r o j } _ { \mathbf { Z } ^ { \perp } } ( \mathbf { X } ) \overset { \mathrm { d e f } } { = } \mathbf { X } - \frac { \langle \mathbf { X } , \mathbf { Z } \rangle } { \langle \mathbf { Z } , \mathbf { Z } \rangle } \mathbf { Z } .\tag{26}
$$

Then,

$$
\langle \mathrm { p r o j } _ { \mathbf { Z } ^ { \perp } } ( \mathbf { X } ) , \mathbf { Z } \rangle = 0 .\tag{27}
$$

Proof. Let $\mathcal { C }$ be the set of vectors spanned by Z, i.e., ${ \mathcal { C } } \ { \stackrel { \mathrm { d e f } } { = } } \ \{ a \mathbf { Z } \ | \ a \in \mathbb { R } \}$ . It is easy to see that $\mathcal { C }$ is a closed subspace of . Then, by Prop. 2, there exists a $\mathbf { Y ^ { \prime } }$ such that

$$
\operatorname* { i n f } _ { \mathbf { Y } \in \mathcal { C } } | | \mathbf { X } - \mathbf { Y } | | _ { \mathcal { H } } = | | \mathbf { X } - \mathbf { Y } ^ { \prime } | | _ { \mathcal { H } } .\tag{28}
$$

By the definition of ${ \mathcal { C } } ,$ we have $\mathbf { Y } ^ { \prime } = a ^ { \prime } \mathbf { Z }$ for some $a ^ { \prime } \in \mathbb { R }$ . Because the infimum is achieved, to determine $a ^ { \prime } .$ we consider the following convex optimization problem:

$$
\begin{array} { r l } & { \underset { a \in \mathbb { R } } { \operatorname { a r g m i n } } \ : | | \mathbf { X } - a \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } = \underset { a \in \mathbb { R } } { \operatorname { a r g m i n } } \ : | | a \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } - 2 \langle a \mathbf { Z } , \mathbf { X } \rangle + | | \mathbf { X } | | _ { \mathcal { H } } ^ { 2 } } \\ & { \qquad = \underset { a \in \mathbb { R } } { \operatorname { a r g m i n } } a ^ { 2 } | | \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } - 2 a \langle \mathbf { Z } , \mathbf { X } \rangle . } \end{array}\tag{29a}
$$

(29b)

Because Eq. (29b) is differentiable in a, we check the first-order optimality conditions

$$
2 a | | \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } - 2 \langle \mathbf { Z } , \mathbf { X } \rangle = 0 ,\tag{30}
$$

so

$$
\underset { a \in \mathbb { R } } { \operatorname { a r g m i n } } a ^ { 2 } | | \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } - 2 a \langle \mathbf { Z } , \mathbf { X } \rangle = \frac { \langle \mathbf { Z } , \mathbf { X } \rangle } { | | \mathbf { Z } | | _ { \mathcal { H } } ^ { 2 } }\tag{31a}
$$

$$
\mathbf { \Omega } = { \frac { \langle \mathbf { Z } , \mathbf { X } \rangle } { \langle \mathbf { Z } , \mathbf { Z } \rangle } } .\tag{31b}
$$

We note that this solution is unique because Eq. (29b) is convex in $a .$ Thus, observing by Prop. 2 that

$$
\left. \mathbf { X } - \frac { \left. \mathbf { Z } , \mathbf { X } \right. } { \left. \mathbf { Z } , \mathbf { Z } \right. } \mathbf { Z } , \mathbf { Z } \right. = 0 ,\tag{32}
$$

we have the desired result that $\langle \mathrm { p r o j } _ { \mathbf { Z } ^ { \perp } } ( \mathbf { X } ) , \mathbf { Z } \rangle = 0 .$

## D Our Method in Relation to Residualization

The technique presented in §3 closely resembles another method used to decorrelate predictors— residualization (see, e.g., Kuperman et al., 2008; Jaeger, 2010; García et al., 2019). Consider a linear regression setting with response $\mathbf { y } \in \mathbb { R } ^ { N }$ and design matrix $\mathbf { X } \in \mathbb { R } ^ { N \times 2 }$ , i.e., N data points for two predictors ${ \bf x } _ { 1 } , { \bf x } _ { 2 } \in \mathbb { R } ^ { N }$ , being column vectors. The idea behind residualization is to decorrelate the predictors by replacing one of them, say $\mathbf { x } _ { 1 }$ , by the residuals obtained from the ordinary least squares solution of the regression model in which $\mathbf { x } _ { 1 }$ is the response and $\mathbf { x } _ { 2 }$ is the (only) predictor. The new predictor will thus take the value

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { 1 - r e s } } = \mathbf { x } _ { 1 } - ( \hat { \beta } _ { 0 } ^ { \mathrm { o l s } } + \hat { \beta } _ { 1 } ^ { \mathrm { o l s } } \mathbf { x } _ { 2 } ) , } \end{array}\tag{33}
$$

where $\hat { \beta } _ { 0 } ^ { \mathrm { o l s } } , \hat { \beta } _ { 1 } ^ { \mathrm { o l s } } \in \mathbb { R }$ are the ordinary least squares estimates for the intercept and slope, respectively, when regressing $\mathbf { x } _ { 1 }$ against $\mathbf { x } _ { 2 }$ . Note that $\hat { \beta } _ { 0 } ^ { \mathrm { { o l s } } } = 0$ and $\begin{array} { r } { \hat { \beta } _ { 1 } ^ { \mathrm { o l s } } = \frac { \mathbf { x } _ { 1 } \top \mathbf { x } _ { 2 } } { \mathbf { x } _ { 2 } \top \mathbf { x } _ { 2 } } } \end{array}$ under mean centering. In that case, Eq. (33) mirrors Eq. (14):

$$
\mathbf { x } _ { \mathrm { 1 - r e s } } = \mathbf { x } _ { 1 } - { \frac { \mathbf { x } _ { 1 } \mathsf { ^ { T } } \mathbf { x } _ { 2 } } { \mathbf { x } _ { 2 } \mathsf { ^ { T } } \mathbf { x } _ { 2 } } } \mathbf { x } _ { 2 } .\tag{34}
$$

In Eq. (14) we had the inner product between two vectors in the Hilbert space $\mathcal { H } .$ , but here we have the inner product between two vectors in Euclidean space (x and x ), which is the dot product. Indeed, residualization is defined only over a sample of data points. Our technique can thus be viewed as a functional generalization of residualization. From a statistical perspective, Eq. (34) provides estimates of residual values which can, ideally, be generalized to data beyond the sample of N data points from which they were estimated. In a Hilbert space, on the other hand, there is no question of generalization. For the predictors we use, which are derived from a language model, we know their true values. We only approximate the inner products in Eq. (14) since computing their exact value would be intractable. That is, the reason for our approximation, which yields the same formula as residualization, is computational, rather than statistical.

Arguments Against Residualization. Residualization has received criticism as a way to obtain more interpretable model coefficients (York, 2012; Wurm and Fisicaro, 2014). Consider the following three linear regression models estimated by ordinary least squares:

$$
\begin{array} { r l } { \mathrm { \bf { M o d e l } } \mathrm { \bf { A } } \mathrm { : } } & { { } \mathrm { \bf { y } } = \beta _ { 0 } ^ { A } + \beta _ { 1 } ^ { A } \mathrm { \bf { x } } _ { 1 } + \beta _ { 2 } ^ { A } \mathrm { \bf { x } } _ { 2 } + \varepsilon } \end{array}\tag{35}
$$

$$
\mathbf { M o d e l } \mathbf { B } \colon \mathbf { \mu } _ { \mathbf { y } } = \beta _ { 0 } ^ { B } + \beta _ { 1 } ^ { B } \mathbf { x } _ { 1 \cdot \mathrm { r e s } } + \beta _ { 2 } ^ { B } \mathbf { x } _ { 2 } + \varepsilon\tag{36}
$$

$$
\mathrm { M o d e l } \mathrm { C } \colon \mathrm { ~ \bf ~ y = } \beta _ { 0 } ^ { C } + \beta _ { 2 } ^ { C } { \bf x } _ { 2 } + \varepsilon ,\tag{37}
$$

where $ { \varepsilon } \in \mathbb { R } ^ { N }$ is a vector of Gaussian noise variables. In the models above, we have that $\beta _ { 1 } ^ { A } = \beta _ { 1 } ^ { B }$ and $\beta _ { 2 } ^ { A } \neq \beta _ { 2 } ^ { B } = \beta _ { 2 } ^ { C }$ (Wurm and Fisicaro, 2014). That is, the effect of the residualized predictor— $\mathbf { - x } _ { 1 }$ in the example above—remains the same after residualization (Model A vs. Model B). On the other hand, the estimated effect of the residualizing predictor on the response—x in the example above—changes between Model B which regresses y $\mathbf { 0 n x _ { \mathrm { 1 - r e s } } }$ and $\mathbf { x } _ { 2 }$ and Model $\mathbf { A }$ which regresses $\textbf { y } 0 \mathbf { n } \ : \mathbf { x } _ { 1 }$ and $\mathbf { x } _ { 2 }$ . The estimated effect of $\mathbf { x } _ { 2 }$ in Model B instead becomes equal to what it would have been under a model with a single predictor, regressing only on $\mathbf { x } _ { 2 }$ (Model C). These outcomes are contrary to what one might expect: The effect of the modified, residualized predictor $\mathbf { x } _ { 1 }$ is the same while the effect of the untouched predictor $\mathbf { x } _ { 2 }$ changes. This has indeed been a source of confusion in the literature (Wurm and Fisicaro, 2014). However, in our case, the fact that $\beta _ { 2 } ^ { A } \neq \beta _ { 2 } ^ { B } = \beta _ { 2 } ^ { C }$ is actually the desired consequence: We sought to estimate the effect of context that is not correlated with frequency. In other words, we want the covariance between frequency and surprisal to be attributed to frequency, as it would be in a model that only regresses reading time on frequency (corresponding to Model C). Moreover, we argue that the estimated coefficient is the wrong quantity to look at when measuring importance—it is the role of the predictor in relation to the others that should matter. While $\beta _ { 1 } ^ { A } = \beta _ { 1 } ^ { \bar { B } }$ does not indicate a difference in importance, a measure of explained variance like LMG does, as we demonstrate by our results in Fig. 1.

We thus advocate for the use of such a measure instead of analyzing coefficients. Another remark relates to whether a residualized variable is interpretable as anything at all. For instance, Breaugh (2006) gives an example of height and weight of basketball players: Residualizing height with respect to weight would result in a residualized predictor that would involve a notion of height disentangled from weight, which is tricky to conceptualize in a real-world context. However, in our case, we are addressing the question of what predictor should be extracted in the first place, i.e., from a language model as a stand-in for context. Our paper argued that orthogonalized surprisal gives a better interpretation for the effect of context than contextual surprisal does.

We hope that our work and this discussion can help shed further light on when residualization is, and is not, suitable.

## E Dataset and Predictor Variables

## E.1 Dataset

We use the Multilingual Eye-movement Corpus (MECO; Siegelman et al., 2022), which consists of eye-tracking-based reading-time data across 13 different languages for 12 Wikipedia-style articles about various topics. The articles have been carefully constructed to contain the same content across languages. Word-level reading time is recorded for between 32 54 participants per language, using several different reading variables. Of these, for our main experiments, we use gaze duration, which is the total fixation time on a word during its first pass (i.e., before the first time the gaze leaves the word). However, below we also show results for two other reading metrics: first fixation duration, which is the duration of the first fixation that lands on the word, and total fixation duration, which is the sum over all fixation durations of a word. In our experiments, we give a reading time of zero to words that were skipped on the first pass. We take the average over the by-subject reading times to obtain the response variables we model.

## E.2 Predictors

Our predictor variables are estimated in the following way: Surprisal estimates are obtained from mGPT (Shliazhko et al., 2024), which is a multilingual variant of GPT-3 that was trained on Wikipedia and the C4 corpus (Raffel et al., 2020). It supports 61 languages, which intersected with the MECO dataset yields 11 languages for our experiments: Dutch, English, Finnish, German, Greek, Hebrew, Italian, Korean, Russian, Spanish, and Turkish. Estonian and Norwegian, which are present in MECO, are unfortunately not supported by mGPT. For each word in MECO, we sum the surprisals estimated by mGPT for each of the tokens that make up that word. We use the estimates from Speer (2022) to obtain word frequency (i.e., unigram surprisal) and length, following previous work. Finally, PMI estimates are obtained from surprisal and frequency through Eq. (10b). All predictors are standardized (i.e., set to mean zero and standard deviation one) before computing orthogonalized surprisal values and fitting the regression models.

## F Additional Experiments

We complement the main experiments presented in §4 with three additional empirical analyses. In App. F.1, we take an alternative position to the one in the main text, using an orthogonalizedfrequency predictor. In App. F.2, we exclude the length predictor and perform the same analysis as we did in the main text. In App. F.3, we compare the predictive power of nonlinear models across different sets of predictors.

## F.1 Orthogonalized Frequency

Here, we provide an additional analysis in which we derive a new frequency predictor, swapping the two variables in Eq. (14). In this case, the shared effect of frequency and surprisal on reading time is attributed to surprisal, and the frequency effect represents the effect beyond what is already explained by surprisal. We present the results in Fig. 2. Comparing this analysis to Fig. 1 may be considered analogous to Shain’s (2019) study, which compares the independent effects of surprisal and frequency by adding them in as predictors on top of baseline models that contain the other. However, in contrast to Shain (2019), we find that frequency appears to be more important than contextual effects in explaining reading times.

![](images/20144c03c547cff6e1464dbb8d5f6371dd55b4801851c088cae01d620a7556e0.jpg)  
Figure 2: This figure is analogous to Fig. 1, with the difference that frequency is projected onto the orthogonal complement of surprisal, resulting in an orthogonalized frequency predictor. Even when the shared effect of frequency and surprisal is attributed to surprisal, we still find that the frequency predictor explains more variance than surprisal for most languages.

## F.2 Results without Length

We complement the experiments in §4 with an additional analysis which excludes length. We follow the same experimental setup as described in the main portion of the text, this time with three linear models that include predictors: (i) surprisal and frequency, (ii) PMI and frequency, and (iii) orthogonalized surprisal and frequency. We include spillover variables from the previous word. The results are shown in Fig. 3. We observe that the implications on context found in Fig. 1 do not change when excluding length.

## F.3 Psychometric Predictive Power

While the surprisal and PMI models are equivalent under a linear model, that relationship does not necessarily need to hold under a nonlinear one. Therefore, we compare the psychometric predictive power by fitting generalized additive models (GAMs). GAMs are a class of additive models that can learn non-linear relationships between predictor and response variables. All the terms in our GAMs are spline-based smooth terms, that include either a contextual predictor variable (i.e., surprisal, PMI, or orthogonalized surprisal), or frequency. We restrict our smooth terms to six basis functions, following the logic outlined in (Hoover et al., 2023). GAMs are fit using the mgcv package in R. Two example calls are given below:

```python
gam(reading_time ~ s(surprisal, bs = 'cr', k = 6) + s(prev_surprisal, bs = 'cr', k = 6), data = .)
```

```python
gam(reading_time ~ s(pmi, bs = 'cr', k = 6) + s(prev_pmi, bs = 'cr', k = 6) + s(frequency, bs =
'cr', k = 6) + s(prev_frequency, bs = 'cr', k = 6), data = .)
```

We consider an additional baseline model which is the average reading time estimated from the training set. We compare delta log-likelihood $\Delta _ { \mathrm { l l h } }$ —the average difference in likelihood between the target models mentioned above and the baseline model—as estimated over ten folds of cross-validation across several different sets of predictor variables.

Results are visualized in Fig. 4. We observe three big trends: First, we find that all predictors lead to positive $\Delta _ { \mathrm { l l h } }$ , indicating that they are useful for predicting reading times. However, second, when looking at models with just one predictor variable, we observe that frequency alone leads to higher $\Delta _ { \mathrm { l l h } }$ than any other single variable, and that surprisal and PMI tend to result in higher $\Delta _ { \mathrm { l l h } }$ than orthogonalized surprisal. This is to be expected. We know from prior research that frequency plays a large role in explaining by-word processing effort, and because orthogonalized variables, by definition, are decorrelated with frequency, they are not expected to be strong predictors of reading times, alone.

![](images/06173815439cb5a14a9397e785ec3a4e792fb47669ce72e3ed375278151e33ac.jpg)  
Figure 3: This figure is analogous to Fig. 1, except that it shows results when excluding the word length predictor.

The right three facets of Fig. 4 show models that combine contextual and non-contextual predictors into a single model. Here, we observe only insignificant, nearly invisible differences between the models’ $\Delta _ { \mathrm { l l h } }$ . We conclude that all three implementations of context are equally good at predicting reading times.

In Fig. 5, we show our generalized additive modeling results, broken down by language, across the x-facets. We also show results for reading time metrics other than gaze duration, including first fixation duration (top row) and total reading times (bottom row). These results are consistent with those reported in Fig. 4. We find that of the four individual predictors, frequency leads to the highest $\Delta _ { \mathrm { l l h } }$ , followed by surprisal, PMI, and then orthogonalized variants. When combining our non-contextual predictor (frequency), alongside these contextual predictors, we do not observe differences in $\Delta _ { \mathrm { l l h } }$

## G Connection with Model Size

The results presented in this article may help explain a trend recently observed in the computational psycholinguistics literature: Surprisal values of larger LMs provide a worse fit to human reading-time data compared to those of medium-sized models (Oh and Schuler, 2023a,b). Specifically, Oh et al. (2024) suggest that this is because larger models are incredibly accurate at predicting rare words in context. Medium-sized models, on the other hand, are not as good at predicting rare words in context. Therefore, surprisal estimates for these words are closer to their unigram frequencies, i.e., non-contextual surprisal. If reading times are primarily driven by frequency effects, as suggested by our analysis, the surprisal predictor should—on its own—yield stronger predictive power if it is closer to frequency, as is the case for medium-sized models. Thus, this could explain why the decoupling of surprisal and frequency in these larger models results in poorer fits to human reading times.

Similarly, this could be a reason for why surprisal estimates derived from lossy contexts have been shown to be more predictive of reading times (Futrell et al., 2020; Kuribayashi et al., 2022): Restricting the context might make the surprisal estimates more similar to unigram frequencies.

![](images/7b6f410aa99a4cb5edaf9e1da05263710faef70b13660453337d2b41880b63aa.jpg)  
Figure 4: $\Delta _ { \mathrm { l l h } } \mathrm { o f } \mathrm { G A M }$ models. Models include only the indicated predictors. Error bars are 95% confidence intervals across folds of data.

![](images/f6c22e8ac6dd3abbc537ea41bc29a2e35a7318d87342af2f0deeb86e4a3013d9.jpg)  
Figure 5: Psychometric Predictive Power by Language. Results are consistent those reported in Fig. 4.