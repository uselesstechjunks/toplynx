#######################################################################
Labeling and Learning Strategies
#######################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***********************************************************************
Resources
***********************************************************************
- [lilianweng.github.io] `Generalized Language Models <https://lilianweng.github.io/posts/2019-01-31-lm/>`_
- [lilianweng.github.io] `Learning with not Enough Data Part 1: Semi-Supervised Learning <https://lilianweng.github.io/posts/2021-12-05-semi-supervised/>`_
- [lilianweng.github.io] `Learning with not Enough Data Part 2: Active Learning <https://lilianweng.github.io/posts/2022-02-20-active-learning/>`_
- [lilianweng.github.io] `Learning with not Enough Data Part 3: Data Generation <https://lilianweng.github.io/posts/2022-04-15-data-gen/>`_ 
- [lilianweng.github.io] `Thinking about High-Quality Human Data <https://lilianweng.github.io/posts/2024-02-05-human-data-quality/>`_
- [ruder.io] `An overview of proxy-label approaches for semi-supervised learning <https://www.ruder.io/semi-supervised/>`_
- [sh-tsang.medium.com] `Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks <https://sh-tsang.medium.com/review-pseudo-label-the-simple-and-efficient-semi-supervised-learning-method-for-deep-neural-aa11b424ac29>`_
- [sh-tsang.medium.com] `Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results <https://sh-tsang.medium.com/review-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-b245d5efa5bf>`_
- [sh-tsang.medium.com] `WSL: Exploring the Limits of Weakly Supervised Pretraining <https://sh-tsang.medium.com/review-wsl-exploring-the-limits-of-weakly-supervised-pretraining-565ff66e0922>`_
- [sh-tsang.medium.com] `Billion-Scale Semi-Supervised Learning for Image Classification <https://sh-tsang.medium.com/review-billion-scale-semi-supervised-learning-for-image-classification-801bb2caa6ce>`_
- [reachsumit.com] `Positive and Negative Sampling Strategies for Representation Learning in Semantic Search <https://blog.reachsumit.com/posts/2023/03/pairing-for-representation/>`_
- [medium.com] `Hard Negative Mining in Nature Language Processing <https://medium.com/@dunnzhang0/hard-negative-mining-in-nature-language-processing-how-to-select-negative-examples-in-66f59da316a4>`_
- [stanford.edu] `Weak Supervision: A New Programming Paradigm for Machine Learning <https://ai.stanford.edu/blog/weak-supervision/>`_
- [ibm.com] `What is knowledge distillation? <https://www.ibm.com/think/topics/knowledge-distillation>`_

***********************************************************************
Label Design
***********************************************************************
1. Class imbalance
=======================================================================
1. Data Augmentation
-----------------------------------------------------------------------
#. Positive Class Upsampling
#. Negative Class Downsampling
#. Hard Negative Mining

	- Instead of uniform downsampling, select high-confidence false negatives (items that nearly got engagement).  
	- Example: For Amazon product recommendations, products that users hovered over but didn’t click could be treated as hard negatives instead of ignored.
	- [towardsdatascience.com] `Two-Tower Networks and Negative Sampling in Recommender Systems <https://towardsdatascience.com/two-tower-networks-and-negative-sampling-in-recommender-systems-fdc88411601b/>`_
	- Paper: On the Theories Behind Hard Negative Sampling for Recommendation
	- Paper: Enhanced Bayesian Personalized Ranking for Robust Hard Negative Sampling in Recommender Systems
#. Synthetic Data  

	- Generate synthetic training data when positive signals are rare.  
	- Example: TikTok might use GAN-based augmentation to create synthetic engagement samples for new videos that lack enough clicks. 

2. Loss weights
-----------------------------------------------------------------------
#. Inverse Propensity Weighting (IPW):  

	- Assign different weights to samples based on their rarity.  
	- Example: TikTok ads ranking models give more weight to rare ad clicks to ensure the model doesn’t ignore small but valuable engagements.  
#. Weighted Loss Functions:  

	- Assign a higher loss weight to the minority class.  
	- Example: Facebook News Feed might increase the loss for underrepresented engagement types like "shares" compared to "likes."  
#. Focal Loss (used in detection models but adapted for ranking/recsys):  

	- Down-weighs easily classified negatives and focuses on hard examples.  
	- Example: Google Search may use focal loss to prioritize rare but meaningful query-document relevance labels over common clicks.
	- [towardsdatascience.com] `Focal Loss : A better alternative for Cross-Entropy <https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/>`_

3. Alternative Label Definitions
-----------------------------------------------------------------------
Redefining what counts as a positive interaction to increase robustness  

#. Using Multiple Engagement Signals:  

	- Instead of just clicks, also use dwell time, scroll depth, likes, comments, and shares.  
	- Example: Twitter/X might train ranking models using both retweets and meaningful replies instead of just likes.  
#. Time-Windowed Engagement Labels:  

	- Look at engagement over time instead of at one interaction snapshot.  
	- Example: Google Discover might track whether users return to read a recommended article later, treating it as a positive implicit signal. 
#. [Related] How to address delayed feedback singals - paper

4. Design for the Future: Explore-Exploit Approaches
-----------------------------------------------------------------------
Balancing learning from existing data with discovering new patterns  

- Multi-Armed Bandits (MAB)  

	- Explore new recommendations even if they don’t have past clicks, balancing exploration and exploitation.  
	- Example: Google Ads may intentionally show low-impression ads to collect new engagement signals.  
- Reinforcement Learning (RL)  

	- Train models to maximize long-term engagement instead of just immediate clicks.  
	- Example: YouTube’s recommendation engine uses RL to balance fresh content vs. already popular videos.

5. Extreme Imbalance in Continuous Training
-----------------------------------------------------------------------
#. Uniform Random Sampling with a Dynamic Candidate Pool:  

	- Regularly sample a fixed, manageable subset of negatives from the entire candidate pool. This ensures that you have a diverse set of negatives over time and keeps computational costs predictable.
	- Provides stability and prevents overfitting to a narrow set of negative examples. It's straightforward to implement in an online setting.
- Uniform negatives might be too easy for the ranker and not always challenge the model, potentially leading to slower improvements in discriminative power.

#. Hard Negative Mining (Dynamic Hard Sampling):  

	- Identify hard negatives (i.e., items that the model mistakenly ranks too high or that are very similar to positive examples) during training and focus on these in subsequent updates.
	- Encourages the model to learn finer distinctions and improves ranking performance by pushing the decision boundary closer to the positive examples.
	- Overemphasis on very hard negatives can sometimes introduce noise or instability if they're outliers. It also requires extra computation to identify these examples dynamically.

#. In-Batch Negative Sampling:  

	- Use the negatives from the same mini-batch as the positive examples. This is computationally efficient since you reuse already processed data.
	- Works seamlessly with continuous training pipelines and ensures that negatives are current with the latest model updates.
	- The diversity of negatives is limited to the mini-batch, so it might not capture the full spectrum of negative examples available in the entire dataset.

Recommended Strategy for Continuous Training: 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Combine Uniform Random Sampling with Dynamic Hard Negative Mining:
- Start by uniformly sampling a pool of negatives periodically from the full candidate set. Then, within that pool (or even within each mini-batch), apply a hard negative mining step to select the most challenging negatives based on the current model's predictions.
- This combination provides a stable baseline (uniform sampling) while ensuring that the model is continually pushed to learn from the most informative negative examples (hard negatives). It adapts as the model evolves, which is crucial for continuous training environments.
- The strategy is computationally manageable since you're not processing all negatives at every update. Instead, you maintain a dynamic candidate pool and update it regularly, ensuring that the system scales to large datasets and adapts to changes over time.

Industry Reference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `PinSage (Ying et al., 2018) <https://arxiv.org/abs/1806.01973>`_: Uses sampling techniques to handle billions of nodes in a graph for recommendation while dynamically updating the model, illustrating how to efficiently mine informative negatives in a large-scale system.

- `FAISS (Facebook AI Similarity Search) <https://github.com/facebookresearch/faiss>`_: While primarily for efficient ANN search, FAISS is an example of a system that supports scalable negative sampling in embedding-based retrieval.  

2. Noisy Labels
=======================================================================
1. Label Smoothing 
-----------------------------------------------------------------------
- Instead of using hard labels (e.g., 0 or 1), use smoothed labels (e.g., 0.9 and 0.1) to make the model more robust to noisy labels.

2. Noise Filtering
-----------------------------------------------------------------------
- Human-in-the-loop: Use human feedback to verify or correct labels in the dataset.
- Confidence-based Filtering: Remove samples with low model confidence or high disagreement between multiple annotators.

3. Outlier Detection
-----------------------------------------------------------------------
- Apply algorithms (e.g., Isolation Forest, Z-score method) to detect outliers in the dataset and remove instances with highly suspicious labels.

3. Low Data Regime
=======================================================================
1. Semi Supervised Learning
-----------------------------------------------------------------------
Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [maddevs.io] `Semi-Supervised Learning Explained: Techniques and Real-World Applications <https://maddevs.io/blog/semi-supervised-learning-explained/>`_
* [ruder.io] `An overview of proxy-label approaches for semi-supervised learning <https://www.ruder.io/semi-supervised/>`_
* [ovgu.de][SSL] `Semi-supervised Learning for Stream Recommender Systems <https://kmd.cs.ovgu.de/pub/matuszyk/Semi-supervised-Learning-for-Stream-Recommender-Systems.pdf>`_

2. Active Learning
-----------------------------------------------------------------------
- extension of semi-supervised learning
- determining and choosing high potential unlabelled data that would make the model more efficient
- these data points are labelled and the classifier gains accuracy.

How to detect informative unlabelled data points?

	- Uncertainty : label the samples for which the model is least confident in its predictions.
	- Variety/Diversity : select samples that are as diverse as possible to best cover the entire input space.
	- Model Improvement : select the samples that will improve the performance of the model (lower loss function).

Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- [burrsettles.com] `Active Learning Literature Survey <https://burrsettles.com/pub/settles.activelearning.pdf>`_

4. Proxy Labels
=======================================================================
Weak Supervision
-----------------------------------------------------------------------
* [medium.com] `Weak Supervision — Learn From Less Information <https://npogeant.medium.com/weak-supervision-learn-from-less-information-dcc8fe54e2a5>`_
* [stanford.edu] `Weak Supervision: A New Programming Paradigm for Machine Learning <https://ai.stanford.edu/blog/weak-supervision/>`_

Common Techniques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Rule-based and Programmatic Supervision  

	- Use heuristic logic or rules to assign labels.  
	- Heuristic rules: Label an image as “shoes” if the filename or product title contains “sneakers”, “boots”, etc.  
	- Labeling functions (Snorkel-style): Write small scripts—e.g., if description contains "battery" and "voltage", label as electronics.

#. External Knowledge and Metadata-Based  

	- Leverage structured knowledge or metadata fields.  
	- Distant supervision: Use a knowledge base like Wikidata to map product names to categories.  
	- Metadata-based: Use seller-provided category or brand fields as weak labels.

#. User Interaction and Crowd Signals  

	- Use engagement or crowd behavior as indirect supervision.  
	- Interaction signals: Label clicked items as relevant in search results.  
	- Crowd feedback: Use user reports or moderation flags to label content as inappropriate.

#. Model-Derived or Proxy Labels  

	- Use model outputs or related tasks to generate pseudo-labels.  
	- Pretrained model outputs: Use an off-the-shelf classifier trained on ImageNet to generate initial labels.  
	- Related task transfer: Use a sentiment classifier trained on English tweets to label product reviews in another platform.

#. Similarity and Co-occurrence Methods  

	- Use embedding proximity or co-occurrence patterns.  
	- Embedding similarity: Label an image as similar to another image with known label based on cosine similarity.  
	- Co-occurrence: If products with tag "eco-friendly" often co-occur with “recycled”, assign “eco-friendly” when “recycled” appears.

Learning Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- In Supervised learning (clean labels), it's safe to use techniques like hard negative mining or adversarial training because the labels are (mostly) correct.

	- These techniques help refine decision boundaries, especially useful when the dataset is large and well-labeled, and you want to maximize generalization.
- In weak supervision, we need to avoid premature overfitting to uncertain signals. 

	- Progressive learning, confidence estimation, and robust negative mining (with semantic distance) are safer than early use of sharp boundary techniques.
	- Applying aggressive boundary-pushing techniques (like hard negative mining or adversarial examples) too early can amplify label noise and lead to overfitting on false signals.
	- Instead, the strategy should emphasize:

		- Confidence calibration
		- Noise-robust loss functions
		- Soft labeling and uncertainty modeling
- Suitable learning strategies under weak supervision:

	- Positives:

		- Confidence-based sampling: Only use pseudo labels above a threshold.
		- Agreement filtering: Use ensemble or co-training and pick examples where models agree.
		- Embedding proximity: Treat nearby examples in the embedding space as positives (soft positives).
	- Negatives:

		- In-batch negatives (default in contrastive learning)
		- Distance-aware sampling: Prefer sampling negatives far from the anchor in representation space to avoid false negatives.
		- Curriculum learning: Start with random negatives, gradually move to harder ones only after early convergence.

Data Centric AI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- aims to re work the labels and have models that better understand the data rather than simply relying on pure labels from the dataset.
- new labels are called Weak Labels because they have additional information that does not directly indicate what we want to predict
- also considered as noisy because their distribution has a margin of error.

different types and technique of weak supervision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Incomplete Supervision

	- Semi Supervised Learning, Active Learning and Transfer Learning
	- Data Programming - creating labelling functions to get more labels for the training instance of the model.
#. Inexact Supervision

	- Multi Instance Learning
#. Inaccurate Supervision

	- bad labels are grouped together and corrected with Data Engineering or a better crowdsourcing process.

5. No Labels
=======================================================================
Self-Supervised Learning
-----------------------------------------------------------------------
- Feature Selection: Mutual information, SHAP, correlation-based selection.
- Dealing with Class Imbalance: SMOTE, focal loss, balanced batch sampling.
- Bias and Fairness: Bias detection, de-biasing strategies, fairness-aware training.
- [paper] Self-Supervised Learning for Recommender Systems: A Survey

Label Transformation
=======================================================================
Label Encoding
-----------------------------------------------------------------------
- Convert categorical labels into numerical format (typically used in classification).
- Applicable to: Categorical labels (nominal or ordinal).

Log Transformation (for regression)
-----------------------------------------------------------------------
- Apply a log transformation to skewed labels for regression tasks (e.g., predicting income, house prices).
- Applicable to: Continuous numerical labels.

Binarization
-----------------------------------------------------------------------
- Convert continuous labels into binary values (e.g., thresholding for classification).
- Applicable to: Continuous labels for binary classification.

Common Label Biases
=======================================================================
.. csv-table::
	:header: "Bias", "Description", "Mitigation Strategy", "Trade-offs"
	:align: left
	:widths: 12, 16, 24, 24

		Popularity Bias, Overexposure of already popular items, Re-weighting; downsampling; diversity re-ranking, May lower CTR on high-performing items
		Position/Exposure Bias, Higher-ranked items get more clicks regardless of relevance, IPS; A/B testing; calibration, Requires accurate exposure estimation; added complexity
		Selection Bias, Interactions are not random; users self-select what to see, Counterfactual reasoning; causal inference; multi-signal integration, Increased computational and modeling complexity
		Feedback Loops, Model reinforces its own biases over time, Periodic re-training; re-ranking; diversity promotion, Can sacrifice short-term engagement for long-term diversity
		Presentation Bias, UI design influences clicks, A/B testing; controlled experiments, May require continuous UI evaluation and adjustments

#. Popularity Bias:

	- Items that are already popular receive more exposure, leading to even higher engagement and reinforcing their popularity.
	- Can limit diversity and make it hard for niche or new items to be discovered.

#. Position/Exposure Bias:

	- Items shown at higher ranks or more prominent positions are more likely to be clicked, irrespective of their true relevance.
	- Can skew click-based labels, as users may click simply because an item is highly visible.

#. Selection Bias:

	- The observed interactions (e.g., clicks, ratings) are not a random sample of all potential interactions. Users self-select what they see or engage with, leading to a biased view of user preferences.
	- Results in models that overfit to popular or easily observable behaviors while neglecting latent interests.

#. Feedback Loops:

	- A model that is trained on biased data may perpetuate or exacerbate the bias in subsequent recommendations, creating a cycle that reinforces the existing bias.
	- Can cause a narrowing of recommendations over time, reducing content diversity.

#. Presentation Bias:

	- The design of the user interface (e.g., ad layout, color schemes) can influence user interactions, introducing bias into the labels.
	- May lead to inflated engagement metrics that are artifacts of UI design rather than true user preference.

Industry Practices to Tackle Label Biases
-----------------------------------------------------------------------
#. Inverse Propensity Scoring (IPS):

	- Adjust training samples by weighting them inversely proportional to the probability of an item being shown.
	- Helps counteract exposure and position bias by compensating for items that are under-exposed.
	- Requires an accurate estimation of exposure probabilities; if these are off, IPS can introduce its own errors.
	- Improved fairness vs. potential instability if propensity scores are noisy.

#. Counterfactual Reasoning and Causal Inference:

	- Use causal modeling to distinguish between true user preference and effects caused by presentation bias.
	- Provides a more principled way to correct for selection and exposure biases.
	- Can be computationally complex and require more sophisticated data collection; often needs strong assumptions about the underlying causal structure.
	- More robust correction vs. increased model complexity and data requirements.

#. A/B Testing and Calibration:

	- Regularly run experiments (A/B tests) to assess the effect of different presentation strategies on engagement metrics, and adjust models accordingly.
	- Provides real-world validation and helps isolate bias effects.
	- Can be expensive, time-consuming, and may not capture long-term effects.
	- Empirical feedback vs. slower iteration speed.

#. Re-Ranking and Diversity Promotion:

	- Incorporate re-ranking strategies (e.g., determinantal point processes, diversity constraints) to ensure a mix of items, mitigating popularity and feedback loop biases.
	- Increases content diversity and breaks echo chambers.
	- May sacrifice some immediate relevance or CTR in favor of broader exposure.
	- Higher long-term engagement and discovery vs. potential short-term drop in engagement metrics.

#. Using Hybrid Signals:

	- Combine explicit feedback (e.g., ratings) with implicit signals (e.g., dwell time, scroll depth) and external data (e.g., contextual signals) to reduce reliance on any single biased signal.
	- Helps smooth out biases that might dominate one type of signal.
	- More complex feature engineering and model design; risk of diluting strong signals if not weighted appropriately.
	- Improved robustness vs. increased model complexity.

***********************************************************************
Data and Feature Engineering
***********************************************************************
Feature Transformation
=======================================================================
Scaling and Normalization
-----------------------------------------------------------------------
- Standardization

	- Transform features to have a mean of 0 and standard deviation of 1. 
	- Applicable to: Continuous numerical variables.
- Min-Max Scaling

	- Rescale features to a fixed range (e.g., [0, 1]). 
	- Applicable to: Continuous numerical variables.
- Robust Scaling

	- Use the median and interquartile range (IQR) to scale, robust to outliers. 
	- Applicable to: Continuous numerical variables, especially with outliers.

Log Transformation
-----------------------------------------------------------------------
- Apply logarithmic transformation to reduce skewness in data with large values. 
- Applicable to: Continuous numerical variables with positive skew (e.g., income, population).

Exponential Decay
-----------------------------------------------------------------------
- Applicable to ggregation features
- Assign more weights to recent events

	.. math:: \text{score} = \sum_{i} w_i \cdot \text{event}_i, \quad \text{where } w_i = \exp\left(-\lambda \cdot (t_{\text{now}} - t_i)\right)

Binning and Discretization
-----------------------------------------------------------------------
- Convert continuous variables into categorical bins (e.g., age groups). 
- Applicable to: Continuous numerical variables.

One-Hot Encoding
-----------------------------------------------------------------------
- Convert categorical variables into binary vectors. 
- Applicable to: Categorical variables (nominal).

Ordinal Encoding
-----------------------------------------------------------------------
- Assign integer values to ordered categories. 
- Applicable to: Ordinal categorical variables.

Polynomial Features
-----------------------------------------------------------------------
- Generate polynomial and interaction features to capture non-linear relationships. 
- Applicable to: Continuous numerical variables.

Handling Missing Values
-----------------------------------------------------------------------
- Impute missing values using mean, median, or more sophisticated methods like KNN or model-based imputation.
- Applicable to: Any type of variable with missing data (both continuous and categorical).

***********************************************************************
Dataset Creation and Curation
***********************************************************************
- [mit.edu] `Dataset Creation and Curation <https://dcai.csail.mit.edu/2024/dataset-creation-curation/>`_
- [mit.edu] `Data Curation for LLMs <https://dcai.csail.mit.edu/2024/data-curation-llms/>`_
- Data curation for LLM pretraining

	- https://medium.com/@zolayola/public-data-sets-in-the-era-of-llms-0a4e89bda658
	- [arxiv.org] `Textbooks Are All You Need II: phi-1.5 technical report <https://arxiv.org/pdf/2309.05463>`_
	- [arxiv.org] `A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity <https://arxiv.org/abs/2305.13169>`_

LLMs for data curation
=======================================================================
#. Evaluating llm output data - hallucination, toxicity, bias

	- use a more powerful llm to evaluate

		- effectiveness
		- challenges
	- ** uncertainty quantification

		- [https://arxiv.org/abs/2308.16175] Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness

#. Data curation for llm applications

	- zero shot
	- few shot - [https://aclanthology.org/2023.acl-long.452.pdf] Data Curation Alone Can Stabilize In-context Learning
	- rag
	- sft

		- Humans provide gold input-output pairs
		- Common paradigm: use LLM to generate synthetic data for fine-tuning

			- Goal: train smaller/cheaper LLM to match performance of larger LLM, for specific task
			
		- Generate synthetic data using powerful LLM

			- Using uncertainty quantification, keeping only high-confidence results
			- Filter out bad synthetic data

				- Separately, for inputs and outputs, train a real vs synthetic classifier
				- use classifier scores to toss out unrealistic examples

			- Clean whole dataset (original + synthetic)
			- Fine-tune the LLM on the full dataset
	- Reinforcement learning from human feedback
