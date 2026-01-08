################################################################################
ML Application
################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

********************************************************************************
ML Design Round Framework
********************************************************************************
- [aayushmnit.com] `Meta MLE Interview Preparation Guide <https://aayushmnit.com/posts/2024-12-15-MLInterviewPrep/MLInterviewPrep.html>`_
- [onedrive.live.com] `MLE Interview Prep Template <https://onedrive.live.com/edit?id=52D2FC816A40F7EB!37560&resid=52D2FC816A40F7EB!37560&ithint=file%2Cxlsx&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvcyFBdXYzUUdxQl9OSlNncVU0TGhPVUp0VGNWNDRXalE&migratedtospo=true&wdo=2&cid=52d2fc816a40f7eb>`_
- https://www.youtube.com/watch?v=jkKAeIx7F8c

Basic Structure
================================================================================
* Problem Understanding:

	- Functional Requirements: Identify the key business problem and the KPIs for success.
	- Non-functional Requirements: Ask about the additional requirement such as

		- imposing compliance policies (geographic, demographic)
		- additional desirable features (diversity, context-awareness, ability to 
* Problem Identification:

	- Abstraction: Think about the observed data as :math:`X` and the target as :math:`Y` (can be :math:`X` itself).

		* Identify whether :math:`X` has structure (sequence, locality) or is unstructured.
		* Identify latent variables :math:`Z` if present.
	- Mapping: Identify ML paradigms. If you can't map to of any, create a new ML paradigm for it!
* Scale Identification:

	- Think about the scale and discuss trade-offs for using different types of ML models for that paradigm. 
	- Decide on a scale for the current problem and draw system diagram. Mark the parts involving ML.
* ML cycle for each parts:

	* Working solution:

		- Uses a SOTA/novel technique.
		- Solves at the right scale.
		- Can go live.
	* Various trade-offs:

		- Model choice (e.g. Offline: DNNs/LLMs; Online: LR, GBDT and NN).
		- Loss (e.g. Imbalanced Dataset: weighted/focal loss).
		- Hyperparameter (overfitting; convergence).
		- Metric (e.g. RecSys: NDCG/MAP for PC vs MRR for Mobile; Classification: P, ROC-AUC vs R, PR-AUC).
	* Identify shortcomings:

		- Parts that can be iterated on.

.. csv-table:: 
	:header: Feature, Machine Learning Design, Machine Learning System Design
	:align: left
	
		Focus, Designing the ML model itself, Deploying; scaling; and maintaining the model in a real-world system
		Key Concerns, Algorithm choice; feature engineering; loss function; evaluation metrics, Infrastructure; scalability; latency; monitoring; retraining pipelines
		Primary Goal, Model accuracy and effectiveness, Efficient; reliable; and scalable deployment
		Output, A trained ML model, A full ML-powered system that runs at scale
		Example, Choosing between a CNN and a Transformer for image classification, Building a scalable real-time image recognition system that serves millions of requests per second
********************************************************************************
Machine Learning Design (ML Model Design)
********************************************************************************
- Scope: Focuses on designing the core machine learning model for a specific task.
- ML Design questions are common in research-focused or model-building roles.
Key question prompts live in `docs/source/gen/interviews/qb.rst`.

********************************************************************************
Machine Learning System Design (MLSD)
********************************************************************************
- Scope: Focuses on building the entire system around the ML model, ensuring it runs efficiently at scale in production.
- ML System Design questions are crucial for senior ML engineers, applied scientists, and ML architects who work on large-scale ML deployments.
- Expectations

	- Define the problem clearly (e.g., how TikTok ranks videos).
	- Break down the system (retrieval, ranking, re-ranking, personalization).
	- Discuss infrastructure trade-offs (real-time vs. batch, model updates, caching strategies).
	- Consider monitoring & reliability (A/B testing, detecting drift, rollback strategies).
- During Interview

	- Integrate Business Context: Always start by clarifying the business problem and how ML can address it.
	- Show End-to-End Thinking: From data collection through feature engineering, modeling, evaluation, and deployment.
	- Discuss Trade-offs: Highlight how your choices affect scalability, accuracy, and user experience.
	- Use Real-World Examples: Cite examples like Netflix for homepage recommendations, YouTube for next-item prediction, and Google for search ranking to demonstrate practical understanding.

********************************************************************************
Study Framework
********************************************************************************
Question prompts live in `docs/source/gen/interviews/qb.rst`.

********************************************************************************
Presentation Framework
********************************************************************************
Question prompts live in `docs/source/gen/interviews/qb.rst`.

********************************************************************************
Resources
********************************************************************************
#. Interview Guide

	#. [trybackprop.com] `FAANG Interview – Machine Learning System Design <https://www.trybackprop.com/blog/ml_system_design_interview>`_
	#. [patrickhalina.com] `ML Systems Design Interview Guide <http://patrickhalina.com/posts/ml-systems-design-interview-guide/>`_
	#. [leetcode.com] `Machine Learning System Design : A framework for the interview day <https://leetcode.com/discuss/interview-question/system-design/566057/Machine-Learning-System-Design-%3A-A-framework-for-the-interview-day>`_
	#. [medium.com] `How to Crack Machine learning Interviews at FAANG! <https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5>`_
	#. [medium.com] `Part 2 — How to Crack Machine learning Interviews at FAANG : Pointers for Junior/Senior/Staff+ levels <https://medium.com/@reachpriyaa/part-2-how-to-crack-machine-learning-interviews-at-faang-pointers-for-junior-senior-staff-4b89e10bff28>`_

	#. [stackexchange.com] `Preparing for a Machine Learning Design Interview <https://datascience.stackexchange.com/questions/69981/preparing-for-a-machine-learning-design-interview>`_
	#. [algoexpert.io] `MLExpert <https://www.algoexpert.io/machine-learning/product>`_
#. Resources

	#. Machine Learning System Design Interview - Alex Xu
	#. Ace The Data Science Interview

********************************************************************************
General System Design interview Tips 
********************************************************************************
#. Start with documenting your summary/overview in Google docs/Excalidraw or Zoom whiteboard. Even if the company hasn’t provided a link and interviewer insists on the conversation to be purely verbal - Document key bullet points. 
#. Present your interview systematically; lead the conversation and don't wait for the interviewer to ask questions. At the beginning of the interview, present the discussion's structure and ask the interviewer about their main areas of interest. 
#. Show your understanding of the business implications by sharing insights on metrics. Understand what the product truly expects from you. 
#. Actively listen to the interviewer and ask about their primary focus. Address the whole process, from collecting and labeling data to defining metrics. 
#. Assess the importance of the modeling process. 
#. Familiarize yourself with the nuances of ML-Ops, such as: At the start of the interview, get a feel for if the interviewer seems interested in ML-Ops. You'll mostly get a clear signal on whether or not they are interested. 

	#. Managing model versions 
	#. Training models 
	#. Using model execution engines 
#. Keep your resume at hand and review it before starting the interview.

********************************************************************************
Recommendation System
********************************************************************************
1. Problem Setup and Label Collection
================================================================================
a. Clarifying Questions

	- Understand the problem context and objectives.
	- Identify constraints and requirements.
b. Definition of Success

	- Define key performance metrics (e.g., accuracy, precision, recall, business metrics).
c. Positive and Negative Labels

	i. Different Options to Define Labels:

		1. Joining a group.
		2. Retention after a week.
		3. Interaction with other users.
		4. Meaningful interaction (e.g., time spent, making friends in the group).
	ii. Fairness Considerations:

		- Ensure adequate data for underrepresented groups.
d. Label Generation

	i. Engagement as Proxy:

		- User click as a positive label, no click as a negative label.
	ii. Use of Labelers:

		1. Utilize semi-supervised or unsupervised methods (e.g., clustering) to enhance labeler efficiency.
		2. Consider visits in a session (e.g., Pinterest or DoorDash) as similar pins or restaurants.
e. Downsampling the Dominant Class using Negative Sampling

	- Only downsample training data while keeping validation and test distributions unchanged.
f. Bias in Training

	- Limit the number of labels per user, video, or restaurant to prevent bias towards active users or popular items.

2. Feature Engineering
================================================================================
a. User Features

	- Demographic information, historical behavior, preferences.
b. Group Features

	- Attributes of the group or community.
c. Cross Features Between Group and Users

	- Interaction-based features.
d. Contextual Features

	- Time of day, holiday, device type, network connection (WiFi vs. 4G).
e. Feature Selection Process

	- Start with basic counters/ratios and refine using Gradient Boosted Decision Trees (GBDT).

3. Modeling
================================================================================
a. Two-Tower Model

	- Separate embedding models for users and items.
b. Embedding Creation

	- Graph embeddings and learned representations.
c. Retrieval (Optimized for Recall)

	- Collaborative filtering-based approaches.
d. Diversification of Sources

	- Ensure variety in retrieved results.
e. Ranking (Optimized for Precision)

	1. Two-Tower Model.
	2. Precision-focused optimization.

4. Measurement
================================================================================
a. Offline vs. Online Evaluation

	- Offline metrics (precision, recall) vs. online business impact.
b. Key Metrics

	- NDCG (Normalized Discounted Cumulative Gain)
	- Precision\@Top-K: Measures relevance of top-K recommendations.
	- Mean Average Precision (MAP\@K): Mean of AP\@K across users.
c. Explanation of Metrics

	- Justify metric choice at each evaluation stage.
d. Online Measurement

	- Prioritize business metrics.
	- Conduct A/B testing or Multi-Armed Bandit experiments.

5. Debugging
================================================================================
a. Structured Debugging Approach

	- Maintain a clear, written log of issues and solutions.
b. Online vs. Offline Model Debugging

	- Identify discrepancies between offline validation and real-world performance.

6. Feature Logging
================================================================================
a. Training Phase

	- Ensure consistency in feature storage and retrieval.
b. Debugging

	- Log model inputs and outputs for analysis.

7. Preparing the Training Pipeline
================================================================================
	- Automate feature extraction, model training, and validation.
	- Ensure reproducibility and scalability.

8. Deployment
================================================================================
a. Novelty Effects

	- Account for temporary engagement spikes post-deployment.
b. Model Refresh Impact

	- Understand how periodic updates influence engagement.

9. Stages of a Ranking System Funnel
================================================================================
	- Retrieval: Reduce millions of candidates to thousands.
	- Filtering: Remove irrelevant or outdated candidates.
	- Feature Extraction: Ensure consistency in train-test splits.
	- Ranking: Apply advanced models to refine selections.

10. Advanced Topics
================================================================================
a. Data Pipeline & Infrastructure

	- Efficient data ingestion, storage, and processing at scale.
	- Real-time vs. batch data pipelines.
	- Feature freshness and consistency.

b. Scalability & Latency Considerations

	- Low-latency serving strategies.
	- Trade-offs between model complexity and inference speed.
	- Caching, pre-computation, and model distillation.

c. Handling Model Drift & Monitoring

	- Detection of data drift and performance degradation.
	- Automated retraining strategies.
	- Monitoring feature distribution shifts over time.

d. Fairness, Interpretability, and Ethics
	
	- Fairness-aware learning to mitigate biases.
	- Interpretability techniques like SHAP, LIME.
	- Ethical considerations in AI-driven recommendations.

********************************************************************************
Paradigms For Applications
********************************************************************************
* Classification 

	* Semantic analysis 
	* Learning to rank 
* Regression 
* Clustering 

	* Anomaly detection 
	* User understanding
* Dimensionality reduction 

	* Topic models
	* Inferred suggestions
* Generative modeling 

	* Structured prediction
* Multimodal learning

********************************************************************************
Broad Application Domains
********************************************************************************
Recommendation and Search
================================================================================
Retrieval
--------------------------------------------------------------------------------
(a) retrieval based on query - query can be text or images (image search)
(b) query-less personalised retrieval for homepage reco (Netflix/YT/Spotify/FB/Amzn homepage)
(c) item-specific recommendation for "suggested items similar to this"

Ranking
--------------------------------------------------------------------------------
(d) context-aware online ranking (CP model or some ranking model)

Policy Enforcement
--------------------------------------------------------------------------------
(e) fraud detection
(f) policy compliance models (age restriction, geo restriction, banned-item restriction)

********************************************************************************
Sample Questions
********************************************************************************
Question prompts moved to the question bank in `docs/source/gen/interviews/qb.rst`.
