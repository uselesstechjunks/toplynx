********************************************************************************
Interview Question Bank
********************************************************************************
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

================================================================================
General Skills
================================================================================
Resume Review
--------------------------------------------------------------------------------
- You are a hiring manager at a large tech company and you have to review multiple resumes from a diverse list of extremely talented candidates per day to shortlist for interviewing in your team as a senior applied ML scientist for search and recommendation. Your friend is applying for similar roles elsewhere. You see the following in his resume and you decided to help him out by suggesting a couple of things to make his resume better. You have been an expert in this fiend for a long time and you know the tricks of the trade. What would you suggest your friend to change this to? Is there anything missing in his description? Is there anything else he should add to make it sound more impressive?

Leadership and Values
--------------------------------------------------------------------------------
- Ethical Considerations: How do they ensure their models are free from biases and uphold ethical standards in their work?
- Software Engineering Practices: How do they ensure the reproducibility, scalability, and maintainability of their machine learning systems?
- Domain Knowledge: How do they leverage domain expertise to inform their machine learning models and address real-world challenges?
- Communication Skills: How do they articulate complex technical concepts to non-technical stakeholders? Can they effectively communicate their ideas, findings, and recommendations to diverse audiences?
- Continuous Learning and Professional Development: How do they stay updated with the latest research, trends, and advancements? Are they involved in communities, conferences, or open-source contributions?
- Team Collaboration and Leadership: How do they contribute to a collaborative team environment, share knowledge, and mentor junior team members? Can they lead and inspire others to achieve common goals?
- Problem-Solving Skills: How do they break down complex problems, identify key insights, and propose innovative solutions using machine learning techniques?
- Cultural Fit and Motivation: What motivates them to work in the field of machine learning, and why are they interested in joining your organization specifically? How do they envision contributing to the team and making an impact?

Handling Difficult Scenarios
--------------------------------------------------------------------------------
- Describe a challenging situation you encountered while leading a project or team. How did you approach and resolve it, and what were the key takeaways from that experience?
- How do you prioritize tasks and manage deadlines in a fast-paced industry environment, especially when facing competing demands and resource constraints?
- Reflecting on the challenging situation you described, how did you prioritize competing objectives and allocate resources effectively to address the issue?
- Can you share a specific example of a time when you had to mediate conflicts or navigate interpersonal dynamics within your team? How did you foster collaboration and maintain team morale?
- In fast-paced environments, how do you ensure that quality is not compromised in pursuit of meeting deadlines? Can you provide examples of strategies you've used to maintain high standards of work under pressure?

================================================================================
ML Design and System Design
================================================================================
ML Design (Model Design)
--------------------------------------------------------------------------------
- What type of model should we use (e.g., linear regression, decision trees, deep learning, transformer-based models, etc.)?
- What features should we select or engineer?
- What kind of loss function should we optimize?
- How do we handle data preprocessing (e.g., normalization, missing values, imbalanced data)?
- What metrics should we optimize (e.g., accuracy, precision, recall, NDCG)?
- How do we train and fine-tune the model (hyperparameter tuning, regularization)?
- How do we validate and evaluate the model (cross-validation, test splits, A/B testing)?
- How do we deal with biases, fairness, and explainability in the model?

ML System Design (MLSD)
--------------------------------------------------------------------------------
- How do we serve the model in production (batch vs. real-time inference)?
- How do we scale the system (distributed training, model sharding, caching, retrieval optimizations)?
- What infrastructure should we use (on-prem vs. cloud, GPU vs. CPU deployment)?
- How do we handle data pipelines (streaming vs. batch processing, feature stores)?
- How do we handle model updates (versioning, retraining, continuous learning)?
- How do we ensure low-latency and high-availability (caching strategies, model distillation, quantization)?
- How do we monitor model performance and drift in production?
- How do we handle logging, debugging, and failure recovery?
- How do we ensure security, compliance, and privacy (differential privacy, federated learning)?

Design Prompts
--------------------------------------------------------------------------------
- Design a system for QA where a user would be able to search with a query and the system answers from an internal knowledge-base.
- What would you do to reduce the latency in the system further?
- How would you apply a content restriction policy in the system (not all users would be able to search through all the knowledge-base).

Study Framework Prompts
--------------------------------------------------------------------------------
- How to check if algorithm converged.
- How to check for overfitting/underfitting. Remedies?
- How to tune hyperparameters and perform model selection.
- Identify signs for model degradation over time. Remedies?

================================================================================
Applied ML (AML)
================================================================================
Generic Applied ML
--------------------------------------------------------------------------------
- Can you explain how you handle scenarios with low data availability?
- Could you elaborate on the different sampling techniques you are familiar with?
- Can you explain the teacher-student paradigm in machine learning? When is a separate teacher model needed?
- Explain a portion from your paper.

Click Prediction and Ranking
--------------------------------------------------------------------------------
- Can you discuss the pros and cons of Gradient Boosting Decision Trees (GBDT) with respect to Deep Neural Networks (DNNs)?
- Can you explain the personalization aspect of your Click Prediction model?
- Can you use a collaborative Filtering approach to solve the Click Prediction problem?
- What are the key metrics that you consider when evaluating your CP model?
- How do you determine when it needs retraining?
- How do you identify when things fail in your model or system?
- How did you handle categorical and ordinal features in your CP problem?
- Why did you frame online-ranking as a CP problem for ranking and not as a learning to rank problem?

Offline Ranking
--------------------------------------------------------------------------------
- Can you discuss the simulation strategy you used for offline ranking?
- What are the pros and cons of the marginalization you had to perform?

Personalization
--------------------------------------------------------------------------------
- Can you discuss the pros and cons of using a similarity score between a user’s history and an item to represent user interest?

Applied Metrics and Evaluation
--------------------------------------------------------------------------------
- What metrics are used for a heavily imbalanced dataset?

Applied ML Mixture
--------------------------------------------------------------------------------
- What is convolution Operation? Code it up.
- What is self attention?
- Derive gradient descent update rule for non negative matrix factorisation.
- Code non negative matrix factorisation.
- Derive gradient descent update rule for linear/logistic regression.
- Code stochastic gradient descent in linear/logistic regression setting.
- Code AUC.
- Questions related to my projects/thesis.
- One question from statistics: was related to Bayes theorem.
- Bias-variance tradeoff.
- Design questions: Let's say some countries don't allow showing ads for knife, gun, etc, how would you go about building a system that can classify safe queries vs unsafe queries?
- What's a language model?
- Explain the working of any click prediction model.
- A couple of questions related to indexing in search engine.
- Convolution vs feedforward.

================================================================================
ML Theory (MLT)
================================================================================
Core ML
--------------------------------------------------------------------------------
- Explain overfitting and regularization.
- Explain the bias-variance tradeoff.
- How do you handle data imbalance issues?
- Explain Gradient descent and Stochastic gradient descent. Which one would you prefer?
- Can you explain logistic regression and derive gradient descent for Logistic regression.
- What do eigenvalues and eigenvectors mean in PCA.
- Explain different types of Optimizers — How is Adam optimizer different from Rmsprop?
- What are the different types of activation functions and explain about vanishing gradient problem?
- How does batch norm help in faster convergence?
- Why does inference take less memory than training?
- What do L1 and L2 regularization mean and when would you use L1 vs. L2? Can you use both?
- When there are highly correlated features in your dataset, how would the weights for L1 and L2 end up being?

Dimensionality Reduction and Generative Models
--------------------------------------------------------------------------------
- Tell me a few dimensionality reduction mechanisms - PCA and autoencoders.
- Explain PCA and probabilistic PCA.
- What is the reconstruction loss in terms of eigenvalues?
- Why are eigenvalues positive in this case? Can you prove that the variance-covariance matrix is PSD?
- How would you select the number of dimensions in PCA?
- Think of an autoencoder with just 1 hidden layer. How would you select the dimension in this case?
- Can you think of a justification for why we'd see a diminishing return as we increase the hidden dimension?
- Is autoencoder related to kernel-PCA?
- What is the loss function for VAE? Explain ELBO and the KL term.
- If we split ELBO further, a reconstruction loss term and another KL term comes out. How is that KL term defined? What are those corresponding distributions?
- Why do we use Gaussians in VAE? Why standard Gaussians? Why assuming standard Gaussian in the latent space doesn't hurt?
- What does this prior signify from a Bayesian perspective?
- How about discrete VAE? How does the reparameterization work in that case?

Statistics and Probability
--------------------------------------------------------------------------------
- You have 3 features, X, Y, Z. X and Y are correlated, Y and Z are correlated. Should X and Z also be correlated always?
- Can you explain what non-parametric two-sample tests are and how they differ from parametric ones?
- Could you provide the intuition behind the Maximum Mean Discrepancy (MMD) estimator that you used?
- Do you know about Bayesian testing? Is Bayesian the same as non-parametric?

Linear Algebra
--------------------------------------------------------------------------------
- Can you list the linear algebra algorithms you are familiar with?
- What is a rational approximation of an operation function?
- Can you discuss the feature selection algorithms that you implemented?
- What are linear operators? How do they differ from non-linear operators?
- Can you explain the estimation strategy that you used in the approximation algorithm?

Classical ML
--------------------------------------------------------------------------------
- What are the different ways to measure performance of a linear regression model.
- Some zero problem on Naive Bayes.
- Difference between gradient boosting and XGBoost.

================================================================================
NLP and LLMs (MLT/AML)
================================================================================
Transformers and Encoders
--------------------------------------------------------------------------------
- Do you have experience with LLMs?
- What is the difference between offline selection and online ranking?
- What are the inputs and outputs of your triplet BERT model?
- Explain triplet BERT architecture, how is it different from normal BERT? Why do you need 3 copies of the identical towers and not just concatenate text with SEP token?
- How do you tackle embeddings of 3 different embeddings?
- What is the meaning of doing a max-pooling over the features in terms of individual dimensions?
- How is max-pooling different than doing concatenation first and then projection?
- Walk me through the entire BERT encoding process from input sequence in natural text to final the layer.
- Explain how wordpiece works. Explain the Embedding matrix. What are its dimensions?
- Why do we need positional encodings? Which part in the transformer layer requires this positional information?
- Why do we need to divide QK^T by sqrt(d_k)?
- Why do we need softmax?
- Why do we need residual connection?
- Explain the FC layer.
- Can you explain how BERT is trained?
- How does BERT differ from models like GPT or T5?
- Can you use BERT for text generation?
- What are the different BERT variants that you have experimented with?
- How do you fine-tune a BERT-based model for your specific domain?
- What is a Sentence-BERT (SBERT) model? How is it different from normal BERT?
- How is SBERT trained and how do you evaluate its quality?
- Other than BERT, what other Encoder Models do you know of?

Multilingual
--------------------------------------------------------------------------------
- How would you approach training a multilingual model?
- What are the key challenges and why this is hard to do?

LLM Evaluation
--------------------------------------------------------------------------------
- What are your evaluation metrics and why do you use them?
- Do you know about metrics that are used for generation?
- Tell me some shortcomings of BLEU and ROUGE. What other metric can we use? How is perplexity defined?
- Do you know about Reflection? How would you evaluate LLM outputs for hallucination and other mistakes?

Prompting
--------------------------------------------------------------------------------
- How do you go about fine-tuning a large language model?
- How did you select which prompts to use in your model?
- Could you share some prompts that didn’t work and how you came up with better ones?

GANs
--------------------------------------------------------------------------------
- How did you use the MMD estimator as a discriminator in a GAN?
- What are the difficulties in training and using GANs? Are there better alternatives out there?

================================================================================
Additional Question Sets (GPT-generated)
================================================================================
Ensemble Learning
--------------------------------------------------------------------------------
- Explain the concept of ensemble learning and the rationale behind combining multiple weak learners to create a strong learner. Provide examples of ensemble methods and their respective advantages and disadvantages.
- Can you discuss any ensemble learning techniques you've used in your projects, such as bagging, boosting, or stacking? How do you select base learners and optimize ensemble performance in practice?
- With the increasing popularity of deep learning models, how do you see the role of ensemble methods evolving in modern machine learning pipelines, and what are the challenges and opportunities in combining deep learning with ensemble techniques?

Dimensionality Reduction Techniques
--------------------------------------------------------------------------------
- Discuss the importance of dimensionality reduction techniques in machine learning, particularly in addressing the curse of dimensionality and improving model efficiency and interpretability.
- Can you explain the difference between linear and non-linear dimensionality reduction methods, and provide examples of algorithms in each category? When would you choose one method over the other?
- Given the exponential growth of data in various domains, how do you adapt dimensionality reduction techniques to handle high-dimensional datasets while preserving meaningful information and minimizing information loss?

Model Evaluation and Validation
--------------------------------------------------------------------------------
- Explain the concept of model evaluation and validation, including common metrics used for assessing classification, regression, and clustering models.
- Can you discuss any strategies or best practices for cross-validation and hyperparameter tuning to ensure robust and reliable model performance estimates?
- Given the prevalence of imbalanced datasets and skewed class distributions in real-world applications, how do you adjust model evaluation metrics and techniques to account for class imbalance and minimize bias in performance estimation?

Statistical Hypothesis Testing
--------------------------------------------------------------------------------
- Discuss the principles of statistical hypothesis testing and the difference between parametric and non-parametric tests. Provide examples of hypothesis tests commonly used in machine learning and statistics.
- Can you explain Type I and Type II errors in the context of hypothesis testing, and how you control for these errors when conducting multiple hypothesis tests or adjusting significance levels?
- With the increasing emphasis on reproducibility and rigor in scientific research, how do you ensure the validity and reliability of statistical hypothesis tests, and what measures do you take to mitigate the risk of false positives or spurious findings?

Bayesian Methods and Probabilistic Modeling
--------------------------------------------------------------------------------
- Explain the Bayesian approach to machine learning and its advantages in handling uncertainty, incorporating prior knowledge, and facilitating decision-making under uncertainty.
- Can you discuss any Bayesian methods or probabilistic models you've applied in your work, such as Bayesian regression, Gaussian processes, or Bayesian neural networks? How do you interpret and communicate Bayesian model outputs to stakeholders?
- Given the computational challenges of Bayesian inference, how do you scale Bayesian methods to large datasets and high-dimensional parameter spaces, and what approximation techniques or sampling methods do you use to overcome these challenges?

Graph Neural Networks (GNNs)
--------------------------------------------------------------------------------
- Explain the theoretical foundations of graph neural networks (GNNs) and their applications in recommendation systems and social network analysis.
- Can you discuss any challenges or limitations in training GNNs on large-scale graphs, particularly in scenarios with heterogeneous node types or dynamic graph structures?
- With the growing interest in heterogeneous information networks and multimodal data, how do you extend traditional GNN architectures to handle diverse types of nodes and edges, and what strategies do you employ to integrate different modalities effectively?

Causal Inference and Counterfactual Reasoning
--------------------------------------------------------------------------------
- Discuss the importance of causal inference in machine learning applications, particularly in domains such as personalized recommendation systems and healthcare analytics.
- Can you explain the difference between causal inference and predictive modeling, and how you incorporate causal reasoning into the design and evaluation of machine learning models?
- Given the challenges of estimating causal effects from observational data, what techniques or methodologies do you use to address confounding variables and selection bias, and what are the limitations of these approaches?

Federated Learning and Privacy-Preserving Techniques
--------------------------------------------------------------------------------
- Explain the concept of federated learning and its advantages in scenarios where data privacy and security are paramount, such as healthcare or financial services.
- Can you discuss any challenges or trade-offs in implementing federated learning systems, particularly in terms of communication overhead, model aggregation, and privacy guarantees?
- With the increasing regulatory scrutiny and consumer concerns around data privacy, how do you ensure compliance with privacy regulations such as GDPR or CCPA while leveraging data for model training and inference, and what techniques do you use to anonymize or encrypt sensitive information?

Meta-Learning and Transfer Learning
--------------------------------------------------------------------------------
- Discuss the principles of meta-learning and its applications in few-shot learning, domain adaptation, and model generalization across tasks and datasets.
- Can you provide examples of meta-learning algorithms or frameworks you've worked with, and how they improve the efficiency and effectiveness of model adaptation and transfer?
- With the increasing complexity and diversity of machine learning models, how do you leverage transfer learning techniques to transfer knowledge from pre-trained models to new tasks or domains, and what strategies do you employ to fine-tune model parameters and hyperparameters effectively?

Interpretability and Explainable AI
--------------------------------------------------------------------------------
- Explain the importance of model interpretability and explainability in machine learning, especially in domains such as finance, healthcare, and law enforcement.
- Can you discuss any techniques or methodologies for explaining black-box models, such as LIME, SHAP, or model distillation, and their advantages and limitations in different contexts?
- Given the trade-offs between model complexity and interpretability, how do you balance model performance with the need for transparency and accountability, and what strategies do you use to communicate complex model decisions to stakeholders or end-users?

================================================================================
Project and Leadership Question Sets (GPT-generated)
================================================================================
Ad-Asset Ranking Models
--------------------------------------------------------------------------------
- Explain the trade-offs between using deep neural networks (DNN) and gradient boosting decision trees (GBDT) for click prediction models in online ad-ranking systems.
- Can you compare the computational complexity and training/inference time between DNN and GBDT models in the context of ad-ranking systems?
- How do you handle language-agnostic historical signals in ad-ranking? Can you elaborate on the challenges and strategies involved?
- How do you handle feature engineering for language-agnostic signals, and what are the challenges in doing so?
- Describe the process of integrating semantic query-context signals with a multilingual BERT-based model. What are the key considerations in this integration?
- Can you discuss any specific techniques or algorithms you implemented for caching embeddings to achieve faster online inference? How did they impact latency and resource utilization?

Offline Selection Problem
--------------------------------------------------------------------------------
- Detail the approach you designed to address the offline selection problem by simulating potential query-contexts with each item. How did you handle the scalability issues with a large item set?
- When simulating potential query-contexts with each item, how do you ensure diversity and relevance in the generated scenarios?
- Explain the sampling strategies you employed in the offline selection problem and their impact on model performance.
- What considerations are important when devising sampling strategies for the offline selection problem, especially when dealing with a large item set?
- Can you elaborate on the process of fine-tuning the semantic model to assign scores in each scenario and how you handle the marginalization step effectively?

Text Feature Engineering and Augmentation
--------------------------------------------------------------------------------
- Discuss your experience in creating homogeneous text features from various user signals and GPT prompts for online ad-ranking. How did you address signal scarcity in this process?
- Can you elaborate on the prompt-based data augmentation techniques you utilized for enhancing signal strength in ad-ranking systems?
- How do you evaluate the effectiveness of prompt-based data augmentation techniques in enhancing signal strength? Are there any risks or limitations associated with these techniques?
- In what ways do you ensure that the augmented text features maintain semantic coherence and relevance to user preferences?
- Could you share examples of specific GPT prompts or augmentation strategies you found particularly effective in your work?

Model Infrastructure Unification
--------------------------------------------------------------------------------
- As a leader in unifying online-ranking modeling infrastructure globally, what challenges did you encounter, especially in coordinating across geographical teams? How did you overcome them?
- Describe your approach to providing hands-on mentorship to new joiners in the team. Can you share a specific example where your mentorship significantly impacted a project or team member?
- Can you discuss any technical or cultural challenges encountered during the process of unifying online-ranking modeling infrastructure globally? How did you address resistance to change or differing opinions among teams?
- How do you balance the need for standardization and consistency with the flexibility required to accommodate diverse market needs and preferences?
- As a mentor, how do you tailor your approach to individual team members with varying levels of experience and expertise?

Research Contributions
--------------------------------------------------------------------------------
- Explain the significance of the cache-friendly algorithm you devised for non-parametric two-sample tests involving the Maximum Mean Discrepancy (MMD) estimator. How does it contribute to computational efficiency?
- Could you elaborate on the implementation details of the multi-threaded variant you developed for the algorithm and its performance improvements over existing solutions?
- What specific optimizations or algorithmic improvements contributed to the significant speed-up achieved by your cache-friendly algorithm for non-parametric two-sample tests?
- Can you elaborate on any practical considerations or trade-offs involved in implementing the multi-threaded variant of the algorithm?
- How does the use of state-of-the-art solvers in your algorithm compare to alternative approaches in terms of scalability and robustness?

Open Source Contributions
--------------------------------------------------------------------------------
- Reflect on your experience co-mentoring in the design of Shogun’s Linear Algebra library. What were the key challenges in ensuring the library's efficiency and usability?
- Discuss the framework you developed for computing rational approximations of linear-operator functions in cases where exact computation is impractical. How did you ensure the accuracy and scalability of the estimator for log-det of high-dimensional, sparse matrices?
- What criteria did you consider when designing and selecting feature selection algorithms for the kernel-based hypothesis tests framework?
- How do you ensure the numerical stability and accuracy of the estimator for log-det of high-dimensional, sparse matrices in your framework?
- Can you discuss any challenges or lessons learned from integrating the framework into existing open-source libraries or ecosystems?

Deep Understanding of Machine Learning Concepts
--------------------------------------------------------------------------------
- Explain the concept of a teacher-student paradigm in machine learning and its relevance in addressing signal sparsity. Provide an example of how you applied this paradigm in your work.
- What are some common challenges in designing personalized recommendation systems, and how do you mitigate them? Can you discuss a specific challenge you faced and how you overcame it?
- How do you balance the trade-off between model complexity and interpretability in personalized recommendation systems, especially when dealing with large-scale data and diverse user preferences?
- Can you provide examples of how you addressed issues such as cold start, data sparsity, or model drift in personalized recommendation systems?
- What are some emerging trends or advancements in recommendation systems that you find particularly exciting or promising?

================================================================================
Reference Links
================================================================================
Sample Questions Links
--------------------------------------------------------------------------------
- https://www.geeksforgeeks.org/machine-learning-interview-questions/
- https://www.turing.com/interview-questions/machine-learning
- https://www.interviewbit.com/machine-learning-interview-questions/
- https://anywhere.epam.com/en/blog/machine-learning-interview-questions
- https://www.mygreatlearning.com/blog/machine-learning-interview-questions/

Related StackExchanges
--------------------------------------------------------------------------------
- `stats.stackexchange <https://stats.stackexchange.com/>`_
- `datascience.stackexchange <https://datascience.stackexchange.com/>`_
- `ai.stackexchange <https://ai.stackexchange.com/>`_
