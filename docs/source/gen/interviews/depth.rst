
################################################################################
ML Depth
################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Fundamentals
********************************************************************************
Metrics
================================================================================
- Regression
- Classification
- Ranking

	- Online ranking: ROC-AUC, PR-AUC  
	- Offline selection: ROC-AUC  
- Generative Models

	- BLUE, ROUGE
	- GPT Labeling: Accuracy, Human-like 

Projects  
================================================================================
- Motivation
- Approach: Metrics, features, architecture  
- Challenges
- Outcome

Transformer-Based Models
================================================================================
Initialization
--------------------------------------------------------------------------------
- Xavier/He initialization  

Tokenizers
--------------------------------------------------------------------------------
- Byte Pair Encoding (BPE)  
- WordPiece  
- SentencePiece  

Activations
--------------------------------------------------------------------------------
- Sigmoid, Tanh, ReLU, LeakyReLU
- SiLU, GELU, Swish
- GLU, GeGLU, ReGLU, SwiGLU

Normalization
--------------------------------------------------------------------------------
- Internal covariate shift  
- Batch normalization  
- Layer normalization (LayerNorm)  
- RMS normalization (RMSNorm)  
- Pre-layer normalization (Pre-LN)  

Position Embedding Schemes
--------------------------------------------------------------------------------
- Learned positional encoding  
- Sinusoidal positional encoding  
- Rotary positional encoding  

Attention Mechanisms
--------------------------------------------------------------------------------
- Functions: Additive, dot product, scaled dot product  
- Architectures:  

	- Full (e.g., Flash, Paged, Ring)  
	- Sparse  
	- Multi-query attention  
	- Grouped-query attention  
- Techniques:  

	- Self-attention (SHA)  
	- Multi-head attention (MHA)  
	- Encoder attention  
	- Decoder attention with KV-cache  

Decoding Techniques
--------------------------------------------------------------------------------
- Beam search  
- Top-p, Top-k sampling  
- Temperature scaling  
- Speculative decoding  

Transformer Architectures
--------------------------------------------------------------------------------
- Models: PyTorch implementation

	- BERT: MLM
	- GPT: CLM  
	- T5: Seq2Seq
- Papers
	- Encoder: BERT, RoBERTa, XLM-R
	- Decoder: GPT, LLama3, DeepSeek
	- Seq2Seq: T5 Learnings, mT5
	- Autoencoder: BART
	- PLM: XLNet
	- RTD: Elextra
	- MoE: Mixtral
	- State-Space: Mamba
	- Retriever Embeddings: Generalizable T5-based Retriever (GTR)

Hardware and Optimization
--------------------------------------------------------------------------------
- Flash attention  
- Quantization (e.g., INT8 LLM)  
- Paged attention  
- Ring attention  

LLM Techniques
--------------------------------------------------------------------------------
- Prompt engineering  
- Prompt tuning  
- Retrieval-Augmented Generation (RAG)  
- LoRA, QLoRA
- Supervised Fine-Tuning (SFT)  
- Reinforcement Learning with Human Feedback (RLHF)  
- Proximal Policy Optimization (PPO)  
- Direct Preference Optimization (DPO)

Convolution-Based Models
================================================================================
- Convolution-based models: ConvNet, ResNet, Graph CN, LightGCN, Graph Transformers.  

Generative Models
================================================================================
- Generative models on latent variable space: VAE, VQVAE, GAN, diffusion models, diffusion transformers.  

Multimodal Models
================================================================================
TODO

Good to Know
================================================================================
- Popular NL tasks and remember the dataset names. 
- Kernel methods: kernel meaning embedding, MMD, other IPMs – read every inch of our paper.  
- Probability and statistics: parametric and non-parametric methods for inference, CI, and hypothesis testing framework.  
- Bayes Net (representation, inference, learning).  
- Causality – how to think systematically about finding the root cause of a problem; Bing search causality paper.  
- Latent variable models: K-means, mixture of Gaussians, PCA, kernel PCA, ICA.  
- Clustering: convex, non-convex, evaluation of clustering performance.  
- Regression and discriminative classification: model assumption, interpretation, evaluation – collinearity, the other stuff.  
- Theory as applied problem statement.  
- Code kernel methods, tree methods, regression, VAE, GAN, diffusion models.

Sample Questions
================================================================================
Transformers
--------------------------------------------------------------------------------
	#. Do you have experience with LLMs?
	#. Explain offline selection problem in detail.
	#. What is the difference between offline selection and online ranking?
	#. What are the inputs and outputs of your triplet BERT model?
	#. Explain triplet BERT architecture, how is it different from normal BERT? Why do you need 3 copies of the identical towers and not just concatenate text with SEP token?
	#. How do you tackle embeddings of 3 different embeddings? 
	#. What is the meaning of doing a max-pooling over the features in terms of individual dimensions? 
	#. How is max-pooling different than doing concatenation first and then projection?
	#. Walk me through the entire BERT encoding process from input sequence in natural text to final the layer.
	#. Explain how wordpiece works. Explain the Embedding matrix. What are its dimensions?
	#. Why do we need positional encodings? Which part in the transformer layer requires this positional information?
	#. Why do we need to divide QK^T by sqrt(d_k)?
	#. Why do we need softmax?
	#. Why do we need residual connection?
	#. Explain the FC layer.
	#. What are your evaluation metrics and why do you use them?
	#. Do you know about metrics that are used for generation?
	#. Tell me some shortcomings of BLEU and ROUGE. What other metric can we use? How is perplexity defined?
	#. Do you know about Reflection? How would you evaluate LLM outputs for hallucination and other mistakes?

Generic
--------------------------------------------------------------------------------
	* Can you explain how you handle scenarios with low data availability?
	* Could you elaborate on the different sampling techniques you are familiar with?
	* Can you explain the teacher-student paradigm in machine learning? When is a separate teacher model needed?
	* Explain a portion from your paper.

Click Prediction
--------------------------------------------------------------------------------
	* Can you discuss the pros and cons of Gradient Boosting Decision Trees (GBDT) with respect to Deep Neural Networks (DNNs)?
	* Can you explain the personalization aspect of your Click Prediction model? 
	* Can you use a collaborative Filtering approach to solve the Click Prediction problem?
	* What are the key metrics that you consider when evaluating your CP model? 
	* How do you determine when it needs retraining?
	* How do you identify when things fail in your model or system?
	* How did you handle categorical and ordinal features in your CP problem? 
	* Why did you frame online-ranking as a CP problem for ranking and not as a learning to rank problem?

Encoder
--------------------------------------------------------------------------------
	* Can you explain how BERT is trained? 
	* How does BERT differ from models like GPT or T5? 
	* Can you use BERT for text generation?
	* What are the different BERT variants that you have experimented with? 
	* How do you fine-tune a BERT-based model for your specific domain?
	* What is a Sentence-BERT (SBERT) model? How is it different from normal BERT?
	* How is SBERT trained and how do you evaluate its quality? 
	* Other than BERT, what other Encoder Models do you know of?

Multilingual
--------------------------------------------------------------------------------
	* How would you approach training a multilingual model?
	* What are the key challenges and why this is hard to do?

Offline Ranking
--------------------------------------------------------------------------------
	* Can you discuss the simulation strategy you used for offline ranking? 
	* What are the pros and cons of the marginalization you had to perform? 

Personalization
--------------------------------------------------------------------------------
	* Can you discuss the pros and cons of using a similarity score between a user’s history and an item to represent user interest?

GAN
--------------------------------------------------------------------------------
	* How did you use the MMD estimator as a discriminator in a GAN? 
	* What are the difficulties in training and using GANs? Are there better alternatives out there?

LLM
--------------------------------------------------------------------------------
	* How do you go about fine-tuning a large language model?
	* How did you select which prompts to use in your model? 
	* Could you share some prompts that didn’t work and how you came up with better ones?

Statistics
--------------------------------------------------------------------------------
	* Can you explain what non-parametric two-sample tests are and how they differ from parametric ones? 
	* Could you provide the intuition behind the Maximum Mean Discrepancy (MMD) estimator that you used? 
	* Do you know about Bayesian testing? Is Bayesian the same as non-parametric?

Linear Algebra
--------------------------------------------------------------------------------
	* Can you list the linear algebra algorithms you are familiar with? 
	* What is a rational approximation of an operation function? 
	* Can you discuss the feature selection algorithms that you implemented? 
	* What are linear operators? How do they differ from non-linear operators? 
	* Can you explain the estimation strategy that you used in the approximation algorithm?

GPT-generated Sample Questions
================================================================================
1. Click Prediction and Ranking Models
--------------------------------------------------------------------------------
	- Can you explain the theoretical underpinnings of gradient boosting decision trees (GBDT) and how they differ from traditional decision tree models in the context of click prediction?
	- How do you handle overfitting in deep neural network (DNN) models for click prediction, especially when dealing with high-dimensional and sparse input features?
	- In your experience, what are the key advantages and limitations of using ensemble methods like GBDT compared to deep learning models in ad-ranking systems?
	- Given the inherent trade-offs between interpretability and performance in ad-ranking models, how do you balance these factors when designing and deploying models in production systems?
	- Can you discuss any challenges you faced in feature engineering for click prediction, particularly when dealing with heterogeneous data sources or unstructured text inputs?
	- With the increasing emphasis on privacy and data protection regulations, how do you ensure that click prediction models remain compliant with legal and ethical standards, especially in the context of user data usage and privacy?
	- Given the dynamic nature of user behavior and ad landscapes, how do you design models that are robust to concept drift and seasonality in online ad-ranking systems?
	- Can you discuss any innovative techniques or algorithms you've developed to handle imbalanced data in click prediction, particularly when dealing with rare events or skewed click-through rates?
	- With the increasing prevalence of adversarial attacks targeting recommendation systems, how do you ensure the resilience and security of ad-ranking models against manipulation and exploitation?
	
2. Multilingual BERT and Sentence BERT
--------------------------------------------------------------------------------
	- Can you explain the architecture and pre-training objectives of BERT models, and how they are adapted for multilingual applications?
	- How do you fine-tune pre-trained BERT models for specific downstream tasks such as ad-ranking or sentiment analysis, and what are the best practices for maximizing performance?
	- With the advent of models like RoBERTa and ALBERT, how do you assess the trade-offs between using BERT-based models and newer architectures for multilingual NLP tasks?
	- What are the main challenges in fine-tuning pre-trained BERT models for low-resource languages, and how do you mitigate these challenges in practice?
	- In your experience, how does the performance of multilingual BERT models compare to domain-specific or language-specific models in tasks such as sentiment analysis or document classification?
	- Can you discuss any recent advancements or research findings in adapting transformer-based models like BERT for cross-lingual transfer learning, and their implications for multilingual NLP applications?
	- How do you address the challenge of domain adaptation when fine-tuning pre-trained BERT models for specific applications or industries, and what strategies do you employ to minimize domain shift?
	- Can you discuss any limitations or biases inherent in pre-trained language models like BERT, especially in the context of multilingual or cross-cultural applications, and how you mitigate these issues?
	- Given the resource-intensive nature of training and fine-tuning large transformer models, how do you optimize model performance and efficiency, particularly in low-resource settings or on edge devices?

3. Prompt Tuning and Prompt-Generated Data Augmentation
--------------------------------------------------------------------------------
	- What role does prompt tuning play in enhancing the performance of large language models (LLMs) such as GPT-3 in downstream tasks like text generation or classification?
	- How do you select and design prompts for specific tasks, and what strategies do you employ to ensure that the generated text adheres to the desired style or content?
	- Can you discuss any recent advancements or research findings in prompt tuning and its applications in improving the efficiency and effectiveness of LLMs?
	- How do you measure the effectiveness of prompt tuning in improving the performance of language models, and what metrics do you use to evaluate the quality of generated text?
	- Can you discuss any challenges or limitations you encountered when tuning prompts for specific tasks or domains, and how you addressed them?
	- With the growing interest in zero-shot and few-shot learning techniques, how do you envision the role of prompt tuning evolving in future developments of large language models?
	- What considerations do you take into account when selecting prompts for different tasks or domains, and how do you ensure that the prompts capture the relevant semantics and context?
	- Can you discuss any challenges or limitations you've encountered when generating diverse and representative prompts for data augmentation, particularly in scenarios with limited labeled data?
	- With the emergence of self-supervised learning approaches like CLIP and DALL-E, how do you see the role of prompt tuning evolving in enabling more versatile and adaptive language models?

4. Linear Algebra and Sampling
--------------------------------------------------------------------------------
	- Explain the importance of linear algebra in machine learning and deep learning, especially in tasks involving matrix operations and optimization.
	- How do you leverage sampling techniques such as Monte Carlo methods or Markov Chain Monte Carlo (MCMC) in machine learning applications, and what are their advantages and limitations?
	- Can you provide examples of how techniques from linear algebra and sampling are applied in probabilistic graphical models or Bayesian inference?
	- Discuss the computational challenges associated with matrix operations in deep learning models, especially when dealing with large-scale datasets or high-dimensional feature spaces.
	- How do you assess the convergence and stability of sampling-based algorithms such as MCMC in probabilistic modeling, and what strategies do you employ to improve their efficiency?
	- Can you provide examples of how techniques from linear algebra and sampling are applied in reinforcement learning or generative modeling, and the specific challenges involved in these applications?
	- Discuss the impact of numerical stability and precision in matrix computations on the performance and reliability of deep learning models, and how you address issues such as numerical instability or overflow.
	- Can you provide examples of how you leverage techniques from randomized linear algebra, such as sketching or random projections, to accelerate computation or reduce memory footprint in large-scale machine learning tasks?
	- With the increasing complexity and dimensionality of modern datasets, how do you ensure scalability and efficiency in sampling-based algorithms for inference or optimization, and what strategies do you employ to parallelize computation or exploit hardware accelerators?
	
5. Probability (Gaussians) and Non-parametric Statistics
--------------------------------------------------------------------------------
	- Discuss the properties and applications of Gaussian distributions in machine learning, and how they are used in modeling continuous-valued variables or noise.
	- What are non-parametric statistical methods, and how do they differ from parametric approaches in terms of flexibility and assumptions?
	- Can you elaborate on specific non-parametric statistical tests or estimators you have used in your work, and the scenarios in which they are preferred over parametric methods?
	- Explain the concept of kernel density estimation (KDE) and its applications in non-parametric density estimation, including its advantages and limitations compared to parametric approaches.
	- How do you address issues such as boundary effects or kernel selection in kernel-based non-parametric methods, and what techniques do you use to optimize their performance?
	- Can you discuss any recent advancements or research findings in non-parametric statistics, such as scalable algorithms for estimating high-dimensional distributions or adaptive kernel methods?
	- Explain the concept of copulas and their applications in modeling complex dependencies in high-dimensional data, and how you incorporate copula-based methods into machine learning pipelines.
	- Can you discuss any challenges or considerations in estimating non-parametric density functions from empirical data, particularly in scenarios with limited sample sizes or high-dimensional feature spaces?
	- Given the increasing availability of data streams and real-time analytics, how do you adapt non-parametric statistical methods for online learning or streaming data analysis, and what techniques do you use to update models dynamically?

GPT-generated Sample Questions on Projects and Leadership
================================================================================
Ad-Asset Ranking Models:
--------------------------------------------------------------------------------
	- Explain the trade-offs between using deep neural networks (DNN) and gradient boosting decision trees (GBDT) for click prediction models in online ad-ranking systems.
	- Can you compare the computational complexity and training/inference time between DNN and GBDT models in the context of ad-ranking systems?
	- How do you handle language-agnostic historical signals in ad-ranking? Can you elaborate on the challenges and strategies involved?    
	- How do you handle feature engineering for language-agnostic signals, and what are the challenges in doing so?
	- Describe the process of integrating semantic query-context signals with a multilingual BERT-based model. What are the key considerations in this integration?
	- Can you discuss any specific techniques or algorithms you implemented for caching embeddings to achieve faster online inference? How did they impact latency and resource utilization?

Offline Selection Problem:
--------------------------------------------------------------------------------
	- Detail the approach you designed to address the offline selection problem by simulating potential query-contexts with each item. How did you handle the scalability issues with a large item set?
	- When simulating potential query-contexts with each item, how do you ensure diversity and relevance in the generated scenarios?
	- Explain the sampling strategies you employed in the offline selection problem and their impact on model performance.
	- What considerations are important when devising sampling strategies for the offline selection problem, especially when dealing with a large item set?	
	- Can you elaborate on the process of fine-tuning the semantic model to assign scores in each scenario and how you handle the marginalization step effectively?
	
Text Feature Engineering and Augmentation:
--------------------------------------------------------------------------------
	- Discuss your experience in creating homogeneous text features from various user signals and GPT prompts for online ad-ranking. How did you address signal scarcity in this process?
	- Can you elaborate on the prompt-based data augmentation techniques you utilized for enhancing signal strength in ad-ranking systems?
	- How do you evaluate the effectiveness of prompt-based data augmentation techniques in enhancing signal strength? Are there any risks or limitations associated with these techniques?
	- In what ways do you ensure that the augmented text features maintain semantic coherence and relevance to user preferences?
	- Could you share examples of specific GPT prompts or augmentation strategies you found particularly effective in your work?

Model Infrastructure Unification:
--------------------------------------------------------------------------------
	- As a leader in unifying online-ranking modeling infrastructure globally, what challenges did you encounter, especially in coordinating across geographical teams? How did you overcome them?
	- Describe your approach to providing hands-on mentorship to new joiners in the team. Can you share a specific example where your mentorship significantly impacted a project or team member?
	- Can you discuss any technical or cultural challenges encountered during the process of unifying online-ranking modeling infrastructure globally? How did you address resistance to change or differing opinions among teams?
	- How do you balance the need for standardization and consistency with the flexibility required to accommodate diverse market needs and preferences?
	- As a mentor, how do you tailor your approach to individual team members with varying levels of experience and expertise?

Research Contributions:
--------------------------------------------------------------------------------
	- Explain the significance of the cache-friendly algorithm you devised for non-parametric two-sample tests involving the Maximum Mean Discrepancy (MMD) estimator. How does it contribute to computational efficiency?
	- Could you elaborate on the implementation details of the multi-threaded variant you developed for the algorithm and its performance improvements over existing solutions?
	- What specific optimizations or algorithmic improvements contributed to the significant speed-up achieved by your cache-friendly algorithm for non-parametric two-sample tests?
	- Can you elaborate on any practical considerations or trade-offs involved in implementing the multi-threaded variant of the algorithm?
	- How does the use of state-of-the-art solvers in your algorithm compare to alternative approaches in terms of scalability and robustness?

Open Source Contributions:
--------------------------------------------------------------------------------
	- Reflect on your experience co-mentoring in the design of Shogun’s Linear Algebra library. What were the key challenges in ensuring the library's efficiency and usability?
	- Discuss the framework you developed for computing rational approximations of linear-operator functions in cases where exact computation is impractical. How did you ensure the accuracy and scalability of the estimator for log-det of high-dimensional, sparse matrices?
	- What criteria did you consider when designing and selecting feature selection algorithms for the kernel-based hypothesis tests framework?
	- How do you ensure the numerical stability and accuracy of the estimator for log-det of high-dimensional, sparse matrices in your framework?
	- Can you discuss any challenges or lessons learned from integrating the framework into existing open-source libraries or ecosystems?

Deep Understanding of Machine Learning Concepts:
--------------------------------------------------------------------------------
	- Explain the concept of a teacher-student paradigm in machine learning and its relevance in addressing signal sparsity. Provide an example of how you applied this paradigm in your work.
	- What are some common challenges in designing personalized recommendation systems, and how do you mitigate them? Can you discuss a specific challenge you faced and how you overcame it?
	- How do you balance the trade-off between model complexity and interpretability in personalized recommendation systems, especially when dealing with large-scale data and diverse user preferences?
	- Can you provide examples of how you addressed issues such as cold start, data sparsity, or model drift in personalized recommendation systems?
	- What are some emerging trends or advancements in recommendation systems that you find particularly exciting or promising?

Handling Difficult Scenarios:
--------------------------------------------------------------------------------
	- Describe a challenging situation you encountered while leading a project or team. How did you approach and resolve it, and what were the key takeaways from that experience?
	- How do you prioritize tasks and manage deadlines in a fast-paced industry environment, especially when facing competing demands and resource constraints?
	- Reflecting on the challenging situation you described, how did you prioritize competing objectives and allocate resources effectively to address the issue?
	- Can you share a specific example of a time when you had to mediate conflicts or navigate interpersonal dynamics within your team? How did you foster collaboration and maintain team morale?
	- In fast-paced environments, how do you ensure that quality is not compromised in pursuit of meeting deadlines? Can you provide examples of strategies you've used to maintain high standards of work under pressure?
