
################################################################################
ML Breadth
################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

********************************************************************************
Resources
********************************************************************************
#. Misc Resources

	#. [github.com] `Machine-Learning-Interview-Prep <https://github.com/QuickLearner171998/Machine-Learning-Interview-Prep/blob/master/README.md>`_
#. Specific Topics

	#. [youtube.com] `ML interviews <https://www.youtube.com/playlist?list=PLXmbE5IFg3EEoSAzuqbu7o8Kh8FFhTFPc>`_
	#. [blog.paperspace.com] `Intro to Optimization in Deep Learning: Busting the Myth About Batch Normalization <https://blog.paperspace.com/busting-the-myths-about-batch-normalization/>`_
	#. [medium.com] `A Visual Explanation of Gradient Descent Methods (Momentum, AdaGrad, RMSProp, Adam) <https://medium.com/towards-data-science/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c>`_
	#. `Clustering evaluation. <https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation>`_
	
		- `Silhouette Coefficient <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_
		- `CH Index <https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index>`_
		- `DB Index <https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index>`_
		- `Rand Index <https://scikit-learn.org/stable/modules/clustering.html#rand-index>`_

#. Interview Guide

	* [medium.com] `How to Crack Machine learning Interviews at FAANG! <https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5>`_
	* [medium.com] `Part 2 — How to Crack Machine learning Interviews at FAANG : Pointers for Junior/Senior/Staff+ levels <https://medium.com/@reachpriyaa/part-2-how-to-crack-machine-learning-interviews-at-faang-pointers-for-junior-senior-staff-4b89e10bff28>`_

********************************************************************************
Revision Topics
********************************************************************************
Fundamentals
================================================================================
#. Math
	
	- Matrix Factorizations - Eigendecomposition, SVD, QR, Cholesky, PD/PSD
	- Matrix Calculus, Vector Calculus, Multivariate Chain Rule, Backprop derivation
	- Taylor's Expansion, First and second order Approximation
	- Convex Optimization: GD, Newton's Mehtod, Gauss-Newton's Approximation
	- Optimizers: SGD, AdaGrad, RMSProp, Adam - Learning Rate, Schedule, Choice of parameters, Effect of Regularisation
	- Constrained Optimization: Lagrange Multipliers, KKT Conditions
	- Mercer Kernels, Reproducing Property
#. Stat

	- Expectations, Higher Moments, Skew, Kurtoisis, Conditional, Marginal, Joint, Bayes Theorem
	- Variance Covariance Matrix, Scatter Matrix, Correlation Matrix, Eigendecomposition
	- Distributions and moments - Bernoulli, Binomial, Categorical, Multinomial, Normal, Poisson, Exponential, Logistic
	- Frequentis Estimation Theory - Point Estimation, Confidence Interval, Hypothesis Testing
	- Point Estimation, Bayes Estimator, Bias Variance Decomposition
	- Minmax Theory, Empirical Risk Minimization
	- Bayesian Estimaton Theory, Conjugate Priors - Normal, Beta-Binomial, Dirichlet-Multinomial
	- Sampling Techniques: CDF-Jacobian, Monte Carlo Estimators, Metropolis Hasting, Adaptive Metropolis Hasting, Ancestor Sampling
#. Learning Theory

	- KL divergence, Entropy, MaxEnt, Cross-Entropy, NLL
	- Graphical Models - BN, MRF, CRF
	- Variational Inference, Belief Propagation, Deep Belief Net
#. Regression and Classification

	- Bayes Estimator - Estimator for conditional mean (regression) or conditional mode (classification)
	- Basis Expansion, Gram matrix, Feature Maps
	- Linear Regression, Logistic Regression, Naive Bayes, SVM
	- Tree Based Methods: DT, Bagging, Boosting, XGBoost
#. Clustering

	- Distance Based: K-Means, Density Based: DBSCAN
	- Hierarchical Clustering, Self-Organizing Maps
	- Metrics - Distance Based (Silhoutte coefficient), DB index, CH index

#. Manifold Learning

	- t-SNE
	- Spectral Clustering
#. Latent Variable Models

	- GMM, PCA, Kernel-PCA, ICA, CCA
	- NMF, LDA
#. Outlier prediction

	- Isolation Forest
	- One-Class SVM
#. Density Estimation

	- Linear/Quadratic Discriminator Analysis
	- Kernel Density Estimator

#. Practical

	- Feature Engineering

#. Reinforcement learning
	
	- SARSA
	- Explore-exploit, bandits (eps-greedy, UCB, Thompson sampling), 
	- Q-learning, DQN

#. Learning To Rank

	- Predicts a relative-order
	- Metrics: MAP, Precision@k, Recall@k, DCG@k/NDCG@k, MRR)
	- Common Approaches: Pairwise

Esoteric Topics
================================================================================
	* Ordinal Regression - predicts a class label/score (check `this <https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf>`_)	
	* Causal reasoning and diagnostics, Causal networks
	* Learning latent representations
	* Bayesian linear regression
	* Gaussian Processes

********************************************************************************
Study Framework
********************************************************************************
* Problem

	* Problem description and assumptions for simplicity.
* Approach and Assumptions

	* Theoretical framework & motivation.
	* Mathematical derivation of training objective (loss) with boundary conditions.
	* What-if scenarios where training fails - mathematical issues (check stack-exchange).
* Training and Validation

	* Design the training algorithm
	* Implementation and computational considerations including complexity.
	* How to check if algorithm converged.
	* What-if scenarios where training fails - computational issues (check stack-exchange).		
* Testing and Model Selection

	* How to check for overfitting/underfitting. Remedies?
	* Metrics to check - different choices and trade-offs.
	* How to tune hyperparameters and perform model selection.
* Inference

	* Computational considerations.
	* Identify signs for model degradation over time. Remedies?

Sample Questions
================================================================================
(a) https://www.geeksforgeeks.org/machine-learning-interview-questions/
(b) https://www.turing.com/interview-questions/machine-learning
(c) https://www.interviewbit.com/machine-learning-interview-questions/
(d) https://anywhere.epam.com/en/blog/machine-learning-interview-questions
(e) https://www.mygreatlearning.com/blog/machine-learning-interview-questions/

Fundamentals
--------------------------------------------------------------------------------
	#. Explain overfitting and regularization
	#. Explain the bias-variance tradeoff.
	#. How do you handle data imbalance issues?
	#. Explain Gradient descent and Stochastic gradient descent. Which one would you prefer?
	#. Can you explain logistic regression and derive gradient descent for Logistic regression
	#. What do eigenvalues and eigenvectors mean in PCA
	#. Explain different types of Optimizers — How is Adam optimizer different from Rmsprop?
	#. What are the different types of activation functions and explain about vanishing gradient problem?
	#. How does batch norm help in faster convergence?
	#. Why does inference take less memory than training?
	#. What do L1 and L2 regularization mean and when would you use L1 vs. L2? Can you use both?
	#. When there are highly correlated features in your dataset, how would the weights for L1 and L2 end up being?

Screening
--------------------------------------------------------------------------------
	#. Explain one project where you faced a challenging or ambiguous problem statement and solved it. What was the business impact?
	#. How do you decide between the model complexity vs the latency budget (I mentioned this during my explanation)?
	#. What is SFT and why it is needed?
	#. What do you understand by PPO in RLHF?
	#. What are LoRA and QLoRA?
	#. Have you worked with other types of generative models like GAN or VAE?
	#. Tell me how GANs are trained. Objective function?
	#. What are some of the problems in training GANs? Said Mode Collapse and Vanishing Gradient (too string discriminator). Asked me to explain both.
	#. How are VAEs different from vanilla autoencoders?
	#. Explain the reparameterisation trick.
	#. For classification trees, what is the splitting criteria?
	#. How are Random Forests different from normal classification trees?
	#. What is regularisation and why do we need it? Explained in RR and DNN? What type of regulariser is used in RR? What is the L1 version called?

In Depth Theory
--------------------------------------------------------------------------------
	#. Tell me a few dimensionality reduction mechanisms - PCA and autoencoders.
	#. Explain PCA and probabilistic PCA.
	#. What is the reconstruction loss in terms of eigenvalues?
	#. Why are eigenvalues positive in this case? Can you prove that the variance-covariance matrix is PSD?
	#. How would you select the number of dimensions in PCA?
	#. Think of an autoencoder with just 1 hidden layer. How would you select the dimension in this case?
	#. Can you think of a justification for why we'd see a diminishing return as we increase the hidden dimension?
	#. Is autoencoder related to kernel-PCA?
	#. What is the loss function for VAE? Explain ELBO and the KL term.
	#. If we split ELBO further, a reconstruction loss term and another KL term comes out. How is that KL term defined? What are those corresponding distributions?
	#. Why do we use Gaussians in VAE? Why standard Gaussians? Why assuming standard Gaussian in the latent space doesn't hurt?
	#. What does this prior signify from a Bayesian perspective?
	#. How about discrete VAE? How does the reparameterization work in that case?		
	#. How would you determine if your click-prediction model has gone bad over time?
	#. If you cannot afford an A/B test, could you still evaluate this? What is the framework for these types of analysis? G-formula and propensity-score reweighting in causal inference.
	#. Can you use MSE for evaluating your classification problem instead of Cross entropy
	#. How does the loss curve for Cross entropy look?
	#. What does the "minus" in cross-entropy mean?
	#. Explain how Momentum differs from RMS prop optimizer?

********************************************************************************
Topics
********************************************************************************
Sample Interview Questions
================================================================================
Feature Engineering
--------------------------------------------------------------------------------
* When do we need to scale features?
* How to handle categorical features for

	* categories with a small number of possible values
	* categories with a very large number of possible values
	* ordinal categories (an order associated with them)

Mathematics
--------------------------------------------------------------------------------
* Different types of matrix factorizations. 
* How are eigenvalues related to singular values.

Statistics
--------------------------------------------------------------------------------
* You have 3 features, X, Y, Z. X and Y are correlated, Y and Z are correlated. Should X and Z also be correlated always?

Classical ML
--------------------------------------------------------------------------------
* Regression

	* What are the different ways to measure performance of a linear regression model.
* Naive Bayes

	* Some zero problem on Naive Bayes
* Trees

	* Difference between gradient boosting and XGBoost.

Applied ML
--------------------------------------------------------------------------------
* What metrics are used for a heavily imbalanced dataset?

Mixture
--------------------------------------------------------------------------------
	#. What is convolution Operation? Code it up.
	#. What is self attention?
	#. Derive gradient descent update rule for non negative matrix factorisation.
	#. Code non negative matrix factorisation.
	#. Derive gradient descent update rule for linear/logistic regression.
	#. Code stochastic gradient descent in linear/logistic regression setting.
	#. Code AUC.
	#. Questions related to my projects/thesis.
	#. One question from statistics: was related to Bayes theorem.
	#. Bias-variance tradeoff.
	#. Design questions: Let's say some countries don't allow showing ads for knife, gun, etc, how would you go about building a system that can classify safe queries vs unsafe queries?
	#. What's a language model?
	#. Explain the working of any click prediction model.
	#. A couple of questions related to indexing in search engine.
	#. Convolution vs feedforward.

Related StackExchanges
================================================================================
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_
