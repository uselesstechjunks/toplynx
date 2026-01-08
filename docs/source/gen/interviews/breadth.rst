
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
	* Convergence checks.
	* What-if scenarios where training fails - computational issues (check stack-exchange).		
* Testing and Model Selection

	* Overfitting/underfitting checks and remedies.
	* Metrics to check - different choices and trade-offs.
	* Hyperparameter tuning and model selection.
* Inference

	* Computational considerations.
	* Model degradation monitoring and remedies.

Sample Questions
================================================================================
Sample questions moved to the question bank in `docs/source/gen/interviews/qb.rst`.

Statistics
--------------------------------------------------------------------------------
Question prompts moved to the question bank in `docs/source/gen/interviews/qb.rst`.

Classical ML
--------------------------------------------------------------------------------
Question prompts moved to the question bank in `docs/source/gen/interviews/qb.rst`.

Applied ML
--------------------------------------------------------------------------------
Question prompts moved to the question bank in `docs/source/gen/interviews/qb.rst`.

Mixture
--------------------------------------------------------------------------------
Question prompts moved to the question bank in `docs/source/gen/interviews/qb.rst`.

Related StackExchanges
================================================================================
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_
