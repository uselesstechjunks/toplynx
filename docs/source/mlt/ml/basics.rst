################################################################################################
Fundamentals of Learning
################################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

************************************************************************************************
Entropy, Cross-Entropy, NLL, KL
************************************************************************************************
- Categorical distribution with number of classes `C`.
- Labels: :math:`y_i=\{0,1\}^{C}\in\mathbb{R}^C`, one-hot.
- Empirical distribution: :math:`\hat{p}=\frac{1}{N}\sum_{i=1}^N y_i\in\mathbb{R}^C`

	.. math:: \hat{p}(k)=\frac{1}{N}\sum_{i=1}^N\mathbb{1}(y_i(k)=1)\in\mathbb{R}
- Predicted probability: 

	- Density function learned: :math:`\pi`, :math:`\pi(k)` for each category.
	- For sample :math:`i`: Sample from :math:`\pi`, :math:`\pi_i\in\mathbb{R}^C`
	- For class :math:`k`: :math:`\pi_i(k)\in\mathbb{R},k=1\dots C`
	- For true class: :math:`\pi(y_i)\in\mathbb{R}=\pi_i^Ty_i`
- Aggregate predictive probability:

	.. math:: \bar{\pi}(k)=\frac{1}{N}\sum_{i=1}^N\pi_i(k)
- Entropy: :math:`H(p)=-p\log(p)=-\sum_{k=1}^C p_k\log(p_k)`
- Empirical entropy:

	.. math:: H(\hat{p})=-\sum_{k=1}^C \hat{p}\log(\hat{p})
- Entropy of predicted probability:

	.. math:: H(\pi)\approx-\frac{1}{N}\sum_{i=1}^N H(\pi_i)=-\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^C \pi_i(k)\log(\pi_i(k))
- Cross-entropy:

	.. math:: H(\hat{p},\pi)=-\sum_{k=1}^C \hat{p}(k)\log(\pi(k))\approx-\frac{1}{N}\sum_{k=1}^C\sum_{i=1}^N\mathbb{1}(y_i(k)=1)\log(\pi_i(k))=-\frac{1}{N}\sum_{i=1}^N y_i^T\log(\pi_i)=-\frac{1}{N}\sum_{i=1}^N \log(\pi(y_i))=NLL
- KL

	.. math:: KL(\hat{p}||\pi)=\hat{p}\log(\frac{\hat{p}}{\pi})=\sum_{k=1}^C \hat{p}(k)\log(\frac{\hat{p}(k)}{\pi(k)})=H(\hat{p},\pi)-H(\hat{p})

************************************************************************************************
Defining the Objective
************************************************************************************************
.. note::
	* Prerequisies:

		* High School Math

			* [Loney] Trigonometry, Coordinate Geometry
			* [Strang] Calculus Volume 1, 2, 3
		* Matrix Algebra

			* [Graybill] Matrices with Applications in Statistics - Chapter 4 Geometric Interpretation
			* [Springer] Matrix Tricks for Linear Statistical Models - Chapter Introduction
			* Matrix Cookbook - Identities - All Things Multinomial and Normal
		* Matrix Calculus

			* [Minka] `Old and New Matrix Algebra Useful for Statistics <https://tminka.github.io/papers/matrix/minka-matrix.pdf>`_
			* [Dattaro] `Convex Optimization - Appendix D <https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/mc.pdf>`_
			* [Abadir Magnus] Matrix Algebra 
		* Probability Theory - Exponential Family, Graphical Models

			* [Bishop] Pattern Recognition and Machine Learning
		* Point Estimation

			* [Lehaman] Theory of Point Estimation - Chapter 1 Preparations
		* Information Theory
	* Estimating Densities

		* Divergence

			* Discriminative Models

				* Cross Entropy and Negative Log-Likelihood
				* Regression - Bayes Estimator: Conditional Expectation Solution
				* Classification - Bayes Estimator: MAP Solution
			* Latent Generative Models

				* Variational Lower Bounds
				* Gaussian Mixture Models
				* Probabilistic PCA
				* Variational Autoencoder
				* Denoising Probabilistic Diffusion
		* Integral Probability Metrics

			* MMD
			* Wasserstein Distance
	* Minmax Theory

		* Adversarial Objective: GAN
		* Constrained Objective Formulation

************************************************************************************************
Optimisation for Optimality
************************************************************************************************
.. note::
	* Prerequisies:

		* Matrix Algebra and Calculus - Geometric View, Identities
		* Taylor Approximation
	* Unconstrained: First and Second Order Methods

		* First Order Methods 

			* Exact: Gradient Descent Variants
			* Approximate: Stochastic Gradient Descent Variants
		* Second Order Methods

			* Exact: Newton's Method
			* Approximate: Gauss-Newton's Hessian Approximation
	* Constrained

		* Lagrange Multipliers
		* KKT
