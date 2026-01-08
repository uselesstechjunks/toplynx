##########################################################################
Adobe
##########################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

**************************************************************************
ML Practicals
**************************************************************************
#. Experience with LLMs
#. ML Ops
	
		- Experience with different parts of the stack - analysis, feature design, feature pipelines, experimentation, registry/versioning, deployment, feature logging, monitoring
		- How would you measure latency increase?
#. Project discussion - Model boundary
#. Model evaluation
	
	- Metrics

		- Explain ROC-AUC and PR-AUC in detail. In what scenarios, one is better than the other?
		- Does a high ROC-AUC imply overfitting? Why/why not? How would you detect overfitting?
	- Slices

		- How to choose slices for evaluation? Numerical, categorical, temporal, edge cases.
#. Feature design
	
		- How would you design features with numerical and categorical signals
#. Productionizing

	- How would you select the criteria for marking a model productionizable?
	- Is it metric based? How would you select the threshold?
	- How would you evaluate your model in presence of feedback loops from production models?
	- Golden dataset creation

		- What are the criteria for selecting a golden set for offline evaluation?
		- Would the criteria for automatic productionizing be any different?
		- How often would you update it?
		- How would you ensure that it has balanced representation for rare but important cases?
#. Systems
	
		- Experience with map reduce

**************************************************************************
ML Theory/Design
**************************************************************************
#. Master's Thesis

	- Explain submodular optimisation with examples.
#. Experience with LLMs

	- Explain different types of finetuning tasks. Is knowledge distillation the same as finetuning?
	- LoRA

		- Explain how LoRA works.
		- Explain rank. Why is row rank = column rank?
		- Suppose you change k entries in a matrix. How much can the rank change?
		- What is the maximum rank of sum of k rank 1 matrices?
		- Why cannot we use LoRA from the beginning of the training?
#. Sampling

		- Given a dataset of 1M examples, how would you design a sampling mechanism to select 10k samples?
		- What is the probability that :math:`x_i` and :math:`x_j` are both part of that selected set?
		- How can you convince a non-technical person that each example is equally likely to be selected?
#. Transformers

	- Explain the difference between encoder, decoder and encoder-decoder architecture.
	- What is the context width of BERT?
	- How would you solve the SQuAD task using a transformer architecture? How would you model the problem and which architecture would you choose?
#. Initialisation

	- How would you initialise the weights? Which technique to use when?
	- What impact would a specific type of initialisation technique have during model training? What is it aiming to solve?
#. Batch Norm

	- Explain batch norm and internal covariate shift.
	- How exactly does the mena and variance calculation work for a given batch?
	- Instead of sample mean, can we use any other convex combination of samples to estimate expectation?
	- What is the difference between biased and unbiased estimators of the variance? Explain bias of an estimator.
#. Contextual bandits

	- How would you approach contextual bandits to tackle assets generated online?
**************************************************************************
Emerging Tech
**************************************************************************
**************************************************************************
Leadership
**************************************************************************
