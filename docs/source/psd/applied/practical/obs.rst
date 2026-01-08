###################################################################################
Observability
###################################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***********************************************************************************
Distribution Shift
***********************************************************************************
Defitions
===================================================================================
.. note::
	* Distribution shift: :math:`p_{\text{train}}(\mathbf{x},y)\neq p_{\text{test}}(\mathbf{x},y)`
	* Covariate shift: 

		* :math:`p_{\text{train}}(\mathbf{x})\neq p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})=p_{\text{test}}(y|\mathbf{x})`
	* Concept shift:

		* :math:`p_{\text{train}}(\mathbf{x})=p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})\neq p_{\text{test}}(y|\mathbf{x})`
	* Label shift:

		* Only in :math:`y\implies\mathbf{x}` problems.
		* :math:`p_{\text{train}}(y)\neq p_{\text{test}}(y)`
		* :math:`p_{\text{train}}(\mathbf{x}|y)=p_{\text{test}}(\mathbf{x}|y)`

Identification 
===================================================================================
Statistical & Distance-Based Methods
-----------------------------------------------------------------------------------
#. Population Stability Index (PSI) / Jensen-Shannon Divergence (JSD)  
#. Kolmogorov-Smirnov (KS) Test 

Model Performance Monitoring
-----------------------------------------------------------------------------------
#. Live A/B Testing with Shadow Models  
#. Error Analysis on Recent Queries  

Embedding-Based Drift Detection
-----------------------------------------------------------------------------------
#. Measuring Drift in Learned Representations (e.g., PCA, t-SNE)  
#. Contrastive Learning for Drift Detection  

Resources
===================================================================================
- [mit.edu] `Class Imbalance, Outliers, and Distribution Shift <https://dcai.csail.mit.edu/2024/imbalance-outliers-shift/>`_	
- [arize.com] `Drift Metrics: a Quickstart Guide <https://arize.com/blog-course/drift/>`_
- [arize.com] `Courses on Observability <https://courses.arize.com/courses/>`_
