#########################################################################
The Big Picture
#########################################################################

*************************************************************************
Probabilistic Framework
*************************************************************************
.. note::
	* For a given task, we can collect the stuff that we care about in a set :math:`\Omega`.
	* The goal of the probabilistic machine learning framework is to be able to define a probability measure, :math:`\mathbb{P}(\omega\in\Omega)`

		* Equivalently, we can define a (or a set of) random variable(s) :math:`S(\omega):2^{\Omega}\mapsto\mathbb{R}` and define a 

			* distribution :math:`F_S(s)=\mathbb{P}(S\leq s)` or a
			* density :math:`f_S(s)` such that 

				.. math:: F_S(s)=\int\limits_{-\infty}^s f_S(t)\mathop{dt}
	* The types of items that can be in the set :math:`\Omega` can be quite diverse, and therefore the associated rv can have the range which can confront to different types of mathematical structures.

		* Discrete variables:

			* A single binary variable, :math:`S\in\{0,1\}`.
			* A categorical variable, :math:`S\in\{1,\cdots,K\}`.
		* Continuous variables:
		
			* Real number :math:`S\in\mathbb{R}`
		* Variables of higher dimensions
		
			* Finite dimensional Euclidean vectors :math:`\mathbf{S}\in\mathbb{R}^d` with a common practice of associating each dimension with its own separate rv such that :math:`S_i\in\mathbb{R}`.
			* Infinite sequences :math:`(S)_{i=1}^\infty` where each :math:`S_i\in\mathbb{R}`.

Defining the Probabilities
*************************************************************************
Discrete variables
=========================================================================
.. note::
	* For discrete variables, we can enumerate the class-marginal probabilities in using a finite set of parameters
	* Bernoulli, :math:`S\sim\mathrm{Ber}(p)`

		.. math:: f_S(s=\{0,1\})=p^s(1-p)^{1-s}
	* Multinoulli, :math:`S\sim\mathrm{Mult}(\pi_1\cdots\pi_K)`

		.. math:: f_S(s=(s_1,\cdots,s_K))=\prod_{k=1}^K\pi_k^{s_k}

Continuous variables
=========================================================================
.. note::
	* For continuous case, explicit definition of the infinite dimensional density function is impossible.
	* We often therefore resort to some known parametric family of distributions

		* Example: Exponential family

Variables of higher dimensions
=========================================================================
Discrete case
-------------------------------------------------------------------------
.. note::
	* TODO - splitting the joint distribution by product of conditionals
	* Mention that enumerating the table is exponential in the number of variables
	* Mention conditional indedepence

Continuous case
-------------------------------------------------------------------------
.. note::
	* Mention conditional independence and using it to define parametric family (GMM)
	* Mention Markov property for sequence models
	* Mention the lower manifold in the data space

Usage of the Framework
*************************************************************************
.. note::
	* We can use this framework to estimate some quantities of interest from the distribution e.g. 
		
		* predict the mean :math:`\mathbb{E}_{S\sim F_S}[S]` [regression framework]
		* predict the mode :math:`s^*=\underset{s}{\arg\max} f_S(s)` [classification framework]
		* generate samples :math:`s^*=s\sim F_S` [generative framework]	

Utilising the Observations
*************************************************************************
.. note::
	* Data can be provided to help learn the parameters of the probability models
	* [Important] For regression and classification :math:`S=Y` and for generative models :math:`S=X`.

The Inference (Learning) Problem
=========================================================================
.. note::
	* Importance of the MLE framework to be able to learn the distributions through means of statistical inference.
