##################################################################################
Statistical Decision Theory
##################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

This puts the prediction task under a statistical inference paradigm.

Statistical machine learning deals with the problem of prediction by using existing data to form a model. There are two distinct aspects of such a task

.. note::
	* Inference:

		* Since the model space can be arbitrarily complex, we assume some structure to the model.
		* The model can be a simplistic mathematical model which predicts a deterministic outcome given an unknown input.
		* The model can also be a probabilistic model which aims at learning the joint or the conditional distribution.
		* The task of inference (learning) is to use data to come up with an optimal model through some meaning of optimality.
	* Prediction:

		* Once we have the model, the task of prediction is simplified if our model is a deterministic one.
		* When our model is probabilistic, then we must instantiate (draw a sample) the outcome to a fixed value given an unknown input.
		* Statistical decision theory deals with the prediction problem which has it's own notion of optimality, usually defined with the help of a loss function.

**********************************************************************************
Statistical Inference
**********************************************************************************
This is defined separately for regression and classification problem.

Analytic Solutions
==================================================================================
.. warning::
	* Under this paradigm, the **data is a realisation of some underlying random process**. 
	* We define the prediction problem under this framework. 
	* The optimal solutions under this often rely on having access to the underlying distribution or some other quantities involving the distribution. 
	* We also discuss ways to define the notion of optimality.

Single Random Variable
----------------------------------------------------------------------------------
.. note::
	* We have a single real-valued rv :math:`X`.
	* We consider the estimation problem where we **find an estimate** :math:`\hat{x}`, **a single value, for any future observation of** :math:`X`.

		* We define Prediction Error (PE): The rv :math:`\tilde{X}=X-\hat{x}`, which has the same pdf as :math:`X`.
	* The **optimality** of our estimate is defined with the help of a **loss function** such that

		.. math:: \hat{X}_{\text{OPT}}=\underset{\hat{X}}{\arg\min} L(X,\hat{X})

		* Loss function is usually some function of PE.
		* MSE loss function is defined as

			.. math:: \mathbb{E}_X[\tilde{X}^2]=\mathbb{E}_X[(X-\hat{x})^2]=\mathbb{E}_X[X^2]-2\mathbb{E}_X[X]\hat{x}+\hat{x}^2

.. tip::
	* To find :math:`\hat{x}`, we can differentiate w.r.t. :math:`\hat{x}` to minimize MSE.

		* Note that :math:`\mathbb{E}_X[X^2]` and :math:`\mathbb{E}_X[X]` are constants.
		* Therefore

			.. math:: \frac{\partial}{\mathop{\partial\hat{x}}}\mathbb{E}_X[(X-\hat{x})^2]=-2\mathbb{E}_X[X]+2\hat{x}\implies\hat{x}_{\text{OPT}}=\mathbb{E}_X[X]

Two Random Variables
----------------------------------------------------------------------------------
.. note::
	* We assume that the data :math:`X` and the target :math:`Y/G` are distributed per a **joint distribution**

		* [Regression] :math:`X,Y\sim F_{X,Y}(x,y)`
		* [Classification] :math:`X,G\sim F_{X,G}(x,g)`
	* We assume that we'll have access to future realisations of :math:`X=x` but not the target.
	* The task is to **find an estimator for the target as function of data**, :math:`\hat{Y}=f(X)` **or** :math:`\hat{G}=g(X)`.
	
		* We use these to **predict future values for the target** as :math:`\hat{y}=f(x)` and :math:`\hat{g}=g(x)`.
	* The **optimality** of our estimate is defined with a non-negative **loss function**, :math:`L`.

		* [Regression] :math:`\hat{Y}_{\text{OPT}}=\underset{\hat{Y}}{\arg\min} L(Y,\hat{Y})`
		* [Classification] :math:`\hat{G}_{\text{OPT}}=\underset{\hat{G}}{\arg\min} L(G,\hat{G})`
	* We wish the predictors to have minimal expected prediction error (EPE) **over the joint**.

		* [Regression] :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y})`
		* [Classification] :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G})`
	* EPE can be reformulated as conditional expectation on observed input variables :math:`X`.

		* [Regression] :math:`EPE=\mathbb{E}_X\left[\mathbb{E}_{Y|X}[L(Y,\hat{Y}|X]\right]`
		* [Classification] :math:`EPE=\mathbb{E}_X\left[\mathbb{E}_{G|X}[L(G,\hat{G}|X]\right]`

.. tip::
	* Since :math:`L` is non-negative, this quantity is minimised when it's minimum at each point :math:`X=x`.
		
		* As we're fixing :math:`X` to a constant, the outer expectation :math:`\mathbb{E}_X` goes away.		
		* [Regression] :math:`\hat{y}_{\text{OPT}}=\underset{\hat{y}}{\arg\min}\left(\mathbb{E}_{Y|X}[L(Y,\hat{y}|X=x]\right)`
		* [Classification] :math:`\hat{g}_{\text{OPT}}=\underset{\hat{g}}{\arg\min}\left(\mathbb{E}_{G|X}[L(G,\hat{g}|X=x]\right)`.
	* For particular choice of loss functions, we arrive as **optimal (Bayes) estimator** definitions

		* [Regression] With MSE loss, :math:`\hat{Y}=\mathbb{E}_{Y|X}[Y|X]`, **mean of the conditional pdf**.
		* [Classification] With 0-1 loss, :math:`\hat{G}` corresponds to the **mode of the conditional pmf**.

Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bayes Estimator
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	* This is the estimator which minimises MSE for each point :math:`X=x`.

		.. math:: L(Y,\hat{y})=\mathbb{E}_{Y|X}[(Y-\hat{y})^2|X=x]
	* To find minimum, we differentiate w.r.t :math:`\hat{y}=f(x)`, a single value

		.. math:: \frac{\partial}{\mathop{\partial\hat{y}}}L(Y,\hat{y})=\frac{\partial}{\mathop{\partial\hat{y}}}\left(\mathbb{E}_{Y|X}[Y^2|X=x]-2\mathbb{E}_{Y|X}[Y|X=x]\hat{y}+\hat{y}^2\right)=-2\mathbb{E}_{Y|X}[Y|X=x]+2\hat{y}
	* Therefore, the optimal estimator at each realisation :math:`X=x` is given by

		.. math:: \hat{y}=f(x)=\mathbb{E}_{Y|X}[Y|X=x]
	* We note that this estimator is unbiased.

.. note::
	TODO - Alternate proof from Sayed and orthogonality conditions !!!IMPORTANT!!!

Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bayes Classifier
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Approximating The Analytic Solutions
==================================================================================
.. warning::
	* In practical problems, we often don't have access to the underlying distribution. 
	* In such cases, we resort to the approximation framework that tries to mimic the optimal solution.
	* We use statistical inference to estimate the unknowns of our model.

Regression - Approximating The Conditional Mean
----------------------------------------------------------------------------------
Assuming locally constant nature of the fucntion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* In kNN regression approach, we approximate Bayes estimator by 

		* replacing expectation with sample average
		* approximating the point :math:`X=x` with a neighbourhood :math:`N(x)` where :math:`|N(x)|=k`
		* The parameter :math:`k` is chosen using model selection approaches.
		* Usually the choice of :math:`k` determines the **roughness** of this model, with larger values resulting in smoother model.
	* In this case :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx\text{Avg}(y_i|x_i\in N(x))`
	* The implicit assumption is that the function behaves locally constant around each point :math:`x`
	* Therefore, it can be estimated with the average value of the target :math:`y_i` for each data point in the neighbourhood :math:`x_i`.

Explicit assumption from a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* In linear regression, we explicitly assume that the estimator is affine in :math:`X_j`.
	
		* In this case, :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx \beta^T x + \beta_0`
		* We usually add a dummy variable :math:`X_0=1` in :math:`X` and write this as a linear function

			.. math:: f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx \beta^T x
	* In basis expansion, we assume that the estimator is an affine in some transform :math:`h(x)\in\mathbb{R}^M`.

		* Example: :math:`x=(x_1,x_2)^T\overset{h}{\longrightarrow}(1,x_1,x_2,x_1x_2,x_1^2,x_2^2)^T`
		* In this case, :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx \beta^T h(x)`

Bias-Variance Tradeoff
==================================================================================

Model Selection and Model Assessment
==================================================================================
Analytic Solutions
----------------------------------------------------------------------------------
AIC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BIC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Experimental Solutions
----------------------------------------------------------------------------------
Bootstrap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**********************************************************************************
Notation
**********************************************************************************
.. warning::
	* All vectors are named for their column vector form. 
	* For row-representation, we use the transpose notation.

.. note::
	* Data is associated with a random variable :math:`X`.
	* Observed data points are instances of the rv, :math:`X=x\in\mathbb{R}^d` for some :math:`d\geq 1`.
	
		* If :math:`d> 1`, :math:`X` is a random vector.
		* In this case, individual components can referred to as :math:`X_j` and :math:`X=(X_1,\cdots,X_d)`.

.. note::
	* [Regression] The target quantity is associated with a continuous rv :math:`Y` taking values :math:`Y=y\in\mathbb{R}^K`, for some :math:`K\geq 1`.

		* It might also be a random vector, with :math:`Y=(Y_1,\cdots,Y_K)`.
		* Single dimensional observations for target are usually written as :math:`Y=y\in\mathbb{R}`.
	* [Classification] The target quantity is associated with a discrete rv :math:`G\in\mathcal{G}` with :math:`|\mathcal{G}|=K`.

.. note::
	* We have a total of :math:`N` observations, and all the observations together are taken in the matrix form

		.. math:: \mathbf{X}_{N\times d}=\begin{bmatrix}-& x_1^T & - \\ \vdots & \vdots & \vdots \\ -& x_N^T & -\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\ \mathbf{x}_1 & \cdots & \mathbf{x}_d \\ |&\cdots&|\end{bmatrix}
	* The vector :math:`\mathbf{x}_j\in\mathbb{R}^N` represents the column vector for all the observations for rv :math:`X_j`.
	* A particular observation for :math:`X=x_i\in\mathbb{R}^d` is taken in the row-vector form, :math:`x_i^T\in\mathbb{R}_{1\times d}`.
	* For :math:`K> 1`, we can also associate the target with the row vector form, :math:`y_i^T\in\mathbb{R}_{1\times K}` [regression] or :math:`g_i^T\in\mathcal{G}_{1\times K}` [classification].

**********************************************************************************
Curse of Dimensionality
**********************************************************************************
.. note::
	* As we move to higher dimensional space, the notion of **distance** doesn't follow our intuition.
	* As this `SO post <https://stats.stackexchange.com/a/99191>`_ puts it (quoting verbatim)

		* Another application, beyond machine learning, is nearest neighbor search: given an observation of interest, find its nearest neighbors (in the sense that these are the points with the smallest distance from the query point). 
		* But in high dimensions, a curious phenomenon arises: the ratio between the nearest and farthest points approaches 1, i.e. the points essentially become uniformly distant from each other. 
		* This phenomenon can be observed for wide variety of distance metrics, but it is more pronounced for the Euclidean metric than, say, Manhattan distance metric. 
		* The premise of nearest neighbor search is that "closer" points are more relevant than "farther" points, but if all points are essentially uniformly distant from each other, the distinction is meaningless.
	* More resource on this:

		* `On the Surprising Behavior of Distance Metrics in High Dimensional Space <https://bib.dbvis.de/uploadedFiles/155.pdf>`_
		* `When Is "Nearest Neighbor" Meaningful? <https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf>`_
		* `Fractional Norms and Quasinorms Do Not Help to Overcome the Curse of Dimensionality <https://www.mdpi.com/1099-4300/22/10/1105/pdf?version=1603175755>`_
