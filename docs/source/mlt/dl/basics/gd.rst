###################################################################################
Gradient Descent
###################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. note::
	* Taylor's series
	* Local linear approximation information captured by gradient.
	* Local quadratic approximation information captured by Hessian.
	* First order approximation of the gradient near a point

***********************************************************************************
Nature of the Error Surface near Stationary Point
***********************************************************************************
.. note::
	* Understanding the nature of local stationary point (maxima/minima/saddle point) with the help of Hessian.

.. warning::
	Largest eigenvalue = direction of slowest descent

***********************************************************************************
Gradient Descent
***********************************************************************************
Batch Gradient Descent
===================================================================================
Stochastic Gradient Descent
===================================================================================
Mini-batch Gradient Descent
===================================================================================

***********************************************************************************
Convergence
***********************************************************************************
.. warning::
	* With fixed learning rate, convergence is not guaranteed.

		* (TODO: derive) In the near proximity of a stationary point where quadratic approximation is reasonable

			* the Learning-rate should be :math:`\eta\le 2/\lambda_\mathrm{max}`
			* With this learning-rate, the optimization improves on the error by a factor of :math:`(1+1/\kappa)` where :math:`\kappa=\lambda_\mathrm{max}/\lambda_\mathrm{min}` is the condition number of the Hessian.
		* Assuming that the LR is set accordingly, it takes infinitely many steps to reach the minimum.
		* Need to set the threshold somewhere.
	* TODO: proof

Learning-Rate Schedule
===================================================================================
.. warning::
	Key idea: Larger LR at the beginning, smaller towards the end.

.. note::
	* Linear decay
	* Exponential decay
	* Power-law decay

Faster Covergence with Momentum
===================================================================================
.. warning::
	* Carry on a little bit extra along the previous direction before stopping and changing direction again.
	* Moves fast near optima as opposed to a fixed variant.

Heavy-Ball Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nesterov Accelerated Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Adaptive Learning Rates
===================================================================================
.. warning::
	* Allow different LR along different Eigen-direction (making up for Newton's Method without having to compute Hessian)
	* Sensible to infrequent updates to certain weights.
	* Moves fast in the beginning, slows down later.

Generic Equation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. math:: w_{t+1}=w_t-\underbrace{\alpha_tH_t^{-1}\widehat{\nabla E_t}(w_t+\gamma_t(w_t-w_{t-1}))}_\text{adaptive gradient component}+\underbrace{\beta_tH_t^{-1}H_{t-1}(w_t-w_{t-1})}_\text{adaptive momemtum component}

.. note::
	* :math:`\widehat{\nabla E_t}` is an estimate of the gradient (stochastic/mini-batch estimate)
	* :math:`H_t=\sqrt{G_t}` is a diagonal matrix where :math:`G_t` is an approximation of the Hessian (only along major axes)
	* With :math:`H_t=I` we recover NAG.

AdaGrad
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	* Keep a running weighted average of the gradient magnitudes to set up the LR
	* :math:`G_t=G_{t-1}+D_t` where :math:`D_t` is a diagonal matrix with squared gradient component.
	* :math:`\gamma_t=0` and :math:`\beta_t=0`

.. warning::
	* Issues: accumulating gradients cause diminishing LR.

RMSProp
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	* Keep more importance to recently computed gradients.
	* :math:`G_t=(1-\beta)G_{t-1}+\beta D_t`.
	* :math:`\gamma_t=0` and :math:`\beta_t=0`

.. warning::
	* Issues: LR can get real close to 0.

Adam
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	RMSProp with momentum

.. tip::
	* Renormalizes the momentum and LR to keep things numerically stable.

***********************************************************************************
Managing Numerical Issues with Gradients
***********************************************************************************
Weight & Bias Initialisation
===================================================================================
Input normalisation
===================================================================================
Weight normalisation
===================================================================================
Batch Normalisation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Layer Normalisation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Resources
===================================================================================
.. note::
	* [iitb.ac.in] `CS769 Optimization in Machine Learning IIT Bombay 2024 <https://www.cse.iitb.ac.in/%7Eganesh/cs769/>`_

		* `Full Playlist on YT <https://www.youtube.com/playlist?list=PLyo3HAXSZD3yhIPf7Luk_ZHM_ss2fFCVV>`_
		* `Unified all GD variants <https://youtu.be/2QNquvof1WA?list=PLyo3HAXSZD3yhIPf7Luk_ZHM_ss2fFCVV&t=865>`_
	* [ruder.io] `An overview of gradient descent optimization algorithms <https://www.ruder.io/optimizing-gradient-descent/>`_
	* [math.stackexchange.com] `This SO post on understanding how adaptive methods try to estimate Hessian <https://math.stackexchange.com/a/2349067>`_
	* [medium.com] `A Visual Explanation of Gradient Descent Methods (Momentum, AdaGrad, RMSProp, Adam) <https://medium.com/towards-data-science/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c>`_
