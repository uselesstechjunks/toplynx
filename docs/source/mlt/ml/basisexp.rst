##################################################################################
Basis Expansion
##################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. warning::
	* We can use the established framework for linear models to move beyond linearity just by using non-linear transforms on the features.
	* We define the transforms as functions :math:`h:\mathbb{R}^d\mapsto\mathcal{\Phi}` where :math:`\mathcal{\Phi}` can be finite or infinite dimensional.
	* Since these methods expand the basis from :math:`d`-dimensions to :math:`\dim\mathcal{\Phi}`, these methods are called **basis expansion** methods.

.. tip::
	* For most of the methods, we consider the case where :math:`d=1`.

**********************************************************************************
Finite Dimensional Expansion
**********************************************************************************
.. note::
	* We define a finite set of transforms :math:`x\mapsto h_m(x)` where :math:`m=1,\cdots,M`.
	* Each observation is mapped to the transformed vector :math:`\mathbf{h}=(h_1(x),\cdots,h_M(x))^T`
	* We use the linear regression or logistic regression frameworks using these transforms

		* Regression: :math:`\hat{y}=f(x)=\sum_{i=1}^M \beta_mh_m(x)=\boldsymbol{\beta}^T\mathbf{h}`
		* Classification: :math:`\log\frac{\mathbb{P}(G=k|X=x)}{\mathbb{P}(G=K|X=x)}=\boldsymbol{\beta}^T\mathbf{h}`

Polynomial
==================================================================================
.. tip::
	* [`Taylor's series <https://en.wikipedia.org/wiki/Taylor_series>`_] An infinite power series can arbitrarily approximate any real-valued infinitely differentiable function around a point :math:`a`.

		.. math:: f(x)=f(a)+\frac{f'(a)}{2!}(x-a)+\frac{f''(a)}{3!}(x-a)^2+\cdots
	* [`Lagrange Polynomial <https://en.wikipedia.org/wiki/Lagrange_polynomial>`_] A polynomial of degree :math:`n` can be fit to pass through exactly :math:`n` points.

		* Say, we have a dataset :math:`\mathbf{X}=[x_1,\cdots,x_N]^T` and target :math:`\mathbf{y}=[y_1,\cdots,y_N]^T`.
		* For each point :math:`(x_k,y_k)`, we create the Lagrange coefficients

			.. math:: l_k(x)=\frac{(x-x_1)(x-x_2)\cdots(x-x_{k-1})(x-x_{k+1})\cdots(x-x_{N-1})(x-x_N)}{(x_k-x_1)(x_k-x_2)\cdots(x_k-x_{k-1})(x_k-x_{k+1})\cdots(x_k-x_{N-1})(x_k-x_N)}
		* The polynomial passing through all these points (interpolation) is given by

			.. math:: f(x)=\sum_{i=1}^N y_1 l_i(x)

.. warning::
	* A polynomial of degree :math:`n` has :math:`n-1` turns.
	* With real data, often not all the turns are utilised for the points in the interior regions.
	* With no other constraints applied, polynomials of such higher degrees oscilate crazily towards the extremes where the data density is usually low.
	* Therefore, in those regions, it provides very poor generalisation of the model.
	* The situation is often worse if we move to higher dimensions, as the proportion of points around the exterior increases in higher dimension.

.. note::
	* With these in mind, a polynomial of smaller degree often works best, such as cubic polynomials.

		.. math:: x\overset{h}\mapsto[1,x,x^2,x^3]
	* However, they often show higher error rate due to increased bias.

Piece-wise Polynomials
==================================================================================
.. note::
	* We can split the input domain into :math:`M` regions using **knots**, :math:`\xi_1,\cdots,\xi_{M-1}`.
	* We choose a function family :math:`\mathcal{H}` and fit it separately as :math:`h_i` in each region :math:`(\xi_{i-1},\xi_i)`.

.. warning::
	* The knot-selection problem has to be taken care of separately.
	* In all such cases, the functions often becomes discontinuous at the knot points which is undesirable for generalisation.

One-hot encoding for regions
----------------------------------------------------------------------------------
.. note::
	* In this case, the basis expansion works with indicator functions

		* :math:`h_1=I(x < \xi_1)`
		* For :math:`i=2,\cdots,M-1, h_i(x)=I(\xi_{i-1} < x < \xi_i)`
		* :math:`h_M(x)=I(\xi_M < x)`
	* For regression problems, the fitted co-efficients are the average of the observation in the current piece.

		.. math:: \hat{\beta_i}=\frac{\sum_{x_k\in(\xi_{i-1},\xi_i)}y_k}{|x_k\in(\xi_{i-1},\xi_i)|}

Polynomial function for regions
----------------------------------------------------------------------------------
.. note::
	* We can also design polynomials of any degree :math:`\mathcal{P}(n)` to each of these regions.

		* :math:`h_1=I(x < \xi_1)\mathcal{P}_1(n)`
		* For :math:`i=2,\cdots,M-1, h_i(x)=I(\xi_{i-1} < x < \xi_i)\mathcal{P}_i(n)`
		* :math:`h_M(x)=I(\xi_M < x)\mathcal{P}_M(n)`
	* Fitting this model with MSE for regression fits a separate polynomial in each region.
	* The fitted function is often discontinuous at the knot points.

Polynomial Spline
==================================================================================
.. note::
	* Here we design the functions for each region in such a way so that the function becomes continuous at each of the knot points.
	* The key idea is to **define additional polynomials of the same target degree** in such a way that, for any given knot-point, the function stays 0 to the left of it but continuously becomes non-zero on the right of it.
	* For cubic splines, the functions are defined as:

		* :math:`h_1(x)=1`
		* :math:`h_2(x)=x`
		* :math:`h_3(x)=x^2`
		* :math:`h_4(x)=x^3`
		* :math:`h_{k+4}(x)=(x-\xi_k)^3_+`, for :math:`k=1,\cdots,M` where

			.. math:: (x-\xi_k)^3_+=\begin{cases}0 & \text{if } x < \xi_k\\ (x-\xi_k)^3 & \text{if } x \ge \xi_k\end{cases}

Natural Spline
==================================================================================
.. note::
	* Since each region of a polynomial spline is fit with less data, often they show crazier behaviour near the boundaries than global polynomials.
	* To alleviate these problems, **natural splines** model the function as a linear function for the left of the leftmost and the right of the rightmost knot points.
	* We use the notation :math:`N_i(x)` instead of :math:`h_i(x)` to emphasize that we're working with natural splines.

.. tip::
	[TODO] Note on the number of parameters and degrees of freedom.

Smoothing Spline
==================================================================================
.. tip::
	* For each of the piece-wise fitting approaches, knot selection remains a key-issue.
	* Smoothing splines address this by implicitly **allowing a knot at every single data-point**.
	* Since this approach can potentially create much higher degree polynomials, the complexity of the model is controlled via regularisation.

.. note::
	* The functions are restricted to be twice-differentiable equipped with norm (e.g. `Sobolev space <https://en.wikipedia.org/wiki/Sobolev_space>`_ :math:`W^{k=2,p=2}`).

		* Any :math:`f\in W^{k,p}` and its (`weak <https://en.wikipedia.org/wiki/Weak_derivative>`_) derivatives up to order :math:`k` have a finite :math:`L_p` norm.
	* The objective function is defined as

		.. math:: \hat{f}=\min_{f\in W^{k=2,p=2}}\left[\sum_{i=1}^N(y-f(x))^2+\lambda\int\left(f''(z)\right)^2\mathop{dz}\right]
	* The smoothness is captured in the double-derivative since it represents curvature.
	* :math:`\lambda\in[0,\infty]` is a smoothing parameter which controls the model complexity

		* :math:`\lambda=0`: Rough fit, equivalent to interpolation using a Lagrange polynomial.
		* :math:`\lambda=\infty`: Linear fit since it reduces to an OLS problem with MSE loss.

.. note::
	* [TODO: Proof?] The solution for this is Natural splines

		.. math:: f(x)=\sum_{i=1}^N \beta_iN_i(x)
	* We have :math:`f''(z)=\sum_{i=1}^N \beta_iN''_i(z)` and 

		.. math:: \int\left(f''(z)\right)^2\mathop{dz}=\sum_{i=1}^N \sum_{j=1}^N\beta_i\beta_j\int N''_i(z)N''_j(z)\mathop{dz}=\boldsymbol{\beta}^T\boldsymbol{\Omega}_N\boldsymbol{\beta}
	* Here :math:`\{\boldsymbol{\Omega}_N\}_{i,j}=\int N''_i(z)N''_j(z)\mathop{dz}`
	* The objective function can therefore be written as a generalised ridge regression

		.. math:: \min_\beta\left[(\mathbf{y}-\mathbf{N}\boldsymbol{\beta})^T(\mathbf{y}-\mathbf{N}\boldsymbol{\beta})+\lambda\boldsymbol{\beta}^T\boldsymbol{\Omega}_N\boldsymbol{\beta}\right]
	* [TODO: Write the final solution]

Non-linear Classification
==================================================================================
.. note::
	* TODO

Moving Beyond 1 Dimension
==================================================================================
.. note::
	* TODO Tensor product

**********************************************************************************
Infinite Dimensional Expansion
**********************************************************************************
.. note::
	* More general regression problems can be formulated using a similar framework for the smoothing splines

		.. math:: \hat{f}=\min_{f\in\mathcal{H}}\left[\sum_{i=1}^NL(y,f(x))+\lambda J(f)\right]
	* Here :math:`\mathcal{H}` is a function class (hypothesis space) (e.g. Sobolev space for smoothing splines).
	* :math:`J(f)` is a regulariser which penalises functions for being too complex (to avoid overfitting).
	* :math:`\lambda` is the regulariser parameter which controls the trade-off between the bias and the variance.

A Point Mapping to a Function
==================================================================================
.. tip::
	* We usually think of points on the number line or in a higher dimensional space (:math:`x\in\mathbb{R}^d`) as a discrete entity.
	* Instead, a point can be thought of as an impulse like `Dirac-delta function <https://en.wikipedia.org/wiki/Dirac_delta_function>`_, which has infinitely high probability of being found at it's location.

		.. math:: \delta_x(t)=\begin{cases}+\infty & t=x \\ 0 & t\neq x\end{cases}

		* It has an infinitely sharp peak at :math:`x` but that dies off immediately to :math:`0` everywhere else.
		* [Side-node] This is a special case of `generalised functions <https://en.wikipedia.org/wiki/Generalized_function>`_ or distributions.
	* We can think of this impulse as a limit of a sequence of functions, :math:`\lim_\limits{\gamma\downarrow 0} f_\gamma`, such as Gaussian bumps around the point :math:`x`.

		.. math:: f_\gamma=\exp\left(-\gamma||x-x'||^2\right)
	* While thinking of a map from a point to a function, we're essentially going backwards from the impulse limit to a bump with a non-zero width.
	* Intuitively, this allows for some uncertainty about the exact location of the point.

Reproducing Kernel Hilbert Space
----------------------------------------------------------------------------------
.. note::
	* We choose `RKHS <https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space>`_ (:math:`\mathcal{H}_K`) to be our function class.
	* We define the basis expansion as a mapping from the original domain to this function space

		.. math:: x\in\mathbb{R}^d\overset{h}\longmapsto  (f:\mathbb{R}^d\mapsto\mathbb{R})\in\mathcal{H}_K
	* For each point :math:`x_i\in\mathbf{X}`, we map it to a function :math:`f_i(\cdot)\in\mathcal{H}_K` with a free parameter.

.. warning::
	* Let :math:`K` be the reproducing kernel of :math:`\mathcal{H}_K` with Dirac evaluation functional :math:`\delta_x:\mathcal{H}_K\mapsto\mathbb{R}`.
	* This means, for any :math:`f\in \mathcal{H}_K`, 

		.. math:: \delta_x(f)=\langle K(\cdot,x), f\rangle_{\mathcal{H}_K}=f(x)
	* We choose a specific mapping :math:`\varphi:\mathbb{R}^d\mapsto\mathcal{H}_K` such that it maps each point :math:`x` to this specific type of functions (canonical feature maps)

		.. math:: \varphi(x)=f_{\delta_x}=K(\cdot,x)
	* We note that the kernel also provides the tool to calculate the inner products (and, hence, similarity measure via a metric) between the points.

		.. math:: \langle \varphi_i(\cdot), \varphi_j(\cdot)\rangle_{{\mathcal{H}}_K}=\langle K(\cdot,x_i), K(\cdot,x_j)\rangle_{{\mathcal{H}}_K}=K(x_i,x_j)

.. note::
	* :math:`\mathcal{H}_K` contains (uncountably) infinite number of basis functions :math:`K(\cdot,x_i)` for :math:`i\in\mathcal{I}`.
	* Every function :math:`f\in\mathcal{H}_K` can be expressed using those basis

		.. math:: f(\cdot)=\sum_{i\in\mathcal{I}}\alpha_i K(\cdot,x_i)

Mercer Kernels: Eigen-decomposition of Linear Space Spanned by Kernel Functions
----------------------------------------------------------------------------------
.. note::
	* From a potentially uncountable number of basis functions, eigen-decomposition reduces the basis functions to a countable case.

		* Requires added structural assumption on the domain of :math:`x` and on the continuity of :math:`K`.
	* Assuming that the kernel has an eigen-decomposition with eigenfunctions :math:`(\phi_i)_{i=1}^\infty\in\mathcal{H}_K`, the kernel also be written in term of these

		.. math:: K(x,y)=\sum_{i=1}^\infty \gamma_i\phi_i(x)\phi_i(y)

		* We note that :math:`\varphi(x)\neq \phi(x)`, and therefore, this produces a different feature map.
	* Since kernels are **symmetric and positive definite**, the eigenvalues 

		* are positive, i.e. :math:`\gamma_i\ge 0`, and 
		* have bounded sum, i.e. :math:`\sum_{i=1}^\infty \gamma_i < \infty`
	* Any function :math:`f\in\mathcal{H}_K` can be expressed as a linear combination of the countable eigenfunctions

		.. math:: f(x)=\sum_{i=1}^\infty c_i\phi_i(x)

Kernel Ridge Regression
==================================================================================
.. note::
	* We use the function norm as the regulariser as this captures how vigorously the function oscilate along the direction of each eigenfunctions.

		.. math:: ||f||^2_{\mathcal{H}_K}\overset{\text{def}}=\sum_{i=1}^\infty c_i^2/\gamma_i < \infty

.. note::
	* The ridge regression problem using functions from kernel family can be expressed as 

		.. math:: \hat{f}=\min_{f\in\mathcal{H}}\left[\sum_{i=1}^N L(y,f(x_i))+\lambda ||f||^2_{\mathcal{H}_K}\right]
	* This reduces to 

		.. math:: \hat{f}=\min_{(c_k)_{k=1}^\infty}\left[\sum_{i=1}^N L(y,\left(\sum_{k=1}^\infty c_k\phi_k(x_i)\right))+\lambda \sum_{k=1}^\infty c_i^2/\gamma_i\right]
	* On a face-value, it seems that we'd need to estimate an infinite number of parameters.
	* [TODO: Proof?] However, the solution reduces to a finite dimensional one from a potentially uncountable one

		.. math:: f(x)=\sum_{i=1}^N\alpha_iK(x,x_i)=\mathbf{K}\boldsymbol{\alpha}
	* The regulariser term is defined as

		.. math:: \lambda||f||_{\mathcal{H}_K}^2=\lambda\langle f,f\rangle_{\mathcal{H}_K}=\sum_{i=1}^N\sum_{j=1}^N K(x_i,x_j)\alpha_i\alpha_j=\lambda\boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha}
	* The minimization problem therefore becomes

		.. math:: \hat{f}=\min_{\boldsymbol{\alpha}}L(\mathbf{y}, \mathbf{K}\boldsymbol{\alpha})+\lambda\boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha}
	* We note that for MSE loss, this reduces to a generalised ridge regression problem.

Connection to Smoothness
----------------------------------------------------------------------------------
.. warning::
	[TODO] Intuition: Lesser weights to more wiggly component functions in the solution.

.. seealso::
	* For intuitive understanding, here is an example:

		* We have 1-dimensioanl data matrix :math:`\mathbf{X}=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}1.1\\1.5\\2.1\end{bmatrix}`.
		* We define the map :math:`x_i\overset{h}\mapsto f_i(\cdot)` using `Gaussian RBF kernel <https://en.wikipedia.org/wiki/Radial_basis_function_kernel>`_.
		* We define our 3 basis functions as

			.. math:: \mathbf{h}=[K(\cdot,x_1),K(\cdot,x_2),K(\cdot,x_3)]=\begin{bmatrix}\exp(-\frac{||x-1.1||^2}{0.05}),\exp(-\frac{||x-1.5||^2}{0.05}),\exp(-\frac{||x-2.1||^2}{0.05})\end{bmatrix}
			
			* One linear combination of these functions is shown here in yellow.

				.. math:: f(x)=3h_1(x)+2h_2(x)-h_3(x)

			.. image:: ../../img/3.png
			  :width: 600
			  :alt: A linear combination of functions
		* Each datapoint :math:`x_i` is the mapped to :math:`\mathbf{h_i}=[K(x_i,x_1),K(x_i,x_2),K(x_i,x_3)]`.

Kernel Support Vector Machine
==================================================================================
