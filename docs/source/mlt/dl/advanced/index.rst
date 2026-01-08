#####################################################################################
Advanced Topics
#####################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. warning::
	* [mbernste.github.io] `The evidence lower bound (ELBO) <https://mbernste.github.io/posts/elbo/>`_
	* [mbernste.github.io] `Expectation-maximization: theory and intuition <https://mbernste.github.io/posts/em/>`_
	* [mbernste.github.io] `Variational inference <https://mbernste.github.io/posts/variational_inference/>`_

*************************************************************************************
Energy Based Model
*************************************************************************************
.. note::
	* `How to Train Your Energy-Based Models <https://arxiv.org/abs/2101.03288>`_
	* `Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One <https://arxiv.org/abs/1912.03263>`_

*************************************************************************************
Generative Adversarial Network
*************************************************************************************
.. note::
	* [GAN] `Generative Adversarial Networks <https://arxiv.org/abs/1406.2661>`_
	* [GMMM] `Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy <https://arxiv.org/abs/1611.04488>`_

.. seealso::
	* [VQ-GAN] `Taming Transformers for High-Resolution Image Synthesis <https://arxiv.org/abs/2012.09841>`_
	* [MMD-GAN] `Demystifying MMD GANs <https://arxiv.org/abs/1801.01401>`_
	* [Sutherland] `Kernel Distances for Better Deep Generative Models <https://www.gatsby.ucl.ac.uk/~dougals/slides/mmd-gans-gpss/#/>`_

*************************************************************************************
Variational Autoencoders
*************************************************************************************
.. note::
	* `Semi-Supervised Learning with Deep Generative Models <https://arxiv.org/abs/1406.5298>`_
	* [VAE] `An Introduction to Variational Autoencoders <https://arxiv.org/abs/1906.02691>`_
	* [dVAE] `The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables <https://arxiv.org/abs/1611.00712>`_
	* [VQ-VAE] `Neural Discrete Representation Learning <https://arxiv.org/abs/1711.00937>`_
	* [VQ-VAE2] `Generating Diverse High-Fidelity Images with VQ-VAE-2 <https://arxiv.org/abs/1906.00446>`_

.. seealso::
	* [mbernste.github.io] `Variational autoencoders <https://mbernste.github.io/posts/vae/>`_
	* [Dall-E] `Zero-Shot Text-to-Image Generation <https://arxiv.org/abs/2102.12092>`_

*************************************************************************************
Diffusion
*************************************************************************************
.. note::
	* [DDPM] `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_
	* [OpenAI] `Improved Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2102.09672>`_
	* [LDM] `High-Resolution Image Synthesis with Latent Diffusion Models <https://arxiv.org/abs/2112.10752>`_
	* [DiT] `Scalable Diffusion Models with Transformers <https://arxiv.org/abs/2212.09748>`_

.. seealso::
	* [mbernste.github.io] Denoising diffusion probabilistic models `Part 1 <https://mbernste.github.io/posts/diffusion_part1/>`_, `Part 2 <https://mbernste.github.io/posts/diffusion_part2/>`_
	* [jalammar.github.io] `The Illustrated Stable Diffusion <https://jalammar.github.io/illustrated-stable-diffusion/>`_
	* [stability.ai] `Generative Models by Stability AI <https://github.com/Stability-AI/generative-models>`_
	* [Github] `Stable Diffusion Repo <https://github.com/CompVis/stable-diffusion>`_

*************************************************************************************
Application
*************************************************************************************
Variational methods in machine learning offer powerful tools for addressing a variety of practical problems encountered with real data. Here are some practical areas where variational methods can be applied, along with examples of how they can tackle common challenges:

1. **Probabilistic PCA (Principal Component Analysis)**:
   - **Application**: Dimensionality reduction while accounting for noise and uncertainty in the data.
   - **Challenges Addressed**: Dealing with noisy data, estimating latent variables when some data are missing or incomplete.

2. **Mixture Models**:
   - **Application**: Clustering data into groups with underlying probabilistic models.
   - **Challenges Addressed**: Handling multimodal distributions in the data, identifying clusters in the presence of noise and uncertainty.

3. **Latent Variable Models**:
   - **Application**: Discovering hidden factors or latent variables that explain observed data.
   - **Challenges Addressed**: Dealing with unobserved variables, inferring missing data, and modeling complex dependencies in high-dimensional data.

4. **Variational Inference**:
   - **Application**: Approximating complex posterior distributions in Bayesian inference.
   - **Challenges Addressed**: Addressing intractable likelihoods or posterior distributions, scaling Bayesian methods to large datasets.

5. **Variational Autoencoders (VAEs)**:
   - **Application**: Learning latent representations of data for tasks like generative modeling and data compression.
   - **Challenges Addressed**: Handling high-dimensional and complex data, learning meaningful representations in an unsupervised or semi-supervised manner, dealing with missing or corrupted data.

6. **Denoising Probabilistic Diffusion Models**:
   - **Application**: Modeling temporal data with noise and uncertainty, such as video processing or sequential data analysis.
   - **Challenges Addressed**: Handling noisy or corrupted data, inferring missing information in time series or sequential data.

Addressing Common Problems with Variational Methods:

- **Unlabelled Data**: Variational methods can perform unsupervised learning to discover hidden patterns or structure in data without relying on explicit labels (e.g., VAEs for learning latent representations).

- **Partially Labelled Data**: Semi-supervised learning techniques can utilize both labelled and unlabelled data to improve model performance, leveraging variational inference for efficient learning.

- **Corrupted by Noise**: Models such as denoising VAEs or diffusion models can effectively denoise data, learning to distinguish signal from noise and reconstruct clean representations.

- **Sparsity**: Variational methods can handle sparse data by incorporating prior distributions or regularization techniques that encourage sparse representations.

- **Multimodality**: Mixture models and advanced VAE architectures can capture multimodal distributions in the data, allowing the model to represent diverse outcomes or clusters.

- **Intractable Likelihood**: Variational inference provides a framework for approximating complex posterior distributions or intractable likelihoods, making Bayesian methods feasible for large-scale data analysis.

Each of these methods relies on variational techniques to optimize model parameters and approximate posterior distributions, balancing model complexity and computational feasibility. They are particularly valuable in scenarios where data are noisy, incomplete, or exhibit complex dependencies that traditional methods struggle to capture. By leveraging variational methods, practitioners can enhance the robustness and flexibility of their machine learning models across a wide range of real-world applications.
