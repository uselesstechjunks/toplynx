
################################################################################
ML Depth
################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Fundamentals
********************************************************************************
Metrics
================================================================================
- Regression
- Classification
- Ranking

	- Online ranking: ROC-AUC, PR-AUC  
	- Offline selection: ROC-AUC  
- Generative Models

	- BLUE, ROUGE
	- GPT Labeling: Accuracy, Human-like 

Projects  
================================================================================
- Motivation
- Approach: Metrics, features, architecture  
- Challenges
- Outcome

Transformer-Based Models
================================================================================
Initialization
--------------------------------------------------------------------------------
- Xavier/He initialization  

Tokenizers
--------------------------------------------------------------------------------
- Byte Pair Encoding (BPE)  
- WordPiece  
- SentencePiece  

Activations
--------------------------------------------------------------------------------
- Sigmoid, Tanh, ReLU, LeakyReLU
- SiLU, GELU, Swish
- GLU, GeGLU, ReGLU, SwiGLU

Normalization
--------------------------------------------------------------------------------
- Internal covariate shift  
- Batch normalization  
- Layer normalization (LayerNorm)  
- RMS normalization (RMSNorm)  
- Pre-layer normalization (Pre-LN)  

Position Embedding Schemes
--------------------------------------------------------------------------------
- Learned positional encoding  
- Sinusoidal positional encoding  
- Rotary positional encoding  

Attention Mechanisms
--------------------------------------------------------------------------------
- Functions: Additive, dot product, scaled dot product  
- Architectures:  

	- Full (e.g., Flash, Paged, Ring)  
	- Sparse  
	- Multi-query attention  
	- Grouped-query attention  
- Techniques:  

	- Self-attention (SHA)  
	- Multi-head attention (MHA)  
	- Encoder attention  
	- Decoder attention with KV-cache  

Decoding Techniques
--------------------------------------------------------------------------------
- Beam search  
- Top-p, Top-k sampling  
- Temperature scaling  
- Speculative decoding  

Transformer Architectures
--------------------------------------------------------------------------------
- Models: PyTorch implementation

	- BERT: MLM
	- GPT: CLM  
	- T5: Seq2Seq
- Papers
	- Encoder: BERT, RoBERTa, XLM-R
	- Decoder: GPT, LLama3, DeepSeek
	- Seq2Seq: T5 Learnings, mT5
	- Autoencoder: BART
	- PLM: XLNet
	- RTD: Elextra
	- MoE: Mixtral
	- State-Space: Mamba
	- Retriever Embeddings: Generalizable T5-based Retriever (GTR)

Hardware and Optimization
--------------------------------------------------------------------------------
- Flash attention  
- Quantization (e.g., INT8 LLM)  
- Paged attention  
- Ring attention  

LLM Techniques
--------------------------------------------------------------------------------
- Prompt engineering  
- Prompt tuning  
- Retrieval-Augmented Generation (RAG)  
- LoRA, QLoRA
- Supervised Fine-Tuning (SFT)  
- Reinforcement Learning with Human Feedback (RLHF)  
- Proximal Policy Optimization (PPO)  
- Direct Preference Optimization (DPO)

Convolution-Based Models
================================================================================
- Convolution-based models: ConvNet, ResNet, Graph CN, LightGCN, Graph Transformers.  

Generative Models
================================================================================
- Generative models on latent variable space: VAE, VQVAE, GAN, diffusion models, diffusion transformers.  

Multimodal Models
================================================================================
TODO

Good to Know
================================================================================
- Popular NL tasks and remember the dataset names. 
- Kernel methods: kernel meaning embedding, MMD, other IPMs – read every inch of our paper.  
- Probability and statistics: parametric and non-parametric methods for inference, CI, and hypothesis testing framework.  
- Bayes Net (representation, inference, learning).  
- Causality – how to think systematically about finding the root cause of a problem; Bing search causality paper.  
- Latent variable models: K-means, mixture of Gaussians, PCA, kernel PCA, ICA.  
- Clustering: convex, non-convex, evaluation of clustering performance.  
- Regression and discriminative classification: model assumption, interpretation, evaluation – collinearity, the other stuff.  
- Theory as applied problem statement.  
- Code kernel methods, tree methods, regression, VAE, GAN, diffusion models.

Sample Questions
================================================================================
Question sections moved to the question bank in `docs/source/gen/interviews/qb.rst`.
