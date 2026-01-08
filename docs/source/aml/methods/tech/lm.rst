#########################################################################################
Language Understanding and Language Models
#########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

*****************************************************************************************
Activations
*****************************************************************************************
.. note::
	* [SiLU] `Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_
	* [GELU] `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_
	* [Swish] `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941v2>`_	
	* [Swish v GELU] `On the Disparity Between Swish and GELU <https://towardsdatascience.com/on-the-disparity-between-swish-and-gelu-1ddde902d64b>`_
	* [GLU] `GLU Variants Improve Transformer <https://arxiv.org/pdf/2002.05202v1>`_

*****************************************************************************************
Normalisation
*****************************************************************************************
* [Internal Covariate Shift][BN] `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
* [LN] `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
* [RMSNorm] `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`_
* [PreLN][Detailed Study with Mean-Field Theory] `On Layer Normalization in the Transformer Architecture <https://arxiv.org/abs/2002.04745>`_

.. warning::
	For theoretical understanding of MFT and NTK, start from this MLSS video `here <https://youtu.be/rzPHnBGmr_E?si=JifFfB9r0Ax373VR>`_.

*****************************************************************************************
Tokenizers
*****************************************************************************************
WordPiece
=========================================================================================
.. seealso::
	`Google's Neural Machine Translation System <https://arxiv.org/abs/1609.08144v2>`_

SentencePiece
=========================================================================================
.. seealso::
	`SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing <https://arxiv.org/abs/1808.06226>`_

Byte-Pair Encoding (BPE)
=========================================================================================
.. seealso::
	`Neural Machine Translation of Rare Words with Subword Units <https://arxiv.org/abs/1508.07909v5>`_

*****************************************************************************************
Word Embeddings
*****************************************************************************************
.. note::
	* Word2Vec: Efficient Estimation of Word Representations in Vector Space
	* GloVe: Global Vectors forWord Representation
	* Evaluation methods for unsupervised word embeddings

*****************************************************************************************
Sequence Modeling
*****************************************************************************************
RNN
=========================================================================================
.. seealso::
	Implementation examples live in :doc:`/code/language-modeling-impls`.

.. note::
	* `On the diffculty of training Recurrent Neural Networks <https://arxiv.org/abs/1211.5063>`_
	* `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_
	* `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473>`_

LSTM
=========================================================================================
.. seealso::
	Implementation examples live in :doc:`/code/language-modeling-impls`.

.. note::
	* `StatQuest on LSTM <https://www.youtube.com/watch?v=YCzL96nL7j0>`_

*****************************************************************************************
Transformer
*****************************************************************************************
General Resources
=========================================================================================
.. warning::
	* [github.com] `LLM101n: Let's build a Storyteller <https://github.com/karpathy/LLM101n>`_
	* [jmlr.org] `Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity <https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf>`_
	* [epoch.ai] `How has DeepSeek improved the Transformer architecture? <https://epoch.ai/gradient-updates/how-has-deepseek-improved-the-transformer-architecture>`_

.. note::
	* [harvard.edu] `The Annotated Transformer <https://nlp.seas.harvard.edu/annotated-transformer/>`_
	* [jalammar.github.io] `The Illustrated Transformer <https://jalammar.github.io/illustrated-transformer/>`_
	* [lilianweng.github.io] `Attention? Attention! <https://lilianweng.github.io/posts/2018-06-24-attention/>`_
	* [d2l.ai] `The Transformer Architecture <https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html>`_
	* [newsletter.languagemodels.co] `The Illustrated DeepSeek-R1: A recipe for reasoning LLMs <https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1>`_

Position Encoding
=========================================================================================
.. note::
	* [arxiv.org] `Position Information in Transformers: An Overview <https://arxiv.org/abs/2102.11090>`_
	* [arxiv.org] `Rethinking Positional Encoding in Language Pre-training <https://arxiv.org/abs/2006.15595>`_
	* [eleuther.ai] `RoPE <https://blog.eleuther.ai/rotary-embeddings/>`_
	* [arxiv.org] `LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens <https://arxiv.org/abs/2402.13753>`_
	* [arxiv.org] `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_

Attention
=========================================================================================
Understanding Einsum
-----------------------------------------------------------------------------------------
.. warning::
	Implementation examples live in :doc:`/code/language-modeling-impls`.

.. note::
	* Attention implementation examples live in :doc:`/code/language-modeling-impls`.

UnitTest
-----------------------------------------------------------------------------------------
.. seealso::
	Unit tests live in :doc:`/code/language-modeling-impls`.

Resources
-----------------------------------------------------------------------------------------
* [MHA] `Attention Is All You Need <https://arxiv.org/abs/1706.03762v7>`_
* [MQA] `Fast Transformer Decoding: One Write-Head is All You Need <https://arxiv.org/abs/1911.02150>`_
* [GQA] `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints <https://arxiv.org/abs/2305.13245v3>`_
* [tinkerd.net] `Multi-Query & Grouped-Query Attention <https://tinkerd.net/blog/machine-learning/multi-query-attention/>`_

Decoding
=========================================================================================
* Beam Search, Top-K, Top-p/Nuclear, Temperature
* [mlabonne.github.io] `Decoding Strategies in Large Language Models <https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html>`_
* Speculative Deocding

*****************************************************************************************
Transformer Architecture
*****************************************************************************************
Encoder [BERT]
=========================================================================================
.. note::
	* BERT: `Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_
	* Additional Resources

		* [tinkerd.net] `BERT Tokenization <https://tinkerd.net/blog/machine-learning/bert-tokenization/>`_
		* [tinkerd.net] `BERT Embeddings <https://tinkerd.net/blog/machine-learning/bert-embeddings/>`_, 
		* [tinkerd.net] `BERT Encoder Layer <https://tinkerd.net/blog/machine-learning/bert-encoder/>`_
	* `A Primer in BERTology: What we know about how BERT works <https://arxiv.org/abs/2002.12327>`_
	* RoBERTa: `A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_
	* XLM: `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_
	* TwinBERT: `Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval <https://arxiv.org/abs/2002.06275>`_

Decoder [GPT]
=========================================================================================
.. note::
	* [jalammar.github.io] `The Illustrated GPT-2 <https://jalammar.github.io/illustrated-gpt2/>`_
	* [github.com] `karpathy/nanoGPT <https://github.com/karpathy/nanoGPT>`_
	* [cameronrwolfe.substack.com] `Decoder-Only Transformers: The Workhorse of Generative LLMs <https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse>`_
	* [openai.com] `GPT-2: Language Models are Unsupervised Multitask Learners <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
	* [openai.com] `GPT-3: Language Models are Few-Shot Learners <https://arxiv.org/abs/2005.14165>`_

Encoder-Decoder [T5]
=========================================================================================
.. note::
	* T5: `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/abs/1910.10683>`_

Autoencoder [BART]
=========================================================================================
.. note::
	* BART: `Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension <https://arxiv.org/abs/1910.13461>`_

Cross-Lingual
=========================================================================================
.. note::
	* [Encoder] XLM-R [Roberta]: `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`_
	* [Decoder] XGLM [GPT-3]: `Few-shot Learning with Multilingual Generative Language Models <https://arxiv.org/abs/2112.10668>`_
	* [Encoder-Decoder] mT5 [T5]: `A Massively Multilingual Pre-trained Text-to-Text Transformer <https://arxiv.org/abs/2010.11934>`_
	* [Autoencoder] mBART [BART]: `Multilingual Denoising Pre-training for Neural Machine Translation <https://arxiv.org/abs/2001.08210>`_

.. seealso::
	* `[ruder.io] The State of Multilingual AI <https://www.ruder.io/state-of-multilingual-ai/>`_
