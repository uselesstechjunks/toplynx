#########################################################################################
Large Language Models
#########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

*****************************************************************************************
Training
*****************************************************************************************
* Training: `DeepSpeed <https://www.deepspeed.ai/training/>`_
* Inference: `DeepSpeed <https://www.deepspeed.ai/inference/>`_, `vLLM <https://docs.vllm.ai/en/latest/index.html>`_
* `The Ultra-Scale Playbook: Training LLMs on GPU Clusters <https://huggingface.co/spaces/nanotron/ultrascale-playbook>`_

Engineering
=========================================================================================
Scaling Large Models
-----------------------------------------------------------------------------------------
* [github.io] `How To Scale Your Model <https://jax-ml.github.io/scaling-book/index>`_
* [mlsyscourse.org] `CMU: 15-442/15-642: Machine Learning Systems <https://mlsyscourse.org/>`_

Quantization
-----------------------------------------------------------------------------------------
* [huggingface.co] `Bits and bytes <https://huggingface.co/docs/bitsandbytes/index>`_

Caching
-----------------------------------------------------------------------------------------
* [arxiv.org] `Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs <https://arxiv.org/html/2310.01801v4>`_

Data Engineering
-----------------------------------------------------------------------------------------
* [github.com] `LLMDataHub: Awesome Datasets for LLM Training <https://github.com/Zjh-819/LLMDataHub>`_
* [arxiv.org] `The Pile: An 800GB Dataset of Diverse Text for Language Modeling <https://arxiv.org/abs/2101.00027>`_	

Hardware Utilisation
-----------------------------------------------------------------------------------------
* [horace.io] `Making Deep Learning Go Brrrr From First Principles <https://horace.io/brrr_intro.html>`_
* [newsletter.maartengrootendorst.com] `A Visual Guide to Quantization <https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization>`_
* [nvidia.com] `Profiling PyTorch Models for NVIDIA GPUs <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31644/>`_
* [pytorch.org] `What Every User Should Know About Mixed Precision Training in PyTorch <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>`_
* [pytorch.org] `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
* [arxiv.org] `Hardware Acceleration of LLMs: A comprehensive survey and comparison <https://arxiv.org/pdf/2409.03384>`_

Pipelines
-----------------------------------------------------------------------------------------
* [huggingface] `LLM Inference at scale with TGI <https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi>`_
* [vLLM] `Easy, Fast, and Cheap LLM Serving with PagedAttention <https://blog.vllm.ai/2023/06/20/vllm.html>`_
* [HuggingFace Blog] `Fine-tuning LLMs to 1.58bit: extreme quantization made easy <https://huggingface.co/blog/1_58_llm_extreme_quantization>`_
* [Paper] `Data Movement Is All You Need: A Case Study on Optimizing Transformers <https://arxiv.org/abs/2007.00072>`_

Tools
-----------------------------------------------------------------------------------------
.. important::
	* [pytorch.org] `PyTorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
	* [tinkerd.net] `Writing CUDA Kernels for PyTorch <https://tinkerd.net/blog/machine-learning/cuda-basics/>`_
	* [spaCy] `Library for NLU/IE Tasks <https://spacy.io/usage/spacy-101>`_, `LLM-variants <https://spacy.io/usage/large-language-models>`_
	* [tinkerd.net] `Distributed Training and DeepSpeed <https://tinkerd.net/blog/machine-learning/distributed-training/>`_

Objectives
=========================================================================================
Pretraining
-----------------------------------------------------------------------------------------
* Improving Language Understanding by Generative Pre-Training
* Universal Language Model Fine-tuning for Text Classification

Domain-Adaptation
-----------------------------------------------------------------------------------------
* SoDA
* [arxiv.org] `LIMO: Less is More for Reasoning <https://arxiv.org/abs/2502.03387>`_

Instruction Fine-Tuning (IFT)
-----------------------------------------------------------------------------------------
Datasets: NaturalInstructions: https://github.com/allenai/natural-instructions/

Supervised Fine-Tuning (SFT)
-----------------------------------------------------------------------------------------
Datasets: UltraChat: https://github.com/thunlp/UltraChat

Preference Optimisation (PO)
-----------------------------------------------------------------------------------------
* Datasets: Ultrafeedback: https://huggingface.co/datasets/argilla/ultrafeedback-curated
* [huggingface.co] `Huggingface TRL <https://huggingface.co/docs/trl/index>`_

Reinforcement Learning with Human Feedback (RLHF)/Proximal Policy Optimisation (PPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [github.io] `The 37 Implementation Details of Proximal Policy Optimization <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>`_
* [arxiv.org] `SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training <https://arxiv.org/abs/2501.17161v1>`_

Direct Preference Optimisation (DPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reinforcement Fine-Tuning (RFT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [philschmid.de] `Bite: How Deepseek R1 was trained <https://www.philschmid.de/deepseek-r1>`_
* [arxiv.org] `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models <https://arxiv.org/abs/2402.03300>`_
* [predibase.com] `How Reinforcement Learning Beats Supervised Fine-Tuning When Data is Scarce <https://predibase.com/blog/how-reinforcement-learning-beats-supervised-fine-tuning-when-data-is-scarce>`_

Long Context LLMs
=========================================================================================
.. csv-table:: 
	:header: "Category","Model","Max sequence length"
	:align: center

		Full Attention,Flash Attention,Not specified
		Augmented Attention,Transformer-XL,Up to 16k tokens (depends on the segment length)
		Augmented Attention,Longformer,Up to 4k tokens
		Recurrence,RMT,Not specified
		Recurrence,xLSTM,Not specified
		Recurrence,Feedback Attention,Not specified
		State Space,Mamba,Not specified
		State Space,Jamba,Not specified

Optimized Full Attention
-----------------------------------------------------------------------------------------
* Flash Attention

Augmented Attention
-----------------------------------------------------------------------------------------
* Receptive Field Modification: Transformer-xl
* Sparse Attention: Longformer

Recurrence
-----------------------------------------------------------------------------------------
* RMT: Recurrent Memory Transformer
* Feedback Attention

Non Transformer
-----------------------------------------------------------------------------------------
* State SpaceModels: Mamba, Jamba

	.. note::
		* [Mamba] `Linear-Time Sequence Modeling with Selective State Spaces <https://arxiv.org/abs/2312.00752>`_
		* `Understanding State Space Models <https://tinkerd.net/blog/machine-learning/state-space-models/>`_

* LSTM: xLSTM

Retrieval Augmented
-----------------------------------------------------------------------------------------
* Bidirectional Attention for encoder: BERT, T5, Electra, Matryoshka, Multimodal

	* Approximate Nearest Neighbour Search
* Causal attention for decoder: GPT, Multimodal generation

Pruning
-----------------------------------------------------------------------------------------
* LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference

Special Techniques
=========================================================================================
Low-Rank Approximations (LoRA)
-----------------------------------------------------------------------------------------
* [huggingface.co] `Performance Efficient Fine-Tuning <https://huggingface.co/docs/peft/index>`_
* [tinkerd.net] `Language Model Fine-Tuning with LoRA <https://tinkerd.net/blog/machine-learning/lora/>`_

Mixture of Experts
-----------------------------------------------------------------------------------------
* [tinkerd.net] `Mixture of Experts Pattern for Transformer Models <https://tinkerd.net/blog/machine-learning/mixture-of-experts/>`_
* Mixtral

Logit Bias
-----------------------------------------------------------------------------------------
Goal: Influence the output probabilities of a language model (LLM) to steer it towards a desired output, such as a "yes" or "no" answer.

	#. Logit Adjustment
	
		- Each token in the vocabulary has an associated logit value.
		- By adding a bias to the logits of specific tokens, you can increase or decrease the likelihood that those tokens will be selected when the model generates text.
	
	#. Softmax Function
	
		- After adjusting the logits, the softmax function is applied to convert these logits into probabilities.
		- Tokens with higher logits will have higher probabilities of being selected.

Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Identify Token IDs

- Determine the token IDs for "yes" and "no" in the model's vocabulary. For instance, suppose "yes" is token ID 345 and "no" is token ID 678.
#. Apply Bias

	- Adjust the logits for these tokens. Typically, you would add a positive bias to both "yes" and "no" tokens to increase their probabilities and/or subtract a bias from all other tokens to decrease their probabilities.
#. Implementing the Bias

	- If using an API or library that supports logit bias (e.g., OpenAI GPT-3), you can specify the bias directly in the request.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: json

	{
	  "prompt": "Is the sky blue?",
	  "logit_bias": {
		"345": 10,  // Bias for "yes"
		"678": 10   // Bias for "no"
	  }
	}

Practical Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Magnitude of Bias

	- The magnitude of the bias determines how strongly the model will favor "yes" or "no." 
	- A larger bias will make the model more likely to choose these tokens.

#. Context Sensitivity

	- The model may still consider the context of the prompt. If the context strongly indicates one answer over the other, the model may lean towards that answer even with a bias.

3. Balanced Bias

	- If you want the model to have an equal chance of saying "yes" or "no," you can apply equal positive biases to both tokens. If you want to skew the response towards one answer, apply a larger bias to that token.

Example in Practice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Consider a scenario where you want the model to respond with "yes" or "no" to the question "Is the sky blue?"
* This setup ensures that the model will highly favor "yes" and "no" as possible outputs. The prompt and biases are designed so that "yes" or "no" are the most likely completions.

.. collapse:: API Implementation Example
	Here's a pseudo-code example of how you might implement this with an API:
	
	.. code-block:: python
	
		import openai
		
		response = openai.Completion.create(
			engine="text-davinci-003",
			prompt="Is the sky blue?",
			max_tokens=1,
			logit_bias={"345": 10, "678": 10}
		)
		
		print(response.choices[0].text.strip())
	
	In this example:
	- The `prompt` is set to "Is the sky blue?"
	- The `logit_bias` dictionary adjusts the logits for the "yes" and "no" tokens to be higher.
	- The `max_tokens` is set to 1 to ensure only one word is generated.
	- By using logit bias in this way, you can guide the LLM to produce a "yes" or "no" answer more reliably.
	
Resources
=========================================================================================
* [openai.com] `OpenAI Docs <https://platform.openai.com/docs/overview>`_
* [HN] `You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
* [Ask HN] `Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
* [HN] `Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_
* [magazine.sebastianraschka.com] `Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
* [Github] `LLM Course <https://github.com/mlabonne/llm-course>`_

*****************************************************************************************
Applied LLMs
*****************************************************************************************
Prompt Engineering
=========================================================================================
Practical
-----------------------------------------------------------------------------------------
* [prompthub.us] `PromptHub Blog <https://www.prompthub.us/blog>`_
* [promptingguide.ai] `Prompt Engineering Guide <https://www.promptingguide.ai/>`_
* [youtube.com] Nice video from OpenAi - https://youtu.be/ahnGLM-RC1Y?si=irFR4SoEfrEzyPh9

Techniques
-----------------------------------------------------------------------------------------
#. [prompthub.us] `The Difference Between System Messages and User Messages in Prompt Engineering <https://www.prompthub.us/blog/the-difference-between-system-messages-and-user-messages-in-prompt-engineering>`_
#. [prompthub.us] `Role-Prompting: Does Adding Personas to Your Prompts Really Make a Difference? <https://www.prompthub.us/blog/role-prompting-does-adding-personas-to-your-prompts-really-make-a-difference>`_
#. [prompthub.us] `Chain of Thought Prompting Guide <https://www.prompthub.us/blog/chain-of-thought-prompting-guide>`_
#. [promptingguide.ai] `Reflexion <https://www.promptingguide.ai/techniques/reflexion>`_
#. [prompthub.us] `Least-to-Most Prompting Guide <https://www.prompthub.us/blog/least-to-most-prompting-guide>`_
#. [prompthub.us] `Prompt Chaining Guide <https://www.prompthub.us/blog/prompt-chaining-guide>`_
#. [prompthub.us] `Fine-Tuning vs Prompt Engineering <https://www.prompthub.us/blog/fine-tuning-vs-prompt-engineering>`_

In Context Learning (ICL)
-----------------------------------------------------------------------------------------
#. [prompthub.us] `The Few Shot Prompting Guide <https://www.prompthub.us/blog/the-few-shot-prompting-guide>`_
#. [prompthub.us] `In Context Learning Guide <https://www.prompthub.us/blog/in-context-learning-guide>`_

Optimisation
-----------------------------------------------------------------------------------------
#. [prompthub.us] `Prompt Caching with OpenAI, Anthropic, and Google Models <https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models>`_
#. [prompthub.us] `Using LLMs to Optimize Your Prompts <https://www.prompthub.us/blog/using-llms-to-optimize-your-prompts>`_
#. [prompthub.us] `How to Optimize Long Prompts <https://www.prompthub.us/blog/how-to-optimize-long-prompts>`_
#. [prompthub.us] `Using Reinforcement Learning and LLMs to Optimize Prompts <https://www.prompthub.us/blog/using-reinforcement-learning-and-llms-to-optimize-prompts>`_

Best Practices
-----------------------------------------------------------------------------------------
#. [prompthub.us] `10 Best Practices for Prompt Engineering with Any Model <https://www.prompthub.us/blog/10-best-practices-for-prompt-engineering-with-any-model>`_
#. [prompthub.us] `Prompt Engineering Principles for 2024 <https://www.prompthub.us/blog/prompt-engineering-principles-for-2024>`_
#. [prompthub.us] `One Size Does Not Fit All: An Analaysis of Model Specific Prompting Strategies <https://www.prompthub.us/blog/one-size-does-not-fit-all-an-analaysis-of-model-specific-prompting-strategies>`_

Application Specific
-----------------------------------------------------------------------------------------
#. [prompthub.us] `Better Summarization with Chain of Density Prompting <https://www.prompthub.us/blog/better-summarization-with-chain-of-density-prompting>`_
#. [prompthub.us] `Prompt Engineering for Content Creation <https://www.prompthub.us/blog/prompt-engineering-for-content-creation>`_
#. [prompthub.us] `RecPrompt: A Prompt Engineering Framework for LLM Recommendations <https://www.prompthub.us/blog/recprompt-a-prompt-engineering-framework-for-llm-recommendations>`_
#. [prompthub.us] `Prompt Engineering for AI Agents <https://www.prompthub.us/blog/prompt-engineering-for-ai-agents>`_

Academic
-----------------------------------------------------------------------------------------
* [arxiv.org][CMU] `Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing <https://arxiv.org/abs/2107.13586>`_
* [arxiv.org] `Reflexion: Language Agents with Verbal Reinforcement Learning <https://arxiv.org/abs/2303.11366>`_
* [arxiv.org] `Chain-of-Thought Prompting Elicits Reasoning in Large Language Models <https://arxiv.org/abs/2201.11903>`_
* [aclanthology.org] `Diverse Demonstrations Improve In-context Compositional Generalization <https://aclanthology.org/2023.acl-long.78.pdf>`_
* [arxiv.org] `A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications <https://arxiv.org/abs/2402.07927>`_
* [arxiv.org] `The Prompt Report: A Systematic Survey of Prompting Techniques <https://arxiv.org/abs/2406.06608>`_
* [arxiv.org] `Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine <https://arxiv.org/abs/2311.16452>`_

	- Zero-shot
	- Random few-shot
	- Random few-shot, chain-of-thought
	- kNN, few-shot, chain-of-though
	- Ensemble w/ choice shuffle
* Key techniques/papers

	- FewShot
	- CoT
	- ReAct: Synergizing Reasoning and Acting in Language Models
	- Reflextion
	- Self-instruct: Aligning Language Models with Self-Generated Instructions
	- PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs
	- Prompt Tuning: The Power of Scale for Parameter-Efficient Prompt Tuning

Embeddings for Retrieval
=========================================================================================
* [techtarget.com] `Embedding models for semantic search: A guide <https://www.techtarget.com/searchenterpriseai/tip/Embedding-models-for-semantic-search-A-guide>`_

Evaluation
-----------------------------------------------------------------------------------------
* [openreview.net] `BEIR <https://openreview.net/pdf?id=wCu6T5xFjeJ>`_
* [arxiv.org] `MTEB <https://arxiv.org/pdf/2210.07316>`_
* For speech and vision, refer to the guide above from TechTarget.

Modeling
-----------------------------------------------------------------------------------------
* [arxiv.org] `Dense Passage Retrieval for Open-Domain Question Answering <https://arxiv.org/abs/2004.04906>`_
* [sbert.net] `SBERT <https://sbert.net/docs/sentence_transformer/pretrained_models.html>`_
* [arxiv.org][Google GTR - T5 Based] `Large Dual Encoders Are Generalizable Retrievers <https://arxiv.org/pdf/2112.07899>`_
* [arxiv.org][`Microsoft E5 <https://github.com/microsoft/unilm/tree/master/e5>`_] `Improving Text Embeddings with Large Language Models <https://arxiv.org/pdf/2401.00368>`_
* [cohere.com][Cohere - Better Perf on RAG] `Embed v3 <https://cohere.com/blog/introducing-embed-v3>`_
* [arxiv.org] SPLADE: `SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval <https://arxiv.org/pdf/2109.10086>`_
* [arxiv.org][Meta] DRAGON: `How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval <https://arxiv.org/pdf/2302.07452>`_
* [huggingface.co] `Matryoshka (Russian Doll) Embeddings <https://huggingface.co/blog/matryoshka>`_ - learning embeddings of different dimensions

Tech
-----------------------------------------------------------------------------------------
Vector DB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [youtube.com] `Pinecone: YouTube Playlist <https://youtube.com/playlist?list=PLRLVhGQeJDTLiw-ZJpgUtZW-bseS2gq9-&si=UBRFgChTmNnddLAt>`_
* Chroma, Weaviate

RAG Focused
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [youtube.com] `LlamaIndex <https://www.llamaindex.ai/>`_: `YouTube Channel <https://www.youtube.com/@LlamaIndex>`_
* [llamaindex.ai] `[LlamaIndex] Structured Hierarchical Retrieval <https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/#structured-hierarchical-retrieval>`_
* [llamaindex.ai] `Child-Parent Recursive Retriever <https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/>`_	

Retrieval Augmented Generation (RAG)
=========================================================================================
* [youtube.com][Stanford] `Stanford CS25: V3 I Retrieval Augmented Language Models <https://www.youtube.com/watch?v=mE7IDf2SmJg>`_
* [arxiv.org] `Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG <https://arxiv.org/abs/2501.09136>`_

Fundamentals
-----------------------------------------------------------------------------------------
* [promptingguide.ai] `Retrieval Augmented Generation (RAG) for LLMs <https://www.promptingguide.ai/research/rag>`_
* [huggingface.co] `RAG paper - RAG Doc <https://huggingface.co/docs/transformers/main/en/model_doc/rag#rag>`_
* [nvidia.com] `RAG 101: Demystifying Retrieval-Augmented Generation Pipelines <https://resources.nvidia.com/en-us-ai-large-language-models/demystifying-rag-blog>`_
* [nvidia.com] `RAG 101: Retrieval-Augmented Generation Questions Answered <https://developer.nvidia.com/blog/rag-101-retrieval-augmented-generation-questions-answered/>`_
* [arxiv.org][MSR] `From Local to Global: A Graph RAG Approach to Query-Focused Summarization <https://arxiv.org/pdf/2404.16130>`_
* [neo4j.com] `The GraphRAG Manifesto: Adding Knowledge to GenAI <https://neo4j.com/blog/graphrag-manifesto/>`_

RAG Eval
-----------------------------------------------------------------------------------------
* [arxiv.org] RAGAS: `Automated Evaluation of Retrieval Augmented Generation <https://arxiv.org/abs/2309.15217>`_
* [arxiv.org] RAGChecker: `A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation <https://arxiv.org/abs/2408.08067>`_

Practical RAG
-----------------------------------------------------------------------------------------
* [arxiv.org] `Improving Retrieval for RAG based Question Answering Models on Financial Documents <https://arxiv.org/pdf/2404.07221>`_
* [community.aws] `Techniques to Enhance Retrieval Augmented Generation (RAG) <https://community.aws/content/2gp2m3BJcl9mSMWT6njCIQNiz0e/techniques-to-enhance-retrieval-augmented-generation-rag?lang=en>`_	
* [medium.com] `Optimizing Retrieval for RAG Applications: Enhancing Contextual Knowledge in LLMs <https://dxiaochuan.medium.com/optimizing-retrieval-for-rag-applications-enhancing-contextual-knowledge-in-llms-79ebcafe5f6e>`_
* [arxiv.org] `Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection <https://arxiv.org/abs/2405.16178>`_
* [stackoverflow.blog] `Practical tips for retrieval-augmented generation (RAG) <https://stackoverflow.blog/2024/08/15/practical-tips-for-retrieval-augmented-generation-rag/>`_

Agents & Tools
-----------------------------------------------------------------------------------------
* [arxiv.org] `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/pdf/2302.04761>`_
* [youtube.com] `TUTORIAL: Large Language Model Powered Agents in the Web <https://www.youtube.com/watch?v=QpXsnd3W7E4>`_

Modeling Choices
-----------------------------------------------------------------------------------------
#. Frozen RAG

	* [arxiv.org][FAIR] `REPLUG: Retrieval-Augmented Black-Box Language Models <https://arxiv.org/pdf/2301.12652>`_
	* [arxiv.org] RALM: `In-Context Retrieval-Augmented Language Models <https://arxiv.org/pdf/2302.00083>`_

#. Trained RAG

	* [arxiv.org][FAIR] RAG: `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks <https://arxiv.org/pdf/2005.11401>`_
	* [arxiv.org][FAIR] FiD: `Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering <https://arxiv.org/pdf/2007.01282>`_
	* [arxiv.org][FAIR] Atlas: `Few-shot Learning with Retrieval Augmented Language Models <https://arxiv.org/pdf/2208.03299>`_	
	* [arxiv.org][FAIR] kNN-LM: `Generalization through Memorization: Nearest Neighbor Language Models <https://arxiv.org/pdf/1911.00172>`_
	* [arxiv.org][Goog] REALM: `Retrieval-Augmented Language Model Pre-Training <https://arxiv.org/pdf/2002.08909>`_
	* [arxiv.org][FAIR] FLARE: `Active Retrieval Augmented Generation <https://arxiv.org/pdf/2305.06983>`_
	* [arxiv.org][FAIR] Toolformer: `Language Models Can Teach Themselves to Use Tools <https://arxiv.org/pdf/2302.04761>`_
	* [arxiv.org] `Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning <https://arxiv.org/abs/2501.15228>`_
	* [arxiv.org] `SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore <https://arxiv.org/pdf/2308.04430>`_
	* [arxiv.org] `Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection <https://arxiv.org/pdf/2310.11511>`_
	* [arxiv.org][FAIR] RA-DIT: `Retrieval-Augmented Dual Instruction Tuning <https://arxiv.org/pdf/2310.01352>`_	
	* Might not work well in practice:

		* [arxiv.org][DeepMind] Retro: `Improving language models by retrieving from trillions of tokens <https://arxiv.org/pdf/2112.04426>`_
		* [arxiv.org][Nvidia] Retro++: `InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining <https://arxiv.org/pdf/2310.07713v2>`_
	* Other stuff:

		* [arxiv.org] Issue with Frozen RAG: `Lost in the Middle: How Language Models Use Long Contexts <https://arxiv.org/pdf/2307.03172>`_
		* [arxiv.org] `Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering <https://arxiv.org/pdf/2210.02627v1>`_
		* [arxiv.org] `FINE-TUNE THE ENTIRE RAG ARCHITECTURE (INCLUDING DPR RETRIEVER) FOR QUESTION-ANSWERING <https://arxiv.org/pdf/2106.11517v1>`_

RAG Pipelines
-----------------------------------------------------------------------------------------
* [llamaindex.ai] `RAG pipeline with Llama3 <https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/#lets-build-rag-pipeline-with-llama3>`_
* [huggingface.co] `Simple RAG for GitHub issues using Hugging Face Zephyr and LangChain <https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain>`_
* [huggingface.co] `Advanced RAG on Hugging Face documentation using LangChain <https://huggingface.co/learn/cookbook/en/advanced_rag>`_
* [huggingface.co] `RAG Evaluation <https://huggingface.co/learn/cookbook/en/rag_evaluation>`_
* [huggingface.co] `Building A RAG Ebook “Librarian” Using LlamaIndex <https://huggingface.co/learn/cookbook/en/rag_llamaindex_librarian>`_

Notes: Modeling
=========================================================================================
.. note::
	* x = query
	* z = doc
	* y = output

Frozen RAG
-----------------------------------------------------------------------------------------
In-context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RALM

		- Retrieve k documents Z_k.
		- Rerank the docs using (1) zero-shot LM or (2) dedicated trained ranker.
		- Select top doc Z_top.
		- Prepend top doc in textual format as-is to the query as a part of the prompt for the LM to generate.
		- What we pass to the decoder: prompt with Z_top in it.
		- Issues: problematic for multiple docs (!)

In-context/Seq2Seq/Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RePLUG

		- Retrieve k documents.
		- Use cosine similarity score to compute p(Z_k | X).
		- What we pass to the decoder: concat{Z_k, X} or prompt with Z_k in it.
		- Make k forward passes in the decoder for each token to compute the likelihood over vocab using softmax p(Y_i | concat{Z_k, X}, Y_1..{i-1}).
		- Rescale the softmax with p(Z_k | X) and marginalize.
		- Pass the marginalized softmax to the decoder.
		- Issues: k forward passes at each token.

Decoder Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	kNN-LN
	
		- For the current token consider X = encode(Y_1...Y_{i-1}).
		- Retrieve k documents Z_k matching X.
		- Make k forward passes in the decoder with the matching doc p_k(Y_i | Z_1..{i-1}).
		- Rescale p_k(Y_i | Z_1..{i-1}) over k and marginalize over the next token Y_i.
		- Do the same in the original sequence p_decode(Y_i | Z_1..{i-1}).
		- Interpolate between these using a hyperparameter.
		- Issues: k forward passes + retrieval at each token.

Retriever trainable RAG
-----------------------------------------------------------------------------------------
Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RePLUG-LSR

		- Uses the parametric LM's output to update the retriever.
		- Loss: KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.

E2E trainable RAG
-----------------------------------------------------------------------------------------
Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	* RAG

		- Per token: same as RePLUG - output probability is marginalised at the time of generation of each token, pass it to beam decoder.
		- Per sequence: output probability is marginalised for the entire sequence.

			- Results in #Y generated sequences.
			- Might require additional passes.

		- Training - NLL loss across predicted tokens.
		- Issues: E2E training makes doc index update problematic, solution: just update the query encoder.
	* Atlas

		- Multiple choice for updating the retriever - simple RePLUG-LSR type formulation based on the KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.
		- Pre-training: same objective as the Seq2Seq (prefixLM or MLM) or decoder-only objective works well.
		- Training:
		- Issues:

Notes: Index Choice
=========================================================================================
Graph RAG
-----------------------------------------------------------------------------------------
.. important::
	- Baseline rag struggles
	
		- answering a question requires traversing disparate pieces of information through their shared attributes
		- holistically understand summarized semantic concepts over large data collections or even singular large documents.
	
	- Graph RAG: https://microsoft.github.io/graphrag/
	
		.. note::
			- Source documents -> Text Chunks: Note: Tradeoff P/R in chunk-size with number of LLM calls vs quality of extraction (due to lost in the middle)
			- Text Chunks -> Element Instances: 
			
				- Multipart LLM prompt for (a) Entity and then (b) Relationship. Extract descriptions as well.
				- Tailor prompt for each domain with FS example. 
				- Additional extraction covariates (e.g. events). 
				- Multiple rounds of gleaning - detect additional entities with high logit bias for yes/no. Prepend "MANY entities were missed".
			- Element Instances -> Element Summaries
			- Element Summaries -> Graph Communities
			- Graph Communities -> Community Summaries
	
				- Leaf level communities
				- Higher level communities
			- Community Summaries -> Community Answers -> Global Answer
	
				- Prepare community summaries: Shuffle and split into chunks to avoid concentration of information and therefore lost in the middle.
				- Map-Reduce community summaries
	
			- Summarisation tasks
	
				- Abstractive vs extractive
				- Generic vs query-focused
				- Single document vs multi-document
	
		- The LLM processes the entire private dataset, creating references to all entities and relationships within the source data, which are then used to create an LLM-generated knowledge graph. 
		- This graph is then used to create a bottom-up clustering that organizes the data hierarchically into semantic clusters This partitioning allows for pre-summarization of semantic concepts and themes, which aids in holistic understanding of the dataset. 
		- At query time, both of these structures are used to provide materials for the LLM context window when answering a question.	
		- Eval:
	
			- Comprehensiveness (completeness within the framing of the implied context of the question)
			- Human enfranchisement (provision of supporting source material or other contextual information)
			- Diversity (provision of differing viewpoints or angles on the question posed)
			- Selfcheckgpt

Notes: RAG vs Long Context
=========================================================================================
- RAG FTW: Xu et al (NVDA): RETRIEVAL MEETS LONG CONTEXT LARGE LANGUAGE MODELS (Jan 2024)

	- Compares between 4k+RAG and 16k/32k LC finetuned with rope trick with 40B+ models
	- Scroll and long bench
- LC FTW: Li et al (DM): Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (Jul 2024)

	- Systematized the eval framework using infty-bench EN.QA (~150k) and EN.MC (~142k) and 7 datasets from long-bench (<20k)
	- 60% of the cases RAG and LC agrees (even makes the same mistakes)
	- Cases where RAG fails 

		(a) multi-hop retrieval 
		(b) general query where semantic similarity doesn't make sense 
		(c) long and complex query 
		(d) implicit query requiring a holistic view of the context
	- Key contribution: Proposes self-reflectory approach with RAG first with an option to respond "unanswerable", then LC
- RAG FTW: Wu et al (NVDA): In Defense of RAG in the Era of Long-Context Language Models (Sep 2024)

	- Same eval method as the above
	- Key contribution: keep the chunks in the same order as they appear in the original text instead of ordering them based on sim measure

Notes: LLM and KG
=========================================================================================
.. seealso::
	* Unifying Large Language Models and Knowledge Graphs: A Roadmap
	* QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering
	* SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

KG-enhanced LLMs
-----------------------------------------------------------------------------------------
- pre-training:

	- ERNIE: Enhanced language representation with informative entities
	- Knowledge-aware language model pretraining
- inference time:

	- Retrieval-augmented generation for knowledge intensive nlp tasks
- KG for facts LLM for reasoning:

	- Language models as knowledge bases?
	- KagNet: Knowledgeaware graph networks for commonsense reasoning

LLM enhanced KGs: KG completion and KG reasoning
-----------------------------------------------------------------------------------------
- LLMs for Knowledge Graph Construction and Reasoning
- Pretrain-KGE: Learning Knowledge Representation from Pretrained Language Models
- From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer

Synergized KG LLM
-----------------------------------------------------------------------------------------
- KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation
- Search: LaMDA: Language Models for Dialog Applications
- RecSys: Is chatgpt a good recommender? a preliminary study
- AI Assistant: ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation

*****************************************************************************************
Known Issues
*****************************************************************************************
Hallucination 
=========================================================================================
Detection & Mitigation
-----------------------------------------------------------------------------------------
Supervised
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Applicable: translation, summarization, image captioning

	- n-gram (bleu/rouge, meteor)

		- reference dependent, usually only one reference
		- often coarse or granular
		- unable to capture semantics: fail to adapt to stylistic changes in the reference
	- ask gpt (selfcheckgpt, g-eval)

		- evaluate on (a) adherence (b) correctness
		- blackbox, unexplainable
		- expensive
Unsupervised
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- perplexity-based (gpt-score, entropy, token confidence) - good second order metric to check
- too granular, represents confusion - not hallucination in particular, often red herring
- not always available

Sycophany
=========================================================================================
Monosemanticity
=========================================================================================
- many neurons are polysemantic: they respond to mixtures of seemingly unrelated inputs.
- neural network represents more independent "features" of the data than it has neurons by assigning each feature its own linear combination of neurons. If we view each feature as a vector over the neurons, then the set of features form an overcomplete linear basis for the activations of the network neurons.
- towards monosemanticity:

	(1) creating models without superposition, perhaps by encouraging activation sparsity; 
	(2) using dictionary learning to find an overcomplete feature basis in a model exhibiting superposition; and 
	(3) hybrid approaches relying on a combination of the two.
- developed counterexamples which persuaded us that the 

	- sparse architectural approach (approach 1) was insufficient to prevent polysemanticity, and that 
	- standard dictionary learning methods (approach 2) had significant issues with overfitting.
- use a weak dictionary learning algorithm called a sparse autoencoder to generate learned features from a trained model that offer a more monosemantic unit of analysis than the model's neurons themselves.
