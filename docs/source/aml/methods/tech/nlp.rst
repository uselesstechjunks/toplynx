*****************************************************************************************
Natural Language Processing
*****************************************************************************************
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Notes from Common NLP Tasks
=========================================================================================
Using In-Context Capability
-----------------------------------------------------------------------------------------
- Language Models as Knowledge Bases
- Language Models are Open Knowledge Graphs

NER
-----------------------------------------------------------------------------------------
- Fixed NER: 

	- classification + chunking - encoder based (NER/POS)

		- token classification:

			- attributing a label to each token by having one class per entity and one class for “no entity.”
			- ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
			- AutoModelForTokenClassification 
		- chunking: 

			- attributing one label (usually B-) to any tokens that are at the beginning of a chunk, 
			- another label (usually I-) to tokens that are inside a chunk, and 
			- a third label (usually O) to tokens that don’t belong to any chunk.
	- The traditional framework used to evaluate token classification prediction is seqeval - classwise p/r/f1/accuracy, overall p/r/f1/accuracy
- Free NER:

	- extract - s2s, decoder
	- Autoregressive Entity Retrieval
	- GPT-NER
	- Universal NER
	
Disambiguation 
-----------------------------------------------------------------------------------------
- Clustering based approach - 
- End to end neural coreference method - all O(n^2) pairs

	- https://huggingface.co/models?other=coreference-resolution
	- https://explosion.ai/blog/coref

Entity Linking
-----------------------------------------------------------------------------------------
- Text-based approaches - tfidf, statistical	
- Graph-based approaches to existing knowledge-base - https://huggingface.co/models?other=named-entity-linking

	- Autoregressive Entity Retrieval - was trained on the full training set of BLINK (i.e., 9M datapoints for entity-disambiguation grounded on Wikipedia).	
	- Blink - Scalable Zero-shot Entity Linking with Dense Entity Retrieval
	- Refined - https://github.com/alexa/ReFinED	

Relation Extraction
-----------------------------------------------------------------------------------------
- SentenceRE
- DocRE

Link Prediction
-----------------------------------------------------------------------------------------
Knowledge Graph Large Language Model (KG-LLM) for Link Prediction

Graph Completion
-----------------------------------------------------------------------------------------
SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

Question Answering
-----------------------------------------------------------------------------------------
QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering

GraphRAG
-----------------------------------------------------------------------------------------
MultiModal
-----------------------------------------------------------------------------------------
Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey

Other NLP Tasks
-----------------------------------------------------------------------------------------
Large Language Models Meet NLP: A Survey

Information Retrieval (IR)
=========================================================================================
0. Language Models in IR
-----------------------------------------------------------------------------------------
- MLM based: BERT, T5
- RTD based: Electra
- Contrastive Learning based:
	- image: OG image and distorted image form pos-pairs
	- text: contriever
		- contrastive learning based embeddings
		- infonce loss: softmax over 1 positive and K negative
		- getting positive: 
			(a) Inverse Cloze Task (contiguous segment as query, rest as doc) - relates with closure of a query
			(b) Independent cropping - sample two independent contiguous pieces of text
		- getting negatives:
			(a) in-batch negatives
			(b) negs from previous batch docs - called keys. either not updated or updated slowly with different parameterization including momentum (moco)
	- text: e5
- Long Context
	- "lost in the middle" using longer context (primacy bias, recency bias) - U-shaped curve
		-> if using only a decoder model, due to masked attention, put the question at the end 
		-> instruction tuned is much better
		-> relevance order of the retriever matters a lot
	
	- extending context length
		- needle in a haystack
		- l-eval, novelqa, infty-bench
		- nocha (fictional, unseen books with true/false q/a pairs 
			- performs better when fact is present in the book at sentence level
			- performs worse if requires global reasoning or if contains extensive world building
		- position embeddings 
			- change the angle hyperparameter in RoPE to deal with longer sequences
		- efficient attention 
			- full attention with hardware-aware algorithm design - flash attention
			- sparse attention techniques: sliding window attention, block attention
		- data engineering - replicate larger model perf using 7b/13b llama
			- continuous pretraining
				- 1-5B new tokens for 
				- upsampling longer sequences
				- same #tokens per batch (adjusted as per sequence length and batch size)
				- 2e-5 lr cosine schedule
				- 2x8 a100 gpu, 7 day training, flashattention (3x time for 80k vs 4k, majority time goes in cpu<->gpu, gpu<->gpu, and hbm<->sm)
			- instruction tuning: rlhf data + self instruct
				- (a) chunk long doc (b) from long doc formulate q/a (c) use OG doc and q/a pair as training
				- 1e-5 lr constant
				- lora/qlora
		- incorporating some form of recurrance relation - transformer-xl, longformer, rmt

- chain-of-agents

1. Document Retrieval
-----------------------------------------------------------------------------------------
Description:
Document retrieval involves finding and ranking relevant documents from a large corpus in response to a user's query.

Example:

- Input: Query: "What are the symptoms of COVID-19?"
- Output: [List of relevant documents about COVID-19 symptoms]

Evaluation Metrics:

- Precision at k (P@k)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

Benchmark Datasets:

- TREC (Text REtrieval Conference)
- CLEF (Conference and Labs of the Evaluation Forum)
- MSMARCO

Example Prompt:
"Retrieve the top 5 documents related to the query: 'What are the symptoms of COVID-19?'"

2. Passage Retrieval
-----------------------------------------------------------------------------------------
Description:
Passage retrieval involves finding and ranking relevant passages or sections within documents in response to a user's query.

Example:

- Input: Query: "What is the capital of France?"
- Output: [List of passages containing information about the capital of France]

Evaluation Metrics:

- Precision at k (P@k)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

Benchmark Datasets:

- MSMARCO Passage Ranking
- TREC Deep Learning

Example Prompt:
"Retrieve the top 5 passages related to the query: 'What is the capital of France?'"

3. Query Expansion
-----------------------------------------------------------------------------------------
Description:
Query expansion involves modifying a user's query by adding additional terms to improve retrieval performance.

Example:

- Input: Query: "COVID-19"
- Output: Expanded Query: "COVID-19 coronavirus symptoms pandemic"

Evaluation Metrics:

- Precision
- Recall
- Mean Average Precision (MAP)

Benchmark Datasets:

- TREC
- CLEF

Example Prompt:
"Expand the following query to improve search results: 'COVID-19'"

4. Question Answering (QA)
-----------------------------------------------------------------------------------------
Description:
QA involves retrieving answers to questions posed in natural language, often using information from a large corpus.

Example:

- Input: Question: "What is the tallest mountain in the world?"
- Output: "Mount Everest"

Evaluation Metrics:

- Exact Match (EM)
- F1 Score

Benchmark Datasets:

- SQuAD (Stanford Question Answering Dataset)
- Natural Questions
- TriviaQA

Example Prompt:
"Answer the following question: 'What is the tallest mountain in the world?'"

Information Extraction (IE) Tasks
=========================================================================================
0. Language Models in IE
-----------------------------------------------------------------------------------------
- NER: named entity recognition, entity-linking
	- predefined entity-classes: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC). 
		- https://huggingface.co/dslim/bert-base-NER
		- https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english			
	- open entity-classes: 
		- UniversalNER: https://universal-ner.github.io/, https://huggingface.co/Universal-NER
		- GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer https://huggingface.co/urchade/gliner_large-v2
		- GLiNER - Multitask: https://www.knowledgator.com/ -> https://huggingface.co/knowledgator/gliner-multitask-large-v0.5
	- Open IE eval: Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction (https://github.com/qijimrc/ROBUST/tree/master)		
	- LLMaAA: Making Large Language Models as Active Annotators https://github.com/ridiculouz/LLMaAA/tree/main
	- A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Graph Construction (https://github.com/zjunlp/DeepKE)
- RE: relationship extraction
	- QA4RE: Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors (ZS Pr) https://github.com/OSU-NLP-Group/QA4RE
	- DocGNRE: Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models (https://github.com/bigai-nlco/DocGNRE)
- EE: event extraction
- Papers to read: UniversalNER, GLiNER

1. Named Entity Recognition (NER)
-----------------------------------------------------------------------------------------
Description:
NER involves identifying and classifying entities in text into predefined categories such as names of people, organizations, locations, dates, etc.

Example:

- Input: "Barack Obama was born in Hawaii."
- Output: [("Barack Obama", "PERSON"), ("Hawaii", "LOCATION")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- CoNLL-2003
- OntoNotes
- WNUT 2017

Example Prompt:
"Identify and classify named entities in the following sentence: 'Barack Obama was born in Hawaii.'"

2. Relation Extraction
-----------------------------------------------------------------------------------------
Description:
Relation extraction involves identifying and classifying the relationships between entities in text.

Example:

- Input: "Barack Obama was born in Hawaii."
- Output: ("Barack Obama", "born in", "Hawaii")

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- TACRED
- SemEval
- ACE 2005

Example Prompt:
"Identify the relationship between entities in the following sentence: 'Barack Obama was born in Hawaii.'"

3. Event Extraction
-----------------------------------------------------------------------------------------
Description:
Event extraction involves identifying events in text and their participants, attributes, and the context in which they occur.

Example:

- Input: "An earthquake of magnitude 6.5 struck California yesterday."
- Output: [("earthquake", "magnitude 6.5", "California", "yesterday")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- ACE 2005
- MUC-4
- TAC KBP

Example Prompt:
"Extract events and their details from the following text: 'An earthquake of magnitude 6.5 struck California yesterday.'"

4. Coreference Resolution
-----------------------------------------------------------------------------------------
Description:
Coreference resolution involves identifying when different expressions in a text refer to the same entity.

Example:

- Input: "Jane went to the market. She bought apples."
- Output: [("Jane", "She")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- CoNLL-2012 Shared Task
- OntoNotes

Example Prompt:
"Identify coreferences in the following text: 'Jane went to the market. She bought apples.'"

Classification Tasks
=========================================================================================
1. Sentiment Analysis
-----------------------------------------------------------------------------------------
Description:
Sentiment analysis involves determining the sentiment or emotional tone behind a piece of text, typically classified as positive, negative, or neutral.

Example:

- Input: "I love this product!"
- Output: "Positive"

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Benchmark Datasets:

- IMDb Movie Reviews
- Sentiment140
- SST (Stanford Sentiment Treebank)

Example Prompt:
"Determine the sentiment of the following text: 'I love this product!'"

Sequence to Sequence Tasks
=========================================================================================
1. Machine Translation
-----------------------------------------------------------------------------------------
Description:
Machine translation involves translating text from one language to another.

Example:

- Input: "Hello, how are you?" (English)
- Output: "Hola, ¿cómo estás?" (Spanish)

Evaluation Metrics:

- BLEU Score
- METEOR
- TER

Benchmark Datasets:

- WMT (Workshop on Machine Translation)
- IWSLT (International Workshop on Spoken Language Translation)

Example Prompt:
"Translate the following text from English to Spanish: 'Hello, how are you?'"

2. Text Summarization
-----------------------------------------------------------------------------------------
Description:
Text summarization involves generating a concise summary of a longer document while preserving key information.

Example:

- Input: "Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry."
- Output: "AI is a branch of computer science aiming to create intelligent machines, essential in technology."

Evaluation Metrics:

- ROUGE Score
- BLEU Score

Benchmark Datasets:

- CNN/Daily Mail
- XSum
- Gigaword

Example Prompt:
"Summarize the following text: 'Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry.'"

3. Text Generation
-----------------------------------------------------------------------------------------
Description:
Text generation involves creating new text that is coherent and contextually relevant based on a given input prompt.

Example:

- Input: "Once upon a time"
- Output: "Once upon a time, in a small village, there lived a brave young girl named Ella."

Evaluation Metrics:

- Perplexity
- BLEU Score
- Human Evaluation

Benchmark Datasets:

- OpenAI GPT-3 Playground
- EleutherAI's Pile
- WikiText

Example Prompt:
"Generate a continuation for the following text: 'Once upon a time, in a small village, there lived a brave young girl named Ella.'"

Multimodal Tasks
=========================================================================================
1. Text-to-Speech (TTS)
-----------------------------------------------------------------------------------------
Description:
TTS involves converting written text into spoken words.

Example:

- Input: "Good morning, everyone."
- Output: [Audio clip saying "Good morning, everyone."]

Evaluation Metrics:

- Mean Opinion Score (MOS)
- Word Error Rate (WER)
- Naturalness

Benchmark Datasets:

- LJSpeech
- LibriSpeech
- VCTK

Example Prompt:
"Convert the following text to speech: 'Good morning, everyone.'"

2. Speech Recognition
-----------------------------------------------------------------------------------------
Description:
Speech recognition involves converting spoken language into written text.

Example:

- Input: [Audio clip saying "Hello, world!"]
- Output: "Hello, world!"

Evaluation Metrics:

- Word Error Rate (WER)
- Sentence Error Rate (SER)

Benchmark Datasets:

- LibriSpeech
- TED-LIUM
- Common Voice

Example Prompt:
"Transcribe the following audio clip: [Audio clip saying 'Hello, world!']"

Extending Vocab for Domain-Adaptation or Fine-Tuning
=========================================================================================
1. Extend the Tokenizer Vocabulary
-----------------------------------------------------------------------------------------
.. code-block:: python

	from transformers import GPT2Tokenizer, GPT2Model
	
	# Load the pre-trained tokenizer and model
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	
	# Example of extending vocabulary with domain-specific terms
	domain_specific_terms = ["term1", "term2", "term3"]
	tokenizer.add_tokens(domain_specific_terms)
	
	# If you are also fine-tuning the model, adjust the model to handle new tokens
	model = GPT2Model.from_pretrained('gpt2')
	model.resize_token_embeddings(len(tokenizer))

.. note::
	* tokenizer.add_tokens(domain_specific_terms): This adds your domain-specific terms to the tokenizer vocabulary.
	* model.resize_token_embeddings(len(tokenizer)): This adjusts the model's embedding layer to accommodate the new tokens. This step is crucial if you plan to fine-tune the model with these new tokens.

2. Tinkering with the Embedding Matrix
-----------------------------------------------------------------------------------------
.. code-block:: python

	import torch
	
	# Load the original model again for clarity
	model = GPT2Model.from_pretrained('gpt2')
	
	# Assuming you have already added new tokens to the tokenizer
	new_token_ids = tokenizer.encode(domain_specific_terms, add_special_tokens=False)
	
	# Initialize the new token embeddings randomly
	new_token_embeddings = torch.randn(len(new_token_ids), model.config.hidden_size)
	
	# Concatenate original embeddings with new token embeddings
	original_embeddings = model.transformer.wte.weight[:tokenizer.vocab_size]
	combined_embeddings = torch.cat([original_embeddings, new_token_embeddings], dim=0)
	
	# Overwrite the original embedding matrix in the model
	model.transformer.wte.weight.data = combined_embeddings

.. note::
	* tokenizer.encode(domain_specific_terms, add_special_tokens=False): This encodes the domain-specific terms to get their token IDs in the tokenizer's vocabulary.
	* torch.randn(len(new_token_ids), model.config.hidden_size): This initializes random embeddings for new tokens. Alternatively, you can initialize them differently based on your specific needs.
	* model.transformer.wte.weight[:tokenizer.vocab_size]: Extracts the original embeddings up to the size of the original vocabulary.
	* torch.cat([original_embeddings, new_token_embeddings], dim=0): Concatenates the original embeddings with the new token embeddings.

Notes
-----------------------------------------------------------------------------------------
* Tokenizer Vocabulary: Ensure that after extending the tokenizer vocabulary, you save it or use it consistently across your tasks.
* Embedding Adjustment: The approach here adds new tokens and initializes their embeddings separately from the pre-trained embeddings. This keeps the original embeddings intact while allowing new tokens to have their embeddings learned during fine-tuning.
* Fine-Tuning: If you plan to fine-tune the model on your specific tasks, you would then proceed with training using your domain-specific data, where the model will adapt not only to the new tokens but also to the specific patterns in your data.
