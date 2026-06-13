######################################################
Canonical Micro-problems
######################################################
***************************************************************
Natural Language Processing
***************************************************************
Extractive QA
===============================================================
- We want to build a customer support tool where agents can paste in a long policy document and ask a question — the system should highlight the exact sentence or phrase in the document that answers it
- BERT + span head (start/end token classification) - SQuAD-style span labels - answer not present in context, requires "no answer" head

Open-domain QA 
===============================================================
- Build an internal knowledge base assistant for a company with thousands of wiki pages. An employee asks a natural language question and expects a direct answer, not a list of links.
- DPR retriever + BERT reader (RAG) - Natural Questions, TriviaQA - retrieval recall is the bottleneck; reader is useless if retriever misses

Abstractive QA / Summarization 
===============================================================
- Our legal team receives hundreds of contracts per day. Build a system that, given a contract and a question like 'what are the termination conditions?', returns a concise natural language answer even when the answer is spread across multiple clauses.
- BART, T5 seq2seq - abstractive QA pairs, CNN/DM - hallucination, faithfulness to source

Semantic textual similarity 
===============================================================
- We have a support ticket system with 10M historical tickets. When a new ticket comes in, surface the 5 most similar resolved tickets so the agent can see how they were handled.
- Siamese BERT (sentence-transformers), contrastive loss - STS-B, NLI pairs as soft supervision - domain shift; cosine similarity doesn't respect asymmetry (entailment is directional)

Natural language inference 
===============================================================
- Given a product claim made in an ad (e.g. 'clinically proven to reduce wrinkles in 7 days') and a scientific paper, determine whether the paper supports, contradicts, or is neutral about the claim.
- BERT + 3-way classification head - SNLI, MultiNLI - annotation artifacts, hypothesis-only shortcuts

Named entity recognition 
===============================================================
- We're ingesting millions of news articles per day. Extract all mentioned companies, people, and locations so they can be linked to structured entities in our knowledge graph
- BERT + token classification + CRF - CoNLL-2003 BIO tags - nested entities, boundary ambiguity, cross-sentence coreference

Relation extraction 
===============================================================
- Given a news article, automatically populate a structured event database: who acquired whom, at what price, on what date
- BERT with entity markers + classification head - TACRED, DocRED - implicit relations not surfaced in a single sentence

Coreference resolution 
===============================================================
- In a long earnings call transcript, every time 'they', 'the company', 'its CEO' is mentioned, resolve it to the correct named entity so downstream extraction is accurate.
- SpanBERT + span scoring - OntoNotes - long-document coreference, pronoun ambiguity

Intent classification 
===============================================================
- We're building a voice assistant for a food delivery app. Given a user utterance like 'I want to change my order', route it to the correct backend handler.
- fine-tuned encoder + softmax head - Dialog datasets (ATIS, SNIPS) - out-of-scope intents, intent overlap

Slot filling 
===============================================================
- In the same voice assistant, after classifying the intent as 'book a restaurant', extract the structured fields: restaurant name, date, time, party size.
- BERT + BIO token classifier (joint with intent) - ATIS - unseen slot values, multi-value slots

Machine translation 
===============================================================
- Our marketplace operates in 30 countries. Automatically translate seller-generated product listings into the buyer's language with minimal human review.
- Transformer seq2seq - parallel corpora (WMT) - low-resource languages, rare token translation

Text generation / controlled generation 
===============================================================
- Given a product's structured spec sheet (dimensions, material, price), generate a compelling marketing description in the brand's tone of voice
- GPT-style decoder, CTRL - LM pretraining + RLHF - repetition, degeneration, constraint satisfaction
