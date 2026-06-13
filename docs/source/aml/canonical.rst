######################################################
Canonical Micro-problems
######################################################
***************************************************************
1. Natural Language Processing
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

***************************************************************
2. Information Retrieval
***************************************************************
Sparse retrieval 
===============================================================
- Build a search bar for an internal HR portal. Employees type queries like 'parental leave policy UK' and expect the most relevant policy document to surface at the top.
- BM25 - TF-IDF statistics, no training - vocabulary mismatch, acronyms, synonyms

Dense retrieval 
===============================================================
- Our e-commerce site has 50M products. A user types 'something cozy to wear on a winter hike' — match them to relevant products even when none of those words appear in the product title.
- Bi-encoder (DPR, sentence-BERT) - in-batch negatives on MS-MARCO / NQ - rare term recall; hard negatives needed for quality

Learned sparse retrieval 
===============================================================
- We need retrieval that handles both exact keyword matching (SKU numbers, product codes) and semantic matching (synonyms, paraphrases) in the same system.
- SPLADE - distillation from cross-encoder - slower than BM25, complex training

Hybrid retrieval 
===============================================================
- Build a search system for a legal research platform where users sometimes search by citation ('Section 12(b) of the Securities Exchange Act') and sometimes by concept ('insider trading safe harbor').
- BM25 + dense, score fusion (RRF or learned) - combined supervision - interpolation weight is domain-sensitive

Cross-encoder reranking 
===============================================================
- Our first-stage retrieval returns 100 candidates. Build a second-stage model that reorders them to maximize the chance the top result is the one the user clicks.
- BERT cross-encoder, MonoT5 - pointwise or pairwise MS-MARCO labels - latency; can only run on top-k from first stage

Query understanding / expansion 
===============================================================
- Users on our medical platform often use lay terms like 'heart attack' instead of clinical terms like 'myocardial infarction'. Improve retrieval by reformulating queries before they hit the index.
- T5 for query rewriting, docT5query for document expansion - weak supervision from click logs - expansion can introduce noise

Learned index / ANN 
===============================================================
- We have 1B product embeddings that need to be searched in under 20ms at query time. Design the indexing and retrieval infrastructure.
- FAISS (IVF, HNSW), ScaNN - embedding quality upstream - recall-latency tradeoff; HNSW strong on recall, IVF better on memory

***************************************************************
3. Computer Vision
***************************************************************
Image classification 
===============================================================
- We run an online marketplace for secondhand goods. When a seller uploads a photo, automatically categorize the item into our product taxonomy so it appears in the right browse category.
- ResNet / ViT + softmax - ImageNet labels - long tail, fine-grained distinctions

Object detection 
===============================================================
- We're building a retail shelf monitoring system. Given a photo of a store shelf, identify which products are present, their positions, and whether any slots are empty.
- DETR, Faster R-CNN - COCO bounding box labels - small object detection, dense scenes

Semantic segmentation 
===============================================================
- We're building an autonomous vehicle perception system. Given a dashcam frame, label every pixel as road, pedestrian, vehicle, or obstacle.
- SegFormer, Mask2Former - pixel-level labels (Cityscapes, ADE20K) - label cost is extreme; boundary ambiguity

Instance segmentation 
===============================================================
- For a fashion try-on app, given a photo of a person, precisely cut out individual clothing items so they can be replaced with items from our catalog.
- Mask R-CNN, SAM - COCO instance masks - occlusion, overlapping instances

Optical character recognition (OCR) 
===============================================================
- We process millions of scanned invoices from vendors. Extract the invoice number, line items, and total amount so they can be ingested into our accounting system
- CNN encoder + CTC / attention decoder - synthetic + real text pairs - arbitrary fonts, low resolution, curved text

Document layout analysis 
===============================================================
- We're digitizing a large archive of academic papers. Given a scanned PDF, identify which regions are body text, which are figures, which are tables, and which are captions.
- LayoutLM, DiT - PubLayNet, FUNSD - mixed layout (tables inside prose), multi-column

Image captioning 
===============================================================
- We're building an accessibility feature for a social platform. Automatically generate alt text for user-uploaded images so screen readers can describe them.
- CNN encoder + transformer decoder (BLIP) - COCO captions - hallucination of objects not in image

Visual QA 
===============================================================
- Build a feature for a home improvement app where a user can photograph a room and ask 'what style is this?' or 'does this furniture match?'
- ViLBERT, BLIP-2, LLaVA - VQA v2, GQA - language bias (models answer from prior without looking)

Image-text retrieval / grounding 
===============================================================
- A user types 'minimalist kitchen with marble countertops' into our interior design platform — return the most visually matching photos from our catalog.
- CLIP contrastive - (image, alt-text) web pairs - loose pairing; spatial/compositional reasoning fails

Phrase grounding / referring expression 
===============================================================
- In a warehouse robotics system, a human operator says 'pick up the small red box on the left side of the shelf' — the robot must identify the exact object being referred to.
- MDETR, Grounding DINO - RefCOCO - complex relational expressions ("the leftmost red cup")

Face verification 
===============================================================
- Build a re-authentication system for a banking app: when a user attempts a high-value transaction, verify that the live selfie matches the ID photo on file.
- ArcFace / CosFace (margin-based softmax) - labeled identity pairs - demographic bias, pose/lighting variation

Anomaly detection (visual) 
===============================================================
- We run a manufacturing line producing circuit boards. Flag any board coming off the line that has a visual defect, without having labeled examples of every defect type.
- PatchCore, normalizing flows - only normal images (no anomaly labels needed) - defining "normal" distribution; novel anomaly types
