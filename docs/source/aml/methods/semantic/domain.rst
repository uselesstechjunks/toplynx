##########################################################################
Domain Understanding
##########################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

**************************************************************************
Topics
**************************************************************************
Key Concepts
==========================================================================
- Supervised Learning Paradigms  
	- Multi-class classification  
	- Multi-label classification  
	- Hierarchical classification  
	- Multi-task learning  
- Representation Learning & Retrieval  
	- Metric learning / Contrastive learning (e.g. InfoNCE, SimCLR)  
	- Embedding learning + Nearest Neighbor Search  
	- Vector quantization / Product quantization  
	- Dense vs Sparse retrieval models  
- Labeling & Supervision Strategies  
	- Weak supervision  
	- Self-supervised learning  
	- Positive-Unlabeled (PU) learning  
	- Active learning  
	- Hard negative mining / Semi-hard negative mining  
- Model Architectures & Adaptation  
	- Fine-tuning pre-trained encoders (CNNs, ViT, BERT, CLIP)  
	- Freezing/unfreezing strategies  
	- Lightweight fine-tuning (LoRA, adapters, prompt tuning)  
	- Multimodal fusion (early vs late fusion)  
- Search & Ranking Infrastructure  
	- Similarity-based search (e.g., FAISS, ANN)  
	- Hybrid retrieval (text + image)  
	- Cross-encoder vs dual-encoder ranking models  
- Data Handling & Preprocessing  
	- Data cleaning and normalization (e.g., noisy title correction)  
	- Image augmentations (crop, flip, blur, resize)  
	- Text normalization and deduplication  
	- Taxonomy mapping and vocabulary standardization  

Models
==========================================================================
#. ResNet
#. ViT/DeiT
#. SimCLR
#. Faster RCNN
#. YOLO
#. DETR
#. CLIP/BLIP
#. BERT/XLM-R, DistilBERT
#. T5 / BART
#. LLAMA
#. ViViT
#. TimeSformer

**************************************************************************
Application
**************************************************************************
Priority 1 – Very High Value (Common + Foundational)
==========================================================================
#. Product Categorization (Classification)  
	- Fixed taxonomy classification from image or metadata  
	- Covers: image encoders (ResNet/ViT), fine-tuning, label scarcity, domain shift  
	- Must know: supervised classification, hierarchical taxonomies, class imbalance, data augmentation, few-shot strategies
#. Dynamic Tag Suggestion (Multi-label Prediction)  
	- Open-ended multi-label prediction using metadata/image  
	- Must know: BCEWithLogitsLoss, multi-label thresholds, label imbalance, tag vocab creation, weak supervision
#. Product Taxonomy Mapping  
	- Mapping noisy/seller-provided categories to structured taxonomy  
	- Must know: text classification, category disambiguation, noisy inputs, hierarchical mappings
Priority 2 – High Value (Used Across Systems)
==========================================================================
#. Attribute Extraction (NER or Slot-filling)  
	- Extract structured attributes like brand, color, size from title/description  
	- Must know: sequence labeling (BIO format), spaCy or BERT-based token classifiers, weak labeling, schema constraints
#. Duplicate Listing Detection  
	- Detect duplicate or near-duplicate listings posted by users  
	- Must know: pairwise embedding similarity, clustering, contrastive learning, efficient retrieval, deduplication heuristics
#. Image-Based Visual Search  
	- Match query images to catalog using visual similarity  
	- Must know: contrastive loss (InfoNCE), SimCLR, in-domain pretraining, feature indexing (FAISS), query augmentation
#. Text-Based Search (Query → Product Metadata)  
	- Users search with queries matched to product text fields  
	- Must know: BM25, dense retrieval (dual encoder), cross-encoder reranking, FAISS, negative sampling

Priority 3 – Medium Value (Niche but insightful)
==========================================================================
#. Multimodal Entity Matching / Linking  
	- Link a product to a known item in a catalog (e.g., brand DB) using both image and text  
	- Must know: multimodal encoders (e.g., CLIP), late fusion vs early fusion, product resolution, text normalization
#. Item Quality / Integrity Detection  
	- Detect suspicious, poor quality, or policy-violating listings  
	- Must know: content moderation, adversarial examples, cross-modal rules, abuse signals, self-supervised pretraining

Priority 4 – Lower Priority but Great for Bonus Points
==========================================================================
#. Product Title Generation  
	- Rewrite or generate SEO-friendly titles from user-written titles/descriptions  
	- Must know: text generation (seq2seq), BART/T5 models, summarization, input pre-processing
#. Title/Description Normalization  
	- Normalize noisy seller-written text for search/ads relevance  
	- Must know: grammar correction, paraphrasing, rule-based + neural hybrid methods
#. Visual Grounding / Region Tagging  
	- Localize object regions corresponding to attributes or tags  
	- Must know: object detection + vision-language grounding, attention maps, weak supervision

**************************************************************************
Problems
**************************************************************************
Text-Based Product Search (Metadata Only)
==========================================================================
- Problem  
	- Allow users to search for products using a free-form text query. The system retrieves and ranks relevant products based on matching against product metadata (title, description).
-  Use Cases  
	- Search bar experience in marketplace  
	- Assistive auto-complete or suggestions  
	- Indexing new products with better retrieval capabilities
-  Input / Output  
	- Input: User text query (e.g., "red running shoes")  
	- Output: Ranked list of product IDs with titles and images
-  Problem Type  
	- Semantic text-to-text retrieval (information retrieval / ranking)
-  Model Choices  
	- Sparse retrieval (baseline):  
		- BM25 over title and description fields  
	- Dense retrieval (modern):  
		- Dual-encoder architecture:  
			- Query encoder (e.g., BERT, DistilBERT)  
			- Product encoder (e.g., same as query encoder)  
		- Similarity via dot product or cosine similarity  
	- Optional: Cross-encoder reranker (e.g., BERT) for top-k reranking
- Labeling Scenarios  
	- Supervised: Click logs or labeller-curated query-product matches  
	- Weak supervision: Synthetic query generation from product text  
	- Noisy signals: Search sessions or co-view logs
- Training Setup  
	- Contrastive learning using positive query-product pairs and in-batch negatives  
	- Loss: InfoNCE or triplet loss  
	- Optional hard negative mining using BM25  
	- Pretraining on large query-product corpora or Wikipedia Q-A pairs
- Evaluation Metrics  
	- Recall@k, NDCG@k, Mean Reciprocal Rank (MRR)  
	- Offline: manual relevance judgments or simulated clicks  
	- Online: click-through rate (CTR), dwell time
- Scaling Considerations  
	- Precompute and index product embeddings using vector database (e.g., FAISS, ScaNN)  
	- Real-time encoding of user query at search time  
	- Efficient reranking within top-N retrieved candidates
- Alternative Methods  
	- Hybrid retrieval: combine BM25 and dense scores  
	- Use knowledge distillation to compress dual encoder  
	- Use entity linking to match structured taxonomy (optional)

Product Taxonomy Mapping
==========================================================================
Task: Design a product categorisation tool for facebook marketplace

Problem
--------------------------------------------------------------------------
#. Use-case
	#. System - multiple possible use-cases
		#. >> Real time assist to the sellers during listing creation time
		#. Post upload clean-up/taxonomy mapping (invisible to the seller)
		#. Creation of category keyword index (invisible to the seller)
		#. Reroute to the quality/compliance/integrity team
	#. Actors - sellers, buyers, platform
	#. Entities - listings, user profiles, history
	#. Interests -
		#. Seller - reduce manual work (selecting from suggested category list)
		#. Buyers - find more relevant listings (search/recommendation)
		#. Platform - increase transactions made on the platform, increase quality/compliance/integrity
#. Scale
	#. 1M sellers, 50M listings live, 1M/day new listings, listings lifespan - days-months
	#. Listings are diverse, sellers are global - needs to generalise well on unseen data
#. Signals
	#. Product database 
		#. Majority of the listings don't have taxonomy - 40M
		#. 10M listings have noisy taxonomy assigned by users (may/may not be correct)
		#. 20k listings with correct taxonomy assigned by human experts
	#. Seller profile, reputation
#. Business kpis
	#. Successful session rate (#success sessions/#sessions)
	#. MRR
	#. CTR on search/recommendation
	#. Taxonomy coverage
#. Misc
	#. Fixed set of categories - flat, 5k categories
	#. Each listing belongs to 1 single leaf category

Solution
--------------------------------------------------------------------------
#. Problem type
	#. Learning to rank - listing as the query, category lists is the doc, pointwise learning to rank
	#. Multi-class classification with fixed leaf labels from a predefined taxonomy list as target categories
	#. Learning to rank is better for 
#. Data
	#. Listings
		#. Content - title, description, images (multiple), metadata (product age, dimensions, colour)
		#. Context - upload time, upload location
	#. Seller 
		#. User profile - demographics - agegroup, gender, geolocation, account age
		#. Activity in communities/groups
		#. Stats - past listings, current listings, reputation (might be useful to determine if user-assigned label is noisy)
#. Feature
#. Learning strategy
#. Dataset curation
#. Model
#. Training
#. Eval
#. Deployment
#. Monitoring
#. Improvements

Dynamic Tag Suggestion System (Image-Only)
==========================================================================
- Problem
	- Suggest relevant tags (attributes, descriptors) for product listings to improve discovery, search, and categorization.
- Use Cases
	- Improves product discoverability.
	- Drives tag-based browsing and filtering.
	- Feeds into downstream categorization or moderation systems.
- Input:
	- One or more images of a product listing (no text input in the basic setup)
	- Tags are from a predefined vocabulary (e.g., 2,000 tags)
- Output:
	- A ranked list or binary vector over the tag vocabulary (multi-label)
- Problem Type
	- Fixed tag vocabulary -> Multi-label classification -> Vector of 0/1 labels or scores per tag
	- Open tag vocabulary -> Retrieval or generative -> Top-k retrieved tags using tag embeddings
- Model Architecture Choices
	- CNNs (e.g., ResNet): Strong baseline, efficient, works with BCE loss
	- Vision Transformers (e.g., ViT): Better generalization, more data-hungry
	- CLIP-style dual encoders: Enables retrieval/zero-shot tagging with tag embeddings
	- Multi-modal models (future): Use image + title/description if available
- Labeling Scenarios
	- Case A: 100k labeled images with tags
		- Finetune a CNN/ViT with BCEWithLogitsLoss
	- Case B: 10k labeled + 1M unlabeled
		- Use semi-supervised learning, self-training, pseudo-labeling
		- Optional: Contrastive pretraining with SimCLR or BYOL
	- Case C: Only curated positive tags, no known negatives
		- Use positive-unlabeled (PU) learning or ranking loss
- Training Setup
	- Preprocessing:
		- Resize, normalize (use dataset-specific mean/std), augmentations
	- Pretraining (optional):
		- Contrastive learning (SimCLR, BYOL) on unlabeled product image corpus
	- Finetuning:
		- Use BCEWithLogitsLoss (independent sigmoid heads)
		- Do not use softmax
		- Optional: Freeze base layers initially, then unfreeze gradually
	- Thresholding:
		- Use global threshold (e.g., 0.5) or tune per-tag thresholds
- Evaluation Metrics
	- Precision@K: How many of top-K predicted tags are correct
	- Recall@K: How many true tags appear in the top-K predictions
	- F1 score (macro and micro)
	- AUC per tag (for threshold tuning)
- Scaling Considerations
	- Multi-GPU training for ViT or large datasets
	- Factorized/tag-bottleneck heads for large vocabularies
	- Index tag embeddings for fast retrieval or zero-shot inference
- Alternative Methods
	- CLIP zero-shot tagging: Embed image and tag descriptions in same space
	- Image-to-tag retrieval: Learn tag embeddings, retrieve nearest
	- Vision-to-text (captioning): Generate pseudo-descriptions, extract tags

Visual Search System (Image-Only)
==========================================================================
- Problem  
	- Enable users to search for products using only an image (e.g., phone-captured photos), matching to semantically similar catalog images.
- Use Cases  
	- Image search via phone camera (e.g., “find similar items”).  
	- Visual discovery experience (Pinterest-style browse).  
	- Helps cold-start users with no typed query.
- Input / Output  
	- Input: Query image (optionally cropped).  
	- Output: Ranked list of product images (or product IDs) from a fixed catalog.
- Problem Type  
	- Image retrieval based on visual similarity (semantic embedding space).  
	- No class prediction, no metadata, no personalization.
- Model Choices - Backbone:  
	- CNN-based: ResNet, EfficientNet, MobileNet (fast inference).  
	- Transformer-based: ViT, DINOv2, DeiT, SAM (better semantics, requires more data).  
- Training Strategy:  
	- Contrastive learning (SimCLR, MoCo, InfoNCE).  
	- Triplet loss or arcface (optional).  
	- Supervised fine-tuning with positive pairs (query ↔ matching catalog images).
- Labeling Scenarios  
	- Case A: 10k manually labeled query ↔ product pairs (positive matches).  
	- Case B: 200M unlabeled mobile photos.  
	- Use clustering, pseudo-labels, weak supervision, or pretraining.  
	- Leverage augmentations on catalog images to synthesize training pairs.
- Training Setup  
	- Pretraining: Contrastive pretraining on product catalog (SimCLR-style) to adapt to product domain.  
	- Finetuning:  
		- On 10k labeled query-product pairs with InfoNCE loss.  
		- Use product embedding = mean pooled embeddings of its multiple images.  
	- Data Augmentations: Blur, crop, resize, grayscale, decolorization to simulate noisy inputs.  
	- Embedding Head: Add projection head (e.g., 2-layer MLP) before retrieval embedding.
- Evaluation Metrics  
	- Recall@k, Precision@k, mAP@k (mean Average Precision).  
	- Retrieval latency and embedding size (efficiency).  
	- Offline: Mean cosine similarity with true match.  
	- Online: Click-through rate (CTR), conversion rate (if measurable).
- Scaling Considerations  
	- Indexing: Use FAISS or ScaNN for approximate nearest neighbors (ANN).  
	- Update index incrementally as new products are added.  
	- Use quantization (PQ/IVF) or knowledge distillation to compress embeddings.  
	- Optional: Use hierarchical retrieval (coarse-to-fine) for speed.
- Alternative Methods  
	- CLIP-style image encoders + product ID supervision (e.g., MIL-NCE).  
	- Self-supervised ViT models (DINOv2) for generalizable embeddings.  
	- Ensemble of CNN + transformer models.  
	- Use DETR/SAM-based region embeddings if user crops objects in the query.

Localized Object Search System (Object-Centric Visual Search)
==========================================================================
- Problem  
	- Users capture an image containing multiple objects and want to search for just one object in the image. 
	- The system detects the region of interest (e.g., via cropping or object detection) and retrieves semantically similar products.
- Use Cases  
	- Tap-to-search on objects (like Google Lens)  
	- Search specific item within a lifestyle image  
	- Visual filters or product detection on seller-uploaded images
- Input / Output  
	- Input: Full image or cropped region from user  
	- Output: Products visually similar to the detected/cropped object
- Problem Type - Two-stage system:  
	- Stage 1: Object detection/localization  
	- Stage 2: Embedding-based retrieval
- Model Choices  
	- Stage 1:  
		- DETR, Faster R-CNN, YOLOv8 (object localization)  
		- SAM for user-assisted segmentation/cropping  
	- Stage 2:  
		- ResNet/ViT/DINOv2 embedding extractor  
		- Projected to common embedding space  
		- Product embedding: mean of region embeddings per product
- Labeling Scenarios  
	- Supervised: object bounding boxes + product match labels  
	- Weakly supervised: click-through logs, cropped images  
	- Self-supervised: augment product images as object crops
- Training Setup  
	- Stage 1: Pretrain detector on product dataset with boxes  
	- Stage 2: Train image embedding model on matched object ↔ product pairs  
	- Optionally fuse detection + embedding (jointly fine-tune)
- Evaluation Metrics  
	- Object localization accuracy (IoU, mAP)  
	- Retrieval metrics: Recall@k, Precision@k for cropped objects  
	- Overall latency (detection + search)
- Scaling Considerations  
	- Cache intermediate crops if common  
	- Use lightweight detectors (YOLO-Nano, MobileSAM)  
	- Optional: Joint detector-embedder model (faster inference)
- Alternative Methods  
	- SAM + embedding on segmented mask  
	- One-stage detector with retrieval head (DELG-style)  
	- Saliency-guided attention cropping without bounding boxes

Product Taxonomy Mapping (Image + Metadata)
==========================================================================
- Problem  
	- Assign a product to a taxonomy node using both the image and product metadata (title and description).
- Input / Output  
	- Input: Product image, title, and description  
	- Output: Category ID (taxonomy node)
- Problem Type  
	Multimodal hierarchical classification
- Model Choices  
	- Multimodal fusion models:  
		- Early fusion: Concatenate image and text embeddings  
		- Late fusion: Separate image and text towers with fusion at classifier level  
	- Base encoders:  
		- Image: ResNet, ViT  
		- Text: BERT, DistilBERT, Sentence-BERT  
	- Fusion techniques: MLP fusion, attention-based fusion, cross-modal transformer
- Labeling Scenarios  
	- Same as image-only, but optionally apply text-based weak supervision  
	- Use keyword extraction to create noisy labels from metadata  
	- Train with human-labeled examples, validate robustness to noisy text
- Training Setup  
	- Pretrain encoders separately or jointly  
	- Finetune with labeled taxonomy classes  
	- Text preprocessing: lowercasing, tokenization, stopword removal  
	- Use dropout and regularization to avoid text overfitting
- Evaluation Metrics  
	- Same as image-only, plus ablations on image-only vs text-only vs multimodal  
	- Optional: evaluate on tail classes separately
- Use Cases  
	- Improved classification performance in ambiguous or visually similar categories  
	- Better coverage for long-tail or rare categories with descriptive text
- Scaling Considerations  
	- Long and noisy text: requires cleaning and truncation  
	- Tradeoff between complexity and latency  
	- Multilingual metadata (requires multilingual text encoder)
- Alternative Methods  
	- Use text-only or image-only when one modality is missing  
	- Use CLIP-like models pretrained on image-text pairs  
	- Train multitask models with auxiliary objectives (e.g., tag prediction)

Dynamic Tag Suggestion (Image + Metadata)
==========================================================================
- Problem
	- Suggest relevant tags (attributes, descriptors) for product listings to improve discovery, search, and categorization.
- Use Cases
	- Improves product discoverability.
	- Drives tag-based browsing and filtering.
	- Feeds into downstream categorization or moderation systems.
- Input / Output
	- Input: Product title, description, and optionally image.
	- Output: Set of 3–10 relevant tags from a fixed tag vocabulary.
- Problem Type
	- Multi-label classification (multiple tags can be correct).
	- Optional: Sequence generation (if tags are open-vocabulary).
- Model Choices
	- Text-only: BERT, DistilBERT, RoBERTa with sigmoid output.
	- Image-text: CLIP-style dual encoders for grounding.
	- Multimodal fusion: Late fusion or cross-attention models.
	- Lightweight: TextCNN or BiGRU + attention for mobile deployment.
- Label Collection - No explicit tags -> weak supervision from seller text
	- Rule-based keyword matching (exact, fuzzy).
	- TF-IDF / RAKE / YAKE for unsupervised keyword extraction.
	- Embedding similarity (BERT/CLIP).
	- Phrase mining (NER, noun phrase chunking).
	- LLM prompting for zero-/few-shot tag extraction.
	- Human-in-the-loop to clean and validate extracted labels.
- Training Setup
	- Loss: Binary cross-entropy with logits.
	- Data imbalance: Weighted sampling or focal loss.
	- Data augmentation: Synonym replacement, dropout, back-translation.
	- Initialization: Pretrained language/image models → fine-tune.
- Evaluation Metrics
	- Precision@k, Recall@k, F1@k.
	- Coverage and diversity of tag suggestions.
	- Manual quality assessment on a small sample.
- Scaling Considerations
	- Efficient inference via pre-computed embeddings.
	- Use tag clustering to reduce vocabulary explosion.
	- Incrementally refresh model with trending tag signals.
- Alternative Methods
	- Tag generation via seq2seq (T5, BART).
	- Retrieval-based tagging (match to nearest products with known tags).
	- Tag co-occurrence graph models.

Multimodal Visual Search System (Image + Text)
==========================================================================
- Problem
	- Enhance search relevance by combining user-provided images with optional free-text (e.g., “red sneakers”) to retrieve matching product entries from the catalog.
- Use Cases
	- “Search this + add description”
	- More accurate queries (“dress like this but in blue”)  
	- Shopping assistants, style filters
- Input / Output  
	- Input:  
		- Query image (phone-captured, optionally cropped)  
		- Optional text query (user-entered keywords)  
	- Output: Ranked product list (by semantic similarity)
- Problem Type  
	- Multimodal retrieval (image + text to image)
- Model Choices  
	- Encoders:  
		- Image: ViT, DINOv2, ResNet (contrastive pretrained)  
		- Text: BERT, DistilBERT, CLIP-Text  
	- Fusion Strategy:  
		- Late fusion: Weighted sum of image/text embeddings  
		- Cross-modal attention (e.g., ALBEF, BLIP)
- Labeling Scenarios  
	- Paired (image, text) examples from product catalog  
	- Manually curated positive query ↔ product matches  
	- Use weak supervision (e.g., co-occurring tags, titles)
- Training Setup  
	- Pretraining: Contrastive alignment of image and text (CLIP-style)  
	- Fine-tuning: Triplet or InfoNCE loss using curated query ↔ product pairs  
	- Fusion tuning: Train a cross-attention head if needed  
	- Embed catalog products with both modalities (combine features)
- Evaluation Metrics  
	- Recall@k, NDCG@k  
	- Multimodal retrieval accuracy  
	- Ablation: image-only, text-only, fused vs. oracle relevance
- Scaling Considerations  
	- Pre-compute and index catalog embeddings  
	- Online combine query embeddings and perform ANN search  
	- Modality dropout during training to handle missing inputs
- Alternative Methods  
	- CLIP or FLAVA for joint image-text space  
	- Late fusion heuristics (weighted linear combination)  
	- Multimodal transformers (e.g., ViLT) for deeper cross-modal reasoning

**************************************************************************
Resources
**************************************************************************
- Multi Modal models

	- [encord.com] `Top 10 Multimodal Models <https://encord.com/blog/top-multimodal-models/>`_
- Vision-text encoder:

	- [medium.com] `Understanding OpenAI’s CLIP model <https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3>`_
	- [amazon.science] `KG-FLIP: Knowledge-guided Fashion-domain Language-Image Pre-training for E-commerce <https://assets.amazon.science/fb/63/9b81471c4b46bad6bd1cbcb591bc/kg-flip-knowledge-guided-fashion-domain-language-image-pre-training-for-e-commerce.pdf>`_
	- [amazon.science] `Unsupervised multi-modal representation learning for high quality retrieval of similar products at e-commerce scale <https://www.amazon.science/publications/unsupervised-multi-modal-representation-learning-for-high-quality-retrieval-of-similar-products-at-e-commerce-scale>`_
- Vision-encoder text-decoder:

	- [amazon.science] `MMT4: Multi modality to text transfer transformer <https://www.amazon.science/publications/mmt4-multi-modality-to-text-transfer-transformer>`_
	- [research.google] `MaMMUT: A simple vision-encoder text-decoder architecture for multimodal tasks <https://research.google/blog/mammut-a-simple-vision-encoder-text-decoder-architecture-for-multimodal-tasks/>`_
	- [medium.com] `Understanding DeepMind’s Flamingo Visual Language Models <https://medium.com/@paluchasz/understanding-flamingo-visual-language-models-bea5eeb05268>`_
- E-commerce publications

	- [amazon.science] `Amazon Science e-Commerce <https://www.amazon.science/publications?q=&f1=0000017b-cb9b-d0be-affb-cbbf08e40000&s=0>`_

Product Categorisation
==========================================================================
- Resources:

	- [arxiv.org] `Semantic Enrichment of E-commerce Taxonomies <https://arxiv.org/abs/2102.05806>`_
	- [arxiv.org] `TaxoEmbed: Product Categorization with Taxonomy-Aware Label Embedding <https://arxiv.org/abs/2010.12862>`_


Multimodal Product Representation
==========================================================================
- Papers:

	- [ieee.org] `Deep Multimodal Representation Learning: A Survey <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8715409>`_
	- [openaccess.thecvf.com] `Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-commerce <https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Learning_Instance-Level_Representation_for_Large-Scale_Multi-Modal_Pretraining_in_E-Commerce_CVPR_2023_paper.pdf>`_
	- [amazon.science] `Unsupervised Multi-Modal Representation Learning for High Quality Retrieval of Similar Products at E-commerce Scale <https://assets.amazon.science/54/5e/df0e19f94b26afb451dd2c156612/unsupervised-multi-modal-representation-learning-for-high-quality-retrieval-of-similar-products-at-e-commerce-scale.pdf>`_

Product Title Normalization & Rewriting
==========================================================================
- Papers:

	- https://paperswithcode.com/task/attribute-value-extraction

Product Deduplication and Matching
==========================================================================
- Goal: Identify duplicate listings across users or platforms (e.g., same product uploaded multiple times).
- Papers:

	- [arxiv.org] `Deep Product Matching for E-commerce Search <https://arxiv.org/abs/1806.06159>`_
	- [arxiv.org] `Multi-modal Product Retrieval in Large-scale E-commerce <https://arxiv.org/abs/2011.09566>`_
