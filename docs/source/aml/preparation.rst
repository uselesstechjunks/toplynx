#####################################################################
Problem Understanding
#####################################################################
*********************************************************************
Level 1: Data
*********************************************************************
This influences whether you use supervised learning, self-supervised pretraining, weak supervision, pseudo-labeling, etc.

	- What modalities are available? (images, text, user logs)
	- Are the labels clean or noisy?
	- How many labeled examples?
	- Any weak, inferred, or behavioral signals I can use?

*********************************************************************
Level 2: Task & Output
*********************************************************************
This defines your loss function, architecture head, evaluation metrics.

	- Is it single-label, multi-label, or ranking?
	- Flat or hierarchical labels?
	- Is the output interpretable or purely latent (like embeddings)?

*********************************************************************
Level 3: System & Constraints
*********************************************************************
This decides model complexity, serving choices, retraining strategies.

	- Is inference real-time or offline?
	- Do we need to support retrieval, tagging, or classification at scale?
	- Can we retrain frequently?
	- Is personalization or user feedback part of the loop?

*********************************************************************
Examples
*********************************************************************
Manual vs Inferred Labels
=====================================================================
- Manual labels (e.g., labeller says: "this is a shoe") → High precision, good for supervised learning.
- Inferred labels (e.g., product clicked after search for "shoes"): → Noisy but abundant. May require:
	- Self-supervised pretraining
	- Positive-unlabeled learning
	- Label smoothing
	- Confidence-based sampling
- If labels are inferred, you can’t blindly fine-tune a classifier. You may overfit to noise, so you’d bring in regularization, semi-supervised learning, or label cleaning.

*********************************************************************
Practice Problems
*********************************************************************
Problem 1: Product Categorization from Images
=====================================================================
You're building a system to classify second-hand products into 500 categories using product photos uploaded by users.

Level 1: Data
	- You have 10k manually labeled images.
	- You also have 1M unlabeled images.
	- Images vary in lighting and quality. Some have blurry backgrounds or text overlays.
Questions:
	- Are the labeled images enough to train a deep model from scratch?
	- Would you use the unlabeled data? If yes, how?
	- Would self-supervised or weakly supervised methods help?

Level 2: Task/Output
	- Each product belongs to exactly one category (single-label classification).
	- Categories are flat (no hierarchy).
	- Evaluation metric: accuracy or top-k accuracy.

Questions:
	- Which loss function fits? Why not BCEWithLogitsLoss?
	- Do you need a softmax output layer?
	- Would label smoothing help?

Level 3: System/Constraints
	- This runs offline on a batch of images every night.
	- Model size and inference time are not major bottlenecks.

Questions:
	- Would you use a ViT or ResNet?
	- Would you unfreeze all layers at once during fine-tuning?

Problem 2: Tag Suggestion from Metadata Text
=====================================================================
You want to suggest up to 5 tags per product based on its title and description. There is no tag label dataset.

Level 1: Data
	- You have 50M product listings with title and description.
	- No human-labeled tags.
	- Tags are often mentioned as keywords in descriptions (e.g., "vintage", "wooden").

Questions:
	- What labeling strategies can help? (e.g., keyword extraction, pseudo-labels)
	- Can you use weak supervision?

Level 2: Task/Output
	- Multi-label classification: many tags can apply to one product.
	- Output is a vector of tag probabilities.

Questions:
	- Which loss function to use?
	- What model architecture can handle text well?

Level 3: System/Constraints
	- Real-time tagging is needed at listing time.
	- Model size matters, latency budget <100ms.

Questions:
	- Can you distill a large model into a smaller one?
	- Would self-distillation or self-ensembling help?

Problem 3: Image-Based Search
=====================================================================
Users upload a photo, and your system returns visually similar products from a catalog of 2M items.

Level 1: Data
	- Each catalog product has 5 images.
	- 10k query-product match examples available from human labels.
	- Additional 200M unlabeled mobile images.

Questions:
	- Would you use contrastive learning? Which strategy?
	- How to handle domain shift between catalog and mobile photos?

Level 2: Task/Output
	- Output is a ranked list of similar items.
	- There's no classification; this is metric learning.

Questions:
	- Which loss function works best (e.g., InfoNCE, Triplet)?
	- Do you need a projection head?

Level 3: System/Constraints
	- Real-time visual search.
	- Must embed all catalog images ahead of time.

Questions:
	- How to structure image indexing (ANN, Faiss)?
	- Can you compress embeddings (e.g., PQ)?

Problem 4: Text-Based Product Search
=====================================================================
Users search using a short text query (e.g., "wooden coffee table"). Your system must return relevant product listings based on title and description.

Level 1: Data
	- 100M product listings with title + description.
	- No query-product match labels.
	- Historical click data available.

Questions:
	- Can you mine pseudo labels from clicks?
	- Would training a dual encoder help?

Level 2: Task/Output
	- Output: ranked list of products based on relevance.
	- Matching task: semantic similarity between query and listing.

Questions:
	- Metric learning vs classification: which is better here?
	- Should you use pointwise, pairwise, or listwise loss?

Level 3: System/Constraints
	- Real-time response required for queries.
	- Embeddings for listings can be precomputed.

Questions:
	- How to design query encoder vs listing encoder?
	- Can you use ANN for retrieval?
	- Is it worth fine-tuning a pretrained text encoder?

Problem 5: Duplicate Product Detection
=====================================================================
You want to flag near-duplicate product listings to improve catalog quality.

Level 1: Data
	- 1M product listings (title, description, images).
	- No labeled duplicates.
	- Some sellers repost similar listings with minor edits.

Questions:
	- Can you mine positives from edit distance or image hash?
	- What makes two listings "duplicates"? Define positive pairs.

Level 2: Task/Output
	- Binary classification: duplicate vs not duplicate.
	- Or: compute similarity score between listing pairs.

Questions:
	- Classification or metric learning?
	- What features do you extract from text and image?

Level 3: System/Constraints
	- Large-scale comparison needed (~billions of pairs).
	- Needs to run offline.

Questions:
	- How to scale pairwise similarity computation?
	- Use blocking or ANN?

Problem 6: Policy Violation Detection
=====================================================================
Detect listings that violate platform policies (e.g., prohibited items, misleading info).

Level 1: Data
	- Small set of labeled violations (5k examples).
	- 100M listings total (text and image).
	- Some listings contain subtle violations.

Questions:
	- Use active learning to expand violation examples?
	- Can self-training or PU learning help?

Level 2: Task/Output
	- Binary classification: violation vs not.
	- Possibly multiple violation types later.

Questions:
	- Should you model it as multi-class or multi-label?
	- Use focal loss to handle class imbalance?

Level 3: System/Constraints
	- Needs to run before listings are published.
	- High precision required to avoid flagging false positives.

Questions:
	- Can you ensemble multiple models (text-only, image-only)?
	- Would hierarchical review pipeline help?
