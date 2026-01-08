###############################################################################
Machine Learning Methods
###############################################################################
.. toctree::
	:maxdepth: 2

	rs/index
	semantic/index
	tech/index

*******************************************************************************
**1. Supervised Learning**
*******************************************************************************
- **Multi-class classification**: One label per input, softmax + cross-entropy.
- **Multi-label classification**: Multiple labels per input, sigmoid + BCE loss.
- **Hierarchical classification**: Class labels follow a tree structure; improves consistency.
- **Metric learning**: Learn embedding space where similar items are close; e.g., InfoNCE, triplet loss.

*******************************************************************************
**2. Self-supervised & Weak Supervision**
*******************************************************************************
- **SimCLR/MoCo**: Contrastive learning using data augmentations to learn representations.
- **BYOL/Barlow Twins**: Learn features without negative samples, using prediction tasks.
- **Pseudo-labeling**: Use confident model predictions as temporary labels.
- **Positive-unlabeled (PU) learning**: Learn from known positives and a large unlabeled pool.
- **Co-training/Democratic Co-training**: Train models on different views or data splits and teach each other.

*******************************************************************************
**3. Fine-tuning Strategies**
*******************************************************************************
- **Linear probing**: Freeze base encoder, train only the classifier head.
- **Full fine-tuning**: Update all layers on downstream data.
- **Gradual unfreezing**: Unfreeze layers progressively during training.
- **LoRA**: Inject low-rank adapters into transformer layers, efficient for fine-tuning.
- **Adapters**: Plug-in small trainable modules between frozen layers.

*******************************************************************************
**4. Labeling Techniques**
*******************************************************************************
- **Manual labels**: Curated by human annotators.
- **Rule-based labels**: Heuristics from metadata or structure.
- **Implicit feedback**: Use user clicks, views, etc. as signals.
- **Distant supervision**: Use external knowledge bases for labels.
- **Active learning**: Query the most informative samples to label.

*******************************************************************************
**5. Training Tricks**
*******************************************************************************
- **Hard negative mining**: Select tough negatives to improve contrastive/matching learning.
- **Self-training**: Train a model on pseudo-labeled data iteratively.
- **Consistency regularization**: Penalize inconsistent predictions under augmentations.
- **Self-ensembling**: Use predictions from multiple model states or augmentations.
- **Label smoothing**: Prevent overconfidence by softening one-hot labels.

*******************************************************************************
**6. Representation & Retrieval**
*******************************************************************************
- **Dual encoders**: Separate encoders for query and document; allows fast retrieval.
- **ANN search (FAISS/ScaNN)**: Approximate nearest neighbors for efficient vector retrieval.
- **Vector quantization (VQ)**: Compress embeddings using codebooks for faster search.
- **Product quantization (PQ)**: Divide vectors into subspaces for scalable retrieval.

*******************************************************************************
**7. Fusion Methods (for multimodal data)**
*******************************************************************************
- **Early fusion**: Combine raw inputs before encoding (e.g., concat text and image).
- **Late fusion**: Combine final outputs or predictions from each modality.
- **Cross-modal attention**: Let one modality attend over another (e.g., in ViLT or Flamingo).

*******************************************************************************
8. Domain Trade-offs
*******************************************************************************
For each domain (e.g., commerce, UGC, jobs, food, news, search):

- Phase 1: Foundation Across Domains
	- Retrieval objective + index design
	- Item and user embedding modeling
	- Label shaping + feedback signal bias
	- Cold-start & tail item strategy
- Phase 2: Ranking Layer Design
	- Crossing techniques: concat, attention, deep crossing, FiLM
	- Personalization fusion: long-term vs short-term interests
	- Loss function trade-offs: point/pair/list/ordinal
	- Feature latency & stale signal impact
- Phase 3: Scaling + Infrastructure Trade-offs
	- ANN system design: PQ vs HNSW vs IVF
	- Embedding refresh frequency + tag injection
	- Multi-source hybrid retrieval + diversity injection
	- Shadow evaluation, drift detection
- Phase 4: Monitoring & Bias
	- Feature delay, bias amplification
	- Observability for ANN and ranking stages
	- Coverage/fairness audits, click model correction
	- Calibration, post-hoc re-ranking, threshold tuning
