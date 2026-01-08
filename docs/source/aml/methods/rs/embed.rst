###############################################################################################
Embeddings
###############################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

In designing recommendation systems, the quality of user and item embeddings is critical for both retrieval (finding a manageable candidate set) and ranking (ordering items for relevance). Our modeling choices often aim to capture three interrelated aspects:

1. Long-Term Interests: The enduring tastes and preferences of a user, learned over a long history.
2. Short-Term/Session Interests: The dynamic, situational, or ephemeral interests that might dominate a single session.
3. Content Diversity: Ensuring that recommendations cover a broad range of topics or styles, which helps with exploration and prevents filter bubbles.

Different approaches tend to emphasize one or more of these aspects, and each comes with trade-offs.

***********************************************************************************************
Modeling Approaches
***********************************************************************************************
===============================================================================================
1. Feedback-Based Latent Factor Models (Collaborative Filtering)
===============================================================================================
- Approach: Methods such as Matrix Factorization (MF) and Neural Collaborative Filtering (NCF) learn user and item embeddings solely from historical feedback (e.g., clicks, ratings, watch time).
- Pros:
	
	- Long-Term Interests: They capture users’ aggregated, long-term preferences very well.
	- Retrieval Efficiency: The embeddings tend to be dense and suitable for approximate nearest neighbor (ANN) search.
- Cons:
	
	- Cold-Start: Struggle with new users/items that lack historical data.
	- Short-Term Interest & Diversity: Tend to reinforce established tastes, possibly creating filter bubbles.
- Key Papers:
	
	- "Matrix Factorization Techniques for Recommender Systems" - Koren et al. (2009)
	- "Neural Collaborative Filtering" - He et al. (2017)

===============================================================================================
2. Unsupervised/Feedback-Free Latent Models (Autoencoders, Self-Supervised Learning)
===============================================================================================
- Approach: Models like autoencoders learn latent representations of items (and sometimes users) without directly relying on explicit feedback. They may reconstruct item features or co-occurrence patterns.
- Pros:
	
	- Content Diversity: Can learn general item topics and characteristics even without user interactions.
	- Cold-Start Mitigation: They can leverage available item metadata to generate embeddings for new items.
- Cons:
	
	- User Profiling: Constructing user embeddings from such item embeddings may require additional aggregation strategies (like max-pooling or attention).
	- Retrieval: The learned latent space might not be as optimized for fast retrieval.
- Key Papers:
	
	- "AutoRec: Autoencoders Meet Collaborative Filtering" - Sedhain et al. (2015)
- Nuance: These methods can serve as pretraining for further supervised fine-tuning using user feedback.

===============================================================================================
3. CBOW-Style Models (Item2Vec)
===============================================================================================
- Approach: Inspired by word2vec, item2vec treats sequences of items (from sessions or purchase logs) as "sentences" and learns embeddings based on co-occurrence.
- Pros:
	
	- Short-Term Interest: Excellent for capturing the context of items within a session.
	- Retrieval: Embeddings tend to be tightly clustered, making them effective for nearest neighbor retrieval.
- Cons:

	- Ranking: Often need additional fine-tuning to be useful in a ranking model.
	- Long-Term Interests: They might miss long-term preference signals if sessions are very short.
- Key Papers:

	- "Item2vec: Neural Item Embedding for Collaborative Filtering" - Barkan & Koenigstein (2016)
- Nuance: Item2vec works well when combined with temporal information or merged with feedback-based models.

===============================================================================================
4. Sequence-Based Models (RNNs, Transformers)
===============================================================================================
- Approach: Sequence models (such as RNNs or Transformers like SASRec, Transformer4Rec) capture the order and context of user interactions, emphasizing short-term behavior.
- Pros:

	- Short-Term Interests: Particularly adept at modeling session dynamics.
	- Context Sensitivity: They adapt to recent user behavior, capturing transient interests.
- Cons:

	- Long-Term Memory: May underrepresent long-term preferences unless explicitly integrated (e.g., via hierarchical models).
	- Retrieval: While excellent for ranking, these models typically require extra steps or approximations to be used for fast retrieval.
- Key Papers:

	- "SASRec: Self-Attentive Sequential Recommendation" - Kang & McAuley (2018)
	- "Transformer4Rec: Sequential Recommendation with Self-Attention" - Sun et al. (2019)
- Nuance: They can incorporate explicit time decay or multi-scale architectures to blend short-term and long-term signals.

===============================================================================================
5. Graph Neural Network (GNN)-Based Models
===============================================================================================
- Approach: GNNs treat users and items as nodes in a graph, leveraging their interactions (edges) to learn embeddings through multi-hop message passing.
- Pros:

	- Long-Term & Cross-Domain Interests: GNNs excel at capturing indirect relationships and discovering latent, multi-hop connections, which can introduce content diversity.
	- Cold-Start: By incorporating side information (e.g., item metadata), GNNs can alleviate cold-start issues.
- Cons:

	- Temporal Dynamics: Standard GNNs may not naturally capture sequential or temporal patterns.
	- Scalability: They can be computationally intensive, especially on large graphs.
- Key Papers:

	- "PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems" - Ying et al. (2018)
	- "Graph Convolutional Matrix Completion" - Berg et al. (2017)
- Nuance: Temporal extensions (e.g., incorporating time-aware graph neural networks) can help bridge the gap between static graph structure and evolving user behavior.

===============================================================================================
6. Hybrid and Ensemble Methods
===============================================================================================
- Approach: Combine two or more of the above techniques to balance long-term interests, short-term behavior, and content diversity. Hybrid approaches might merge collaborative signals with content-based features using multi-view learning or re-ranking strategies.
- Pros:

	- Flexibility: Can leverage the strengths of multiple models (e.g., using MF for long-term signals and Transformers for session-level signals).
	- Diversity: Hybrid methods tend to increase diversity by integrating complementary information.
- Cons:

	- Complexity: They require careful engineering to balance contributions from each component.
	- Computational Cost: Ensembles are typically more resource-intensive.
- Key Papers:

	- "Deep Hybrid Recommender Systems" - Zhang et al. (2019) (as an example)
- Nuance: A common approach is to use a two-stage pipeline—first, a retrieval phase (e.g., using item2vec or MF) and then a ranking phase (e.g., using Transformers or GNN-based re-ranking) augmented with content features.

===============================================================================================
7. Utilizing Domain-Specific Content Understanding
===============================================================================================
- Across all these approaches, incorporating content features can significantly enhance performance, especially for cold-start problems and diversity:
- Text-Based Items:

	- Use pretrained models like word2vec, GloVe, or BERT to extract semantic embeddings from descriptions, reviews, or titles.
- Image-Based Items:  

	- Employ models like CLIP to generate image embeddings that capture visual semantics.
- Video-Based Items:  

	- Use video-specific models (e.g., VideoBERT, TimeSformer) to capture both visual and temporal aspects.
- Fusion Strategies:  

	- Combine these content embeddings with collaborative signals through techniques like concatenation, attention-based fusion, or multi-view learning, providing richer and more robust representations.

***********************************************************************************************
Summary
***********************************************************************************************
- Each of these modeling approaches has distinct strengths and trade-offs in addressing long-term interests, short-term dynamics, and content diversity:

	- Feedback-based models are excellent at capturing enduring tastes but can risk filter bubbles and cold-start issues.
	- Unsupervised/autoencoder approaches help in learning general item representations without relying solely on user feedback, aiding cold-start.
	- CBOW/Item2Vec methods excel in short-term, session-level similarities ideal for fast retrieval.
	- Sequence-based models capture the temporal context, useful for session-based or sequential recommendations.
	- Graph Neural Networks integrate multi-hop relationships, aiding in discovering diverse and latent associations.
	- Hybrid models combine multiple signals to balance personalization with exploration.
	- Content integration (using BERT, CLIP, VideoBERT, etc.) further enriches these embeddings, especially in addressing new items or complex content semantics.
- Integrating domain-specific content features (via pretrained embeddings) can further address cold-start issues and enhance the semantic richness of both user and item representations.

***********************************************************************************************
Table
***********************************************************************************************
.. csv-table::
	:header: "Modeling Approach", "Captures Long-Term Interests", "Captures Short-Term Interests", "Handles Cold-Start/Content Diversity", "Pros", "Cons", "Key Papers"
	:align: center
	
	Feedback-Based Latent Models (MF; NCF), High, Moderate, Low (needs augmentation), Effective at modeling long-term preferences; efficient ANN retrieval, Cold-start issues; risk of filter bubbles, Koren et al. (2009); He et al. (2017)
	Unsupervised/Autoencoder Models, Moderate, Low, Moderate (with content features), Learns unsupervised representations; good for pretraining, May need additional strategies to aggregate user profiles; retrieval not optimized, Sedhain et al. (2015)
	CBOW/Item2Vec Models, Moderate, High, Low (pure co-occurrence), Excellent for short-term; context-based similarity; effective for retrieval, May require fine-tuning for ranking; limited long-term signals, Barkan & Koenigstein (2016)
	Sequence-Based Models (Transformers/RNNs), Low-Moderate, Very High, Low (unless fused with content), Excels in session-based recommendations; captures sequential dependencies, May underrepresent long-term interests; needs extra tuning for fast retrieval, Kang & McAuley (2018); Sun et al. (2019)
	Graph Neural Networks, High, Moderate, Moderate (with side-information), Captures complex multi-hop relationships; promotes exploration and diversity, Computationally expensive; may miss temporal dynamics, Ying et al. (2018); Berg et al. (2017)
	Hybrid/Ensemble Approaches, High (if well-combined), High (if including sequential models), High (leverages content-based features), Balances multiple facets of user behavior; increased robustness, Higher system complexity; tuning challenges; more computational cost, Zhang et al. (2019) (for hybrid approaches)

***********************************************************************************************
Real World Applications
***********************************************************************************************
.. csv-table::
	:header: "Modeling Method", "Real-World Application", "Source / Link"
	:align: center

	Feedback-Based Latent Models (MF; Neural CF), Netflix: Uses matrix factorization techniques to capture long-term movie preferences for recommendations, Amazon: Employs item-to-item collaborative filtering (a variant of MF) for product suggestions; `Netflix Tech Blog: Beyond the 5 Stars <https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-12ea2c9a1b26>`_
	Unsupervised/Autoencoder Models, AutoRec-based Approaches: Academic studies (influencing industrial systems like those at LinkedIn or Alibaba) use autoencoders to learn latent representations from sparse interaction data; which can be fine-tuned later for ranking., `AutoRec: Autoencoders Meet Collaborative Filtering (Sedhain et al.; 2015) <https://dl.acm.org/doi/10.1145/2783258.2783304>`_
	CBOW/Item2Vec Models, Pinterest: Leverages item2vec–like models to learn visual item embeddings for pin recommendations.Spotify: Utilizes similar co-occurrence based embeddings for song similarity and retrieval., `Item2vec: Neural Item Embedding for Collaborative Filtering (Barkan & Koenigstein; 2016) <https://arxiv.org/abs/1602.03410>`_
	Sequence-Based Models (Transformers/RNNs), YouTube: Employs sequence models (e.g.; SASRec; Transformers) to capture session context for video recommendations, TikTok: Uses RNN/Transformer architectures to model short-term user behavior on its “For You” page.; `YouTube Recommendations (Covington et al.; 2016) <https://www.youtube.com/watch?v=2U9U7ThBzI8>`_; `TikTok Business Blog <https://newsroom.tiktok.com/en-us/>`_
	Graph Neural Network-Based Models, Pinterest’s PinSage: Uses GNNs to generate high-quality image embeddings from a large-scale pin-interaction graph.Alibaba: Applies GNN-based methods to incorporate user–item interaction graphs for product recommendations., `PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Ying et al.; 2018) <https://arxiv.org/abs/1806.01973>`_
	Hybrid/Ensemble Approaches, Spotify: Combines collaborative filtering with deep audio content features (extracted via CNNs or pretrained models) to produce diverse; personalized playlists, Google News: Fuses collaborative signals with content understanding (e.g.; BERT embeddings) to tailor news recommendations.; `Spotify Engineering Blog <https://engineering.atspotify.com/2015/11/spotify-recommendation-engineering/>`_; `Google News Blog <https://blog.google/products/google-news/>`_
	Utilizing Domain-Specific Content Features, Amazon: Augments collaborative signals with text embeddings from models like BERT on product descriptions and reviews to address cold-start and enrich similarity measures; YouTube: Uses video embeddings from models such as VideoBERT to capture rich visual and temporal semantics for video retrieval and ranking., `VideoBERT: A Joint Model for Video and Language Representation Learning <https://arxiv.org/abs/1904.01766>`_
