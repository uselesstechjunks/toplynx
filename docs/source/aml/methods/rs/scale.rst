###############################################################################################
Scale
###############################################################################################
***********************************************************************************************
Typical Scale for ML Methods 
***********************************************************************************************
.. csv-table:: 
	:header: Modeling Method, Typical Scale, Real-World Example, Key References / Links
	:align: center
	
	Matrix Factorization / Neural CF, Millions of users; tens–hundreds of thousands of items, Netflix (movie recommendations); Amazon product recommendations, `Koren et al. (2009) <https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-12ea2c9a1b26>`_; `He et al. (2017) <https://arxiv.org/abs/1708.05031>`_
	Autoencoder-Based Models, Up to several million users/items (with careful engineering), Prototype systems in academia and industry (e.g.; LinkedIn/Alibaba experiments), `Sedhain et al. (2015) <https://dl.acm.org/doi/10.1145/2783258.2783304>`_
	CBOW/Item2Vec Models, Tens of millions of items; millions of users, Pinterest’s recommendation pipeline (visual item embeddings), `Barkan & Koenigstein (2016) <https://arxiv.org/abs/1602.03410>`_
	Sequence-Based Models (RNNs/Transformers), Millions of users; operate on session-level subsets (dozens per session), YouTube and TikTok session‑based recommendations, `Kang & McAuley (2018) <https://arxiv.org/abs/1808.09781>`_; `Sun et al. (2019) <https://arxiv.org/abs/1904.01766>`_
	Graph Neural Network-Based Models (PinSage; LightGCN), Up to tens of millions of nodes (combined users and items; often on relevant subgraphs), Pinterest (PinSage) for visual recommendations; e‑commerce systems at Alibaba, `PinSage (Ying et al.; 2018) <https://arxiv.org/abs/1806.01973>`_; `LightGCN (He et al.; 2020) <https://arxiv.org/abs/2002.02126>`_
	Hybrid/Ensemble Approaches, Varies by design; typically millions to tens of millions overall, Spotify’s hybrid playlist generation; Google News combining content and CF, `Spotify Engineering Blog <https://engineering.atspotify.com/>`_; `Google News Blog <https://blog.google/products/google-news/>`_

***********************************************************************************************
Practical Methods
***********************************************************************************************
When scaling from millions to billions of entities (such as YouTube videos, Google search pages, or Facebook’s social graph), several additional engineering and modeling adjustments become necessary. Here are some key considerations and strategies:

1. Distributed and Parallel Processing:  
	
	- Engineering: You must distribute computation across clusters (using frameworks like Apache Spark, TensorFlow, or PyTorch Distributed). This ensures that both training and inference can handle massive datasets.  
	- Example: Google’s search infrastructure and YouTube’s recommendation engine use distributed systems to compute embeddings and run large‑scale inference.
	- Source: `Google’s Bigtable and distributed training systems <https://research.google/pubs/pub38115/>`_

2. Graph Sampling and Partitioning:  
	
	- Engineering: For GNN-based methods, processing the full graph is infeasible. Techniques like neighbor sampling (used in GraphSAGE) or mini‑batch training (used in LightGCN) help process only relevant subgraphs.  
	- Example: Pinterest’s PinSage employs random walks and sampling to generate embeddings on their large-scale graph.
	- Source: `PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Ying et al., 2018) <https://arxiv.org/abs/1806.01973>`_

3. Hierarchical or Multi-Stage Modeling:  
	
	- Modeling: Instead of one monolithic model, use hierarchical models where an initial retrieval model reduces the candidate set, followed by a more refined ranking model. This two-stage process is common in search engines and recommendation systems at Google and Facebook.  
	- Example: YouTube employs a candidate retrieval stage using a two-tower model, then uses a ranking model that might combine sequence models, content signals, and GNNs.
	- Source: `YouTube Recommendations: Deep Neural Networks for YouTube Recommendations (Covington et al., 2016) <https://www.youtube.com/watch?v=2U9U7ThBzI8>`_

4. Approximate Nearest Neighbor (ANN) Search:  

	- Engineering: With billions of items, exact nearest neighbor search becomes too slow. ANN libraries (such as FAISS by Facebook AI) enable efficient retrieval from extremely large embedding spaces.
	- Source: `FAISS: Facebook AI Similarity Search <https://github.com/facebookresearch/faiss>`_

5. Model Compression and Distillation:  

	- Modeling: Large models may need to be compressed or distilled into smaller, more efficient versions that can run in real-time at scale. This can involve quantization, pruning, or teacher-student distillation.
	- Source: `Model Compression <https://arxiv.org/abs/1710.09282>`_

6. Efficient Indexing and Caching:  

	- Engineering: Systems at this scale benefit from specialized data structures (like inverted indices, Bloom filters) and caching layers to quickly serve recommendations and search results.
	- Example: Google Search uses massive indexing systems and caching to ensure low latency.
	- Source: `Google’s Indexing and Ranking Systems <https://research.google/pubs/>`_

7. Adaptive and Incremental Learning:  

	- Modeling: With billions of nodes and edges, the data is continuously evolving. Incremental or online learning techniques allow models to update without retraining from scratch.
	- Example: Facebook uses incremental learning for its social graph updates.
	- Source: `PyTorch-BigGraph <https://research.fb.com/blog/2019/08/pytorch-biggraph-a-large-scale-graph-embedding-system/>`_

.. csv-table:: 
	:header: Adjustment, What It Entails , Real-World Example, Reference
	:align: center

		Distributed & Parallel Processing , Use distributed frameworks (Spark; TensorFlow Distributed) for training/inference , Google Search; YouTube Recommendations , `Google’s Bigtable <https://research.google/pubs/pub38115/>`_
		Graph Sampling & Partitioning , Employ neighbor sampling/mini-batch training in GNNs, Pinterest’s PinSage, `PinSage (Ying et al.; 2018) <https://arxiv.org/abs/1806.01973>`_
		Hierarchical / Multi-Stage Models , Two-stage retrieval and ranking systems, YouTube’s candidate retrieval followed by ranking model , `YouTube Recommendations (Covington et al.; 2016) <https://www.youtube.com/watch?v=2U9U7ThBzI8>`_
		Approximate Nearest Neighbor (ANN), Use ANN search (e.g.; FAISS) for efficient retrieval , Facebook’s similarity search in recommendation systems, `FAISS <https://github.com/facebookresearch/faiss>`_
		Model Compression & Distillation , Compress large models to run in real-time at scale , Applied in many industry systems (Google; Facebook), `Model Compression <https://arxiv.org/abs/1710.09282>`_
		Efficient Indexing & Caching, Specialized indexing data structures and caching layers , Google Search indexing and caching , `Google’s Indexing <https://research.google/pubs/>`_
		Adaptive & Incremental Learning , Update models continuously using online learning techniques, Facebook’s incremental updates on its social graph , `PyTorch-BigGraph <https://research.fb.com/blog/2019/08/pytorch-biggraph-a-large-scale-graph-embedding-system/>`_

