####################################################################################
Notes on Search & Recommendation
####################################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

************************************************************************************
Overview
************************************************************************************
.. warning::

	* Overview of search and recsys - different stages
	* Metrics, Modelling for different stages
	* Application of LLMs at different stages
	* General problems
	* Domain specific problems

.. important::
	- Entities

		- Users, items (text, image, video, nodes), interactions, context
	- Labels

		- Supervised, semi-supervised (proxy label), self-supervised, unsupervised
	- Patterns

		- Query-Item, User-Item, Item-Item, Session, User-User
	- Objectives & metrics

		- Accuracy Precision@k, Recall@k, MAP@k, NDCG@k, MRR@k, HR@k
		- Behavioral Diversity, Novelty, Serendipity, Popularity-bias, Personalisation, Fairness
		- Monitoring Drift metrics
	- Considerations in model training

		- Training window Seasonality, Data leak
		- Deciding on labels
	- Stages

		- Retrieval, Filtering, Rerank
	- Models

		- Retrieval

			- Content-based Filtering
			- Collaborative Filtering - MF/Neural CF
			- GCN - LightGCN
			- Sequence - Transformers
		- Filtering

			- Ruled based
		- Rerank
		
			- GBDT, NN, DCN, WDN, DPP
	- Domains

		- Search Advertising
		- Music
		- Video
		- E-commerce
		- Social media
	- Issues

		- General

			#. Cold-start
			#. Diversity vs. personalization Trade-Off
			#. Popularity bias & fairness
			#. Short-term engagement vs. long-term user retention trade-off
			#. Privacy concerns & compliance (GDPR, CCPA)
			#. Distribution shift (data/input, concept/target)
		- Advanced

			#. Multi-touch Attribution
			#. Real-time personalization & latency trade-Offs
			#. Cross-device and cross-session personalization
			#. Multi-modality & cross-domain recommendation challenges
		- Domain-Specific

			#. Search Query understanding & intent disambiguation
			#. E-Commerce Balancing revenue & user satisfaction
			#. Video & Music Streaming Content-length bias in recommendations

Metrics
====================================================================================
Accuracy
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Metric", "Full Name", "Formula", "Desc", "Drawback"
	:align: center
	
		HR@k, Hit-rate at k, , ,
		Recall@k, Recall at k, , ,
		NDCG@k, Normalized Discounted Cumulative Gain at k, , ,

Popularity Bias
------------------------------------------------------------------------------------
.. note::
	* :math:`U`: Set of all users
	* :math:`I`: Set of all items
	* :math:`L_u`: List of items (concatenated) impressed for user :math:`u`
	* :math:`L`: All list of items (concatenated)

.. csv-table::
	:header: "Metric", "Full Name", "Formula", "Note", "Drawback"
	:align: center
	
		ARP, Average Recommendation Popularity, :math:`\frac{1}{|U|}\sum_{u\in U}\frac{\sum_{i\in L_u}\phi(i)}{|L_u|}`, Average CTR across users, Good (low) value doesn't indicate coverage
		Agg-Div, Aggregate Diversity, :math:`\frac{|\bigcup_{u\in U}L_u|}{|I|}`, Item Coverage, Doesn't detect skew in impression
		Gini, Gini Index, :math:`1-\frac{1}{|I|-1}\sum_{k}^{|I|}(2k-|I|-1)p(i_k|L)`, :math:`p(i_k|L)` how many times :math:`i_k` occured in `L`, Ignores user preference
		UDP, User Popularity Deviation, , ,

Diversity
------------------------------------------------------------------------------------
Personalsation
------------------------------------------------------------------------------------
Issues
====================================================================================
Distribution Shift
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Problem", "How to Detect", "How to Fix", "Trade-Offs"
	:align: center 

		Model Degradation, Performance drop (CTR; engagement), Frequent model retraining, Computationally expensive
		Popularity Mismatch, PSI; JSD; embeddings drift, Adaptive reweighting of historical data, Hard to balance long vs. short-term relevance
		Bias Reinforcement, Disparity in exposure metrics, Fairness-aware ranking, May hurt engagement
		Cold-Start for New Trends, Increase in unseen queries, Session-based personalization, Requires fast inference
		Intent Drift in Search, Increase in irrelevant search rankings, Online learning models, Real-time training is costly

Stages
====================================================================================
.. csv-table::
	:header: "Stage", "Goals", "Key Metrics", "Common Techniques"
	:align: center

		Retrieval, Fetch diverse candidates from multiple sources, Recall@K; Coverage; Latency, Multi-tower models; ANN; User embeddings
		Combining & Filtering, Merge candidates; remove duplicates; apply business rules, Diversity; Precision@K; Fairness, Weighted merging; Min-hashing; Rule-based filtering
		Re-Ranking, Optimize order of recommendations for engagement, CTR; NDCG; Exploration Ratio, Neural Rankers; Bandits; DPP for diversity

Patterns
====================================================================================
.. csv-table::
	:header: "Pattern", "Traditional Approach", "LLM Augmentations"
	:align: center

		Query-Item, BM25; TF-IDF; Neural Ranking, LLM-based reranking; Query expansion
		Item-Item, Co-occurrence; Similarity Matching, Semantic matching; Multimodal embeddings
		User-Item, CF; Content-Based; Deep Learning, Personalized generation; Zero-shot preferences
		Session-Based, Sequential Models; Transformers, Few-shot reasoning; Context-aware personalization
		User-User, Graph-Based; Link Prediction, Profile-text analysis; Social graph augmentation

************************************************************************************
Stages
************************************************************************************
Retrieval 
====================================================================================
(Fetching an initial candidate pool from multiple sources) 

Task
------------------------------------------------------------------------------------
	- Reduce a large item pool (millions of candidates) to a manageable number (thousands). 
	- Retrieve diverse candidates from multiple sources that might be relevant to the user. 
	- Balance long-term preferences vs. short-term intent. 

Comon Metrics
------------------------------------------------------------------------------------
	- Recall@K – How many relevant items are in the top-K retrieved items? 
	- Coverage – Ensuring diversity by retrieving from multiple pools. 
	- Latency – Efficient retrieval in milliseconds at large scales. 

Common Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Heterogeneous Candidate Retrieval, Multi-tower models; Hybrid retrieval (Collaborative Filtering + Content-Based)
		Personalization, User embeddings (e.g.; Two-Tower models; Matrix Factorization)
		Exploration & Freshness, Real-time embeddings; Bandit-based exploration
		Scalability & Efficiency, Approximate Nearest Neighbors (ANN); FAISS; HNSW
		Cold-Start Handling, Content-based retrieval (TF-IDF; BERT); Popularity-based heuristics

Example - YouTube Recommendation 
------------------------------------------------------------------------------------
	- Candidate pools Watched videos, partially watched videos, topic-based videos, demographically popular videos, newly uploaded videos, videos from followed channels. 
	- Techniques used Two-Tower model for retrieval, Approximate Nearest Neighbors (ANN) for fast lookup. 

Combining & Filtering 
====================================================================================
(Merging retrieved candidates from different sources and removing low-quality items) 

Task
------------------------------------------------------------------------------------
	- Merge multiple retrieved pools and assign confidence scores to each source. 
	- Filter out irrelevant, duplicate, or low-quality candidates. 
	- Apply business rules (e.g., compliance filtering, removing expired content). 

Comon Metrics
------------------------------------------------------------------------------------
	- Diversity – Ensuring different content types are represented. 
	- Precision@K – How many retrieved items are actually relevant? 
	- Fairness & Representation – Avoiding over-exposure of popular items. 
	- Latency – Keeping the filtering process efficient. 

Common Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Merging Multiple Candidate Pools, Weighted aggregation based on confidence scores
		Duplicate Removal, Min-hashing; Jaccard similarity; clustering-based deduplication
		Quality Filtering, Heuristic filters; Rule-based filters; Adversarial detection
		Business Constraints, Compliance rules (e.g.; sensitive content removal); Content freshness checks
		Balancing Diversity, Re-weighting based on underrepresented categories
		Scaling Up, Streaming pipelines (Kafka; Flink); Pre-filtering with Bloom Filters

Example - Newsfeed Recommendation 
------------------------------------------------------------------------------------
	- Candidate sources Text posts, image posts, video posts. 
	- Filtering techniques Removing duplicate posts, blocking low-quality content, filtering based on engagement thresholds. 

Re-Ranking 
====================================================================================
Task
------------------------------------------------------------------------------------
	- Optimize the order of candidates to maximize engagement. 
	- Balance personalization with exploration (ensuring new content gets surfaced). 
	- Ensure fairness and representation (avoid showing only highly popular items). 

Metrics
------------------------------------------------------------------------------------
	- [Offline] AUC (ROC-AUC, PR-AUC) – Measures prediction accuracy if modeled as a binary classification problem.
	- [Offline] NDCG@k, MRR@k, HR@k – Measures ranking quality.
	- [Online] CTR (Click-Through Rate) – Measures immediate engagement.
	- [Online] Long-Term Engagement – Holdout -> Measures retention and repeat interactions.
	- [?] Exploration Ratio – Tracks new content shown to users.

Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Fast Re-Ranking, Tree-based models (GBDT); LightGBM; XGBoost
		Personalized Ranking, Embed + MLP Models (e.g.; DeepFM; Wide & Deep; Transformer-based rankers)
		Diversity Promotion, Re-ranking by category (e.g.; Round Robin); Determinantal Point Processes (DPP)
		Explore-Exploit Balance, Multi-Armed Bandits (Thompson Sampling; UCB); Randomized Ranking
		Handling Highly Popular Items, Popularity dampening; Re-ranking with popularity decay
		Fairness & Representation, Re-weighting models; Exposure-aware ranking		

Resources
------------------------------------------------------------------------------------
Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Features
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	- User profile (captures long term user's preferences)
	- Item profile (captures item metadata and content understanding)
	- Contextual features (e.g, device, geolocation, temporal)
	- Interaction features

************************************************************************************
Patterns
************************************************************************************
Query-Item Recommendation 
====================================================================================
- Search systems
- text-to-item search
- image-to-item search
- query expansion techniques

Key Concept 
------------------------------------------------------------------------------------
- Common Approaches

	- Lexical Matching (TF-IDF, BM25, keyword-based retrieval) 
	- Semantic Matching (Word embeddings, Transformer models like BERT, CLIP for vision-text matching) 
	- Hybrid Search (Combining lexical and semantic search, e.g., BM25 + embeddings) 
	- Learning-to-Rank (LTR) models optimizing ranking performance based on user interactions) 
	- Multimodal Search (Image-to-text retrieval, video search, voice search, etc.) 
- LLM Applications

	- LLMs enhance ranking via reranking models (ColBERT, T5-based retrieval). 
	- Can be used for query expansion, understanding user intent, and handling ambiguous queries. 
	- Example use case Google Search, AI-driven Q&A search (Perplexity AI). 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (clicked vs. not clicked) or relevance score (explicit ratings, dwell time). 
	- Data sources Search logs, query-click data, user feedback (thumbs up/down). 
	- Challenges Noisy labels (e.g., clicks may not always indicate relevance). 
#. Semi-Supervised Learning 

	- Use query expansion techniques (e.g., weak supervision from similar queries). 
	- Leverage pseudo-labeling (e.g., use a weaker ranker to generate labels). 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., train embeddings by pulling query and relevant items closer). 
	- Masked query prediction (e.g., predicting missing words in search queries). 

Common Features
------------------------------------------------------------------------------------
- Query Features Term frequency, query length, part-of-speech tagging. 
- Item Features Title, description, category, metadata, embeddings. 
- Interaction Features Click history, query-to-item dwell time, CTR. 
- Contextual Features Time of query, device type, user history. 
- Embedding-Based Features Pretrained word embeddings (Word2Vec, FastText, BERT embeddings). 

Resources
------------------------------------------------------------------------------------
#. Traditional Information Retrieval 

	- "An Introduction to Information Retrieval" – Manning et al. (2008) 
	- "BM25 and Beyond" – Robertson et al. (2009) 
#. Neural Ranking Models 

	- "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding" – Devlin et al. (2018) 
	- "Dense Passage Retrieval for Open-Domain Question Answering" – Karpukhin et al. (2020) 
#. Multimodal & Deep Learning-Based Search 

	- "CLIP Learning Transferable Visual Models from Natural Language Supervision" – Radford et al. (2021) 
	- "DeepRank A New Deep Architecture for Relevance Ranking in Information Retrieval" – Pang et al. (2017) 
#. LLM-Based Search Ranking 

	- "ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT" – Khattab et al. (2020) 
	- "T5 for Information Retrieval" – Nogueira et al. (2020) 
#. LLM-Augmented Search 

	- "InstructGPT for Information Retrieval" – Ouyang et al. (2023) 
	- "GPT-4 for Web Search Augmentation" – Bender et al. (2023) 

Item-Item Recommendation 
====================================================================================
- Similar Products
- Related Videos
- "Customers Who Bought This Also Bought"

Key Concept 
------------------------------------------------------------------------------------
- Item-item recommendation focuses on suggesting similar items based on user interactions. This is widely used in e-commerce, streaming platforms, and content discovery systems. 

	- Typically modeled as an item simi-larity problem. 
	- Unlike user-item recommendation, the goal is to find related items rather than predicting a user’s preferences. 
- Common Approaches

	- Item-Based Collaborative Filtering (Similarity between item interaction histories) 
	- Content-Based Filtering (Similarity using item attributes like text, image, category) 
	- Graph-Based Approaches (Item-item similarity using co-purchase graphs) 
	- Deep Learning Methods (Representation learning, embeddings) 
	- Hybrid Methods (Combining multiple approaches) 
- LLM Applications

	- LLMs improve semantic similarity scoring, identifying nuanced item relationships.
	- Multimodal LLMs (e.g., CLIP) combine text, images, and metadata to enhance recommendations.
	- Example use case E-commerce (Amazon's “similar items”), content platforms (Netflix’s related videos).

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (1 = two items are similar, 0 = not similar). 
	- Data sources Co-purchase data, co-click data, content similarity. 
	- Challenges Defining meaningful similarity when explicit labels don’t exist. 
#. Semi-Supervised Learning 

	- Clustering similar items based on embeddings or co-occurrence. 
	- Weak supervision from user-generated tags, reviews. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., learning embeddings by pushing dissimilar items apart). 
	- Masked item prediction (e.g., predicting missing related items in a session). 

Common Features
------------------------------------------------------------------------------------
- Item Features Category, brand, price, textual description, images. 
- Interaction Features Co-purchase counts, view sequences, co-engagement. 
- Graph Features Item co-occurrence in user sessions, citation networks. 
- Embedding-Based Features Learned latent item representations. 
- Contextual Features Time decay (trending vs. evergreen items).  

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches 

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001) 
	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009) 
#. Content-Based Approaches 

	- "Learning Deep Representations for Content-Based Recommendation" – Wang et al. (2015) 
	- "Deep Learning Based Recommender System A Survey and New Perspectives" – Zhang et al. (2019) 
#. Graph-Based & Hybrid Approaches 

	- "Amazon.com Recommendations Item-to-Item Collaborative Filtering" – Linden et al. (2003) 
	- "PinSage Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018) 
#. Multimodal LLMs for Recommendation 

	- "CLIP-Recommend Multimodal Learning for E-Commerce Recommendations" – Xu et al. (2023) 
	- "Unified Vision-Language Pretraining for E-Commerce Recommendations" – Wang et al. (2022) 
#. Semantic Similarity Using LLMs 

	- "Semantic-Aware Item Matching with Large Language Models" – Chen et al. (2023) 
	- "Contextual Item Recommendation with Pretrained LLMs" – Li et al. (2022) 

User-Item Recommendation 
====================================================================================
- Homepage recommendations
- product recommendations
- videos you might like, etc

Key Concept 
------------------------------------------------------------------------------------
- User-item recommendation focuses on predicting a user's preference for an item based on historical interactions. This can be framed as 

	- Explicit feedback (e.g., ratings, thumbs up/down) 
	- Implicit feedback (e.g., clicks, watch time, purchases) 
- Common Approaches

	- Collaborative Filtering (CF) (Matrix Factorization, Neural CF) 
	- Content-Based Filtering (Feature-based models) 
	- Hybrid Models (Combining CF and content-based methods) 
	- Deep Learning Approaches (Neural networks, Transformers) 
- LLM Applications

	- LLMs enhance this by learning richer user and item embeddings, capturing nuanced interactions. 
	- LLMs can generate user preferences dynamically via zero-shot/few-shot learning, improving personalization. 
	- Example use case Personalized product descriptions, interactive recommendation assistants. 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label binary (clicked/not clicked, purchased/not purchased) or continuous (watch time, rating). 
	- Data sources user interactions, purchase logs, watch history. 
	- Challenges Class imbalance (many more non-clicked items than clicked ones). 
#. Semi-Supervised Learning 

	- Use self-training (pseudo-labeling) to expand labeled data. 
	- Graph-based methods to propagate labels across similar users/items. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., SimCLR, BERT-style masked item prediction). 
	- Learning representations via session-based modeling (e.g., predicting the next item a user interacts with). 

Common Features
------------------------------------------------------------------------------------
- User Features Past interactions, demographics, engagement signals. 
- Item Features Category, text/image embeddings, historical engagement. 
- Cross Features User-item interactions (e.g., user’s affinity to a category). 
- Contextual Features Time of day, device, location. 
- Embedding-based Features Learned latent factors from models like Word2Vec for items/users. 

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering 

	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009) 
	- "Neural Collaborative Filtering" – He et al. (2017) 
#. Deep Learning Approaches 

	- "Deep Neural Networks for YouTube Recommendations" – Covington et al. (2016) 
	- "Wide & Deep Learning for Recommender Systems" – Cheng et al. (2016) 
#. Hybrid and Production Systems 

	- "Netflix Recommendations Beyond the 5 Stars" – Gomez-Uribe et al. (2015) 
#. Transformer-Based RecSys 

	- "BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019) 
	- "SASRec Self-Attentive Sequential Recommendation" – Kang & McAuley (2018) 
#. LLM-powered Recommendation 

	- "GPT4Rec A Generative Framework for Personalized Recommendation" – Wang et al. (2023) 
	- "LLM-based Collaborative Filtering Enhancing Recommendations with Large Language Models" – Liu et al. (2023) 

Session-Based Recommendation 
====================================================================================
- Personalized recommendations based on recent user actions
- short-term intent modeling
- sequential recommendations

Key Concept 
------------------------------------------------------------------------------------
- Session-based recommendation focuses on predicting the next relevant item for a user based on their recent interactions, rather than long-term historical data. This is useful when 

	- Users don’t have extensive histories (e.g., guest users). 
	- Preferences shift dynamically (e.g., browsing sessions in e-commerce). 
	- Recent behavior is more indicative of intent than long-term history. 
- Common Approaches

	- Rule-Based Methods (Most popular, trending, or recently viewed items) 
	- Markov Chains & Sequential Models (Predicting next item based on state transitions) 
	- Recurrent Neural Networks (RNNs, GRUs, LSTMs) (Capturing sequential dependencies) 
	- Graph-Based Approaches (Session-based Graph Neural Networks) 
	- Transformer-Based Models (Attention-based architectures for session modeling) 
- LLM Applications

	- Traditional methods use sequential models (RNNs, GRUs, Transformers) to predict next-item interactions. 
	- LLMs enhance session modeling by leveraging sequential reasoning and contextual awareness. 
	- Few-shot prompting allows LLMs to infer session preferences without explicit training. 
	- Example use case Dynamic content feeds (TikTok), real-time recommendations (Spotify session playlists). 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Next item in sequence (e.g., clicked/purchased item). 
	- Data sources User sessions, browsing logs, cart abandonment data. 
	- Challenges Short sessions make training harder; sparse interaction data. 
#. Semi-Supervised Learning 

	- Use self-supervised tasks like predicting masked interactions. 
	- Graph-based node propagation to learn session similarities. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., predict next item from different user sessions). 
	- Next-click prediction using masked sequence modeling (BERT-style). 

Common Features
------------------------------------------------------------------------------------
- Session Features Time spent, number of items viewed, recency of last interaction. 
- Item Features Product category, textual embeddings, popularity trends. 
- Sequence Features Click sequences, time gaps between interactions. 
- Contextual Features Device type, time of day, geographical location. 
- Embedding-Based Features Pretrained session embeddings (e.g., Word2Vec-like for items). 

Resources
------------------------------------------------------------------------------------
#. Traditional Approaches & Sequential Models 

	- "Session-Based Recommendations with Recurrent Neural Networks" – Hidasi et al. (2016) 
	- "Neural Architecture for Session-Based Recommendations" – Tang & Wang (2018) 
#. Graph-Based Methods 

	- "Session-Based Recommendation with Graph Neural Networks" – Wu et al. (2019) 
	- "Next Item Recommendation with Self-Attention" – Sun et al. (2019) 
#. Transformer-Based Methods 

	- "SASRec Self-Attentive Sequential Recommendation" – Kang & McAuley (2018) 
	- "BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019) 
#. LLM-Driven Dynamic Recommendation 

	- "LLM-Powered Dynamic Personalized Recommendations" – Guo et al. (2023) 
	- "Next-Item Prediction Using Pretrained Language Models" – Sun et al. (2021) 
	- "Real-Time Recommendation with Large Language Models" – Zhang et al. (2023) 

User-User Recommendation 
====================================================================================
- People You May Know
- Friend Suggestions
- Follower Recommendations

Key Concept 
------------------------------------------------------------------------------------
- User-user recommendation focuses on predicting connections between users based on their behavior, interests, or existing social networks.

	#. Typically modeled as a link prediction problem in graphs. 
	#. Used for social networks, professional connections, or matchmaking systems. 
- Common Approaches

	#. Collaborative Filtering (User-Based CF) 
	#. Graph-Based Approaches (Graph Neural Networks, PageRank, Node2Vec, etc.) 
	#. Feature-Based Matching (Demographic and behavior similarity) 
	#. Hybrid Approaches (Graph + CF + Deep Learning) 
- LLM Applications

	- Typically modeled as a graph-based link prediction problem, where users are nodes. 
	- LLMs can enhance user similarity computations by processing richer profile texts (e.g., bios, chat history). 
	- Social connections can be inferred by analyzing natural language data, rather than relying solely on structural graph features. 
	- Example use case Professional networking (LinkedIn), AI-assisted friend suggestions. 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (1 = connection exists, 0 = no connection). 
	- Data sources Friendship graphs, follow/unfollow actions, mutual interests. 
	- Challenges Highly imbalanced data (most user pairs are not connected). 

#. Semi-Supervised Learning 

	- Graph-based label propagation (e.g., predicting missing edges in a user graph). 
	- Use unlabeled users with weak supervision from social structures. 

#. Self-Supervised Learning 

	- Contrastive learning (e.g., learning embeddings where connected users are closer in vector space). 
	- Masked edge prediction (e.g., hide some connections and train the model to reconstruct them). 

Common Features
------------------------------------------------------------------------------------
- User Features Profile attributes (age, location, industry, interests). 
- Graph Features Common neighbors, Jaccard similarity, Adamic-Adar score. 
- Interaction Features Message frequency, engagement level. 
- Embedding-Based Features Node2Vec or GNN-based embeddings. 
- Contextual Features Activity time, shared communities.

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches 

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001) 
	- "A Guide to Neural Collaborative Filtering" – He et al. (2017) 
#. Graph-Based Approaches 

	- "DeepWalk Online Learning of Social Representations" – Perozzi et al. (2014) 
	- "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018) 
	- "Graph Neural Networks A Review of Methods and Applications" – Wu et al. (2021) 
#. Hybrid and Large-Scale User-User Recommendation 

	- "Link Prediction Approaches and Applications" – Liben-Nowell et al. (2007) 
	- "Who to Follow Recommending People in Social Networks" – Twitter Research (2010) 
#. Graph-Based LLMs 

	- "Graph Neural Networks Meet Large Language Models A Survey" – Wu et al. (2023) 
	- "LLM-powered Social Graph Completion for Friend Recommendations" – Huang et al. (2023) 
#. Hybrid Graph and LLMs 

	- "LLM-Augmented Node Classification in Social Networks" – Zhang et al. (2023) 
	- "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018)  

************************************************************************************
Deep Dives
************************************************************************************
Personalisation
====================================================================================
Diversity
====================================================================================
.. important::
	- Music & video platforms (Spotify, YouTube, TikTok) use DPP and Bandits to introduce diverse content.
	- E-commerce (Amazon, Etsy) balances popularity-based downsampling with weighted re-ranking.
	- Newsfeeds (Google News, Facebook, Twitter) use category-sensitive filtering to prevent echo chambers.

- Goal

	- improving user engagement
	- avoiding filter bubbles
	- preventing over-reliance on popular content.
- Metric

	- TODO

- LLMs for Diversity in Recommendations

	.. note::	
		- YouTube - Uses LLMs for multi-modal retrieval (text, video, audio). 
		- Spotify - Uses LLMs for playlist diversification and exploration-based re-ranking. 
		- Netflix - Uses GPT-like models for diverse genre-based recommendations. 
		- Google Search & News - Uses BERT-based fairness filters for diverse search results. 

- Technique Summary

	.. csv-table::
		:header: "Technique", "Stage", "Pros", "Cons"
		:align: center

			Multi-Pool Retrieval, Retrieval, High diversity; multiple candidate sources, Computationally expensive
			Popularity-Based Downsampling, Retrieval, Prevents over-recommendation of trending items, May reduce engagement
			Minimum-Item Representation Heuristics, Filtering, Ensures fairness across categories, Might reduce personalization
			Category-Sensitive Filtering, Filtering, Adapts to user preferences dynamically, High computation cost
			Determinantal Point Processes (DPP), Re-Ranking, Mathematical diversity control, Computationally expensive
			Re-Ranking with Diversity Constraints, Re-Ranking, Tunable for personalization vs. diversity, Requires careful tuning
			Multi-Armed Bandits, Re-Ranking, Balances personalization and exploration, Hard to tune in real-world scenarios

- LLMs for Diversity at Each Stage 

	.. csv-table::
		:header: "Stage", "LLM Enhancements", "Pros", "Cons"
		:align: center

			Retrieval, Query expansion; Multi-modal retrieval, Increases recall & heterogeneity, Higher latency; Loss of precision
			Filtering & Merging, Semantic deduplication; Bias correction, Prevents redundancy; Fairer recommendations, Computationally expensive
			Re-Ranking, Diversity-aware reranking; Counterfactuals, Balances personalization & exploration, Risk of over-exploration; Expensive inference

Retrieval Stage
------------------------------------------------------------------------------------
.. note::
	Goal Ensuring Diversity in Candidate Selection

Multi-Pool Retrieval (Heterogeneous Candidate Selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Retrieves candidates from multiple independent sources (e.g., popularity-based pool, collaborative filtering pool, content-based retrieval).
	- Ensures that recommendations are not solely based on one dominant factor (e.g., trending items).

Pros:

	- Increases coverage by considering multiple types of items.
	- Helps balance long-term preferences vs. short-term interest.

Cons:

	- If not weighted properly, can introduce irrelevant or low-quality recommendations.
	- Computationally expensive when handling large numbers of pools.

Example:

	- YouTube retrieves candidates from watched videos, partially watched videos, new uploads, and popular in demographic to balance diversity.

Popularity-Based Downsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Reduces the dominance of highly popular items in the candidate pool.
	- Ensures niche items have a fair chance of being retrieved.

Pros:

	- Prevents "rich-get-richer" feedback loops.
	- Encourages long-tail item discovery.

Cons:

	- Might hurt immediate engagement metrics (CTR, Watch Time).
	- New users may still prefer popular items over niche ones.

Example:

	- Spotifys Discover Weekly uses a mix of popular and long-tail recommendations to balance engagement and discovery.

LLMs for Diverse Candidate Selection 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Query Expansion for Better Recall 

		- LLMs generate query variations to retrieve diverse candidates beyond exact keyword matching. 
		- Example Instead of just retrieving laptops, LLMs expand queries to include notebooks, MacBooks, ultrabooks. 
		- Technique Use T5/BERT-based semantic expansion to increase retrieval diversity. 
	
	#. Multi-Modal Understanding for Heterogeneous Retrieval 

		- LLMs bridge different modalities (text, image, video) to retrieve richer candidate pools. 
		- Example In YouTube Recommendations, an LLM can link a users watched TED Talk to blog articles on the same topic. 
		- Technique Use CLIP (for text-image-video embeddings) to retrieve across modalities. 

	#. User Preference Understanding for Contextual Retrieval 

		- Instead of static retrieval models, LLMs generate dynamic search queries based on user conversation history. 
		- Example A user searching for travel backpacks may also receive recommendations for hiking gear if LLMs infer the intent. 
		- Technique Use GPT-like models to rewrite user queries dynamically based on session context. 

Pros 

	- Improves Recall - LLMs retrieve more diverse content that traditional CF models miss. 
	- Better Cold-Start Handling - Generates synthetic preferences for new users. 

Cons 

	- High Latency - Generating queries dynamically can be slower than precomputed embeddings. 
	- Loss of Precision - More diverse candidates mean a higher risk of retrieving irrelevant results. 

Filtering & Merging Stage
------------------------------------------------------------------------------------
.. note::
	Goal Balancing Diversity Before Re-Ranking

Minimum-Item Representation Heuristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Ensures that each category, genre, or provider has a minimum number of candidates before merging.
	- Helps prevent over-representation of any single category.

Pros:

	- Easy to implement with rule-based heuristics.
	- Ensures fairness in content exposure.

Cons:

	- Can sacrifice relevance by forcing underrepresented items.
	- Hard to scale for fine-grained personalization.

Example:

	- News Feeds (Facebook, Twitter, Google News) ensure a minimum number of international vs. local news, avoiding content silos.

Category-Sensitive Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Computes category entropy to measure diversity across different categories.
	- If a users recommendations lack category diversity, it enforces rebalancing by boosting underrepresented categories.

Pros:

	- Dynamically adapts to different users.
	- Can be optimized for long-term user retention.

Cons:

	- Requires real-time category tracking, which can be computationally expensive.
	- Poor tuning may result in irrelevant recommendations.

Example:

	- Netflix ensures that recommendations contain a mix of different genres rather than overloading one.

LLMs for Diversity-Aware Candidate Selection 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Semantic Deduplication & Cluster Merging 

		- LLMs identify semantically similar items (even if they differ in wording) to prevent redundancy. 
		- Example In news recommendations, LLMs group articles covering the same event to avoid repetition. 
		- Technique Use sentence embeddings (SBERT) to cluster semantically duplicate items. 

	#. Bias & Fairness Control 

		- LLMs detect biased patterns (e.g., over-representing a certain demographic) and adjust recommendations accordingly. 
		- Example A job recommendation system might over-recommend tech jobs to menLLMs can balance exposure. 
		- Technique Use LLM-based fairness models (e.g., DebiasBERT) to adjust recommendations. 

	#. Context-Aware Filtering 

		- LLMs generate filtering rules on-the-fly based on user profile, session history, or external trends. 
		- Example If a user browses vegetarian recipes, LLMs downrank meat-based recipes dynamically. 
		- Technique Use GPT-powered filtering prompts to dynamically adjust content selection. 

Pros 

	- Prevents Repetitive Recommendations - Ensures users dont see redundant items. 
	- Improves Fairness & Representation - Adjusts for bias in candidate selection. 

Cons 

	- Computationally Expensive - Filtering millions of candidates using LLMs can increase inference costs. 
	- Difficult to Fine-Tune - Over-filtering may hide relevant recommendations. 

Re-Ranking Stage
------------------------------------------------------------------------------------
.. note::
	Goal Final Diversity Adjustments

Determinantal Point Processes (DPP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Uses probabilistic modeling to diversify ranked lists.
	- Given a candidate set, DPP selects a subset that maximizes diversity while maintaining relevance.
	- Works by modeling similarity between items and ensuring that similar items are not ranked too closely together.

Pros:

	- Mathematically principled and ensures diversity without arbitrary rules.
	- Used successfully in Spotify and Amazon for playlist & product recommendations.

Cons:

	- Computationally expensive, especially in large-scale deployments.
	- Needs proper similarity functions to be effective.

Example:

	- Spotify Playlist Generation - Ensures a playlist has a variety of artists and genres instead of only one type of song.

Re-Ranking with Diversity Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Uses weighted re-ranking algorithms that explicitly penalize redundant recommendations.
	- Can be tuned to balance diversity vs. personalization dynamically.

Pros:

	- Adjustable trade-off between diversity and user preferences.
	- Works well for personalized recommendations.

Cons:

	- Needs constant tuning to find the right balance.
	- If misconfigured, can make recommendations feel random or irrelevant.

Example:

	- YouTubes Ranking Model applies re-ranking constraints to prevent over-recommendation of a single creator in a session.

Multi-Armed Bandits for Explore-Exploit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Balances exploitation (showing relevant, known content) with exploration (introducing new, diverse content).
	- Upper Confidence Bound (UCB), Thompson Sampling are commonly used bandit techniques.

Pros:

	- Encourages personalized discovery while ensuring exploration.
	- Automatically adapts over time.

Cons:

	- Hard to tune exploration parameters in production settings.
	- May result in temporary engagement drops during exploration phases.

Example:

	- TikToks For You Page mixes known preferences with new content using bandit-based ranking.

LLMs for Diversity-Aware Ranking 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Diversity-Aware Ranking Models 

		- LLMs act as personalization-aware rerankers, balancing relevance with diversity dynamically. 
		- Example Instead of showing only Marvel movies to a fan, LLMs inject DC movies or indie superhero films. 
		- Technique Use LLM-powered diversity re-ranking prompts in post-processing. 

	#. Personalized Exploration vs. Exploitation 

		- LLMs simulate user preferences in real-time and adjust ranking to include more exploration. 
		- Example In TikTok, if a user likes cooking videos, LLMs inject some fitness or travel videos to encourage exploration. 
		- Technique Use GPT-powered bandit re-ranking for adaptive diversity balancing. 

	#. Diversity-Aware Re-Ranking via Counterfactual Predictions 

		- LLMs generate counterfactual recommendations to test how users might respond to different recommendation lists. 
		- Example Instead of showing only trending news, LLMs inject underrepresented topics and measure user responses. 
		- Technique Use LLMs for offline counterfactual testing before deployment. 

Pros 

	- Balances Personalization & Diversity - Prevents filter bubbles. 
	- Improves Long-Term Engagement - Users are less likely to get bored. 

Cons 

	- Higher Inference Cost - Re-ranking every session in real-time increases server load. 
	- Risk of Over-Exploration - If diversity is forced, users may feel the system is less relevant.

Distribution Shift
====================================================================================
Identification
------------------------------------------------------------------------------------
Refer to Observability page

Addressal
------------------------------------------------------------------------------------
(A) Continuous Model Updating & Online Learning 

	- Solution Train fresh models on recent data to ensure up-to-date recommendations. 
	- Trade-Offs 

		- Frequent retraining is computationally expensive. 
		- Requires robust online learning pipelines (feature stores, incremental updates). 

Example 

	- Google Search updates its ranking models regularly to adapt to evolving search trends. 
	- Spotify retrains user embeddings frequently to reflect shifting music preferences. 

(B) Adaptive Sampling & Reweighting Older Data 

	- Solution Weight recent data more heavily while retaining historical knowledge for long-term trends. 
	- Trade-Offs 

		- Overweighting recent data may cause catastrophic forgetting of long-term preferences. 
		- Requires tuning of decay rates (e.g., exponential decay). 

Example 

	- E-Commerce platforms (Amazon, Walmart) use time-decayed embeddings to keep recommendations fresh. 

(C) Real-Time Personalization Using Session-Based Models 

	- Solution Use short-term session-based models (Transformers, RNNs) that adapt to recent interactions. 
	- Trade-Offs 

		- Session models work well short-term but lack long-term personalization. 
		- Requires fast inference pipelines (low latency). 

Example 

	- TikToks recommender adapts within a session, adjusting based on user behavior in real-time. 

(D) Reinforcement Learning for Adaptive Ranking 

	- Solution Use reinforcement learning (RL) models to dynamically adapt rankings based on user feedback. 
	- Trade-Offs 

		- RL models require a lot of data to converge. 
		- Training RL models online is computationally expensive. 

Example 

	- YouTubes ranking system adapts via reinforcement learning to balance freshness & engagement. 

(E) Hybrid Ensembles (Mixing Old & New Models) 

	- Solution Use an ensemble of multiple models trained on different time periods, allowing a blend of fresh & historical preferences. 
	- Trade-Offs 

		- Combining models increases complexity. 
		- Requires ensemble weighting tuning to balance long-term vs. short-term data. 

Example 

		- Netflix blends long-term preference models with session-based recommendations. 
