#############################################################
Problems in Search & RecSys
#############################################################
*************************************************************
Scenario Classification
*************************************************************
1. Homepage Recommendation (Feed/Personalized Front Page):

	- Context: When a user lands on a platform, the system must quickly serve a personalized selection of items (articles, products, posts, videos) from a large pool.  
	- Goals: Balance long-term user interests with fresh content; ensure diversity; and achieve low latency.

2. Next Item Recommendation (Sequential/Session-Based):  

	- Context: Given a user's recent interactions (e.g., clicks in a session), predict the immediate next item they are likely to engage with.  
	- Goals: Capture short-term context and evolving interests, while possibly integrating long-term preferences.

3. Search (Query-Based):  

	- Context: When a user issues an explicit query (text, voice, or image), the system must retrieve the most relevant items and rank them appropriately.  
	- Goals: Understand query intent, balance lexical matching with semantic understanding, and deliver highly relevant results with efficient ranking.

4. Suggested Playlist Curation (Curated Collections):  

	- Context: For domains like music or video streaming, the system must generate a cohesive, diverse, and context-aware playlist that aligns with both the user's historical tastes and current mood or context.  
	- Goals: Blend long-term preferences with short-term mood or situational signals; foster serendipity and diversity; maintain smooth transitions between items.

5. Other Domain-Specific Scenarios (e.g., Friend/Content Discovery in Social Networks, Product Recommendations in E-Commerce):  

	- While these can often be seen as variations of the above scenarios, they might emphasize factors such as social connections or detailed item attributes.

*************************************************************
Modeling Approaches & Trade-offs
*************************************************************
1. Homepage Recommendation
=============================================================
- Collaborative Filtering (CF) / Neural CF (e.g., Matrix Factorization, Neural Collaborative Filtering):

	- Pros:
	
		- Excels at capturing long-term, aggregated user interests from historical interactions.  
		- Can scale efficiently using approximate nearest neighbor (ANN) search techniques.  
	- Cons:
	
		- May suffer from cold-start issues (new users/items) and risk creating "filter bubbles" by overemphasizing past preferences.
	- Scale:
	
		- Typically deployed at scales of millions of users and items.
	- Justification:
	
		- A strong baseline for personalization; can be augmented with side information to improve diversity.
	- Real-World Example:
	
		- Netflix uses MF-based approaches combined with neural models for its homepage recommendations.

- Content-Based Filtering (CBF):
	- Pros:
	
		- Handles cold-start for new items by relying on item features (text, image, etc.).  
	- Cons:
	
		- May narrow the focus to items very similar to what the user already saw, reducing diversity.
	- Scale:
	
		- Requires efficient feature extraction pipelines; often works well when combined with vector search engines.
	- Justification:
	
		- Useful when rich item metadata is available.
	- Real-World Example:
	
		- Google News uses BERT-based content matching alongside collaborative signals.
	
- Hybrid Models:

	- Pros:
	
		- Leverage both CF and CBF strengths; can provide a good mix of familiar and novel content.  
	- Cons:
	
		- Increased complexity and higher computational cost.
	- Scale:
	
		- Can be applied at large scale with proper infrastructure (distributed computing, caching).  
	- Justification:
	
		- Balances personalization and exploration, ensuring a diverse homepage feed.
	- Real-World Example:
	
		- Amazon's homepage recommendation system combines collaborative and content features.

2. Next Item Recommendation
=============================================================
- Sequence-Based Models (RNNs, Transformers like SASRec, Transformer4Rec):

	- Pros:
 
		- Excellent for capturing short-term session dynamics and sequential patterns in user behavior.  
		- Can adjust quickly to context changes.
	- Cons:  
 
		- May underrepresent long-term stable interests unless combined with long-term signals.
		- Typically more complex and computationally demanding.
	- Scale:  
	
 		- Effective for session-level data; usually operates on a subset of data per user session.
	- Justification:  
	
 		- Tailored to real-time or near-real-time prediction of the next interaction.
	- Real-World Example:  
	
 		- YouTube uses sequential models to predict the next video on the "Up Next" list.

- Item2Vec / CBOW Models:  
	- Pros:  
 
		- Efficiently capture co-occurrence patterns from user sessions.  
		- Ideal for fast retrieval from a large catalog.
	- Cons:  
 
		- Generally provide embeddings optimized for retrieval rather than fine-grained ranking.
	- Scale:  
 
		- Can operate at scales of tens of millions of items with fast ANN search methods.
	- Justification:  
 
		- Provides a lightweight mechanism for session-level recommendations.
	- Real-World Example:  
 
		- TikTok's "For You" recommendations leverage such co-occurrence signals.

- Hybrid Sequence Models:  
	- Pros:  
 
		- Combine sequential modeling with long-term user profiles, often by integrating collaborative signals.  
	- Cons:  
 
		- Increased complexity; need to balance short-term dynamics with long-term stability.
	- Scale:  
 
		- Deployed on platforms with millions of daily active users, using distributed training.
	- Justification:  
 
		- Offers a more comprehensive view of user behavior.
	- Real-World Example:  
 
		- Spotify's recommendation system blends session-based signals with overall user preferences.

3. Search (Query-Based Recommendation)
=============================================================
- Learning-to-Rank Models (e.g., using Gradient Boosted Decision Trees or Neural Ranking Models like BERT-based re-rankers):  

	- Pros:  
 
		- Capable of integrating both lexical and semantic features; high precision in ranking search results.  
	- Cons:  
 
		- Neural re-rankers (like BERT) are computationally expensive, impacting latency.
	- Scale:  
 
		- Effective for web-scale search with billions of pages when used in a two-stage (candidate generation + re-ranking) setup.
	- Justification:  
 
		- Balances retrieval efficiency with relevance and semantic understanding.
	- Real-World Example:  
 
		- Google Search uses BM25 for initial retrieval, followed by BERT-based re-ranking.  
		- [Google's Neural Ranking paper](https://arxiv.org/abs/1904.01766)

- Vector Space Models (e.g., using pre-computed embeddings and ANN search):  

	- Pros:  
 
		- Scales efficiently for massive document collections; captures semantic similarities.  
	- Cons:  
 
		- May require periodic updates to embeddings to capture evolving content.
	- Scale:  
 
		- Scalable to billions of documents with approximate nearest neighbor libraries like FAISS.
	- Justification:  
 
		- Provides efficient and semantically rich retrieval.
	- Real-World Example:  
 
		- Google's embedding-based search methods used in voice and image search.

4. Suggested Playlist Curation
=============================================================
- Hybrid Models Combining Collaborative Filtering & Content-Based Approaches:  

	- Pros:  
 
		- Can balance user's long-term listening history (CF) with immediate context (sequence models) and incorporate audio/textual features for diversity.  
	- Cons:  
 
		- Complexity increases, and enforcing diversity may lower short-term CTR.  
	- Scale:  
 
		- Deployed on platforms with hundreds of millions of songs/users; often uses efficient retrieval (e.g., ANN) followed by re-ranking.
	- Justification:  
 
		- Needed to blend familiarity (long-term preferences) with novelty (short-term trends) while maintaining a coherent playlist flow.
	- Real-World Example:  
 
		- Spotify's "Discover Weekly" and "Daily Mix" are produced using hybrid models that merge collaborative signals with audio feature analysis (via CNNs or pretrained audio embeddings).  
		- [Spotify Engineering Blog](https://engineering.atspotify.com/)

- Sequence-Based and Reinforcement Learning Approaches:  

	- Pros:  
 
		- Can dynamically adjust the playlist order based on immediate user behavior and feedback.  
	- Cons:  
 
		- More difficult to balance between user satisfaction and diversity; increased latency in real-time updates.
	- Scale:  
 
		- Applied on streaming platforms with millions of active sessions; may use caching and periodic re-ranking for real-time performance.
	- Justification:  
 
		- Effective for continuously adapting playlists to changing user contexts.
	- Real-World Example:  
 
		- Apple Music's curated playlists often leverage sequence modeling and reinforcement learning signals.
