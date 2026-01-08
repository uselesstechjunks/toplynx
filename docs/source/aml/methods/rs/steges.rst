#########################################################################
Stages in Search & Recommendation Systems
#########################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

*************************************************************************
Item Embedding Pretraining
*************************************************************************
- Goal: 
	- Learn dense item embeddings useful for retrieval, search, and recommendation.
	- Build item index for search and homepage recsys
	- Cold-start robustness by using rich content + weak interaction signals
- Signals:

	- Item2Vec or CBOW-style pretraining using session data
	- Co-engagement graphs: Co-liked, co-favorited, co-contacted seller
	- Cross-platform metadata: seller network embeddings, profile stats
- Modalities & Features:

	- Text (title, description)
	- Image/video (thumbnail, listing video)
	- Metadata (category, tags, location)
	- Author features (demographics, group activity, reputation)
- Embedding Spaces:
	
	- Separate: one for retrieval, one for semantic search
	- Joint (with fine-tuning for task-specific objectives)

*************************************************************************
User Embedding for Retrieval
*************************************************************************
- Goal: Personalize retrieval using long-term user behavior
- Retrieval embeddings = similarity optimized
- Challenges: Sparse on-platform interactions
- Signals:

	- Watch/browse/contact history on other Meta surfaces for cross-platform activity mining: pages liked, groups joined
	- Facebook friends/groups/interests
	- Location, device, demographics
	- Seller interaction graphs (contacted, purchased from)
- Models:

	- DSSM-style dual tower models (user tower with user profile + behavior history + item tower)
	- Pretrained item embeddings reused from Step 5
	- InfoNCE with in-batch negatives

*************************************************************************
Ranking
*************************************************************************
- Goal: Rank retrieved items based on relevance, intent, quality, and engagement likelihood.
- Models:

	- Wide & Deep, DeepFM (crossed features)
	- BST (Behavioral Sequence Transformer)
	- Multitask models (CTR, message likelihood, time spent)
- Features:

	- User features: demographics, location, device, time, profile
	- User behavior: short-term session sequences (click/view/pause), long-term interest
	- Item features: embeddings, quality score, popularity, recency
	- Contextual signals: entry point, seasonality, intent cues
	- Seller: profile, reputation
- Embedding Reuse:

	- Ranking embeddings = intent + context optimized
	- Reuse item embeddings from pretraining as input features
	- Reuse user embeddings from retrieval model
	- Fine-tune during ranking task training
