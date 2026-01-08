####################################################################################
Issues in Search & Recommendation Systems
####################################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

- Distribution Shift

	.. csv-table::
		:header: "Problem", "How to Detect", "How to Fix", "Trade-Offs"
		:align: center 
	
			Model Degradation, Performance drop (CTR; engagement), Frequent model retraining, Computationally expensive
			Popularity Mismatch, PSI; JSD; embeddings drift, Adaptive reweighting of historical data, Hard to balance long vs. short-term relevance
			Bias Reinforcement, Disparity in exposure metrics, Fairness-aware ranking, May hurt engagement
			Cold-Start for New Trends, Increase in unseen queries, Session-based personalization, Requires fast inference
			Intent Drift in Search, Increase in irrelevant search rankings, Online learning models, Real-time training is costly
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

************************************************************************************
General Issues
************************************************************************************
Cold-Start Problem (Users & Items) 
====================================================================================
- Why It Matters 

	- New users No interaction history makes personalization difficult. 
	- New items Struggle to get exposure due to lack of engagement signals. 

- Strategic Solutions & Trade-Offs 

	- Content-Based Methods (Text embeddings, Image/Video features) -> Good for new items, but lacks user personalization. 
	- Demographic-Based Recommendations (Cluster similar users) -> Generalizes well but risks oversimplification. 
	- Randomized Exploration (Show new items randomly) -> Increases fairness but can reduce CTR. 

- Domain-Specific Notes 

	- E-commerce (Amazon, Etsy) -> Cold-start for new sellers & niche products. 
	- Video Streaming (Netflix, YouTube) -> Cold-start for newly released content. 

Popularity Bias & Feedback Loops
====================================================================================
- Why It Matters 

	- Over-recommending already popular items creates a "rich-get-richer" effect affecting fairness, novelty.
	- Reinforces biases in user engagement, making it harder to surface niche or novel content.

- Common Approaches:

	- Changing objective

		- ReGularization (RG)

			- [depaul.edu] `Controlling Popularity Bias in Learning to Rank Recommendation <https://scds.cdm.depaul.edu/wp-content/uploads/2017/05/SOCRS_2017_paper_5.pdf>`_
			- Controls the ratio of popular and less popular items via a regularizer added to the objective function
			- Penalizes lists that contain only one group of items and hence attempting to reduce the concentration on popular items
		- Discrepancy Minimization (DM)

			- [cmu.edu] `Post Processing Recommender Systems for Diversity <https://www.contrib.andrew.cmu.edu/~ravi/kdd17.pdf>`_
			- Optimizes for aggregate diversity
			- Define a target distribution of item exposure as a constraint for the objective function
			- Goal is therefore to minimize the discrepancy of the recommendation frequency for each item and the target distribution
		- FA*IR (FS)

			- [arxiv.org] `FA*IR A Fair Top-k Ranking Algorithm <https://arxiv.org/abs/1706.06368>`_
			- Creates queues of protected (long-tail) and unprotected (head) items so that protected items get more exposure
		- Personalized Long-tail Promotion (XQ)

			- [arxiv.org] `Managing Popularity Bias in Recommender Systems with Personalized Re-ranking <https://arxiv.org/abs/1901.07555>`_
			- Query result diversification
			 -The objective for a final recommendation list is a balanced ratio of popular and less popular (long-tail) items.
		- Calibrated Popularity (CP)

			- [arxiv.org] `User-centered Evaluation of Popularity Bias in Recommender Systems - Abdollahpouri et. al <https://arxiv.org/pdf/2103.06364>`_
			- Takes user's affinity towards popular, diverse and niche contents into account
	- Randomisation

		- Contextual Bandits
	- Position debiasing
- Domain-Specific Notes:

	- Social Media (TikTok, Twitter, Facebook) Celebrity overexposure (e.g., verified users dominating feeds). 
	- News Aggregators (Google News, Apple News) Same sources getting recommended (e.g., mainstream news over independent journalism). 

Diversity vs. Personalization Trade-Off 
====================================================================================
- Resources:

	- [engineering.fb.com] `On the value of diversified recommendations <https://engineering.fb.com/2020/12/17/ml-applications/diversified-recommendations/>`_
- Why It Matters:

	- Highly personalized feeds reinforce user preferences, limiting exposure to new content.
	- Leads to boredom of users in long-term which might reduce retention rate.
	- Users may get stuck in content silos (e.g., political polarization, filter bubbles).

- Understanding the issue:
	
	- Theoretical framework
	
		- Personalization

			- Polya process
			- self reinforcement
			- pros short term gains
			- cons leads to boredom and retention
		- Balancing

			- balancing process
			- Negative reinforcement
			- Pros doesn't lead to boredom
			- Cons affects short term gains
	- Complexities in real world personal preferences

		- Multidimensional (dark comedy = dark thriller + general comedy)
		- Soft (30% affinity towards comedy, 90% affinity towards sports)
		- Contextual (mood, time of day, current trends)
		- Dynamic (evolves over time)

- Heuristics on diversifying recommendation:

	- Author level diversity -> strafification -> pick candidates from different authors
	- Media type diversity -> applicable for multimedia platforms -> intermix modality
	- Semantic diversity -> content understanding system -> classify user's affinity to topics -> sample across topics
	- Explore similar semantic nodes -> knowledge tree/graph

		- Explore parents, siblings, children of topics
		- Explore long tail for niche topics
		- Explore items that covers multiple topics
	- Maintain separate pool for short-term and long-term preferences
	- Utilize explore-exploit framework -> eps-greedy, ucb, thompson sampling
	- Prioritize behavioural metrics as much as accuracy metrics
	- Priotitize explicit negative feedbacks from users

- Strategic Solutions & Trade-Offs 

	- Diversity-Promoting Re-Ranking (DPP, Exploration Buffers) -> Reduces filter bubbles but may decrease engagement. 
	- Diversity-Constrained Search (Re-weighting ranking models) -> Promotes varied content but risks reducing precision. 
	- Hybrid User-Item Graphs (Graph Neural Networks for diversification) -> Balances exploration but requires expensive training. 

- Domain-Specific Notes 

	- Social Media (Facebook, Twitter, YouTube) -> Political echo chambers & misinformation bubbles. 
	- E-commerce (Amazon, Etsy, Zalando) -> Users seeing only one type of product repeatedly.

Short-Term Engagement vs. Long-Term User Retention 
====================================================================================
- Why It Matters 

	- Systems often optimize for immediate engagement (CTR, watch time, purchases), which can lead to addictive behaviors or content fatigue.
	- Over-exploitation of "sticky content" (clickbait, sensationalism, autoplay loops) may reduce long-term satisfaction.

- Strategic Solutions & Trade-Offs:

	- Multi-Objective Optimization (CTR + Long-Term Retention) -> Complex to balance but essential for sustainability.
	- Delayed Reward Models (Reinforcement Learning) -> Great for long-term user retention but slow learning process.
	- Personalization Decay (Balancing Freshness vs. Relevance) -> Introduces diverse content but can feel random to users.

- Domain-Specific Notes:

	- YouTube, TikTok, Instagram -> Prioritizing sensational viral content over educational material.
	- E-Commerce (Amazon, Alibaba) -> Short-term discounts vs. long-term brand loyalty.

Real-Time Personalization & Latency Trade-Offs 
====================================================================================
- Why It Matters 

	- Personalized recommendations require real-time feature updates and low-latency inference. 
	- Search relevance depends on immediate context (e.g., location, time of day, trending topics). 

- Strategic Solutions & Trade-Offs 

	- Precomputed User Embeddings (FAISS, HNSW, Vector DBs) -> Speeds up search but sacrifices personalization flexibility. 
	- Edge AI for On-Device Personalization -> Reduces latency but increases computational costs. 
	- Session-Based Recommendation Models (Transformers for Session-Based Context) -> Great for short-term personalization but expensive for large user bases. 

- Domain-Specific Notes 

	- E-Commerce (Amazon, Walmart, Shopee) -> Latency constraints for similar item recommendations. 
	- Search Engines (Google, Bing, Baidu) -> Needing real-time personalization without slowing down results. 

************************************************************************************
Domain-Specific
************************************************************************************
Search
==================================================================================== 
- Query Understanding & Intent Disambiguation

	- Users enter ambiguous or vague queries, requiring intent inference. 
	- Example Searching for “apple” – Is it a fruit, a company, or a music service? 
	- Solutions & Trade-Offs 

		- LLM-Powered Query Rewriting (T5, GPT) -> Improves relevance but risks over-modifying queries. 
		- Session-Aware Query Expansion -> Helps disambiguation but increases computational cost. 

E-Commerce
====================================================================================
- Balancing Revenue & User Satisfaction 

	- Revenue-driven recommendations (sponsored ads, promoted products) vs. organic recommendations. 
	- Example Amazon mixing sponsored and personalized search results. 
	- Solutions & Trade-Offs 

		- Hybrid Models (Re-ranking with Fairness Constraints) -> Balances organic vs. paid but hard to tune for revenue goals. 
		- Trust-Based Ranking (Reducing deceptive sellers, fake reviews) -> Improves satisfaction but may lower short-term sales. 

Video & Music Streaming
====================================================================================
- Content-Length Bias in Recommendations 

	- Recommendation models often favor shorter videos (TikTok, YouTube Shorts) over long-form content. 
	- Example YouTube's watch-time optimization may prioritize clickbaity short videos over educational content. 
	- Solutions & Trade-Offs 

		- Normalized Engagement Metrics (Watch Percentage vs. Watch Time) -> Improves long-form content exposure but may reduce video diversity. 
		- Hybrid-Length Recommendations (Mixing Shorts & Full Videos) -> Enhances variety but harder to rank effectively.
