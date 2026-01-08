########################################################################################
Search Engine
########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none
****************************************************************************************
Minimal Event Logs
****************************************************************************************
Problem: `This question from Leetcode discussion forum <https://leetcode.com/discuss/interview-experience/512591/google-machine-learning-engineer-bangalore-dec-2019-reject/911775>`_

.. note::
	- You are given user_id, timestamp, search_query and list of links clicked by user while doing google search.

		.. csv-table::
			:header: user_id, timestamp, search_query, list_of_links_clicked
			:align: center

				5844364346, 1355563265.81, 'hotels in london', [linkurl1; linkurl2; linkurl2]
				2657673352, 1355565575.36, 'flowers', [linkurl2]
				3686586523, 1355547455.81, 'insurance', []
	- Design a machine learning system which predicts which links to show based on search query.
	- Assumption: distinct user_ids: 5B, distinct linkurls: 30B, <TODO: others>
	- Business KPIs: HitRate@K (successful sessions/sessions), CTR, Total Clicks
========================================================================================
What Would I Have During Inference Time?
========================================================================================
- user_id, timestamp, search_query, [list of links]
========================================================================================
What To Predict?
========================================================================================
#. Pointwise Ranking using estimated CTR: KPI - link CTR

	.. csv-table::
		:header: user_id, timestamp, search_query, link, ctr
		:align: center

		58443665466, 1355563675.81, 'hotels in paris', linkurl1, 0.35
#. Ranked List of K Links: KPI - total clicks per search_query

	.. csv-table::
		:header: user_id, timestamp, search_query, ranked_links
		:align: center

		58443665466, 1355563675.81, 'hotels in paris', [linkurl1; linkurl4; linkurl6]
========================================================================================
What Matters?
========================================================================================
#. Statistics: 

	- Overall (popularity)
	- Latest (trending)
	- At a givem time (seasonality)
	- Last active (age)
#. Quantities of interest:

	- User's search activity, click propensity, time since last query/click
	- Query's popularity, click propensity, time since last searched/clicked
	- Link's popularity, click propensity, time since last click
	- User's affinity towards certain queries
	- User's affinity towards certain links
	- Link's click propensity with certain queries
	- User's activity at certain time, query's popularity at a certain time, links popularity at a certain time
========================================================================================
How Do I Design Features?
========================================================================================
Even without any semantics, we can cook a ton of features from just these 4 attributes in the log.

What Can I Measure?
----------------------------------------------------------------------------------------
- Transforms: 

	#. Direct numeric value:

		- Pros: Less number of model parameters
		- Cons: Same weight (importance) across the entire numeric range, multiple collision
	#. Bin counting: Since right skewed -> :math:`f(cnt)=\log(1+cnt)` -> binning -> one-hot encoding -> learned embedding

		- Pros: Different weights (importance) for different range
		- Cons: Cannot handle distribution change (learned weights no longer optimal), still collision, mostly zero

Numeric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	.. csv-table::
		:header: Numeric, Measure, Pivot
		:align: center
	
			time_since, active, <user_id; last search query>
			time_since, active, <user_id; last link clicked>
			time_since, active, <search_query; last searched>
			time_since, active, <search_query; last link clicked>
			time_since, active, <link; last clicked>

Counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	.. csv-table::
		:header: Count, Measure, Pivot
		:align: center
	
			query, activity, <user_id; all_time>; <user_id; since_last_year>; <user_id; since_last_month>; <user_id; since_last_week>; <user_id; since_last_day>; <user_id; since_last_hour>; <user_id; since_last_10_mins>
			query, seasonality, <user_id; month_of_year>; <user_id; day_of_month>; <user_id; day_of_week>; <user_id; time_of_day>
			click, activity, <user_id; all_time>; <user_id; since_last_year>; <user_id; since_last_month>; <user_id; since_last_week>; <user_id; since_last_day>; <user_id; since_last_hour>; <user_id; since_last_10_mins>
			click, seasonality, <user_id; month_of_year>; <user_id; day_of_month>; <user_id; day_of_week>; <user_id; time_of_day>
			user, popularity, <search_query; all_time>; <search_query; since_last_year>
			user, trending, <search_query; since_last_month>; <search_query; since_last_week>; <search_query; since_last_day>; <search_query; since_last_hour>; <search_query; since_last_10_mins>
			user, seasonality, <search_query; month_of_year>; <search_query; day_of_month>; <search_query; day_of_week>; <search_query; time_of_day>
			click, popularity, <search_query; all_time>; <search_query; since_last_year>
			click, trending, <search_query; since_last_month>; <search_query; since_last_week>; <search_query; since_last_day>; <search_query; since_last_hour>; <search_query; since_last_10_mins>
			click, seasonality, <search_query; month_of_year>; <search_query; day_of_month>; <search_query; day_of_week>; <search_query; time_of_day>
			user, popularity, <link; all_time>; <link; since_last_year>
			user, trending, <link; since_last_month>; <link; since_last_week>; <link; since_last_day>; <link; since_last_hour>; <link; since_last_10_mins>
			user, seasonality, <link; month_of_year>; <link; day_of_month>; <link; day_of_week>; <link; time_of_day>
			click, popularity, <link; all_time>; <link; since_last_year>
			click, trending, <link; since_last_month>; <link; since_last_week>; <link; since_last_day>; <link; since_last_hour>; <link; since_last_10_mins>
			click, seasonality, <link; month_of_year>; <link; day_of_month>; <link; day_of_week>; <link; time_of_day>
			click, popularity, <user_id; link; all_time>; <user_id; link; since_last_year>
			click, trending, <user_id; link; since_last_month>; <user_id; link; since_last_week>; <user_id; link; since_last_day>; <user_id; link; since_last_hour>; <user_id; link; since_last_10_mins>
			click, seasonality, <user_id; link; month_of_year>; <user_id; link; day_of_month>; <user_id; link; day_of_week>; <user_id; link; time_of_day>
			click, popularity, <user_id; search_query; all_time>; <user_id; search_query; since_last_year>
			click, trending, <user_id; search_query; since_last_month>; <user_id; search_query; since_last_week>; <user_id; search_query; since_last_day>; <user_id; search_query; since_last_hour>; <user_id; search_query; since_last_10_mins>
			click, seasonality, <user_id; search_query; month_of_year>; <user_id; search_query; day_of_month>; <user_id; search_query; day_of_week>; <user_id; search_query; time_of_day>
			click, popularity, <user_id; search_query; link; all_time>; <user_id; search_query; link; since_last_year>
			click, trending, <user_id; search_query; link; since_last_month>; <user_id; search_query; link; since_last_week>; <user_id; search_query; link; since_last_day>; <user_id; search_query; link; since_last_hour>; <user_id; search_query; link; since_last_10_mins>
			click, seasonality, <user_id; search_query; link; month_of_year>; <user_id; search_query; link; day_of_month>; <user_id; search_query; link; day_of_week>; <user_id; search_query; link; time_of_day>

.. note::
	Most of the counts will fall in the zero bucket, especially cross counting features

What Can I Extract?
----------------------------------------------------------------------------------------
#. We can extract these from the search_query and link attributes

	- search_query: normalized_search_query -> wordbreaker -> words
	- link: domain
#. We can a form a vocabulary of words with an out_of_vocabulary word for accomodating unseen words during inference time. Then we can construct

	- Bag of words features for query
	- Tf-idf features for query
	- Check `sklearn feature extractors <https://scikit-learn.org/stable/api/sklearn.feature_extraction.html>`_ for details.
	- Question: Can we form hypothetical documents associated with links by merging words from all queries that resulted in a click?

.. note::
	- We can estimate out_of_vocabularity words by analyzing training data (:math:`x\%`). Plot new out of vocab words that comes every single day. That number should be diminishing as we consider longer seen windows.
	- Use robust training (forcefully masking :math:`x\%` words during training time)

What Can I Hash?
----------------------------------------------------------------------------------------
- Direct embedding table for 5B users or 30B links is impractical
- Use hashing trick -> reduce cardinality -> one hot

	- Pros: manageable size
	- Cons: collision
========================================================================================
How Do I Create Training Data?
========================================================================================
Negative sampling

========================================================================================
How Do I Train Model?
========================================================================================
BCE loss with clicked/not
#. Architecture 1: MLP, wide and deep, deep and cross
#. Architecture 2:

========================================================================================
How Do I Evaluate Model?
========================================================================================
========================================================================================
How Do I Debug Model?
========================================================================================
========================================================================================
How Do I Deploy Model?
========================================================================================
========================================================================================
How Do I Monitor Model?
========================================================================================
========================================================================================
Can I Build a 2-Stage Ranking System?
========================================================================================
Embeddings can be reused for/finetuned with ranking as well.

Latent Features for Non-Personalised Retrieval
----------------------------------------------------------------------------------------
Supervised
----------------------------------------------------------------------------------------
#. Neural CF: 

		search_query -> double hashing -> embedding; search_query -> bag of words -> embedding
		link -> double hashing -> embedding
		label: clicked/not
		InfoNCE loss for optimal ANN search - negative sampling
		BPR loss for pairwise ranking

Self Supervised
----------------------------------------------------------------------------------------
Pretrain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Item2Vec

	#. Query embedding: connect search_query by edge if same link was clicked for both
	#. Link embedding: connect link by edge if clicked for the same search_query
	#. Learn embeddings separately by forming random walk sequence of 5 and then CBOW (word2vec) for masked middle
	#. NOTE: doesn't consider temporal features, learns long term understanding

#. GNN

	#. Connect search_query and link by edge if clicked and run message passing GCN algorithm
	#. Initiate from scratch or from neural CF learned embedding
	#. NOTE: doesn't consider temporal features, learns long term understanding

Finetune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use learned embeddings from pretraining step in neural CF

Latent Features for Personalised Retrieval
----------------------------------------------------------------------------------------
========================================================================================
Can I Build a Diversity Promoting Reranker?
========================================================================================
DPP click score with Gaussian/learned kernel

****************************************************************************************
Rich Event Logs
****************************************************************************************
****************************************************************************************
Content
****************************************************************************************
****************************************************************************************
Context
****************************************************************************************
