#################################################################################
Commerce
#################################################################################
.. image:: ../../img/commerce.png
	:width: 600
	:alt: Framework

.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

*********************************************************************************
Features
*********************************************************************************
Item Embedding
=================================================================================
Ranker
=================================================================================
1. User features
---------------------------------------------------------------------------------
2. Item features
---------------------------------------------------------------------------------
3. Context features
---------------------------------------------------------------------------------
4. Statistical features
---------------------------------------------------------------------------------
5. Fusion features
---------------------------------------------------------------------------------
.. csv-table::
	:header: "Index", "Feature Type", "Description"
	:widths: 4 12 32
	:align: center
	
		1, dense_score, Cosine or dot similarity with query embedding
		2, bm25_score, Raw or normalized BM25 score
		3, query_term_match_ratio, # query tokens matched / total query tokens
		4, category_match, 1 if same category; 0 otherwise
		5, brand_match, 1 if same brand; 0 otherwise
		6, image_sim_score, Optional: if query is image
		7, retrieval_source, One-hot: [dense only; sparse only; both]
		8, retrieved_rank_dense, Position in dense top-k list
		9, retrieved_rank_sparse, Position in sparse top-k list
		10, dense_to_sparse_rank_gap, Sparse rank - Dense rank

*********************************************************************************
Domain Understanding
*********************************************************************************
Listings
=================================================================================
.. csv-table::
	:header: "Attribute", "Sub-attribute", "Examples", "Characteristics"
	:widths: 16 12 12 32
	:align: center
	
		1. Title/Desc, , , uninformative; misleading; spelling/grammar errors
		2. Images, , , low quality
		3. Location, , ,
		, Postal, , user provided -> low coverage; incorrect
		, Lat-Long, , gps inferred -> high coverage; incorrect; upload location might be different than product availability
		4. Price, , , incorrect; misleading/scam
		5. Category, , , mostly missing; possibly incorrect
		6. Tags, , , category dependent; mostly missing; possibly incorrect
		, 1. Attributes, colour; size,
		, 2. Condition, new; refurbished, 
		, 3. Style, minimalistic; vintage; casual,
		, 4. Use-case, gift-ideas; travel friendly,
		, 5. Occasion, wedding; office; gym,
		, 6. Catchphrases, huge discount, open-ended; clickbaity

*********************************************************************************
Product Understanding
*********************************************************************************
Taxonomy classification
=================================================================================
Attribute extraction
=================================================================================
Entity linking
=================================================================================
*********************************************************************************
Product Quality & Integrity
*********************************************************************************
Duplicate detection
=================================================================================
Moderation
=================================================================================
*********************************************************************************
Product Search
*********************************************************************************
Problem Understanding
=================================================================================
#. Use-case
   - System:
     - text queries
     - system returns a list of listings
     - sorted to maximise engagement
     - filtered by geolocation
     - [*] personalisation
     - [*] contextualisation
     - available across different surfaces
   - Actions (users):
     - click -> product details page
       - save to wishlist
       - contact seller
     - scroll past
   - Actors:
     - users, sellers, platform
   - Interests:
     - users: find most relevant results
     - sellers: increase coverage of their listings
     - platform:
       - [out of scope] quality: results should not contain listings that violate policies
       - user engagement
#. Business KPIs
   - CTR, CVR, coverage, QBR, DwellTime
#. Scale
   - 1M sellers, 50M listings, 1M/day new listings
   - 1B users, 95% on mobile device
   - low latency req (50ms for retrieval, 200ms for rerank)
#. Signals
   - Search logs
     - events: click, dwell-time, contacted-seller, added-to-wishlist
       - clicks: 10-20%, noisy (weak signal - curiosity, clickbaits)
       - dwell-time
       - added-to-wishlist: 1-3%, (stronger - delayed feedback, sparse, niche/personalised)
       - contacted-seller: 0.1-0.5% (delayed feedback)
     - depends on:
       - platform: surface, display-pos
       - seller: listing-quality, seller reputation, previous engagement with seller
       - user: user's click propensity overall/query-specific/category-specific/attribute-specific
   - baseline - kw search
#. Misc
   - subsystems
     - listings side
       - kw extraction
       - taxonomy classification
       - attribute extraction
     - query side
       - query segmentation
       - query intent - browse, buy, brand
       - query rewrite/expansion

Sparse Retrieval
=================================================================================
Dense Retrieval
=================================================================================
Fusion
=================================================================================
Re-ranking
=================================================================================
Personalised Search
=================================================================================

*********************************************************************************
Product Recommendation
*********************************************************************************
Similar listings recommendation
=================================================================================
Homepage recommendation
=================================================================================
