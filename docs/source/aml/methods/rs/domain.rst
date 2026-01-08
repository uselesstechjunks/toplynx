####################################################################################
Domain Knowledge
####################################################################################
************************************************************************************
Industry Practice
************************************************************************************
.. csv-table::
	:header: "SNo", "Domain", "Company", "Retrieval Methodology", "Class"
	:align: left
	:widths: 2, 16, 8, 48, 16

		a, Commerce, Amazon, Item-to-item CF, Item-based
		b, UGC, Facebook, Embedding-based retrieval; multiple user-centric sources, User-based
		b, UGC, Instagram, Multi-source retrieval; Two-Tower: user and item embeddings, User-item
		c, Short video, TikTok, Deep retrieval; user-video interaction, User-item
		c, Short video, YouTube Shorts, Deep learning-based candidate generation; user-video, User-item
		d, Long video, YouTube, Deep learning-based candidate generation; user-video, User-item
		e, Published media, Netflix, Specialized ML personalization models, User-item
		e, Published media, Prime Video, Matrix factorization; content-based, User-item
		e, Published media, Spotify, Generative retrieval; content + user-item interaction, User-item
		f, Inventory retail, Walmart, Hybrid retrieval; traditional + embedding, Item-based
		f, Inventory retail, Target, Contextual multi-arm bandits; personalization focus, User-based
		g, Travel, Airbnb, Embedding-based retrieval; item2vec-style, Item-based
		h, Food delivery, Uber Eats, Two-Tower embeddings; graph learning; user-food, User-item
		i, News, Google News, CF on click behavior; user-article, User-item
		j, Search engine, Google, Traditional IR; query-document matching; evolving toward user-item based ranking, User-item

************************************************************************************
Design Consideration
************************************************************************************
- Item-based is more robust; scalable; and explainable under uncertainty.
- User-item is more powerful and fine-grained when you have dense; high-quality user interaction data.

.. csv-table::
	:header: "SNo", "Situation", "Why Item-Based Wins", "Practical Examples"
	:align: left
	:widths: 2, 16, 32, 24

		1, Stable; low-churn catalogs, If item properties don't change rapidly; item-item relationships stay valid for a long time., Retail (Walmart; Target); long-tail marketplaces.
		2, Cold-start users, New or infrequent users have no behavioral data. Item-item similarity avoids the need for personalization., Food delivery (e.g.; recommend "popular in your area" restaurants).
		3, Sparse user history, If users interact very little; you can't learn good user embeddings. But item similarity can still be robust., New users on Amazon: recommend similar products without needing deep profiles.
		4, Highly diverse user base, When user behavior varies too much (by region; culture; device); item-based models generalize better., Global apps with fragmented user bases (e.g.; Spotify in different countries).
		5, Heavy multi-user devices, If multiple people use the same device/account; user profiles become noisy. Item-based CF is safer and more accurate., Shared smart TVs; public kiosks; family shopping accounts.
		6, Explainability, Item-based recommendations ("users who bought X also bought Y") are easier to justify to users and auditors., E-commerce and B2B sales platforms.
************************************************************************************
Search
************************************************************************************
.. note::
	- [fennel.ai] `Feature Engineering for Personalized Search <https://fennel.ai/blog/feature-engineering-for-personalized-search/>`_

************************************************************************************
Search Advertising
************************************************************************************
.. csv-table::
	:header: "Issue", "Why It Matters", "Strategic Fixes", "Trade-Offs"
	:align: center

		Relevance vs. Revenue, Showing high-bid but low-relevance ads hurts trust, Hybrid ranking (bid + quality), Too much relevance filtering lowers revenue
		Click Fraud & Ad Spam, Inflated clicks drain budgets, ML-based fraud detection, False positives can hurt advertisers
		Ad Auction Manipulation, AI-driven bid shading exploits system, Second-price auctions, Reduced ad revenue
		Ad Fatigue & Banner Blindness, Users ignore repetitive ads, Adaptive ad rotation, Frequent ad refreshing increases costs
		Query Intent Mismatch, Poor ad matching frustrates users, BERT-based intent detection, Over-restricting ads lowers monetization
		Landing Page Experience, High bounce rate = low conversion, Quality Score rules, Strict rules limit advertiser flexibility
		Multi-Touch Attribution, Last-click attribution undervalues early ad exposures, Shapley-based attribution, More complexity; slower optimization
		Ad Bias & Fairness, Favoring large advertisers hurts competition, Fairness-aware bidding, Less revenue from high bidders

Relevance vs. Revenue Trade-Off
====================================================================================
Why It Matters 

	- Advertisers bid for visibility, but their ads may not always be relevant to the user's query. 
	- If high-bid but low-relevance ads are shown, users may lose trust in the search engine. 

Strategic Solutions & Trade-Offs 

	- Quality Score (Google Ads' Approach)  Ranks ads based on a combination of CTR, relevance, and landing page experience, not just bid amount. 
	- Hybrid Ranking Model (Revenue + User Engagement)  Balances ad revenue vs. user satisfaction. 

Trade-Offs 

	- Prioritizing high-relevance, low-bid ads reduces short-term revenue. 
	- Prioritizing high-bid, low-relevance ads hurts user trust & long-term retention. 

Click Spam & Ad Fraud
====================================================================================
Why It Matters 

	- Bots & malicious actors inflate clicks to waste competitor ad budgets (click fraud). 
	- Some advertisers run low-quality, misleading ads to generate fake engagement. 

Strategic Solutions & Trade-Offs 

	- Click Fraud Detection (Googles Invalid Click Detection)  Uses IP tracking, anomaly detection, and ML models to filter fraudulent clicks. 
	- Post-Click Analysis (User Behavior Analysis)  Detects bots based on engagement (bounce rate, session length, interactions). 

Trade-Offs 

	- False Positives  May block legitimate traffic, harming advertisers. 
	- False Negatives  Fraudulent clicks still get monetized, increasing costs for real advertisers. 

Ad Auction Manipulation & Bid Shading
====================================================================================
Why It Matters 

	- Sophisticated advertisers use AI-driven bidding strategies to game real-time auctions. 
	- Bid shading techniques lower ad costs while maintaining high visibility. 

Strategic Solutions & Trade-Offs 

	- Second-Price Auctions (Vickrey Auctions)  Advertisers only pay the second-highest bid price, reducing manipulation. 
	- Multi-Objective Bidding Models  Balances advertiser cost efficiency and search engine revenue. 

Trade-Offs 

	- Too much bid control reduces revenue  Search engines may earn less per click. 
	- Aggressive bid adjustments can reduce advertiser trust  If advertisers feel theyre losing transparency, they may pull budgets. 

Ad Fatigue & Banner Blindness
====================================================================================
Why It Matters 

	- Users ignore repetitive ads after multiple exposures, reducing CTR over time. 
	- If ads look too much like organic results, users may feel deceived. 

Strategic Solutions & Trade-Offs 

	- Adaptive Ad Rotation (Google Ads Optimize for Best Performing Mode)  Dynamically swaps low-performing ads with higher-engagement creatives. 
	- Ad Labeling Transparency  Clearer Sponsored tags improve user trust but reduce click rates. 

Trade-Offs 

	- Refreshing ads too frequently raises advertiser costs. 
	- Too much ad transparency leads to lower revenue per impression. 

Query Intent Mismatch
====================================================================================
Why It Matters 

	- Search queries are often ambiguous, and poor ad matching leads to bad user experience. 
	- Example Searching for Apple  Should the search engine show Apple iPhones (commercial intent) or apple fruit (informational intent)? 

Strategic Solutions & Trade-Offs 

	- Intent Classification Models (BERT, T5-based Models)  Classify queries into commercial vs. informational intent. 
	- Negative Keyword Targeting (Google Ads' Negative Keywords)  Advertisers block unrelated queries from triggering their ads. 

Trade-Offs 

	- Restricting ads based on intent can lower revenue. 
	- Allowing broad ad targeting risks user dissatisfaction. 

Landing Page Experience & Conversion Rate Optimization
====================================================================================
Why It Matters 

	- Even if an ad gets high CTR, if the landing page is misleading or slow, users bounce without converting. 
	- Google penalizes low-quality landing pages via Quality Score reductions. 

Strategic Solutions & Trade-Offs 

	- Landing Page Quality Audits (Googles Ad Quality Guidelines)  Checks for page speed, relevance, mobile-friendliness. 
	- Post-Click Engagement Monitoring  Uses bounce rate, time-on-site, conversion tracking to refine ranking. 

Trade-Offs 

	- Strict landing page rules limit advertiser flexibility. 
	- Relaxed rules allow low-quality ads, reducing long-term trust. 

Multi-Touch Attribution & Ad Budget Allocation
====================================================================================
Why It Matters 

	- Users may see an ad but not convert immediately  Traditional last-click attribution ignores earlier touchpoints. 
	- Advertisers struggle to allocate budgets across search, display, social, and video ads. 

Strategic Solutions & Trade-Offs 

	- Multi-Touch Attribution Models (Shapley Value, Markov Chains)  Assigns fair credit to different ad exposures. 
	- Cross-Channel Conversion Tracking  Tracks user journeys across search & display ads. 

Trade-Offs 

	- More complex attribution models require longer training times. 
	- Over-attributing upper-funnel ads can inflate costs without clear ROI. 

Fairness & Ad Bias Issues
====================================================================================
Why It Matters 

	- Some ad auctions are biased against small advertisers, favoring large ad budgets. 
	- Discriminatory ad targeting (e.g., gender/race bias in job/housing ads) can lead to regulatory penalties. 

Strategic Solutions & Trade-Offs 

	- Fairness-Constrained Bidding (Googles Fairness-Aware Ad Auctions)  Adjusts auction weights to prevent dominance by large advertisers. 
	- Bias Detection in Ad Targeting (Auditing Models for Discriminatory Targeting)  Ensures fair exposure of diverse ads. 

Trade-Offs 

	- Too much fairness correction may reduce revenue from high-bidding advertisers. 
	- Too little correction risks regulatory lawsuits (e.g., Facebooks 2019 lawsuit for discriminatory ad targeting). 

************************************************************************************
Music
************************************************************************************
.. csv-table::
	:header: "Challenge", "Why Its Important", "Trade-Offs"
	:align: center

		Personalization vs. Serendipity, Users want relevant music but also expect some new discoveries., Too much personalization  Feels repetitive. Too much exploration  Feels random.
		Repetition & Content Fatigue, Users get frustrated if the same songs appear too often., Strict anti-repetition  May exclude user favorites. Loose constraints  Risk of overplaying certain songs.
		Context & Mood Adaptation, Users listen to music differently based on mood; time; activity (workout; relaxation)., Explicit mood tagging is effective but requires manual input. Implicit context detection risks wrong assumptions.
		Balancing Popular & Niche Tracks, Highly popular songs dominate engagement; making it hard for lesser-known songs to gain exposure., Boosting niche tracks improves diversity; but may lower engagement metrics.
		Cold-Start for New Songs & Artists, Newly released songs struggle to get exposure due to lack of engagement signals., Over-boosting new music can lead to reduced user satisfaction.
		Playlist Length & Engagement Optimization, Users may not finish long playlists; leading to low engagement metrics., Shorter playlists increase completion rate; but longer ones improve session duration.

Playlist Generation & Curation in Music Recommendation Systems
====================================================================================
Types of Playlists & Their Challenges
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Playlist Type", "Example", "Key Challenges"
	:align: center

		Personalized Playlists, Spotifys Discover Weekly; YouTube Musics Your Mix, Ensuring balance between familiar & new tracks.
		Mood/Activity-Based Playlists, Workout Mix; Chill Vibes; Focus Music, Detecting mood & intent dynamically.
		Trending & Algorithmic Playlists, Spotifys Top 50; Apple Musics Charts, Avoiding popularity bias while staying relevant.
		Collaborative & Social Playlists, Spotify Blend; Apple Musics Shared Playlists, Handling conflicting preferences in shared lists.
		Genre/Artist-Centric Playlists, Best of 90s Rock; Jazz Classics, Ensuring diversity within a theme.

Solutions to Key Playlist Challenges
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Challenge", "Solution", "Trade-Off"
	:align: center

		Over-Personalization (Echo Chamber), Inject 5-20% exploration (Multi-Armed Bandits), Too much exploration may decrease CTR
		Repetition & Content Fatigue, Anti-repetition rules (e.g.; same song cannot appear in back-to-back sessions), May prevent users from hearing favorite tracks
		Cold-Start for New Songs, Boost underexposed songs using metadata (tempo; genre), Over-promoting new songs may harm engagement
		Context-Aware Playlists, Use real-time signals (e.g.; running mode detects movement; adjusts tempo), Misinterpreted context may cause poor recommendations
		Playlist Completion Rate, Optimize for average session length (shorter playlists for casual users; longer for engaged users), Shorter playlists may reduce playtime per session

Common Problems
====================================================================================
Cold-Start Problem for New Artists & Songs
------------------------------------------------------------------------------------
- Why It Matters:

	- New artists and newly released tracks struggle to get exposure since they have no engagement history.

- Strategic Solutions & Trade-Offs:

	- Metadata-Based Recommendations (Genre, BPM, lyrics embeddings)  Useful for early exposure but lacks engagement feedback.
	- Collaborative Boosting (Linking new artists to known artists)  Improves visibility but risks inaccurate pairing.
	- User-Driven Exploration (Playlists like Fresh Finds)  Promotes new songs but may not reach mainstream listeners.

- Example:

	- Spotifys Fresh Finds is a human-curated playlist designed for emerging artists.

Popularity Bias & Lack of Exposure for Niche Artists
------------------------------------------------------------------------------------
- Why It Matters:

	- Big-label artists dominate recommendations, making it hard for new/independent musicians to gain visibility.
	- Overemphasis on top charts and algorithmic repetition reinforces the same mainstream music.

- Strategic Solutions & Trade-Offs:

	- Fairness-Aware Re-Ranking (Exposing lesser-known artists)  Promotes diversity but may reduce engagement.
	- User Preference-Based Exploration (Blending familiar & new artists)  Increases discovery but harder to balance.
	- Contextual Boosting (Surfacing niche content in certain playlists)  Encourages exploration but risks user dissatisfaction.

- Spotifys Fix:

	- Discover Weekly and Release Radar to highlight emerging artists.

Balancing Exploration vs. Personalization in Playlists
------------------------------------------------------------------------------------
- Why It Matters:

	- Users want to hear familiar songs but also expect discovery of new tracks.
	- Too much exploration reduces engagement, too little keeps users stuck in their existing preferences.

- Strategic Solutions & Trade-Offs:

	- Reinforcement Learning-Based Ranking (Balancing Novelty & Familiarity)  Dynamically adjusts exploration but requires more data.
	- Hybrid Personalized Playlists (50% known, 50% new)  Encourages discovery but still risks disengagement.
	- Diversity Re-Ranking Models (Ensuring mix of different artist popularity levels)  Enhances engagement but increases complexity.

- Spotifys Fix:

	- Discover Weekly mixes familiar artists with newly recommended artists.

Repetition & Content Fatigue (Avoiding Overplayed Songs)
------------------------------------------------------------------------------------
- Why It Matters:

	- Users dislike hearing the same songs too frequently in personalized playlists.
	- Music recommendation systems tend to reinforce top tracks due to high past engagement.

- Strategic Solutions & Trade-Offs:

	- Play-Session Awareness (Avoiding recently played tracks)  Prevents fatigue but risks reducing personalization strength.
	- Diversified Playlist Generation (Embedding Clustering)  Encourages discovery but may introduce unrelated tracks.
	- Temporal Diversity Constraints (Recommender-aware time gaps)  Reduces overexposure but adds complexity to ranking models.

- Spotify & Apple Musics Fix:

	- Autogenerated playlists (e.g., Daily Mix, Radio) have anti-repetition constraints.

Context-Aware Recommendations (Music for Different Situations)
------------------------------------------------------------------------------------
- Why It Matters:

	- Music preferences vary by context (workout, driving, studying, relaxing), but most recommenders treat all listening the same.

- Strategic Solutions & Trade-Offs:

	- User-Controlled Context Tags (Spotifys Mood Playlists, YouTube Musics Activity Mode)  More control but adds friction.
	- Implicit Context Detection (Using location, time, device, previous context switches)  Improves automation but risks privacy concerns.
	- Adaptive Playlist Generation (Real-time context-aware re-ranking)  Better real-world usability but increases computational costs.

- Industry Example:

	- Spotifys Made for You mixes genres based on past listening sessions.

Short-Term vs. Long-Term Personalization
------------------------------------------------------------------------------------
- Why It Matters:

	- Users music preferences change over time, but most recommendation models overly rely on recent activity.
	- Recommending only recently played songs can overfit short-term moods and ignore long-term preferences.

- Strategic Solutions & Trade-Offs:

	- Session-Based Personalization (Short-Term Context Models)  Captures mood-based preferences but can overfit recent choices.
	- Hybrid Long-Term + Short-Term Embeddings (Contrastive Learning on Listening History)  Balances nostalgia & discovery but computationally expensive.
	- Decay-Based Weighting on Past Behavior  Helps phase out stale preferences but requires careful tuning.

- Spotifys Approach:

	- Balances On Repeat (long-term) and Discover Weekly (exploration).

Multi-Modal Recommendation (Lyrics, Podcasts, Audio Similarity)
------------------------------------------------------------------------------------
- Why It Matters:

	- Music discovery can be driven by lyrics, themes, artist backstories, and spoken content (podcasts).
	- Traditional recommendation models focus only on collaborative filtering (listening history).

- Strategic Solutions & Trade-Offs:

	- Lyrics-Based Embeddings (Thematic music recommendations)  Enhances meaning-based recommendations but requires NLP processing.
	- Cross-Domain Music-Podcast Recommendation (Shared interests)  Improves discovery but harder to rank relevance.
	- Audio Similarity-Based Retrieval (Matching based on timbre, rhythm)  Better for organic discovery but requires deep learning models.

- Industry Example:

	- YouTube Music cross-recommends music & podcasts based on topics.

************************************************************************************
Social Media
************************************************************************************
************************************************************************************
Video
************************************************************************************
************************************************************************************
E-Commerce
************************************************************************************
