####################################################################################
Data and Label Collection
####################################################################################
************************************************************************************
Collaborative Filtering (CF)
************************************************************************************
- Relies on user-item interactions to recommend items. 
- Since users rarely provide explicit ratings, implicit signals are inferred from engagement behaviors.  

User Engagement-Based Labels  
====================================================================================
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: left
	:widths: 16, 16, 24

		Clicks, Count clicks on an item.,  Simple; scalable.  Clicking  liking (accidental clicks).
		Watch Time / Dwell Time, Measure time spent on videos/articles.,  Captures engagement depth.  Long duration  satisfaction (e.g.; passive watching).
		Purchase / Conversion, Track purchases (e-commerce; rentals; subscriptions).,  Strongest preference signal.  Sparse data (only a few items are purchased).
		Add to Cart / Wishlist, Users mark interest without purchasing.,  Softer preference signal.  Users may abandon carts.
		Scrolling & Hovering, Detect mouse hover time over items.,  Early preference signal.  May be unintentional.
		Search Queries & Item Views, Items viewed after searching for a term.,  Strong relevance signal.  Some users browse randomly.

Social & Community-Based Signals  
====================================================================================
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: left
	:widths: 16, 16, 24

		Likes / Upvotes, Count "likes" on posts; videos; or comments.,  Clear positive feedback.  Some users never like items.
		Shares / Retweets, Count how often users share content.,  Strong endorsement.  May share for controversy.
		Follows / Subscriptions, Followed creators or product wishlists.,  Indicates long-term interest.  Users may follow without deep engagement.

Negative Feedback & Implicit Dislikes  
====================================================================================
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: left
	:widths: 16, 16, 24

		Skip / Bounce Rate, Detect when a user skips a song/video quickly.,  Identifies disinterest.  May skip for reasons unrelated to content.
		Negative Actions, "Not Interested" clicks; downvotes; blocking content.,  Explicit dislike signal.  Only a subset of users take these actions.

CF Use Case Example:  
- Spotify uses play count, skip rate, and playlist additions to infer user preferences.  
- Netflix monitors watch completion rate, rewatches, and early exits for movie recommendations.  

************************************************************************************
Content-Based Filtering (CBF)
************************************************************************************
Session-Based & Short-Term Context Labels
====================================================================================
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: left
	:widths: 16, 16, 24

		Recent Search Context, Track evolving search terms.,  Captures short-term needs.  Trends change quickly.
		Location-Based Preferences, Match user location with nearby content.,  Useful for local recommendations.  Privacy-sensitive.
		Time of Day / Activity Patterns, Suggest different items based on morning/evening behavior.,  Improves context relevance.  Needs continuous adaptation.

Self-Supervised Paradigm
====================================================================================
TODO

************************************************************************************
Knowledge Graphs for Hybrid Labeling
************************************************************************************
- Uses entities and relationships to enhance recommendations.
