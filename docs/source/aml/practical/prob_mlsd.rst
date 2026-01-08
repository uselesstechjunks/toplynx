#########################################################################
MLSD Problems
#########################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

*************************************************************************
Section 1: Modeling Paradigm and Loss Selection
*************************************************************************
1. You are designing a recommender for a fashion ecommerce app. Users can like, save, or share an item. How would you frame this task? What modeling paradigm and loss functions would you use?

2. You need to categorize uploaded products into one of 300 categories. Some categories are similar (e.g., smartphones vs. feature phones). Should you model this as multi-class or ordinal regression? Why?

3. You're building a video watch-next model where you care about clicks, watch duration, and replays. How would you model this: multi-task, ranking, or regression?

4. You're tasked with reranking a short list of ads that were retrieved using keyword match. You have click, hover, and conversion data. How would you design your model and loss function?

*************************************************************************
Section 2: Evaluation Metric Alignment
*************************************************************************
5. A music app wants to predict top genres for users. The model is multi-label. How would you evaluate this model offline? What metrics would you avoid and why?

6. You deploy a ranking model using MAP@10, but notice poor NDCG@3 online. What might be causing this discrepancy?

7. You train a retrieval model using InfoNCE, but CTR in production drops. You find out that ANN recall@k improved. How would you debug this?

8. Your model performs well on macro F1 for a multi-label toxic comment detector, but still fails to flag some severe toxic classes. What metric might better reflect business needs?

*************************************************************************
Section 3: Joint and Multi-Task Training
*************************************************************************
9. You are jointly training a retrieval and ranking model. Retrieval improves (recall@100↑), but ranking degrades (NDCG@10↓). Shared encoder. What might be the issue?

10. You're training click, cart, and purchase tasks together. Click improves, but purchase plateaus. List three changes to make the training more effective.

11. You train a multi-task model with shared towers, but notice oscillating performance across tasks during training. How can PCGrad or GradNorm help?

12. You want to personalize feed ranking for a short video app using both implicit feedback (dwell time) and explicit signals (likes). How would you model and train this system?

*************************************************************************
Section 4: Architecture and Scalability Trade-offs
*************************************************************************
13. You're building a real-time search system over 50M listings. You want both relevance and personalization. How do you balance between precomputed item embeddings and real-time ranking?

14. You're asked to re-use your item encoder from attribute prediction to train a semantic retrieval model. What changes would you make to adapt it?

15. You want to use metric learning for user-item matching, but also optimize for downstream click-through-rate. How would you structure training and inference?

16. For a social feed, you're using a BST-style session transformer for ranking. Latency becomes a problem. What trade-offs and simplifications would you consider?

21. You serve search results using a dual-tower model with 100M product embeddings. ANN recall@100 is good, but users complain about repetitive results. What architectural changes would you explore to inject more diversity?

22. You're asked to migrate a BERT-based query encoder to a production search system with <10ms latency budget. What architectural simplifications and trade-offs would you make?

23. You're training a unified user tower for retrieval and ranking across 4 surfaces (search, home, notifications, ads). What's your strategy for shared vs. surface-specific components?

24. You want to rank listings on a marketplace using both image and title. Inference latency doubles after adding a vision encoder. How would you reduce the overhead while preserving semantic quality?

25. You're training a reranker with cross-encoders (user + item in same tower). Training performance is great, but you can't deploy due to ANN constraints. How would you redesign for deployment?

26. You've pre-trained your item tower using self-supervised contrastive learning. When used in a ranking model, it underperforms compared to a CE-trained model. What architectural alignment issues could be at play?

*************************************************************************
Section 5: Debugging and Failure Mode Diagnosis
*************************************************************************
17. You observe that your item embeddings collapse near origin after triplet training. What might be the issue?

18. Your reranker, trained with listwise loss, is overfitting to top popular items. How would you debug this and improve diversity?

19. Your cold-start user model retrieves the same popular jobs for everyone. Where would you intervene: model, data, loss?

20. You're training with a noisy click signal. Precision@k looks good, but users complain about irrelevant content. What could be wrong and how would you fix it?

27. Your pairwise LTR model performs well offline, but online CTR drops. You suspect false positives in the training negatives. How do you trace and fix this?

28. A product recommendation model starts surfacing out-of-stock or policy-violating items. You already have a quality filter. What part of the pipeline might be causing this and how do you fix it?

29. Your shared user tower for search and home feed is overfitting to head queries. Recall@100 for tail queries drops. What architecture or sampling fixes might help?

30. Your model NDCG improves steadily, but add-to-cart conversion drops. Your team suspects overfitting to position bias in training. How would you detect and correct this?

31. A fine-tuned reranker trained on clicks performs poorly when deployed on "new arrivals." The CTR gap between new and old items widens. What's likely happening?

32. During joint training, one task shows noisy loss oscillations while others converge. You suspect instability in the sampling logic. What are the common culprits?

33. During online A/B testing, CTR improves, but downstream business metrics (purchases, returns) degrade. What might be happening?

34. Your reranker performs poorly on longer product descriptions. Top-5 retrieval recall is good. What could be going wrong, and where would you fix it?

*************************************************************************
Section 6: Cross-Modal Retrieval, Ranking, and Personalization
*************************************************************************
35. You're training a dual-encoder for multimodal product listings (title + image). In deployment, only text is available for user queries. How do you ensure your model still learns strong cross-modal alignment?

36. A vision encoder trained on clean product studio images underperforms on mobile-uploaded photos from users. How would you adapt the encoder for user-taken image queries?

37. You jointly train an item encoder using image, title, and description. But when you ablate image features, ranking improves. What could explain this?

38. You want to personalize fashion search results using both listing images and user preferences (past clicks, style). What architecture would let you combine static image features and dynamic user embeddings efficiently?

*************************************************************************
Section 7: Latency Constraints and Inference Optimizations
*************************************************************************
39. You deploy a ranking model with 3 heads (click, save, purchase). Latency increases non-linearly with each head. What architecture changes could reduce inference time while preserving multi-objective performance?

40. Your Transformer-based re-ranker is too slow for live ranking. You try distillation but the accuracy drops. What alternatives would you try to preserve ordering quality?

41. A production image tower is bottlenecking your feed ranking system. What methods could you use to cache or partially precompute features to stay within latency budget?

*************************************************************************
Section 8: ANN-Specific Retrieval Challenges
*************************************************************************
42. You fine-tune your user-item towers with InfoNCE and ANN recall@100 improves. But downstream ranking quality degrades. You suspect ANN misalignment. What are the most likely failure points?

43. You swap cosine similarity with dot product in your ANN retriever to enable popularity-weighted scores. Performance drops. Why might this happen?

44. Your ANN index has high recall, but retrieved items are dominated by a few popular clusters. How would you fix embedding drift or improve diversity?

*************************************************************************
Section 9: Tail Query Recovery and Head Bias
*************************************************************************
45. You train a dense retriever on search queries. Head queries dominate logs. During eval, recall@100 for tail queries is low. What model and sampling strategies would help?

46. You pre-train your item tower on co-clicks and train your user tower on click logs. The model over-personalizes and fails to generalize on rare or new queries. How do you debug and fix it?

47. You train a transformer reranker on full impression logs. For long-tail queries, it often gives irrelevant results even when candidates are fine. What could be going wrong?

*************************************************************************
Section 10: Product Categorization in Marketplace
*************************************************************************
Problem:
Given a noisy listing (title, description, image, maybe user tags), assign it to a category from a flat taxonomy of 300 classes, which are semantically related and possibly hierarchical (e.g., Electronics → Phones → Smartphones).

*************************************************************************
Section 11: Ads Moderation: Modeling + System Design
*************************************************************************
Q1.
You are building an ad moderation classifier that must detect multiple violations such as:
- prohibited item (e.g., drugs, weapons)
- misinformation
- sensational claims
- political content

Ads often violate multiple policies at once.
How would you frame this modeling task? What loss and evaluation metric would you use?

Q2.
Your ads moderation model is missing many violations in edge cases (e.g., subtle wording, region-specific political terms), even though precision is high.
How would you improve recall without exploding false positives?

Q3.
Policy violations are reviewed by human moderators. Most examples labeled as “clean” are never manually reviewed. You suspect some positives are missed.
How would you modify your training setup or model to handle this label noise?

Q4.
You train a binary classifier for each violation type using shared encoders. During training, you observe that some heads overfit (training AUC > 0.99, val AUC < 0.7), while others underperform.
What’s the likely cause, and how would you fix it?

Q5.
You need to moderate ads across 20 countries. Certain violation types (e.g., political content) vary by region.
Would you use one model or multiple? How would you share information across regions without hurting precision?

*************************************************************************
Section 12: Content Understanding: Taxonomy + Semantics
*************************************************************************
Q6. You are building a content classifier that tags posts into topics:
- parenting, dating, career, mental health, etc.
Each post can belong to multiple overlapping topics.

What modeling and loss design would you use? How would you deal with overlapping labels?

Q7. Your topic classifier performs poorly on long posts. Investigation shows that key topics are mentioned late in the text.
What architectural changes would you consider?

Q8. You’re tagging posts using a 4-level topic hierarchy. You only have partial labels for most training examples (e.g., only level-1 or level-2).
How would you design the model and loss to train on this partially labeled data?

Q9. You use a flat softmax over 500 topics. Most errors are near-misses (e.g., “career coaching” vs. “job hunting”).
What can you change in the architecture or loss to make the model confusion-aware?

Q10. Your content classifier is used for downstream moderation (e.g., escalation to reviewers). Reviewers complain that the top-k predictions often skip low-frequency but critical categories.
How would you redesign the loss, training data, or post-processing to account for this?

*************************************************************************
Section 13: Difficult Data Regimes
*************************************************************************
Q1. You’re training a product quality classifier. You have:

- 1% manually reviewed “high-quality” items (positive)
- 50% of data labeled via heuristics (e.g., co-view count, description length)
- 49% unlabeled

You want to train a binary classifier. What’s the best strategy?

A. Filter the heuristic labels using a threshold and train with BCE  
B. Train with positive+unlabeled (PU learning) using reviewed items as positives  
C. Use the heuristic as soft labels and apply BCE with label smoothing  
D. Train contrastive loss using co-view pairs as positives

Q2. In a region-specific moderation task, the “hate speech” class is:

- Rare
- Labeled inconsistently
- Shows severe overfitting after only a few epochs

What’s the best modeling strategy?

A. Upweight the loss and add task-specific dropout  
B. Use BPR-style ranking loss  
C. Downsample negatives to rebalance  
D. Freeze shared layers and adapt via low-rank adapter layers + gradual unfreezing

Q3. You’re training a co-purchase product similarity model:

- Use co-purchase pairs as positives
- Use in-batch negatives
- Some pairs include unrelated items due to bulk-buying by bots

What helps most?

A. Dropout in input projection layer  
B. Weighted InfoNCE with confidence score per positive  
C. Increase temperature in softmax  
D. Add domain classifier to detect bot-bought patterns

Q4. You’re fine-tuning a text encoder on a noisy tagging task where tags come from user-generated hashtags:

- Some tags are correct
- Some are ambiguous
- Many are missing entirely

Which architecture choice helps the most?

A. Add per-tag loss weights  
B. Use mean pooled CLS + sigmoid layer  
C. Add a contrastive head alongside tag prediction  
D. Use temperature-scaled softmax and top-1 supervision
