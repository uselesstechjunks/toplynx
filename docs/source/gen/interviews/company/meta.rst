##################################
Meta
##################################
Source: https://www.teamblind.com/post/Meta-MLE-E6-ML-System-Design-Interview-XaoxCs0c/41332921

While you're expected to ask (a lot of) clarifying Qs primarily for understanding system constraints (Scale of Users, Scale of posts, SLAs, Personalization, Locale/Language constraints and modes of input e.g. text/video etc.), the main focus should be on different modeling architectures and their tradeoffs. Spend about 10 minutes on the Data and Feature Tx/Engg. , but the bulk of it should be focused on different RecSys architectures. Read up on Multi-stage Recommenders (FB has an excellent one here: https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) and talk about different architectures e.g. Two-towers v/s DCN v/s Multi-task architectures (this is an extremely good resource: https://aman.ai/recsys/architectures/). The key challenge is threading the needle on the ML System focused on modeling while also touching base on defining the ML objective, Data, Features, Models, Offline/Online Metrics, Pipelines, Deployment (A/B Test etc.) in 35+ minutes.

ML Design structure was primarily from Alex Xu's wonderful book on ML System Design (https://bytebytego.com/intro/machine-learning-system-design-interview) and finally keeping updated with latest Research papers and blogs from most companies. A typical ML System Design structure involves the following stages:

0. Outline the System Requirements and constraints: Scale of Users, Scale of posts, SLAs, Personalization, Locale/Language constraints and modes of input e.g. text/video etc.

1. Given a Business/Product Use-case, define the ML Objective (e.g. "Design a (News) Feed for App Users" translates to something like: "Given a Series of unseen/seen posts, display a set of posts ordered by engagement (you need to define what engagement is e.g. views, clicks, likes, shares, reports, comments etc.))

2. Outline the various Data sources or Actors involved in this System: e.g. User-Post interactions, Users, Contexts, Posts etc.

3. Outline aspects of Feature Transformations (Generating dense features e.g. embedding based features, sparse features e.g. user-post historical interactions etc., post features (embeddings etc.)), user features (context, past aggregate stats etc.) and finally what labels you may want to use ; this is probably among the complex components to design for a specific product as the richer the feature set, the better the model generalizes.

4. Model Architectures: Since the scale of modern RecSys is massive, maybe a multi-stage system is better. Refer to the Instagram Explore blogpost I mentioned above. Here is where your main focus and Modeling experience comes through. How do we use a Two-Tower model from the features and labels generated above for our training phase? How do we choose a Ranking objective (Single-objective v/s Multi-objective), what loss functions power those objectives, do Deep-Cross Networks capture better interactions than a vanilla two-tower network, or do we choose something more complex e.g. MMoE etc. These are some of the various architectures you need to be aware of and talk through the tradeoffs depending upon product maturity etc.

5. Metrics and Model Evaluation: Online/Offine

6. Model Deployment: A/B Testing and/or Exploration/Exploitation when better justified.

7. Model Serving: If time permits then talk about the Training pipeline setup (Data Prep, Feature Prep (Feature Tx + Feat Stores), Model training, Model registry/stores etc.) and Inference pipelines (Batch/Online etc.) along with a continual training pipeline to help prevent model degradation.

8. Model Monitoring (Read more on that here: https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide)

These were some of the excellent resources I referred to in my areas of expertise (Recommenders and Search) for different model architectures:

1. Snap's Spotlight system: https://eng.snap.com/embedding-based-retrieval
2. Instacart Search system: https://tech.instacart.com/how-instacart-uses-embeddings-to-improve-search-relevance-e569839c3c36
3. AirBnb Journey Ranker for Multi-stage ranking: https://arxiv.org/pdf/2305.18431
4. Muti-task RecSys: https://blog.reachsumit.com/posts/2024/01/multi-task-learning-recsys/

Behavioural
- Tell me about yourself.
- Why Meta?
- How do you approach balancing multiple projects and deadlines?
- Give me an example of a project where you used data and machine learning.
- Describe a time you used data to influence a product or business decision.
- Tell me about a time you faced an obstacle and how did you resolve it?
- How do you solve a disagreement with a team member?


