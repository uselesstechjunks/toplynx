#####################################################################
Practice Problems
#####################################################################
.. note::
For each, aim to hit: 

  (1) clarify scope (DAU, latency budget, label availability), 
  (2) decompose the system (data flywheel → offline training → serving → monitoring), 
  (3) make explicit tradeoffs rather than just stating choices, 
  (4) name what you'd measure and how online/offline metrics can diverge. 

Staff-level bar specifically rewards knowing why a design decision breaks down at scale and what the next failure mode would be.

*********************************************************************
1. Ranking & Retrieval
*********************************************************************
1. Design a feed ranking system for a social platform with 500M DAU. Walk through candidate generation, feature engineering, model architecture, and how you'd handle cold start, position bias, and feedback loops. Probes: Two-tower vs. cross-attention tradeoffs, online/offline metric alignment, exploration-exploitation, consistency at scale.
2. Design a semantic search system for a large document corpus. How would you handle embedding freshness, approximate nearest neighbor tradeoff vs. recall, and reranking? Probes: ANN index choices (HNSW vs. IVF), embedding model selection, hybrid sparse-dense retrieval, latency SLAs.

*********************************************************************
2. Ads & Monetization
*********************************************************************
1. Design a click-through rate (CTR) prediction system. How would you handle feature sparsity, embedding collision, training/serving skew, and calibration? Probes: DLRM-style architectures, hash trick vs. vocabulary, real-time feature pipelines, isotonic regression for calibration.
2. Design a bidding system that jointly optimizes for advertiser ROI and platform revenue. Probes: Constrained optimization, multi-objective reward shaping, counterfactual estimation, VCG vs. GSP auction dynamics.

*********************************************************************
3. Recommendations
*********************************************************************
1. Design a video recommendation system. How do you avoid filter bubbles, handle multi-objective optimization (engagement vs. satisfaction vs. diversity), and measure long-term user value? Probes: Surrogate metrics vs. north star, exploration policies, sequence modeling, multi-task learning heads.

*********************************************************************
4. Abuse / Safety
*********************************************************************
1. Design a real-time content moderation system for user-generated text and images. How do you handle adversarial actors, concept drift, and the cost asymmetry between false positives and false negatives? Probes: Multi-stage classifiers, human-in-the-loop integration, policy-model coupling, severity tiering.
2. Design a system to detect coordinated inauthentic behavior (fake accounts / bots) at scale. Probes: Graph-based features vs. behavioral sequences, streaming inference, adversarial robustness, labeling pipelines.

*********************************************************************
5. ML Platform / Infrastructure
*********************************************************************
1. Your team's production model has silent accuracy degradation over 6 months. Design an end-to-end system to detect, diagnose, and remediate model drift. Probes: Population stability index, covariate vs. concept drift, shadow models, rollback strategy, continuous training pipelines.
2. Design a hyperparameter optimization and model selection platform for hundreds of concurrent training jobs across multiple teams. Probes: Bayesian optimization vs. successive halving, resource scheduling, reproducibility, experiment tracking at scale.


