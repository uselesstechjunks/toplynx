########################################################################
Important Issues
########################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

************************************************************************
Cold Start & Long-Tail Handling
************************************************************************
- Use pretrained content encoders
- Use popularity prior & meta-learned embeddings
- Rely on seller/group identity for long-tail bootstrapping

************************************************************************
Data Labeling and Feedback Loops
************************************************************************
- Weakly supervised methods
- Use LLMs to generate labels from noisy data
- Online learning or continual training pipelines

************************************************************************
System Design Aspects
************************************************************************
- Real-time embedding update pipelines
- Feature freshness (e.g., contact count in last hour)
- Ranking latency vs. relevance trade-offs
- Efficient ANN index sharding
