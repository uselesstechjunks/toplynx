###############################################################################
Designing Systems
###############################################################################
*******************************************************************************
Resources
*******************************************************************************
- [seangoedecke.com] `Everything I know about good system design <https://www.seangoedecke.com/good-system-design/>`_

*******************************************************************************
Product Systems
*******************************************************************************
Domain / Data Model
===============================================================================
- entities & relationships
- Huyen: Ch.3 Data Engineering Fundamentals (data models: relational/NoSQL; dataflow). 
- DDIA: 

  - Ch.2 Data Models & Query Languages; 
  - Ch.4 Encoding & Evolution (schemas/versioning). 
  - (Optionally Ch.3 Storage & Retrieval for indexes). 

Data Plane
===============================================================================
- ingestion, storage, batch/stream, features
- Huyen: 

  - Ch.3 Data Engineering Fundamentals; 
  - Ch.4 Training Data; 
  - Ch.10 Infrastructure & Tooling for MLOps. 
- DDIA: 
  
  - Ch.3 Storage & Retrieval; 
  - Ch.5 Replication; 
  - Ch.10 Batch Processing; 
  - Ch.11 Stream Processing. 

Intelligence Plane
===============================================================================
- modeling, embeddings, serving
- Huyen: 

  - Ch.5 Feature Engineering; 
  - Ch.6 Model Dev & Offline Eval; 
  - Ch.7 Model Deployment & Prediction Service; 
  - Ch.9 Continual Learning & Test in Prod. 
- DDIA: (Not ML-specific) see Ch.10–11 for batch/stream infra that underpins training/inference. 

Product Plane
===============================================================================
- surfaces/APIs, UX hooks to models
- Huyen: 

  - Ch.7 Prediction Service; 
  - Ch.11 The Human Side of ML (UX/consistency). 
- DDIA: 

  - Ch.4 Encoding & Evolution → “Dataflow Through Services: REST & RPC” (API/service contracts). 

Control Plane
===============================================================================
- orchestration, deploys, experiments, monitoring, reliability
- Huyen: 

  - Ch.8 Data Shifts & Monitoring; 
  - Ch.9 Test in Production (A/B, canary); 
  - Ch.10 Infra & Tooling (schedulers, feature/model stores). 
- DDIA: 

  - Ch.1 Reliable, Scalable, Maintainable Apps; 
  - Ch.8 Trouble with Distributed Systems; 
  - Ch.9 Consistency & Consensus (coordination).

*******************************************************************************
LLM Systems
*******************************************************************************
- https://www.systemdesignhandbook.com/guides/llm-system-design/
- https://www.linkedin.com/posts/hoang-van-hao_25-llms-system-design-interview-questions-activity-7396785599728422913-NXbn

*******************************************************************************
Interview Resources
*******************************************************************************
- [hellointerview.com] `Learn System Design in a Hurry <https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction>`_
- [interviewing.io] `12 fundamental (technical) system design concepts <https://interviewing.io/guides/system-design-interview/part-two#12-fundamental-technical-system-design-concepts>`_
- [linkedin] https://www.linkedin.com/posts/bastyajayshenoy_google-amazon-meta-all-want-ml-system-activity-7397976890612961281-WGYy?rcm=ACoAAARRsNsBS0nkhGM_ofPjXrSrx2KAhC69cdU
