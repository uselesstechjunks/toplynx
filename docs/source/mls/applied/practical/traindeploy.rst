#######################################################################
Training & Deployment
#######################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

***********************************************************************
Training
***********************************************************************
Large-Scale ML & Distributed Training
=======================================================================
- Parallelization: Data parallelism vs model parallelism.
- Gradient Accumulation: Handling large batch sizes.
- Federated Learning: Privacy-preserving distributed learning.
- ML Monitoring & Logging: Model drift detection, feature monitoring, data pipelines.
- Serving at Scale: TFX, Ray Serve, TorchServe, Kubernetes-based deployments.

Fine-Tuning & LLMs
=======================================================================
- Efficient Fine-Tuning: LoRA, QLoRA, adapters, prompt tuning.
- Memory-Efficient Training: Flash Attention, ZeRO Offloading, activation checkpointing.
- Inference Optimization: KV caching, speculative decoding, grouped-query attention.
- Long-Context Adaptation: RoPE interpolation, Hyena operators, recurrent memory transformers.
- Safety & Alignment: RLHF, constitutional AI, preference tuning.

***********************************************************************
Deployment
***********************************************************************
Model Productionization & Scaling
=======================================================================
- Latency vs Accuracy Tradeoffs: Quantization, distillation, pruning.
- Efficient Inference: TensorRT, ONNX, model sharding, mixed precision training.
- Retraining Strategies: Online learning, active learning, incremental updates.
- Data Drift and Concept Drift: Detection techniques, adaptive retraining pipelines.
- A/B Testing and Shadow Deployment: Canary rollouts, offline vs online evaluation.

Applied Causal Inference & Uplift Modeling
=======================================================================
- Causal ML in Production: A/B testing pitfalls, Simpson's paradox.
- Uplift Modeling: Net lift estimation for interventions.
- DoWhy & Causal Discovery: Counterfactual analysis in ML pipelines.

Retraining
=======================================================================
#. How often to retrain?
   
   - Depends on drift: Frequent updates if data shifts, otherwise periodic (weekly, monthly, quarterly).
#. Periodic vs Continuous Training?

   - Periodic: Easier to manage, avoids instability.
   - Continuous: Needed when real-time adaptation is required (e.g., dynamic pricing, recommendation systems).
