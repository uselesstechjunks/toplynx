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
