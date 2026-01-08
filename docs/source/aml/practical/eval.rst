######################################################################################
Evaluation
######################################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***********************************************************************
Offline Evaluation
***********************************************************************
Golden Set Curation
=======================================================================
- Criteria for selection

	- Coverage: Includes all relevant feature distributions.
	- Accuracy: Labels verified by experts.
	- Diversity: Edge cases, rare conditions.
- Update frequency?
   
	- Periodically (e.g., quarterly) or when drift is detected.
- How to balance representation?

	- Maintain real-world distribution while oversampling rare cases.

Metric
=======================================================================
- Basic Building Blocks

	- TP = True Positives  
	- FP = False Positives  
	- FN = False Negatives  
	- TN = True Negatives  
- From these:

	- Precision = TP / (TP + FP)  
	- Recall = TP / (TP + FN)  
	- F1-score = 2 × (Precision × Recall) / (Precision + Recall)

Binary Classification
----------------------------------------------------------------
- ROC-AUC: Measures ability to distinguish classes across all thresholds; useful when class balance is not extreme.
- PR-AUC: Focuses on positive class performance (precision vs recall); useful when positives are rare.
- When to prefer ROC-AUC vs PR-AUC?

	- ROC-AUC: When positives and negatives are balanced.
	- PR-AUC: When positives are rare (e.g., fraud detection, rare disease prediction).

Multi-class Classification (1 label/sample)
----------------------------------------------------------------
- :math:`n_i` is the number of true instances for class i, and :math:`N` is the total number of samples.
- Accuracy = (Number of correct predictions) / (Total samples)
- Macro Precision = average of per-class precision  

	- .. math:: \text{MacroPrecision} = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FP_i}
- Macro Recall = average of per-class recall  

	- .. math:: \text{MacroRecall} = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FN_i}
- Macro F1 = average of per-class F1  

	- .. math:: \text{MacroF1} = \frac{1}{C} \sum_{i=1}^{C} \text{F1}_i
- Weighted F1  

	- .. math:: \text{WeightedF1} = \sum_{i=1}^{C} \frac{n_i}{N} \cdot \text{F1}_i

Multi-label Classification (multiple labels/sample)
----------------------------------------------------------------
Let :math:`Y_i` be the true labels and :math:`\hat{Y}_i` the predicted labels for instance i.

- Subset Accuracy (Exact Match)  

	.. math:: \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(Y_i = \hat{Y}_i)
- Micro Precision / Recall / F1 (aggregate TP/FP/FN across all labels)  
  
	- .. math:: \text{MicroPrecision} = \frac{\sum_{l} TP_l}{\sum_{l} (TP_l + FP_l)}
	- .. math:: \text{MicroRecall} = \frac{\sum_{l} TP_l}{\sum_{l} (TP_l + FN_l)}
	- .. math:: \text{MicroF1} = 2 \cdot \frac{\text{MicroPrecision} \cdot \text{MicroRecall}}{\text{MicroPrecision} + \text{MicroRecall}}
- Macro F1 (average across labels)  

	- .. math:: \text{MacroF1} = \frac{1}{L} \sum_{l=1}^{L} \text{F1}_l

Multi-task Binary Classification (1 binary label per task)
----------------------------------------------------------------
Let there be T tasks. Each task is a binary classification problem.

- For each task, compute:

  - :math:`Accuracy\_t = (TP + TN) / (TP + FP + TN + FN)`
  - :math:`AUC\_t`, :math:`F1\_t` (as per binary classification)
- MacroAUC / MacroF1 (across tasks)  

  .. math:: \text{MacroF1}_{\text{tasks}} = \frac{1}{T} \sum_{t=1}^{T} \text{F1}_t

Slice-based Performance Evaluation
=======================================================================
How to choose slices for evaluation?

	- Numerical features: Quantile-based bins (e.g., age groups).
	- Categorical features: Stratify by value distribution.
	- Temporal features: Time-based slices (e.g., recent vs past).
	- Edge cases: Identify rare but critical scenarios.

When is a model ready for production?

	- Stable performance across test & validation sets.
	- Performs better than baseline (existing model or heuristic).
	- Low failure rate in stress tests (edge cases, adversarial inputs).

Model Evaluation Beyond AUC
=======================================================================
- Calibration: Platt scaling, isotonic regression.
- Expected Calibration Error (ECE): Ensuring confidence scores are well-calibrated.
- Robustness Testing: Adversarial robustness, stress testing with synthetic data.

***********************************************************************
Online Evaluation
***********************************************************************
- A/B testing - `Interleaving <https://www.amazon.science/publications/interleaved-online-testing-in-large-scale-systems>`_ vs non-interleaving
- Bayesian A/B testing
- Metrics

	- Resource: [transparency.meta.com] `How Meta improves <https://transparency.meta.com/en-gb/policies/improving/>`_

**************************************************************************************
LLM App Evaluation
**************************************************************************************
Practical
=========================================================================================
* [github.com] `The LLM Evaluation guidebook <https://github.com/huggingface/evaluation-guidebook>`_
* [confident.ai] `LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide <https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation>`_
* [confident-ai.com] `How to Evaluate LLM Applications: The Complete Guide <https://www.confident-ai.com/blog/how-to-evaluate-llm-applications>`_
* [arize.com] `The Definitive Guide to LLM App Evaluation <https://arize.com/llm-evaluation/overview/>`_
* [arize.com] `RAG Evaluation <https://arize.com/blog-course/rag-evaluation/>`_
* [guardrailsai.com] `Guardrails AI Docs <https://www.guardrailsai.com/docs>`_

Academic
=========================================================================================
* [acm.org] `A Survey on Evaluation of Large Language Models <https://dl.acm.org/doi/pdf/10.1145/3641289>`_
* [arxiv.org] `The Responsible Foundation Model Development Cheatsheet: A Review of Tools & Resources <https://arxiv.org/abs/2406.16746>`_
* [arxiv.org] `Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering <https://arxiv.org/pdf/2101.00774>`_
* Evaluation of instruction tuned/pre-trained models

	* MMLU

		* [arxiv.org] `Measuring Massive Multitask Language Understanding <https://arxiv.org/pdf/2009.03300>`_
		* Dataset: https://huggingface.co/datasets/cais/mmlu
	* Big-Bench

		* [arxiv.org] `Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models <https://arxiv.org/pdf/2206.04615>`_
		* Dataset: https://github.com/google/BIG-bench
