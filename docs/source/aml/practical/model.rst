###########################################################################
Model
###########################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***************************************************************************
Resources
***************************************************************************
- [github.com] `RecSysPapers <https://github.com/tangxyw/RecSysPapers/tree/main/Multi-Task>`_
- [readthedocs.io] `DeepCTR Models API <https://deepctr-doc.readthedocs.io/en/latest/Models.html>`_

***************************************************************************
Retrieval
***************************************************************************
Two Tower
===========================================================================
Int Tower
===========================================================================
***************************************************************************
Ranking
***************************************************************************
DLRM
===========================================================================
DeepFM/DCN
===========================================================================
MTL
===========================================================================
PHASE 1: Foundational Understanding
---------------------------------------------------------------------------
#. Suppose you’re building a ranking model for a job portal. You want to predict both click and apply. Give two principled reasons why modeling these together in an MTL setup is better than training separate models.
#. You're training an MTL model for click and purchase on an ecommerce feed. The purchase signal is extremely sparse and delayed. If you remove the click task and train on purchase alone, the model fails to converge meaningfully. Mechanistically, what role did the click task play in making training work? (Be specific about gradient behavior, representation learning, and optimization landscape.)
#. MTL is often said to "regularize" a sparse task using dense tasks. What does regularization mean in this context, and why is it more effective than traditional methods like L2 or dropout when learning from sparse labels?
#. You're modeling click, like, and comment in a UGC app. You suspect comment is suffering from overfitting. It's sparse, noisy, and high variance across users. How might the click and like tasks stabilize learning for comment? Be precise about how this affects representations, gradients, and generalization.

PHASE 2: MTL Design Choices
---------------------------------------------------------------------------
Architecture Design
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
- Shared-bottom vs Task-Specific Towers

	#. Why might shared-bottom work well for a pair like click and like, but poorly for click and purchase? Explain in terms of representation bias, gradient dynamics, and task semantics.
	#. You're using a shared-bottom model for click, like, and comment. You notice that click and like metrics improve, but comment does not — even though comment is semantically close to like. Upon inspection, you find that comment gradients are small and noisy. Explain how this can happen despite semantic proximity, and outline two architectural fixes (other than PCGrad or reweighting).
- Soft parameter sharing – MMoE: Multi-Gate Mixture-of-Experts

	#. In MMoE, what kind of behaviors do experts tend to specialize in during training? Assume tasks are diverse — e.g., click, comment, share, purchase. What properties of the tasks or data influence this specialization?
	#. Give two scenarios where using MMoE might be overkill or even counterproductive compared to a simpler architecture.
	#. You’ve trained an MMoE model with 4 experts and 3 tasks: click, add-to-cart, and purchase. Post-training: click task distributes its gate weights evenly across all experts. purchase uses only Expert 2 and Expert 3. add-to-cart uses Expert 2 most, but sometimes Expert 1. What does this pattern suggest about your task distribution and expert behavior? What would you do next to interpret or optimize further?
- Entropy Regularization on Gates and Task-Conditioned Gating

	#. You’re training MMoE with 4 experts for 6 tasks. After training: One expert is never used by any task (mean gate weight ≈ 0). Two experts are heavily used by all tasks. Performance gains are marginal compared to shared-bottom. What’s going wrong? How would you fix it?
	#. Your gates produce near-uniform outputs across all experts — for all tasks. There’s no clear differentiation between expert usage. What does this suggest about your gating input or network? What architectural changes would you explore?
	#. Your setup has 5 related tasks (click, like, view, hover, share). When trained with MMoE: All gates collapse to one expert. All tasks converge, but show slight overfitting. Expert activations are indistinguishable across tasks. What’s your diagnosis? What would you change in the architecture or task grouping?
- Progressive Layered Extraction (PLE)

	#. You’re modeling click, like, comment, and share for a social video feed. Initial experiments with shared-bottom and MMoE show: High offline CTR, but unstable comment/share training loss, like and click dominate expert usage. Gradient traces reveal interference in early shared layers. You decide to switch to PLE. How would you structure the first PLE layer to prevent early-stage interference while maintaining representation sharing? Be explicit about expert counts, sharing logic, and gate behavior.
	#. After deploying a 2-layer PLE model, you see the following patterns in logs: Shared experts are heavily used in both layers by all tasks Task-specific experts are barely used click and like are improving, comment and share plateau PCA of task expert outputs shows strong overlap with shared expert space What might be happening? How would you adjust the architecture or training to fix it?
	#. You're scaling a PLE model from 4 tasks to 12 tasks (some sparse, some dense). Your infra supports deeper models but limits total parameter count. What architectural trade-offs would you make in PLE design to scale effectively? Mention expert sharing, gate complexity, and head strategy.

PHASE 3: Data Pipeline Decisions
---------------------------------------------------------------------------
- Label Uncertainty Weighting
- Sampling Strategies

PHASE 4: Learning Dynamics and Stabilization
---------------------------------------------------------------------------
Loss Balancing Strategies
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
- GradNorm

	#. If a task has a very small loss, but its gradient norm w.r.t shared parameters is very large, what will GradNorm do to its weight?
	#. You noticed your training is unstable after enabling GradNorm. Upon inspection: Some task gradient norms are nearly zero. Others are >100. GradNorm loss explodes periodically. What’s going wrong? How do you fix it?
	#. Your model has already task-specific towers. Shared layers are shallow (1–2 layers). Loss curves for all tasks are stable, but task A converges slower. Would you still use GradNorm? Why or why not?
- Manual Gating / Scheduling / Curriculum sampling

	#. You’re training click, like, and purchase in a shared-bottom model. click is dense, purchase is sparse and noisy. You don’t want to use GradNorm. Which two strategies would you combine to stabilize training? Why?
	#. You’re seeing high variance in share and comment task loss. You suspect feedback quality is inconsistent. What can you do to prevent these tasks from hurting shared layers?

PHASE 5: Debugging and Failure Modes
---------------------------------------------------------------------------
#. You notice that add-to-cart (ATC) AUC has dropped 4% in the past 7 days, but: click and like are stable. Online CTR is flat. No model release happened during this window. What’s your diagnosis plan? What would you check first?
#. Your 2-layer PLE model’s logs show: Layer 1 gate usage: evenly distributed. Layer 2 gate usage: over 90% of all tasks route to Expert 1. Tasks share, comment, and purchase have degraded 2–3% in offline metrics. What’s likely happening? What actions would you take?
#. You run a shadow evaluation on purchase. You find that: Precision@top-1k is up 2%. But calibrated probability bucket (0.9–1.0) only converts ~10% of the time. Historical value was ~40%. What’s the likely issue? How would you fix it without retraining?

PHASE 6: Scaling, Shadow Evaluation & Monitoring Infra
---------------------------------------------------------------------------
- Shadow Evaluation

	#. Your shadow logs show that for task share: Logit distribution is highly peaked near 1.0 (many predictions > 0.9). Actual share rate in top-1k scored items is only ~5%. This task was not recently retrained. Other tasks show no drift. What’s going wrong? How do you fix it?
	#. In your MMoE shadow logs: Expert 3 receives ~70% of the gate weight across all tasks. Previously in training, expert usage was balanced. CTR is unchanged, but CVR degraded. What’s your diagnosis and next action?
	#. In your PLE shadow setup: Task comment logits are stuck near 0.5 (low confidence). Gate logs show gating still routes to correct experts. Feedback volume is high, but precision/AUC degraded. No code change in model. What specific logging or feature traces would you inspect next?

- Task Drift Detection
- Expert Usage Logs
- Task-Specific A/B Surfacing
- Serving Time Routing Checks
- Scheduled Refresh / Fine-Tune Strategy

BST
===========================================================================
