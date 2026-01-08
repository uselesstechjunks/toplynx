###########################################################################
Integrity Systems
###########################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

***************************************************************************
Banned Item Sale
***************************************************************************
- Problem: Prevent banned item sale on Marketplace
  - Actors: sellers (users), platform, users
  - Entities: listings
  - Interests: sellers, platform, users
- Assumptions
  - Scale
    - 1M sellers, 50M listings
    - 1M/day new listings
    - Banned rate: 0.02-0.05% -> 200-500/day banned items
  - Labels
    - Review budget: 2k/day (clean labels: banned + not banned)
    - Feedbacks: report listing (illegal/policy violation), report seller (fraud/scam, unresponsive) - 1000/day
  - System behaviour
    - Creation time trigger (RT/NRT-batched) - policy violation checks (banned words, banned objects)
    - Feedback based trigger -> manual review
  - Business metrics
    - Exposure: reduce (number of users exposed to banned items)
    - Review cost: reduce (number of listings correctly banned by ML - increase)
  - Others
    - Banned item list - fixed, country specific
    - Banned listings fall under single policy violation
- Problem type
  - Multi-class classification
  - Metric: per class precision, recall, f1 -> macro precision, recall, f1
- Data
  - Listings
    - content: text (title, description, metadata, tags), images, video
    - context: upload location, upload time
  - Seller
    - user profile (demographics, age, gender, account age)
    - community stats - reputation (#pos feedbacks/total feedbacks), response rate, report rate
    - activity stats - upload time based (time of day, day of week)
    - conversation history with buyers - text messages (last 5 text messages)
- Features
  - categorical -> one hot -> learned embedding -> concat
  - numerical -> normalised (account age), log-transform (stats based)
  - pretrained embeddings -> mBERT/distillBERT, ViT, ViViT
- Learning strategy
  - 2 weeks for data collection
  - 30k clean labelled examples -> upsample rare classes if required
  - Strategies
    - Direct supervision: from 30k clean labels
      - Pros: simple
      - Cons: overfitting risk, regularise by dropout, early stopping
    - Data augmentation: upsample 3X = 100k
      - text: back translation, synonym replacement, masked token pred
      - image: rand augment (crop + resize, flip, blur, noise)
      - video: frame drop, jitter
      - Pros: more robust
      - Cons: need careful augmentation techniques
    - Semi supervision: 10M unlabelled examples + 30k clean labels with consistency/entropy regularisation
      - Mean teacher, ICT, UDA with randaugment, self-training
      - Pros: better generalisation, learns useful embeddings
      - Cons: more resource, complicated training process, requires tuning for semi supervised loss weight
      - If we go for this then a further distillation would be useful
- Model + Training
  - Arch
    - early fusion between modalities for listings for interaction learning
  - Choice
    - concat -> MLP [2-3x] -> classification head (simple)
    - project + concat + MLP -> classification head
    - project + cross-attention + concat + project + classification head
  - Training
    - cross entropy loss, dropout, backprop, frozen pretrained encoder
- Eval
  - Offline - golden eval set
  - Online - live traffic A/B testing
    - random traffic alloc - might not be invokved at all
    - user alloc - users change their profile and try again
    - geo alloc - better reliable
    - baseline: basic keyword based filtering
- Deployment
  - Distributed deployment, horizontal scaling, NRT system with batch
  - Continuous training? Learn from mistakes (items that are banned but missed by the system)
- Monitoring
  - ML metrics, drift metrics
- Improvements
  - Use domain pretrained encoders for different modalities (e.g., encoders for product search)
  - Use proxy labels from LLMs
  - Explore hard negative mining strategies

***************************************************************************
Banned Product Ads
***************************************************************************
- Problem: Banned product ads sale on facebook news feed
- Assumptions
  - Scale
    - 10M advertisers, 100M/day ad creatives (text/image/video)
    - 1B/day ad impression
    - Banned rate: 0.01-0.05%, 10-50k/day
  - Labels
    - Expert labels: 10k/day label budget
    - User flags: 100k/day flagged by users
    - Policy matching
  - System behaviour
    - Submission time queue/block (if high confidence)
    - real-time trigger based filter
  - Business metrics
    - Exposure to banned items
    - Rejection cost
    - Review cost

***************************************************************************
Spam Listings Detection
***************************************************************************
- Problem: You’re building a model to detect scammy product listings.
  - Only 0.2% of listings are manually flagged as scams
  - The flagged data is diverse, but inconsistent (some borderline spam, some extreme abuse)
  - There are many obvious false negatives (scams that went unflagged)
  - You also have an older “suspicion score” model that outputs a confidence ∈ [0, 1] from past rule-based signals

- Signals
  - Hard positive: 0.2% manually flagged scam listings
  - Weak positive: 1% of total listings (diverse, inconsistent, varying intent)
  - False negative: 5% of total listings
  - True negatives: 93.8% of total listings
- Paradigm
  - Since scam categories are not well defined, we have to work with binary labels
  - Binary classification with BCE loss
  - Assume we have label disagreement stats as well (e.g., 66% agreement) for each manually labeled set
#. Phase 1: Train with confidence-based pseudo labels for the entire dataset

   - Just one head at this stage
   - Use older classifier to obtain confidence scores and use as pseudo labels
   - Label issue: contains bias from previous classifier, not variance
   - Regularization techniques to prevent bias transfer:
     - What would work:
       - Loss weights: lower weights (e.g., 0.1) to reduce loss gradient magnitude
       - Question: Do we need to tune loss weights with gold set at this phase as well?
     - What won’t work:
       - Dropout regularization: would have helped if we had hard labels but less diversity (avoids overfitting)
       - Temperature scaling: higher temperature would have helped if we had inconsistent labels for boundary cases (fattens density)
       - Label smoothing: not required as labels are not noisy or of high variance (i.e., not accidentally flagged)
   - This phase should help with false negatives (5%)
#. Phase 2: Train with weak labels from user-flagged positives

   - Still no data for confidence head (since users flag items, there is no action to explicitly mark an item as 'not spam')
   - Still just one head at this stage
   - Use label smoothing \( y_i^s = (1 - \epsilon)y + 0.5\epsilon \) using user-flagged labels
   - Label smoothing would help with diversity and inconsistency in user-flagged items (accounts for variance in labels)
   - Need to tune smoothing parameter with gold set, aiming for precision @ fixed recall
   - Dropout: not useful as labels already cover a diverse set
   - Temperature scaling: lowering might help with inconsistency (if they indicate borderline cases); we shouldn’t use high temperature as labels are weak (used in phase 4)
   - Train for a few epochs until the gain plateaus
#. Phase 3: Train with hard labels from reviewed positives

   - Add hard positives (y = 1) into the training dataset from phase 2 (replace soft labels from previous step if there is overlap)
   - Reduce loss weights for weak positives
   - Split hard positives across batches for stable learning
   - Continue training from phase 2 for a few more epochs until the gain plateaus
#. Phase 4: Train with hard negatives

   - Calibrate output probabilities using validation gold set
   - Tune threshold for recall @ fixed precision
   - Run inference on reviewed set to find high-confidence false negatives
   - Reshuffle batch to include both hard positives and hard negatives in each batch
   - Train with higher temperature; post-training, redo calibration and threshold tuning
#. Phase 5: Add confidence head

   - Add confidence head, freeze everything else
   - Use true disagreement labels for reviewed items
   - To obtain proxy labels for the rest, run inference with three different dropout setups on the training set
   - Use calibrated threshold to find positive and negative predictions; use voting to obtain disagreement score
   - Assign higher loss weight to actual disagreement score, lower loss weight to proxy disagreement score
   - Train until convergence
   - Calibrate disagreement head with gold set
   - Tune disagreement threshold for recall @ fixed precision for manual trigger
