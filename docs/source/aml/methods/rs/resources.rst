####################################################################################
Resources
####################################################################################
************************************************************************************
Survey
************************************************************************************
* [arxiv.org][Retrieval] `A Comprehensive Survey on Retrieval Methods in Recommender Systems <https://arxiv.org/pdf/2407.21022>`_
* [ijcai.org][UBM] `A Survey on User Behavior Modeling in Recommender Systems <https://www.ijcai.org/proceedings/2023/0746.pdf>`_
* [arxiv.org][CTR] `Deep Learning for Click-Through Rate Estimation <https://arxiv.org/abs/2104.10584>`_
* [arxiv.org][SSL] `Self-Supervised Learning for Recommender Systems: A Survey <https://arxiv.org/abs/2203.15876>`_
* [arxiv.org][Embedding] `Embedding in Recommender Systems: A Survey <https://arxiv.org/abs/2310.18608v2>`_
* [arxiv.org][CTR] `Click-Through Rate Prediction in Online Advertising: A Literature Review <https://arxiv.org/abs/2202.10462>`_
* [le-wu.com][Ranking] `A Survey on Accuracy-Oriented Neural Recommendation <https://le-wu.com/files/Publications/JOURNAL/A_Survey_of_Neural_Recommender_Systems.pdf>`_
* [mdpi.com] `A Comprehensive Survey of Recommender Systems Based on Deep Learning <https://www.mdpi.com/2076-3417/13/20/11378/pdf?version=1697524018>`_
* [youtube.com] `TUTORIAL: Neural Contextual Bandits for Personalized Recommendation <https://www.youtube.com/watch?v=esOd-tsdEco>`_ (`arxiv <https://arxiv.org/pdf/2312.14037>`_)
* [youtube.com] `TUTORIAL: Privacy in Web Advertising: Analytics and Modeling <https://www.youtube.com/watch?v=qaiDxriCEmQ>`_
* [youtube.com] `TUTORIAL: Multimodal Pretraining and Generation for Recommendation <https://www.youtube.com/watch?v=Pw1eW0rMzSU>`_
* [youtube.com] `TUTORIAL: Large Language Models for Recommendation Progresses and Future Directions <https://www.youtube.com/watch?v=zcuOrWxJ2k8>`_

************************************************************************************
Metrics & QA
************************************************************************************
.. important::

	* [evidentlyai.com] `10 metrics to evaluate recommender and ranking systems <https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems>`_
	* [docs.evidentlyai.com] `Ranking metrics <https://docs.evidentlyai.com/reference/all-metrics/ranking-metrics>`_
	* [arize.com] `A Quick Survey of Drift Metrics <https://arize.com/blog-course/drift/>`_
	* [github.com] `50 Fundamental Recommendation Systems Interview Questions <https://github.com/Devinterview-io/recommendation-systems-interview-questions>`_
	* [devinterview.io] `50 Recommendation Systems interview questions <https://devinterview.io/questions/machine-learning-and-data-science/recommendation-systems-interview-questions/>`_

************************************************************************************
Videos
************************************************************************************
- [youtube.com] `Stanford CS224W Machine Learning w/ Graphs I 2023 I GNNs for Recommender Systems <https://www.youtube.com/watch?v=OV2VUApLUio>`_
.. note::
	- Mapped as an edge prediction problem in a bipartite graph
	- Ranking

		- Metric Recall@k (non differentiable)
		- Other metrics HR@k, nDCG
		- Differentiable Discriminative loss - binary loss (similar to cross entropy), Bayesian prediction loss (BPR)
		- Issue with binary, BPR solves the ranking problem better
		- Trick to choose neg samples
		- Not suitable for ANN
	- Collaborative filtering

		- DNN to capture user item similarity with cosine or InfoNCE loss
		- ANN friendly
		- Doesn't consider longer than 1 hop in the bipartite graph
	- GCN

		- Smoothens the embeddings by GCN layer interactions using undirected edges to enforce similar user and similar item signals
		- Neural GCN or LightGCN
		- Application similar image recommendation in Pinterest
		- Issue doesn't have contextual awareness or session/temporal awareness

************************************************************************************
Course, Books & Papers
************************************************************************************
CTR Prediction Papers
====================================================================================
- [paperswithcode.com] `CTR Prediction <https://paperswithcode.com/task/click-through-rate-prediction>`_

.. csv-table::
	:header: "Technique", "Resource"
	:align: center

		LR, `Distributed training of Large-scale Logistic models <https://proceedings.mlr.press/v28/gopal13.pdf>`_		
		Embed + MLP, `Deep Neural Networks for YouTube Recommendations <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf>`_
		Embed + MLP, `Real-time Personalization using Embeddings for Search Ranking at Airbnb <https://dl.acm.org/doi/pdf/10.1145/3219819.3219885>`_
		Wide & Deep, `Wide & Deep Learning for Recommender Systems <https://arxiv.org/abs/1606.07792>`_
		DeepFM, `DeepFM: A Factorization-Machine based Neural Network for CTR Prediction <https://arxiv.org/abs/1703.04247>`_
		xDeepFM, `xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems <https://arxiv.org/abs/1803.05170>`_
		DCN, `Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_
		DCNv2, `DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems <https://arxiv.org/abs/2008.13535>`_
		DIN, `Deep Interest Network for Click-Through Rate Prediction <https://arxiv.org/abs/1706.06978>`_
		BST, `Behavior Sequence Transformer for E-commerce Recommendation in Alibaba <https://arxiv.org/abs/1905.06874>`_

Embeddings Papers
====================================================================================
.. csv-table::
	:header: "Technique", "Resource"
	:align: center

		Hash, `Hash Embeddings for EfficientWord Representations <https://arxiv.org/abs/1709.03933>`_
		Deep Hash, `Learning to Embed Categorical Features without Embedding Tables for Recommendation <https://arxiv.org/abs/2010.10784>`_
		Survey, `Embedding in Recommender Systems: A Survey <https://arxiv.org/abs/2310.18608v2>`_

Modeling Methods Papers
====================================================================================
	- BOF = Bag of features
	- NG = N-Gram
	- CM = Causal Models (autoregressive)

.. csv-table::
	:header: "Tag", "Title"
	:align: center

		QU;Search,`Better search through query understanding <https://queryunderstanding.com/>`_
		IR;QU;Search,`Using Query Contexts in Information Retrieval <http://www-rali.iro.umontreal.ca/rali/sites/default/files/publis/10.1.1.409.2630.pdf>`_
		IR;Course;Stanford,`CS 276 / LING 286 Information Retrieval and Web Search <https://web.stanford.edu/class/cs276/>`_
		IR;Book,`Introduction to Information Retrieval <https://nlp.stanford.edu/IR-book/information-retrieval-book.html>`_
		Retrival;RS,`Simple but Efficient A Multi-Scenario Nearline Retrieval Framework for Recommendation on Taobao <https://arxiv.org/pdf/2408.00247v1>`_
		Retrival;Ranking;Embed+MLP,`Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_
		Retrival;Two Tower;BOF,`StarSpace Embed All The Things! <https://arxiv.org/abs/1709.03856>`_
		Retrival;Ranking;Two Tower;NG+BOF,`Embedding-based Retrieval in Facebook Search <https://arxiv.org/abs/2006.11632>`_		
		Ranking;L2R,`DeepRank: Learning to rank with neural networks for recommendation <http://zhouxiuze.com/pub/DeepRank.pdf>`_
		GCN,`Graph Convolutional Neural Networks for Web-Scale Recommender Systems <https://arxiv.org/abs/1806.01973>`_
		GCN,`LightGCN - Simplifying and Powering Graph Convolution Network for Recommendation <https://arxiv.org/abs/2002.02126>`_
		CM;Session,`Transformers4Rec Bridging the Gap between NLP and Sequential / Session-Based Recommendation <https://scontent.fblr25-1.fna.fbcdn.net/v/t39.8562-6/243129449_615285476133189_8760410510155369283_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=b8d81d&_nc_ohc=WDJcULkgkY8Q7kNvgHspPmM&_nc_zt=14&_nc_ht=scontent.fblr25-1.fna&_nc_gid=A_fmEzCPOHil7q9dPSpYsHS&oh=00_AYDCkVOnyZufYEGHEQORBbfI-blNODNIrePL4TaB8p_82A&oe=67A8FEDE>`_			
		Diversity;DPP,`Improving the Diversity of Top-N Recommendation via Determinantal Point Process <https://arxiv.org/abs/1709.05135v1>`_
		Diversity;DPP,`Practical Diversified Recommendations on YouTube with Determinantal Point Processes <https://jgillenw.com/cikm2018.pdf>`_
		Diversity;DPP,`Personalized Re-ranking for Improving Diversity in Live Recommender Systems <https://arxiv.org/abs/2004.06390>`_
		Diversity;DPP,`Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity <https://proceedings.neurips.cc/paper_files/paper/2018/file/dbbf603ff0e99629dda5d75b6f75f966-Paper.pdf>`_
		Diversity;Multi-Stage,`Representation Online Matters Practical End-to-End Diversification in Search and Recommender Systems <https://arxiv.org/pdf/2305.15534>`_
		Polularity Bias,`Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking <https://cdn.aaai.org/ocs/18199/18199-78818-1-PB.pdf>`_
		Polularity Bias,`User-centered Evaluation of Popularity Bias in Recommender Systems <https://dl.acm.org/doi/fullHtml/10.1145/3450613.3456821>`_
		Polularity Bias,`Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System <https://arxiv.org/pdf/2010.15363>`_
		Fairness,`Fairness in Ranking Part II Learning-to-Rank and Recommender Systems <https://dl.acm.org/doi/pdf/10.1145/3533380>`_
		Fairness,`Fairness Definitions Explained <https://fairware.cs.umass.edu/papers/Verma.pdf>`_
		LLM,`A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys) <https://arxiv.org/abs/2404.00579>`_
		LLM,`Collaborative Large Language Model for Recommender Systems <https://arxiv.org/abs/2311.01343>`_
		LLM,`Recommendation as Instruction Following A Large Language Model Empowered Recommendation Approach <https://arxiv.org/abs/2305.07001>`_

More Papers
====================================================================================
.. csv-table::
	:header: "Year", "Title"
	:align: center

		2001,Item-Based Collaborative Filtering Recommendation Algorithms – Sarwar et al.
		2003,Amazon.com Recommendations Item-to-Item Collaborative Filtering – Linden et al.
		2007,Link Prediction Approaches and Applications – Liben-Nowell et al.
		2008,An Introduction to Information Retrieval – Manning et al.
		2009,BM25 and Beyond – Robertson et al.
		2009,Matrix Factorization Techniques for Recommender Systems – Koren et al.
		2010,Who to Follow Recommending People in Social Networks – Twitter Research
		2014,DeepWalk Online Learning of Social Representations – Perozzi et al.
		2015,Learning Deep Representations for Content-Based Recommendation – Wang et al.
		2015,Netflix Recommendations Beyond the 5 Stars – Gomez-Uribe et al.
		2016,Deep Neural Networks for YouTube Recommendations – Covington et al.
		2016,Wide & Deep Learning for Recommender Systems – Cheng et al.
		2016,Session-Based Recommendations with Recurrent Neural Networks – Hidasi et al.
		2017,DeepRank A New Deep Architecture for Relevance Ranking in Information Retrieval – Pang et al.
		2017,Neural Collaborative Filtering – He et al.
		2017,A Guide to Neural Collaborative Filtering – He et al.
		2018,BERT Pre-training of Deep Bidirectional Transformers for Language Understanding – Devlin et al.
		2018,PinSage Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2018,Neural Architecture for Session-Based Recommendations – Tang & Wang
		2018,SASRec Self-Attentive Sequential Recommendation – Kang & McAuley
		2018,Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2019,Deep Learning Based Recommender System A Survey and New Perspectives – Zhang et al.
		2019,Session-Based Recommendation with Graph Neural Networks – Wu et al.
		2019,Next Item Recommendation with Self-Attention – Sun et al.
		2019,BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations – Sun et al.
		2020,Dense Passage Retrieval for Open-Domain Question Answering – Karpukhin et al.
		2020,ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT – Khattab et al.
		2020,T5 for Information Retrieval – Nogueira et al.
		2021,CLIP Learning Transferable Visual Models from Natural Language Supervision – Radford et al.
		2021,Transformers4Rec Bridging the Gap Between NLP and Sequential Recommendation – De Souza et al.
		2021,Graph Neural Networks A Review of Methods and Applications – Wu et al.
		2021,Next-Item Prediction Using Pretrained Language Models – Sun et al.
		2022,Unified Vision-Language Pretraining for E-Commerce Recommendations – Wang et al.
		2022,Contextual Item Recommendation with Pretrained LLMs – Li et al.
		2023,InstructGPT for Information Retrieval – Ouyang et al.
		2023,GPT-4 for Web Search Augmentation – Bender et al.
		2023,CLIP-Recommend Multimodal Learning for E-Commerce Recommendations – Xu et al.
		2023,Semantic-Aware Item Matching with Large Language Models – Chen et al.
		2023,GPT4Rec A Generative Framework for Personalized Recommendation – Wang et al.
		2023,LLM-based Collaborative Filtering Enhancing Recommendations with Large Language Models – Liu et al.
		2023,LLM-Powered Dynamic Personalized Recommendations – Guo et al.
		2023,Real-Time Recommendation with Large Language Models – Zhang et al.
		2023,Graph Neural Networks Meet Large Language Models A Survey – Wu et al.
		2023,LLM-powered Social Graph Completion for Friend Recommendations – Huang et al.
		2023,LLM-Augmented Node Classification in Social Networks – Zhang et al.
