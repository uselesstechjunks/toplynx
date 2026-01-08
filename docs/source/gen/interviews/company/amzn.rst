##########################################################################
Amazon
##########################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

**************************************************************************
Leadership Principles
**************************************************************************
.. csv-table:: 
	:header: "Rank","Vertical","Desc","Relevancy"
	:align: center

		5,2. **Ownership**:,Think long-term; act on behalf; embrace responsibility.,Taking ownership ensures accountability and long-term strategic thinking; which are essential for developing and deploying ML models.
		5,3. **Invent and Simplify**:,Innovate; simplify; encourage external ideas.,Innovation and simplification are core to developing effective ML algorithms and systems.
		5,5. **Learn and Be Curious**:,Continuous learning; explore new ideas.,Continuous learning is critical in the rapidly evolving field of machine learning.
		5,11. **Earn Trust**:,Listen; speak candidly; benchmark against best.,Trust is crucial for collaboration and acceptance of ML solutions.
		5,12. **Dive Deep**:,Stay connected to details; skeptical of metrics.,Understanding technical details deeply is essential for effective ML modeling and problem-solving.
		5,14. **Deliver Results**:,Focus on key inputs; timely delivery; quality.,Ultimately; delivering results is the goal of applied machine learning projects.
		4,1. **Customer Obsession**:,Start with customer; earn trust; obsess over needs,Understanding and addressing customer needs is crucial for developing impactful machine learning solutions.
		4,4. **Are Right; A Lot**:,Strong judgment; seek diverse perspectives.,Having strong judgment and seeking diverse perspectives aids in making informed decisions in ML research and application.
		4,7. **Insist on the Highest Standards**:,High expectations; drive quality.,High standards ensure quality in ML model development and deployment.
		4,8. **Think Big**:,Bold vision; inspire innovation; think differently.,Thinking big helps in envisioning impactful applications of machine learning.
		4,9. **Bias for Action**:,Value speed; encourage calculated risk.,Speed in experimentation and prototyping is beneficial in agile ML development.
		4,13. **Have Backbone; Disagree and Commit**:,Challenge respectfully; commit wholly.,Respectful disagreement and commitment to decisions are important in collaborative ML research.
		3,6. **Hire and Develop the Best**:,Raise performance; develop leaders; coach.,While important; direct leadership of teams may be less relevant compared to technical expertise in this role.
		3,16. **Success and Scale Bring Broad Responsibility**:,Be humble; impact positively; improve daily.,Awareness of broader impacts is important; though less directly tied to day-to-day ML tasks.
		2,10. **Frugality**:,Resourceful; accomplish more with less.,Resourcefulness is valuable; but excessive frugality may hinder innovation in machine learning.
		2,15. **Strive to be Earth’s Best Employer**:,Create safe; productive; empathetic environment.,While a positive work environment is important; the direct impact on machine learning work may be indirect.

**************************************************************************
Kindle
**************************************************************************
Understanding the Tech Stack
==========================================================================
22/07: LLM long-context model capabilities awareness

	- Review `Amazon Bedrock User Guide <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>`_ for models: `Claude <https://aws.amazon.com/bedrock/claude/>`_ (Anthropic), `Titan <https://aws.amazon.com/bedrock/titan/>`_ (Amazon), and `Jurassic-2 <https://aws.amazon.com/bedrock/ai21/>`_ (AI21 Labs)
	- Study long-context capabilities: context window sizes, attention mechanisms
	- Explore techniques like sliding window attention, sparse attention, and recursive chunking
	- Understand the implications of long-context models for book analysis and summarization

23/07: LLM eval and pitfall awareness

	- Study evaluation metrics: perplexity, BLEU, ROUGE, BERTScore
	- Explore `Amazon's SageMaker <https://aws.amazon.com/sagemaker/>`_ for model evaluation
	- Review common LLM pitfalls: hallucination, bias, toxicity
	- Understand techniques for prompt engineering and few-shot learning
	- Study ethical considerations in LLM deployment

24/07: e2e ML pipeline awareness

	- Review Amazon SageMaker for end-to-end ML workflows
	- Study data preprocessing techniques for text data
	- Explore model fine-tuning strategies on Amazon Bedrock
	- Understand deployment options: batch inference vs. real-time inference
	- Review monitoring and maintenance of ML models in production

25/07: Brainstorm on key potential applications

	- Retrieval Augmented Generation (RAG) for enhanced book recommendations
	- Few-shot learning for genre classification
	- Zero-shot learning for content moderation
	- Multi-task learning for simultaneous summary generation and sentiment analysis
	- Explore potential applications of diffusion models for book cover generation
	- Consider state-space models for time series analysis of reading patterns

26/07: Design actual systems, review all previous days, code for transformer

	- Design a complete system for automated book categorization using Amazon Bedrock
	- Review key concepts from previous days
	- Implement a basic transformer model using PyTorch or TensorFlow
	- Explore Amazon SageMaker's built-in algorithms for NLP tasks
	- Study integration of custom models with Amazon Bedrock

Additional topics to consider throughout:

	1. `Stable Diffusion <https://aws.amazon.com/bedrock/stable-diffusion/>`_ for image generation tasks
	2. `Amazon Textract <https://aws.amazon.com/textract/>`_ for extracting text and data from scanned documents
	3. `Amazon Comprehend <https://aws.amazon.com/comprehend/>`_ for natural language processing tasks
	4. `Amazon Polly <https://aws.amazon.com/polly/>`_ for text-to-speech capabilities
	5. `Amazon Kendra <https://aws.amazon.com/kendra/>`_ for intelligent search applications

LLM design patterns to explore:

	1. In-context learning and prompt engineering
	2. Chain-of-thought prompting for complex reasoning tasks
	3. Constitutional AI for safer and more controlled LLM outputs
	4. Retrieval-augmented generation (RAG) for grounding LLMs in factual data
	5. Fine-tuning strategies for domain-specific tasks

Understanding the Domain
==========================================================================
Resources
--------------------------------------------------------------------------
- How Many Words in a Novel? `reedsy <https://blog.reedsy.com/how-many-words-in-a-novel/>`_, `thewritelife <https://thewritelife.com/how-many-words-in-a-novel/>`_

.. csv-table:: 
	:header: "Genre","Min Word Count","Max Word Count"
	:align: center

		Flash Fiction,300,1500
		Short Story,1500,30000
		Novellas,30000,50000
		Novels,50000,110000
		Mainstream Romance,70000,100000
		Subgenre Romance,40000,100000
		Science Fiction / Fantasy,90000,150000
		Historical Fiction,80000,100000
		Thrillers / Horror / Mysteries / Crime,70000,90000
		Young Adult,50000,80000
		Picture Books,300,800
		Early Readers,200,3500
		Chapter Books,4000,10000
		Middle Grade,25000,40000
		Standard Nonfiction,70000,80000
		Memoir,80000,100000
		Biography,80000,200000
		Self-Help,40000,50000

Existing Features
--------------------------------------------------------------------------
1. X-Ray Feature Enhancement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: Intelligent X-Ray Content Generation
- ML Problem Description: Automatically generating and enhancing X-Ray content for books
- Data Sources: Book content, existing X-Ray data, user interactions with X-Ray features
- Modeling Approach: Named Entity Recognition (`GENRE <https://github.com/facebookresearch/GENRE>`_), Relation Extraction, Summarization models
- Key KPIs: X-Ray usage rate, time spent using X-Ray features
- ML Metrics: F1 score for entity and relation extraction, ROUGE scores for generated content
- Quality Metrics: Accuracy of information, relevance to reader's current position in the book
- Resources: [WordDumb] `Calibre Plugin <https://xxyzz.github.io/WordDumb/index.html>`_ (could possibly be useful)

2. Book Content Quality Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: ML-Driven Content Quality Enhancement
- ML Problem Description: Identifying and suggesting improvements for book content, especially for nonfiction and children's books
- Data Sources: eBook content, editorial guidelines, user feedback on content quality
- Modeling Approach: Text classification for issue detection, sequence-to-sequence models for suggesting improvements
- Key KPIs: Reduction in customer complaints about content quality, improvement in book ratings
- ML Metrics: Precision and recall for issue detection, BLEU score for improvement suggestions
- Quality Metrics: Editor acceptance rate of ML suggestions, improvement in readability scores

3. Customer Review Summaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: Automated Review Summarization
- ML Problem Description: Extracting key points and sentiment from multiple customer reviews to create concise summaries
- Data Sources: Customer reviews, ratings, verified purchase data
- Modeling Approach: Abstractive summarization models (e.g., BART, T5), sentiment analysis
- Key KPIs: Summary usage rate, impact on purchase decisions
- ML Metrics: ROUGE scores for summarization quality, sentiment classification accuracy
- Quality Metrics: Readability of summaries, coverage of key review points, balance of positive and negative feedback

4. Rufus Experience for Books
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: Rufus AI Assistant for Books
- ML Problem Description: Natural language understanding and generation for book-related queries and recommendations
- Potential Data Sources: Book metadata, user interactions, purchase history, book content summaries
- Modeling Approach: Large Language Models (LLMs) fine-tuned on book-related data, potentially using retrieval-augmented generation
- Key KPIs: User engagement rate, query resolution rate, conversion rate from Rufus interactions
- ML Metrics: Perplexity, BLEU score for response quality, relevance of recommendations
- Quality Metrics: User satisfaction, accuracy of information provided, diversity of recommendations

5. Next Read Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: Personalized Book Discovery Engine
- ML Problem Description: Recommending the next book for readers based on their reading history and preferences
- Data Sources: User reading history, book metadata, user demographics, reading speed data
- Modeling Approach: Collaborative filtering, content-based filtering, and hybrid models (e.g., neural collaborative filtering)
- Key KPIs: Click-through rate on recommendations, conversion rate, user retention
- ML Metrics: Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG)
- Quality Metrics: Diversity of recommendations, serendipity, user satisfaction with recommendations

6. Book Club Recommendation System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: AI-Powered Book Club Matcher
- ML Problem Description: Matching readers with suitable book clubs based on reading preferences and social factors
- Data Sources: User reading history, book club data, social interaction data from Goodreads
- Modeling Approach: Clustering algorithms, Graph Neural Networks for social connections
- Key KPIs: Book club join rate, member retention rate, engagement in book discussions
- ML Metrics: Silhouette score for clustering quality, link prediction accuracy in social graphs
- Quality Metrics: User satisfaction with book club matches, diversity of book club suggestions

7. Image Rendering Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Product Name: Adaptive eBook Image Optimization
- ML Problem Description: Optimizing image rendering for various devices and screen sizes while maintaining quality
- Data Sources: Original book images, device specifications, user feedback on image quality
- Modeling Approach: Computer Vision models for image quality assessment, Generative AI for image enhancement
- Key KPIs: User satisfaction with image quality, reduction in image-related complaints
- ML Metrics: Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR)
- Quality Metrics: Loading speed of images, preservation of important details across devices

For each of these products, it's important to also consider:

- Scalability of the ML solutions to handle Amazon's vast user base and book catalog
- Privacy and security measures, especially when dealing with user data
- Fairness and bias mitigation in recommendations and content generation
- Interpretability of ML models, where applicable, to provide transparent recommendations or decisions to users and stakeholders

Claude Generated Problem List
--------------------------------------------------------------------------
1. Automated Book Categorization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Develop a system to automatically categorize books into genres and sub-genres based on their content, cover images, and metadata.

2. Content Quality Assessment:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a model to assess the quality of submitted manuscripts, considering factors like grammar, style, structure, and potential reader engagement.

3. Book Summary Generation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Design an AI system that can generate concise, accurate summaries of books to help readers quickly understand the main points and decide if they want to read the full text.

4. Cross-lingual Book Recommendation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Develop a recommendation system that can suggest books to readers across different languages, considering content similarity and user preferences.

5. Automated Content Moderation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a system to automatically flag potentially inappropriate or sensitive content in submitted manuscripts, considering various cultural and age-appropriate contexts.

6. Enhanced eBook Layout Optimization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Design an AI-driven system that can automatically optimize the layout and formatting of eBooks for different devices and screen sizes, ensuring a consistent reading experience.

7. Author Style Analysis and Ghostwriting Detection:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Develop a model to analyze writing styles and potentially detect ghostwritten content or verify author consistency across multiple works.

8. Intelligent Text-to-Speech for Audiobooks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create an AI system that can convert eBooks into natural-sounding audiobooks, including appropriate pacing, emphasis, and potentially different voices for dialogue.

9. Automated Illustration Generation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Design a system that can generate relevant illustrations or suggest image placements based on the textual content of a book.

10. Reading Engagement Prediction:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Develop a model to predict reader engagement and completion rates for books based on various factors like writing style, genre, length, and historical user data.

For each of these problems, you should be prepared to discuss:

	- Clarifying questions about the specific goals and constraints
	- Potential data sources and annotation strategies
	- Suitable modeling approaches (e.g., which ML/NLP techniques might be appropriate)
	- Evaluation metrics and methodologies
	- Potential challenges and pitfalls in implementation
	- Ethical considerations and biases to be aware of
	- Trade-offs between different approaches or model architectures

GPT Generated Problem List
--------------------------------------------------------------------------
1. Reading Experience
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Intelligent Chapter Summaries: Enhances reader engagement by providing a preview of content and facilitates easier navigation within books.
	- Description: Using AI to generate concise summaries of chapters or sections within book. This helps readers quickly grasp key points and decide if they want to delve deeper into specific parts.

- Personalized Reading Recommendations: Increases book discoverability and encourages continued engagement by offering tailored suggestions based on individual reading habits.
	- Description: AI algorithms analyze reader preferences and behavior to suggest books within KDP's library that match their interests.

2. Publishing (Creation of Books Process)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Automated Genre Classification: Streamlines the publishing process for authors by automatically assigning accurate genres, aiding in better metadata tagging and targeting specific reader demographics.
	Description: AI categorizes manuscripts into specific genres (e.g., mystery, romance, sci-fi) based on semantic analysis of content.

- Content Enhancement through AI Editing: Helps authors polish their work before publishing, leading to higher quality books and potentially better reader reception.
	Description: AI-powered tools assist authors in refining their manuscripts by suggesting improvements in writing style, grammar, and structure, improving readability and engagement.

3. Reporting (Improvement through Sales & Business Growth)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Predictive Sales Analytics: Empowers authors with insights into potential sales trajectories, allowing them to make informed decisions on marketing strategies and promotions.
	Description: AI models forecast book sales based on historical data, market trends, and content analysis.

- Automated Performance Insights: Enables authors to iterate and enhance subsequent editions based on real-time feedback and performance metrics.
	Description: AI algorithms analyze reader reviews, engagement metrics, and sales data to provide authors with actionable insights for improving their books.

4. Cross-Cutting Ideas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- AI-driven Content Translation: Expands the reach of books to international markets, increasing sales potential and accessibility for diverse readers.
	Description: Utilizing AI for accurate and context-aware translation of books into multiple languages, preserving the author's voice and style.

- Visual Content Analysis for Enhanced eBooks: Improves the overall reading experience for genres like comics, children's books, and cookbooks by maintaining visual fidelity and clarity.
	Description: AI identifies and enhances visual elements (images, graphics) within eBooks, ensuring optimal display across different devices and formats.

5. Vague Ideas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Content Moderation and Quality Assurance:
	Description: Develop AI systems for automatic content moderation, ensuring adherence to publishing standards and identifying potentially problematic content.

Enhanced Kindle eBook Publishing Process Overview
--------------------------------------------------------------------------
1. Manuscript Preparation: Authors write and format their manuscripts using advanced AI tools that ensure proper formatting and suggest improvements.
2. Conversion to Kindle Format: AI tools automatically convert manuscripts to Kindle-compatible formats, minimizing manual adjustments.
3. Metadata Entry: AI systems suggest optimal metadata to improve discoverability on Amazon.
4. Cover Design: Generative AI tools assist in creating visually appealing covers that resonate with the book's genre and content.
5. Uploading and Previewing: Enhanced preview tools ensure proper formatting across all Kindle devices.
6. Pricing and Rights: AI-driven tools recommend optimal pricing strategies based on market analysis.
7. Publishing and Marketing: AI tools provide marketing insights and strategies to help authors reach their target audience effectively.

Potential Features and AI/ML Technologies
--------------------------------------------------------------------------
1. Automated Formatting and Conversion: AI-powered tool that formats manuscripts according to Kindle standards and converts them to the appropriate format with minimal manual intervention.
   	- Technology: NLP for understanding document structure, computer vision for image placement, DL models for format conversion.
2. Intelligent Metadata Generation: Tool that suggests optimal keywords, categories, and metadata to enhance discoverability.
   	- Technology: LLMs for understanding manuscript content and suggesting relevant keywords, classification models for category suggestions.
3. Cover Design Assistance: AI-driven design tool that generates cover design options based on the book's content and genre.
   	- Technology: Generative AI for image creation, style transfer models to match the genre-specific aesthetics.
4. Advanced Preview and Validation: Smart preview tool that simulates how the ebook will look across different Kindle devices and flags potential formatting issues.
   	- Technology: Computer vision to analyze and compare layout consistency across devices, regression models to predict readability issues.
5. Content Quality and Consistency Checker: AI tool that checks for grammar, style, and consistency within the manuscript, offering suggestions for improvement.
   	- Technology: NLP models for grammar and style checking, LLMs for content consistency analysis.
6. Dynamic Pricing Recommendations: AI-driven pricing advisor that suggests optimal pricing based on market trends, genre, and competitive analysis.
   	- Technology: Predictive modeling and reinforcement learning to analyze market data and suggest pricing strategies.
7. Marketing and Promotion Insights: Tool that provides marketing insights and strategies tailored to the book’s genre and target audience.
   	- Technology: Data analytics for market trend analysis, NLP for sentiment analysis on reader reviews, and recommendation systems for personalized marketing strategies.
8. Interactive Editing Assistant: Smart assistant within the KDP platform that offers real-time suggestions and corrections as authors upload and edit their manuscripts.
   	- Technology: NLP and LLMs for understanding context and providing relevant suggestions.
9. Personalized Author Dashboard: Dashboard that uses ML to provide personalized insights, such as sales trends, reader demographics, and marketing effectiveness.
   	- Technology: Data analytics and visualization tools.
10. Voice-to-Text and Text-to-Voice Tools: Tools that allow authors to dictate their manuscripts and listen to their books read aloud, using advanced speech recognition and synthesis technologies.
   	- Technology: Speech-to-text and text-to-speech models.
11. Enhanced Analytics for Reader Engagement: Tools that analyze reader behavior (e.g., highlights, notes, read-through rates) to provide feedback to authors on which parts of their books are most engaging.
   	- Technology: Data analytics and NLP for understanding reader interactions.

Supporting Technologies
--------------------------------------------------------------------------
- Natural Language Processing (NLP): For understanding and processing text data, including metadata generation, content analysis, and grammar checking.
- Large Language Models (LLM): For generating text, understanding context, and offering suggestions related to content and marketing.
- Generative AI: For creating cover designs and other visual elements.
- Computer Vision: For analyzing document layouts and ensuring consistent formatting across devices.
- Deep Learning (DL): For complex model building, such as format conversion, content quality checking, and predictive analytics.
- Reinforcement Learning (RL): For dynamic pricing and other adaptive strategies.
- Data Analytics: For market analysis, trend prediction, and recommendation systems.

**************************************************************************
Sample Questions
**************************************************************************
Shared by Recruiter
==========================================================================
ML Breadth
--------------------------------------------------------------------------
Expectation: Candidates should demonstrate a solid understanding of standard methods relevant to their scientific field. A good measure of suitable breadth includes the ability to discuss concepts/methods commonly covered in relevant graduate-level university courses and apply these methods to construct a functional, scalable system. 

Additionally, familiarity with concepts such as experimental design, system evaluation, and optimal decision making across various scientific domains is important. The evaluation process can incorporate the following approaches:

Methods Survey: An assessment of the candidate's knowledge of techniques includes:

- How do you identify and address overfitting?
- Can you develop a query embedding for Amazon teams?
- Explain ensemble algorithms (e.g., Random forest; handling features and data; reducing variance).
- What methods can be used to split a decision tree?
- Which metrics would you utilize in a classification problem?
- How do you handle imbalanced datasets?
- What loss function is suitable for measuring multi-label problems?
- Suppose you need to determine a threshold for a classifier predicting customer sign-up for Prime. What criteria could be used to determine this threshold?
- In a model with one billion positive samples and 200,000 negative samples, what would you examine to ensure its quality before deployment?
- Describe the training process for a Context-awareness entity ranking model.

ML Depth
---------------------------------------------------------------------------
Expectation: Candidates are expected to exhibit mastery in their specific area of expertise, preferably assessed by a recognized authority in the field. They should demonstrate the ability to discern methodological trade-offs, contextualize solutions within both classical and contemporary research, and possess familiarity with the nuanced skill of devising solutions within their domain. Ideally, they would have a track record of publications in their field. The assessment process should delve into the following aspects:

- Methods: Candidates should provide detailed insights into the methodologies employed in their research and projects, including rationale for their choices (such as highlighting strengths and weaknesses of methods and justifying their selection).
- Innovation vs Practicality: Assessment should explore candidates' past projects to gauge their level of creativity and pragmatism.
- Deep Dives: Evaluation should examine whether candidates delved deeply into projects where relevant, such as investigating outliers, misclassified examples, and edge cases.
- Model Evaluation: Candidates should elaborate on how they evaluated their models, including rationale behind specific trade-offs and methods used to identify key model dynamics.
- Fundamentals: Assessment should cover candidates' understanding of the fundamental principles in their field.

Scrapped from the Internet
==========================================================================
Data Preprocessing and Handling:
--------------------------------------------------------------------------
1. How would you handle missing or corrupted data in a dataset?
2. How would you find thresholds for a classifier?
3. What are some ways to split a tree in a decision tree algorithm?
4. How does pruning work in Decision Trees?
5. What methods would you employ to forecast sales figures for Samsung phones?

Supervised Learning:
--------------------------------------------------------------------------
1. State the applications of supervised machine learning in modern businesses.
2. How will you determine which machine learning algorithm to use for a classification problem?
3. How does the Amazon recommendation engine work when recommending other things to buy?
4. Differentiate between logistic regression and support vector machines.
5. Give an example of using logistic regression over SVM and vice versa.
6. What does the F1 score represent?
7. How do the results change if we use logistic regression over the decision tree in a random forest?
8. Describe linear regression vs. logistic regression.
9. How would you define log loss in the context of model evaluation?
10. Could you discuss the key assumptions that govern linear regression models and explain the significance of taking these assumptions into account when interpreting statistical results?

Ensemble Learning:
--------------------------------------------------------------------------
1. Explain the ensemble learning technique in machine learning.
2. Differentiate between bagging and boosting.
3. What distinguishes the model performance between bagging and boosting?
4. Can you elaborate on how gradient boost is used in machine learning and how it works?
5. How does the assumption of error in linear regression influence the accuracy of our models, and what does it entail?
6. How do you perceive the role of DMatrix in XGBoost, and how does it differ from other gradient boosting data structures?

Clustering and Dimensionality Reduction:
--------------------------------------------------------------------------
1. How is KNN different from K-means clustering?
2. Explain the K-means and K Nearest Neighbor algorithms and differentiate between them.
3. How are PCA with a polynomial kernel and a single layer autoencoder related?
4. Differentiate between Lasso and Ridge regression.
5. Explain ICA, CCA, and PCA.
6. State some ways of reducing dimensionality.
7. How would you get a CCA objective function from PCA?

Model Evaluation and Performance:
--------------------------------------------------------------------------
1. Considering that you already have labeled data for your clustering project, what are some of the methods that you can use to evaluate model performance?
2. What does an ROC curve tell you about a model’s performance?
3. Could you define the concepts of overfitting and underfitting in machine learning, and explain their relevance in model development?

Deep Learning and Neural Networks:
--------------------------------------------------------------------------
1. Can you elaborate on what an attention model entails?
2. Can you differentiate between batch normalization and instance normalization and their respective uses?
3. Can you walk me through the functioning of a 1D CNN?
4. Can you describe the difference in application between RNNs and LSTMs?

Miscellaneous:
--------------------------------------------------------------------------
1. Design an Email Spam Filter.
2. What steps would you take to ensure a scalable, efficient architecture for Bing’s image search system?
3. How can you perform a dot product operation on two sparse matrices?
4. Walk me through a Monte Carlo simulation to estimate Pi.

**************************************************************************
Interview Experience (Scrapped from the Internet)
**************************************************************************
Science Breadth
==========================================================================
In the ML Breadth round, the focus was on assessing the depth of my understanding across machine learning concepts. I encountered a mix of theoretical questions and practical scenarios related to applied science at Amazon. It tested my ability to grasp a broad spectrum of ML topics, showcasing the importance of a well-rounded foundation in machine learning. This would include topics in supervised and unsupervised learning 

.. note::
	* KNN, logistic regression, SVM, Naive Bayes, Decision Trees, Random Forests, Ensemble Models, Boosting, 
	* Regression, Clustering, Dimensionality Reduction
	* Feature Engineering, Overfitting, Regularization, best practices for hyperparameter tuning, Evaluation metrics
	* Neural Networks, RNNs, CNNs, Transformers.

Science Depth
==========================================================================
The Science Depth segment involved a resume deep dive, where detailed questions probed into my past work experiences. This round aimed to uncover the depth of my expertise in specific areas, emphasizing the practical application of my knowledge. This would entail understanding the tradeoffs made during the project, the different design decisions, results and impact on the organization and understanding how successful was the project at solving the problem at hand using business metrics if required. Nitty gritty details of implementation are enquired during the interview and its important to take a look at past projects and know every little detail of it and study its impact.

Science Application
==========================================================================
The Machine Learning Case Study in the domain of the job role provided a practical challenge to assess my ability to apply theoretical knowledge to real-world scenarios. This segment gauged my problem-solving skills within the context of the job, giving me an opportunity to showcase my ability to translate theoretical concepts into actionable solutions. This would entail first understanding the business problem, and then methodically come up with steps for problem formulation and a solid reason to go for a machine learning based solution. The next part would be to come up with the data collection, feature engineering and talk about the different machine learning models and finally talk about evaluation metrics, training strategies and understanding the business metric and A/B testing the model to understand feasibility for replacing the existing model.

Leadership Principles
==========================================================================
The Behavioral Style questions in the Leadership Principles round were designed to evaluate my alignment with Amazon’s core leadership principles. Through scenarios drawn from my past work experiences, I was assessed for various leadership skills. This round, often conducted by a bar raiser, held significant importance in determining my suitability for the role, underscoring Amazon’s commitment to strong leadership qualities. A strong emphasis is given on the STAR format — Situation, Task, Action and Result hence it’s very important to follow this structure when answering any scenario based question.

Coding
==========================================================================
The Coding segment comprised LeetCode-style Data Structures and Algorithms questions. This component tested my coding proficiency and problem-solving abilities. Topics would include 

.. note::
	* Data Structures
		* Arrays, Hash maps, Graphs, Trees, Heaps, Linked List, Stack, Queue
	* Algorithms
		* Binary Search, Sliding Window, Two Pointer, Backtracking, Recursion, Dynamic Programming, Greedy. 
	* Data Manipulation libraries
		* Pandas and SQL.
	* Coding concepts from Machine Learning, Probability and Statistics.

Tech Talk
==========================================================================
An intriguing component of the interview process was the Tech Talk, a platform for me to showcase one of my previous projects. This session involved a 45-minute presentation, allowing me to delve into the details of the project, its objectives, methodologies employed, and, most importantly, the outcomes achieved. This presentation was a chance to demonstrate my communication skills, presenting complex technical information in an accessible manner. Following the presentation, the last 15 minutes were dedicated to a Q&A session facilitated by the panelists.

**************************************************************************
Links
**************************************************************************
.. note::
	* `Amazon Interview Experience for Applied Scientist <https://www.geeksforgeeks.org/amazon-interview-experience-for-applied-scientist/>`_
	* `Amazon data scientist interview (questions, process, prep) <https://igotanoffer.com/blogs/tech/amazon-data-science-interview>`_
	* `Amazon | Senior Applied Scientist L6 | Seattle <https://leetcode.com/discuss/compensation/685178/amazon-senior-applied-scientist-l6-seattle>`_
	* `Leadership Principles <https://www.amazon.jobs/content/en/our-workplace/leadership-principles>`_
