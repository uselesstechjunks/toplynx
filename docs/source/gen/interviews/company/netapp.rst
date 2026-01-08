########################################################################################
NetApp
########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

****************************************************************************************
Round 1: Machine Learning / Data Science Coding
****************************************************************************************
Python-Based Questions
========================================================================================
Data Preprocessing:
	.. note::
		Problem: Given a dataset with missing values, normalize all numerical columns after imputing the missing values with their column means.

Data Aggregation:
	.. note::
		Problem: Given a dataset of file logs with columns user_id, file_type, and file_size, calculate the total size of each file_type uploaded by each user.

Data Pipeline:
	.. note::
		Problem: Write a Python script that reads a large CSV file in chunks, filters rows based on a condition (e.g., file size > 100 MB), and writes the filtered rows to a new file.

Model Evaluation:
	.. note::
		Problem: Write a function to compute precision, recall, and F1-score given true and predicted labels.

Custom Metric Implementation:
	.. note::
		Problem: Implement a function to compute the Area Under the Precision-Recall Curve (AUPRC) for given true and predicted probabilities.
	
Feature Engineering:
	.. note::
		Problem: Given a dataset of file uploads with columns file_id, file_size, and upload_date, create a new feature representing the file size as a percentage of the average file size for its upload date.
	
Time Series Analysis:
	.. note::
		Problem: Write a function to detect outliers in a time series data based on a rolling window standard deviation.
	
Clustering:
	.. note::
		Problem: Implement k-means clustering from scratch in Python and cluster a given dataset into 3 groups.

Dimensionality Reduction:
	.. note::
		Problem: Use Principal Component Analysis (PCA) to reduce the dimensions of a high-dimensional dataset and retain 95% of the variance.
	
Decision Trees:
	.. note::
		Problem: Implement a simple decision tree classifier to predict whether a file is likely to be accessed frequently based on features like file size, user ID, and file type.

Natural Language Processing:
	.. note::
		Problem: Implement a simple sentiment analysis model using Naive Bayes to classify user reviews as positive or negative.
	
API Data Fetching:
	.. note::
		Problem: Fetch data from a public API (e.g., GitHub repositories), clean it, and find the top 5 repositories with the most stars.

Optimization Problem:
	.. note::
		Problem: Given a list of file sizes and a storage limit, write a function to find the maximum number of files that can fit within the storage limit.
	
SQL-Based Questions
========================================================================================
Basic Query:
	.. note::
		Problem: Find the average file size from a table files with columns file_id, file_name, and file_size.

Join and Aggregation:
	.. note::
		Problem: Given two tables, users (with user_id, name) and files (with file_id, user_id, file_size), find the total file size uploaded by each user.

Window Functions:
	.. note::
		Problem: Write a query to calculate the rank of each user based on their total file size uploaded in descending order.Data Cleaning:

Data Cleaning:
	.. note::
		Problem: Find and delete duplicate rows in a table files based on the columns file_name and upload_date.

Complex Joins:
	.. note::
		Problem: Given three tables—users, files, and tags—find all files tagged as "important" by users who have uploaded more than 100 files.

Dynamic Queries:
	.. note::
		Problem: Create a query to find the average file size for each file_type, and return only those averages above a threshold (e.g., 100 MB).

Recursive Queries:
	.. note::
		Problem: Write a query to find all parent-child relationships in a hierarchical table folders with columns folder_id and parent_id.
Pivot Table:
	.. note::
		Problem: Write a query to convert rows of file types and their counts into a column format for better visualization.

Multi-Table Analysis:
	.. note::
		Problem: Given two tables—files (with file_id, user_id, file_size) and file_tags (with file_id, tag)—write a query to find the top 3 tags associated with the largest files.

Temporal Analysis:
	.. note::
		Problem: Write a query to find the average file size uploaded per day over the past 30 days.

Data Validation:
	.. note::
		Problem: Write a query to identify rows in a table files where upload_date is later than the modification_date.

Case Statement:
	.. note::
		Problem: Write a query to classify files into size categories ("Small", "Medium", "Large") based on predefined thresholds.

Index Optimization:
	.. note::
		Problem: Write a query to analyze the performance of an index on the file_name column in a large files table.

****************************************************************************************
Round 2: Machine Learning System Design
****************************************************************************************
Design a Scalable Recommendation System for File Storage Optimization:
	.. note::
		- Discuss data sources: user behavior logs, file metadata.
		- Feature engineering: file access frequency, user preferences.
		- Model: Collaborative filtering or content-based filtering.
		- System architecture: Data ingestion pipeline, model training (batch), real-time inference using a microservices-based architecture.

Monitoring and Maintaining a ML Model for Anomaly Detection in Cloud Storage:
	.. note::
		- Metrics: Precision, recall, drift detection.
		- Automation: Retraining pipelines, model versioning.
		- Infrastructure: Use of Docker/Kubernetes for deployment, cloud services for scalability.

Scalable File Deduplication System:
	.. note::
		- Problem: Design a system that detects duplicate files in a distributed storage system.
		- Considerations: Hashing, sharding strategies, and handling partial duplicates.

Content-Based Search for Cloud Files:
	.. note::
		- Problem: Design a system that allows users to search files based on their content (e.g., text or metadata) instead of just file names.
		- Include indexing, embedding generation, and retrieval strategies.

Predictive Maintenance for Cloud Servers:
	.. note::
		- Problem: Design a system to predict potential failures in cloud servers based on historical sensor data.
		- Considerations: Handling time-series data, real-time alerts, and scalability.

Usage Pattern Anomaly Detection:
	.. note::
		- Problem: Design a system that detects unusual user behavior in file access patterns to prevent unauthorized access.
		- Include: Model architecture (e.g., autoencoders or isolation forests) and deployment pipeline.

Data Compression System:
	.. note::
		- Problem: Propose a machine learning-based system to identify optimal compression algorithms for different file types uploaded by users.
