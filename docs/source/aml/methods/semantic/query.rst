################################################################
Query understanding
################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none
****************************************************************
Query segmentation
****************************************************************
- task: partition query into predefined segments
- example segments: brand, product, size, colour, location
- input: query
- output: structured output corrsponding to segments
- use-case:
	- enables broad match in sparse index. query -> segments -> enables 'OR' based search
	- can be used for intent classification
	- can be used for generating rewrites by reordering via templates
	- can be used as input features in query tower for dense index
- techniques:
	(a) tokenize -> dictionary lookup
	(b) pos tagging
	(c) templated/rule based

****************************************************************
Query Rewrites
****************************************************************
- task: generate variations of given query
- input: query
- output: a set of k=5/10 rewrites
- use-case:
	- enables extended match in sparse index. query -> rewrites -> multiple lookups -> union result
	- can be used for data augmentation in dense retrieval/reranking modeling
- techniques:
	(a) co-click graph based
	(b) semantic similarity based
	(c) canonical replaement/synonym replacement
	(d) LLM rewrites

****************************************************************
Query Intent
****************************************************************
- task: map intent to a set of predefined intent classes
- target: browse, compare, planning, buy
- output: one or more intent classes
- use-case:
	- template selection based on intent -> can be used for segmentation
	- can be used as features in downstream query tower encoder
- techniques:
	(a) lookup based
	(b) embedding + classifier heads
	(c) sequence based
