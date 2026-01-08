#######################################################################
Language Modeling Implementations
#######################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

***********************************************************************
Sequence Modeling
***********************************************************************
RNN
=========================================================================================
.. literalinclude:: code/rnn.py
	:language: python
	:linenos:

LSTM
=========================================================================================
.. literalinclude:: code/lstm.py
	:language: python
	:linenos:

***********************************************************************
Attention
***********************************************************************
Understanding Einsum
=========================================================================================
.. literalinclude:: code/einsum.py
	:language: python
	:linenos:

Dot product Attention (single query)
=========================================================================================
.. literalinclude:: code/attn.py
	:language: python
	:linenos:

Multi-head Attention (single query)
=========================================================================================
.. literalinclude:: code/mha.py
	:language: python
	:linenos:

Multi-head Attention (sequential query)
=========================================================================================
.. literalinclude:: code/mha_seq.py
	:language: python
	:linenos:

Masked Multi-head Attention (parallel query)
=========================================================================================
.. literalinclude:: code/mha_par.py
	:language: python
	:linenos:

Masked Multi-head Attention Batched (parallel query)
=========================================================================================
.. literalinclude:: code/mha_par_batched.py
	:language: python
	:linenos:

Multi-head Attention Batched (sequential query)
=========================================================================================
.. literalinclude:: code/mha_seq_batched.py
	:language: python
	:linenos:

Masked Multi-query Attention Batched (parallel query)
=========================================================================================
.. literalinclude:: code/mqa_par_batched.py
	:language: python
	:linenos:

Multi-query Attention Batched (sequential query)
=========================================================================================
.. literalinclude:: code/mqa_seq_batched.py
	:language: python
	:linenos:

Unit Test
=========================================================================================
.. literalinclude:: code/attn_test.py
	:language: python
	:linenos:
