How Python Compiles and Executes The Source Code
##########################################################################
To understand how Python compiles and executes the source code, I'll follow along the talk `Demystifying Python’s Internals <https://www.youtube.com/watch?v=HYKGZunmF50>`_ by Sebastiaan Zeeff at PyCon US and note down any useful insight I find. In the talk, Sebastiaan shows a step-by-step guide on how to implement a custom operator in python. My plan is to eventually implement a haskell like Monad operator ``>>=`` in Python (see `this <http://learnyouahaskell.com/a-fistful-of-monads>`_).

The Big Picture
*****************************************
.. image:: ../img/001pyflow.png
  :width: 800
  :alt: Python Compilation Flow

Prerequisite: Setting up devenv
*****************************************
* Used the base Ubuntu docker image on Linux and installed gcc, python3, make, git and vim.
* Cloned cpython `git repo <https://github.com/python/cpython.git>`_.
* Switched to branch 3.11 for my experiments for reproducibility (base commit SHA: ``4a7612fbecbdd81a6e708e29aab0dc4c6555948d``).
* Compiled cpython source which creates a binary named ``python`` inside the build dir.

.. code-block:: bash

  mkdir build && cd build && ./configure && make -j 8

Adding ``|>`` Operator
**********************************************************************************
The goal here is to implement a pipe operator, ``|>``, (similar to Linux ``|``) so that we can chain function calls like the following:

.. code-block:: python

    def sq(x):
      return x**2

    def double(x):
      return x*2
      
    # usage: functionally the same as sq(sq(double(sq(42)))))
    42 |> sq |> double |> sq |> sq 

Part 1: From Source Code to Abstract Syntax Tree
====================================================================================
When Python reads a source code, it first builds an abstract syntax tree (AST). This consists of two parts:

  #. from the source code characters, it creates a sequence of tokens using a ``tokenizer``.
  #. from the tokenized output, it then creates AST using grammar rules that are defined in the parser.

Adding tokenizer support
==========================================

TL;DR
-------------------------
* Tokens are defined inside the config file ``Grammar/Tokens``.
* To Add a new token, we need to specify a ``key`` (e.g. ``MYTOKENIDENTIFIERKEY``) and a ``value`` (``=/=``) in this file and regerate tokens.
* The tokens are then automatically given a numeric code (defined in ``token.h``).
* Python uses a C functionality (as provided by ``tokenizer.c``) to tokenize the source code.
* The same numeric code, key and value are copied inside a python file ``token.py``.

In-Depth
-----------------------
Added a token key ``VBARGREATER`` with value ``|>`` in this file ``/cpython/Grammar/Tokens`` and ran ``make regen-token`` to regenerate the tokenizer.

.. collapse:: Expand snippet

  .. code-block:: bash

      root@008f4044fac9:/cpython# git diff Grammar/Tokens
      diff --git a/Grammar/Tokens b/Grammar/Tokens
      index 1f3e3b0991..13aac4c7b6 100644
      --- a/Grammar/Tokens
      +++ b/Grammar/Tokens
      @@ -53,6 +53,7 @@ ATEQUAL                 '@='
       RARROW                  '->'
       ELLIPSIS                '...'
       COLONEQUAL              ':='
      +VBARGREATER             '|>'

       OP
       AWAIT

I could see a difference in terms of how a source code is tokenized. To see this, I created a test file with the same python code as above and ran: ``python -m tokenize test/test.py``. Earlier, ``|`` and ``>`` were identified as separate tokens. Now, each instance of ``|>`` is treated as single token.

.. collapse:: Expand snippet

  .. code-block:: bash

      # part of tokenizer output
      ...
      7,0-7,0:            DEDENT         ''
      7,0-7,2:            NUMBER         '42'
      7,3-7,5:            OP             '|>'
      7,6-7,8:            NAME           'sq'
      7,9-7,11:           OP             '|>'
      7,12-7,18:          NAME           'double'
      7,19-7,21:          OP             '|>'
      7,22-7,24:          NAME           'sq'
      7,25-7,27:          OP             '|>'
      7,28-7,30:          NAME           'sq'
      7,30-7,31:          NEWLINE        '\n'
      8,0-8,0:            ENDMARKER      ''

I could also see that a bunch of other files has also been changed automatically after the token-regeneration.

.. code-block:: bash

    modified:   Doc/library/token-list.inc
    modified:   Grammar/Tokens
    modified:   Include/token.h
    modified:   Lib/token.py
    modified:   Parser/token.c

Let's dig deep into see what changes were made in each of these files and try to wrap our heads around what these files are for.

* ``Doc/library/token-list.inc``

    This one is for the Python doc file. It created an entry in the docs for the new token key and value.

    .. collapse:: Expand snippet
    
      .. code-block:: bash

          root@008f4044fac9:/cpython# git diff Doc/library/token-list.inc
          diff --git a/Doc/library/token-list.inc b/Doc/library/token-list.inc
          index 1a99f0518d..b8d2bd5185 100644
          --- a/Doc/library/token-list.inc
          +++ b/Doc/library/token-list.inc
          @@ -201,6 +201,10 @@

              Token value for ``":="``.

          +.. data:: VBARGREATER
          +
          +   Token value for ``"|>"``.
          +
           .. data:: OP

           .. data:: AWAIT

* ``Lib/token.py``

    This one assigns a numerical code to each of the tokens. Since I added the token in the middle and not at the end, it reassigned the numeric codes for the following tokens as well. Newly added ``VBARGREATER`` gets a code, 54. Number of tokens (``N_TOKENS``) has increased from 64 to 65. Also, there is a dictionary, ``EXACT_TOKEN_TYPES``, which maintains an inverse map from value to key (e.g. ``|>`` to key ``VBARGREATER``).

    .. collapse:: Expand snippet
    
      .. code-block:: bash

          root@008f4044fac9:/cpython# git diff Lib/token.py
          diff --git a/Lib/token.py b/Lib/token.py
          index 9d0c0bf0fb..8b8d2c1a09 100644
          --- a/Lib/token.py
          +++ b/Lib/token.py
          @@ -57,18 +57,19 @@ ATEQUAL = 50
           RARROW = 51
           ELLIPSIS = 52
           COLONEQUAL = 53
          -OP = 54
          -AWAIT = 55
          -ASYNC = 56
          -TYPE_IGNORE = 57
          -TYPE_COMMENT = 58
          -SOFT_KEYWORD = 59
          +VBARGREATER = 54
          +OP = 55
          +AWAIT = 56
          +ASYNC = 57
          +TYPE_IGNORE = 58
          +TYPE_COMMENT = 59
          +SOFT_KEYWORD = 60
           # These aren't used by the C tokenizer but are needed for tokenize.py
          -ERRORTOKEN = 60
          -COMMENT = 61
          -NL = 62
          -ENCODING = 63
          -N_TOKENS = 64
          +ERRORTOKEN = 61
          +COMMENT = 62
          +NL = 63
          +ENCODING = 64
          +N_TOKENS = 65
           # Special definitions for cooperation with parser
           NT_OFFSET = 256

          @@ -123,6 +124,7 @@ EXACT_TOKEN_TYPES = {
               '{': LBRACE,
               '|': VBAR,
               '|=': VBAREQUAL,
          +    '|>': VBARGREATER,
               '}': RBRACE,
               '~': TILDE,
           }

* ``Include/token.h``

    Added the same numeric code as above but in the C header.

    .. collapse:: Expand snippet
    
      .. code-block:: bash

          root@008f4044fac9:/cpython# git diff Include/token.h
          diff --git a/Include/token.h b/Include/token.h
          index eb1b9ea47b..efc42f7825 100644
          --- a/Include/token.h
          +++ b/Include/token.h
          @@ -64,14 +64,15 @@ extern "C" {
           #define RARROW          51
           #define ELLIPSIS        52
           #define COLONEQUAL      53
          -#define OP              54
          -#define AWAIT           55
          -#define ASYNC           56
          -#define TYPE_IGNORE     57
          -#define TYPE_COMMENT    58
          -#define SOFT_KEYWORD    59
          -#define ERRORTOKEN      60
          -#define N_TOKENS        64
          +#define VBARGREATER     54
          +#define OP              55
          +#define AWAIT           56
          +#define ASYNC           57
          +#define TYPE_IGNORE     58
          +#define TYPE_COMMENT    59
          +#define SOFT_KEYWORD    60
          +#define ERRORTOKEN      61
          +#define N_TOKENS        65
           #define NT_OFFSET       256

* ``Parser/token.c``

    This has an array of token names, ``_PyParser_TokenNames``, in which it added the new token. In general, this file defines functions that returns numeric codes for tokens of different length (as defined in ``token.h``), such as, ``int PyToken_OneChar(int c1)``, ``int PyToken_TwoChars(int c1, int c2)`` and ``int PyToken_ThreeChars(int c1, int c2, int c3)``. In the current change, it added a new line of code inside ``PyToken_TwoChars`` in the switch statement to differentiate between ``|=`` (already existing token in Python) and the newly added ``|>``. This function is utilised in a giant function ``static int tok_get(struct tok_state *tok, const char **p_start, const char **p_end)`` inside ``Parser/tokenizer.c``.

    .. collapse:: Expand snippet
    
      .. code-block:: bash

          root@008f4044fac9:/cpython# git diff Parser/token.c
          diff --git a/Parser/token.c b/Parser/token.c
          index 74bca0eff6..6c3ea72316 100644
          --- a/Parser/token.c
          +++ b/Parser/token.c
          @@ -60,6 +60,7 @@ const char * const _PyParser_TokenNames[] = {
               "RARROW",
               "ELLIPSIS",
               "COLONEQUAL",
          +    "VBARGREATER",
               "OP",
               "AWAIT",
               "ASYNC",
          @@ -184,6 +185,7 @@ PyToken_TwoChars(int c1, int c2)
               case '|':
                   switch (c2) {
                   case '=': return VBAREQUAL;
          +        case '>': return VBARGREATER;
                   }
                   break;
               }

Adding grammar support
==========================================
TL;DR
-------------------------
* Current parser for Python uses a Parsing Expression Grammar (PEG) syntax.


In-Depth
-----------------------
.. image:: ../img/002pyast.png
  :width: 400
  :alt: Python Compilation Flow
