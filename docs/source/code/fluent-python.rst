Fluent Python
##########################################################################

Chapter 1: Python data model
**************************************************

Special Methods are cool and are written like ``__special_method__``. They are called by the Python framework to enable various functionality. We're not supposed to call them directly (except for ``__init__`` to call superclass constructor maybe).

Special Methods for Collection Objects
============================================

#. ``__getitem__``: Random-access operator.
#. ``__len__``: Provides a functionality to obtain length, with ``len(object)``.

Together, they allow the framework to do the following:

  #. Access any element with index (``object[key]``).
  #. Access n-th element from last with negative indexing (``object[-index_from_last]``).
  #. Obtain random element using ``random.choice``.

      .. code-block: python

          from random import choice

          item = choice(object) # returns a random item from object

  #. Slicing (``object[key1:key2]``) (TODO read more about slicing).
  #. Make the object iterable.

      .. code-block:: python
      
          for item in object:
            do_stuff(item)

  #. Generate a reverse iterator.
  
      .. code-block:: python
      
          for item in reverse(object):
            do_stuff(item)

  #. Enable querying for existance of an item by performing sequential scanning.
  
      .. note::
          Implement a ``__contains__`` function, then ``in`` would use that one.

  #. If we provice a custom ``item_ranker`` function, then we can also sort the items in the object using ``sorted`` interface.
  
      .. code-block:: python
          
          def item_ranker(item):
            return rank(item)
          
          for item in sorted(object, item_ranker):
            do_stuff(item)
            
            
Special Methods for Numeric Objects
============================================

#. ``__add__(self, other)`` implements ``self + other``.
#. ``__mul__(self, other)`` implements ``self * other``.
#. ``__abs__(self)`` implements ``abs(self)``.
#. ``__repr__(self)`` implements a printable representation (enables ``print(object)`` and usage in ``%r``).
#. ``__str__(self)`` implements a string representation (enables ``str(object)`` and usage in ``%s``).
#. ``__bool__(self)`` returns ``True/False`` to be used in ``if/else/and/or/not``.

    .. note::
    
      #. ``__repr__`` usually encodes a hint about how to construct an object of the class as-well (e.g. ``MyClass(a=x, b=y)``).
      #. ``__str__`` may represent it as ``[x,y]``. 
      #. In absence of a ``__str__``, it falls back to ``__repr__``.
      #. Delegate the task of representing items in object by using ``item!r`` inside format string.

          .. code-block:: python

              def __repr__(self):
                return f'MyClass(a={self.a!r}, b={self.b!r})'


Collections API
==============================

Refere to image 1.2 in the book for UML diagram. In a nutshell, any collection object should implement:

  #. ``Iterable`` to enable ``for``.
  #. ``Sized`` to enable ``len``.
  #. ``Container`` to enable ``in``.
  
Specialization of Collection class:
  
  #. ``Sequence``
  #. ``Mapping``
  #. ``Set``
  
Refer to table 1-1 and 1-2 in the book for a list of special methods for various functionalities.

Chapter 2: An Array of Sequences
**************************************************

.. note::
    Each python object contains metadata fields (such as reference counts, type-information).

Sequences provide common functionalities such as iteration, slicing, sorting and concatenation.

Classification of Sequences
=========================================

  #. Storage:
  
    #. Container Sequences: Contains pointers to python objects, potentially heterogeneous. Example: ``list/tuple``.
    #. Flat Sequences: Contains a contiguous chuck of memory for homogenous python objects. Example: ``str/array``.
    
  #. Mulatibility:
  
    #. Mutable Sequences: Items can be updated in-place. Example: ``list/array``.
    #. Immutable Sequences: Items cannot be updated. Creates a new instance instead. Example: ``tuple/str``.

