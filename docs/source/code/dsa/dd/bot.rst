*********************************************************************************
Bag of Tricks
*********************************************************************************
Goal: Map the problem to known tasks.

Thought Process
=====================================================================
.. note::
	#. Does it form a group, chain, tree, graph? - union find, parent-child hashmap (set if parent-child is implicit).
	#. Does it have a search space? Is there a contiguous segment within this search space for which a condition satisfies? Is the problem about finding the boundaries of that segment? - binary search.
	#. Does the quantity grow monotonically with number of elements? - VLW.
	#. What bookkeeping is required? What involves recomputation? What else can we track to avoid it? - hashmap, bst, stack, queue, heap.
	#. Can we solve it in parts and combine the results? - divide and conquer, recursion, DP.
	#. What choices can be greedily eliminated? - two pointers, greedy, quicksort partitioning.

Find something
=================================================================================
Types of Queries
---------------------------------------------------------------------------------
#. OGQ - Optimal Goal Query: VLW - variable length window + aux bookkeeping (monotonic goal), FLW - fixed length window (works for non-monotone)
#. RSQ - Range Sum Query: :math:`\sum(l,r)`: Prefix sum, BIT, segment tree
#. MSQ - Maximum Sum Query: :math:`[0,n)->\max(\sum(l,r))`: Prefix sum->BIT, VLW->Kadane, divide and conquer->segment tree
#. RMQ - Range Min Query: :math:`\min(l,r)`: [unordered] monotonic stack, monotonic queue (VLW), Cartesian tree, segment tree, [ordered] binary search, BST (VLW)
#. RFQ - Range Frequency Query: :math:`c(l,r,key)`: Dict, segment tree
#. EEQ - Earlier Existance Query: set, dict, bitmap
#. LSQ - Latest Smaller Query: :math:`\max(l | l<r, v(l)<v(r))`: Monotonic stack (v1), Cartesian tree
#. ESQ - Earliest Smaller Query: :math:`\min(l | l<r, v(l)<v(r))`: Monotonic stack (v2), (???) inversions/pointers?
#. SEQ - Smallest Earlier Query: :math:`\min(v(l) | l<r, v(l)<v(r))`: pointer, bst, heap
#. TKQ - Top K Query: heap
#. RIQ - Range Intersect Query: Given point, find ranges that contains it: Interval tree
#. ROQ - Range Overlap Query: Find intervals that overlaps with given range: Sorting + binary search, sorting + stack
#. ECQ - Equivalence Class Query: Whether (x,y) belonds to the same group: Union find, dict for parent-child
#. MEX - Minimum Excluded Element: (???)
#. LCS - Longest common/increasing/palindromic subsequence: VLW, DP
#. RUQ - Range Update Query: Prefix sum->BIT (+delta at begin, -delta at end), segment tree

General Techniques
---------------------------------------------------------------------------------
Ordered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Values explicit - vanilla Binary search.

	#. Range that satisfies the condition is on the left of the search space. Pruned range might move past it. Need to allow right to move past left.
	#. Range that satisfies the condition is on the right of the search space. Pruned range always contains it. Always reduces to size 1.
	#. Range that satisfies the condition is in the middle of the search space.
#. Values NOT explicit 

	#. Values bounded? Binary search on range. Condition check O(T(n)) -> total T(n)lg(W), W=precision
	#. Bounded either above/below? One way binary search from one end - move from i -> 2i or i -> i/2
	#. Target forms a convex function? Optimal exists at root? 

		#. Can compute gradient? GD.
		#. Can compute Hessian? Newton.
Unordered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Linear search
#. Divide & conquer 
#. Use bookkeeping techniques

Bookkeeping
---------------------------------------------------------------------------------
#. KV map - multiple O(1) use-cases

	- freq counts - histogram delta >= 0, distinct count >=k, min freq count >= k, majority-non majority count (max freq - sum V)
	- detect earlier occurance, obtain earliest/latest occurance with paired value
	- parent-child relation (stores parent/child pointer), alternative to union-find
#. Stack (maintains insert seq in rev + can maintain first k inserted + latest in O(1))
#. Queue (maintains insert seq + can maintain last k inserted + earliest/latest in O(1))
#. Dequeue (maintains insert seq + can maintain first+last k inserted + earliest/latest in O(1))
#. BST (all earlier values searchable in O(lg n) - doesn't maintain insert seq) - sortedcontainers.SortedList
#. Order statistics tree (???)
#. Heap (smallest/largest values from earlier range in O(1) + can maintain k smallest/largest - doesn't maintain insert seq)
#. Cartesian tree (RMQ tasks) - heap with insert seq: range min at root O(1). Constructive requires stack. Unbalanced.
#. Monotonic stack - 2 types 

	#. Type I: RMQ (Range Min/Max Query): Simulates Cartesian tree.

		#. Maintains longest monotonic subsequence from min (max) (including curr) ending at curr.
		#. At finish, corresponds to the rightmost branch of a Cartesian tree.
		#. Everything that has ever been on the stack is the min of some range. This covers all possible range mins.
		#. For every curr, all larger (smaller) values are popped - curr is RM of everything since popped.
		#. Once pushed, top is range min (max) of [S[-2]+1, top]. S[-2] is range min of [S[-3]+1, top]		
		#. Bot is range min (max) for [0, top] (i.e., root of the Cartesian tree)
		#. Each value gets to be at the stack at some point.
	#. Type II: ESQ (Earliest Smaller/Larger Query)

		#. Maintains longest monotonic subsequence from first element.
		#. Everything that comes after, only pushed onto the stack if it's larger (smaller)
#. Monotonic queue - Same as monotonic stack except it works for sliding window as we can skip ranges by popping root (at front).
#. Min (max) stack (maintains range min (max) for [0, curr] at top + keeps all elements + obtain in O(1))
#. Min (max) queue (maintains range min (max) for [0, curr] at back + keeps all elements + obtain in O(1))
#. Segment tree (RSQ/RMQ, all subarray sums with prefix/suffix/sum in tree) - mutable, extends to 2d
#. Interval tree (find value in range)
#. Multidimensional - KD tree
#. Binary indexed tree (???) - mutable
#. Sparse table (RMQ)	
#. Union find (equivalence classes)
#. Trie (prefix matching)
#. String hashing - Rabin Karp
#. Make bookkeeping faster - sqrt decomposition

Count something
=================================================================================
#. Can we count compliment instead?

Modify something
=================================================================================
#. Two pointers + swap
#. Dutch national flag

Schedule something
=================================================================================
#. Priority queue + optional external dict for value - greedy
#. [Tarjan][Kahn] Topological sort

Assign something
=================================================================================
#. Two pointers
#. [Kuhn] Maximal bipartite matching

Optimise something
=================================================================================
#. DP - Classic problems

	#. 0-1 knapsack
	#. Complete knapsack
	#. Multiple knapsack
	#. Monotone queue optimization
	#. Subset sum
	#. Longest common subsequence
	#. Longest increasing subsequence (LIS)
	#. Longest palindromic subsequence
	#. Rod cutting
	#. Edit distance
	#. Counting paths in a 2D array
	#. Longest Path in DAG
	#. Divide and conquer DP
	#. Knuth's optimisation
	#. ASSP [Floyd Warshall]
#. Greedy 

	#. Two pointers
	#. Sliding window
	#. Shortest path - SSSP [Dijkstra][Bellman Ford]
	#. Lightest edge - MST [Prim][Kruskal]

Check connectivity, grouping & cyclic dependencies
=================================================================================
#. Tortoise & hare algorithm
#. BFS for bipartite detection
#. DFS with edge classification, union-find
#. Lowest common ancestor - tree/graph - [Euler's tour],[Tarjan],[Farach-Colton and Bender]
#. Connected components
#. Articulation vertex and biconneted components
#. [Kosaraju] Strongly connected components
#. Eulerian circuit for cycle visiting all vertices

Combine something
=================================================================================
#. Backtracking

Design something 
=================================================================================
#. Mostly bookkeeping

Validate something
=================================================================================
#. Paring problems - Stack
#. Regex problems - DP

Involves intervals
=================================================================================
#. Sort them - overlap check left-end >= right-start
#. Sort by start - benefit (???)
#. Sort by end - benefit (???)
