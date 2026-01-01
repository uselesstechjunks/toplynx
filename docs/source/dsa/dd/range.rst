================================================================================
Order, Search and Range Queries
================================================================================
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Data Structures
--------------------------------------------------------------------------------
#. Segment Tree:

	- Sum, min, max, and custom range queries.
	- Lazy propagation for range updates.
	- Variants like mergeable segment trees.
	- Sample implementation

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/segtree.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. Fenwick Tree (Binary Indexed Tree):

	- Point updates and prefix/range queries.
	- Multidimensional Fenwick Trees.
	- Sample implementation

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/bit.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	- 2D: Sample implementation

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/bit2d.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. Sparse Table:

	- Efficient for immutable data (static range queries like min, max, or GCD).
#. Order Statistics Tree (Augmented BST or Fenwick Tree with Order Statistics):

	- Find kth smallest element.
	- Count of elements less than or greater than a given value.
	- Fundamental tree algorithms

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/bst.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	- Check if a tree is a valid BST

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/validbst.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	- Tree traversals with stack

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/inorder.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. RMQ (Range Minimum Query):

	- Hybrid solutions combining segment tree and sparse table for efficiency.
#. Wavelet Tree:

	- Handles range frequency queries and range kth order statistics.
#. Mo’s Algorithm:

	- Square-root decomposition for offline range queries.
#. Merge Sort Tree:

	- Efficient for range queries involving sorted data.
#. Interval Tree and KD-Tree:

	- For multidimensional range queries.
#. Monotonic Stack/Queue:

	- Span porblems in static data.
#. Augmented Trie:

	- For string search.
	- Example Problem: Design Search Autocomplete System with HitCount

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/searchautocomplete.py
		      :language: python
		      :tab-width: 2
		      :linenos:

Algorithms
--------------------------------------------------------------------------------
#. Binary search

	#. Define search space
	#. Define condition which specifies a contiguous range in that search space touching either ends.
	#. Decide whether to find the left boundary of that space or right.
	#. Choose whether to search for max from left or min from right.

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/binsearch.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	#. Application: Unsorted Array - `Finding Peak Element <https://leetcode.com/problems/find-peak-element/description/>`_

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/peak.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	#. Application: `Search Suggestions System <https://leetcode.com/problems/search-suggestions-system/description/>`_

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/searchsuggestion.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	#. Find max by checking condition on solution space (`Social Distancing <https://usaco.org/index.php?page=viewproblem2&cpid=1038>`_)

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/socdist.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. Binary Search Rotated

	#. Find Pivot

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/findmin_rot.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	#. Search in rotated

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/binsearch_rot.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. Divide-and-Conquer approaches (e.g., inversion count with merge sort).
#. Sliding window techniques (efficient for specific range problems).

	.. note::
		- fixed length
	
			- fixed sum with constant extra bookkeeping
			- fixed sum with auxiliary data structures
		- variable length
	
			- fixed sum with constant extra bookkeeping - aggregate >= value
			- fixed sum with auxiliary data structures - frequency, prefix sums -> dict, monotonic queue, bst
	.. attention::
		- sequential grouping
		- sequential criteria - longest, smallest, contained, largest, smallest

#. Two-pointer methods for range problems in sorted data.
#. Offline processing for batch queries using Mo's Algorithm or persistent data structures.
#. Cycle sort

	- `Missing Number <https://leetcode.com/problems/missing-number/>`_
	- `Find All Numbers Disappeared in an Array <https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/>`_
	- `Find the Duplicate Number <https://leetcode.com/problems/find-the-duplicate-number/>`_
#. Cycle detection: Floyd

	- `Find the Duplicate Number <https://leetcode.com/problems/find-the-duplicate-number/>`_

Fundamental Problems
--------------------------------------------------------------------------------
MEX - Minimum Excluded Element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Cycle sort - `First missing positive in range [1,n] <https://leetcode.com/problems/first-missing-positive/>`_
#. `Design data structure that pops smallest available numbers in infiite set with addback <https://leetcode.com/problems/smallest-number-in-infinite-set/>`_

	.. collapse:: Implicit MEX

	   .. literalinclude:: ../../code/smallest_infinite.py
	      :language: python
	      :tab-width: 2
	      :linenos:
#. `MEX on array with updates <https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/description/>`_
#. TODO: https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-i/description/
#. TODO: https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-ii/

Pointer Gynmastics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Rotate array

	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/rotatearr.py
	      :language: python
	      :tab-width: 2
	      :linenos:

Binary Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. `Minimum Number of Days to Make m Bouquets <https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/>`_

	- Problem: Given an array where each element represents the number of days it takes for a flower to bloom, and integers :math:`m` and :math:`k`, find the minimum number of days required to make :math:`m` bouquets, where each bouquet requires :math:`k` adjacent flowers.
	- Hints: Use binary search on the minimum days.

#. `Allocate Books (or Minimum Maximum Partition) <https://www.geeksforgeeks.org/allocate-minimum-number-pages/>`_

	- Problem: Given :math:`n` books and :math:`m` students, where each book has a certain number of pages, partition the books such that the maximum pages assigned to a student is minimized.
	- Hints: Binary search on the maximum pages.

#. `Koko Eating Bananas <https://leetcode.com/problems/koko-eating-bananas/>`_

	- Problem: Given :math:`n` piles of bananas and an integer :math:`h`, find the minimum eating speed :math:`k` such that Koko can finish all the bananas in :math:`h` hours.
	- Hints: Binary search on the eating speed.

#. `Find Median in a Row-Wise Sorted Matrix <https://www.geeksforgeeks.org/find-median-row-wise-sorted-matrix/>`_

	- Problem: Given a row-wise sorted matrix, find its median.
	- Hints: Use binary search on the value range, with a helper function to count elements smaller than or equal to the mid.

#. `Aggressive Cows (or Maximum Minimum Distance) <https://www.geeksforgeeks.org/assign-stalls-to-k-cows-to-maximize-the-minimum-distance-between-them/>`_

	- Problem: Given :math:`n` stalls and :math:`c` cows, place the cows in the stalls such that the minimum distance between any two cows is maximized.
	- Hints: Binary search on the minimum distance.

#. `Search in a Rotated Sorted Array <https://leetcode.com/problems/search-in-rotated-sorted-array/>`_

	- Problem: Given a rotated sorted array, find a target value in :math:`O(\log n)`.
	- Hints: Binary search with conditions to identify the rotated segment.

#. `Split Array Largest Sum <https://leetcode.com/problems/split-array-largest-sum/>`_

	- Problem: Split an array into :math:`m` non-empty subarrays to minimize the largest sum among the subarrays.
	- Hints: Binary search on the maximum subarray sum.

#. `Find Peak Element in an Unsorted Array <https://leetcode.com/problems/find-peak-element/>`_

	- Problem: Given an unsorted array, find a peak element (an element greater than its neighbors) in :math:`O(\log n)`.
	- Hints: Apply binary search with local comparison.

#. `Longest Subsequence with Limited Sum <https://leetcode.com/problems/longest-subsequence-with-limited-sum/>`_

	- Problem: Given an array and queries, for each query, find the maximum number of elements in the array whose sum is less than or equal to the query value.
	- Hints: Binary search with prefix sums.

#. `Minimize the Maximum Difference Between Pairs <https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/>`_

	- Problem: Given an array of integers and a number :math:`p`, partition the array into :math:`p` pairs such that the maximum absolute difference of any pair is minimized.
	- Hints: Binary search on the maximum difference.

#. `Maximize Minimum Distance Between Points <https://www.geeksforgeeks.org/place-k-elements-such-that-minimum-distance-is-maximized/>`_

	- Problem: Given points on a line and a fixed number of segments, maximize the minimum distance between the segment boundaries.
	- Hints: Binary search on the answer.

Inversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. `Shortest Unsorted Continuous Subarray to Sort <https://leetcode.com/problems/shortest-unsorted-continuous-subarray/description/>`_

	.. collapse:: Two approaches - Two pointers, monotonic stack

	   .. literalinclude:: ../../code/shortestUnsortedSubarray.py
	      :language: python
	      :tab-width: 2
	      :linenos:
#. `Shortest Unsorted Continuous Subarray to Reove <https://leetcode.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/>`_

	.. collapse:: Two pointers

	   .. literalinclude:: ../../code/shortestUnsortedRemove.py
	      :language: python
	      :tab-width: 2
	      :linenos:

Order Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. `Kth Largest/Smallest Element in a Stream <https://leetcode.com/problems/kth-largest-element-in-a-stream/>`_

	- Maintain the top k elements in a stream of data.
	- Hints: Leverage min-heaps or order statistics trees.

#. `Kth Largest/Smallest Element in an Array <https://leetcode.com/problems/kth-largest-element-in-an-array/description/>`_

	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/quickselect.py
	      :language: python
	      :tab-width: 2
	      :linenos:

#. `Find the Median of a Running Stream <https://leetcode.com/problems/find-median-from-data-stream/>`_

	- Use two heaps (max-heap and min-heap) for efficiency.

#. `Count of Smaller/Larger Numbers After Self <https://leetcode.com/problems/count-of-smaller-numbers-after-self/>`_

	- Given an array, for each element, count how many elements are smaller/larger to its right.
	- Solution: Fenwick Tree, segment tree, or merge sort.

#. `Find the Kth Largest Element in an Unsorted Array <https://leetcode.com/problems/kth-largest-element-in-an-array/>`_

	- Variants where you cannot sort directly (e.g., use Quickselect).

#. kth Element in the Cartesian Product

	- Problem: Given two sorted arrays :math:`A` and :math:`B`, find the :math:`k`-th smallest tuple :math:`(a, b)` in :math:`A \times B` under the order relation defined above (based on the sum :math:`a + b`). 
	- Hints: Use a min-heap with tuples to track possible combinations efficiently.

#. `Median in a Sliding Window <https://leetcode.com/problems/sliding-window-median/>`_

	- Problem: Given an array of integers and a sliding window of size :math:`k`, find the median of each window as it slides from left to right.
	- Hints: Use two heaps (max-heap and min-heap) to dynamically maintain the window.

#. `Inversion Count in Subarrays <https://www.geeksforgeeks.org/counting-inversions-in-an-subarrays/>`_

	- Problem: For an array :math:`A`, process :math:`q` queries of the form :math:`(L, R)` where you need to count the number of inversions in the subarray :math:`A[L:R]`.
	- Hints: Use a segment tree with merge-sort logic at each node.

#. Range k-th Smallest Element

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R, k)`, find the :math:`k`-th smallest element in the range :math:`[L, R]`.
	- Hints: Use a merge sort tree or wavelet tree for efficient query processing.

#. Count of Numbers in Range with a Given Frequency

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R, F)`, count how many numbers in the range :math:`[L, R]` appear exactly :math:`F` times.
	- Hints: Use Mo’s Algorithm with frequency tracking or segment trees with custom nodes.

Range Query Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Range Sum Query with Updates

	- Hints: Solve using segment trees or Fenwick trees with range updates.

#. Range Minimum/Maximum Query

	- Hints: Solve using segment trees, sparse tables, or hybrid methods.

#. Dynamic Range Median Queries

	- Hints: Maintain a dynamic dataset and answer queries for the median of a range.

#. Range XOR Query

	- Hints: Solve using segment trees.

#. Sum of Range Products

	- Hints: Given an array, answer the sum of products of all pairs in the range [L, R].

#. Number of Distinct Elements in Range

	- Hints: Use Mo’s Algorithm or a segment tree with a map structure.

#. Range Frequency Query

	- Hints: Solve using a wavelet tree or merge sort tree.

#. Dynamic Range Median Queries

	- Problem: Maintain a dynamic array supporting

		1. Insertion of an element.
		2. Deletion of an element.
		3. Querying the median of any range :math:`[L, R]`.
	- Hints: Combine balanced BST or heaps with a range query structure like segment trees.

#. Range XOR with Updates

	- Problem: Given an array of integers, process the following operations efficiently

		1. Update the :math:`i` -th element to :math:`x`.
		2. Query the XOR of elements in the range :math:`[L, R]`.
	- Hints: Use a segment tree with XOR as the operation and point updates.

#. Maximum Frequency in a Range

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R)`, find the most frequent number in the range :math:`[L, R]`.
	- Hints: Use a segment tree with frequency maps stored at each node.

#. Maximum Subarray Sum in a Range

	- Problem: Process queries of the form :math:`(L, R)`, where you must find the maximum subarray sum in the range :math:`[L, R]`.
	- Hints: Augment the segment tree to store max subarray sums and handle overlapping subranges efficiently.

#. Range Updates with a Custom Function

	- Problem: Design a data structure to efficiently handle

		1. Updates: Apply a custom function :math:`f(x)` to all elements in the range :math:`[L, R]`.
		2. Queries: Retrieve the sum of all elements in the range :math:`[L, R]`.
	- Hints: Use a segment tree with lazy propagation where :math:`f(x)` can be propagated efficiently.

Hybrid Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Dynamic Skyline Problem

	- Given a list of intervals, dynamically insert or delete intervals and determine the current skyline.

#. Maximum Sum Rectangle in a 2D Matrix

	- Use a 1D segment tree approach for optimal results.

#. Range GCD Query

	- Find the GCD of elements in the range [L, R] using a segment tree or sparse table.

#. Number of Rectangles Containing a Point

	- Problem: You are given a list of :math:`n` rectangles (defined by two opposite corners) and :math:`q` points. For each point, count how many rectangles contain it.
	- Hints: Use a segment tree or 2D Fenwick Tree to maintain active ranges as you sweep through one coordinate.

#. Dynamic Skyline

	- Problem: Maintain the skyline (maximum height of buildings seen from a distance) as you dynamically add and remove buildings.
	- Hints: Use an interval tree or segment tree to handle dynamic range updates efficiently.

#. Count Subarrays with Given Sum in Range

	- Problem: For :math:`q` queries :math:`(L, R, S)`, count how many contiguous subarrays in the range :math:`[L, R]` have a sum equal to :math:`S`.
	- Hints: Use prefix sums with a Fenwick Tree to count valid subarray sums efficiently.

#. Maximum Overlap of Intervals

	- Problem: Given a list of intervals, process :math:`q` queries to find the maximum overlap of intervals in a given range :math:`[L, R]`.
	- Hints: Use a difference array combined with prefix sums or a segment tree for dynamic updates.

#. Submatrix Sum Queries

	- Problem: Given a 2D grid, process

		1. Updates: Add a value to all elements in a submatrix.
		2. Queries: Find the sum of elements in any submatrix.
	- Hints: Use a 2D Fenwick Tree or segment tree for efficient query and update operations.

Problems Using Monotonic Stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Largest Rectangle in Histogram

	- Problem: Given an array of heights representing a histogram, find the area of the largest rectangle.
	- Hints: Use a monotonic stack to track bars in increasing order.
	- Sample implementation

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/maxhist.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	- Related: Maximum rectangle in binary matrix. Can be reduced to above.

		.. collapse:: Expand Code
	
		   .. literalinclude:: ../../code/maxrect.py
		      :language: python
		      :tab-width: 2
		      :linenos:
#. Trapping Rain Water

	- Problem: Given an array representing heights, calculate how much water can be trapped after it rains.
	- Hints: Use a monotonic stack to find the bounds of trapped water.

#. Next Greater Element (NGE)

	- Problem: For an array, find the next greater element for each element.
	- Hints: Traverse from the end and use a monotonic stack to maintain greater elements.

#. Next Smaller Element

	- Problem: For an array, find the next smaller element for each element.
	- Hints: Similar to NGE, but with a decreasing monotonic stack.

#. Sum of Subarray Minimums

	- Problem: Given an array, find the sum of the minimum values of all subarrays.
	- Hints: Use a monotonic stack to find the nearest smaller elements on both sides.

#. 132 Pattern

	- Problem: Find if there exists a 132 pattern in an array.
	- Hints: Use a monotonic stack to maintain potential "3" values while iterating.

#. Daily Temperatures

	- Problem: For each day's temperature, find how many days you’d have to wait for a warmer temperature.
	- Hints: Monotonic stack tracks indices of temperatures.

#. Asteroid Collision

	- Problem: Simulate asteroid collisions where larger ones destroy smaller ones.
	- Hints: Use a monotonic stack to simulate collisions.

Problems Using Monotonic Queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. `Sliding Window Maximum <https://leetcode.com/problems/sliding-window-maximum/description>`_

	- Problem: Find the maximum element in every sliding window of size :math:`k`.
	- Hints: Maintain a monotonic queue to store potential maxima.

#. `Shortest Subarray with Sum at Least K <https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/>`_

	- Problem: Given an array, find the shortest subarray with a sum :math:`\geq K`.
	- Hints: Use a monotonic queue to optimize prefix sums.

		.. collapse:: Monotonic queue for rightmost left index
	
		   .. literalinclude:: ../../code/shortestsubarrsumk.py
		      :language: python
		      :tab-width: 2
		      :linenos:
	- This can also be solved using segment tree but it's suboptimal

		.. collapse:: Shortest Subarray with Sum at Least K
	
		   .. literalinclude:: ../../code/minlencumsum.py
		      :language: python
		      :tab-width: 2
		      :linenos:

Interview Problems
--------------------------------------------------------------------------------
#. Sliding Window Maximum

	#. Basic Variant
	
		- Problem: Find the maximum element in every sliding window of size :math:`k` in an array.
		- Hints: Use a monotonic deque to store indices of potential maxima, maintaining decreasing order.
	
	#. Dynamic Data (Real-Time Updates) 
	
		- Change: The array is dynamic, and elements can be added/removed in real-time. 
		- Hints: Use a Segment Tree or Fenwick Tree to track maxima in specific ranges. 

	#. Multiple Queries 
	
		- Change: Instead of just one pass, answer multiple queries of the form :math:`[L, R]` to find the maximum in subarrays. 
		- Hints: Preprocess with a Sparse Table (for static queries) or Segment Tree (for dynamic updates). 

#. Largest Rectangle in Histogram

	#. Basic Variant
	
		- Problem: Find the area of the largest rectangle that can be formed in a histogram. 
		- Hints: Use a monotonic stack to find the next smaller and previous smaller heights for each bar.

	#. 2D Matrix (Maximal Rectangle) 
	
		- Change: Extend the to a binary matrix to find the largest rectangle containing only 1s. 
		- Hints: Treat each row as a histogram and use the stack approach iteratively.

	#. Dynamic Histogram Updates 
	
		- Change: Allow updates to histogram heights and dynamically compute the largest rectangle. 
		- Hints: Use a Segment Tree to store and query the largest rectangle efficiently. 

#. Trapping Rain Water

	#. Basic Variant
	
		- Problem: Given an array of heights, calculate the total water trapped after rain. 
		- Hints: Use two-pointer technique or monotonic stack to find bounds for water levels.
		
			.. collapse:: 
		
			   .. literalinclude:: ../../code/trap.py
			      :language: python
			      :tab-width: 2
			      :linenos:
	
	#. Dynamic Updates 
	
		- Change: Heights can be updated, and the total trapped water must be recalculated efficiently. 
		- Hints: Use a Fenwick Tree to maintain prefix max values and efficiently compute water levels. 

	#. Multiple Queries 
	
		- Change: For multiple ranges :math:`[L, R]`, calculate the water trapped in those ranges. 
		- Hints: Precompute prefix max/min values for efficient range queries. 

#. Next Greater Element (NGE)

	#. Basic Variant
	
		- Problem: For an array, find the next greater element for each element. 
		- Hints: Use a monotonic stack while iterating from the end of the array.
	
	#. Circular Array 
	
		- Change: The array is circular, so elements wrap around. 
		- Hints: Simulate wrapping by iterating twice through the array with a stack. 

	#. Dynamic Updates 
	
		- Change: Support updates to the array and answer NGE queries efficiently. 
		- Hints: Use a Segment Tree or Ordered Set to dynamically track and query next greater elements. 

#. Range Sum Query

	#. Basic Variant
	
		- Problem: Given an array, calculate the sum of elements in a range :math:`[L, R]` . 
		- Hints: Use a prefix sum array for efficient range queries.

	#. Dynamic Updates 
	
		- Change: Allow updates to the array and answer range sum queries. 
		- Hints: Use a Fenwick Tree or Segment Tree for :math:`O(\log n)` updates and queries. 

	#. Range Sum with Modulo or Constraints 
	
		- Change: Add a constraint to compute range sums modulo :math:`k`, or find if the sum in a range satisfies certain conditions. 
		- Hints: Use a Segment Tree with custom lazy propagation to handle constraints. 

#. Stock Span Problem

	#. Basic Variant
	
		- Problem: For each day’s stock price, find the number of consecutive days before it with a price less than or equal to the current day. 
		- Hints: Use a monotonic stack to track indices.
	
	#. Dynamic Price Updates 
	
		- Change: Allow updates to stock prices and recalculate the span dynamically. 
		- Hints: Use a Segment Tree to maintain range queries for stock prices. 

	#. Multiple Queries for Ranges 
	
		- Change: Answer span queries for multiple subranges :math:`[L, R]` . 
		- Hints: Combine Segment Tree or Sparse Table with preprocessing for efficient queries. 

#. Sum of Subarray Minimums

	#. Basic Variant
	
		- Problem: Find the sum of minimum values of all subarrays of an array. 
		- Hints: Use a monotonic stack to find the nearest smaller elements on both sides.
	
	#. Dynamic Array Updates 
	
		- Change: Support updates to array elements and recompute the sum of subarray minimums. 
		- Hints: Use a Segment Tree to track minimums and their contributions dynamically. 

	#. Additional Constraints 
	
		- Change: Add constraints like subarray sums must be within a given range or subarray lengths must be limited. 
		- Hints: Combine a Fenwick Tree with constraint checks for efficient processing. 

#. Binary Search Variants

	#. Basic Variant
	
		- Problem: Find an element in a sorted array using binary search. 
		- Hints: Divide and conquer to find the target element.
	
	#. Rotated Sorted Array 
	
		- Change: The array is rotated; find the target element. 
		- Hints: Modify binary search to handle rotations. 

	#. Minimum in Rotated Sorted Array with Duplicates 
	
		- Change: The rotated array contains duplicates. 
		- Hints: Adapt binary search with careful handling of duplicate elements. 

	#. Find Median in a Stream 
	
		- Change: Support dynamic updates and find the median efficiently. 
		- Hints: Use a combination of Heaps or Balanced BSTs for dynamic median maintenance. 
