from sortedcontainers import SortedSet

"""
Think of it as if we're forming an array by calling popSmallest.
On that array, we're calling MEX operation.

Observations
- The array isn't explicit here.
- we know that the array doesn't contain duplicates.
- array can only contain unique elements.
- we won't need to store a hashmap for counts.

Bookkeeping
- keep track of the mex elements.
- keep track of the range of popped elements
- added back elements form holes in that range
- anything that's in that hole is smaller than popped max
	- needs to be sorted
	- needs fast retrieval
- how does addback change the popped range?
"""

class MexImplSetAndHeap:
	def __init__(self):
		self.is_missing = set()
		self.missing_minheap = []
		self.max_added = 0

	def popSmallest(self) -> int:
		mex = self.max_added + 1
		if self.is_missing:
			mex = min(mex, heapq.heappop(self.missing_minheap))
			self.is_missing.remove(mex)
		self.max_added = max(self.max_added, mex)
		return mex
	
	def addBack(self, num: int) -> None:
		if num <= self.max_added and num not in self.is_missing:
			self.is_missing.add(num)
			heapq.heappush(self.missing_minheap, num)

class MexImplSortedSet:
	def __init__(self):
		self.missing_numbers = SortedSet()
		self.max_added = 0

	def popSmallest(self) -> int:
		mex = self.missing_numbers.pop(0) if self.missing_numbers else self.max_added + 1
		self.max_added = max(self.max_added, mex)
		return mex

	def addBack(self, num: int) -> None:
		if num <= self.max_added:
			self.missing_numbers.add(num)

class SmallestInfiniteSet:

	def __init__(self):
		# self.impl = MexImplSortedSet()
		self.impl = MexImplSetAndHeap()

	def popSmallest(self) -> int:
		return self.impl.popSmallest()

	def addBack(self, num: int) -> None:
		self.impl.addBack(num)


# Your SmallestInfiniteSet object will be instantiated and called as such:
# obj = SmallestInfiniteSet()
# param_1 = obj.popSmallest()
# obj.addBack(num)