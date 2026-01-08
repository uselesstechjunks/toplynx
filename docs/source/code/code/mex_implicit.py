from sortedcontainers import SortedSet
import heapq

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
- why does it work even if we don't maintain it properly?
"""
class SmallestInfiniteSet1:
	def __init__(self):
		self.addedBack = SortedSet()
		self.poppedMax = 0

	def popSmallest(self) -> int:
		mex = self.addedBack.pop(0) if self.addedBack else self.poppedMax + 1
		self.poppedMax = max(self.poppedMax, mex)
		return mex

	def addBack(self, num: int) -> None:
		if num in self.addedBack:
			return
		if num > self.poppedMax:
			return
		self.addedBack.add(num)
	
	def __repr__(self):
		return f'SmallestInfiniteSet(poppedMax={self.poppedMax}, addedBack={self.addedBack})'

class SmallestInfiniteSet2:
	def __init__(self):
		self.addedBack = set()
		self.mexHeap = []
		self.poppedMax = 0

	def popSmallest(self) -> int:
		mex = None
		if self.mexHeap:
			mex = heapq.heappop(self.mexHeap)
			self.addedBack.remove(mex)
		else:
			mex = self.poppedMax + 1
		self.poppedMax = max(self.poppedMax, mex)
		return mex

	def addBack(self, num: int) -> None:
		if num in self.addedBack:
			return
		if num > self.poppedMax:
			return
		self.addedBack.add(num)
		heapq.heappush(self.mexHeap, num)
	
	def __repr__(self):
		return f'SmallestInfiniteSet(poppedMax={self.poppedMax}, addedBack={self.addedBack}, mexHeap={self.mexHeap})'

def test(obj):
	print(obj)
	for _ in range(5):
		print(f'next={obj.popSmallest()}')
		print(obj)
	
	obj.addBack(3)
	print(obj)
	obj.addBack(1)
	print(obj)
	obj.addBack(5)
	print(obj)
	obj.addBack(4)
	print(obj)
	obj.addBack(2)
	print(obj)

if __name__ == '__main__':
	test(SmallestInfiniteSet1())
	test(SmallestInfiniteSet2())
