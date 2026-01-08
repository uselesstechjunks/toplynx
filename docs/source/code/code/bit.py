class BinaryIndexedTree:

	def __init__(self, nums: List[int]):
		def build():
			for i in range(self.n):
				self.add(i, self.nums[i])

		self.n = len(nums)
		self.nums = nums
		self.bit = [0]*self.n
		build()
	
	def nextIndex(self, index):
		return index | (index + 1)

	def prevIndex(self, index):
		return (index & (index + 1)) - 1
	
	def add(self, index, delta):
		curr_index = index
		while curr_index < self.n:
			self.bit[curr_index] += delta
			curr_index = self.nextIndex(curr_index)
	
	def sum(self, right):
		res = 0
		curr_index = right
		while curr_index >= 0:
			res += self.bit[curr_index]
			curr_index = self.prevIndex(curr_index)
		return res

	def update(self, index: int, val: int) -> None:
		delta = val - self.nums[index]
		self.nums[index] = val
		self.add(index, delta)

	def sumRange(self, left: int, right: int) -> int:
		return self.sum(right) - self.sum(left-1)
