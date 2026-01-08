from random import randint
from collections import Counter

class RangeSumTree:
	def __init__(self, nums: List[int]):
		def build():
			for i, val in enumerate(nums):
				self.update_impl(1, 0, self.n-1, i, val)

		self.n = len(nums)
		self.tree = [0] * 4 * self.n
		build()

	def update_impl(self, root, t_left, t_right, index, val):
		"""
		Updates the leaf value at index and then subsequently intermeidate tree values.

		Params:
			root: root index of the tree
			t_left: left index in the original array spanned by the current subtree
      t_right: right index in the original array spanned by the current subtree
			index: index in the array for update
			val: value to be updated at index
		Returns:
			None
		"""
		if t_left > t_right:
			return

		if t_left == t_right:
			self.tree[root] = val
			return

		t_mid = (t_left + t_right) // 2
		
		left_child = self.left_child_index(root)
		right_child = self.right_child_index(root)

		if index <= t_mid:
			self.update_impl(left_child, t_left, t_mid, index, val)
		else:
			self.update_impl(right_child, t_mid + 1, t_right, index, val)
		
		self.tree[root] = self.tree[left_child] + self.tree[right_child]

	def left_child_index(self, root):
		return root * 2

	def right_child_index(self, root):
		return root * 2 + 1

	def update(self, index: int, val: int) -> None:
		self.update_impl(1, 0, self.n-1, index, val)

	def query(self, root, t_left, t_right, left, right):
		"""
		Returns the range sum in the original array between left and right indices (both included)

		Params:
			root: root index of the tree
   		t_left: left index in the original array spanned by the current subtree
      t_right: right index in the original array spanned by the current subtree
			left: left index of the range in the array
			right: right index of the range in the array
		Returns:
			Sum of the numbers within range [left, right]
		"""
		if t_left > right or t_right < left:
			return 0
		
		if t_left == left and t_right == right:
			return self.tree[root]
		
		t_mid = (t_left + t_right) // 2
		left_child = self.left_child_index(root)
		right_child = self.right_child_index(root)

		if right <= t_mid:
			return self.query(left_child, t_left, t_mid, left, right)
		if t_mid < left:
			return self.query(right_child, t_mid + 1, t_right, left, right)
		else:
			return self.query(left_child, t_left, t_mid, left, t_mid) + self.query(right_child, t_mid + 1, t_right, t_mid + 1, right)

	def sumRange(self, left: int, right: int) -> int:
		return self.query(1, 0, self.n-1, left, right)

"""
The following is a more generic version of segment tree to be used for multiple range query tasks.
"""
class SegmentTree:
	def __init__(self, nums, combine):
		def build(root_index, range_left, range_right):
			nonlocal nums
			if range_left > range_right:
				return
			if range_left == range_right:
				self.tree[root_index] = nums[range_left]
				return
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			build(left_index, range_left, range_mid)
			build(right_index, range_mid + 1, range_right)
			self.tree[root_index] = self.combine(self.tree[left_index], self.tree[right_index])

		self.n = len(nums)
		self.tree = [0] * 4 * self.n
		self.combine = combine
		build(1, 0, self.n-1)

	def update(self, index: int, val: int) -> None:
		def impl(root_index, range_left, range_right, insert_index, val):
			if range_left > range_right:
				return
			if range_left == range_right:
				self.tree[root_index] = val
				return
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			if insert_index <= range_mid:
				impl(left_index, range_left, range_mid, insert_index, val)
			else:
				impl(right_index, range_mid + 1, range_right, insert_index, val)
			self.tree[root_index] = self.combine(self.tree[left_index], self.tree[right_index])
		
		impl(1, 0, self.n-1, index, val)

	def sumRange(self, left: int, right: int) -> int:
		def impl(root_index, range_left, range_right, left, right):
			if range_left > range_right:
				return 0
			if range_left == left and range_right == right:
				return self.tree[root_index]
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			if right <= range_mid:
				return impl(left_index, range_left, range_mid, left, right)
			elif range_mid < left:
				return impl(right_index, range_mid+1, range_right, left, right)
			return impl(left_index, range_left, range_mid, left, range_mid) + impl(right_index, range_mid + 1, range_right, range_mid + 1, right)
		return impl(1, 0, self.n-1, left, right)

class RangeSum(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: x+y)

class RangeMin(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: min(x, y))

class RangeFrequency(SegmentTree):
    def __init__(self, arr):
        counts = [(x,1) for x in arr]
        def combine(x, y):
            if x[0] < y[0]:
                return x
            if x[0] > y[0]:
                return y
            return (x[0], x[1]+y[1])
        super().__init__(arr=counts, combine=combine)

class RangeOrderStatistics(SegmentTree):
    def __init__(self, arr):
        counts = [1 if x == 0 else 0 for x in arr]
        super().__init__(arr=counts, combine=lambda x,y: x+y)
    def find_kth_idx(self, k):
        def impl(i, tl, tr, k):
            if k > self.tree[i]:
                return -1
            if tl == tr:
                return tl
            tm = tl+(tr-tl)//2
            if k <= self.tree[2*i]:
                return impl(2*i, tl, tm, k)
            else:
                return impl(2*i+1, tm+1, tr, k-self.tree[2*i])
        return impl(1, 0, self.n-1, k)
