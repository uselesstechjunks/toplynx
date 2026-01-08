def permute(self, nums: List[int]) -> List[List[int]]:
	def backtrack(index, n):
		nonlocal res
		if index == n:
			res.append(nums[:])
			return
		for i in range(index, n):
			nums[i], nums[index] = nums[index], nums[i]
			backtrack(index + 1, n)
			nums[i], nums[index] = nums[index], nums[i]
	res = []
	backtrack(0, len(nums))
	return res

def combine(self, n: int, k: int) -> List[List[int]]:
	def optimized():
		""" prevents wasteful subset formation with additional state information """
		# [1,2,3]      curr                start   range 
		# remaining=2, []                      1-> [1,2]
		# remaining=1, [1],        [2]         2-> [2,3], 3 -> [3]
		# remaining=0, [1,2],[1,3],[2,3]
		def backtrack(curr, start):
			nonlocal res
			remaining = k - len(curr)
			if not remaining:
				res.append(curr[:])
				return
			for num in range(start, n - remaining + 2):
				curr.append(num)
				backtrack(curr, num + 1)
				curr.pop()
		res = []
		backtrack([], 1)
		return res
		"""
  		with indices
    		def backtrack(curr, start_index):
			nonlocal nums, res
			remaining = k - len(curr)
			if not remaining:
				res.append(curr[:])
				return
			for index in range(start_index, len(nums) - remaining + 1):
				curr.append(nums[index])
				backtrack(curr, index + 1)
				curr.pop()		
		res = []
		nums = list(range(1, n+1))
		backtrack([], 0)
		return res
  		"""

	def simple():
		""" In this formulation, subsets get wasted, e.g. [3] """
		# [1,2,3]                    start    range 
		# k=0, []                      1-> [1,2,3]
		# k=1, [1],        [2], [3]    2-> [2,3], 3->[3]
		# k=2, [1,2],[1,3],[2,3]
		def backtrack(curr, start):
			nonlocal res
			if len(curr) == k:
				res.append(curr[:])
				return
			for num in range(start, n + 1):
				curr.append(num)
				""" NOTE: NOT START + 1"""
				backtrack(curr, num + 1)
				curr.pop()
		res = []
		backtrack([], 1)
		return res

def generateParenthesis(self, n: int) -> List[str]:
	# n = 3               left   right
	# (                     1      0
	# ((,      ()           1      1
	# (((,((), ()(          2      1
	# (((,(()(,()((,()()
	""" 
	key idea:
	(1) when to add ( => whenever there are still left paranthesis left to add
	(2) when to add ) => whenever left count is greater than right count
	"""
	def backtrack(curr, left_count, right_count):
		nonlocal res
		if right_count == n:
			res.append(''.join(curr))
			return
		# add ( if applicable
		if left_count < n:
			curr.append('(')
			backtrack(curr, left_count + 1, right_count)
			curr.pop()
		# add ) if applicable
		if right_count < left_count:
			curr.append(')')
			backtrack(curr, left_count, right_count + 1)
			curr.pop()
	res = []
	backtrack([], 0, 0)
	return res

def subsets(self, nums: List[int]) -> List[List[int]]:
	def choice():
		""" Best approach """
		# $: []
		# 1: [],[1]
		# 2: [],[1],[2],[1,2]
		# 3: [],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]
		def backtrack(curr, index):
			nonlocal res
			if index == len(nums):
				res.append(curr[:])
				return
			# leave it
			backtrack(curr, index + 1)
			# take it
			curr.append(nums[index])
			backtrack(curr, index + 1)
			curr.pop()
		
		res = []
		backtrack([], 0)
		return res

	def forward():
		# []
		# [],[1]
		# [],[1],[2],[1,2]
		# [],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]
		def backtrack(curr, index):
			if index == len(nums):
				return curr
			prev = copy.deepcopy(curr)
			for ans in curr:
				ans.append(nums[index])
			return backtrack(prev + curr, index + 1)
		return backtrack([[]], 0)

	def backward():
		# [1,2,3]
		# []
		# [3],[]
		# [2,3],[2],[3],[]
		# [1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]
		def backtrack(index):
			if index == len(nums):
				return [[]]
			res = backtrack(index + 1)
			n = len(res)
			for i in range(n):
				curr = copy.deepcopy(res[i])
				curr.append(nums[index])
				res.append(curr)
			return res
		return backtrack(0)
