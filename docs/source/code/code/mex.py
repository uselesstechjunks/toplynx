def withExtraSpace(self, nums) -> int:
	nums = set(nums)
	mex = 1
	while mex in nums:
		mex += 1
	return mex
	
def withoutExtraSpace(self, nums) -> int:
	# use the array index itself as the set using index as lookup key
	# traverse from left to right
	# when we see a negative number, ignore
	# when we see a positive number:
	#   case 0: number at its correct index. do nothing and move right
	#   case 1: number is larger than index, need to move left
	#           overwrite if there's negative at it's correct index
	#           swap if there's a positive at it's correct index
	#           if swapped, that positive must have been from case 2 earlier
	#           if swapped, do not increase the current index
	#   case 2: number if smaller than index, need to move right
	#           if there exists some number on the right should should be sitting here
	#           if would be swapped in a later stage
	# once processed the whole array, traverse from left to right
	# return the first element that's not in its right place
	curr_index, length = 0, len(nums)-1
	while curr_index < length:
		if nums[curr_index] <= 0:
			curr_index += 1
			continue
		correct_index = nums[curr_index]-1
		if correct_index > curr_index and correct_index < length:
			nums[curr_index], nums[correct_index] = nums[correct_index], nums[curr_index]
		elif correct_index < curr_index and correct_index >= 0:
			nums[curr_index], nums[correct_index] = nums[correct_index], nums[curr_index]

	index = 0
	while index < length:
		if nums[index] <= 0:
			return index + 1
		index += 1
	return length+1

""" a dumber way of doing the above """
def constant_space():
	"""
	idea: use index as a proxy for the set above
	"""
	# NOTE
	# 1. anyting that's negative doesn't matter
	# 2. anything that's larger than the max index doesn't matter
	# we need to ensure that we don't overwrite anything
	# [0,3,4,-1,1]
	# [0,3,4,-1,1] i = 1
	# [0,0,4,3,1] i = 2
	# [0,0,1,3,4] i = 2
	numbers = [0] + nums
	index = 1

	while index < len(numbers):
		if numbers[index] <= 0 or numbers[index] > len(nums):
			numbers[index] = 0
			index += 1
		elif numbers[index] == index:
			index += 1
		elif numbers[numbers[index]] == numbers[index]: # this takes care of duplicates
			numbers[index] = 0
			index += 1
		else:
			tmp = numbers[numbers[index]]
			numbers[numbers[index]] = numbers[index]
			numbers[index] = tmp

	curr = 1
	while curr <= len(nums) and numbers[curr]:
		curr += 1
	return curr

def firstMissingPositive(self, nums: List[int]) -> int:
	return self.withoutExtraSpace(nums)

# https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/description/
def findSmallestInteger(self, nums: List[int], value: int) -> int:
	# NOTE this doesn't quite pass all testcases yet.
	"""
	The goal is to find maximum possible MEX, therefore we should
	try to look at filling out as many as possible

	the idea is to use modular arithmatic.

	If we add value to any number x multiple times, we can transform
	it to the range [0, value-1]. We can check x % value to find this number.

	The problem then is to find MEX in [0, value-1] range.
	"""
	transformed = set()
	for num in nums:
		curr = num % value
		while curr in transformed:
			curr += value
		transformed.add(curr)
	curr = 0
	while curr in transformed:
		curr += 1
	return curr
