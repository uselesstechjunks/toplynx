def threeSum(self, nums: List[int]) -> List[List[int]]:
	# can convert this to a two sum problem after sorting
	# why sorting? because the complexity is O(n^2) anyway.
	# example:
	# [-4,-1,-1,-1,0,1,2]
	# [ 0  1  2  3 4 5 6] <- index
	# if index (1,2,6) is included [-1,-1,2], then (1,3,6) or (2,3,6) cannot be included
	# when our first is stuck at index 1, we need to skip from left ONLY WHEN
	# it's not the immediate next one from first
	# we should also skip from caller to avoid searching from 2
	def two_sum(first_index):
		nonlocal result
		n = len(nums)
		first = nums[first_index]
		left, right = first_index + 1, n - 1

		while left < right:
			if left > first_index + 1 and nums[left-1] == nums[left]:
				left += 1
				continue
			if right < n - 1 and nums[right] == nums[right + 1]:
				right -= 1
				continue
			curr_sum = first + nums[left] + nums[right]
			if curr_sum == 0:
				result.append([first, nums[left], nums[right]])
				left += 1
				right -= 1
			elif curr_sum < 0:
				# need to add something larger
				left += 1
			else:
				# need to add something smaller
				right -= 1
	
	nums.sort()
	result = []
	for i, _ in enumerate(nums):
		if i > 0 and nums[i - 1] == nums[i]:
			continue
		two_sum(i)
	
	return result
