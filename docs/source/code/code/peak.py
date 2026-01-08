def findPeakElement(self, nums: List[int]) -> int:
	if len(nums) == 1:
		return 0
	# the fact that we're asked to return ANY peak is the key
	# for ANY array, it has a segment of monotonically decreasing segment
	# on the right, even if it's of length 1. We need to find the beginning of that
	# segment.
	# example: 1 2 3 4, the subarray that's monotonically decreasing is [4]
	# example: 1,2,5,4, the subarray that's monotonically decreasing is [5,4]
	# we can use binary search to find this segment.
	left, right = 0, len(nums)-1
	while left < right:
		mid = (left + right) // 2
		if nums[mid] > nums[mid + 1]:
			right = mid
		else:
			left = mid + 1
	return left