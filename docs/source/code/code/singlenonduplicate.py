def singleNonDuplicate(self, nums: List[int]) -> int:
	# the array is odd lengthed
	# the answer must lie in an odd lengthed subarray
	# the search gradually reduces the array to an array of length 1
	left, right = 0, len(nums) - 1
	while left != right:
		mid = (left + right) // 2
		if nums[mid] == nums[mid+1]:
			# left half [left, mid-1]
			left_half_even = (mid - left) % 2 == 0
			if left_half_even:
				left = mid + 2
			else:
				right = mid - 1
		elif nums[mid-1] == nums[mid]:
			# it's also safe to check mid-1 since min size of the array is 3
			# right half [mid+1, right]
			right_half_even = (right - mid) % 2 == 0
			if right_half_even:
				right = mid - 2
			else:
				left = mid + 1
		else:
			left = right = mid
	return nums[left]