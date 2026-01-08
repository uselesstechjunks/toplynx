def search_right_for_min():
	# searching for the min of the segment num >= target
	n = len(nums)
	left, right = 0, n-1
	while left != right:
		mid = (left + right) // 2
		if nums[mid] >= target:
			right = mid
		else:
			left = mid + 1
	return -1 if nums[left] != target else left
def search_left_for_max():
	# searching for the max of the segment num <= target
	n = len(nums)
	left, right = 0, n-1
	while left <= right:
		mid = (left + right) // 2
		if nums[mid] <= target:
			left = mid + 1
		else:
			right = mid - 1
	return -1 if nums[right] != target else right