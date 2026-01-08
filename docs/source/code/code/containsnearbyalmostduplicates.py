from sortedcontainers import SortedList
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
	"""
	observations:
	window that satisfies: assume l < r
	size at most k=indexDiff
	|nums[i]-nums[j]| <= v
	(a) nums[i]-nums[j] >= 0
		nums[i]-nums[j] <= v => nums[i] <= nums[j]+v
		find a number larger than or equal to v more than curr
	(b) nums[i]-nums[j] < 0
		-nums[i]+nums[j] <= v => nums[i] >= nums[j]-v
		find a number smaller than or equal to v less than curr
	for nums[j]
	search if there exists a value on the left
	range = [nums[j]-v, nums[j]+v]
	"""
	k, v = indexDiff, valueDiff
	bst = SortedList()
	for r in range(len(nums)):
		l = bst.bisect_left(nums[r])
		if l < len(bst) and bst[l] <= nums[r] + v:
			return True

		if l > 0 and bst[l-1] >= nums[r] - v:
			return True
		
		bst.add(nums[r])
		if len(bst) > k:
			bst.remove(nums[r-k])
	return False